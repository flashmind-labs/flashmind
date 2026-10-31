use std::io;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ratatui::widgets::Paragraph;

use flashmind_llm::{
    AnthropicProvider, MistralProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider,
};
use flashmind_tui::widgets::{
    ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse, EXIT_HINT, ExitConfirm,
};
use flashmind_types::{
    LlmProvider, ModelCategory, Provider, ProviderPreferences, ProviderSort, ReasoningLevel,
};

use crate::config::Config;
use crate::paths::config_dir;

// ---------------------------------------------------------------------------
// Choice picker helper
// ---------------------------------------------------------------------------

/// Run a picker. Esc and Ctrl-C twice both close it with `None`.
pub fn run_choice(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<Option<ChoiceResponse>> {
    Ok(pick_choice(tui, picker)?.chosen())
}

/// Outcome of one picker in a multi-step flow.
#[derive(Debug)]
enum Pick<T> {
    Chosen(T),
    /// Esc: go back one step.
    Back,
    /// Ctrl-C twice: leave the whole flow.
    Quit,
}

impl<T> Pick<T> {
    fn chosen(self) -> Option<T> {
        match self {
            Pick::Chosen(value) => Some(value),
            Pick::Back | Pick::Quit => None,
        }
    }
}

/// Unwrap `Pick::Chosen`, or return `Back`/`Quit` from the enclosing function.
macro_rules! chosen {
    ($pick:expr) => {
        match $pick {
            Pick::Chosen(value) => value,
            Pick::Back => return Ok(Pick::Back),
            Pick::Quit => return Ok(Pick::Quit),
        }
    };
}

fn pick_choice(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<Pick<ChoiceResponse>> {
    loop {
        match run_choice_action(tui, picker)? {
            ChoicePickerAction::Select(r) => return Ok(Pick::Chosen(r)),
            ChoicePickerAction::Cancel => return Ok(Pick::Back),
            ChoicePickerAction::Quit => return Ok(Pick::Quit),
            ChoicePickerAction::Delete(_) => {}
        }
    }
}

pub fn run_choice_action(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<ChoicePickerAction> {
    use ratatui::crossterm::event;

    let mut screen = tui.fullscreen()?;
    loop {
        screen.draw(|frame| picker.render(frame))?;
        if !event::poll(std::time::Duration::from_millis(100))? {
            continue;
        }
        if let Some(action) = picker.handle_event(screen.read_event()?) {
            return Ok(action);
        }
    }
}

// ---------------------------------------------------------------------------
// Model picker
// ---------------------------------------------------------------------------

fn format_context(tokens: u32) -> String {
    flashmind_types::humanize::format_tokens(tokens)
}

fn format_price(price: Option<rust_decimal::Decimal>) -> String {
    match price {
        None => "—".into(),
        Some(p) if p.is_zero() => "free".into(),
        Some(p) => {
            let mtok = p * rust_decimal::Decimal::from(1_000_000);
            if mtok < rust_decimal::Decimal::ONE {
                format!("${}/M", mtok.round_dp(4).normalize())
            } else {
                format!("${}/M", mtok.round_dp(2).normalize())
            }
        }
    }
}

fn model_features(model: &flashmind_types::ModelInfo) -> Vec<&'static str> {
    let c = &model.capabilities;
    let has = |category| model.categories.contains(&category);
    let mut features = Vec::new();
    for (enabled, label) in [
        (has(ModelCategory::Chat), "chat"),
        (c.tool_calling, "tools"),
        (c.reasoning || has(ModelCategory::Reasoning), "reasoning"),
        (c.images || has(ModelCategory::Vision), "vision"),
        (c.documents, "documents"),
        (c.audio, "audio-input"),
        (c.video || has(ModelCategory::VideoInput), "video-input"),
        (
            c.image_generation || has(ModelCategory::ImageGeneration),
            "image",
        ),
        (
            c.video_generation || has(ModelCategory::VideoGeneration),
            "video",
        ),
        (c.audio_output || has(ModelCategory::Tts), "speech"),
        (has(ModelCategory::Stt), "transcribe"),
        (
            model.image.editing || has(ModelCategory::ImageEditing),
            "image-edit",
        ),
        (
            model.image.style_references == Some(true),
            "style-reference",
        ),
    ] {
        if enabled {
            features.push(label);
        }
    }
    features
}

fn matches_catalog_query(model: &flashmind_types::ModelInfo, query: &str) -> bool {
    query.split_whitespace().all(|term| {
        if let Some(capability) = term.strip_prefix("cap:") {
            return model_features(model).contains(&capability);
        }
        for (prefix, price) in [
            ("in", model.pricing.prompt),
            ("out", model.pricing.completion),
        ] {
            let Some(rest) = term.strip_prefix(prefix) else {
                continue;
            };
            let (operator, value) = if let Some(value) = rest.strip_prefix("<=") {
                ("<=", value)
            } else if let Some(value) = rest.strip_prefix(">=") {
                (">=", value)
            } else if let Some(value) = rest.strip_prefix('<') {
                ("<", value)
            } else if let Some(value) = rest.strip_prefix('>') {
                (">", value)
            } else if let Some(value) = rest.strip_prefix('=') {
                ("=", value)
            } else {
                continue;
            };
            let Ok(limit) = value
                .trim_start_matches('$')
                .parse::<rust_decimal::Decimal>()
            else {
                return false;
            };
            let Some(price) = price else { return false };
            let limit = limit / rust_decimal::Decimal::from(1_000_000);
            return match operator {
                "<=" => price <= limit,
                ">=" => price >= limit,
                "<" => price < limit,
                ">" => price > limit,
                "=" => price == limit,
                _ => false,
            };
        }
        model.id.to_lowercase().contains(term)
            || model
                .name
                .as_deref()
                .unwrap_or("")
                .to_lowercase()
                .contains(term)
    })
}

struct ModelPicker<'a> {
    models: &'a [flashmind_types::ModelInfo],
    filtered: Vec<usize>,
    selected: usize,
    scroll: usize,
    query: String,
    visible: usize,
    esc_hint: &'static str,
    exit: ExitConfirm,
    browse: bool,
    focus: CatalogFocus,
    input_limit: String,
    output_limit: String,
    feature_cursor: usize,
    features: Vec<&'static str>,
    sort: Option<CatalogSort>,
    details: bool,
    details_scroll: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CatalogSort {
    column: CatalogFocus,
    descending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogFocus {
    Model,
    Context,
    Created,
    Input,
    Output,
    Features,
}

impl CatalogFocus {
    fn next(self, reverse: bool) -> Self {
        use CatalogFocus::*;
        match (self, reverse) {
            (Model, false) | (Created, true) => Context,
            (Context, false) | (Input, true) => Created,
            (Created, false) | (Output, true) => Input,
            (Input, false) | (Features, true) => Output,
            (Output, false) | (Model, true) => Features,
            (Features, false) | (Context, true) => Model,
        }
    }
}

const CATALOG_FEATURES: &[&str] = &[
    "chat",
    "tools",
    "reasoning",
    "vision",
    "documents",
    "audio-input",
    "video-input",
    "image",
    "video",
    "speech",
    "transcribe",
    "image-edit",
    "style-reference",
];

fn within_price_limit(price: Option<rust_decimal::Decimal>, limit: &str) -> bool {
    if limit.is_empty() {
        return true;
    }
    let Ok(limit) = limit.parse::<rust_decimal::Decimal>() else {
        return false;
    };
    price.is_some_and(|price| price <= limit / rust_decimal::Decimal::from(1_000_000))
}

fn model_details(model: &flashmind_types::ModelInfo) -> String {
    let style = match model.image.style_references {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    };
    let max_images = model
        .image
        .max_edit_images
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".into());
    let mut text = format!(
        "Model: {}\nName: {}\nContext: {}\nInput: {}\nOutput: {}\nCapabilities: {}\nStyle references: {style}\nMaximum edit images: {max_images}",
        model.id,
        model.name.as_deref().unwrap_or("unknown"),
        model
            .context_length
            .map(format_context)
            .unwrap_or_else(|| "unknown".into()),
        format_price(model.pricing.prompt),
        format_price(model.pricing.completion),
        model_features(model).join(", ")
    );
    if !model.image.parameters.is_empty() {
        text.push_str("\n\nImage options (provider-reported):");
        if model.id.starts_with("openrouter:") {
            text.push_str("\nAvailable options are combined across endpoints.");
        }
        for (name, descriptor) in &model.image.parameters {
            let option = match name.as_str() {
                "resolution" => " (--size / --resolution)",
                "aspect_ratio" => " (--aspect-ratio)",
                "output_format" => " (--format)",
                "quality" => " (--quality)",
                "input_references" => " (--reference / --image)",
                "n" => " (-n)",
                _ => "",
            };
            let value = if let Some(values) = descriptor
                .get("values")
                .and_then(serde_json::Value::as_array)
            {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| value.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            } else if let (Some(min), Some(max)) = (descriptor.get("min"), descriptor.get("max")) {
                format!("{min} to {max}")
            } else {
                descriptor.to_string()
            };
            text.push_str(&format!("\n{name}{option}: {value}"));
        }
    }
    text.push_str("\n\nUp/Down: scroll  Esc or F2: return");
    text
}

impl<'a> ModelPicker<'a> {
    const VISIBLE: usize = 10;

    fn new(models: &'a [flashmind_types::ModelInfo]) -> Self {
        let mut picker = Self {
            models,
            filtered: Vec::new(),
            selected: 0,
            scroll: 0,
            query: String::new(),
            visible: Self::VISIBLE,
            esc_hint: "cancel",
            exit: ExitConfirm::default(),
            browse: false,
            focus: CatalogFocus::Model,
            input_limit: String::new(),
            output_limit: String::new(),
            feature_cursor: 0,
            features: Vec::new(),
            sort: None,
            details: false,
            details_scroll: 0,
        };
        picker.refilter();
        picker
    }

    fn refilter(&mut self) {
        let q = self.query.to_lowercase();
        self.filtered = (0..self.models.len())
            .filter(|&i| self.browse || self.models[i].categories.contains(&ModelCategory::Chat))
            .filter(|&i| {
                !self.browse
                    || (within_price_limit(self.models[i].pricing.prompt, &self.input_limit)
                        && within_price_limit(
                            self.models[i].pricing.completion,
                            &self.output_limit,
                        )
                        && self
                            .features
                            .iter()
                            .all(|feature| model_features(&self.models[i]).contains(feature)))
            })
            .filter(|&i| {
                if q.is_empty() {
                    return true;
                }
                let m = &self.models[i];
                if self.browse {
                    return matches_catalog_query(m, &q);
                }
                let name = m.name.as_deref().unwrap_or("");
                let id = &m.id;
                name.to_lowercase().contains(&q) || id.to_lowercase().contains(&q)
            })
            .collect();
        self.filtered.sort_by(|&a, &b| {
            let (a, b) = (&self.models[a], &self.models[b]);
            if let Some(sort) = self.sort.filter(|_| self.browse) {
                let cmp_option =
                    |a: Option<rust_decimal::Decimal>, b: Option<rust_decimal::Decimal>| match (
                        a, b,
                    ) {
                        (Some(a), Some(b)) => {
                            if sort.descending {
                                b.cmp(&a)
                            } else {
                                a.cmp(&b)
                            }
                        }
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    };
                let result = match sort.column {
                    CatalogFocus::Model => {
                        if sort.descending {
                            b.id.cmp(&a.id)
                        } else {
                            a.id.cmp(&b.id)
                        }
                    }
                    CatalogFocus::Context => cmp_option(
                        a.context_length.map(rust_decimal::Decimal::from),
                        b.context_length.map(rust_decimal::Decimal::from),
                    ),
                    CatalogFocus::Created => cmp_option(
                        a.created_at
                            .map(|value| rust_decimal::Decimal::from(value.timestamp())),
                        b.created_at
                            .map(|value| rust_decimal::Decimal::from(value.timestamp())),
                    ),
                    CatalogFocus::Input => cmp_option(a.pricing.prompt, b.pricing.prompt),
                    CatalogFocus::Output => cmp_option(a.pricing.completion, b.pricing.completion),
                    CatalogFocus::Features => std::cmp::Ordering::Equal,
                };
                result.then_with(|| a.id.cmp(&b.id))
            } else {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            }
        });
        self.selected = 0;
        self.scroll = 0;
    }

    fn adjust_scroll(&mut self) {
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + self.visible {
            self.scroll = self.selected + 1 - self.visible;
        }
    }

    fn handle_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> Option<Pick<usize>> {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};
        if ExitConfirm::is_ctrl_c(&key) {
            return self.exit.press().then_some(Pick::Quit);
        }
        if self.details {
            match key.code {
                KeyCode::Esc | KeyCode::F(2) => {
                    self.details = false;
                    self.details_scroll = 0;
                }
                KeyCode::Up => self.details_scroll = self.details_scroll.saturating_sub(1),
                KeyCode::Down => self.details_scroll = self.details_scroll.saturating_add(1),
                _ => {}
            }
            return None;
        }
        match key.code {
            KeyCode::F(2) if self.browse && !self.filtered.is_empty() => {
                self.details = true;
                None
            }
            KeyCode::Tab if self.browse => {
                self.focus = self.focus.next(key.modifiers.contains(KeyModifiers::SHIFT));
                None
            }
            KeyCode::BackTab if self.browse => {
                self.focus = self.focus.next(true);
                None
            }
            KeyCode::Left if self.browse && self.focus == CatalogFocus::Features => {
                self.feature_cursor =
                    (self.feature_cursor + CATALOG_FEATURES.len() - 1) % CATALOG_FEATURES.len();
                None
            }
            KeyCode::Right if self.browse && self.focus == CatalogFocus::Features => {
                self.feature_cursor = (self.feature_cursor + 1) % CATALOG_FEATURES.len();
                None
            }
            KeyCode::Char(' ') if self.browse && self.focus == CatalogFocus::Features => {
                let feature = CATALOG_FEATURES[self.feature_cursor];
                if let Some(index) = self.features.iter().position(|value| *value == feature) {
                    self.features.remove(index);
                } else {
                    self.features.push(feature);
                }
                self.refilter();
                None
            }
            KeyCode::Enter if self.browse && self.focus != CatalogFocus::Features => {
                let descending = self
                    .sort
                    .is_some_and(|sort| sort.column == self.focus && !sort.descending);
                self.sort = Some(CatalogSort {
                    column: self.focus,
                    descending,
                });
                self.refilter();
                None
            }
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                self.adjust_scroll();
                None
            }
            KeyCode::Down => {
                let max = self.filtered.len().saturating_sub(1);
                self.selected = (self.selected + 1).min(max);
                self.adjust_scroll();
                None
            }
            KeyCode::Enter if !self.browse => {
                if self.filtered.is_empty() {
                    None
                } else {
                    Some(Pick::Chosen(self.filtered[self.selected]))
                }
            }
            KeyCode::Esc => Some(Pick::Back),
            KeyCode::Backspace => {
                match self.focus {
                    CatalogFocus::Input if self.browse => {
                        self.input_limit.pop();
                    }
                    CatalogFocus::Output if self.browse => {
                        self.output_limit.pop();
                    }
                    CatalogFocus::Features if self.browse => {
                        self.features.clear();
                    }
                    _ => {
                        self.query.pop();
                    }
                }
                self.refilter();
                None
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match self.focus {
                    CatalogFocus::Input if self.browse && (c.is_ascii_digit() || c == '.') => {
                        self.input_limit.push(c)
                    }
                    CatalogFocus::Output if self.browse && (c.is_ascii_digit() || c == '.') => {
                        self.output_limit.push(c)
                    }
                    CatalogFocus::Model => self.query.push(c),
                    _ => {}
                }
                self.refilter();
                None
            }
            _ => None,
        }
    }

    fn lines(&self, max_width: u16) -> Vec<ratatui::text::Line<'static>> {
        use ratatui::style::{Color, Modifier, Style};
        use ratatui::text::{Line, Span};

        let dim = flashmind_tui::styles::S_DIM;
        let mut lines = Vec::new();

        lines.push(Line::from(vec![
            Span::styled(
                if self.browse && self.focus == CatalogFocus::Model {
                    "> Search: "
                } else {
                    "  Search: "
                },
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!("{}_", self.query),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(format!("  ({} models)", self.filtered.len()), dim),
        ]));
        lines.push(Line::default());

        let vis_end = (self.scroll + self.visible).min(self.filtered.len());
        let visible = &self.filtered[self.scroll..vis_end];

        let w = max_width as usize;
        let hdr_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let focused = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
        let header = |focus| {
            if self.browse && self.focus == focus {
                focused
            } else {
                hdr_style
            }
        };
        let name_w = (w / 4).max(12);
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<name_w$}", "Model"),
                header(CatalogFocus::Model),
            ),
            Span::styled(format!("{:>7}", "Ctx"), header(CatalogFocus::Context)),
            Span::styled(format!("{:>11}", "Released"), header(CatalogFocus::Created)),
            Span::styled(format!("{:>10}", "In $/M"), header(CatalogFocus::Input)),
            Span::styled(format!("{:>10}", "Out $/M"), header(CatalogFocus::Output)),
            Span::styled("  Features", header(CatalogFocus::Features)),
        ]));
        if self.browse {
            let input = if self.input_limit.is_empty() {
                "any".to_owned()
            } else {
                format!("<={}", self.input_limit)
            };
            let output = if self.output_limit.is_empty() {
                "any".to_owned()
            } else {
                format!("<={}", self.output_limit)
            };
            let feature = CATALOG_FEATURES[self.feature_cursor];
            let checked = if self.features.contains(&feature) {
                "x"
            } else {
                " "
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<name_w$}{:>7}{:>11}", "", "", ""), dim),
                Span::styled(format!("{:>10}", input), header(CatalogFocus::Input)),
                Span::styled(format!("{:>10}", output), header(CatalogFocus::Output)),
                Span::styled(
                    format!("  [{checked}] {feature}"),
                    header(CatalogFocus::Features),
                ),
            ]));
        }
        lines.push(Line::from(Span::styled(
            format!("  {}", "─".repeat(w.saturating_sub(2).min(100))),
            dim,
        )));

        if self.filtered.is_empty() {
            lines.push(Line::from(Span::styled("  No matching models.", dim)));
        }

        for (i, &model_idx) in visible.iter().enumerate() {
            let abs_i = self.scroll + i;
            let is_sel = abs_i == self.selected;
            let m = &self.models[model_idx];

            let name = if self.browse {
                &m.id
            } else {
                m.name.as_deref().unwrap_or(&m.id)
            };
            let name_display: String = if name.len() > name_w - 1 {
                let truncated: String = name.chars().take(name_w - 2).collect();
                format!("{truncated}…")
            } else {
                name.to_string()
            };

            let ctx = m
                .context_length
                .map(format_context)
                .unwrap_or_else(|| "—".into());
            let in_price = format_price(m.pricing.prompt);
            let out_price = format_price(m.pricing.completion);

            let feats_str = model_features(m).join(", ");

            let (indicator, row_style) = if is_sel {
                (
                    "▸ ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ("  ", Style::default().fg(Color::White))
            };

            lines.push(Line::from(vec![
                Span::styled(indicator.to_string(), if is_sel { row_style } else { dim }),
                Span::styled(format!("{name_display:<name_w$}"), row_style),
                Span::styled(format!("{ctx:>7}"), dim),
                Span::styled(
                    format!(
                        "{:>11}",
                        m.created_at
                            .map(|date| date.format("%Y-%m-%d").to_string())
                            .unwrap_or_else(|| "-".into())
                    ),
                    dim,
                ),
                Span::styled(format!("{in_price:>10}"), row_style),
                Span::styled(format!("{out_price:>10}"), row_style),
                Span::styled(format!("  {feats_str}"), dim),
            ]));
        }

        let total = self.filtered.len();
        if total > self.visible {
            let has_above = self.scroll > 0;
            let has_below = self.scroll + self.visible < total;
            let arrows = match (has_above, has_below) {
                (true, true) => "↑↓",
                (true, false) => "↑",
                (false, true) => "↓",
                _ => "",
            };
            lines.push(Line::from(Span::styled(
                format!("  {arrows} {}/{total}", self.selected + 1),
                dim,
            )));
        }

        lines.push(Line::default());
        if self.browse {
            let order = match self.sort {
                None => "release date (newest first)".to_owned(),
                Some(sort) => format!(
                    "{} ({})",
                    match sort.column {
                        CatalogFocus::Model => "model",
                        CatalogFocus::Context => "context",
                        CatalogFocus::Created => "release date",
                        CatalogFocus::Input => "input price",
                        CatalogFocus::Output => "output price",
                        CatalogFocus::Features => "features",
                    },
                    if sort.descending {
                        "descending"
                    } else {
                        "ascending"
                    }
                ),
            };
            lines.push(Line::from(format!("  Sort: {order}")));
            let filters = if self.features.is_empty() {
                "any".to_owned()
            } else {
                self.features.join(", ")
            };
            lines.push(Line::from(format!("  Feature filters: {filters}")));
            if let Some(index) = self.filtered.get(self.selected) {
                let features = model_features(&self.models[*index]);
                let label = if features.is_empty() {
                    "unknown".to_owned()
                } else {
                    features.join(", ")
                };
                lines.push(Line::from(format!("  Capabilities: {label}")));
            }
        }
        let keys = if self.browse {
            "  Tab: column  Enter: sort  F2: details  Left/Right: feature  Space: toggle  Esc: close"
                .to_string()
        } else {
            format!(
                "  Type to search  ↑/↓: navigate  Enter: select  Esc: {}",
                self.esc_hint
            )
        };
        lines.push(self.hint(keys));

        lines
    }

    /// The key hint, or the exit prompt after a first Ctrl-C.
    fn hint(&self, keys: String) -> ratatui::text::Line<'static> {
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};

        if self.exit.pending() {
            Line::from(Span::styled(
                format!("  {EXIT_HINT}"),
                Style::default().fg(Color::Yellow),
            ))
        } else {
            Line::from(Span::styled(keys, flashmind_tui::styles::S_DIM))
        }
    }
}

pub fn run_model_picker(
    tui: &mut flashmind_tui::Tui,
    models: &[flashmind_types::ModelInfo],
) -> io::Result<Option<usize>> {
    Ok(pick_model(tui, models, None, "cancel", false)?.chosen())
}

pub(crate) fn browse_models(
    tui: &mut flashmind_tui::Tui,
    models: &[flashmind_types::ModelInfo],
) -> io::Result<()> {
    pick_model(tui, models, None, "close", true)?;
    Ok(())
}

fn pick_model(
    tui: &mut flashmind_tui::Tui,
    models: &[flashmind_types::ModelInfo],
    initial: Option<usize>,
    esc_hint: &'static str,
    browse: bool,
) -> io::Result<Pick<usize>> {
    use ratatui::crossterm::event::{self, Event};

    let mut picker = ModelPicker::new(models);
    picker.esc_hint = esc_hint;
    picker.browse = browse;
    picker.refilter();
    if let Some(i) = initial.and_then(|idx| picker.filtered.iter().position(|&m| m == idx)) {
        picker.selected = i;
    }
    let mut screen = tui.fullscreen()?;
    loop {
        screen.draw(|frame| {
            if picker.details {
                let model = &picker.models[picker.filtered[picker.selected]];
                let text = model_details(model);
                frame.render_widget(
                    Paragraph::new(text)
                        .wrap(ratatui::widgets::Wrap { trim: false })
                        .scroll((picker.details_scroll, 0)),
                    frame.area(),
                );
                return;
            }
            picker.visible = (frame.area().height as usize)
                .saturating_sub(if browse { 11 } else { 7 })
                .max(1);
            picker.adjust_scroll();
            let lines = picker.lines(frame.area().width);
            frame.render_widget(Paragraph::new(lines), frame.area());
        })?;
        if !event::poll(std::time::Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = screen.read_event()? else {
            continue;
        };
        if let Some(result) = picker.handle_key(key) {
            return Ok(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Setup wizard
// ---------------------------------------------------------------------------

const PROVIDERS: &[(&str, Provider)] = &[
    ("Ollama (local, no API key)", Provider::Ollama),
    ("OpenRouter (many models, one key)", Provider::OpenRouter),
    ("Anthropic (Claude models)", Provider::Anthropic),
    ("OpenAI (GPT models)", Provider::OpenAi),
    ("Mistral (Mistral models)", Provider::Mistral),
];

fn options(labels: &[&str]) -> Vec<ChoiceOption> {
    labels
        .iter()
        .map(|label| ChoiceOption {
            label: (*label).into(),
            accepts_input: false,
        })
        .collect()
}

/// Ask for ZDR, then sorting. Esc on the sorting picker returns to the ZDR
/// picker; Esc on the ZDR picker returns `Back`.
fn prompt_openrouter_routing(
    existing: Option<&ProviderPreferences>,
    mut choose: impl FnMut(&mut ChoicePicker) -> io::Result<Pick<ChoiceResponse>>,
) -> io::Result<Pick<ProviderPreferences>> {
    let mut routing = existing.cloned().unwrap_or_default();
    let sorts = [
        ("OpenRouter default", None),
        (
            "Throughput (fastest token generation)",
            Some(ProviderSort::Throughput),
        ),
        ("Latency (fastest first token)", Some(ProviderSort::Latency)),
        ("Price (lowest cost)", Some(ProviderSort::Price)),
    ];
    loop {
        let mut zdr_picker = ChoicePicker::new(
            "OpenRouter zero data retention (ZDR)".into(),
            options(&["Require ZDR endpoints", "Use account settings"]),
        )
        .esc_hint("back");
        zdr_picker.selected = if routing.zdr == Some(true) { 0 } else { 1 };
        let zdr = chosen!(choose(&mut zdr_picker)?);
        routing.zdr = Some(zdr.selected == 0);

        let labels: Vec<&str> = sorts.iter().map(|(label, _)| *label).collect();
        let mut sort_picker =
            ChoicePicker::new("OpenRouter provider sorting".into(), options(&labels))
                .esc_hint("back");
        sort_picker.selected = sorts
            .iter()
            .position(|(_, sort)| *sort == routing.sort)
            .unwrap_or(0);
        match choose(&mut sort_picker)? {
            Pick::Chosen(sort) => {
                routing.sort = sorts[sort.selected].1.clone();
                return Ok(Pick::Chosen(routing));
            }
            Pick::Back => {}
            Pick::Quit => return Ok(Pick::Quit),
        }
    }
}

/// Prompt for an API key until a non-empty one is entered.
fn prompt_key(
    tui: &mut flashmind_tui::Tui,
    title: String,
    previous: Option<&str>,
    esc_hint: &'static str,
) -> io::Result<Pick<String>> {
    let mut picker = ChoicePicker::new(
        title,
        vec![ChoiceOption {
            label: "API key".into(),
            accepts_input: true,
        }],
    )
    .esc_hint(esc_hint)
    .cancel_on_input_esc()
    .secret();
    picker.input_buffer = previous.unwrap_or_default().to_string();
    loop {
        let resp = chosen!(pick_choice(tui, &mut picker)?);
        if !resp.input.trim().is_empty() {
            return Ok(Pick::Chosen(resp.input.trim().to_string()));
        }
    }
}

/// Setup steps in order. Steps that don't apply to the current choices are
/// skipped, both going forward and going back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Provider,
    ApiKey,
    Model,
    Reasoning,
    Routing,
}

const STEPS: &[Step] = &[
    Step::Provider,
    Step::ApiKey,
    Step::Model,
    Step::Reasoning,
    Step::Routing,
];

/// Choices made so far. Kept across back/forward so revisited steps start
/// on the previous answer.
struct Wizard<'a> {
    config: &'a Config,
    provider_idx: usize,
    entered_key: Option<String>,
    models: Vec<flashmind_types::ModelInfo>,
    models_for: Option<(usize, Option<String>)>,
    model_idx: Option<usize>,
    reasoning_idx: usize,
    routing: Option<ProviderPreferences>,
    /// Error from the last step, shown in the next picker's title.
    notice: Option<String>,
}

impl<'a> Wizard<'a> {
    fn new(config: &'a Config) -> Self {
        Self {
            config,
            provider_idx: 0,
            entered_key: None,
            models: Vec::new(),
            models_for: None,
            model_idx: None,
            reasoning_idx: 0,
            routing: config.openrouter.routing.clone(),
            notice: None,
        }
    }

    fn provider(&self) -> Provider {
        PROVIDERS[self.provider_idx].1.clone()
    }

    fn provider_label(&self) -> &'static str {
        match self.provider() {
            Provider::OpenRouter => "OpenRouter",
            Provider::Anthropic => "Anthropic",
            Provider::OpenAi => "OpenAI",
            Provider::Mistral => "Mistral",
            _ => "API",
        }
    }

    fn existing_key(&self) -> Option<String> {
        self.config.api_key_for(&self.provider()).map(str::to_owned)
    }

    fn api_key(&self) -> Option<String> {
        match self.provider() {
            Provider::Ollama => None,
            _ => self.existing_key().or_else(|| self.entered_key.clone()),
        }
    }

    fn model(&self) -> Option<&flashmind_types::ModelInfo> {
        self.model_idx.and_then(|i| self.models.get(i))
    }

    fn applies(&self, step: Step) -> bool {
        match step {
            Step::Provider | Step::Model => true,
            Step::ApiKey => self.provider() != Provider::Ollama && self.existing_key().is_none(),
            Step::Reasoning => self.model().is_some_and(|m| m.capabilities.reasoning),
            Step::Routing => self.provider() == Provider::OpenRouter,
        }
    }

    fn next_after(&self, step: Step) -> Option<Step> {
        let pos = STEPS.iter().position(|&s| s == step)?;
        STEPS[pos + 1..].iter().copied().find(|&s| self.applies(s))
    }

    fn title(&mut self, base: &str) -> String {
        match self.notice.take() {
            Some(notice) => format!("{base}  ({notice})"),
            None => base.to_string(),
        }
    }

    /// Pick from `labels`, starting on `selected`.
    fn choose(
        &mut self,
        tui: &mut flashmind_tui::Tui,
        title: &str,
        labels: &[&str],
        selected: usize,
        esc_hint: &'static str,
    ) -> io::Result<Pick<usize>> {
        let mut picker = ChoicePicker::new(self.title(title), options(labels)).esc_hint(esc_hint);
        picker.selected = selected.min(labels.len().saturating_sub(1));
        Ok(Pick::Chosen(
            chosen!(pick_choice(tui, &mut picker)?).selected,
        ))
    }

    /// Run one step. `Chosen` means move on to the next step.
    async fn run_step(
        &mut self,
        tui: &mut flashmind_tui::Tui,
        step: Step,
        esc_hint: &'static str,
    ) -> Result<Pick<()>> {
        match step {
            Step::Provider => {
                let labels: Vec<&str> = PROVIDERS.iter().map(|(label, _)| *label).collect();
                let idx = chosen!(self.choose(
                    tui,
                    "Choose a provider",
                    &labels,
                    self.provider_idx,
                    esc_hint,
                )?);
                if idx != self.provider_idx {
                    self.provider_idx = idx;
                    self.entered_key = None;
                    self.model_idx = None;
                }
                Ok(Pick::Chosen(()))
            }
            Step::ApiKey => {
                let label = self.provider_label();
                let key = chosen!(prompt_key(
                    tui,
                    self.title(&format!("Enter {label} API key")),
                    self.entered_key.as_deref(),
                    esc_hint,
                )?);
                if self.entered_key.as_ref() != Some(&key) {
                    self.model_idx = None;
                }
                self.entered_key = Some(key);
                Ok(Pick::Chosen(()))
            }
            Step::Model => {
                let source = (self.provider_idx, self.api_key());
                if self.models_for.as_ref() != Some(&source) {
                    match self.fetch_models(tui).await {
                        Ok(models) if !models.is_empty() => {
                            self.models = models;
                            self.models_for = Some(source);
                        }
                        Ok(_) => {
                            self.notice = Some(
                                "no models found — check your API key or that the provider is running"
                                    .into(),
                            );
                            return Ok(Pick::Back);
                        }
                        Err(err) => {
                            self.notice = Some(format!("failed to list models: {err:#}"));
                            return Ok(Pick::Back);
                        }
                    }
                }
                let idx = chosen!(pick_model(
                    tui,
                    &self.models,
                    self.model_idx,
                    esc_hint,
                    false
                )?);
                self.model_idx = Some(idx);
                Ok(Pick::Chosen(()))
            }
            Step::Reasoning => {
                let idx = chosen!(self.choose(
                    tui,
                    "Thinking / reasoning level",
                    &["Off", "Low", "Medium", "High"],
                    self.reasoning_idx,
                    esc_hint,
                )?);
                self.reasoning_idx = idx;
                Ok(Pick::Chosen(()))
            }
            Step::Routing => {
                let routing = chosen!(prompt_openrouter_routing(
                    self.routing.as_ref(),
                    |picker| { pick_choice(tui, picker) }
                )?);
                self.routing = Some(routing);
                Ok(Pick::Chosen(()))
            }
        }
    }

    async fn fetch_models(
        &self,
        tui: &mut flashmind_tui::Tui,
    ) -> Result<Vec<flashmind_types::ModelInfo>> {
        let config = self.config;
        let api_key = self.api_key();
        let require_key = || api_key.clone().context("API key is required");
        let llm_provider: Arc<dyn LlmProvider> = match self.provider() {
            Provider::Ollama => Arc::new(OllamaProvider::new(config.ollama_url.clone(), None)?),
            Provider::OpenRouter => Arc::new(OpenRouterProvider::new(require_key()?)),
            Provider::Anthropic => Arc::new(AnthropicProvider::new(require_key()?)),
            Provider::OpenAi => {
                let base = config
                    .openai_base_url
                    .as_deref()
                    .unwrap_or("https://api.openai.com/v1");
                Arc::new(
                    OpenAiProvider::builder(base)
                        .api_key(require_key()?)
                        .build()?,
                )
            }
            Provider::Mistral => Arc::new(MistralProvider::new(require_key()?)),
            other => bail!("unsupported provider: {other:?}"),
        };

        tui.println(&ratatui::text::Line::from(format!(
            "\n  Fetching models from {}...",
            llm_provider.name()
        )))?;
        llm_provider
            .list_models()
            .await
            .context("check your API key and network")
    }

    fn config_for_save(&self) -> Result<Config> {
        let provider = self.provider();
        let model = self.model().context("no model selected")?;
        let reasoning = if model.capabilities.reasoning {
            match self.reasoning_idx {
                1 => ReasoningLevel::Low,
                2 => ReasoningLevel::Medium,
                3 => ReasoningLevel::High,
                _ => ReasoningLevel::Off,
            }
        } else {
            ReasoningLevel::Off
        };

        let mut config = self.config.clone();
        config.model = Some(format!("{provider}:{}", model.id));
        config.reasoning = reasoning.is_on().then_some(reasoning);
        if let Some(key) = self.api_key() {
            config
                .providers
                .entry(provider.to_string())
                .or_default()
                .api_key = Some(crate::ApiKey(key));
        }
        match provider {
            Provider::OpenRouter => {
                config.openrouter_api_key = None;
                config.openrouter.routing = self.routing.clone();
            }
            Provider::Anthropic => config.anthropic_api_key = None,
            Provider::OpenAi => config.openai_api_key = None,
            Provider::Mistral => config.mistral_api_key = None,
            _ => {}
        }
        Ok(config)
    }

    fn save(self, tui: &mut flashmind_tui::Tui) -> Result<()> {
        let config = self.config_for_save()?;
        let model_str = config.model.as_deref().context("no model selected")?;
        let config_path = config_dir()?.join("config.toml");
        std::fs::write(&config_path, toml::to_string(&config)?)?;

        tui.println(&ratatui::text::Line::default())?;
        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("  Saved to {}", config_path.display()),
            flashmind_tui::styles::S_AGENT,
        )))?;
        tui.println(&ratatui::text::Line::from(format!("  Model: {model_str}")))?;
        tui.println(&ratatui::text::Line::from(
            "  Run `flsh` to start chatting.",
        ))?;

        tui.suspend()?;
        println!("Configuration saved. Run `flsh` to start chatting.");

        Ok(())
    }
}

/// Guided setup. Esc goes back one step; Esc on the first step cancels.
/// Ctrl-C twice quits from any step without writing the config.
pub async fn run_setup(config: &Config) -> Result<()> {
    let mut tui = flashmind_tui::Tui::new();
    let mut wizard = Wizard::new(config);
    let mut history: Vec<Step> = Vec::new();
    let mut step = Step::Provider;

    loop {
        let esc_hint = if history.is_empty() { "cancel" } else { "back" };
        let prev = match wizard.run_step(&mut tui, step, esc_hint).await? {
            Pick::Chosen(()) => {
                history.push(step);
                match wizard.next_after(step) {
                    Some(next) => step = next,
                    None => break,
                }
                continue;
            }
            Pick::Back => history.pop(),
            Pick::Quit => None,
        };
        match prev {
            Some(prev) => step = prev,
            None => {
                tui.suspend()?;
                println!("Setup cancelled.");
                return Ok(());
            }
        }
    }

    wizard.save(&mut tui)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::{ModelCapabilities, ModelInfo};

    #[test]
    fn openrouter_setup_saves_selected_routing_and_keeps_other_options() {
        let existing = ProviderPreferences {
            order: vec!["example-provider".into()],
            data_collection: Some("deny".into()),
            allow_fallbacks: Some(false),
            ..Default::default()
        };
        let mut selections = [0, 1].into_iter();
        let routing = prompt_openrouter_routing(Some(&existing), |picker| {
            picker.selected = selections.next().unwrap();
            Ok(Pick::Chosen(picker.respond()))
        })
        .unwrap()
        .chosen()
        .unwrap();
        let saved = format!(
            "[openrouter.routing]\n{}",
            toml::to_string(&routing).unwrap()
        );
        let config: Config = toml::from_str(&saved).unwrap();
        let routing = config.openrouter.routing.unwrap();
        assert_eq!(routing.zdr, Some(true));
        assert_eq!(routing.sort, Some(ProviderSort::Throughput));
        assert_eq!(routing.order, existing.order);
        assert_eq!(routing.data_collection, existing.data_collection);
        assert_eq!(routing.allow_fallbacks, existing.allow_fallbacks);
    }

    #[test]
    fn openrouter_setup_preselects_existing_choices_and_allows_defaults() {
        let existing = ProviderPreferences {
            zdr: Some(true),
            sort: Some(ProviderSort::Latency),
            ..Default::default()
        };
        let routing =
            prompt_openrouter_routing(Some(&existing), |picker| Ok(Pick::Chosen(picker.respond())))
                .unwrap()
                .chosen()
                .unwrap();
        assert_eq!(routing.zdr, existing.zdr);
        assert_eq!(routing.sort, existing.sort);

        let routing = prompt_openrouter_routing(None, |picker| Ok(Pick::Chosen(picker.respond())))
            .unwrap()
            .chosen()
            .unwrap();
        assert_eq!(routing.zdr, Some(false));
        assert_eq!(routing.sort, None);
    }

    #[test]
    fn cancelling_routing_setup_does_not_change_existing_settings() {
        let existing = ProviderPreferences {
            zdr: Some(true),
            sort: Some(ProviderSort::Price),
            ..Default::default()
        };
        assert!(
            prompt_openrouter_routing(Some(&existing), |_| Ok(Pick::Back))
                .unwrap()
                .chosen()
                .is_none()
        );
        let mut calls = 0;
        let cancelled = prompt_openrouter_routing(Some(&existing), |picker| {
            calls += 1;
            if calls == 1 {
                picker.selected = 1;
                Ok(Pick::Chosen(picker.respond()))
            } else {
                Ok(Pick::Back)
            }
        })
        .unwrap();
        assert!(matches!(cancelled, Pick::Back));
        assert_eq!(existing.zdr, Some(true));
        assert_eq!(existing.sort, Some(ProviderSort::Price));
    }

    #[test]
    fn esc_on_routing_sort_returns_to_zdr() {
        let mut calls = 0;
        let routing = prompt_openrouter_routing(None, |picker| {
            calls += 1;
            match calls {
                1 => picker.selected = 0,
                2 => return Ok(Pick::Back),
                3 => {
                    assert_eq!(picker.selected, 0, "ZDR keeps the earlier answer");
                    picker.selected = 1;
                }
                _ => picker.selected = 3,
            }
            Ok(Pick::Chosen(picker.respond()))
        })
        .unwrap()
        .chosen()
        .unwrap();
        assert_eq!(calls, 4);
        assert_eq!(routing.zdr, Some(false));
        assert_eq!(routing.sort, Some(ProviderSort::Price));
    }

    #[test]
    fn quit_on_routing_sort_leaves_instead_of_going_back() {
        let mut calls = 0;
        let result = prompt_openrouter_routing(None, |picker| {
            calls += 1;
            if calls == 1 {
                Ok(Pick::Chosen(picker.respond()))
            } else {
                Ok(Pick::Quit)
            }
        })
        .unwrap();
        assert!(matches!(result, Pick::Quit));
        assert_eq!(calls, 2);
    }

    #[test]
    fn model_picker_quits_on_second_ctrl_c_without_searching() {
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let models = [];
        let mut picker = ModelPicker::new(&models);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(picker.handle_key(ctrl_c).is_none());
        assert!(picker.query.is_empty());
        assert!(matches!(picker.handle_key(ctrl_c), Some(Pick::Quit)));
    }

    #[test]
    fn wizard_skips_steps_that_do_not_apply() {
        let config = Config::default();
        let mut wizard = Wizard::new(&config);

        // Ollama needs no key and finishes after selecting a model without reasoning.
        assert_eq!(wizard.next_after(Step::Provider), Some(Step::Model));
        assert_eq!(wizard.next_after(Step::Model), None);

        // OpenRouter asks for an API key and finishes with routing.
        wizard.provider_idx = 1;
        assert_eq!(wizard.next_after(Step::Provider), Some(Step::ApiKey));
        assert_eq!(wizard.next_after(Step::Model), Some(Step::Routing));
        assert_eq!(wizard.next_after(Step::Routing), None);

        let config = Config {
            openrouter_api_key: Some("example-key".into()),
            ..Config::default()
        };
        let mut wizard = Wizard::new(&config);
        wizard.provider_idx = 1;
        assert_eq!(wizard.next_after(Step::Provider), Some(Step::Model));
    }

    fn setup_model(reasoning: bool) -> ModelInfo {
        ModelInfo {
            image: Default::default(),
            id: "example/model".into(),
            name: None,
            created_at: None,
            context_length: None,
            max_completion_tokens: None,
            capabilities: ModelCapabilities {
                reasoning,
                ..Default::default()
            },
            categories: vec![ModelCategory::Chat],
            pricing: Default::default(),
        }
    }

    #[test]
    fn model_browser_filters_input_and_output_prices_per_million() {
        use rust_decimal::Decimal;
        let mut model = setup_model(false);
        model.id = "openrouter:example/model".into();
        model.pricing.prompt = Some(Decimal::new(1, 6));
        model.pricing.completion = Some(Decimal::new(5, 6));
        assert!(matches_catalog_query(&model, "openrouter in<=1 out<=5"));
        assert!(!matches_catalog_query(&model, "in<1"));
        assert!(!matches_catalog_query(&model, "out<5"));
        assert!(matches_catalog_query(&model, "in>=1 out>4"));
        model.pricing.prompt = None;
        assert!(!matches_catalog_query(&model, "in<=1"));
        assert!(!matches_catalog_query(&model, "out<=oops"));
    }

    #[test]
    fn model_browser_filters_reported_features_with_prices_and_names() {
        let mut model = setup_model(true);
        model.id = "mistral:example".into();
        model.capabilities.images = true;
        model.capabilities.tool_calling = true;
        model.categories.push(ModelCategory::Tts);
        model.pricing.prompt = Some(rust_decimal::Decimal::new(1, 6));
        assert!(matches_catalog_query(
            &model,
            "mistral cap:vision cap:tools cap:speech in<=1"
        ));
        assert!(!matches_catalog_query(&model, "cap:transcribe"));
        assert!(!matches_catalog_query(&model, "cap:unknown"));
        assert_eq!(
            model_features(&model),
            vec!["chat", "tools", "reasoning", "vision", "speech"]
        );
    }

    #[test]
    fn model_details_show_provider_image_options_with_cli_flags() {
        let mut model = setup_model(false);
        model.id = "openrouter:example/model".into();
        model.image.parameters.insert(
            "resolution".into(),
            serde_json::json!({"type":"enum","values":["1K","2K"]}),
        );
        model.image.parameters.insert(
            "input_references".into(),
            serde_json::json!({"type":"range","min":0,"max":4}),
        );
        let details = model_details(&model);
        assert!(details.contains("resolution (--size / --resolution): 1K, 2K"));
        assert!(details.contains("input_references (--reference / --image): 0 to 4"));
        assert!(details.contains("combined across endpoints"));
    }

    #[test]
    fn model_browser_filters_columns_with_keyboard() {
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use rust_decimal::Decimal;
        let models: Vec<_> = [
            ("example/a", Some(1), true),
            ("example/b", Some(3), true),
            ("example/c", None, false),
        ]
        .into_iter()
        .map(|(id, price, vision)| ModelInfo {
            id: id.into(),
            pricing: flashmind_types::ModelPricing {
                prompt: price.map(|value| Decimal::new(value, 6)),
                completion: Some(Decimal::new(5, 6)),
                ..Default::default()
            },
            capabilities: ModelCapabilities {
                images: vision,
                ..Default::default()
            },
            ..setup_model(false)
        })
        .collect();
        let mut picker = ModelPicker::new(&models);
        picker.browse = true;
        picker.refilter();
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        picker.handle_key(key(KeyCode::Tab));
        picker.handle_key(key(KeyCode::Tab));
        picker.handle_key(key(KeyCode::Tab));
        assert_eq!(picker.focus, CatalogFocus::Input);
        picker.handle_key(key(KeyCode::Char('1')));
        assert_eq!(picker.filtered, [0]);
        picker.handle_key(key(KeyCode::Tab));
        picker.handle_key(key(KeyCode::Char('4')));
        assert!(picker.filtered.is_empty());
        picker.handle_key(key(KeyCode::Backspace));
        picker.handle_key(key(KeyCode::Char('5')));
        assert_eq!(picker.filtered, [0]);
        picker.handle_key(key(KeyCode::Tab));
        for _ in 0..3 {
            picker.handle_key(key(KeyCode::Right));
        }
        picker.handle_key(key(KeyCode::Char(' ')));
        assert_eq!(picker.features, ["vision"]);
        picker.handle_key(key(KeyCode::BackTab));
        picker.handle_key(key(KeyCode::BackTab));
        picker.handle_key(key(KeyCode::Backspace));
        assert_eq!(picker.filtered, [0, 1]);
        picker.handle_key(key(KeyCode::BackTab));
        picker.handle_key(key(KeyCode::BackTab));
        picker.handle_key(key(KeyCode::BackTab));
        assert_eq!(picker.focus, CatalogFocus::Model);
        picker.handle_key(key(KeyCode::Char('b')));
        assert_eq!(picker.filtered, [1]);
    }

    #[test]
    fn browser_sorts_by_price_and_release_date_and_opens_details() {
        use chrono::DateTime;
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use rust_decimal::Decimal;
        let models: Vec<_> = [
            ("a", Some(3), Some(100)),
            ("b", Some(1), Some(300)),
            ("c", None, None),
        ]
        .into_iter()
        .map(|(id, price, date)| ModelInfo {
            id: id.into(),
            created_at: date.and_then(|value| DateTime::from_timestamp(value, 0)),
            pricing: flashmind_types::ModelPricing {
                prompt: price.map(|value| Decimal::new(value, 6)),
                ..Default::default()
            },
            ..setup_model(false)
        })
        .collect();
        let mut picker = ModelPicker::new(&models);
        picker.browse = true;
        picker.focus = CatalogFocus::Input;
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        picker.handle_key(key(KeyCode::Enter));
        assert_eq!(picker.filtered, [1, 0, 2]);
        picker.handle_key(key(KeyCode::Enter));
        assert_eq!(picker.filtered, [0, 1, 2]);
        picker.focus = CatalogFocus::Created;
        picker.handle_key(key(KeyCode::Enter));
        assert_eq!(picker.filtered, [0, 1, 2]);
        picker.handle_key(key(KeyCode::Enter));
        assert_eq!(picker.filtered, [1, 0, 2]);
        picker.handle_key(key(KeyCode::F(2)));
        assert!(picker.details);
        picker.handle_key(key(KeyCode::Esc));
        assert!(!picker.details);
    }

    #[test]
    fn model_browser_searches_provider_and_includes_media_models() {
        let models = vec![
            ModelInfo {
                id: "openrouter:example/image".into(),
                categories: vec![ModelCategory::ImageGeneration],
                ..setup_model(false)
            },
            ModelInfo {
                id: "ollama:example/chat".into(),
                ..setup_model(false)
            },
        ];
        let mut picker = ModelPicker::new(&models);
        picker.browse = true;
        picker.refilter();
        assert_eq!(picker.filtered.len(), 2);
        picker.query = "OPENROUTER".into();
        picker.refilter();
        assert_eq!(picker.filtered, [0]);
    }

    #[test]
    fn model_picker_excludes_media_only_models_even_when_searching() {
        let models: Vec<ModelInfo> = [
            vec![ModelCategory::ImageGeneration],
            vec![ModelCategory::Chat],
            vec![ModelCategory::VideoGeneration],
            vec![
                ModelCategory::Chat,
                ModelCategory::Vision,
                ModelCategory::ImageGeneration,
            ],
            vec![ModelCategory::Tts],
            vec![ModelCategory::Stt],
        ]
        .into_iter()
        .map(|categories| ModelInfo {
            categories,
            ..setup_model(false)
        })
        .collect();
        let mut picker = ModelPicker::new(&models);
        assert_eq!(picker.filtered, [1, 3]);
        picker.query = "example".into();
        picker.refilter();
        assert_eq!(picker.filtered, [1, 3]);
    }

    #[test]
    fn model_picker_orders_newest_first_and_keeps_order_when_searching() {
        use chrono::DateTime;
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let models: Vec<ModelInfo> = [
            ("unknown/z", None),
            ("dated/old", Some(100)),
            ("dated/z", Some(300)),
            ("dated/a", Some(300)),
            ("unknown/a", None),
        ]
        .into_iter()
        .map(|(id, created)| ModelInfo {
            id: id.into(),
            created_at: created.and_then(|timestamp| DateTime::from_timestamp(timestamp, 0)),
            ..setup_model(false)
        })
        .collect();
        let mut picker = ModelPicker::new(&models);
        assert_eq!(picker.filtered, [3, 2, 1, 4, 0]);

        for c in "dated".chars() {
            picker.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert_eq!(picker.filtered, [3, 2, 1]);
        picker.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert!(matches!(
            picker.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(Pick::Chosen(2))
        ));

        for _ in 0..5 {
            picker.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        }
        assert_eq!(picker.filtered, [3, 2, 1, 4, 0]);
        assert!(matches!(
            picker.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(Pick::Chosen(3))
        ));
    }

    #[test]
    fn wizard_includes_reasoning_only_when_supported() {
        let config = Config::default();
        let mut wizard = Wizard::new(&config);
        wizard.models = vec![setup_model(true)];
        wizard.model_idx = Some(0);
        wizard.reasoning_idx = 3;
        assert_eq!(wizard.next_after(Step::Model), Some(Step::Reasoning));
        assert_eq!(wizard.next_after(Step::Reasoning), None);
        assert_eq!(
            wizard.config_for_save().unwrap().reasoning,
            Some(ReasoningLevel::High)
        );

        wizard.models[0].capabilities.reasoning = false;
        assert_eq!(wizard.next_after(Step::Model), None);
        assert_eq!(wizard.config_for_save().unwrap().reasoning, None);
    }

    #[test]
    fn setup_save_preserves_optional_settings_and_other_provider_keys() {
        let config: Config = toml::from_str(
            r#"
            system_prompt = "Example prompt"
            reasoning = "high"
            openrouter_api_key = "example-openrouter-key"
            anthropic_api_key = "example-anthropic-key"
            openai_base_url = "https://example.com/v1"
            vision_model = "ollama:example-vision"
            skill_dirs = ["/path/to/skills"]
            pet = "off"
            [openrouter.routing]
            zdr = true
            sort = "throughput"
            "#,
        )
        .unwrap();
        let mut wizard = Wizard::new(&config);
        wizard.provider_idx = 3;
        wizard.entered_key = Some("example-openai-key".into());
        wizard.models = vec![setup_model(false)];
        wizard.model_idx = Some(0);
        let saved: Config =
            toml::from_str(&toml::to_string(&wizard.config_for_save().unwrap()).unwrap()).unwrap();
        assert_eq!(saved.model.as_deref(), Some("openai:example/model"));
        assert!(saved.reasoning.is_none());
        assert_eq!(
            saved.api_key_for(&Provider::OpenAi),
            Some("example-openai-key")
        );
        let mut expected = config;
        expected.model = saved.model.clone();
        expected.reasoning = None;
        expected
            .providers
            .entry("openai".into())
            .or_default()
            .api_key = Some(crate::ApiKey("example-openai-key".into()));
        assert_eq!(
            serde_json::to_value(&saved).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
}
