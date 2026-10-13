use std::io;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ratatui::widgets::Paragraph;

use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_tui::widgets::{
    ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse, EXIT_HINT, ExitConfirm,
};
use flashmind_types::{LlmProvider, Provider, ProviderPreferences, ProviderSort, ReasoningLevel};

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

struct ModelPicker<'a> {
    models: &'a [flashmind_types::ModelInfo],
    filtered: Vec<usize>,
    selected: usize,
    scroll: usize,
    query: String,
    visible: usize,
    esc_hint: &'static str,
    exit: ExitConfirm,
}

impl<'a> ModelPicker<'a> {
    const VISIBLE: usize = 10;

    fn new(models: &'a [flashmind_types::ModelInfo]) -> Self {
        let filtered: Vec<usize> = (0..models.len()).collect();
        Self {
            models,
            filtered,
            selected: 0,
            scroll: 0,
            query: String::new(),
            visible: Self::VISIBLE,
            esc_hint: "cancel",
            exit: ExitConfirm::default(),
        }
    }

    fn refilter(&mut self) {
        let q = self.query.to_lowercase();
        self.filtered = (0..self.models.len())
            .filter(|&i| {
                if q.is_empty() {
                    return true;
                }
                let m = &self.models[i];
                let name = m.name.as_deref().unwrap_or("");
                let id = &m.id;
                name.to_lowercase().contains(&q) || id.to_lowercase().contains(&q)
            })
            .collect();
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
        match key.code {
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
            KeyCode::Enter => {
                if self.filtered.is_empty() {
                    None
                } else {
                    Some(Pick::Chosen(self.filtered[self.selected]))
                }
            }
            KeyCode::Esc => Some(Pick::Back),
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
                None
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.query.push(c);
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
            Span::styled("  Search: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}_", self.query),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(format!("  ({} models)", self.filtered.len()), dim),
        ]));
        lines.push(Line::default());

        if self.filtered.is_empty() {
            lines.push(Line::from(Span::styled("  No matching models.", dim)));
            lines.push(Line::default());
            lines.push(self.hint(format!("  Type to search  Esc: {}", self.esc_hint)));
            return lines;
        }

        let vis_end = (self.scroll + self.visible).min(self.filtered.len());
        let visible = &self.filtered[self.scroll..vis_end];

        let w = max_width as usize;
        let hdr_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let name_w = (w / 3).max(20);
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<name_w$}", "Model"), hdr_style),
            Span::styled(format!("{:>7}", "Ctx"), hdr_style),
            Span::styled(format!("{:>10}", "In $/M"), hdr_style),
            Span::styled(format!("{:>10}", "Out $/M"), hdr_style),
            Span::styled("  Features", hdr_style),
        ]));
        lines.push(Line::from(Span::styled(
            format!("  {}", "─".repeat(w.saturating_sub(2).min(100))),
            dim,
        )));

        for (i, &model_idx) in visible.iter().enumerate() {
            let abs_i = self.scroll + i;
            let is_sel = abs_i == self.selected;
            let m = &self.models[model_idx];

            let name = m.name.as_deref().unwrap_or(&m.id);
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

            let mut feats = Vec::new();
            if m.capabilities.tool_calling {
                feats.push("tools");
            }
            if m.capabilities.reasoning {
                feats.push("reason");
            }
            if m.capabilities.images {
                feats.push("vision");
            }
            if m.capabilities.audio {
                feats.push("audio");
            }
            let feats_str = feats.join(", ");

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
        lines.push(self.hint(format!(
            "  Type to search  ↑/↓: navigate  Enter: select  Esc: {}",
            self.esc_hint
        )));

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
    Ok(pick_model(tui, models, None, "cancel")?.chosen())
}

fn pick_model(
    tui: &mut flashmind_tui::Tui,
    models: &[flashmind_types::ModelInfo],
    initial: Option<usize>,
    esc_hint: &'static str,
) -> io::Result<Pick<usize>> {
    use ratatui::crossterm::event::{self, Event};

    let mut picker = ModelPicker::new(models);
    picker.esc_hint = esc_hint;
    if let Some(i) = initial.and_then(|idx| picker.filtered.iter().position(|&m| m == idx)) {
        picker.selected = i;
    }
    let mut screen = tui.fullscreen()?;
    loop {
        screen.draw(|frame| {
            picker.visible = (frame.area().height as usize).saturating_sub(7).max(1);
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
            _ => "API",
        }
    }

    fn existing_key(&self) -> Option<String> {
        match self.provider() {
            Provider::OpenRouter => self.config.openrouter_api_key.clone(),
            Provider::Anthropic => self.config.anthropic_api_key.clone(),
            Provider::OpenAi => self.config.openai_api_key.clone(),
            _ => None,
        }
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
                let idx = chosen!(pick_model(tui, &self.models, self.model_idx, esc_hint)?);
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
        match provider {
            Provider::OpenRouter => {
                config.openrouter_api_key = self.api_key();
                config.openrouter.routing = self.routing.clone();
            }
            Provider::Anthropic => config.anthropic_api_key = self.api_key(),
            Provider::OpenAi => config.openai_api_key = self.api_key(),
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
            id: "example/model".into(),
            name: None,
            context_length: None,
            max_completion_tokens: None,
            capabilities: ModelCapabilities {
                reasoning,
                ..Default::default()
            },
            categories: Vec::new(),
            pricing: Default::default(),
        }
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
            command_allowlist = ["git status"]
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
        assert_eq!(saved.openai_api_key.as_deref(), Some("example-openai-key"));
        let mut expected = config;
        expected.model = saved.model.clone();
        expected.reasoning = None;
        expected.openai_api_key = Some("example-openai-key".into());
        assert_eq!(
            serde_json::to_value(&saved).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
}
