use std::io;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse};
use flashmind_types::{LlmProvider, Provider, ReasoningLevel};

use crate::config::{Config, config_dir};

// ---------------------------------------------------------------------------
// Choice picker helper
// ---------------------------------------------------------------------------

pub fn run_choice(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<Option<ChoiceResponse>> {
    loop {
        match run_choice_action(tui, picker)? {
            ChoicePickerAction::Select(r) => return Ok(Some(r)),
            ChoicePickerAction::Cancel => return Ok(None),
            ChoicePickerAction::Delete(_) => {}
        }
    }
}

pub fn run_choice_action(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<ChoicePickerAction> {
    use ratatui::crossterm::event::{self, Event};

    let _raw = tui.raw_mode()?;
    let mut drawn = tui.draw_lines(&picker.lines())?;

    loop {
        if !event::poll(std::time::Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if let Some(action) = picker.handle_key(key) {
            tui.erase(drawn)?;
            drop(_raw);
            return Ok(action);
        }
        tui.erase(drawn)?;
        drawn = tui.draw_lines(&picker.lines())?;
    }
}

fn prompt_api_key(
    tui: &mut flashmind_tui::Tui,
    label: &str,
    existing: &Option<String>,
) -> Result<Option<String>> {
    if let Some(key) = existing {
        tui.println(&ratatui::text::Line::from(format!(
            "  Using existing {label} API key from config/env."
        )))?;
        return Ok(Some(key.clone()));
    }
    let mut key_picker = ChoicePicker::new(
        format!("Enter {label} API key"),
        vec![ChoiceOption {
            label: "API key".into(),
            accepts_input: true,
        }],
    );
    match run_choice(tui, &mut key_picker)? {
        Some(resp) if !resp.input.is_empty() => Ok(Some(resp.input)),
        _ => Ok(None),
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
        } else if self.selected >= self.scroll + Self::VISIBLE {
            self.scroll = self.selected + 1 - Self::VISIBLE;
        }
    }

    fn handle_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> Option<Option<usize>> {
        use ratatui::crossterm::event::KeyCode;
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
                    Some(Some(self.filtered[self.selected]))
                }
            }
            KeyCode::Esc => Some(None),
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
                None
            }
            KeyCode::Char(c) => {
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
            lines.push(Line::from(Span::styled(
                "  Type to search  Esc: cancel",
                dim,
            )));
            return lines;
        }

        let vis_end = (self.scroll + Self::VISIBLE).min(self.filtered.len());
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
            format!("  {}", "─".repeat((w - 2).min(100))),
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
        if total > Self::VISIBLE {
            let has_above = self.scroll > 0;
            let has_below = self.scroll + Self::VISIBLE < total;
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
        lines.push(Line::from(Span::styled(
            "  Type to search  ↑/↓: navigate  Enter: select  Esc: cancel",
            dim,
        )));

        lines
    }
}

pub fn run_model_picker(
    tui: &mut flashmind_tui::Tui,
    models: &[flashmind_types::ModelInfo],
) -> io::Result<Option<usize>> {
    use ratatui::crossterm::event::{self, Event};

    let width = tui.width()?;
    let mut picker = ModelPicker::new(models);
    let _raw = tui.raw_mode()?;
    let mut drawn = tui.draw_lines(&picker.lines(width))?;

    loop {
        if !event::poll(std::time::Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if let Some(result) = picker.handle_key(key) {
            tui.erase(drawn)?;
            drop(_raw);
            return Ok(result);
        }
        drawn = tui.redraw_lines(&picker.lines(width), drawn)?;
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

pub async fn run_setup(config: &Config) -> Result<()> {
    let mut tui = flashmind_tui::Tui::new();

    // 1. Pick provider
    let options: Vec<ChoiceOption> = PROVIDERS
        .iter()
        .map(|(label, _)| ChoiceOption {
            label: label.to_string(),
            accepts_input: false,
        })
        .collect();
    let mut picker = ChoicePicker::new("Choose a provider".into(), options);

    let provider = match run_choice(&mut tui, &mut picker)? {
        Some(resp) => {
            tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                format!("  → {}", resp.label),
                flashmind_tui::styles::S_AGENT,
            )))?;
            PROVIDERS[resp.selected].1
        }
        None => {
            tui.println(&ratatui::text::Line::from("Setup cancelled."))?;
            return Ok(());
        }
    };

    // 2. Get API key if needed
    let api_key = match provider {
        Provider::Ollama => None,
        _ => {
            let env_key = match provider {
                Provider::OpenRouter => config.openrouter_api_key.clone(),
                Provider::Anthropic => config.anthropic_api_key.clone(),
                Provider::OpenAi => config.openai_api_key.clone(),
                _ => None,
            };
            if let Some(key) = env_key {
                tui.println(&ratatui::text::Line::from(
                    "  Using existing API key from config/env.",
                ))?;
                Some(key)
            } else {
                let label = match provider {
                    Provider::OpenRouter => "OpenRouter",
                    Provider::Anthropic => "Anthropic",
                    Provider::OpenAi => "OpenAI",
                    _ => "API",
                };
                let mut key_picker = ChoicePicker::new(
                    format!("Enter {label} API key"),
                    vec![ChoiceOption {
                        label: "API key".into(),
                        accepts_input: true,
                    }],
                );
                match run_choice(&mut tui, &mut key_picker)? {
                    Some(resp) if !resp.input.is_empty() => Some(resp.input),
                    _ => bail!("API key is required for {label}"),
                }
            }
        }
    };

    // 3. Build a temporary provider to list models
    let llm_provider: Arc<dyn LlmProvider> = match provider {
        Provider::Ollama => Arc::new(OllamaProvider::new(config.ollama_url.clone(), None)?),
        Provider::OpenRouter => Arc::new(OpenRouterProvider::new(api_key.clone().unwrap())),
        Provider::Anthropic => Arc::new(AnthropicProvider::new(api_key.clone().unwrap())),
        Provider::OpenAi => {
            let base = config
                .openai_base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1");
            Arc::new(
                OpenAiProvider::builder(base)
                    .api_key(api_key.clone().unwrap())
                    .build()?,
            )
        }
        other => bail!("unsupported provider: {other:?}"),
    };

    tui.println(&ratatui::text::Line::from(format!(
        "\n  Fetching models from {}...",
        llm_provider.name()
    )))?;
    let models = llm_provider
        .list_models()
        .await
        .context("failed to list models (check your API key and network)")?;

    if models.is_empty() {
        bail!("no models found — check your API key or that the provider is running");
    }

    // 4. Pick model
    let model_idx = match run_model_picker(&mut tui, &models)? {
        Some(idx) => idx,
        None => {
            tui.println(&ratatui::text::Line::from("Setup cancelled."))?;
            return Ok(());
        }
    };
    let chosen_model = &models[model_idx];
    let model_id = &chosen_model.id;

    // 5. Pick reasoning level
    let reasoning = if chosen_model.capabilities.reasoning {
        let reasoning_options = vec![
            ChoiceOption {
                label: "Off".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "Low".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "Medium".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "High".into(),
                accepts_input: false,
            },
        ];
        let mut reasoning_picker =
            ChoicePicker::new("Thinking / reasoning level".into(), reasoning_options);
        match run_choice(&mut tui, &mut reasoning_picker)? {
            Some(resp) => match resp.selected {
                1 => ReasoningLevel::Low,
                2 => ReasoningLevel::Medium,
                3 => ReasoningLevel::High,
                _ => ReasoningLevel::Off,
            },
            None => ReasoningLevel::Off,
        }
    } else {
        ReasoningLevel::Off
    };

    // 6. Memory embedding provider
    let has_openrouter_key = provider == Provider::OpenRouter
        || api_key
            .as_ref()
            .filter(|_| provider == Provider::OpenRouter)
            .is_some()
        || config.openrouter_api_key.is_some();
    let has_openai_key = provider == Provider::OpenAi || config.openai_api_key.is_some();

    let mut memory_provider_name: Option<String> = None;
    let mut memory_model_name: Option<String> = None;
    let mut new_openrouter_key = None;
    let mut new_openai_key = None;

    {
        let (or_label, oai_label) = if has_openrouter_key {
            ("OpenRouter (key available)", "OpenAI")
        } else if has_openai_key {
            ("OpenRouter", "OpenAI (key available)")
        } else {
            ("OpenRouter", "OpenAI")
        };

        let memory_options = vec![
            ChoiceOption {
                label: or_label.into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: oai_label.into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "None".into(),
                accepts_input: false,
            },
        ];
        let mut memory_picker =
            ChoicePicker::new("Long-term memory (embeddings)".into(), memory_options);
        if let Some(resp) = run_choice(&mut tui, &mut memory_picker)? {
            match resp.selected {
                0 => {
                    memory_provider_name = Some("openrouter".into());
                    memory_model_name = Some("openai/text-embedding-3-small".into());

                    if !has_openrouter_key {
                        new_openrouter_key = prompt_api_key(&mut tui, "OpenRouter", &None)?;
                        if new_openrouter_key.is_none() {
                            bail!("OpenRouter API key required for memory embeddings");
                        }
                    }

                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → memory via OpenRouter (openai/text-embedding-3-small)",
                        flashmind_tui::styles::S_AGENT,
                    )))?;
                }
                1 => {
                    memory_provider_name = Some("openai".into());
                    memory_model_name = Some("text-embedding-3-small".into());

                    if !has_openai_key {
                        new_openai_key = prompt_api_key(&mut tui, "OpenAI", &None)?;
                        if new_openai_key.is_none() {
                            bail!("OpenAI API key required for memory embeddings");
                        }
                    }

                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → memory via OpenAI (text-embedding-3-small)",
                        flashmind_tui::styles::S_AGENT,
                    )))?;
                }
                _ => {
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → no memory",
                        flashmind_tui::styles::S_DIM,
                    )))?;
                }
            }
        }
    }

    // 7. Web search
    let mut new_brave_key = config.brave_api_key.clone();
    let mut new_firecrawl_key = config.firecrawl_api_key.clone();

    {
        let search_options = vec![
            ChoiceOption {
                label: "Brave Search".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "Firecrawl".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "None".into(),
                accepts_input: false,
            },
        ];
        let mut search_picker = ChoicePicker::new("Web search provider".into(), search_options);
        if let Some(resp) = run_choice(&mut tui, &mut search_picker)? {
            match resp.selected {
                0 => {
                    if new_brave_key.is_none() {
                        new_brave_key = prompt_api_key(&mut tui, "Brave Search", &None)?;
                        if new_brave_key.is_none() {
                            bail!("Brave API key is required");
                        }
                    } else {
                        tui.println(&ratatui::text::Line::from(
                            "  Using existing Brave API key from config/env.",
                        ))?;
                    }
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → Brave Search",
                        flashmind_tui::styles::S_AGENT,
                    )))?;
                }
                1 => {
                    if new_firecrawl_key.is_none() {
                        new_firecrawl_key = prompt_api_key(&mut tui, "Firecrawl", &None)?;
                        if new_firecrawl_key.is_none() {
                            bail!("Firecrawl API key is required");
                        }
                    } else {
                        tui.println(&ratatui::text::Line::from(
                            "  Using existing Firecrawl API key from config/env.",
                        ))?;
                    }
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → Firecrawl",
                        flashmind_tui::styles::S_AGENT,
                    )))?;
                }
                _ => {
                    new_brave_key = None;
                    new_firecrawl_key = None;
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  → no web search",
                        flashmind_tui::styles::S_DIM,
                    )))?;
                }
            }
        }
    }

    // 8. Write config
    let prefix = match provider {
        Provider::Ollama => "ollama",
        Provider::OpenRouter => "openrouter",
        Provider::Anthropic => "anthropic",
        Provider::OpenAi => "openai",
        _ => "ollama",
    };
    let model_str = format!("{prefix}:{model_id}");

    let config_path = config_dir()?.join("config.toml");
    let mut out: Vec<String> = Vec::new();
    out.push(format!("model = {model_str:?}"));
    if reasoning.is_on() {
        out.push(format!("reasoning = {:?}", reasoning.to_string()));
    }

    let final_openrouter_key = new_openrouter_key
        .or_else(|| api_key.clone().filter(|_| provider == Provider::OpenRouter))
        .or_else(|| config.openrouter_api_key.clone());
    let final_anthropic_key = api_key
        .clone()
        .filter(|_| provider == Provider::Anthropic)
        .or_else(|| config.anthropic_api_key.clone());
    let final_openai_key = new_openai_key
        .or_else(|| api_key.clone().filter(|_| provider == Provider::OpenAi))
        .or_else(|| config.openai_api_key.clone());

    if let Some(ref key) = final_openrouter_key {
        out.push(format!("openrouter_api_key = {key:?}"));
    }
    if let Some(ref key) = final_anthropic_key {
        out.push(format!("anthropic_api_key = {key:?}"));
    }
    if let Some(ref key) = final_openai_key {
        out.push(format!("openai_api_key = {key:?}"));
    }
    if let Some(ref url) = config.ollama_url {
        out.push(format!("ollama_url = {url:?}"));
    }

    if let Some(ref key) = new_brave_key {
        out.push(format!("brave_api_key = {key:?}"));
    }
    if let Some(ref key) = new_firecrawl_key {
        out.push(format!("firecrawl_api_key = {key:?}"));
    }

    if let Some(ref mp) = memory_provider_name {
        out.push(format!("memory_provider = {mp:?}"));
    }
    if let Some(ref mm) = memory_model_name {
        out.push(format!("memory_model = {mm:?}"));
    }

    if let Some(ref prompt) = config.system_prompt {
        out.push(format!("system_prompt = {prompt:?}"));
    }

    let content = out.join("\n") + "\n";
    std::fs::write(&config_path, &content)?;

    tui.println(&ratatui::text::Line::default())?;
    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Saved to {}", config_path.display()),
        flashmind_tui::styles::S_AGENT,
    )))?;
    tui.println(&ratatui::text::Line::from(format!("  Model: {model_str}")))?;
    if memory_provider_name.is_some() {
        tui.println(&ratatui::text::Line::from("  Memory: enabled"))?;
    }
    if new_brave_key.is_some() {
        tui.println(&ratatui::text::Line::from("  Search: Brave"))?;
    } else if new_firecrawl_key.is_some() {
        tui.println(&ratatui::text::Line::from("  Search: Firecrawl"))?;
    }
    tui.println(&ratatui::text::Line::from(
        "  Run `flsh` to start chatting.",
    ))?;

    Ok(())
}
