use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use serde::Deserialize;

use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry, EntryKind};
use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_memory::session::{self, SessionEntry, SessionEntryKind, SessionStore};
use flashmind_tools::{ToolBuilder, protected::ProtectedPaths};
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse};
use flashmind_types::{AgentEvent, AgentInput, AgentLlmConfig, LlmProvider, Model, Provider};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "flashmind", about = "AI chat & coding agent")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Model (e.g. "ollama:llama3.2", "openrouter:anthropic/claude-sonnet-4")
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// One-shot prompt — prints the response and exits
    #[arg(short, long)]
    prompt: Option<String>,

    /// System prompt override
    #[arg(long)]
    system: Option<String>,

    /// Skip restoring the previous session
    #[arg(long)]
    no_restore: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive setup — choose a provider and model
    Setup,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
struct Config {
    model: Option<String>,
    system_prompt: Option<String>,

    openrouter_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
    ollama_url: Option<String>,

    brave_api_key: Option<String>,
    firecrawl_api_key: Option<String>,
}

fn config_dir() -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .context("cannot determine home directory")?
        .join(".flashmind");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

const DEFAULT_CONFIG: &str = r#"# model = "ollama:llama3.2"
# system_prompt = "You are a helpful coding assistant."

# openrouter_api_key = ""
# anthropic_api_key = ""
# openai_api_key = ""
# ollama_url = "http://localhost:11434"

# brave_api_key = ""
# firecrawl_api_key = ""
"#;

fn load_config() -> Result<Config> {
    let path = config_dir()?.join("config.toml");
    let mut config: Config = if path.exists() {
        let text = std::fs::read_to_string(&path)?;
        toml::from_str(&text).with_context(|| format!("invalid config: {}", path.display()))?
    } else {
        std::fs::write(&path, DEFAULT_CONFIG)?;
        Config::default()
    };

    // Env vars override config file.
    macro_rules! env_override {
        ($field:ident, $var:literal) => {
            if config.$field.is_none() {
                config.$field = std::env::var($var).ok().filter(|s| !s.is_empty());
            }
        };
    }
    env_override!(openrouter_api_key, "OPENROUTER_API_KEY");
    env_override!(anthropic_api_key, "ANTHROPIC_API_KEY");
    env_override!(openai_api_key, "OPENAI_API_KEY");
    env_override!(openai_base_url, "OPENAI_BASE_URL");
    env_override!(ollama_url, "OLLAMA_URL");
    env_override!(brave_api_key, "BRAVE_API_KEY");
    env_override!(firecrawl_api_key, "FIRECRAWL_API_KEY");

    Ok(config)
}

// ---------------------------------------------------------------------------
// Provider construction
// ---------------------------------------------------------------------------

fn resolve_model(cli: &Cli, config: &Config) -> Result<Model> {
    let raw = cli
        .model
        .as_deref()
        .or(config.model.as_deref())
        .unwrap_or("ollama:llama3.2");
    raw.parse()
        .with_context(|| format!("invalid model string: {raw}"))
}

fn build_provider(model: &Model, config: &Config) -> Result<Arc<dyn LlmProvider>> {
    match model.provider {
        Provider::Ollama => {
            let p = OllamaProvider::new(config.ollama_url.clone(), None)?;
            Ok(Arc::new(p))
        }
        Provider::OpenRouter => {
            let key = config
                .openrouter_api_key
                .as_ref()
                .context("openrouter_api_key required for OpenRouter models")?;
            Ok(Arc::new(OpenRouterProvider::new(key.clone())))
        }
        Provider::Anthropic => {
            let key = config
                .anthropic_api_key
                .as_ref()
                .context("anthropic_api_key required for Anthropic models")?;
            Ok(Arc::new(AnthropicProvider::new(key.clone())))
        }
        Provider::OpenAi => {
            let key = config
                .openai_api_key
                .as_ref()
                .context("openai_api_key required for OpenAI models")?;
            let base = config
                .openai_base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1");
            let p = OpenAiProvider::builder(base).api_key(key).build()?;
            Ok(Arc::new(p))
        }
        other => bail!("unsupported provider: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn build_tools(config: &Config) -> flashmind_types::ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let protected = Arc::new(ProtectedPaths::new(&cwd));
    ToolBuilder::new()
        .file_ops(None, &protected)
        .bash(vec![], &protected, vec![], None)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .time()
        .sqlite()
        .http()
        .json()
        .build()
}

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------

const CHAT_KEY: &str = "default";

async fn open_session_store() -> Result<SessionStore> {
    let db_path = config_dir()?.join("sessions.db");
    let conn = tokio_rusqlite::Connection::open(db_path).await?;
    conn.call(
        |c| -> std::result::Result<(), flashmind_memory::rusqlite::Error> {
            session::schema::init_session_schema(c)?;
            Ok(())
        },
    )
    .await?;
    Ok(SessionStore::new(conn))
}

fn conv_to_session(entry: &ConversationEntry, turn_index: i64) -> SessionEntry {
    let (kind, tool_calls, tool_call_id) = match &entry.kind {
        EntryKind::SystemPrompt(_) => (SessionEntryKind::SystemPrompt, None, None),
        EntryKind::Developer { tag, .. } => {
            (SessionEntryKind::Developer { tag: tag.clone() }, None, None)
        }
        EntryKind::User { .. } => (SessionEntryKind::User, None, None),
        EntryKind::Assistant { tool_calls, .. } => {
            let tc = tool_calls
                .as_ref()
                .and_then(|v| serde_json::to_value(v).ok());
            (SessionEntryKind::Assistant, tc, None)
        }
        EntryKind::Tool { call_id, .. } => (SessionEntryKind::Tool, None, Some(call_id.clone())),
    };

    SessionEntry {
        id: 0,
        chat_key: CHAT_KEY.to_string(),
        entry_kind: kind,
        content: entry.content().to_string(),
        tool_calls,
        tool_call_id,
        tool_name: None,
        metadata: match &entry.kind {
            EntryKind::Developer { metadata, .. } => metadata.clone(),
            _ => None,
        },
        turn_index,
        created_at: entry.timestamp.timestamp(),
    }
}

fn session_to_conv(entry: &SessionEntry) -> ConversationEntry {
    let ts = chrono::DateTime::from_timestamp(entry.created_at, 0).unwrap_or_else(chrono::Utc::now);

    let kind = match &entry.entry_kind {
        SessionEntryKind::SystemPrompt => EntryKind::SystemPrompt(entry.content.clone()),
        SessionEntryKind::Developer { tag } => EntryKind::Developer {
            content: entry.content.clone(),
            tag: tag.clone(),
            metadata: entry.metadata.clone(),
        },
        SessionEntryKind::User => EntryKind::User {
            content: entry.content.clone(),
            parts: None,
        },
        SessionEntryKind::Assistant => {
            let tool_calls = entry
                .tool_calls
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            EntryKind::Assistant {
                content: entry.content.clone(),
                tool_calls,
            }
        }
        SessionEntryKind::Tool => EntryKind::Tool {
            call_id: entry.tool_call_id.clone().unwrap_or_default(),
            output: entry.content.clone(),
        },
    };

    ConversationEntry {
        kind,
        timestamp: ts,
    }
}

async fn load_conversation(
    store: &SessionStore,
    system_prompt: &str,
    restore: bool,
) -> Result<Conversation> {
    let mut conversation = Conversation::new();
    conversation.set_system(system_prompt);

    if !restore {
        return Ok(conversation);
    }

    let entries = store.load(CHAT_KEY).await?;
    if entries.is_empty() {
        return Ok(conversation);
    }

    // Skip the stored system prompt — we always use the current one.
    for entry in &entries {
        if matches!(entry.entry_kind, SessionEntryKind::SystemPrompt) {
            continue;
        }
        conversation.add(session_to_conv(entry));
    }

    Ok(conversation)
}

async fn save_turn(store: &SessionStore, conversation: &Conversation) -> Result<()> {
    let entries: Vec<SessionEntry> = conversation
        .entries()
        .iter()
        .enumerate()
        .map(|(i, e)| conv_to_session(e, i as i64))
        .collect();

    store.rewrite(CHAT_KEY, &entries).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// One-shot mode
// ---------------------------------------------------------------------------

async fn run_oneshot(
    agent: &mut Agent,
    conversation: &mut Conversation,
    prompt: String,
) -> Result<()> {
    let cancel = CancellationToken::new();
    let stream = agent.start(conversation, cancel, AgentInput::user(prompt), None);
    tokio::pin!(stream);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {e}");
                std::process::exit(1);
            }
            AgentEvent::Done(_) => {
                println!();
                return Ok(());
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive mode
// ---------------------------------------------------------------------------

async fn run_interactive(
    agent: &mut Agent,
    conversation: &mut Conversation,
    store: &SessionStore,
    model_display: &str,
    restored: bool,
) -> Result<()> {
    use flashmind_tui::{Repl, ReplConfig, ReplEvent};

    let greeting = if restored {
        format!("flashmind — {model_display} — session restored — Ctrl-D to quit")
    } else {
        format!("flashmind — {model_display} — Ctrl-D to quit")
    };

    let config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: Some(greeting),
        ..Default::default()
    };

    let mut repl = Repl::new(config);
    repl.print_greeting()?;

    while let ReplEvent::UserInput(text) = repl.read_input()? {
        conversation.mark_turn_start();
        let cancel = CancellationToken::new();
        let stream = agent.start(conversation, cancel.clone(), AgentInput::user(text), None);
        repl.stream_response(cancel, Box::pin(stream)).await?;
        save_turn(store, conversation).await?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const PROVIDERS: &[(&str, Provider)] = &[
    ("Ollama (local, no API key)", Provider::Ollama),
    ("OpenRouter (many models, one key)", Provider::OpenRouter),
    ("Anthropic (Claude models)", Provider::Anthropic),
    ("OpenAI (GPT models)", Provider::OpenAi),
];

fn run_choice(
    tui: &mut flashmind_tui::Tui,
    picker: &mut ChoicePicker,
) -> io::Result<Option<ChoiceResponse>> {
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
            return Ok(match action {
                ChoicePickerAction::Select(r) => Some(r),
                ChoicePickerAction::Cancel => None,
            });
        }
        tui.erase(drawn)?;
        drawn = tui.draw_lines(&picker.lines())?;
    }
}

// ---------------------------------------------------------------------------
// Model picker — searchable table with pricing and capabilities
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

        // Search bar
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

        // Compute column widths from visible slice
        let vis_end = (self.scroll + Self::VISIBLE).min(self.filtered.len());
        let visible = &self.filtered[self.scroll..vis_end];

        // Header
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

        // Rows
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

        // Scroll indicator
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

fn run_model_picker(
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

async fn run_setup(config: &Config) -> Result<()> {
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

    // 2. Get API key if needed (before fetching models)
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

    // 4. Pick model with searchable table
    let model_idx = match run_model_picker(&mut tui, &models)? {
        Some(idx) => idx,
        None => {
            tui.println(&ratatui::text::Line::from("Setup cancelled."))?;
            return Ok(());
        }
    };
    let model_id = &models[model_idx].id;

    // 5. Write config
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
    if let Some(ref key) = api_key {
        let key_field = match provider {
            Provider::OpenRouter => "openrouter_api_key",
            Provider::Anthropic => "anthropic_api_key",
            Provider::OpenAi => "openai_api_key",
            _ => "",
        };
        if !key_field.is_empty() {
            out.push(format!("{key_field} = {key:?}"));
        }
    }
    if let Some(ref url) = config.ollama_url {
        out.push(format!("ollama_url = {url:?}"));
    }
    if let Some(ref key) = config.brave_api_key {
        out.push(format!("brave_api_key = {key:?}"));
    }
    if let Some(ref key) = config.firecrawl_api_key {
        out.push(format!("firecrawl_api_key = {key:?}"));
    }
    if let Some(ref prompt) = config.system_prompt {
        out.push(format!("system_prompt = {prompt:?}"));
    }
    if provider != Provider::OpenRouter
        && let Some(ref key) = config.openrouter_api_key
    {
        out.push(format!("openrouter_api_key = {key:?}"));
    }
    if provider != Provider::Anthropic
        && let Some(ref key) = config.anthropic_api_key
    {
        out.push(format!("anthropic_api_key = {key:?}"));
    }
    if provider != Provider::OpenAi
        && let Some(ref key) = config.openai_api_key
    {
        out.push(format!("openai_api_key = {key:?}"));
    }

    let content = out.join("\n") + "\n";
    std::fs::write(&config_path, &content)?;

    tui.println(&ratatui::text::Line::default())?;
    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Saved to {}", config_path.display()),
        flashmind_tui::styles::S_AGENT,
    )))?;
    tui.println(&ratatui::text::Line::from(format!("  Model: {model_str}")))?;
    tui.println(&ratatui::text::Line::from(
        "  Run `flashmind` to start chatting.",
    ))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let config = load_config()?;

    if let Some(Command::Setup) = cli.command {
        return run_setup(&config).await;
    }

    let model = resolve_model(&cli, &config)?;
    let provider = build_provider(&model, &config)?;
    let tools = build_tools(&config);

    let system_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(flashmind_prompts::CODING_AGENT);

    let mut agent = Agent::builder(provider)
        .tools(tools)
        .llm(AgentLlmConfig::new(model.clone()))
        .auto_compact(true)
        .build()
        .await;

    let store = open_session_store().await?;
    let restore = !cli.no_restore && cli.prompt.is_none();
    let mut conversation = load_conversation(&store, system_prompt, restore).await?;
    let restored = restore && conversation.entries().len() > 1;

    if let Some(prompt) = cli.prompt {
        run_oneshot(&mut agent, &mut conversation, prompt).await?;
    } else {
        let model_display = model.name();
        run_interactive(
            &mut agent,
            &mut conversation,
            &store,
            model_display,
            restored,
        )
        .await?;
        save_turn(&store, &conversation).await?;
    }

    Ok(())
}
