use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry, EntryKind};
use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_memory::session::{self, SessionEntry, SessionEntryKind, SessionStore};
use flashmind_memory::{EmbeddingProviderConfig, MemoryStore, create_embedding_provider};
use flashmind_tools::{Tool, ToolBuilder, ToolContext, ToolResult, protected::ProtectedPaths};
use flashmind_tui::widgets::{
    ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse, StatusInfo,
};
use flashmind_types::memory::{MemoryMetadata, MemoryProvider};
use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, LlmProvider, Model, ModelPricing, Provider,
    ReasoningLevel,
};

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
}

#[derive(Subcommand)]
enum Command {
    /// Interactive setup — choose a provider and model
    Setup,
    /// Resume a previous session
    Resume,
    /// Manage MCP servers
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
}

#[derive(Subcommand)]
enum McpCommand {
    /// Add an MCP server (stdio or HTTP/SSE)
    Add {
        /// Unique name for this server
        name: String,
        /// Command to spawn (stdio transport)
        #[arg(short, long)]
        command: Option<String>,
        /// Arguments for the command
        #[arg(short, long)]
        args: Vec<String>,
        /// HTTP/SSE endpoint URL
        #[arg(short, long)]
        url: Option<String>,
        /// Environment variables (KEY=VALUE)
        #[arg(short, long, value_parser = parse_env_pair)]
        env: Vec<(String, String)>,
    },
    /// Remove an MCP server
    Remove {
        /// Server name to remove
        name: String,
    },
    /// List configured MCP servers
    List,
}

fn parse_env_pair(s: &str) -> Result<(String, String), String> {
    let (k, v) = s.split_once('=').ok_or("expected KEY=VALUE")?;
    Ok((k.to_string(), v.to_string()))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
struct Config {
    model: Option<String>,
    system_prompt: Option<String>,
    reasoning: Option<ReasoningLevel>,

    openrouter_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
    ollama_url: Option<String>,

    brave_api_key: Option<String>,
    firecrawl_api_key: Option<String>,

    memory_provider: Option<String>,
    memory_model: Option<String>,
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
# reasoning = "off"  # off, low, medium, high

# openrouter_api_key = ""
# anthropic_api_key = ""
# openai_api_key = ""
# ollama_url = "http://localhost:11434"

# brave_api_key = ""
# firecrawl_api_key = ""

# memory_provider = "openrouter"  # openrouter or openai
# memory_model = "openai/text-embedding-3-small"
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

async fn fetch_pricing(provider: &Arc<dyn LlmProvider>, model: &Model) -> ModelPricing {
    let name = model.capability_name();
    let models = provider.list_models().await.unwrap_or_default();
    models
        .iter()
        .find(|m| m.id == name)
        .map(|m| m.pricing.clone())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn mcp_config_dir() -> Result<PathBuf> {
    let dir = config_dir()?.join("mcp");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

async fn build_tools(
    config: &Config,
) -> (
    flashmind_types::ToolRegistry,
    flashmind_tools::tool_sync::ToolSync,
) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let protected = Arc::new(ProtectedPaths::new(&cwd));

    let mcp_config_dir = config_dir()
        .map(|d| d.join("mcp"))
        .unwrap_or_else(|_| PathBuf::from(".flashmind/mcp"));
    let _ = std::fs::create_dir_all(&mcp_config_dir);
    let mcp_provider = flashmind_tools::mcp::McpDiskConfig::new(mcp_config_dir);

    ToolBuilder::new()
        .file_ops(None, &protected)
        .bash(vec![], &protected, vec![], None)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .time()
        .mcp(mcp_provider, None)
        .build_with_sync()
        .await
}

async fn run_mcp(cmd: McpCommand) -> Result<()> {
    use flashmind_tools::mcp::{McpConfigProvider, McpDiskConfig, McpRegistry, McpServerConfig};

    let config_provider = McpDiskConfig::new(mcp_config_dir()?);
    let registry = McpRegistry::new(config_provider.clone(), None);

    match cmd {
        McpCommand::Add {
            name,
            command,
            args,
            url,
            env,
        } => {
            if command.is_none() && url.is_none() {
                bail!("must provide either --command or --url");
            }
            let server_config = McpServerConfig {
                name: name.clone(),
                command,
                args,
                url,
                env: env.into_iter().collect(),
                ..Default::default()
            };
            config_provider.save_config(&server_config).await?;
            match registry.connect(server_config).await {
                Ok(tools) => {
                    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                    println!(
                        "Connected to '{name}' — {} tools: {}",
                        tools.len(),
                        names.join(", ")
                    );
                }
                Err(e) => {
                    println!("Saved '{name}' but failed to connect: {e}");
                    println!("It will retry on next launch.");
                }
            }
        }
        McpCommand::Remove { name } => {
            config_provider.delete_config(&name).await?;
            println!("Removed '{name}'");
        }
        McpCommand::List => {
            let configs = config_provider.list_configs().await?;
            if configs.is_empty() {
                println!("No MCP servers configured.");
            } else {
                for cfg in &configs {
                    let transport = if cfg.command.is_some() {
                        format!("stdio: {}", cfg.command.as_deref().unwrap_or("?"))
                    } else if let Some(url) = &cfg.url {
                        format!("http: {url}")
                    } else {
                        "unknown".into()
                    };
                    let tool_count = cfg.cached_tools.len();
                    println!(
                        "  {:<20} {transport}  ({tool_count} cached tools)",
                        cfg.name
                    );
                }
            }
        }
    }

    registry.shutdown_all().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory tools
// ---------------------------------------------------------------------------

struct MemoryStoreTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryStoreTool {
    fn name(&self) -> &str {
        "memory_store"
    }

    fn description(&self) -> &str {
        "Store a fact in long-term memory for future conversations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or information to remember."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags for categorization (e.g. \"preference\", \"project\")."
                },
                "context": {
                    "type": "string",
                    "description": "Brief context about why this is being stored."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let content: String = ctx
            .args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if content.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "content is required"));
        }

        let tags: Vec<String> = ctx
            .args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let context = ctx
            .args
            .get("context")
            .and_then(|v| v.as_str())
            .map(String::from);

        let meta = MemoryMetadata {
            context,
            tags,
            expires_at: None,
        };

        let id = MemoryProvider::store(self.store.as_ref(), &content, meta).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Stored memory {id}"),
        ))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("…");
        let preview: String = content.chars().take(60).collect();
        format!("remember: {preview}")
    }
}

struct MemoryRecallTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }

    fn description(&self) -> &str {
        "Search long-term memory for relevant facts from previous conversations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let query = ctx.args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "query is required"));
        }

        let limit = ctx.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let results = MemoryProvider::search(self.store.as_ref(), query, limit).await?;

        if results.is_empty() {
            return Ok(ToolResult::success(ctx.tool_call_id, "No memories found."));
        }

        let mut out = String::new();
        for entry in &results {
            out.push_str(&format!(
                "[{}] (score: {:.2}) {}\n",
                entry.id, entry.score, entry.content
            ));
            if !entry.metadata.tags.is_empty() {
                out.push_str(&format!("  tags: {}\n", entry.metadata.tags.join(", ")));
            }
        }
        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("…");
        format!("recall: {query}")
    }
}

struct MemoryForgetTool {
    store: Arc<MemoryStore>,
}

#[async_trait]
impl Tool for MemoryForgetTool {
    fn name(&self) -> &str {
        "memory_forget"
    }

    fn description(&self) -> &str {
        "Remove a specific memory by ID."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The memory ID to forget (from memory_recall results)."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let id = ctx.args.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if id.is_empty() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "id is required"));
        }

        MemoryProvider::forget(self.store.as_ref(), id).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Forgot memory {id}"),
        ))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("…");
        format!("forget: {id}")
    }
}

fn build_embedding_config(config: &Config) -> Option<EmbeddingProviderConfig> {
    let provider = config.memory_provider.as_deref()?;
    match provider {
        "openrouter" => {
            let model = config
                .memory_model
                .clone()
                .unwrap_or_else(|| "openai/text-embedding-3-small".into());
            Some(EmbeddingProviderConfig::OpenRouter {
                api_key: config.openrouter_api_key.clone(),
                model,
            })
        }
        "openai" => {
            let model = config
                .memory_model
                .clone()
                .unwrap_or_else(|| "text-embedding-3-small".into());
            Some(EmbeddingProviderConfig::OpenAI {
                api_key: config.openai_api_key.clone(),
                model,
                base_url: config.openai_base_url.clone(),
            })
        }
        _ => None,
    }
}

async fn open_memory_store(config: &Config) -> Result<Option<Arc<MemoryStore>>> {
    let Some(embed_config) = build_embedding_config(config) else {
        return Ok(None);
    };

    let embedder = create_embedding_provider(&embed_config, None)
        .context("failed to create embedding provider for memory")?;

    let db_path = config_dir()?.join("memory.db");
    let store = MemoryStore::connect(&db_path, embedder)
        .await
        .context("failed to open memory store")?;

    Ok(Some(Arc::new(store)))
}

fn register_memory_tools(registry: &mut flashmind_types::ToolRegistry, store: &Arc<MemoryStore>) {
    registry.register(Arc::new(MemoryStoreTool {
        store: Arc::clone(store),
    }));
    registry.register(Arc::new(MemoryRecallTool {
        store: Arc::clone(store),
    }));
    registry.register(Arc::new(MemoryForgetTool {
        store: Arc::clone(store),
    }));
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
// Resume mode
// ---------------------------------------------------------------------------

async fn run_resume(cli: &Cli, config: &Config) -> Result<()> {
    let store = open_session_store().await?;
    let sessions = store.list_sessions().await?;

    if sessions.is_empty() {
        println!("No sessions to resume.");
        return Ok(());
    }

    let mut tui = flashmind_tui::Tui::new();

    let options: Vec<ChoiceOption> = sessions
        .iter()
        .map(|s| {
            let age = format_session_age(s.last_updated);
            ChoiceOption {
                label: format!("{} ({} entries, {})", s.chat_key, s.entry_count, age),
                accepts_input: false,
            }
        })
        .collect();

    let mut picker = ChoicePicker::new("Select a session to resume:".into(), options);
    let Some(resp) = run_choice(&mut tui, &mut picker)? else {
        return Ok(());
    };

    let chat_key = &sessions[resp.selected].chat_key;

    let model = resolve_model(cli, config)?;
    let provider = build_provider(&model, config)?;
    let (mut tools, tool_sync) = build_tools(config).await;

    let memory_store = open_memory_store(config).await?;
    if let Some(ref ms) = memory_store {
        register_memory_tools(&mut tools, ms);
    }

    let base_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(flashmind_prompts::CODING_AGENT);
    let system_prompt = if memory_store.is_some() {
        format!(
            "{base_prompt}\n\n{}",
            flashmind_prompts::MEMORY_INSTRUCTIONS
        )
    } else {
        base_prompt.to_string()
    };

    let reasoning = config.reasoning.unwrap_or(ReasoningLevel::Off);
    let llm_config = AgentLlmConfig::new(model.clone()).with_reasoning(reasoning);

    let mut agent = Agent::builder(provider.clone())
        .tools(tools)
        .llm(llm_config)
        .auto_compact(true)
        .build()
        .await;

    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = provider.context_window(&model).await;

    let mut conversation = Conversation::new();
    conversation.set_system(&system_prompt);
    let entries = store.load(chat_key).await?;
    for entry in &entries {
        if matches!(entry.entry_kind, SessionEntryKind::SystemPrompt) {
            continue;
        }
        conversation.add(session_to_conv(entry));
    }

    let model_display = model.name();
    print_banner(&mut tui, model_display, reasoning)?;

    {
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        tui.println(&Line::from(Span::styled(
            format!("  session restored: {chat_key}"),
            Style::default().fg(Color::Green),
        )))?;
        tui.println(&Line::default())?;
    }

    run_interactive(
        &mut agent,
        &mut conversation,
        &store,
        config,
        SessionState {
            model_display: model_display.to_string(),
            reasoning,
            pricing,
            context_window,
        },
        &tool_sync,
    )
    .await?;
    save_turn(&store, &conversation).await?;
    tool_sync.shutdown().await;

    Ok(())
}

fn format_session_age(unix_ts: i64) -> String {
    let now = chrono::Utc::now().timestamp();
    let secs = (now - unix_ts).max(0);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
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

fn configured_providers(config: &Config) -> Vec<(&'static str, Provider)> {
    let mut providers = Vec::new();
    if config.ollama_url.is_some() || cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        providers.push(("Ollama", Provider::Ollama));
    }
    if config.openrouter_api_key.is_some() {
        providers.push(("OpenRouter", Provider::OpenRouter));
    }
    if config.anthropic_api_key.is_some() {
        providers.push(("Anthropic", Provider::Anthropic));
    }
    if config.openai_api_key.is_some() {
        providers.push(("OpenAI", Provider::OpenAi));
    }
    providers
}

async fn handle_model_command(
    args: &str,
    agent: &mut Agent,
    config: &Config,
    tui: &mut flashmind_tui::Tui,
) -> Result<Option<(String, ModelPricing, Option<u32>)>> {
    let input = args.trim();

    if !input.is_empty() {
        let model: Model = input
            .parse()
            .with_context(|| format!("invalid model format: {input}"))?;
        let provider = build_provider(&model, config)?;
        let pricing = fetch_pricing(&provider, &model).await;
        let context_window = provider.context_window(&model).await;
        let display = model.name().to_string();

        agent.set_provider(provider);
        agent.llm_mut().model = model;
        agent.refresh_features().await;

        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("  Switched to {display}"),
            flashmind_tui::styles::S_AGENT,
        )))?;
        return Ok(Some((display, pricing, context_window)));
    }

    // No args — show interactive picker
    let providers = configured_providers(config);
    if providers.is_empty() {
        tui.println(&ratatui::text::Line::from(
            "  No providers configured. Run `flashmind setup` first.",
        ))?;
        return Ok(None);
    }

    // Pick provider (skip if only one)
    let chosen_provider = if providers.len() == 1 {
        providers[0].1
    } else {
        let current = agent.llm().model.provider;
        let options: Vec<ChoiceOption> = providers
            .iter()
            .map(|(label, p)| {
                let suffix = if *p == current { " ◀" } else { "" };
                ChoiceOption {
                    label: format!("{label}{suffix}"),
                    accepts_input: false,
                }
            })
            .collect();
        let mut picker = ChoicePicker::new("Provider".into(), options);
        match run_choice(tui, &mut picker)? {
            Some(resp) => providers[resp.selected].1,
            None => return Ok(None),
        }
    };

    // Build temporary provider and fetch models
    let tmp_provider = build_provider(
        &Model {
            provider: chosen_provider,
            model: flashmind_types::AliasedModel {
                name: String::new(),
                real_name: None,
            },
        },
        config,
    )?;

    tui.println(&ratatui::text::Line::from(format!(
        "  Fetching models from {}...",
        tmp_provider.name()
    )))?;
    let models = tmp_provider
        .list_models()
        .await
        .context("failed to list models")?;
    if models.is_empty() {
        tui.println(&ratatui::text::Line::from("  No models found."))?;
        return Ok(None);
    }

    // Show model picker
    let model_idx = match run_model_picker(tui, &models)? {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let chosen = &models[model_idx];
    let prefix = match chosen_provider {
        Provider::Ollama => "ollama",
        Provider::OpenRouter => "openrouter",
        Provider::Anthropic => "anthropic",
        Provider::OpenAi => "openai",
        _ => "ollama",
    };
    let model: Model = format!("{prefix}:{}", chosen.id).parse()?;
    let pricing = fetch_pricing(&tmp_provider, &model).await;
    let context_window = tmp_provider.context_window(&model).await;
    let display = model.name().to_string();

    agent.set_provider(tmp_provider);
    agent.llm_mut().model = model;
    agent.refresh_features().await;

    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Switched to {display}"),
        flashmind_tui::styles::S_AGENT,
    )))?;
    Ok(Some((display, pricing, context_window)))
}

struct SessionState {
    model_display: String,
    reasoning: ReasoningLevel,
    pricing: ModelPricing,
    context_window: Option<u32>,
}

async fn run_interactive(
    agent: &mut Agent,
    conversation: &mut Conversation,
    store: &SessionStore,
    config: &Config,
    state: SessionState,
    tool_sync: &flashmind_tools::tool_sync::ToolSync,
) -> Result<()> {
    use flashmind_tui::{Repl, ReplConfig, ReplEvent};

    let mut tui = flashmind_tui::Tui::new();
    print_banner(&mut tui, &state.model_display, state.reasoning)?;

    let repl_config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: None,
        ..Default::default()
    };

    let mut repl = Repl::new(repl_config);
    let mut current_model = state.model_display;
    let mut current_reasoning = state.reasoning;
    let mut current_pricing = state.pricing;
    let mut current_context_window = state.context_window;

    repl.set_status(StatusInfo {
        model: current_model.clone(),
        thinking: Some(current_reasoning),
        ..Default::default()
    });

    let mut total_cost = rust_decimal::Decimal::ZERO;

    while let ReplEvent::UserInput(text) = repl.read_input()? {
        // Slash command dispatch
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
            match cmd {
                "model" => {
                    if let Some((display, new_pricing, new_cw)) =
                        handle_model_command(args, agent, config, &mut tui).await?
                    {
                        current_model = display;
                        current_pricing = new_pricing;
                        current_context_window = new_cw;
                        repl.set_status(StatusInfo {
                            model: current_model.clone(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window.map(|cw| (0, cw)),
                        });
                    }
                    continue;
                }
                "thinking" => {
                    let input = args.trim();
                    let level = if input.is_empty() {
                        let options = vec![
                            ChoiceOption { label: "Off".into(), accepts_input: false },
                            ChoiceOption { label: "Low".into(), accepts_input: false },
                            ChoiceOption { label: "Medium".into(), accepts_input: false },
                            ChoiceOption { label: "High".into(), accepts_input: false },
                        ];
                        let mut picker = ChoicePicker::new(
                            format!("Thinking (current: {current_reasoning})"),
                            options,
                        );
                        run_choice(&mut tui, &mut picker)?.map(|resp| match resp.selected {
                            1 => ReasoningLevel::Low,
                            2 => ReasoningLevel::Medium,
                            3 => ReasoningLevel::High,
                            _ => ReasoningLevel::Off,
                        })
                    } else {
                        match input {
                            "off" => Some(ReasoningLevel::Off),
                            "low" => Some(ReasoningLevel::Low),
                            "medium" | "on" => Some(ReasoningLevel::Medium),
                            "high" => Some(ReasoningLevel::High),
                            _ => {
                                tui.println(&ratatui::text::Line::from(
                                    "  Usage: /thinking [off|low|medium|high]",
                                ))?;
                                None
                            }
                        }
                    };
                    if let Some(level) = level {
                        agent.llm_mut().reasoning = level;
                        current_reasoning = level;
                        repl.set_status(StatusInfo {
                            model: current_model.clone(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window.map(|cw| (0, cw)),
                        });
                        tui.println(&ratatui::text::Line::from(
                            ratatui::text::Span::styled(
                                format!("  Thinking: {level}"),
                                flashmind_tui::styles::S_AGENT,
                            ),
                        ))?;
                    }
                    continue;
                }
                "help" => {
                    tui.println(&ratatui::text::Line::default())?;
                    tui.println(&ratatui::text::Line::from(
                        "  /model [provider:name]     Switch model (interactive picker if no args)",
                    ))?;
                    tui.println(&ratatui::text::Line::from(
                        "  /thinking [off|low|med|high]  Show or set reasoning level",
                    ))?;
                    tui.println(&ratatui::text::Line::from(
                        "  /help                      Show this help",
                    ))?;
                    tui.println(&ratatui::text::Line::default())?;
                    continue;
                }
                _ => {} // unknown slash commands fall through to the agent
            }
        }

        conversation.mark_turn_start();
        let cancel = CancellationToken::new();
        let stream = agent.start(conversation, cancel.clone(), AgentInput::user(text), None);
        repl.stream_response(cancel, Box::pin(stream)).await?;
        tool_sync.sync(agent.tools_mut());

        if let Some(usage) = repl.last_usage() {
            if let Some(turn_cost) = usage.cost(&current_pricing) {
                total_cost += turn_cost;
            }
            repl.set_status(StatusInfo {
                model: current_model.clone(),
                thinking: Some(current_reasoning),
                cost: Some(total_cost),
                context: current_context_window.map(|cw| (usage.prompt_tokens, cw)),
            });
        }

        save_turn(store, conversation).await?;
    }

    Ok(())
}

fn print_banner(
    tui: &mut flashmind_tui::Tui,
    model_display: &str,
    reasoning: ReasoningLevel,
) -> io::Result<()> {
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    let dim = flashmind_tui::styles::S_DIM;
    let bold = Style::default().add_modifier(Modifier::BOLD);

    tui.println(&Line::default())?;
    tui.println(&Line::from(vec![
        Span::styled(
            "  flashmind",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" — ", dim),
        Span::styled(model_display, bold),
    ]))?;

    let mut info_spans = Vec::new();
    info_spans.push(Span::styled("  ", dim));

    if reasoning.is_on() {
        info_spans.push(Span::styled(
            format!("thinking: {reasoning}"),
            Style::default().fg(Color::Yellow),
        ));
        info_spans.push(Span::styled("  ", dim));
    }

    info_spans.push(Span::styled("Esc to cancel, Ctrl-D to quit", dim));
    tui.println(&Line::from(info_spans))?;
    tui.println(&Line::default())?;

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
    let chosen_model = &models[model_idx];
    let model_id = &chosen_model.id;

    // 5. Pick reasoning level (only if model supports it)
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

    // 6. Memory — embedding provider for long-term memory
    //    Default to OpenRouter when the user already has an OpenRouter key.
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
        // Put the provider with an existing key first so it's pre-selected.
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

    // 7. Web search — Brave or Firecrawl
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

    // Collect final API keys — merge new keys from setup with existing config.
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
        "  Run `flashmind` to start chatting.",
    ))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let log_dir = config_dir().unwrap_or_else(|_| PathBuf::from(".")).join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::daily(&log_dir, "flashmind.log");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .with_ansi(false)
        .with_writer(file_appender)
        .init();

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = ratatui::crossterm::terminal::disable_raw_mode();
        let _ = ratatui::crossterm::execute!(io::stdout(), ratatui::crossterm::cursor::Show);
        original_hook(info);
    }));

    let cli = Cli::parse();
    let config = load_config()?;

    match cli.command {
        Some(Command::Setup) => return run_setup(&config).await,
        Some(Command::Resume) => {
            return run_resume(&cli, &config).await;
        }
        Some(Command::Mcp { command }) => return run_mcp(command).await,
        None => {}
    }

    let model = resolve_model(&cli, &config)?;
    let provider = build_provider(&model, &config)?;
    let (mut tools, tool_sync) = build_tools(&config).await;

    let memory_store = open_memory_store(&config).await?;
    if let Some(ref ms) = memory_store {
        register_memory_tools(&mut tools, ms);
    }

    let base_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(flashmind_prompts::CODING_AGENT);
    let system_prompt = if memory_store.is_some() {
        format!(
            "{base_prompt}\n\n{}",
            flashmind_prompts::MEMORY_INSTRUCTIONS
        )
    } else {
        base_prompt.to_string()
    };

    let reasoning = config.reasoning.unwrap_or(ReasoningLevel::Off);
    let llm_config = AgentLlmConfig::new(model.clone()).with_reasoning(reasoning);

    let mut agent = Agent::builder(provider.clone())
        .tools(tools)
        .llm(llm_config)
        .auto_compact(true)
        .build()
        .await;

    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = provider.context_window(&model).await;

    let store = open_session_store().await?;
    let mut conversation = load_conversation(&store, &system_prompt, false).await?;

    if let Some(prompt) = cli.prompt {
        run_oneshot(&mut agent, &mut conversation, prompt).await?;
        tool_sync.shutdown().await;
    } else {
        run_interactive(
            &mut agent,
            &mut conversation,
            &store,
            &config,
            SessionState {
                model_display: model.name().to_string(),
                reasoning,
                pricing,
                context_window,
            },
            &tool_sync,
        )
        .await?;
        save_turn(&store, &conversation).await?;
        tool_sync.shutdown().await;
    }

    Ok(())
}
