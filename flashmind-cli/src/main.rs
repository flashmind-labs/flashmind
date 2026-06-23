mod agents;
mod config;
mod interactive;
mod memory;
mod mention;
mod ops;
mod provider;
mod session;
mod setup;
mod tools;

use std::io;
use std::path::PathBuf;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use clap_complete::generate;

use flashmind_core::Conversation;
use flashmind_types::{AgentLlmConfig, ReasoningLevel};

use crate::config::{config_dir, load_config};
use crate::interactive::{SessionState, run_interactive, run_oneshot, run_resume};
use crate::ops::{run_clean, run_logs};
use crate::provider::{build_provider, fetch_pricing, resolve_model};
use crate::setup::run_setup;
use crate::tools::{build_tools, build_tools_full, run_mcp};

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to tools for reading files, editing code, and searching the web. You can run commands through skills — reusable procedures with sandboxed environments.

Use your tools to answer questions, complete tasks, and solve problems. When a task involves code:
- Read the relevant files before editing. Understand the surrounding code and conventions.
- Make minimal, targeted changes. Preserve existing style.
- Fix root causes, not symptoms. Form a hypothesis before making changes.

General principles:
- Be direct. Match the depth of your response to the complexity of the question.
- If something is ambiguous, ask before guessing.
- Don't add abstractions, error handling, or features beyond what was requested."#;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "flsh", about = "AI chat & coding agent")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Model (e.g. "ollama:llama3.2", "openrouter:anthropic/claude-sonnet-4")
    #[arg(short, long, global = true)]
    pub model: Option<String>,

    /// One-shot prompt — prints the response and exits
    #[arg(short, long)]
    prompt: Option<String>,

    /// System prompt override
    #[arg(long)]
    pub system: Option<String>,

    /// Comma-separated list of tools to enable (e.g. "file_read,grep,exec")
    #[arg(short, long, value_delimiter = ',')]
    pub tools: Option<Vec<String>>,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive setup — choose a provider and model
    #[command(alias = "s")]
    Setup,
    /// Resume a previous session
    #[command(alias = "r")]
    Resume {
        /// Show sessions from all directories
        #[arg(short, long)]
        all: bool,
    },
    /// Manage MCP servers
    #[command(alias = "m")]
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
    /// Generate shell completion scripts
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
    /// View agent logs
    Logs {
        /// Date to view: "today" (default), "yesterday", or YYYY-MM-DD
        #[arg(default_value = "today")]
        date: String,
        /// Follow log output in real-time (like tail -f)
        #[arg(short, long)]
        follow: bool,
        /// Number of lines to show (default: 50)
        #[arg(short = 'n', long)]
        lines: Option<usize>,
    },
    /// Remove expired data (logs, sessions, display logs)
    Clean {
        /// Retention period in days (default: 7)
        #[arg(long, default_value = "7")]
        days: u64,
        /// Don't actually delete, just show what would be removed
        #[arg(short, long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
pub enum McpCommand {
    /// Add an MCP server (stdio or HTTP/SSE)
    #[command(alias = "a")]
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
    #[command(alias = "rm")]
    Remove {
        /// Server name to remove
        name: String,
    },
    /// Authenticate with an MCP server
    Auth {
        /// Server name to authenticate
        name: String,
    },
    /// List configured MCP servers
    #[command(alias = "ls")]
    List,
    /// Manage tool permissions for an MCP server
    #[command(alias = "p")]
    Permissions {
        /// Server name
        name: String,
    },
    /// Import MCP servers from another CLI
    Import {
        /// Import from Claude (Desktop + Code)
        #[arg(long)]
        claude: bool,
    },
}

fn parse_env_pair(s: &str) -> Result<(String, String), String> {
    let (k, v) = s.split_once('=').ok_or("expected KEY=VALUE")?;
    Ok((k.to_string(), v.to_string()))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let log_dir = config_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::daily(&log_dir, "flsh.log");
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
        Some(Command::Resume { all }) => {
            return run_resume(&cli, &config, all).await;
        }
        Some(Command::Mcp { command }) => return run_mcp(command).await,
        Some(Command::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "flsh", &mut io::stdout());
            return Ok(());
        }
        Some(Command::Logs {
            date,
            follow,
            lines,
        }) => {
            return run_logs(date, follow, lines);
        }
        Some(Command::Clean { days, dry_run }) => {
            return run_clean(days, dry_run).await;
        }
        None => {}
    }

    let model = resolve_model(&cli, &config)?;
    let provider = build_provider(&model, &config)?;

    let reasoning = config.reasoning.unwrap_or(ReasoningLevel::Off);
    let llm_config = AgentLlmConfig::new(model.clone()).with_reasoning(reasoning);

    let full_mode = cli
        .tools
        .as_ref()
        .is_some_and(|t| t.iter().any(|a| a == "all"));
    let (mut tools, tool_sync, skill_index, skill_provider, skill_runner) = if full_mode {
        build_tools_full(&config, provider.clone(), llm_config.clone()).await
    } else {
        build_tools(&config, provider.clone(), llm_config.clone()).await
    };

    let memory_store = memory::open_memory_store(&config).await?;
    if let Some(ref ms) = memory_store {
        memory::register_memory_tools(&mut tools, ms);
    }

    if let Some(ref allowed) = cli.tools
        && !full_mode
    {
        tools.retain(|name| allowed.iter().any(|a| a == name));
    }

    let base_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(SYSTEM_PROMPT);
    let mut system_prompt = base_prompt.to_string();
    if memory_store.is_some() {
        system_prompt.push_str(&format!("\n\n{}", flashmind_prompts::MEMORY_INSTRUCTIONS));
    }
    system_prompt.push_str(&format!("\n\n{}", flashmind_prompts::MCP_INSTRUCTIONS));
    system_prompt.push_str(&format!("\n\n{}", flashmind_prompts::SKILL_INSTRUCTIONS));
    system_prompt.push_str(&skill_index.0);

    let mut agent = flashmind_core::Agent::builder(provider.clone())
        .tools(tools)
        .llm(llm_config)
        .auto_compact(true)
        .build()
        .await;

    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = provider.context_window(&model).await;

    let store = session::open_session_store().await?;

    if let Some(prompt) = cli.prompt {
        let mut conversation = Conversation::with_system(&system_prompt);
        run_oneshot(&mut agent, &mut conversation, prompt).await?;
        tool_sync.shutdown().await;
    } else {
        let session_key = session::new_session_key();
        let mut conversation = Conversation::with_system(&system_prompt);

        run_interactive(
            &mut agent,
            &mut conversation,
            &store,
            &config,
            SessionState {
                display_log_path: crate::interactive::display_log_path(&session_key),
                session_key,
                model: model.clone(),
                reasoning,
                pricing,
                context_window,
                system_prompt,
                skip_banner: false,
                skill_provider: Some(skill_provider),
                skill_runner: Some(skill_runner),
                total_cost: rust_decimal::Decimal::ZERO,
                last_usage: None,
            },
            &tool_sync,
        )
        .await?;
        tool_sync.shutdown().await;
    }

    Ok(())
}
