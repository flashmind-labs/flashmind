//! Flashmind CLI — interactive AI agent.

mod attachments;
mod commands;
mod config;
mod display;
mod mcp_auth;
mod memory;
mod oneshot;
mod repl;
mod session;
mod tui;
mod wizard;

use std::io::{self, IsTerminal};

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use flashmind_tools::mcp::{McpConfigProvider, McpDiskConfig, McpRegistry, McpServerConfig};
use flashmind_types::model::Model;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "flashmind-cli", about = "Interactive AI agent CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Override the model (e.g. "ollama:llama3.2", "openrouter:anthropic/claude-sonnet-4").
    #[arg(short, long, global = true)]
    model: Option<Model>,

    /// Start a new session instead of resuming.
    #[arg(long = "no-restore", global = true)]
    no_restore: bool,

    /// Run a single prompt non-interactively and exit.
    /// Combine with stdin piping: `echo "context" | flashmind-cli -p "summarize this"`
    #[arg(short, long)]
    prompt: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Create default config at ~/.flashmind/config.toml.
    Init,
    /// Interactive config setup wizard.
    Setup,
    /// Manage saved sessions.
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Edit SOUL.md (system prompt) in $EDITOR.
    Soul,
    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for.
        shell: Shell,
    },
    /// Manage MCP servers.
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Show recent logs.
    Logs {
        /// Number of lines to show.
        #[arg(short, long, default_value = "50")]
        lines: usize,
    },
}

#[derive(Subcommand)]
enum SessionAction {
    /// List all saved sessions.
    List,
    /// Delete a session by key.
    Delete {
        /// Session key to delete.
        key: String,
    },
    /// Export a session to a JSON file.
    Export {
        /// Session key to export.
        key: String,
        /// Output file path (defaults to <key>.json).
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum McpAction {
    /// List configured MCP servers.
    List,
    /// Add an MCP server (URL for HTTP, or command for stdio).
    ///
    /// Examples:
    ///   flashmind-cli mcp add github https://mcp.github.com/sse
    ///   flashmind-cli mcp add sqlite -- uvx mcp-server-sqlite --db-path ./data.db
    ///   flashmind-cli mcp add gmail --reauth "npx @gongrzhe/server-gmail-autoauth-mcp auth" -- npx @gongrzhe/server-gmail-autoauth-mcp
    #[command(trailing_var_arg = true)]
    Add {
        /// Server name.
        name: String,
        /// Shell command to run when the server needs reauthentication.
        #[arg(long)]
        reauth: Option<String>,
        /// URL or command followed by its arguments.
        /// Use `--` before commands with flags: `mcp add myserver -- cmd --flag`.
        #[arg(required = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Remove an MCP server.
    Remove {
        /// Server name.
        name: String,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TLS provider for HTTP tools
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cli = Cli::parse();

    // Logging
    let log_dir = Config::log_dir();
    std::fs::create_dir_all(&log_dir)?;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("flashmind-cli.log"))?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(file)
        .with_ansi(false)
        .init();

    match cli.command {
        Some(Commands::Init) => {
            let path = Config::init()?;
            println!("Config created at {}", path.display());
        }
        Some(Commands::Setup) => {
            Config::init()?;
            wizard::run(cli.model).await?;
        }
        Some(Commands::Sessions { action }) => {
            let sessions = session::Sessions::connect(&Config::db_path()).await?;
            match action {
                SessionAction::List => {
                    let list = sessions.list_local().await?;
                    if list.is_empty() {
                        println!("No saved sessions.");
                    } else {
                        for s in &list {
                            let age = format_age(s.updated_at);
                            let label = s.title.as_deref().unwrap_or(&s.prompt);
                            println!("{} [{age}] {} — {}", s.key, s.model, label);
                        }
                    }
                }
                SessionAction::Delete { key } => {
                    sessions.delete(&key).await?;
                    let display_path = Config::session_display_path(&key);
                    let _ = std::fs::remove_file(&display_path);
                    println!("Deleted session {key}");
                }
                SessionAction::Export { key, output } => {
                    let conv = sessions.load(&key).await?;
                    match conv {
                        Some(conv) => {
                            let messages = conv.to_messages();
                            let json = serde_json::to_string_pretty(&messages)?;
                            let out_path = output.unwrap_or_else(|| format!("{key}.json"));
                            std::fs::write(&out_path, &json)?;
                            println!("Exported {} messages to {}", messages.len(), out_path);
                        }
                        None => {
                            println!("Session not found: {key}");
                        }
                    }
                }
            }
        }
        Some(Commands::Soul) => {
            Config::init()?;
            let path = Config::soul_path();
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
            let status = std::process::Command::new(&editor).arg(&path).status()?;
            if !status.success() {
                eprintln!("Editor exited with {status}");
            }
        }
        Some(Commands::Completions { shell }) => {
            let mut cmd = Cli::command();
            generate(shell, &mut cmd, "flashmind-cli", &mut io::stdout());
        }
        Some(Commands::Logs { lines }) => {
            let log_path = Config::log_dir().join("flashmind-cli.log");
            if log_path.exists() {
                let content = std::fs::read_to_string(&log_path)?;
                let all: Vec<&str> = content.lines().collect();
                let start = all.len().saturating_sub(lines);
                for line in &all[start..] {
                    println!("{line}");
                }
            } else {
                println!("No log file found.");
            }
        }
        Some(Commands::Mcp { action }) => {
            Config::init()?;
            let mcp = McpDiskConfig::new(Config::mcp_dir());
            match action {
                McpAction::List => {
                    let configs = mcp.list_configs().await?;
                    if configs.is_empty() {
                        println!("No MCP servers configured.");
                    } else {
                        for cfg in &configs {
                            let transport = if cfg.url.is_some() { "HTTP" } else { "stdio" };
                            println!("  {} ({})", cfg.name, transport);
                        }
                    }
                }
                McpAction::Add { name, reauth, args } => {
                    let endpoint = &args[0];
                    let is_url =
                        endpoint.starts_with("http://") || endpoint.starts_with("https://");
                    let (url, command, cmd_args) = if is_url {
                        (Some(endpoint.clone()), None, vec![])
                    } else {
                        (None, Some(endpoint.clone()), args[1..].to_vec())
                    };
                    let config = McpServerConfig {
                        name: name.clone(),
                        command,
                        args: cmd_args,
                        url,
                        env: Default::default(),
                        client_id: None,
                        client_secret: None,
                        scopes: vec![],
                        credentials: None,
                        reauth: reauth.clone(),
                        cached_tools: vec![],
                    };

                    let registry = McpRegistry::new(mcp.clone(), None);
                    registry.add(config).await?;

                    // Run reauth command if configured.
                    if let Some(ref cmd) = reauth {
                        print!("Running reauth command... ");
                        let out = tokio::process::Command::new("sh")
                            .args(["-c", cmd])
                            .output()
                            .await?;
                        if !out.status.success() {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            eprintln!("reauth failed ({}): {}", out.status, stderr.trim());
                        } else {
                            println!("ok");
                        }
                    }

                    // Connect (runs OAuth via browser if needed).
                    print!("Connecting to {name}... ");
                    match registry.reconnect(&name).await {
                        Ok(tools) => {
                            println!("{} tools available", tools.len());
                        }
                        Err(e)
                            if e.downcast_ref::<flashmind_tools::mcp::McpAuthRequired>()
                                .is_some() =>
                        {
                            println!(
                                "Opening browser for authentication... (waiting up to 5 minutes)"
                            );
                            match mcp_auth::browser_oauth(&registry, &name).await {
                                Ok(()) => {
                                    let tools = registry.current_mcp_tools();
                                    let count = tools.get(&name).map(|t| t.len()).unwrap_or(0);
                                    println!("Authenticated. {count} tools available");
                                }
                                Err(e) => eprintln!("{e}"),
                            }
                        }
                        Err(e) => {
                            eprintln!("failed: {e}");
                        }
                    }
                }
                McpAction::Remove { name } => {
                    mcp.delete_config(&name).await?;
                    println!("Removed MCP server: {name}");
                }
            }
        }
        None => {
            // Non-interactive mode: -p flag or stdin pipe
            let stdin_input = if !IsTerminal::is_terminal(&io::stdin()) {
                let mut buf = String::new();
                io::Read::read_to_string(&mut io::stdin(), &mut buf)?;
                if buf.trim().is_empty() {
                    None
                } else {
                    Some(buf)
                }
            } else {
                None
            };

            if cli.prompt.is_some() || stdin_input.is_some() {
                let prompt = match (cli.prompt, stdin_input) {
                    (Some(p), Some(stdin)) => format!("{stdin}\n\n{p}"),
                    (Some(p), None) => p,
                    (None, Some(stdin)) => stdin,
                    (None, None) => unreachable!(),
                };
                oneshot::run(cli.model, &prompt).await?;
            } else {
                repl::run(cli.model, cli.no_restore).await?;
            }
        }
    }

    Ok(())
}

fn format_age(ts: i64) -> String {
    let now = chrono::Utc::now().timestamp();
    let secs = now - ts;
    if secs < 60 {
        "just now".into()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}
