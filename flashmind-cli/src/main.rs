//! Flashmind CLI — interactive AI agent.

mod commands;
mod config;
mod display;
mod enrichment;
mod repl;
mod session;
mod tui;
mod wizard;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use crate::config::Config;
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
}

#[derive(Subcommand)]
enum Commands {
    /// Create default config at ~/.flashmind/config.toml.
    Init,
    /// Manage saved sessions.
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
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
                    // Also remove display log
                    let display_path = Config::session_display_path(&key);
                    let _ = std::fs::remove_file(&display_path);
                    println!("Deleted session {key}");
                }
            }
        }
        None => {
            repl::run(cli.model, cli.no_restore).await?;
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
