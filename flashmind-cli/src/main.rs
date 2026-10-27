mod agents;
mod autocompact;
mod config;
mod file_tools;
mod instructions;
mod interactive;
mod mention;
mod model_completion;
mod ops;
mod paths;
mod pi_import;
mod plugins;
mod prompt;
mod provider;
mod session;
mod setup;
mod subagents;
mod terminal_title;
mod tools;
mod undo;
mod usage;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use clap_complete::generate;

use flashmind_core::{Agent, AgentManager, Conversation};
use flashmind_skills::DiskSkillProvider;
use flashmind_tools::ToolSync;
use flashmind_types::{
    AgentLlmConfig, Model, ModelPricing, ReasoningLevel, SessionId, SessionUsage,
};
use tokio::sync::RwLock;

use crate::config::{Config, load_config};
use crate::instructions::{global_level, is_claude_model, project_instructions};
use crate::interactive::{SessionState, run_interactive, run_oneshot, run_resume};
use crate::ops::{run_clean, run_logs, run_update};
use crate::paths::skill_flag_dirs;
use crate::provider::{build_provider, fetch_pricing, resolve_model};
use crate::setup::run_setup;
use crate::tools::{AlwaysOnSkills, ToolSetup, always_on_skills, build_tools, skill_instructions};

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

    /// With --prompt, print every agent event as one JSON object per line
    #[arg(long, requires = "prompt")]
    json: bool,

    /// System prompt override
    #[arg(long)]
    pub system: Option<String>,

    /// Comma-separated tools to enable; use "all" for the full set
    #[arg(short, long, value_delimiter = ',')]
    pub tools: Option<Vec<String>>,

    /// Run without shell command approval prompts
    #[arg(long, global = true)]
    pub dangerously_skip_permissions: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Update flsh from its Git repository using cargo install
    Update,
    /// Import sessions from another agent
    Import {
        #[command(subcommand)]
        source: ImportSource,
    },
    /// Interactive setup — choose a provider and model
    #[command(alias = "s")]
    Setup,
    /// Resume a previous session
    #[command(alias = "r")]
    Resume {
        /// Session ID to resume directly, skipping the picker
        session_id: Option<SessionId>,
        /// Show sessions from all directories
        #[arg(short, long)]
        all: bool,
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
enum ImportSource {
    /// Import pi JSONL sessions (defaults to pi's sessions directory)
    Pi { path: Option<PathBuf> },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let log_dir = crate::paths::state_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("flashmind"))
        .join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::daily(&log_dir, "flsh.log");
    let filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into());

    // The file always, and stderr as well when asked for. Under a supervisor or
    // in a container the file is somewhere nobody looks, and the whole point of
    // RUST_LOG_STDERR is that `docker logs` shows what the run did.
    //
    // Off by default because stderr is also where the resume hint and the token
    // count go, and an interactive session should not have log lines in the way.
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;
    let to_file = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_ansi(false)
        .with_writer(file_appender);
    let stderr_requested = std::env::var("RUST_LOG_STDERR")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false);
    if stderr_requested {
        let to_stderr = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_ansi(false)
            .with_writer(std::io::stderr);
        tracing_subscriber::registry()
            .with(filter)
            .with(to_file)
            .with(to_stderr)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(to_file)
            .init();
    }

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = ratatui::crossterm::terminal::disable_raw_mode();
        let _ = ratatui::crossterm::execute!(
            io::stdout(),
            ratatui::crossterm::event::DisableMouseCapture,
            ratatui::crossterm::event::DisableBracketedPaste,
            ratatui::crossterm::terminal::LeaveAlternateScreen
        );
        let _ = ratatui::crossterm::execute!(io::stdout(), ratatui::crossterm::cursor::Show);
        original_hook(info);
    }));

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Update) => return run_update().await,
        Some(Command::Import {
            source: ImportSource::Pi { path },
        }) => {
            return pi_import::run(path).await;
        }
        Some(Command::Setup) => return run_setup(&load_config()?).await,
        Some(Command::Resume {
            ref session_id,
            all,
        }) => {
            return run_resume(&cli, &load_config()?, all, session_id.as_ref()).await;
        }
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

    let config = load_config()?;
    let model = resolve_model(&cli, &config)?;
    let session_key = session::new_session_key();
    let Startup {
        mut agent,
        subagents,
        tool_sync,
        skill_provider,
        system_prompt,
        reasoning,
        pricing,
        context_window,
        always_skills,
    } = start_session(&cli, &config, &model, &session_key).await?;

    let store = session::open_session_store().await?;

    let mut conversation = Conversation::with_system(&system_prompt);
    if let Some(prompt) = cli.prompt {
        let usage = run_oneshot(&mut agent, &mut conversation, prompt, cli.json, &pricing).await?;
        session::save_turn(&store, &session_key, &conversation).await?;
        store
            .save_meta(
                &session_key,
                None,
                Some(&model.to_string()),
                agent.working_dir().and_then(Path::to_str),
            )
            .await?;
        store
            .update_session_usage(&session_key, &usage, None)
            .await?;
        eprintln!("Resume: flsh resume {session_key}");
    } else {
        run_interactive(
            &mut agent,
            &mut conversation,
            &store,
            &config,
            SessionState {
                subagents,
                display_log_path: Some(crate::interactive::display_log_path(&session_key)?),
                session_key,
                model,
                reasoning,
                pricing,
                context_window,
                system_prompt,
                skip_banner: false,
                skill_provider: Some(skill_provider),
                usage: SessionUsage::default(),
                last_usage: None,
                always_skills,
            },
            &tool_sync,
        )
        .await?;
    }

    Ok(())
}

/// Agent, tools and model details shared by new and resumed sessions.
pub(crate) struct Startup {
    pub subagents: Arc<AgentManager>,
    pub agent: Agent,
    pub tool_sync: ToolSync,
    pub skill_provider: Arc<RwLock<DiskSkillProvider>>,
    pub system_prompt: String,
    pub reasoning: ReasoningLevel,
    pub pricing: ModelPricing,
    pub context_window: Option<u32>,
    pub always_skills: AlwaysOnSkills,
}

/// Build the provider, tools, system prompt and agent for `model`, tagging
/// requests with `session_key`.
pub(crate) async fn start_session(
    cli: &Cli,
    config: &Config,
    model: &Model,
    session_key: &str,
) -> Result<Startup> {
    let provider = build_provider(&model.provider, config)?;

    let reasoning = config.reasoning.unwrap_or(ReasoningLevel::Off);
    let llm_config = AgentLlmConfig::new(model.clone())
        .with_reasoning(reasoning)
        .with_session_id(session_key.parse::<SessionId>()?);

    let ToolSetup {
        subagents,
        registry: tools,
        events,
        sync: tool_sync,
        skill_index,
        skill_provider,
    } = build_tools(
        config,
        provider.clone(),
        llm_config.clone(),
        cli.dangerously_skip_permissions,
        cli.tools.as_deref(),
    )
    .await;

    let cwd = std::env::current_dir()?;
    let mut system_prompt = match cli.system.as_deref().or(config.system_prompt.as_deref()) {
        Some(prompt) => prompt.to_string(),
        None => prompt::default_prompt(&tools, &cwd),
    };
    system_prompt.push_str(&project_instructions(
        &cwd,
        global_level(),
        is_claude_model(model),
    ));
    system_prompt.push_str(&skill_instructions(
        &skill_index,
        tools.contains("skill_load"),
    ));
    let always_skills = always_on_skills(
        &*skill_provider.read().await,
        config.always_skills.as_deref().unwrap_or_default(),
        &skill_flag_dirs()?,
    );
    system_prompt.push_str(&always_skills.prompt);

    let auto_compact = config.auto_compact_for(model);
    let tool_reminder = prompt::tool_reminder(&tools);
    let agent = Agent::builder(provider.clone())
        .periodic_reminder(tool_reminder)
        .user_update_reminder(prompt::USER_UPDATE_REMINDER)
        .tools(tools)
        .background_events(events)
        .llm(llm_config)
        .auto_compact(auto_compact.enabled)
        .compact_at_tokens(auto_compact.tokens)
        .working_dir(cwd)
        .allow_parallel_tool_calls(true)
        .build()
        .await;

    let pricing = fetch_pricing(&provider, model).await;
    let context_window = provider.context_window(model).await;

    Ok(Startup {
        subagents,
        agent,
        tool_sync,
        skill_provider,
        system_prompt,
        reasoning,
        pricing,
        context_window,
        always_skills,
    })
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn tool_selection_is_unrestricted_by_default_and_can_be_explicitly_filtered() {
        for args in [vec!["flsh"], vec!["flsh", "resume"]] {
            let default = Cli::try_parse_from(args).unwrap();
            assert!(default.tools.is_none());
        }
        let all = Cli::try_parse_from(["flsh", "--tools", "all"]).unwrap();
        assert_eq!(all.tools.unwrap(), ["all"]);
        let selected = Cli::try_parse_from(["flsh", "--tools", "read,exec"]).unwrap();
        assert_eq!(selected.tools.unwrap(), ["read", "exec"]);
    }

    #[test]
    fn skip_permissions_is_opt_in_and_works_with_resume() {
        assert!(
            !Cli::try_parse_from(["flsh"])
                .unwrap()
                .dangerously_skip_permissions
        );
        for args in [
            vec!["flsh", "--dangerously-skip-permissions"],
            vec!["flsh", "--dangerously-skip-permissions", "resume"],
            vec!["flsh", "resume", "--dangerously-skip-permissions"],
        ] {
            assert!(
                Cli::try_parse_from(args)
                    .unwrap()
                    .dangerously_skip_permissions
            );
        }
    }
}
