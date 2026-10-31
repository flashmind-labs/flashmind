mod agents;
mod autocompact;
mod config;
mod file_tools;
mod instructions;
mod interactive;
mod mention;
mod metadata;
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
mod usage;

use std::convert::Infallible;
use std::fmt::{Debug, Formatter};
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use url::Url;

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

    /// Base URL for an OpenAI-compatible endpoint (overrides config)
    #[arg(long, global = true)]
    base_url: Option<Url>,

    /// API key for an OpenAI-compatible endpoint (overrides config)
    #[arg(long, visible_alias = "key", global = true)]
    api_key: Option<ApiKey>,

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
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
#[serde(transparent)]
struct ApiKey(String);

impl Debug for ApiKey {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[redacted]")
    }
}

impl FromStr for ApiKey {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self(value.to_owned()))
    }
}

impl Cli {
    fn apply_provider_overrides(&self, mut config: Config) -> Config {
        config.endpoint_override.url = self.base_url.clone();
        config.endpoint_override.api_key = self.api_key.clone();
        if let Some(base_url) = &self.base_url {
            config.openai_base_url = Some(base_url.to_string());
        }
        if let Some(api_key) = &self.api_key {
            config.openai_api_key = Some(api_key.0.clone());
        }
        config
    }
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
        Some(Command::Setup) => {
            return run_setup(&cli.apply_provider_overrides(load_config()?)).await;
        }
        Some(Command::Resume {
            ref session_id,
            all,
        }) => {
            return run_resume(
                &cli,
                &cli.apply_provider_overrides(load_config()?),
                all,
                session_id.as_ref(),
            )
            .await;
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

    let config = cli.apply_provider_overrides(load_config()?);
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
    if tools.contains("delegate") {
        system_prompt.push_str(&crate::agents::AgentStore::open()?.instructions(true));
    }
    let always_skills = always_on_skills(
        &*skill_provider.read().await,
        config.always_skills.as_deref().unwrap_or_default(),
        &skill_flag_dirs()?,
    );
    system_prompt.push_str(&always_skills.prompt);

    let auto_compact = config.auto_compact_for(model);
    let tool_reminder = prompt::tool_reminder(&tools);
    let mut agent = Agent::builder(provider.clone())
        .periodic_reminder(tool_reminder)
        .user_update_reminder(prompt::USER_UPDATE_REMINDER)
        .tools(tools)
        .background_events(events)
        .llm(llm_config)
        .auto_compact(auto_compact.enabled)
        .compact_at_tokens(auto_compact.tokens)
        .working_dir(cwd)
        .allow_parallel_tool_calls(true)
        .build_sync();

    let mut pricing = ModelPricing::default();
    let mut context_window = None;
    if cli.prompt.is_some() {
        agent.refresh_features().await;
        pricing = fetch_pricing(&provider, model).await;
        context_window = provider.context_window(model).await;
    }

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

    #[tokio::test]
    async fn interactive_startup_does_not_contact_the_model_provider() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config {
            ollama_url: Some(format!("http://{}", listener.local_addr().unwrap())),
            ..Default::default()
        };
        let cli = Cli::try_parse_from(["flsh"]).unwrap();
        let model = "ollama:example".parse().unwrap();
        let key = session::new_session_key();
        tokio::select! {
            result = start_session(&cli, &config, &model, &key) => {
                let startup = result.unwrap();
                startup.tool_sync.shutdown().await;
            }
            connection = listener.accept() => {
                connection.unwrap();
                panic!("Interactive startup must show the UI before contacting the provider.");
            }
        }
    }

    #[test]
    fn endpoint_flags_override_config_for_chat_and_resume() {
        for command in [None, Some("resume")] {
            for key_flag in ["--api-key", "--key"] {
                let mut args = vec!["flsh"];
                if let Some(command) = command {
                    args.push(command);
                }
                args.extend([
                    "--model",
                    "openai:example-model",
                    "--base-url",
                    "https://provider.example/v1",
                    key_flag,
                    "example-key",
                ]);
                let cli = Cli::try_parse_from(args).unwrap();
                let config = cli.apply_provider_overrides(Config {
                    openai_base_url: Some("https://configured.example/v1".into()),
                    openai_api_key: Some("configured-key".into()),
                    ..Default::default()
                });
                assert_eq!(
                    config.openai_base_url.as_deref(),
                    Some("https://provider.example/v1")
                );
                assert_eq!(config.openai_api_key.as_deref(), Some("example-key"));
                let model = resolve_model(&cli, &config).unwrap();
                assert!(build_provider(&model.provider, &config).is_ok());
                assert_eq!(format!("{:?}", cli.api_key.as_ref().unwrap()), "[redacted]");
            }
        }
    }

    #[test]
    fn absent_endpoint_flags_preserve_config() {
        let cli = Cli::try_parse_from(["flsh"]).unwrap();
        let config = cli.apply_provider_overrides(Config {
            openai_base_url: Some("https://configured.example/v1".into()),
            openai_api_key: Some("configured-key".into()),
            ..Default::default()
        });
        assert_eq!(
            config.openai_base_url.as_deref(),
            Some("https://configured.example/v1")
        );
        assert_eq!(config.openai_api_key.as_deref(), Some("configured-key"));
        assert!(Cli::try_parse_from(["flsh", "--base-url", "invalid"]).is_err());
    }

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
}
