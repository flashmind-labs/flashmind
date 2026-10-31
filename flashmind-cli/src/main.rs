mod agents;
mod autocompact;
mod config;
mod file_tools;
mod instructions;
mod interactive;
mod media;
mod mention;
mod metadata;
mod model_completion;
mod models;
mod ocr;
mod ops;
mod paths;
mod pi_import;
mod plugins;
mod prompt;
mod provider;
mod session;
mod setup;
mod shell_completion;
mod subagents;
mod terminal_title;
mod three_d;
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
use clap::{Parser, Subcommand};

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
#[command(
    name = "flsh",
    about = "AI chat & coding agent",
    after_help = "Guide: https://github.com/flashmind-labs/flashmind/blob/main/flashmind-cli/GUIDE.md"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Model (e.g. "ollama:llama3.2", "openrouter:anthropic/claude-sonnet-4")
    #[arg(short, long, global = true)]
    pub model: Option<String>,

    /// Add a JSON request field as KEY=JSON_VALUE (repeatable)
    #[arg(
        short = 'e',
        long = "extra-args",
        global = true,
        value_name = "KEY=JSON_VALUE"
    )]
    extra_args: Vec<ExtraArg>,

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

#[derive(Clone)]
struct ExtraArg {
    key: String,
    value: serde_json::Value,
}

impl FromStr for ExtraArg {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (key, value) = input.split_once('=').ok_or("expected KEY=JSON_VALUE")?;
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(
                "extra field name must contain only letters, numbers, or underscores".into(),
            );
        }
        if [
            "model",
            "messages",
            "prompt",
            "input",
            "stream",
            "voice",
            "voice_id",
            "response_format",
            "size",
            "aspect_ratio",
            "resolution",
            "quality",
            "style",
            "n",
            "duration",
            "seconds",
            "ratio",
            "generate_audio",
        ]
        .contains(&key)
        {
            return Err(format!(
                "{key} is controlled by the command; use its dedicated option"
            ));
        }
        let value =
            serde_json::from_str(value).map_err(|error| format!("invalid JSON value: {error}"))?;
        Ok(Self {
            key: key.into(),
            value,
        })
    }
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
        config.endpoint_override.extra_args = self
            .extra_args
            .iter()
            .map(|arg| (arg.key.clone(), arg.value.clone()))
            .collect();
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
    /// Browse and search available models
    Models {
        /// Limit the catalog to one provider (custom providers are also accepted)
        #[arg(value_parser = shell_completion::providers(), hide_possible_values = true)]
        provider: Option<flashmind_types::Provider>,
    },
    /// List voices available from a provider
    Voices {
        /// Provider to query (custom providers are also accepted)
        #[arg(value_parser = shell_completion::providers(), hide_possible_values = true)]
        provider: flashmind_types::Provider,
    },
    /// Extract text from a document with Mistral OCR
    Ocr(ocr::OcrArgs),
    /// Generate a 3D model from text or an image
    #[command(name = "3d")]
    ThreeD(three_d::ThreeDArgs),
    /// Generate text with a model
    Text(media::TextArgs),
    /// Generate images
    #[command(
        subcommand_precedence_over_arg = true,
        args_conflicts_with_subcommands = true
    )]
    Image(media::ImageArgs),
    /// Generate a video
    Video(media::VideoArgs),
    /// Generate speech
    Speech(media::SpeechArgs),
    /// Transcribe an audio file
    Transcribe(media::TranscribeArgs),
    /// Generate or install shell completions
    Completions {
        /// Shell to generate for, or install to configure your shell
        target: shell_completion::CompletionTarget,
        /// Shell to configure (defaults to the login shell)
        #[arg(long)]
        shell: Option<shell_completion::InstallShell>,
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
        Some(Command::Models { ref provider }) => {
            return models::run(
                provider.as_ref(),
                &cli.apply_provider_overrides(load_config()?),
            )
            .await;
        }
        Some(Command::Voices { ref provider }) => {
            let config = cli.apply_provider_overrides(load_config()?);
            let llm = build_provider(provider, &config)?;
            let voices =
                tokio::time::timeout(std::time::Duration::from_secs(15), llm.list_voices(""))
                    .await
                    .map_err(|_| anyhow::anyhow!("voice listing timed out"))?
                    .ok_or_else(|| anyhow::anyhow!("could not list voices for {provider}"))?;
            for voice in voices {
                println!("{}\t{}", voice.id, voice.name);
            }
            return Ok(());
        }
        Some(Command::Ocr(ref args)) => {
            let config = cli.apply_provider_overrides(load_config()?);
            return ocr::run(args, cli.model.as_deref(), &config).await;
        }
        Some(Command::ThreeD(ref args)) => {
            let model = media_model(&cli)?;
            let config = cli.apply_provider_overrides(load_config()?);
            return three_d::run(&model, &config, args.clone()).await;
        }
        Some(Command::Text(ref args)) => {
            let (model, provider) = media_provider(&cli)?;
            return media::text(provider.as_ref(), model, args.clone()).await;
        }
        Some(Command::Image(ref args)) => {
            let (model, provider) = media_provider(&cli)?;
            match &args.command {
                Some(media::ImageCommand::Edit(edit)) => {
                    return media::image_edit(provider.as_ref(), &model, edit.clone()).await;
                }
                Some(media::ImageCommand::Gen(generation)) => {
                    return media::image(provider.as_ref(), &model, generation.clone()).await;
                }
                None => {
                    return media::image(provider.as_ref(), &model, args.generation.clone()).await;
                }
            }
        }
        Some(Command::Video(ref args)) => {
            let (model, provider) = media_provider(&cli)?;
            return media::video(provider.as_ref(), &model, args.clone()).await;
        }
        Some(Command::Speech(ref args)) => {
            let (model, provider) = media_provider(&cli)?;
            return media::speech(provider.as_ref(), &model, args.clone()).await;
        }
        Some(Command::Transcribe(ref args)) => {
            anyhow::ensure!(
                cli.extra_args.is_empty(),
                "--extra-args is not supported for multipart transcription; use --provider-options with OpenRouter"
            );
            let model = media_model(&cli)?;
            if model.provider == flashmind_types::Provider::OpenRouter {
                let config = cli.apply_provider_overrides(load_config()?);
                let key = config
                    .api_key_for(&model.provider)
                    .ok_or_else(|| anyhow::anyhow!("OpenRouter API key required"))?;
                let provider = flashmind_llm::OpenRouterProvider::new(key.to_owned());
                return media::transcribe(&provider, &model, args.clone()).await;
            }
            let (_, provider) = media_provider(&cli)?;
            return media::transcribe_compatible(provider.as_ref(), &model, args.clone()).await;
        }
        Some(Command::Completions { target, shell }) => {
            if let Some(target_shell) = target.shell() {
                anyhow::ensure!(
                    shell.is_none(),
                    "--shell is only valid with completions install"
                );
                shell_completion::generate(target_shell, io::stdout())?;
            } else {
                shell_completion::install(shell).await?;
            }
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

fn media_model(cli: &Cli) -> Result<Model> {
    let raw = cli
        .model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("pass a model with --model provider:name"))?;
    let qualified = if raw.contains(':') {
        raw.to_owned()
    } else {
        format!("openrouter:{raw}")
    };
    qualified.parse().map_err(Into::into)
}

fn media_provider(cli: &Cli) -> Result<(Model, Arc<dyn flashmind_types::LlmProvider>)> {
    let model = media_model(cli)?;
    let config = cli.apply_provider_overrides(load_config()?);
    let provider = build_provider(&model.provider, &config)?;
    Ok((model, provider))
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

    #[test]
    fn image_edit_command_accepts_multiple_images_and_global_options() {
        let cli = Cli::try_parse_from([
            "flsh",
            "image",
            "edit",
            "-m",
            "venice:example-edit",
            "--image",
            "base.webp",
            "--image",
            "https://example.com/second.png",
            "--resolution",
            "2K",
            "-o",
            "result.png",
            "change the sky",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Image(media::ImageArgs {
                command: Some(media::ImageCommand::Edit(_)),
                ..
            }))
        ));
        assert!(
            Cli::try_parse_from([
                "flsh",
                "image",
                "edit",
                "-m",
                "venice:example-edit",
                "change the sky"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "flsh",
                "image",
                "-m",
                "venice:example",
                "generate a landscape"
            ])
            .is_ok()
        );
    }

    #[test]
    fn image_gen_command_accepts_generation_options() {
        let cli = Cli::try_parse_from([
            "flsh",
            "image",
            "gen",
            "-m",
            "venice:example",
            "--size",
            "1K",
            "-o",
            "result.png",
            "generate a landscape",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Image(media::ImageArgs {
                command: Some(media::ImageCommand::Gen(_)),
                ..
            }))
        ));
    }

    #[test]
    fn extra_args_parse_json_values_and_reject_managed_fields() {
        let cli = Cli::try_parse_from([
            "flsh",
            "image",
            "-m",
            "venice:example",
            "-e",
            "lora_strength=50",
            "-e",
            r#"negative_prompt="clouds,rain""#,
            "-e",
            "enhance_prompt=true",
            "example",
        ])
        .unwrap();
        let config = cli.apply_provider_overrides(Config::default());
        assert_eq!(config.endpoint_override.extra_args["lora_strength"], 50);
        assert_eq!(
            config.endpoint_override.extra_args["negative_prompt"],
            "clouds,rain"
        );
        assert_eq!(config.endpoint_override.extra_args["enhance_prompt"], true);
        for bad in ["negative_prompt=clouds", "model=1", "=1", "bad key=1"] {
            assert!(
                Cli::try_parse_from([
                    "flsh",
                    "image",
                    "-m",
                    "venice:example",
                    "-e",
                    bad,
                    "example"
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn generation_commands_accept_provider_models_and_options() {
        for args in [
            vec![
                "flsh",
                "image",
                "-m",
                "openrouter:example/image",
                "--aspect-ratio",
                "1:1",
                "prompt",
            ],
            vec![
                "flsh",
                "video",
                "-m",
                "openrouter:example/video",
                "--duration",
                "5",
                "prompt",
            ],
            vec![
                "flsh",
                "speech",
                "-m",
                "openrouter:example/speech",
                "--voice",
                "example",
                "hello",
            ],
            vec![
                "flsh",
                "transcribe",
                "-m",
                "openrouter:example/stt",
                "audio.mp3",
                "--timestamps",
                "segment,word",
            ],
            vec![
                "flsh",
                "text",
                "-m",
                "example/text",
                "--reasoning",
                "high",
                "hello",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(
                media_model(&cli).unwrap().provider,
                flashmind_types::Provider::OpenRouter
            );
        }
    }

    #[test]
    fn transcription_rejects_invalid_provider_options() {
        for options in ["[]", "null", r#"{"example":true}"#] {
            assert!(
                Cli::try_parse_from([
                    "flsh",
                    "transcribe",
                    "-m",
                    "openrouter:example/stt",
                    "audio.mp3",
                    "--provider-options",
                    options,
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn three_d_command_accepts_text_or_image() {
        let cli = Cli::try_parse_from([
            "flsh",
            "3d",
            "-m",
            "meshy:example",
            "-o",
            "model.glb",
            "--texture",
            "a chair",
        ])
        .unwrap();
        assert!(matches!(cli.command, Some(Command::ThreeD(_))));
        assert_eq!(media_model(&cli).unwrap().provider.to_string(), "meshy");
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "tripo:example",
                "-o",
                "model.glb",
                "--image",
                "input.png",
                "--image",
                "left.png"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "tripo:example",
                "-o",
                "model.glb",
                "--image",
                "input.png",
                "extra text"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from(["flsh", "3d", "-m", "tripo:example", "-o", "model.glb"]).is_err()
        );
    }

    #[test]
    fn three_d_rig_options_require_a_rig_and_accept_existing_models() {
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "meshy:latest",
                "--rig",
                "--rig-model",
                "input.glb",
                "--remesh",
                "--height",
                "1.8",
                "-o",
                "rigged.glb"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "meshy:latest",
                "--rig-model",
                "input.glb",
                "-o",
                "rigged.glb"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "meshy:latest",
                "--remesh",
                "-o",
                "model.glb",
                "character"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "flsh",
                "3d",
                "-m",
                "meshy:latest",
                "--rig",
                "--rig-model",
                "input.glb",
                "--image",
                "image.png",
                "-o",
                "rigged.glb"
            ])
            .is_err()
        );
    }

    #[test]
    fn elevenlabs_voice_and_speech_commands_parse() {
        let cli = Cli::try_parse_from(["flsh", "voices", "elevenlabs"]).unwrap();
        assert!(
            matches!(cli.command, Some(Command::Voices { provider: flashmind_types::Provider::Custom(ref name) }) if name == "elevenlabs")
        );
        let cli = Cli::try_parse_from([
            "flsh",
            "speech",
            "-m",
            "elevenlabs:example-model",
            "--voice",
            "example-voice",
            "hello",
        ])
        .unwrap();
        assert_eq!(
            media_model(&cli).unwrap().provider.to_string(),
            "elevenlabs"
        );
    }

    #[test]
    fn completion_install_accepts_a_shell_override() {
        let cli =
            Cli::try_parse_from(["flsh", "completions", "install", "--shell", "fish"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Completions {
                target: shell_completion::CompletionTarget::Install,
                shell: Some(shell_completion::InstallShell::Fish)
            })
        ));
        assert!(Cli::try_parse_from(["flsh", "completions", "zsh"]).is_ok());
    }

    #[test]
    fn shell_completions_include_commands_options_and_provider_suggestions() {
        for shell in [
            clap_complete::Shell::Zsh,
            clap_complete::Shell::Bash,
            clap_complete::Shell::Fish,
        ] {
            let mut script = Vec::new();
            shell_completion::generate(shell, &mut script).unwrap();
            let script = String::from_utf8(script).unwrap();
            for item in [
                "models",
                "3d",
                "install",
                "speech",
                "reference",
                "elevenlabs",
                "venice",
                "wav",
            ] {
                assert!(script.contains(item), "{shell:?} is missing {item}");
            }
        }
        let cli = Cli::try_parse_from(["flsh", "models", "example-provider"]).unwrap();
        assert!(
            matches!(cli.command, Some(Command::Models { provider: Some(flashmind_types::Provider::Custom(ref name)) }) if name == "example-provider")
        );
    }

    #[test]
    fn models_command_accepts_an_optional_provider() {
        let cli = Cli::try_parse_from(["flsh", "models"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Models { provider: None })
        ));
        let cli = Cli::try_parse_from(["flsh", "models", "openrouter"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Models {
                provider: Some(flashmind_types::Provider::OpenRouter)
            })
        ));
    }

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
