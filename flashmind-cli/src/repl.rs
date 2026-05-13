//! Main REPL loop — wires together config, sessions, TUI, and agent.

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use ratatui::crossterm::event::Event;
use rust_decimal::Decimal;
use tokio::sync::mpsc;

use flashmind_core::{Agent, Conversation, ConversationEntry};
use flashmind_types::AgentInput;
use flashmind_types::model::{Model, ReasoningLevel};

use crate::commands::{self, Command};
use crate::config::Config;
use crate::display::{self, DisplayLog};
use crate::session::{self, SessionPick, Sessions};
use crate::tui::{TuiAction, TuiApp, TuiState, spawn_key_reader};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(model_override: Option<Model>, no_restore: bool) -> Result<()> {
    let config = Config::load()?;
    Config::init()?;

    let sessions = Sessions::connect(&Config::db_path()).await?;

    // Pick or create session
    let (session_key, is_resume) = if no_restore {
        (session::new_session_key(), false)
    } else {
        let local = sessions.list_local().await?;
        let (pick, deleted) = session::pick_session(&local)?;
        for key in &deleted {
            sessions.delete(key).await?;
            let _ = std::fs::remove_file(Config::session_display_path(key));
        }
        match pick {
            SessionPick::Resume(key) => (key, true),
            SessionPick::New(key) => (key, false),
        }
    };

    // Build agent
    let llm_config = config.build_llm_config(model_override.as_ref())?;
    let provider = config.build_provider_for(&llm_config.model.provider)?;
    let model_display = llm_config.model.to_string();
    let tools = config.build_tools();

    let agent = Agent::builder(provider)
        .llm(llm_config)
        .tools(tools)
        .build();

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system(config.system_prompt()));

    let mut app = TuiApp::new()?;
    let key_rx = spawn_key_reader();

    // Load history
    let history_path = Config::base_dir().join("history.jsonl");
    app.load_history(&history_path);

    let mut state = ReplState {
        config,
        sessions,
        session_key,
        agent,
        conversation,
        display_log: DisplayLog::new(),
        app,
        key_rx,
        model_display,
    };

    // Fetch context window and model capabilities
    state.agent.refresh_features().await;
    state.app.context_window = state.agent.context_window();

    // Restore session
    if is_resume {
        if let Some(conv) = state.sessions.load(&state.session_key).await? {
            state.conversation = conv;
        }
        let display_path = Config::session_display_path(&state.session_key);
        let events = display::load(&display_path);
        if !events.is_empty() {
            state.app.load_display_log(&events);
            state.display_log.extend(events);
        }
        state
            .app
            .add_system_message(&format!("Session resumed ({})", state.session_key));
    } else {
        state
            .app
            .add_system_message(&format!("Flash — {}", state.model_display));
        state.app.newline();
    }
    state
        .app
        .set_status(format!("Flash — {}", state.model_display));

    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    state
        .conversation
        .prepend(ConversationEntry::system(state.config.system_prompt()));

    loop {
        let action = state.app.read_input(&mut state.key_rx).await?;
        let text = match action {
            TuiAction::Submit(text) => text,
            TuiAction::Quit => break,
            TuiAction::Cancel | TuiAction::None => continue,
        };

        // Slash commands
        if let Some(cmd) = commands::parse(&text) {
            match state.handle_command(cmd).await? {
                Flow::Continue => continue,
                Flow::Quit => break,
            }
            // unreachable, but keeps the pattern clear
        }

        // Record user input
        state.app.add_user_message(&text);
        state.display_log.log_user(text.clone());

        // Stream agent response
        {
            let mut tui_state = TuiState::new();
            let stream = state.agent.start(
                &mut state.conversation,
                AgentInput::user(text.clone()),
                None,
            );
            state
                .app
                .stream_response(Box::pin(stream), &mut tui_state, &mut state.key_rx, |ev| {
                    state.display_log.log_agent_event(ev)
                })
                .await?;
        }

        // Persist
        state
            .sessions
            .save(&state.session_key, &state.conversation)
            .await?;
        state
            .sessions
            .save_metadata(&state.session_key, &text, &state.model_display, &cwd)
            .await?;
        display::save(
            &Config::session_display_path(&state.session_key),
            state.display_log.events(),
        )?;

        // Title enrichment (fire-and-forget)
        crate::enrichment::spawn_title_enrichment(
            state.session_key.clone(),
            state.conversation.to_messages(),
            state.agent.llm().model.clone(),
            state.agent.provider_arc().clone(),
            state.sessions.connection().clone(),
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// REPL state
// ---------------------------------------------------------------------------

struct ReplState<'a> {
    config: Config,
    sessions: Sessions,
    session_key: String,
    agent: Agent,
    conversation: Conversation,
    display_log: DisplayLog,
    app: TuiApp<'a>,
    key_rx: mpsc::UnboundedReceiver<Event>,
    model_display: String,
}

enum Flow {
    Continue,
    Quit,
}

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------

impl ReplState<'_> {
    async fn handle_command(&mut self, cmd: Command) -> Result<Flow> {
        match cmd {
            Command::Help => {
                self.app.add_system_message(commands::help_text());
            }
            Command::Clear => {
                self.conversation = Conversation::new();
                self.conversation
                    .prepend(ConversationEntry::system(self.config.system_prompt()));
                self.display_log.log_clear();
                self.sessions.delete(&self.session_key).await?;
                self.app.add_system_message("Conversation cleared.");
            }
            Command::Compact => {
                self.handle_compact().await?;
            }
            Command::Context => {
                let entries = self.conversation.entries().len();
                let ctx_window = self.agent.context_window();
                let info = if let Some((total, prompt, completion)) = self.app.last_usage() {
                    let pct = if ctx_window > 0 {
                        (prompt as f64 / ctx_window as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    format!(
                        "Context: {}p + {}c ({} total) — {}% of {}k window — {} entries",
                        prompt,
                        completion,
                        total,
                        pct,
                        ctx_window / 1000,
                        entries,
                    )
                } else {
                    format!("Context: {} entries (no usage data yet)", entries)
                };
                self.app.add_system_message(&info);
            }
            Command::Info => {
                let llm = self.agent.llm();
                let reasoning = match llm.reasoning {
                    ReasoningLevel::On => "on",
                    ReasoningLevel::Off => "off",
                };
                let temp = llm
                    .sampling
                    .temperature
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "default".into());
                let entries = self.conversation.entries().len();
                let ctx = self.agent.context_window();
                let info = format!(
                    "Session: {}\n\
                     Model: {}\n\
                     Temperature: {temp}\n\
                     Reasoning: {reasoning}\n\
                     Context window: {}k\n\
                     Entries: {entries}",
                    self.session_key,
                    llm.model,
                    ctx / 1000,
                );
                self.app.add_system_message(&info);
            }
            Command::Model(new_model) => {
                if let Some(name) = new_model {
                    match name.parse::<Model>() {
                        Ok(model) => {
                            let result =
                                self.config
                                    .build_provider_for(&model.provider)
                                    .and_then(|p| {
                                        self.config
                                            .build_llm_config(Some(&model))
                                            .map(|llm| (p, llm))
                                    });
                            match result {
                                Ok((provider, new_llm)) => {
                                    self.model_display = new_llm.model.to_string();
                                    self.agent = Agent::builder(provider)
                                        .llm(new_llm)
                                        .tools(self.config.build_tools())
                                        .build();
                                    self.agent.refresh_features().await;
                                    self.app.context_window = self.agent.context_window();
                                    self.app
                                        .set_status(format!("Flash — {}", self.model_display));
                                    self.app.add_system_message(&format!(
                                        "Switched to {}",
                                        self.model_display
                                    ));
                                }
                                Err(e) => {
                                    self.app.add_system_message(&format!("Error: {e}"));
                                }
                            }
                        }
                        Err(e) => {
                            self.app.add_system_message(&format!("Error: {e}"));
                        }
                    }
                } else {
                    self.app
                        .add_system_message(&format!("Current model: {}", self.model_display));
                }
            }
            Command::Temperature(arg) => {
                if let Some(val) = arg {
                    match val.parse::<Decimal>() {
                        Ok(t) => {
                            self.agent.llm_mut().sampling.temperature = Some(t);
                            self.app
                                .add_system_message(&format!("Temperature set to {t}"));
                        }
                        Err(_) => {
                            self.app
                                .add_system_message("Invalid temperature — use a number like 0.7");
                        }
                    }
                } else {
                    let t = self
                        .agent
                        .llm()
                        .sampling
                        .temperature
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "default".into());
                    self.app.add_system_message(&format!("Temperature: {t}"));
                }
            }
            Command::Thinking(arg) => {
                if let Some(val) = arg {
                    match val.as_str() {
                        "on" | "true" | "1" => {
                            self.agent.llm_mut().reasoning = ReasoningLevel::On;
                            self.app.add_system_message("Reasoning enabled.");
                        }
                        "off" | "false" | "0" => {
                            self.agent.llm_mut().reasoning = ReasoningLevel::Off;
                            self.app.add_system_message("Reasoning disabled.");
                        }
                        _ => {
                            self.app.add_system_message("Usage: /thinking [on|off]");
                        }
                    }
                } else {
                    let mode = match self.agent.llm().reasoning {
                        ReasoningLevel::On => "on",
                        ReasoningLevel::Off => "off",
                    };
                    self.app.add_system_message(&format!("Reasoning: {mode}"));
                }
            }
            Command::Title(arg) => {
                if let Some(title) = arg {
                    if let Err(e) = session::set_title(
                        self.sessions.connection(),
                        &self.session_key,
                        Some(&title),
                    )
                    .await
                    {
                        self.app.add_system_message(&format!("Error: {e}"));
                    } else {
                        self.app.add_system_message(&format!("Title set: {title}"));
                    }
                } else {
                    match self.sessions.list_local().await {
                        Ok(list) => {
                            let title = list
                                .iter()
                                .find(|s| s.key == self.session_key)
                                .and_then(|s| s.title.clone());
                            match title {
                                Some(t) => self.app.add_system_message(&format!("Title: {t}")),
                                None => self.app.add_system_message("No title set."),
                            }
                        }
                        Err(e) => self.app.add_system_message(&format!("Error: {e}")),
                    }
                }
            }
            Command::Sessions => {
                let list = self.sessions.list_local().await?;
                if list.is_empty() {
                    self.app.add_system_message("No saved sessions.");
                } else {
                    let mut buf = String::from("Sessions:\n");
                    for s in &list {
                        let active = if s.key == self.session_key { " *" } else { "" };
                        let label = s.title.as_deref().unwrap_or(&s.prompt);
                        buf.push_str(&format!(
                            "  {} — {} — {}{}\n",
                            s.key, s.model, label, active
                        ));
                    }
                    self.app.add_system_message(buf.trim_end());
                }
            }
            Command::Save(arg) => {
                save_output(&mut self.app, arg.as_deref().unwrap_or(""));
            }
            Command::Setup => {
                let provider = self.agent.provider_arc().clone();
                let llm_config = self.agent.llm().clone();
                crate::wizard::run_setup(
                    provider,
                    llm_config,
                    &mut self.app,
                    &mut self.key_rx,
                    &mut self.display_log,
                )
                .await?;
            }
            Command::Sub(arg) => {
                if let Some(task) = arg {
                    let prompt =
                        format!("Spawn a background subagent to work on this task: {task}");
                    self.send_to_agent(&prompt).await?;
                } else {
                    self.app.add_system_message("Usage: /sub <task>");
                }
            }
            Command::Subagents => {
                self.send_to_agent("List all active subagents and their current status.")
                    .await?;
            }
            Command::Quit => return Ok(Flow::Quit),
            Command::Unknown(name) => {
                self.app.add_system_message(&format!(
                    "Unknown command: /{name}. Type /help for available commands."
                ));
            }
        }
        Ok(Flow::Continue)
    }

    async fn handle_compact(&mut self) -> Result<()> {
        let last_usage = self.app.last_usage();
        let estimated = last_usage.map(|(_, p, _)| p).unwrap_or(0);
        let ctx_window = self.agent.context_window();

        if estimated == 0 || ctx_window == 0 {
            self.app
                .add_system_message("No token usage data — send a message first.");
            return Ok(());
        }

        self.app.add_system_message(&format!(
            "Compacting ({}/{}k tokens)...",
            estimated,
            ctx_window / 1000
        ));
        self.app.draw(None)?;

        let provider: Arc<dyn flashmind_types::LlmProvider> = self.agent.provider_arc().clone();
        let model = self.agent.llm().model.clone();

        {
            let compact_stream = flashmind_core::compaction::try_compact(
                &mut self.conversation,
                estimated,
                ctx_window,
                &*provider,
                &model,
            );
            tokio::pin!(compact_stream);

            let mut tui_state = TuiState::new();
            while let Some(ev) = compact_stream.next().await {
                self.display_log.log_agent_event(&ev);
                self.app.handle_agent_event(&ev, &mut tui_state);
            }
        }
        self.app.draw(None)?;

        self.sessions
            .save(&self.session_key, &self.conversation)
            .await?;
        display::save(
            &Config::session_display_path(&self.session_key),
            self.display_log.events(),
        )?;

        Ok(())
    }

    async fn send_to_agent(&mut self, prompt: &str) -> Result<()> {
        self.app.add_user_message(prompt);
        self.display_log.log_user(prompt.to_string());

        {
            let mut tui_state = TuiState::new();
            let stream = self.agent.start(
                &mut self.conversation,
                AgentInput::user(prompt.to_string()),
                None,
            );
            self.app
                .stream_response(Box::pin(stream), &mut tui_state, &mut self.key_rx, |ev| {
                    self.display_log.log_agent_event(ev)
                })
                .await?;
        }

        self.sessions
            .save(&self.session_key, &self.conversation)
            .await?;
        display::save(
            &Config::session_display_path(&self.session_key),
            self.display_log.events(),
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn save_output(app: &mut TuiApp, path: &str) {
    if path.is_empty() {
        app.add_system_message("Usage: /save <filename>");
        return;
    }

    let mut output = String::new();
    for line in &app.lines {
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        output.push_str(&text);
        output.push('\n');
    }

    let line_count = app.lines.len();
    match std::fs::write(path, &output) {
        Ok(()) => {
            app.add_system_message(&format!("[saved {} lines to {}]", line_count, path));
        }
        Err(e) => {
            app.add_system_message(&format!("[save error: {}]", e));
        }
    }
}
