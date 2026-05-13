//! Main REPL loop — wires together config, sessions, TUI, and agent.

use anyhow::Result;
use ratatui::crossterm::event::Event;
use rust_decimal::Decimal;
use tokio::sync::mpsc;

use flashmind_core::{Agent, Conversation, ConversationEntry};
use flashmind_types::model::{Model, ReasoningLevel};
use flashmind_types::{AgentEvent, AgentInput};

use crate::commands::{self, Command};
use crate::config::Config;
use crate::display::{self, DisplayLog};
use crate::session::{self, SessionPick, Sessions};
use flashmind_tools::tool_sync::ToolSync;

use crate::tui::{StreamInterrupt, TuiAction, TuiApp, TuiState, spawn_key_reader};

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
    let tool_set = config.build_tools(provider.clone(), &llm_config).await?;

    let tool_sync = tool_set.tool_sync;
    let agent = Agent::builder(provider)
        .llm(llm_config)
        .tools(tool_set.tools)
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
        tool_sync,
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
        state.app.show_tools(state.agent.tools());
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

        // Parse file attachments (@file syntax)
        let cwd_path = std::env::current_dir().unwrap_or_default();
        let att = crate::attachments::collect(&text, &cwd_path);
        for err in &att.errors {
            state
                .app
                .add_system_message(&format!("Attachment error: {err}"));
        }
        let input = if att.parts.is_empty() {
            AgentInput::user(text.clone())
        } else {
            let n = att.parts.len();
            state
                .app
                .add_system_message(&format!("[{n} file(s) attached]"));
            AgentInput::User {
                content: att.text,
                context: None,
                parts: Some(att.parts),
            }
        };

        // Sync any MCP tools registered by background connections
        state.tool_sync.sync(state.agent.tools_mut());

        // Stream agent response
        {
            let mut tui_state = TuiState::new();
            let stream = state.agent.start(&mut state.conversation, input, None);
            let interrupt = state
                .app
                .stream_response(Box::pin(stream), &mut tui_state, &mut state.key_rx, |ev| {
                    state.display_log.log_agent_event(ev)
                })
                .await?;

            if let Some(si) = interrupt {
                state.handle_interrupt(si).await?;
            }
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
    tool_sync: ToolSync,
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
                self.app.clear_lines();
                self.app
                    .add_system_message(&format!("Flash — {}", self.model_display));
                self.app.show_tools(self.agent.tools());
            }
            Command::Compact => {
                self.handle_compact().await?;
            }
            Command::Context => {
                let entries = self.conversation.entries().len();
                let ctx_window = self.agent.context_window();
                let cum_total = self.app.cumulative_prompt + self.app.cumulative_completion;
                let info = if let Some((total, prompt, completion)) = self.app.last_usage() {
                    let pct = if ctx_window > 0 {
                        (prompt as f64 / ctx_window as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    format!(
                        "Context: {}p + {}c ({} total) — {}% of {}k window — {} entries\n\
                         Session total: {}p + {}c ({} tokens)",
                        prompt,
                        completion,
                        total,
                        pct,
                        ctx_window / 1000,
                        entries,
                        self.app.cumulative_prompt,
                        self.app.cumulative_completion,
                        cum_total,
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
                                    match self.config.build_tools(provider.clone(), &new_llm).await
                                    {
                                        Ok(tool_set) => {
                                            self.model_display = new_llm.model.to_string();
                                            self.tool_sync = tool_set.tool_sync;
                                            self.agent = Agent::builder(provider)
                                                .llm(new_llm)
                                                .tools(tool_set.tools)
                                                .build();
                                            self.agent.refresh_features().await;
                                            self.app.context_window = self.agent.context_window();
                                            self.app.set_status(format!(
                                                "Flash — {}",
                                                self.model_display
                                            ));
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
            Command::Soul => {
                let path = Config::soul_path();
                let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());

                ratatui::crossterm::terminal::disable_raw_mode()?;
                let status = std::process::Command::new(&editor).arg(&path).status();
                ratatui::crossterm::terminal::enable_raw_mode()?;
                self.app.draw(None)?;

                match status {
                    Ok(s) if s.success() => {
                        let new_prompt = self.config.system_prompt();
                        self.conversation = Conversation::new();
                        self.conversation
                            .prepend(ConversationEntry::system(new_prompt));
                        self.app
                            .add_system_message("SOUL.md updated. Conversation reset.");
                    }
                    Ok(s) => {
                        self.app
                            .add_system_message(&format!("Editor exited with {s}"));
                    }
                    Err(e) => {
                        self.app.add_system_message(&format!("Error: {e}"));
                    }
                }
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
        if self.conversation.entries().len() <= 1 {
            self.app
                .add_system_message("Nothing to compact — send a message first.");
            return Ok(());
        }

        self.app.add_system_message("Compacting conversation...");
        self.app.draw(None)?;

        let summary = self
            .agent
            .compact_conversation(&mut self.conversation)
            .await;

        let ev = match summary {
            Some(s) => AgentEvent::Compacted(s),
            None => AgentEvent::Compacted(
                "[compacted via fallback — LLM summarization unavailable]".into(),
            ),
        };
        let mut tui_state = TuiState::new();
        self.display_log.log_agent_event(&ev);
        self.app.handle_agent_event(&ev, &mut tui_state);
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

    async fn handle_interrupt(&mut self, interrupt: StreamInterrupt) -> Result<()> {
        let mcp = match self.tool_sync.mcp_registry() {
            Some(r) => r,
            None => {
                self.app
                    .add_system_message("Interrupt received but no MCP registry available.");
                return Ok(());
            }
        };

        let server: String = serde_json::from_str::<serde_json::Value>(&interrupt.output)
            .ok()
            .and_then(|v| v.get("server").and_then(|s| s.as_str()).map(String::from))
            .unwrap_or_default();

        if server.is_empty() {
            self.app
                .add_system_message(&format!("Unhandled interrupt: {}", interrupt.output));
            return Ok(());
        }

        self.app.add_system_message(&format!(
            "Opening browser for OAuth with '{server}'... (waiting up to 5 minutes)"
        ));
        self.app.draw(None)?;

        match crate::mcp_auth::browser_oauth(mcp, &server).await {
            Ok(()) => {
                self.tool_sync.sync(self.agent.tools_mut());
                self.app
                    .add_system_message(&format!("Authenticated with '{server}'."));
            }
            Err(e) => {
                self.app.add_system_message(&format!("{e}"));
            }
        }

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
