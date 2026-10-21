use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::text::{Line, Span};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use flashmind_core::{
    Agent, AgentManager, CancellationToken, Conversation, ConversationEntry, EntryKind,
    MAX_CONTINUATIONS, RecoveryAttempts,
};
use flashmind_memory::session::SessionStore;
use flashmind_memory::session::SessionSummary;
use flashmind_skills::{DiskSkillProvider, SkillProvider};
use flashmind_tui::styles::{S_AGENT, S_DIM, S_TOOL_FAIL};
use flashmind_tui::widgets::{
    ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse, PastedImage, StatusInfo,
};
use flashmind_tui::{CommandInfo, Repl, Tui};
use flashmind_types::llm::TokenUsage;
use flashmind_types::utils::truncate_utf8;
use flashmind_types::{
    AgentEvent, AgentInput, CompactionReason, CompletionRequest, ContentPart, LlmProvider, Message,
    Model, ModelPricing, Provider, ReasoningLevel, SessionId, StreamEvent, TurnStatus,
};
use tokio::sync::RwLock;

use crate::autocompact::AutoCompactCommand;
use crate::config::Config;
use crate::provider::{build_provider, fetch_pricing};
use crate::session::{format_session_age, load_conversation_from, new_session_key, save_turn};
use crate::setup::{run_choice, run_choice_action};
use crate::subagents::SubagentNotifications;
use crate::terminal_title::TerminalTitle;
use crate::tools::AlwaysOnSkills;
use flashmind_types::SessionUsage;

use crate::usage::{ModelPriceSummary, UsageSummary};

// ---------------------------------------------------------------------------
// Display log — stores raw events for session restore
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DisplayEvent {
    User { text: String },
    Skill { name: String, body: String },
    Agent { event: AgentEvent },
    Clear,
}

struct DisplayLog {
    path: PathBuf,
    events: Vec<DisplayEvent>,
    saved_len: usize,
    rewrite_required: bool,
}

impl DisplayLog {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            events: Vec::new(),
            saved_len: 0,
            rewrite_required: false,
        }
    }

    /// Open the log at `path`, keeping any events already saved there.
    fn open(path: PathBuf) -> Self {
        let events = Self::load(&path);
        let saved_len = events.len();
        Self {
            events,
            path,
            saved_len,
            rewrite_required: false,
        }
    }

    fn log_user(&mut self, text: &str) {
        self.events.push(DisplayEvent::User {
            text: text.to_string(),
        });
    }

    fn log_skill(&mut self, name: &str, body: &str) {
        self.events.push(DisplayEvent::Skill {
            name: name.to_string(),
            body: body.to_string(),
        });
    }

    fn log_event(&mut self, event: &AgentEvent) {
        // The summary replaces the history, so a resume starts from it.
        if matches!(event, AgentEvent::Compacted(_)) {
            self.events.clear();
            self.rewrite_required = true;
        }
        self.events.push(DisplayEvent::Agent {
            event: event.clone(),
        });
    }

    /// Drop the last user message and everything after it.
    fn pop_last_turn(&mut self) {
        if let Some(i) = self
            .events
            .iter()
            .rposition(|event| matches!(event, DisplayEvent::User { .. }))
        {
            self.events.truncate(i);
            self.rewrite_required = true;
        }
    }

    fn clear(&mut self) {
        self.events.clear();
        self.events.push(DisplayEvent::Clear);
        self.rewrite_required = true;
    }

    fn save(&mut self) -> io::Result<()> {
        let rewrite = self.rewrite_required || !self.path.exists();
        let file = if rewrite {
            std::fs::File::create(&self.path)?
        } else {
            std::fs::OpenOptions::new().append(true).open(&self.path)?
        };
        let mut writer = io::BufWriter::new(file);
        let result = (|| {
            for event in &self.events[if rewrite { 0 } else { self.saved_len }..] {
                serde_json::to_writer(&mut writer, event).map_err(io::Error::other)?;
                writeln!(writer)?;
            }
            writer.flush()
        })();
        if result.is_ok() {
            self.saved_len = self.events.len();
            self.rewrite_required = false;
        } else {
            self.rewrite_required = true;
        }
        result
    }

    fn load(path: &Path) -> Vec<DisplayEvent> {
        let Ok(file) = std::fs::File::open(path) else {
            return Vec::new();
        };
        io::BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect()
    }
}

fn save_display_log(log: &mut DisplayLog, repl: &mut Repl<'_>) -> io::Result<()> {
    if let Err(error) = log.save() {
        repl.emit_event(&AgentEvent::Error(format!(
            "Could not save session display log: {error}"
        )))?;
    }
    Ok(())
}

fn replay_display_log(repl: &mut Repl<'_>, events: &[DisplayEvent]) -> io::Result<()> {
    for event in events {
        match event {
            DisplayEvent::User { text } => {
                repl.replay_user_input(text)?;
            }
            DisplayEvent::Skill { name, body } => {
                show_skill(repl, name, body)?;
            }
            DisplayEvent::Agent { event } => {
                repl.replay_event(event)?;
            }
            DisplayEvent::Clear => repl.clear_transcript()?,
        }
    }
    repl.replay_finish_turn()
}

fn show_skill(repl: &mut Repl<'_>, name: &str, body: &str) -> io::Result<()> {
    let mut lines = vec![
        Line::default(),
        Line::from(Span::styled(format!("  Skill: /{name}"), S_AGENT)),
    ];
    lines.extend(
        body.lines()
            .map(|line| Line::from(Span::styled(format!("  {line}"), S_DIM))),
    );
    lines.push(Line::default());
    repl.print_lines(&lines)
}

/// Replay the saved display log when it has events, else rebuild the
/// transcript from the conversation.
fn replay_session(
    repl: &mut Repl<'_>,
    display_log: Option<&DisplayLog>,
    conversation: &Conversation,
) -> io::Result<()> {
    match display_log.filter(|log| !log.events.is_empty()) {
        Some(log) => replay_display_log(repl, &log.events),
        None => replay_conversation(repl, conversation),
    }
}

fn replay_conversation(repl: &mut Repl<'_>, conversation: &Conversation) -> io::Result<()> {
    let tool_names: HashMap<&str, &str> = conversation
        .entries()
        .iter()
        .filter_map(|entry| match &entry.kind {
            EntryKind::Assistant { tool_calls, .. } => tool_calls.as_ref(),
            _ => None,
        })
        .flatten()
        .map(|call| (call.id.as_str(), call.name.as_str()))
        .collect();
    for entry in conversation.entries() {
        match &entry.kind {
            EntryKind::SystemPrompt(_) => {}
            EntryKind::User { content, .. } => repl.replay_user_input(content)?,
            EntryKind::Assistant {
                content, reasoning, ..
            } => {
                if let Some(reasoning) = reasoning {
                    repl.replay_event(&AgentEvent::ReasoningDelta(reasoning.clone()))?;
                }
                repl.replay_event(&AgentEvent::TextDelta(content.clone()))?;
                repl.replay_finish_turn()?;
            }
            EntryKind::Tool { call_id, output } => {
                let name = tool_names.get(call_id.as_str()).copied().unwrap_or("tool");
                repl.replay_event(&AgentEvent::Status(format!("{name} result")))?;
                repl.replay_event(&AgentEvent::TextDelta(output.clone()))?;
                repl.replay_finish_turn()?;
            }
            EntryKind::Developer { content, .. } => {
                repl.replay_event(&AgentEvent::Status(content.clone()))?
            }
        }
    }
    repl.replay_finish_turn()
}

fn save_pasted_images(images: &[PastedImage]) -> Vec<PathBuf> {
    use base64::Engine;
    let dir = PathBuf::from("/tmp/flashmind-images");
    let _ = std::fs::create_dir_all(&dir);
    let mut paths = Vec::with_capacity(images.len());
    for img in images {
        let ext = match img.media_type.as_str() {
            "image/png" => "png",
            "image/gif" => "gif",
            "image/webp" => "webp",
            _ => "jpg",
        };
        let name = format!("{}.{ext}", uuid::Uuid::new_v4().as_simple());
        let path = dir.join(name);
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&img.data)
            && std::fs::write(&path, &bytes).is_ok()
        {
            paths.push(path);
        }
    }
    paths
}

pub fn display_log_path(session_key: &str) -> Result<PathBuf> {
    let dir = crate::paths::display_logs_dir()?;
    crate::paths::migrate_display_logs(&dir)?;
    Ok(dir.join(format!("display-{session_key}.jsonl")))
}

// ---------------------------------------------------------------------------
// Slash command list (for autocomplete)
// ---------------------------------------------------------------------------

pub(crate) fn slash_commands() -> Vec<CommandInfo> {
    vec![
        CommandInfo::new("/model", "Switch model").with_args("[provider:name]"),
        CommandInfo::new("/thinking", "Set reasoning level").with_args("[off|low|medium|high]"),
        CommandInfo::new("/reasoning", "Expand or collapse reasoning blocks")
            .with_args("[show|hide|toggle]"),
        CommandInfo::new("/status", "Show model, context, cost, session info"),
        CommandInfo::new("/context", "Show context window usage breakdown"),
        CommandInfo::new("/new", "Start a new session"),
        CommandInfo::new("/clear", "Clear conversation"),
        CommandInfo::new("/compact", "Compress conversation context"),
        CommandInfo::new("/autocompact", "Set when to compact automatically")
            .with_args("[global|model] [on|off|auto|inherit|<tokens>]"),
        CommandInfo::new("/undo", "Remove last turn"),
        CommandInfo::new("/retry", "Redo last turn, optionally with more context")
            .with_args("[extra context]"),
        CommandInfo::new("/resume", "List and switch sessions"),
        CommandInfo::new("/rename", "Rename current session").with_args("<title>"),
        CommandInfo::new("/system", "View or set system prompt").with_args("[prompt]"),
        CommandInfo::new("/export", "Export conversation to markdown").with_args("[path]"),
        CommandInfo::new("/fork", "Fork current session"),
        CommandInfo::new("/tools", "List tools, or show/hide output")
            .with_args("[expand|collapse]"),
        CommandInfo::new("/skills", "List installed skills"),
        CommandInfo::new("/agents", "List custom agents"),
        CommandInfo::new("/subagents", "Open a subagent conversation (Ctrl+B)"),
        CommandInfo::new("/agents-create", "Create an agent (Tab Tab enriches)")
            .with_args("<name>")
            .with_example("/agents-create helper"),
        CommandInfo::new("/help", "Show this help"),
        CommandInfo::new("/quit", "Exit flsh"),
    ]
}

async fn build_available_commands(
    skill_provider: &Option<Arc<RwLock<DiskSkillProvider>>>,
    agents: &crate::agents::AgentStore,
) -> Vec<CommandInfo> {
    let mut cmds = slash_commands();
    if let Some(provider) = skill_provider {
        let guard = provider.read().await;
        for skill in guard.list() {
            let name = format!("/{}", skill.meta.name);
            if !cmds.iter().any(|c| c.name == name) {
                let description = skill.meta.description.clone().unwrap_or_default();
                cmds.push(CommandInfo::new(name, description));
            }
        }
    }
    for def in agents.list() {
        let name = format!("/{}", def.name);
        if !cmds.iter().any(|c| c.name == name) {
            cmds.push(CommandInfo::new(name, "Custom agent"));
        }
    }
    cmds.sort_by(|a, b| a.name.cmp(&b.name));
    cmds
}

// ---------------------------------------------------------------------------
// Status bar helpers
// ---------------------------------------------------------------------------

/// Detect the current git branch by shelling out to `git`.
/// Returns `None` if not in a git repo or git is unavailable.
fn current_git_branch() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Refresh the status bar with full state (used after turns and model/thinking changes).
fn refresh_status(
    repl: &mut Repl<'_>,
    active: &ModelChoice,
    thinking: ReasoningLevel,
    cost: Decimal,
) {
    repl.set_status(StatusInfo {
        model: active.model.name().to_string(),
        thinking: Some(thinking),
        cost: Some(cost),
        context: active
            .context_window
            .and_then(|cw| repl.last_usage().map(|u| (u.prompt_tokens, cw))),
        git_branch: current_git_branch(),
    });
}

/// Reset the status bar (model + thinking + git branch only) for new/cleared sessions.
fn reset_status(repl: &mut Repl<'_>, model: &str, thinking: ReasoningLevel) {
    repl.set_status(StatusInfo {
        model: model.to_string(),
        thinking: Some(thinking),
        cost: None,
        context: None,
        git_branch: current_git_branch(),
    });
}

/// Output lines of a `!` command kept for the agent, counted from the end.
const SHELL_CONTEXT_LINES: usize = 200;

/// Run a shell-escape command (input starting with `!`) and print its output
/// to the transcript.  Uses the user's shell (`$SHELL`, falling back to `sh`)
/// with `-c`.  Output is streamed line-by-line so long-running commands feel
/// responsive.  A non-zero exit status is reported but doesn't abort the REPL.
///
/// Returns a note with the command and the tail of its output for the
/// conversation, so the agent sees what the user ran. Returns `None` when the
/// shell could not start or the command was interrupted.
async fn run_shell_escape(repl: &mut Repl<'_>, cmdline: &str) -> Result<Option<String>> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
    let mut child = match Command::new(shell)
        .arg("-c")
        .arg(cmdline)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            repl.println(Line::from(Span::styled(
                format!("Cannot run shell: {error}"),
                S_TOOL_FAIL,
            )))?;
            return Ok(None);
        }
    };
    let mut stdout =
        BufReader::new(child.stdout.take().context("shell stdout was not piped")?).lines();
    let mut stderr =
        BufReader::new(child.stderr.take().context("shell stderr was not piped")?).lines();
    let (sender, output) = tokio::sync::mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task = async move {
        let mut stdout_open = true;
        let mut stderr_open = true;
        let mut captured = VecDeque::new();
        let mut keep = |line: &str| {
            if captured.len() == SHELL_CONTEXT_LINES {
                captured.pop_front();
            }
            captured.push_back(line.to_string());
        };
        while stdout_open || stderr_open {
            tokio::select! {
                _ = task_cancel.cancelled() => {
                    child.kill().await?;
                    return Ok::<_, io::Error>(None);
                }
                line = stdout.next_line(), if stdout_open => match line? {
                    Some(line) => { keep(&line); let _ = sender.send(line); }
                    None => stdout_open = false,
                },
                line = stderr.next_line(), if stderr_open => match line? {
                    Some(line) => {
                        let line = format!("stderr: {line}");
                        keep(&line);
                        let _ = sender.send(line);
                    }
                    None => stderr_open = false,
                },
            }
        }
        tokio::select! {
            _ = task_cancel.cancelled() => { child.kill().await?; Ok(None) }
            status = child.wait() => status.map(|status| Some(ShellRun { status, captured })),
        }
    };
    repl.set_activity("shell");
    let status = repl.run_task_with_output(&cancel, task, output).await?;
    repl.clear_activity();
    let note = match status? {
        Some(ShellRun { status, captured }) => {
            if !status.success() {
                repl.println(Line::from(Span::styled(
                    format!("[exit {status}]"),
                    S_TOOL_FAIL,
                )))?;
            }
            Some(shell_note(cmdline, &status.to_string(), captured))
        }
        None => {
            repl.println(Line::from(Span::styled("Shell command interrupted", S_DIM)))?;
            None
        }
    };
    repl.println(Line::default())?;
    Ok(note)
}

/// A finished `!` command: its exit status and the tail of its output.
struct ShellRun {
    status: ExitStatus,
    captured: VecDeque<String>,
}

/// The conversation note for a `!` command the user ran.
fn shell_note(cmdline: &str, status: &str, captured: VecDeque<String>) -> String {
    let mut note = format!(
        "The user ran a shell command themselves ({status}). Its output is shown for context; it is not a request.\n\n$ {cmdline}\n"
    );
    if captured.len() == SHELL_CONTEXT_LINES {
        note.push_str(&format!(
            "[earlier output omitted, last {SHELL_CONTEXT_LINES} lines follow]\n"
        ));
    }
    for line in captured {
        note.push_str(&line);
        note.push('\n');
    }
    note
}

// ---------------------------------------------------------------------------
// One-shot mode
// ---------------------------------------------------------------------------

pub async fn run_oneshot(
    agent: &mut Agent,
    conversation: &mut Conversation,
    prompt: String,
    json: bool,
    pricing: &ModelPricing,
) -> Result<SessionUsage> {
    let mut usage = SessionUsage::default();
    let cancel = CancellationToken::new();
    // Resolve @mentions in the one-shot prompt: attach file contents as
    // system-level context injected before the user message.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mentions = crate::mention::extract_mentions(&prompt);
    let mut context = crate::mention::build_context_block(&cwd, &mentions);
    let searches = crate::mention::extract_content_searches(&prompt);
    if let Some(search_ctx) = crate::mention::build_search_context_block(&cwd, &searches) {
        context = Some(match context {
            Some(c) => format!("{c}\n{search_ctx}"),
            None => search_ctx,
        });
    }
    let input = AgentInput::User {
        content: prompt,
        context,
        parts: None,
    };
    let stream = agent.start(conversation, cancel, input, None);
    tokio::pin!(stream);

    while let Some(event) = stream.next().await {
        if let Some(request_usage) = event_usage(&event) {
            usage.add(request_usage, pricing);
        }
        if json {
            println!("{}", serde_json::to_string(&event)?);
            match event {
                AgentEvent::Error(_) => std::process::exit(1),
                AgentEvent::Done(_) => break,
                _ => continue,
            }
        }
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {e}");
                std::process::exit(1);
            }
            AgentEvent::Done(_) => {
                println!();
                break;
            }
            _ => {}
        }
    }
    eprintln!("{}", UsageSummary(&usage));
    Ok(usage)
}

// ---------------------------------------------------------------------------
// Resume mode
// ---------------------------------------------------------------------------

pub async fn run_resume(
    cli: &crate::Cli,
    config: &Config,
    all: bool,
    session_id: Option<&SessionId>,
) -> Result<()> {
    // Always load every directory's sessions; tabs scope the view in-UI.
    let store_sessions: Vec<(SessionStore, Vec<SessionSummary>)> =
        crate::session::list_all_sessions().await?;

    let mut flat = flatten_sessions(&store_sessions);
    let selected_index = if let Some(session_id) = session_id {
        flat.iter()
            .position(|entry| entry.summary.chat_key == session_id.as_str())
            .with_context(|| format!("Session not found: {session_id}"))?
    } else {
        if flat.is_empty() {
            println!("No sessions to resume.");
            return Ok(());
        }

        let mut tui = Tui::new();
        let mut picker = session_picker(&flat, "Select a session to resume:", None, all);
        loop {
            match run_choice_action(&mut tui, &mut picker)? {
                ChoicePickerAction::Select(r) => break r.selected,
                ChoicePickerAction::Cancel | ChoicePickerAction::Quit => return Ok(()),
                ChoicePickerAction::Delete(idx) => {
                    let entry = &flat[idx];
                    let store = &store_sessions[entry.store_idx].0;
                    store.delete_session(&entry.summary.chat_key).await?;
                    flat.remove(idx);
                    picker.remove(idx);
                    if flat.is_empty() {
                        println!("No sessions to resume.");
                        return Ok(());
                    }
                }
            }
        }
    };

    let selected = &flat[selected_index];
    let session = &selected.summary;
    let store = &store_sessions[selected.store_idx].0;
    let chat_key = session.chat_key.clone();

    // If the session was from a different directory, chdir to it.
    if let Some(ref wd) = session.working_dir {
        let target = Path::new(wd);
        if target.is_dir() {
            std::env::set_current_dir(target)?;
        } else {
            anyhow::bail!("session working directory no longer exists: {wd}");
        }
    }

    // Restore model from session metadata, fall back to CLI/config default
    let model: Model = session
        .model
        .as_deref()
        .and_then(|m| m.parse().ok())
        .unwrap_or_else(|| {
            crate::provider::resolve_model(cli, config)
                .unwrap_or_else(|_| "ollama:llama3.2".parse().unwrap())
        });
    let crate::Startup {
        mut agent,
        subagents,
        tool_sync,
        skill_provider,
        system_prompt,
        reasoning,
        pricing,
        context_window,
        always_skills,
    } = crate::start_session(cli, config, &model, &chat_key).await?;

    let mut conversation = load_conversation_from(store, &chat_key, &system_prompt).await?;

    // Restore running cost and last token usage from session metadata so the
    // status bar and cost accumulator pick up where the previous session left
    // off.
    let usage = summary_totals(session);
    let last_usage = summary_usage(session);

    run_interactive(
        &mut agent,
        &mut conversation,
        store,
        config,
        SessionState {
            subagents,
            session_key: chat_key.clone(),
            model: model.clone(),
            reasoning,
            pricing,
            context_window,
            system_prompt,
            skip_banner: false,
            display_log_path: Some(display_log_path(&chat_key)?),
            skill_provider: Some(skill_provider),
            usage,
            last_usage,
            always_skills,
        },
        &tool_sync,
    )
    .await?;
    tool_sync.shutdown().await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive REPL
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

/// The active model with what the session needs to know about it.
pub(crate) struct ModelChoice {
    pub(crate) model: Model,
    pub(crate) pricing: ModelPricing,
    pub(crate) context_window: Option<u32>,
}

/// Parse a `provider:name` model string, swap the agent's provider/model, and
/// return the new model with its pricing and context window. Shared by the
/// `/model` command and agent switching.
pub(crate) async fn apply_model(
    agent: &mut Agent,
    model_str: &str,
    config: &Config,
) -> Result<ModelChoice> {
    let model: Model = model_str
        .parse()
        .with_context(|| format!("invalid model format: {model_str}"))?;
    let provider = build_provider(&model.provider, config)?;
    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = provider.context_window(&model).await;

    agent.set_provider(provider);
    agent.llm_mut().model = model.clone();
    agent.refresh_features().await;
    crate::autocompact::apply_settings(agent, config);

    Ok(ModelChoice {
        model,
        pricing,
        context_window,
    })
}

async fn handle_model_command(
    args: &str,
    agent: &mut Agent,
    config: &Config,
    tui: &mut Tui,
) -> Result<Option<ModelChoice>> {
    let input = args.trim();

    if !input.is_empty() {
        let choice = apply_model(agent, input, config).await?;
        tui.println(&Line::from(Span::styled(
            format!(
                "  Switched to {} {}",
                choice.model.name(),
                ModelPriceSummary(&choice.pricing)
            ),
            S_AGENT,
        )))?;
        return Ok(Some(choice));
    }

    let mut providers = configured_providers(config);
    if providers.is_empty() {
        tui.println(&Line::from(
            "  No providers configured. Run `flsh setup` first.",
        ))?;
        return Ok(None);
    }

    let chosen_provider = if providers.len() == 1 {
        providers[0].1.clone()
    } else {
        let current = agent.llm().model.provider.clone();
        // Put the current provider first.
        providers.sort_by_key(|(_, p)| if *p == current { 0 } else { 1 });
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
            Some(resp) => providers[resp.selected].1.clone(),
            None => return Ok(None),
        }
    };

    let tmp_provider = build_provider(&chosen_provider, config)?;

    tui.println(&Line::from(format!(
        "  Fetching models from {}...",
        tmp_provider.name()
    )))?;
    let models = tmp_provider
        .list_models()
        .await
        .context("failed to list models")?;
    if models.is_empty() {
        tui.println(&Line::from("  No models found."))?;
        return Ok(None);
    }

    let model_idx = match crate::setup::run_model_picker(tui, &models)? {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let chosen = &models[model_idx];
    // `configured_providers` only offers built-in providers, whose display
    // form is the model-string prefix.
    let model: Model = format!("{chosen_provider}:{}", chosen.id).parse()?;
    let pricing = fetch_pricing(&tmp_provider, &model).await;
    let context_window = tmp_provider.context_window(&model).await;

    tui.println(&Line::from(Span::styled(
        format!(
            "  Switched to {} {}",
            model.name(),
            ModelPriceSummary(&pricing)
        ),
        S_AGENT,
    )))?;

    agent.set_provider(tmp_provider);
    agent.llm_mut().model = model.clone();
    agent.refresh_features().await;
    crate::autocompact::apply_settings(agent, config);

    Ok(Some(ModelChoice {
        model,
        pricing,
        context_window,
    }))
}

pub struct SessionState {
    pub subagents: Arc<AgentManager>,
    pub session_key: String,
    pub model: Model,
    pub reasoning: ReasoningLevel,
    pub pricing: ModelPricing,
    pub context_window: Option<u32>,
    pub system_prompt: String,
    pub skip_banner: bool,
    pub display_log_path: Option<PathBuf>,
    pub skill_provider: Option<Arc<RwLock<DiskSkillProvider>>>,
    /// Cumulative usage restored on resume.
    pub usage: SessionUsage,
    /// Last token usage to restore (status bar + context display).
    pub last_usage: Option<TokenUsage>,
    pub always_skills: AlwaysOnSkills,
}

pub async fn run_interactive(
    agent: &mut Agent,
    conversation: &mut Conversation,
    store: &SessionStore,
    config: &Config,
    state: SessionState,
    tool_sync: &flashmind_tools::tool_sync::ToolSync,
) -> Result<()> {
    use flashmind_tui::{ReplConfig, ReplEvent};

    // Shadow with an owned clone so /resume can swap the active store when
    // switching to a session from a different working directory.
    let mut store: SessionStore = store.clone();
    let store = &mut store;

    let mut session_config = config.clone();
    let config = &mut session_config;

    let mut skill_context = if let Some(provider) = &state.skill_provider {
        let provider = provider.read().await;
        crate::tools::skill_instructions(&crate::tools::skill_index(&*provider))
    } else {
        String::new()
    };
    skill_context.push_str(&state.always_skills.prompt);
    let terminal_title = TerminalTitle::new();
    let sessions = store.list_sessions().await?;
    let initial_task = sessions
        .iter()
        .find(|session| session.chat_key == state.session_key)
        .and_then(|session| session.title.as_deref())
        .or_else(|| {
            conversation
                .entries()
                .iter()
                .rev()
                .find(|entry| entry.is_user())
                .map(|entry| entry.content())
        })
        .unwrap_or("");
    terminal_title.set(initial_task);
    let mut tui = Tui::new();

    let agent_store = crate::agents::AgentStore::open()?;

    let history_file = crate::paths::state_dir().ok().map(|d| d.join("history"));
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mention_provider = Arc::new(crate::mention::CwdMentionProvider::new(cwd.clone()))
        as Arc<dyn flashmind_tui::MentionProvider>;
    let providers = configured_providers(config).into_iter().map(|(_, p)| p);
    let model_completion = crate::model_completion::ModelCompletion::spawn(providers, config);
    let repl_config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: None,
        history_file,
        available_commands: build_available_commands(&state.skill_provider, &agent_store).await,
        mention_provider: Some(mention_provider),
        argument_provider: Some(model_completion.clone()),
        ..Default::default()
    };

    let mut repl = Repl::new(repl_config).with_tui(&tui);
    repl.set_subagent_source(Arc::new(crate::subagents::SubagentConversations::new(
        state.subagents.clone(),
    )));
    repl.enable_subagent_progress();
    repl.set_expand_reasoning(false)?;
    // Enable the pet companion from config: a name, "random"/unset → random,
    // "off"/"none" → disabled.
    match config.pet.as_deref() {
        Some(v) => {
            let v = v.trim();
            if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("none") {
                // disabled
            } else if v.eq_ignore_ascii_case("random") {
                repl.enable_pet(flashmind_tui::random_pet());
            } else if let Some(kind) = flashmind_tui::Pet::from_name(v) {
                repl.enable_pet(kind);
            } else {
                tracing::warn!(pet = %v, "unknown pet name in config; assigning random");
                repl.enable_pet(flashmind_tui::random_pet());
            }
        }
        None => repl.enable_pet(flashmind_tui::random_pet()),
    }
    if let Ok((w, _)) = ratatui::crossterm::terminal::size() {
        repl.set_renderer_width(w.saturating_sub(1) as usize);
    }
    let mut session_key = state.session_key;
    let mut active = ModelChoice {
        model: state.model,
        pricing: state.pricing,
        context_window: state.context_window,
    };
    let mut current_reasoning = state.reasoning;
    let mut system_prompt = state.system_prompt;
    let mut title_generated = false;
    let mut turn_count: usize = 0;

    // Default agent state, captured so `/agents reset` can restore it after a
    // custom agent has been switched in.
    let default_system_prompt = system_prompt.clone();
    let default_model_str = active.model.to_string();
    let default_reasoning = current_reasoning;

    // Replay display log if restoring a session
    let mut display_log = state.display_log_path.map(DisplayLog::open);
    replay_session(&mut repl, display_log.as_ref(), conversation)?;
    if !state.skip_banner {
        let tool_names: Vec<&str> = agent.tools().list();
        print_banner(
            &mut tui,
            active.model.name(),
            current_reasoning,
            &tool_names,
            &state.always_skills,
        )?;
    }

    // Restore running cost and last token usage from the session state (resume).
    let mut usage = state.usage;
    if let Some(ref u) = state.last_usage {
        repl.set_usage(u.prompt_tokens, u.completion_tokens);
    }
    refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);

    let mut notifications = SubagentNotifications::new(state.subagents);
    loop {
        let mut completed = Vec::new();
        let input = repl.read_input_until(|repl| {
            for event in agent.drain_background_events() {
                if let Some(request_usage) = event_usage(&event) {
                    let pricing =
                        event_pricing(&event, &active.model, &active.pricing, &model_completion);
                    accrue_cost(repl, &mut usage, &pricing, request_usage.clone());
                }
                emit_logged(repl, &mut display_log, &event)?;
            }
            completed = notifications.take_completed();
            Ok(!completed.is_empty())
        })?;
        let background = input.is_none();
        let (mut text, pasted_images) = match input {
            Some(ReplEvent::UserInput(text, images)) => (text, images),
            Some(ReplEvent::Quit) => break,
            None => (String::new(), Vec::new()),
        };
        for message in completed {
            conversation.add(message);
        }
        let mut invoked_skill = None;
        // Shell escape: lines starting with `!` run as a shell command. The
        // command and its output go into the conversation for the next turn
        // without starting one.
        if let Some(cmdline) = text.strip_prefix('!') {
            let cmdline = cmdline.trim();
            if cmdline.is_empty() {
                continue;
            }
            if let Some(note) = run_shell_escape(&mut repl, cmdline).await? {
                conversation.add(ConversationEntry::system_message(note));
            }
            continue;
        }
        // Slash command dispatch
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
            match cmd {
                "subagents" => {
                    repl.open_subagents()?;
                    continue;
                }
                "agents" => {
                    let trimmed = args.trim();
                    let (sub, rest) = trimmed.split_once(' ').unwrap_or((trimmed, ""));
                    let rest = rest.trim();
                    match sub {
                        "" | "list" => {
                            let defs = agent_store.list();
                            if defs.is_empty() {
                                repl.println(Line::from(
                                    "  No agents yet. Create one with /agents-create <name>",
                                ))?;
                            } else {
                                repl.println(Line::default())?;
                                for d in defs {
                                    let model = d.model.as_deref().unwrap_or("(current model)");
                                    repl.println(Line::from(vec![
                                        Span::styled(format!("  /{}", d.name), S_AGENT),
                                        Span::styled(format!("  {model}"), S_DIM),
                                    ]))?;
                                    if let Some(first) = d.system_prompt.lines().next() {
                                        let snippet: String = first.chars().take(72).collect();
                                        repl.println(Line::from(Span::styled(
                                            format!("      {snippet}"),
                                            S_DIM,
                                        )))?;
                                    }
                                }
                                repl.println(Line::default())?;
                            }
                        }
                        "create" => {
                            let name = (!rest.is_empty()).then_some(rest);
                            if crate::agents::create_agent_flow(
                                &mut tui,
                                agent,
                                &agent_store,
                                name,
                                None,
                            )
                            .await?
                            .is_some()
                            {
                                repl.set_available_commands(
                                    build_available_commands(&state.skill_provider, &agent_store)
                                        .await,
                                );
                            }
                        }
                        "delete" | "rm" => {
                            if rest.is_empty() {
                                repl.println(Line::from("  Usage: /agents delete <name>"))?;
                            } else if agent_store.delete(rest)? {
                                repl.println(Line::from(Span::styled(
                                    format!("  Deleted agent '{rest}'"),
                                    S_AGENT,
                                )))?;
                                repl.set_available_commands(
                                    build_available_commands(&state.skill_provider, &agent_store)
                                        .await,
                                );
                            } else {
                                repl.println(Line::from(format!("  No agent named '{rest}'")))?;
                            }
                        }
                        "reset" => {
                            conversation.set_system(&default_system_prompt);
                            system_prompt = default_system_prompt.clone();
                            active = apply_model(agent, &default_model_str, config).await?;
                            agent.llm_mut().reasoning = default_reasoning;
                            current_reasoning = default_reasoning;
                            refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                            repl.println(Line::from(Span::styled(
                                "  Reset to default agent",
                                S_AGENT,
                            )))?;
                        }
                        _ => {
                            repl.println(Line::from(
                                "  Usage: /agents [list|create <name>|delete <name>|reset]",
                            ))?;
                        }
                    }
                    continue;
                }
                "agents-create" => {
                    let trimmed = args.trim();
                    let (name, draft) = trimmed.split_once(' ').unwrap_or((trimmed, ""));
                    let name = (!name.is_empty()).then_some(name);
                    let draft = (!draft.trim().is_empty()).then(|| draft.trim());
                    if crate::agents::create_agent_flow(&mut tui, agent, &agent_store, name, draft)
                        .await?
                        .is_some()
                    {
                        repl.set_available_commands(
                            build_available_commands(&state.skill_provider, &agent_store).await,
                        );
                    }
                    continue;
                }
                "model" => {
                    if let Some(choice) =
                        handle_model_command(args, agent, config, &mut tui).await?
                    {
                        active = choice;
                        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                    }
                    continue;
                }
                "thinking" => {
                    let input = args.trim();
                    let level = if input.is_empty() {
                        let options = plain_options(&["Off", "Low", "Medium", "High"]);
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
                            "medium" | "med" | "on" => Some(ReasoningLevel::Medium),
                            "high" => Some(ReasoningLevel::High),
                            _ => {
                                repl.println(Line::from(
                                    "  Usage: /thinking [off|low|medium|high]",
                                ))?;
                                None
                            }
                        }
                    };
                    if let Some(level) = level {
                        agent.llm_mut().reasoning = level;
                        current_reasoning = level;
                        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                        repl.println(Line::from(Span::styled(
                            format!("  Thinking: {level}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "reasoning" => {
                    // Toggle or set reasoning-block display: expanded (full
                    // text) vs collapsed (one-line summary).
                    let input = args.trim();
                    let new_val = match input {
                        "" | "toggle" => Some(!repl.expand_reasoning()),
                        "show" | "expand" | "on" => Some(true),
                        "hide" | "collapse" | "off" => Some(false),
                        _ => {
                            repl.println(Line::from("  Usage: /reasoning [show|hide|toggle]"))?;
                            None
                        }
                    };
                    if let Some(v) = new_val {
                        repl.set_expand_reasoning(v)?;
                        repl.println(Line::from(Span::styled(
                            format!(
                                "  Reasoning display: {}",
                                if v { "expanded" } else { "collapsed" }
                            ),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "tools" => {
                    let input = args.trim();
                    match input {
                        "" => {
                            let names = agent.tools().list();
                            repl.println(Line::from(Span::styled(
                                format!("  {} tools registered:", names.len()),
                                S_AGENT,
                            )))?;
                            for name in &names {
                                repl.println(Line::from(format!("    {name}")))?;
                            }
                        }
                        "expand" | "show" | "on" => {
                            repl.set_expand_tools(true)?;
                            repl.println(Line::from(Span::styled(
                                "  Tool output: expanded",
                                S_AGENT,
                            )))?;
                        }
                        "collapse" | "hide" | "off" => {
                            repl.set_expand_tools(false)?;
                            repl.println(Line::from(Span::styled(
                                "  Tool output: collapsed",
                                S_AGENT,
                            )))?;
                        }
                        _ => {
                            repl.println(Line::from("  Usage: /tools [expand|collapse]"))?;
                        }
                    }
                    continue;
                }
                "new" => {
                    persist_cost_usage(store, &session_key, &usage, &repl).await?;
                    repl.clear_transcript()?;
                    if turn_count > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    session_key = new_session_key();
                    agent.llm_mut().session_id = Some(session_key.parse()?);
                    *conversation = Conversation::with_system(&system_prompt);
                    display_log = Some(DisplayLog::new(display_log_path(&session_key)?));
                    title_generated = false;
                    terminal_title.set("");
                    turn_count = 0;
                    usage = SessionUsage::default();
                    reset_status(&mut repl, active.model.name(), current_reasoning);
                    repl.println(Line::from(Span::styled("  New session started", S_AGENT)))?;
                    continue;
                }
                "clear" => {
                    repl.clear_transcript()?;
                    *conversation = Conversation::with_system(&system_prompt);
                    terminal_title.set("");
                    turn_count = 0;
                    usage = SessionUsage::default();
                    if let Some(ref mut log) = display_log {
                        log.clear();
                        save_display_log(log, &mut repl)?;
                    }
                    save_turn(store, &session_key, conversation).await?;
                    reset_status(&mut repl, active.model.name(), current_reasoning);
                    persist_cost_usage(store, &session_key, &usage, &repl).await?;
                    repl.println(Line::from(Span::styled("  Conversation cleared", S_AGENT)))?;
                    continue;
                }
                "autocompact" => {
                    match AutoCompactCommand::parse(args) {
                        Some(command) => {
                            command.apply(config, &active.model);
                            crate::autocompact::apply_settings(agent, config);
                            repl.println(Line::from(command.describe(config, &active.model)))?;
                            repl.println(Line::from(Span::styled(
                                crate::autocompact::describe(agent, config),
                                S_AGENT,
                            )))?;
                            if command.is_change()
                                && let Err(error) = crate::config::update_config(|doc| {
                                    command.save(doc, config, &active.model)
                                })
                            {
                                repl.println(Line::from(format!(
                                    "  Could not save to config: {error}"
                                )))?;
                            }
                        }
                        None => repl.println(Line::from(crate::autocompact::USAGE))?,
                    }
                    continue;
                }
                "compact" => {
                    {
                        let stream = flashmind_core::compaction::compact_manually(
                            conversation,
                            agent.provider(),
                            &agent.llm().model,
                        );
                        let tapped = log_stream(stream, &mut display_log);
                        let cancel = CancellationToken::new();
                        tokio::pin!(tapped);
                        repl.stream_events(&cancel, tapped).await?;
                    }
                    save_turn(store, &session_key, conversation).await?;
                    if let Some(log) = display_log.as_mut() {
                        save_display_log(log, &mut repl)?;
                    }

                    if steer(&mut repl, conversation, &mut display_log)? {
                        conversation.mark_turn_start();
                        run_turn_loop(
                            agent,
                            conversation,
                            &mut repl,
                            UsageAccounting {
                                total: &mut usage,
                                pricing: &active.pricing,
                                model: &active.model,
                                catalog: &model_completion,
                                store,
                                session_key: &session_key,
                            },
                            &mut display_log,
                            &mut notifications,
                        )
                        .await?;
                        tool_sync.sync(agent.tools_mut()).await;
                        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                        save_turn(store, &session_key, conversation).await?;
                    }
                    continue;
                }
                "undo" => {
                    let removed = pop_last_turn(conversation);
                    if removed > 0 {
                        turn_count = turn_count.saturating_sub(1);
                        save_turn(store, &session_key, conversation).await?;
                        if let Some(log) = display_log.as_mut() {
                            log.pop_last_turn();
                            save_display_log(log, &mut repl)?;
                        }
                        repl.clear_transcript()?;
                        replay_session(&mut repl, display_log.as_ref(), conversation)?;
                        repl.println(Line::from(Span::styled(
                            format!("  Removed last turn ({removed} entries)"),
                            S_AGENT,
                        )))?;
                    } else {
                        repl.println(Line::from(Span::styled("  Nothing to undo", S_DIM)))?;
                    }
                    continue;
                }
                "retry" => {
                    let extra = args.trim();
                    // Find the last user message before popping
                    let last_user_msg = conversation
                        .entries()
                        .iter()
                        .rev()
                        .find(|e| e.is_user())
                        .map(|e| e.content().to_string());
                    pop_last_turn(conversation);
                    if let Some(original) = last_user_msg {
                        let retry_msg = if extra.is_empty() {
                            original
                        } else {
                            format!("{original}\n\n{extra}")
                        };
                        conversation.add(ConversationEntry::user(&retry_msg));
                        conversation.mark_turn_start();
                        run_turn_loop(
                            agent,
                            conversation,
                            &mut repl,
                            UsageAccounting {
                                total: &mut usage,
                                pricing: &active.pricing,
                                model: &active.model,
                                catalog: &model_completion,
                                store,
                                session_key: &session_key,
                            },
                            &mut display_log,
                            &mut notifications,
                        )
                        .await?;
                        tool_sync.sync(agent.tools_mut()).await;
                        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                        save_turn(store, &session_key, conversation).await?;
                    } else {
                        repl.println(Line::from(Span::styled("  Nothing to retry", S_DIM)))?;
                    }
                    continue;
                }
                "sessions" | "resume" => {
                    persist_cost_usage(store, &session_key, &usage, &repl).await?;
                    if turn_count > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    // Load every directory's sessions so /resume can switch
                    // across working directories; tabs scope the view in-UI.
                    let store_sessions = crate::session::list_all_sessions().await?;
                    let mut flat = flatten_sessions(&store_sessions);
                    if flat.is_empty() {
                        repl.println(Line::from("  No sessions saved."))?;
                        continue;
                    }
                    let mut picker =
                        session_picker(&flat, "Switch session:", Some(&session_key), false);
                    let selected = loop {
                        match run_choice_action(&mut tui, &mut picker)? {
                            ChoicePickerAction::Select(r) => break Some(r),
                            ChoicePickerAction::Cancel | ChoicePickerAction::Quit => {
                                break None;
                            }
                            ChoicePickerAction::Delete(idx) => {
                                let entry = &flat[idx];
                                if entry.summary.chat_key == session_key {
                                    continue;
                                }
                                store_sessions[entry.store_idx]
                                    .0
                                    .delete_session(&entry.summary.chat_key)
                                    .await?;
                                flat.remove(idx);
                                picker.remove(idx);
                                if flat.is_empty() {
                                    break None;
                                }
                            }
                        }
                    };
                    if let Some(resp) = selected {
                        let chosen = &flat[resp.selected];
                        if chosen.summary.chat_key != session_key {
                            // If the session lives in another directory, chdir
                            // there first so relative tool paths line up.
                            if let Some(ref wd) = chosen.summary.working_dir {
                                let target = Path::new(wd);
                                if target != std::env::current_dir()?.as_path() {
                                    if target.is_dir() {
                                        std::env::set_current_dir(target)?;
                                    } else {
                                        repl.println(Line::from(format!(
                                            "  Session directory no longer exists: {wd}"
                                        )))?;
                                        continue;
                                    }
                                }
                            }
                            *store = store_sessions[chosen.store_idx].0.clone();
                            session_key = chosen.summary.chat_key.clone();
                            agent.llm_mut().session_id = Some(session_key.parse()?);
                            *conversation =
                                load_conversation_from(store, &session_key, &system_prompt).await?;
                            title_generated = chosen.summary.title.is_some();
                            terminal_title.set(&session_display_name(&chosen.summary));
                            turn_count = conversation
                                .entries()
                                .iter()
                                .filter(|e| e.is_user())
                                .count();
                            // Restore the running cost and last token usage from
                            // the chosen session so the status bar continues
                            // where that session left off (parity with startup
                            // `flashmind resume`).
                            usage = summary_totals(&chosen.summary);
                            if let Some(u) = summary_usage(&chosen.summary) {
                                repl.set_usage(u.prompt_tokens, u.completion_tokens);
                            }
                            refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                            display_log = Some(DisplayLog::open(display_log_path(&session_key)?));
                            repl.clear_transcript()?;
                            replay_session(&mut repl, display_log.as_ref(), conversation)?;
                            let title = chosen
                                .summary
                                .title
                                .as_deref()
                                .unwrap_or(truncate_utf8(&session_key, 20));
                            repl.println(Line::from(Span::styled(
                                format!("  Switched to: {title}"),
                                S_AGENT,
                            )))?;
                            for line in always_skill_lines(&state.always_skills) {
                                repl.println(line)?;
                            }
                        }
                    }
                    continue;
                }
                "rename" => {
                    let new_title = args.trim();
                    if new_title.is_empty() {
                        repl.println(Line::from("  Usage: /rename <new title>"))?;
                    } else {
                        store.update_title(&session_key, new_title).await?;
                        title_generated = true;
                        terminal_title.set(new_title);
                        repl.println(Line::from(Span::styled(
                            format!("  Session renamed to: {new_title}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "system" => {
                    let input = args.trim();
                    if input.is_empty() {
                        let current = conversation
                            .entries()
                            .first()
                            .filter(|e| e.is_system())
                            .map(|e| e.content())
                            .unwrap_or("(none)");
                        repl.println(Line::default())?;
                        for line in current.lines() {
                            repl.println(Line::from(format!("  {line}")))?;
                        }
                        repl.println(Line::default())?;
                    } else {
                        system_prompt = format!("{input}{skill_context}");
                        conversation.set_system(&system_prompt);
                        repl.println(Line::from(Span::styled("  System prompt updated", S_AGENT)))?;
                    }
                    continue;
                }
                "export" => {
                    let path = if args.trim().is_empty() {
                        let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
                        format!("flsh-export-{ts}.md")
                    } else {
                        args.trim().to_string()
                    };
                    let mut output = String::new();
                    for entry in conversation.entries() {
                        match &entry.kind {
                            EntryKind::SystemPrompt(s) => {
                                output.push_str("## System Prompt\n\n");
                                output.push_str(s);
                                output.push_str("\n\n---\n\n");
                            }
                            EntryKind::User { content, .. } => {
                                output.push_str("## User\n\n");
                                output.push_str(content);
                                output.push_str("\n\n");
                            }
                            EntryKind::Assistant { content, .. } => {
                                output.push_str("## Assistant\n\n");
                                output.push_str(content);
                                output.push_str("\n\n");
                            }
                            EntryKind::Tool { call_id, output: o } => {
                                output.push_str(&format!(
                                    "### Tool Result ({call_id})\n\n```\n{o}\n```\n\n"
                                ));
                            }
                            EntryKind::Developer { content, tag, .. } => {
                                let tag_str = tag.as_deref().unwrap_or("developer");
                                output.push_str(&format!("### {tag_str}\n\n{content}\n\n"));
                            }
                        }
                    }
                    if let Err(e) = std::fs::write(&path, &output) {
                        repl.println(Line::from(Span::styled(
                            format!("  Export failed: {e}"),
                            S_TOOL_FAIL,
                        )))?;
                    } else {
                        repl.println(Line::from(Span::styled(
                            format!("  Exported to {path}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "fork" => {
                    save_turn(store, &session_key, conversation).await?;
                    let new_key = new_session_key();
                    let count = store.branch(&session_key, &new_key).await?;
                    // Copy metadata with a "(fork)" suffix
                    let sessions = store.list_sessions().await?;
                    let old_title = sessions
                        .iter()
                        .find(|s| s.chat_key == session_key)
                        .and_then(|s| s.title.as_deref())
                        .unwrap_or("untitled");
                    let fork_title = format!("{old_title} (fork)");
                    let cwd = std::env::current_dir()
                        .ok()
                        .and_then(|p| p.to_str().map(String::from));
                    store
                        .save_meta(
                            &new_key,
                            Some(&fork_title),
                            Some(&active.model.to_string()),
                            cwd.as_deref(),
                        )
                        .await?;
                    session_key = new_key;
                    agent.llm_mut().session_id = Some(session_key.parse()?);
                    title_generated = true;
                    terminal_title.set(&fork_title);
                    repl.println(Line::from(Span::styled(
                        format!("  Forked session ({count} entries) — {fork_title}"),
                        S_AGENT,
                    )))?;
                    continue;
                }
                "skills" => {
                    use ratatui::style::{Color, Style};
                    repl.println(Line::default())?;
                    match &state.skill_provider {
                        None => {
                            repl.println(Line::from(Span::styled(
                                "  Skills not available in this session.",
                                Style::default().fg(Color::Yellow),
                            )))?;
                        }
                        Some(provider) => {
                            let guard = provider.read().await;
                            let skills = guard.list();
                            if skills.is_empty() {
                                let dir = crate::paths::config_dir()
                                    .map(|d| d.join("skills").display().to_string())
                                    .unwrap_or_else(|_| "the skills directory".into());
                                repl.println(Line::from(format!(
                                    "  No skills installed. Add a SKILL.md in {dir}/.",
                                )))?;
                            } else {
                                repl.println(Line::from(Span::styled(
                                    format!("  {} skill(s):", skills.len()),
                                    Style::default().fg(Color::Cyan),
                                )))?;
                                for s in &skills {
                                    let name = &s.meta.name;
                                    let desc =
                                        s.meta.description.as_deref().unwrap_or("(no description)");
                                    repl.println(Line::from(vec![
                                        Span::styled(
                                            format!("  /{name:<23} "),
                                            Style::default().fg(Color::Green),
                                        ),
                                        Span::raw(desc.to_string()),
                                    ]))?;
                                }
                            }
                        }
                    }
                    repl.println(Line::default())?;
                    continue;
                }
                "status" => {
                    repl.println(Line::default())?;
                    let mut rows: Vec<(&str, String)> = Vec::new();
                    rows.push(("model", active.model.name().to_string()));
                    rows.push(("thinking", format!("{current_reasoning}")));
                    if let Some(cw) = active.context_window {
                        let used = repl.last_usage().map(|u| u.prompt_tokens).unwrap_or(0);
                        let pct = context_pct(used, cw);
                        rows.push(("context", format!("{used}/{cw} ({pct}%)")));
                    } else {
                        rows.push(("context", "unknown".to_string()));
                    }
                    rows.push(("cost", format!("${}", usage.total_cost)));
                    rows.push(("session", session_key.clone()));
                    rows.push(("turns", format!("{turn_count}")));
                    if let Some(branch) = current_git_branch() {
                        rows.push(("branch", branch));
                    }
                    for (k, v) in &rows {
                        repl.println(Line::from(vec![
                            Span::styled(format!("  {k:<10}"), S_DIM),
                            Span::raw(v.clone()),
                        ]))?;
                    }
                    repl.println(Line::default())?;
                    continue;
                }
                "context" => {
                    repl.println(Line::default())?;
                    let entries = conversation.entries();
                    let total = entries.len();
                    let mut system = 0usize;
                    let mut user = 0usize;
                    let mut assistant = 0usize;
                    let mut tool = 0usize;
                    let mut developer = 0usize;
                    for e in entries {
                        match &e.kind {
                            EntryKind::SystemPrompt(_) => system += 1,
                            EntryKind::User { .. } => user += 1,
                            EntryKind::Assistant { .. } => assistant += 1,
                            EntryKind::Tool { .. } => tool += 1,
                            EntryKind::Developer { .. } => developer += 1,
                        }
                    }
                    let rows: Vec<(&str, String)> = vec![
                        ("entries", format!("{total}")),
                        ("  system", format!("{system}")),
                        ("  user", format!("{user}")),
                        ("  assistant", format!("{assistant}")),
                        ("  tool", format!("{tool}")),
                        ("  developer", format!("{developer}")),
                    ];
                    for (k, v) in &rows {
                        repl.println(Line::from(vec![
                            Span::styled(format!("  {k:<12}"), S_DIM),
                            Span::raw(v.clone()),
                        ]))?;
                    }
                    if let Some(u) = repl.last_usage() {
                        repl.println(Line::from(vec![
                            Span::styled(format!("  {:<12}", "last tokens"), S_DIM),
                            Span::raw(format!(
                                "{}↑ {}↓ {}",
                                u.prompt_tokens, u.completion_tokens, u.total_tokens
                            )),
                        ]))?;
                    }
                    if let Some(cw) = active.context_window {
                        let used = repl.last_usage().map(|u| u.prompt_tokens).unwrap_or(0);
                        let pct = context_pct(used, cw);
                        repl.println(Line::from(vec![
                            Span::styled(format!("  {:<12}", "window"), S_DIM),
                            Span::raw(format!("{cw} ({pct}% used)")),
                        ]))?;
                    }
                    repl.println(Line::default())?;
                    continue;
                }
                "quit" => break,
                "help" => {
                    repl.println(Line::default())?;
                    let commands = slash_commands();
                    let mut cmds: Vec<(String, &str)> = commands
                        .iter()
                        .map(|c| {
                            let usage = if c.args_hint.is_empty() {
                                c.name.clone()
                            } else {
                                format!("{} {}", c.name, c.args_hint)
                            };
                            (usage, c.description.as_str())
                        })
                        .collect();
                    cmds.extend(
                        [
                            ("/<skill> [args]", "Invoke an installed skill"),
                            ("/<agent> [message]", "Switch to a custom agent"),
                            ("", ""),
                            ("@path", "Mention a file; contents attached as context"),
                            ("!cmd", "Run a shell command; output to transcript"),
                            ("Ctrl+B", "Open a subagent conversation; Esc returns"),
                            ("Ctrl+V", "Paste image from clipboard"),
                            ("Ctrl+L", "Redraw screen"),
                            ("PgUp / PgDn", "Scroll transcript"),
                            ("Ctrl+Home / Ctrl+End", "Jump to first/latest message"),
                            ("Ctrl+F", "Search; Enter/Shift+Enter next/previous"),
                            ("Mouse wheel / drag", "Scroll / select and copy text"),
                        ]
                        .map(|(usage, desc)| (usage.to_string(), desc)),
                    );
                    for (usage, desc) in &cmds {
                        repl.println(Line::from(format!("  {usage:<30} {desc}")))?;
                    }
                    repl.println(Line::default())?;
                    continue;
                }
                _ => {
                    // Custom agent: `/{name} [message]` switches the active
                    // agent persistently. With a message, the switch happens and
                    // the message is sent this turn; without one, it just switches.
                    if let Some(def) = agent_store.get(cmd) {
                        let applied =
                            crate::agents::apply_agent(agent, config, &def, &mut tui).await?;
                        system_prompt = format!("{}{skill_context}", def.system_prompt);
                        conversation.set_system(&system_prompt);
                        if let Some(choice) = applied.model {
                            active = choice;
                        }
                        if let Some(r) = applied.reasoning {
                            current_reasoning = r;
                        }
                        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);
                        if args.trim().is_empty() {
                            continue;
                        }
                        text = args.trim().to_string();
                    } else if let Some(provider) = &state.skill_provider {
                        let guard = provider.read().await;
                        if let Some(skill) = guard.get(cmd) {
                            let skill = skill.clone();
                            drop(guard);
                            let body = &skill.body;
                            conversation.add(ConversationEntry::system_message(format!(
                                "<skill name=\"{cmd}\">\n{body}\n</skill>"
                            )));
                            invoked_skill = Some(skill);
                            text = if args.trim().is_empty() {
                                format!("Run the /{cmd} skill.")
                            } else {
                                args.trim().to_string()
                            };
                        }
                    }
                }
            }
        }

        if !background {
            terminal_title.set(&text);

            // Add user message to conversation
            if let Some(ref mut log) = display_log {
                log.log_user(&text);
            }
            if let Some(skill) = invoked_skill {
                show_skill(&mut repl, &skill.meta.name, &skill.body)?;
                if let Some(ref mut log) = display_log {
                    log.log_skill(&skill.meta.name, &skill.body);
                }
            }
            // Resolve @mentions: attach referenced file contents as a context
            // developer entry placed immediately before the user message.
            let mentions = crate::mention::extract_mentions(&text);
            if let Some(ctx) = crate::mention::build_context_block(&cwd, &mentions) {
                conversation.add(ConversationEntry::system_message(ctx));
            }
            let searches = crate::mention::extract_content_searches(&text);
            if let Some(ctx) = crate::mention::build_search_context_block(&cwd, &searches) {
                conversation.add(ConversationEntry::system_message(ctx));
            }
            if pasted_images.is_empty() || agent.capabilities().images {
                conversation.add(user_entry(&text, pasted_images));
            } else {
                let saved = save_pasted_images(&pasted_images);
                let paths: Vec<String> = saved.iter().map(|p| p.display().to_string()).collect();
                let hint = if paths.len() == 1 {
                    format!(
                        "{text}\n\n[The user shared an image saved at {}. \
                         Use the `image_read` tool to analyze it.]",
                        paths[0]
                    )
                } else {
                    format!(
                        "{text}\n\n[The user shared {} images saved at: {}. \
                         Use the `image_read` tool to analyze them.]",
                        paths.len(),
                        paths.join(", ")
                    )
                };
                conversation.add(ConversationEntry::user(&hint));
            }
        }

        // Fold any messages queued during the previous turn into this same
        // turn, so the agent responds to everything typed at once instead of
        // surfacing each queued message as its own sequential follow-up turn.
        // Each is echoed (it was only shown dimmed while queued) and appended
        // as its own user entry, preserving message boundaries.
        steer(&mut repl, conversation, &mut display_log)?;

        conversation.mark_turn_start();

        run_turn_loop(
            agent,
            conversation,
            &mut repl,
            UsageAccounting {
                total: &mut usage,
                pricing: &active.pricing,
                model: &active.model,
                catalog: &model_completion,
                store,
                session_key: &session_key,
            },
            &mut display_log,
            &mut notifications,
        )
        .await?;
        tool_sync.sync(agent.tools_mut()).await;
        repl.set_available_commands(
            build_available_commands(&state.skill_provider, &agent_store).await,
        );
        turn_count += 1;

        if let Some(ref mut log) = display_log {
            save_display_log(log, &mut repl)?;
        }

        refresh_status(&mut repl, &active, current_reasoning, usage.total_cost);

        save_turn(store, &session_key, conversation).await?;

        // Generate session title after the first exchange
        if turn_count == 1 && !title_generated {
            title_generated = true;
            let cwd = std::env::current_dir()
                .ok()
                .and_then(|p| p.to_str().map(String::from));
            store
                .save_meta(
                    &session_key,
                    None,
                    Some(&active.model.to_string()),
                    cwd.as_deref(),
                )
                .await?;
            let provider = agent.provider_arc().clone();
            let model = agent.llm().model.clone();
            let first_user = conversation
                .entries()
                .iter()
                .find(|e| e.is_user())
                .map(|e| e.content().to_string());
            let first_assistant = conversation
                .entries()
                .iter()
                .find(|e| e.is_assistant())
                .map(|e| {
                    let c = e.content();
                    if c.len() > 200 {
                        format!("{}...", truncate_utf8(c, 200))
                    } else {
                        c.to_string()
                    }
                });
            if let (Some(user_msg), Some(asst_msg)) = (first_user, first_assistant) {
                let store_clone = store.clone();
                let key = session_key.clone();
                tokio::spawn(async move {
                    if let Ok(title) = generate_title(&provider, &model, &user_msg, &asst_msg).await
                    {
                        let _ = store_clone.update_title(&key, &title).await;
                    }
                });
            }
        }

        persist_cost_usage(store, &session_key, &usage, &repl).await?;
    }

    // Final save on exit
    if turn_count > 0 {
        save_turn(store, &session_key, conversation).await?;
    }
    persist_cost_usage(store, &session_key, &usage, &repl).await?;

    drop(repl);
    tui.suspend()?;
    println!("{}", UsageSummary(&usage));
    println!("Resume: flsh resume {session_key}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Turn-based agent loop
// ---------------------------------------------------------------------------

/// Run the agent turn loop: stream one LLM turn, execute tool calls, repeat.
async fn run_turn_loop(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    mut accounting: UsageAccounting<'_>,
    display_log: &mut Option<DisplayLog>,
    notifications: &mut SubagentNotifications,
) -> Result<()> {
    let mut cancel = CancellationToken::new();
    let mut compactions: u8 = 0;
    let mut continuations: u8 = 0;
    let mut recovery = RecoveryAttempts::default();

    repl.mark_turn_start();

    loop {
        notifications.deliver(conversation);
        // Messages submitted while the model streams wait for the next
        // request: `steer` adds them after a tool batch, and a reply that
        // ends the turn leaves them for `read_input` as the next turn.

        // Stream one LLM turn (text/reasoning deltas).
        // Scoped so the mutable borrows on agent+conversation are released
        // before we touch them for tool execution.
        let mut partial_content = String::new();
        let mut partial_reasoning = String::new();
        let (cancelled, result) = {
            let mut turn = agent.run_turn(conversation, &cancel);
            let cancelled = {
                let tapped = log_stream(&mut turn, display_log).inspect(|event| match event {
                    AgentEvent::TextDelta(delta) => partial_content.push_str(delta),
                    AgentEvent::ReasoningDelta(delta) => partial_reasoning.push_str(delta),
                    _ => {}
                });
                tokio::pin!(tapped);
                stream_accounted_events(repl, &cancel, tapped, &mut accounting).await?
            };
            let result = turn
                .take_result()
                .unwrap_or_else(|| Err(anyhow::anyhow!("stream ended without result")));
            (cancelled, result)
        };

        if cancelled {
            if !partial_content.is_empty() || !partial_reasoning.is_empty() {
                conversation.add(ConversationEntry::assistant_with_reasoning(
                    partial_content,
                    (!partial_reasoning.is_empty()).then_some(partial_reasoning),
                ));
            }
            finish_cancelled_turn(repl, conversation, display_log)?;
            if resume_queued_turn(repl, conversation, display_log, &mut cancel)? {
                compactions = 0;
                recovery = RecoveryAttempts::default();
                continue;
            }
            break;
        }

        if result.is_ok() {
            recovery.record_success();
        }

        match result {
            Ok(TurnStatus::Done { .. }) => {
                if notifications.deliver(conversation) {
                    repl.finish_turn()?;
                    repl.mark_turn_start();
                    continue;
                }
                repl.finish_turn()?;
                break;
            }

            // The reply hit the output limit; the agent left a note asking
            // the model to carry it on.
            Ok(TurnStatus::Continue { .. }) => {
                compactions = 0;
                if continuations >= MAX_CONTINUATIONS {
                    repl.finish_turn()?;
                    break;
                }
                continuations += 1;
                steer(repl, conversation, display_log)?;
                continue;
            }

            Ok(TurnStatus::ToolCalls { tool_calls, .. }) => {
                compactions = 0;

                if execute_tools(
                    agent,
                    conversation,
                    repl,
                    &tool_calls,
                    &cancel,
                    display_log,
                    &mut accounting,
                )
                .await?
                {
                    if cancel.is_cancelled() {
                        finish_cancelled_turn(repl, conversation, display_log)?;
                        if resume_queued_turn(repl, conversation, display_log, &mut cancel)? {
                            recovery = RecoveryAttempts::default();
                            continue;
                        }
                    }
                    break; // tool interrupted
                }
                steer(repl, conversation, display_log)?;
                // Persist after tool execution for crash recovery
                if let Err(error) =
                    save_turn(accounting.store, accounting.session_key, conversation).await
                {
                    emit_logged(
                        repl,
                        display_log,
                        &AgentEvent::Error(format!("Could not save session checkpoint: {error}")),
                    )?;
                }
                continue;
            }

            Ok(TurnStatus::Interrupted { .. }) => {
                repl.finish_turn()?;
                break;
            }

            Ok(TurnStatus::CompactionNeeded { reason, .. }) => {
                compactions += 1;
                if compactions > 2 {
                    repl.emit_event(&AgentEvent::Error(
                        "Context too small — compaction loop".into(),
                    ))?;
                    repl.finish_turn()?;
                    break;
                }
                let threshold = matches!(reason, CompactionReason::ContextThreshold(_));
                // With auto-compact off, only compact when the reply was cut
                // short; context overflow errors still recover in try_recover.
                if !threshold || agent.auto_compact() {
                    compact(agent, conversation, repl, reason, display_log).await?;
                }
                if threshold {
                    repl.finish_turn()?;
                    break;
                }
                continue;
            }

            Err(e) => {
                let recovered = try_recover(
                    agent,
                    conversation,
                    repl,
                    &cancel,
                    &e,
                    &mut recovery,
                    display_log,
                )
                .await?;
                if cancel.is_cancelled() {
                    finish_cancelled_turn(repl, conversation, display_log)?;
                    if resume_queued_turn(repl, conversation, display_log, &mut cancel)? {
                        compactions = 0;
                        recovery = RecoveryAttempts::default();
                        continue;
                    }
                    break;
                }
                if recovered {
                    continue;
                }
                repl.emit_event(&AgentEvent::Error(e.to_string()))?;
                repl.finish_turn()?;
                break;
            }
        }
    }

    conversation.strip_agent_progress();
    conversation.strip_memories();
    conversation.strip_reminders();
    Ok(())
}

// ---------------------------------------------------------------------------
// Turn-loop helpers
// ---------------------------------------------------------------------------

struct UsageAccounting<'a> {
    total: &'a mut SessionUsage,
    pricing: &'a ModelPricing,
    model: &'a Model,
    catalog: &'a crate::model_completion::ModelCompletion,
    store: &'a SessionStore,
    session_key: &'a str,
}

async fn stream_accounted_events(
    repl: &mut Repl<'_>,
    cancel: &CancellationToken,
    stream: impl Stream<Item = AgentEvent> + Unpin,
    accounting: &mut UsageAccounting<'_>,
) -> io::Result<bool> {
    let result = repl
        .stream_events_with(cancel, stream, |repl, event| {
            if let Some(usage) = event_usage(event) {
                let pricing = event_pricing(
                    event,
                    accounting.model,
                    accounting.pricing,
                    accounting.catalog,
                );
                accrue_cost(repl, accounting.total, &pricing, usage.clone());
            }
        })
        .await;
    // Checkpoint requests and tool batches, including interrupted streams.
    persist_cost_usage(
        accounting.store,
        accounting.session_key,
        accounting.total,
        repl,
    )
    .await
    .map_err(io::Error::other)?;
    result
}

fn event_usage(event: &AgentEvent) -> Option<&TokenUsage> {
    match event {
        AgentEvent::Usage(usage) => Some(usage),
        AgentEvent::SpawnedEvent { event, .. } => event_usage(event),
        _ => None,
    }
}

fn event_pricing(
    event: &AgentEvent,
    parent: &Model,
    pricing: &ModelPricing,
    catalog: &crate::model_completion::ModelCompletion,
) -> ModelPricing {
    if let AgentEvent::SpawnedEvent {
        model: Some(model), ..
    } = event
        && model != parent
    {
        return catalog.pricing(model).unwrap_or_default();
    }
    pricing.clone()
}

/// Add a request's cost to the session total and its tokens to the turn
/// summary.
fn accrue_cost(
    repl: &mut Repl<'_>,
    total: &mut SessionUsage,
    pricing: &ModelPricing,
    usage: TokenUsage,
) {
    repl.add_usage(&usage, Some(pricing));
    total.add(&usage, pricing);
    if !total.cost_unknown {
        repl.set_cost(total.total_cost);
    }
}

/// A user entry carrying any pasted images as image parts.
pub(crate) fn user_entry(text: &str, images: Vec<PastedImage>) -> ConversationEntry {
    if images.is_empty() {
        return ConversationEntry::user(text);
    }
    let parts: Vec<ContentPart> = images
        .into_iter()
        .map(|img| ContentPart::Image {
            media_type: img.media_type,
            data: img.data,
        })
        .collect();
    ConversationEntry::user_with_parts(text, parts)
}

/// Remove the last turn: trailing tool and assistant entries, then the user
/// message before them. Returns how many entries were removed.
fn pop_last_turn(conversation: &mut Conversation) -> usize {
    let entries = conversation.entries_mut();
    let before = entries.len();
    while entries
        .last()
        .is_some_and(|e| e.is_tool() || e.is_assistant())
    {
        entries.pop();
    }
    if entries.last().is_some_and(|e| e.is_user()) {
        entries.pop();
    }
    before - entries.len()
}

/// Save the running cost and last token usage so a resumed session restores
/// them.
async fn persist_cost_usage(
    store: &SessionStore,
    session_key: &str,
    usage: &SessionUsage,
    repl: &Repl<'_>,
) -> Result<()> {
    store
        .update_session_usage(session_key, usage, repl.last_usage())
        .await
        .context("Could not save session cost and usage")?;
    Ok(())
}

/// Add messages typed during the running turn to the conversation, so the
/// model reads them on its next request instead of after the turn ends.
/// Commands remain queued for the main input loop. Returns whether messages
/// were added, so commands alone do not start another model request.
fn steer(
    repl: &mut Repl<'_>,
    conversation: &mut Conversation,
    display_log: &mut Option<DisplayLog>,
) -> Result<bool> {
    let mut added = false;
    while let Some((text, images)) = repl.take_pending_message() {
        repl.replay_user_input(&text)?;
        if let Some(log) = display_log {
            log.log_user(&text);
        }
        conversation.add(user_entry(&text, images));
        added = true;
    }
    Ok(added)
}

fn finish_cancelled_turn(
    repl: &mut Repl<'_>,
    conversation: &mut Conversation,
    display_log: &mut Option<DisplayLog>,
) -> io::Result<()> {
    conversation.add(ConversationEntry::system_message(
        "The user interrupted the previous turn.",
    ));
    emit_logged(repl, display_log, &AgentEvent::Cancelled)?;
    repl.finish_turn()
}

fn resume_queued_turn(
    repl: &mut Repl<'_>,
    conversation: &mut Conversation,
    display_log: &mut Option<DisplayLog>,
    cancel: &mut CancellationToken,
) -> Result<bool> {
    if !repl.has_pending_input() {
        return Ok(false);
    }
    conversation.strip_agent_progress();
    conversation.strip_memories();
    conversation.strip_reminders();
    if !steer(repl, conversation, display_log)? {
        return Ok(false);
    }
    conversation.mark_turn_start();
    repl.mark_turn_start();
    *cancel = CancellationToken::new();
    Ok(true)
}

/// Execute tool calls. Returns `true` if a tool interrupted (caller should break).
async fn execute_tools(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    tool_calls: &[flashmind_types::ToolCall],
    cancel: &CancellationToken,
    display_log: &mut Option<DisplayLog>,
    accounting: &mut UsageAccounting<'_>,
) -> Result<bool, io::Error> {
    let mut answered = Vec::new();
    for batch in agent.tool_batches(tool_calls) {
        let completed = {
            let mut execution = agent.execute_tool_batch(&batch, cancel);
            {
                let tapped = log_stream(&mut execution, display_log);
                tokio::pin!(tapped);
                stream_accounted_events(repl, cancel, tapped, accounting).await?;
            }
            execution
                .take_result()
                .ok_or_else(|| io::Error::other("Tool batch ended without results"))?
        };
        let mut interrupted = false;
        for done in completed {
            if cancel.is_cancelled() {
                record_cancelled_tools(conversation, tool_calls, &answered);
                return Ok(true);
            }
            let tc = &done.call;
            if let Some(approval) = done
                .result
                .payload()
                .filter(|_| done.result.is_interrupt())
                .and_then(|payload| {
                    payload
                        .as_any()
                        .downcast_ref::<flashmind_tools::bash::CommandApproval>()
                })
            {
                repl.clear_input_bar()?;
                let mut tui = Tui::new();
                let mut picker = ChoicePicker::new(
                    format!("Allow command: {}", approval.command),
                    plain_options(&["Allow this command for this session", "Deny"]),
                );
                let allowed = matches!(
                    run_choice(&mut tui, &mut picker)?,
                    Some(ChoiceResponse { selected: 0, .. })
                );
                let output = if allowed {
                    approval.approve_for_session();
                    "Approved for this session. Execute the tool again."
                } else {
                    "Command denied by user."
                };
                conversation.add(ConversationEntry::tool(&tc.id, output));
                emit_logged(
                    repl,
                    display_log,
                    &AgentEvent::ToolResult {
                        name: tc.name.clone(),
                        id: tc.id.clone(),
                        output: output.into(),
                        success: allowed,
                        elapsed_ms: done.elapsed_ms,
                        sources: Vec::new(),
                    },
                )?;
                answered.push(tc.id.clone());
                continue;
            }
            interrupted |= done.result.is_interrupt();
            answered.push(tc.id.clone());
            for event in done.finish(conversation) {
                emit_logged(repl, display_log, &event)?;
            }
        }
        if interrupted {
            repl.finish_turn()?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn record_cancelled_tools(
    conversation: &mut Conversation,
    calls: &[flashmind_types::ToolCall],
    answered: &[String],
) {
    for call in calls.iter().filter(|call| !answered.contains(&call.id)) {
        conversation.add(ConversationEntry::tool(&call.id, "Cancelled by user."));
    }
}

async fn compact(
    agent: &Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    reason: CompactionReason,
    display_log: &mut Option<DisplayLog>,
) -> Result<()> {
    let provider = agent.provider_arc().clone();
    let model = agent.llm().model.clone();

    match reason {
        CompactionReason::OutputLength => {
            if let Err(e) = conversation.compact_with_llm(&*provider, &model).await {
                tracing::warn!("Output-length compaction failed: {e}");
            }
        }
        CompactionReason::ContextThreshold(prompt_tokens) => {
            let stream = flashmind_core::compaction::try_compact(
                conversation,
                prompt_tokens,
                agent.context_window(),
                agent.keep_recent_tokens(),
                &*provider,
                &model,
            );
            tokio::pin!(stream);
            while let Some(ev) = stream.next().await {
                emit_logged(repl, display_log, &ev)?;
            }
        }
    }
    Ok(())
}

/// Show an event and record it in the display log.
fn emit_logged(
    repl: &mut Repl<'_>,
    display_log: &mut Option<DisplayLog>,
    event: &AgentEvent,
) -> io::Result<()> {
    if let Some(log) = display_log.as_mut() {
        log.log_event(event);
    }
    repl.emit_event(event)
}

/// Record each event of `stream` in the display log as it passes through.
fn log_stream<'a>(
    stream: impl Stream<Item = AgentEvent> + 'a,
    display_log: &'a mut Option<DisplayLog>,
) -> impl Stream<Item = AgentEvent> + 'a {
    stream.map(move |event| {
        if let Some(log) = display_log.as_mut() {
            log.log_event(&event);
        }
        event
    })
}

/// Attempt error recovery (compaction, stream retry, binary strip).
/// Returns `true` if recovery succeeded and the caller should `continue`.
async fn try_recover(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    cancel: &CancellationToken,
    error: &anyhow::Error,
    attempts: &mut RecoveryAttempts,
    display_log: &mut Option<DisplayLog>,
) -> Result<bool, io::Error> {
    let mut recovery = agent.recover(conversation, cancel, error, attempts);
    {
        let tapped = log_stream(&mut recovery, display_log);
        tokio::pin!(tapped);
        repl.stream_events(cancel, tapped).await?;
    }
    Ok(recovery.take_result().unwrap_or(false))
}

// ---------------------------------------------------------------------------
// Title generation
// ---------------------------------------------------------------------------

/// Run a one-off completion and return its text with surrounding whitespace
/// and quotes removed. Stream errors are skipped.
pub(crate) async fn complete_text(
    provider: &Arc<dyn LlmProvider>,
    request: CompletionRequest,
) -> String {
    let stream = provider.complete(request);
    tokio::pin!(stream);

    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if let Ok(StreamEvent::ContentDelta(delta)) = event {
            text.push_str(&delta);
        }
    }

    text.trim().trim_matches('"').trim().to_string()
}

async fn generate_title(
    provider: &Arc<dyn LlmProvider>,
    model: &Model,
    user_msg: &str,
    assistant_msg: &str,
) -> Result<String> {
    let mut request = CompletionRequest::new(
        model.clone(),
        vec![
            Message::system(
                "Generate a short title (3-6 words) for the following conversation. Reply with ONLY the title, nothing else.",
            ),
            Message::user(format!("User: {user_msg}\n\nAssistant: {assistant_msg}")),
        ],
    );
    request.max_tokens = Some(30);

    let title = complete_text(provider, request).await;
    if title.is_empty() {
        anyhow::bail!("empty title");
    }
    Ok(title)
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

pub fn print_banner(
    tui: &mut Tui,
    model_display: &str,
    reasoning: ReasoningLevel,
    tool_names: &[&str],
    always_skills: &AlwaysOnSkills,
) -> io::Result<()> {
    use ratatui::style::{Color, Modifier, Style};

    let dim = S_DIM;
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let cyan = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    // Compact, pi-style banner: name + version, model, key state, hints.
    // The verbose tool list is dropped in favor of a count.
    tui.println(&Line::default())?;
    tui.println(&Line::from(vec![
        Span::styled("  flsh", cyan),
        Span::styled(format!(" v{}", env!("CARGO_PKG_VERSION")), dim),
        Span::styled("  ", dim),
        Span::styled(model_display, bold),
    ]))?;

    let mut info_spans: Vec<Span<'static>> = Vec::new();
    info_spans.push(Span::styled("  ", dim));
    if reasoning.is_on() {
        info_spans.push(Span::styled(
            format!("thinking:{reasoning}"),
            Style::default().fg(Color::Yellow),
        ));
        info_spans.push(Span::styled("  ", dim));
    }
    if let Some(branch) = current_git_branch() {
        info_spans.push(Span::styled(
            format!("⎇ {branch}"),
            Style::default().fg(Color::Magenta),
        ));
        info_spans.push(Span::styled("  ", dim));
    }
    if !tool_names.is_empty() {
        info_spans.push(Span::styled(format!("{} tools", tool_names.len()), dim));
        info_spans.push(Span::styled("  ", dim));
    }
    info_spans.push(Span::styled(
        "/help for commands, Esc to cancel, Ctrl-D to quit",
        dim,
    ));
    tui.println(&Line::from(info_spans))?;

    for line in always_skill_lines(always_skills) {
        tui.println(&line)?;
    }

    tui.println(&Line::default())?;

    Ok(())
}

fn always_skill_lines(skills: &AlwaysOnSkills) -> Vec<Line<'static>> {
    let loaded = if skills.loaded.is_empty() {
        "none".into()
    } else {
        skills.loaded.join(", ")
    };
    let mut lines = vec![Line::from(Span::styled(
        format!("  Always-on skills: {loaded}"),
        S_AGENT,
    ))];
    if !skills.missing.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  Missing always-on skills: {}", skills.missing.join(", ")),
            S_TOOL_FAIL,
        )));
    }
    lines
}

/// A saved session and the index of the per-directory store it lives in.
struct FlatSession {
    store_idx: usize,
    summary: SessionSummary,
}

/// Every session across `store_sessions`, newest first.
fn flatten_sessions(store_sessions: &[(SessionStore, Vec<SessionSummary>)]) -> Vec<FlatSession> {
    let mut flat: Vec<FlatSession> = Vec::new();
    for (store_idx, (_, sessions)) in store_sessions.iter().enumerate() {
        for s in sessions {
            flat.push(FlatSession {
                store_idx,
                summary: s.clone(),
            });
        }
    }
    flat.sort_by_key(|e| std::cmp::Reverse(e.summary.last_updated));
    flat
}

/// Session picker with This dir / All / Recent tabs. `current` marks the
/// active session. The picker opens on This dir when it has sessions, else
/// Recent; `force_all` opens it on All.
fn session_picker(
    flat: &[FlatSession],
    title: &str,
    current: Option<&str>,
    force_all: bool,
) -> ChoicePicker {
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(String::from));
    let options: Vec<ChoiceOption> = flat
        .iter()
        .map(|e| {
            let mut opt = session_choice_option(&e.summary);
            if current == Some(e.summary.chat_key.as_str()) {
                opt.label.push_str(" ◀");
            }
            opt
        })
        .collect();
    let previews: Vec<String> = flat
        .iter()
        .map(|e| session_preview(&e.summary, true))
        .collect();
    let this_dir: Vec<usize> = flat
        .iter()
        .enumerate()
        .filter(|(_, e)| e.summary.working_dir.as_deref() == cwd.as_deref())
        .map(|(i, _)| i)
        .collect();
    let all_indices: Vec<usize> = (0..flat.len()).collect();
    let recent: Vec<usize> = (0..flat.len()).take(20).collect();
    let default_tab = if force_all {
        1
    } else if this_dir.is_empty() {
        2
    } else {
        0
    };
    let mut picker = ChoicePicker::new(title.into(), options)
        .with_previews(previews)
        .with_tabs(vec![
            ("This dir".into(), this_dir),
            ("All".into(), all_indices),
            ("Recent".into(), recent),
        ])
        .searchable(true);
    for _ in 0..default_tab {
        picker.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    }
    picker
}

/// Running cost saved with a session, stored as a decimal string.
fn summary_cost(s: &SessionSummary) -> Option<Decimal> {
    s.total_cost.as_deref().and_then(|c| c.parse().ok())
}

fn summary_totals(summary: &SessionSummary) -> SessionUsage {
    summary.total_usage.clone().unwrap_or_else(|| SessionUsage {
        total_cost: summary_cost(summary).unwrap_or_default(),
        tokens_unknown: true,
        cost_unknown: summary_cost(summary).is_none(),
        ..Default::default()
    })
}

/// Last token usage saved with a session, stored as JSON.
fn summary_usage(s: &SessionSummary) -> Option<TokenUsage> {
    s.last_usage
        .as_deref()
        .and_then(|u| serde_json::from_str(u).ok())
}

/// Width the session title is padded to in the picker row so the trailing
/// `N msgs · age` column lines up across rows.
const SESSION_TITLE_COL: usize = 36;

/// Collapse a path's home-directory prefix to `~` for compact display.
fn shorten_dir(d: &str) -> String {
    let home = dirs::home_dir()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_default();
    d.strip_prefix(&home)
        .map(|rest| format!("~{rest}"))
        .unwrap_or_else(|| d.to_string())
}

/// Share of the context window `used` takes, as a whole percentage.
fn context_pct(used: u32, context_window: u32) -> u64 {
    if context_window == 0 {
        return 0;
    }
    u64::from(used) * 100 / u64::from(context_window)
}

/// Picker options that take no typed input.
fn plain_options(labels: &[&str]) -> Vec<ChoiceOption> {
    labels
        .iter()
        .map(|label| ChoiceOption {
            label: (*label).into(),
            accepts_input: false,
        })
        .collect()
}

/// One-line label for a session row: padded title followed by message count
/// and age. Richer metadata (model, dir, cost, message previews) lives in the
/// preview pane built by [`session_preview`].
fn session_choice_option(s: &SessionSummary) -> ChoiceOption {
    let age = format_session_age(s.last_updated);
    let full_title = session_display_name(s);
    let title: String = full_title.chars().take(SESSION_TITLE_COL).collect();
    let pad = SESSION_TITLE_COL.saturating_sub(title.chars().count());
    ChoiceOption {
        label: format!(
            "{title}{:pad$}  {} msgs · {age}",
            "",
            s.entry_count,
            pad = pad
        ),
        accepts_input: false,
    }
}

/// Multi-line preview shown for the highlighted session: model, working dir
/// (when `show_dir`), cost/token usage, and the first + last messages.
fn session_preview(s: &SessionSummary, show_dir: bool) -> String {
    let mut lines: Vec<String> = Vec::new();

    if let Some(model) = s.model.as_deref() {
        lines.push(format!("model: {model}"));
    }
    if show_dir && let Some(dir) = s.working_dir.as_deref() {
        lines.push(format!("dir: {}", shorten_dir(dir)));
    }

    // Cost + token usage on one line when available.
    let mut usage = Vec::new();
    if let Some(cost) = summary_cost(s).filter(|c| !c.is_zero()) {
        usage.push(format!("${cost:.4}"));
    }
    if let Some(u) = summary_usage(s) {
        usage.push(format!("{} tokens", u.total_tokens));
    }
    if !usage.is_empty() {
        lines.push(usage.join(" · "));
    }

    let one_line = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(first) = s.first_message.as_deref().filter(|t| !t.trim().is_empty()) {
        lines.push(format!("first: {}", one_line(first)));
    }
    if let Some(last) = s.last_message.as_deref().filter(|t| !t.trim().is_empty()) {
        // Avoid repeating the same text when first == last (single-message sessions).
        let first = s.first_message.as_deref().unwrap_or("");
        if last.trim() != first.trim() {
            lines.push(format!("last: {}", one_line(last)));
        }
    }

    lines.join("\n")
}

/// Human-readable name for a session row. Prefers the generated title; falls
/// back to a cleaned snippet of the first user message; finally a dated label.
/// Never returns the raw chat key.
fn session_display_name(s: &SessionSummary) -> String {
    const MAX_WORDS: usize = 6;
    const MAX_CHARS: usize = 40;

    if let Some(t) = s.title.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        return t.to_string();
    }

    if let Some(first) = s.first_message.as_deref() {
        let collapsed = first.split_whitespace().collect::<Vec<_>>();
        if !collapsed.is_empty() {
            let snippet = collapsed
                .iter()
                .take(MAX_WORDS)
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            let snippet: String = snippet.chars().take(MAX_CHARS).collect();
            let snippet = snippet.trim_end_matches(['.', ',', '!', '?', ':', ';']);
            if !snippet.is_empty() {
                return snippet.to_string();
            }
        }
    }

    use chrono::TimeZone;
    let date = chrono::Local
        .timestamp_opt(s.last_updated, 0)
        .single()
        .map(|d| d.format("%b %-d").to_string())
        .unwrap_or_default();
    if date.is_empty() {
        "session".to_string()
    } else {
        format!("session · {date}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn startup_distinguishes_loaded_and_missing_always_on_skills() {
        let skills = AlwaysOnSkills {
            loaded: vec!["terse".into()],
            missing: vec!["missing".into()],
            ..Default::default()
        };
        let lines = always_skill_lines(&skills);
        assert_eq!(lines[0].to_string(), "  Always-on skills: terse");
        assert_eq!(lines[1].to_string(), "  Missing always-on skills: missing");
        let lines = always_skill_lines(&AlwaysOnSkills::default());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].to_string(), "  Always-on skills: none");
    }

    #[test]
    fn shell_note_keeps_the_command_and_marks_a_cut_tail() {
        let short = shell_note("ls", "exit status: 0", VecDeque::from(["a".to_string()]));
        assert!(short.contains("$ ls\na\n"));
        assert!(!short.contains("omitted"));

        let full = (0..SHELL_CONTEXT_LINES).map(|i| i.to_string()).collect();
        assert!(shell_note("seq", "exit status: 0", full).contains("earlier output omitted"));
    }

    use super::*;
    use flashmind_memory::session::SessionSummary;

    #[test]
    fn parent_and_subagent_usage_include_cache_discounts_in_session_cost() {
        let pricing = ModelPricing {
            prompt: Some(Decimal::new(2, 6)),
            completion: Some(Decimal::new(10, 6)),
            cache_read: Some(Decimal::new(2, 7)),
            cache_write: Some(Decimal::new(25, 7)),
            ..Default::default()
        };
        let usage = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 20,
            total_tokens: 1020,
            cache_read_tokens: 800,
            cache_creation_tokens: 100,
        };
        let events = [
            AgentEvent::Usage(usage.clone()),
            AgentEvent::SpawnedEvent {
                id: "worker".into(),
                task: "Example task".into(),
                model: None,
                event: Box::new(AgentEvent::Usage(usage)),
            },
            AgentEvent::Done("Done.".into()),
        ];
        let mut repl = Repl::new(Default::default());
        let mut total = SessionUsage {
            total_cost: Decimal::ONE,
            ..Default::default()
        };
        for event in events {
            if let Some(usage) = event_usage(&event) {
                accrue_cost(&mut repl, &mut total, &pricing, usage.clone());
            }
        }
        // Each request: 100 fresh + 800 cached + 100 written + 20 output.
        assert_eq!(total.total_cost, Decimal::new(100162, 5));
        assert_eq!(total.input_tokens, 2000);
        assert_eq!(total.cached_input_tokens, Some(1600));
        assert_eq!(total.output_tokens, 40);
    }

    #[test]
    fn overridden_models_do_not_use_the_parent_price_when_their_price_is_unknown() {
        let parent = "openrouter:example/parent".parse().unwrap();
        let parent_pricing = ModelPricing {
            prompt: Some(Decimal::ONE),
            completion: Some(Decimal::ONE),
            ..Default::default()
        };
        let catalog = crate::model_completion::ModelCompletion::default();
        let usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 2,
            ..Default::default()
        };
        let event = AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "Example task".into(),
            model: Some("openrouter:example/worker".parse().unwrap()),
            event: Box::new(AgentEvent::Usage(usage.clone())),
        };
        let pricing = event_pricing(&event, &parent, &parent_pricing, &catalog);
        let mut total = SessionUsage::default();
        total.add(&usage, &pricing);
        assert!(total.cost_unknown);
        assert_eq!(total.input_tokens, 10);
        assert_eq!(
            event_pricing(
                &AgentEvent::Usage(usage),
                &parent,
                &parent_pricing,
                &catalog
            )
            .prompt,
            Some(Decimal::ONE)
        );
    }

    #[test]
    fn legacy_sessions_keep_cost_without_inventing_token_totals() {
        let mut summary = sample_summary();
        summary.total_cost = Some("0.1234".into());
        summary.last_usage = Some(
            serde_json::to_string(&TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 10,
                ..Default::default()
            })
            .unwrap(),
        );
        let usage = summary_totals(&summary);
        assert!(usage.tokens_unknown);
        assert!(!usage.cost_unknown);
        assert_eq!(usage.total_cost, Decimal::new(1234, 4));
    }

    #[tokio::test]
    async fn session_cost_survives_metadata_updates_and_reopening() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sessions.db");
        let store = crate::session::open_session_db(path.clone()).await.unwrap();
        let key = "example-session";
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::user("Example question"));
        save_turn(&store, key, &conversation).await.unwrap();

        let request = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 20,
            ..Default::default()
        };
        let pricing = ModelPricing {
            prompt: Some(Decimal::new(123456789, 12)),
            completion: Some(Decimal::new(10, 6)),
            ..Default::default()
        };
        let mut repl = Repl::new(Default::default());
        repl.set_usage(request.prompt_tokens, request.completion_tokens);
        let mut usage = SessionUsage::default();
        accrue_cost(&mut repl, &mut usage, &pricing, request.clone());
        persist_cost_usage(&store, key, &usage, &repl)
            .await
            .unwrap();
        store
            .save_meta(key, Some("Example"), Some("openai:example"), None)
            .await
            .unwrap();
        drop(store);

        let store = crate::session::open_session_db(path.clone()).await.unwrap();
        let sessions = store.list_sessions().await.unwrap();
        let summary = &sessions[0];
        assert_eq!(summary_totals(summary), usage);
        assert_eq!(summary_cost(summary), Some(usage.total_cost));
        assert_eq!(summary_usage(summary).unwrap().prompt_tokens, 1000);

        let mut resumed = summary_totals(summary);
        accrue_cost(&mut repl, &mut resumed, &pricing, request);
        persist_cost_usage(&store, key, &resumed, &repl)
            .await
            .unwrap();
        store
            .save_meta(key, Some("Renamed"), Some("openai:other-example"), None)
            .await
            .unwrap();
        drop(store);

        let store = crate::session::open_session_db(path).await.unwrap();
        let sessions = store.list_sessions().await.unwrap();
        assert_eq!(summary_totals(&sessions[0]), resumed);
        assert_eq!(resumed.total_cost, usage.total_cost * Decimal::from(2));
    }

    #[test]
    fn display_log_reports_failed_saves() {
        let directory = tempfile::tempdir().unwrap();
        let mut log = DisplayLog::new(directory.path().to_path_buf());
        assert!(log.save().is_err());
    }

    #[test]
    fn display_log_preserves_invoked_skill_text() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("display.jsonl");
        let mut log = DisplayLog::new(path.clone());
        log.log_user("Run the /example skill.");
        log.log_skill("example", "# Instructions\n\nRead the code.\nCheck it.");
        log.save().unwrap();

        let events = DisplayLog::load(&path);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[1],
            DisplayEvent::Skill { name, body }
                if name == "example" && body == "# Instructions\n\nRead the code.\nCheck it."
        ));
    }

    #[test]
    fn display_log_appends_new_events_and_rewrites_after_undo() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("display.jsonl");
        let mut log = DisplayLog::new(path.clone());
        log.log_user("first");
        log.save().unwrap();

        let mut log = DisplayLog::open(path.clone());
        log.log_event(&AgentEvent::TextDelta("one".into()));
        log.log_user("second");
        log.save().unwrap();
        assert_eq!(DisplayLog::load(&path).len(), 3);

        log.pop_last_turn();
        log.save().unwrap();
        assert_eq!(DisplayLog::load(&path).len(), 2);
        log.log_event(&AgentEvent::TextDelta("more".into()));
        log.save().unwrap();
        assert_eq!(DisplayLog::load(&path).len(), 3);
    }

    #[test]
    fn cancelled_tools_do_not_duplicate_completed_results() {
        let calls: Vec<_> = ["first", "second", "third"]
            .into_iter()
            .map(|id| flashmind_types::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: serde_json::json!({}),
            })
            .collect();
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::tool("first", "completed"));
        record_cancelled_tools(&mut conversation, &calls, &["first".into()]);
        let entries = conversation.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].content(), "completed");
        assert!(
            entries[1..]
                .iter()
                .all(|entry| entry.content() == "Cancelled by user.")
        );
    }

    fn sample_summary() -> SessionSummary {
        SessionSummary {
            chat_key: "session_abcdef0123456789".into(),
            entry_count: 4,
            last_updated: 1_700_000_000,
            title: None,
            model: None,
            first_message: None,
            last_message: None,
            working_dir: None,
            total_cost: None,
            last_usage: None,
            total_usage: None,
        }
    }

    #[test]
    fn display_log_pop_last_turn_drops_last_prompt_and_reply() {
        let mut log = DisplayLog::new(PathBuf::from("unused"));
        log.log_user("first");
        log.log_event(&AgentEvent::TextDelta("one".into()));
        log.log_user("second");
        log.log_skill("example", "Read the code.");
        log.log_event(&AgentEvent::TextDelta("two".into()));
        log.pop_last_turn();
        assert_eq!(log.events.len(), 2);
        assert!(matches!(&log.events[0], DisplayEvent::User { text } if text == "first"));
    }

    #[test]
    fn display_log_starts_over_at_compaction() {
        let mut log = DisplayLog::new(PathBuf::from("unused"));
        log.log_user("first");
        log.log_event(&AgentEvent::TextDelta("one".into()));
        log.log_event(&AgentEvent::Compacted("summary".into()));
        log.log_event(&AgentEvent::TextDelta("two".into()));
        assert_eq!(log.events.len(), 2);
        assert!(matches!(
            &log.events[0],
            DisplayEvent::Agent { event: AgentEvent::Compacted(summary) } if summary == "summary"
        ));
    }

    #[test]
    fn display_name_prefers_title() {
        let mut s = sample_summary();
        s.title = Some("Fix the parser bug".into());
        assert_eq!(session_display_name(&s), "Fix the parser bug");
    }

    #[test]
    fn display_name_derives_from_first_message() {
        let mut s = sample_summary();
        s.title = None;
        s.first_message = Some("  how do   I add a new tab\nto the picker widget please  ".into());
        // collapsed, first 6 words, trailing punctuation trimmed
        assert_eq!(session_display_name(&s), "how do I add a new");
    }

    #[test]
    fn display_name_caps_length() {
        let mut s = sample_summary();
        s.title = None;
        s.first_message =
            Some("supercalifragilistic expialidocious antidisestablishmentarian".into());
        let name = session_display_name(&s);
        assert!(name.chars().count() <= 40, "got {name:?}");
    }

    #[test]
    fn display_name_falls_back_to_dated_label() {
        let mut s = sample_summary();
        s.title = None;
        s.first_message = None;
        let name = session_display_name(&s);
        assert!(name.starts_with("session"), "got {name:?}");
        assert!(
            !name.contains(&s.chat_key),
            "must not leak chat_key: {name:?}"
        );
    }
}
