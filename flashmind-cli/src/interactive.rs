use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{Stream, StreamExt};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::text::{Line, Span};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use flashmind_core::subagent::Notifications;
use flashmind_core::{
    Agent, AgentManager, AgentRun, CancellationToken, CompletedToolCall, Conversation,
    ConversationEntry, EntryKind, RunBoundary, ToolBatch,
};
use flashmind_memory::session::SessionStore;
use flashmind_memory::session::SessionSummary;
use flashmind_skills::{DiskSkillProvider, SkillProvider};
use flashmind_tui::styles::{S_AGENT, S_DIM, S_TOOL_FAIL};
use flashmind_tui::widgets::{
    ChoiceOption, ChoicePicker, ChoicePickerAction, PastedImage, StatusInfo,
};
use flashmind_tui::{CommandInfo, Repl, Tui};
use flashmind_types::llm::TokenUsage;
use flashmind_types::utils::truncate_utf8;
use flashmind_types::{
    AgentEvent, AgentInput, CompletionRequest, ContentPart, InteractionMode, LlmProvider, Message,
    Model, ModelPricing, Provider, ReasoningLevel, SessionId, StreamEvent,
};
use tokio::sync::RwLock;

use crate::autocompact::AutoCompactCommand;
use crate::config::Config;
use crate::metadata::{MetadataLoader, ModelMetadata};
use crate::provider::{build_provider, fetch_pricing};
use crate::session::{format_session_age, load_conversation_from, new_session_key, save_turn};
use crate::setup::{run_choice, run_choice_action};
use crate::terminal_title::TerminalTitle;
use crate::tools::AlwaysOnSkills;
use flashmind_types::SessionUsage;

use crate::usage::{ModelPriceSummary, UsageSummary};

// ---------------------------------------------------------------------------
// Display log — stores raw events for session restore
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        self.events.push(DisplayEvent::Agent {
            event: event.clone(),
        });
    }

    fn restore_events(&self) -> &[DisplayEvent] {
        let start = self
            .events
            .iter()
            .rposition(|event| {
                matches!(
                    event,
                    DisplayEvent::Clear
                        | DisplayEvent::Agent {
                            event: AgentEvent::Compacted(_)
                        }
                )
            })
            .unwrap_or(0);
        &self.events[start..]
    }

    fn truncate(&mut self, index: usize) {
        self.events.truncate(index);
        self.rewrite_required = true;
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

/// Restore the latest compaction summary and all subsequent events.
fn replay_session(
    repl: &mut Repl<'_>,
    display_log: Option<&DisplayLog>,
    conversation: &Conversation,
) -> io::Result<()> {
    match display_log.filter(|log| !log.events.is_empty()) {
        Some(log) => replay_display_log(repl, log.restore_events()),
        None => replay_conversation(repl, restore_conversation_entries(conversation)),
    }
}

fn restore_conversation_entries(conversation: &Conversation) -> &[ConversationEntry] {
    let entries = conversation.entries();
    let start = entries
        .iter()
        .rposition(ConversationEntry::is_summary)
        .unwrap_or(0);
    &entries[start..]
}

fn replay_transcript(
    repl: &mut Repl<'_>,
    display_log: Option<&DisplayLog>,
    conversation: &Conversation,
) -> io::Result<()> {
    match display_log.filter(|log| !log.events.is_empty()) {
        Some(log) => replay_display_log(repl, &log.events),
        None => replay_conversation(repl, conversation.entries()),
    }
}

fn replay_conversation(repl: &mut Repl<'_>, entries: &[ConversationEntry]) -> io::Result<()> {
    let tool_names: HashMap<&str, &str> = entries
        .iter()
        .filter_map(|entry| match &entry.kind {
            EntryKind::Assistant { tool_calls, .. } => tool_calls.as_ref(),
            _ => None,
        })
        .flatten()
        .map(|call| (call.id.as_str(), call.name.as_str()))
        .collect();
    for entry in entries {
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
                repl.replay_event(&AgentEvent::ToolResult {
                    id: call_id.clone(),
                    name: name.into(),
                    output: output.clone(),
                    success: !output.starts_with("[exit code:")
                        || output.starts_with("[exit code: 0]"),
                    elapsed_ms: 0,
                    sources: Vec::new(),
                })?;
            }
            EntryKind::Developer {
                content,
                tag: Some(tag),
                ..
            } if tag == "summary" => {
                repl.replay_event(&AgentEvent::Compacted(content.clone()))?;
            }
            EntryKind::Developer { tag: Some(tag), .. } if tag == "interaction_mode" => {}
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
        CommandInfo::new("/models", "Browse models, newest first").with_args("[provider:name]"),
        CommandInfo::new("/mode", "Switch Read or Act mode (Shift+Tab)").with_args("[read|act]"),
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
        CommandInfo::new("/undo", "Remove the last user message and its replies"),
        CommandInfo::new("/retry", "Retry last message, optionally with more context")
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
        CommandInfo::new("/subagents", "Open a subagent conversation (Ctrl+G)"),
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
    let mut extras = Vec::new();
    if let Some(provider) = skill_provider {
        let guard = provider.read().await;
        for skill in guard.list() {
            let description = skill.meta.description.clone().unwrap_or_default();
            extras.push(CommandInfo::new(
                format!("/{}", skill.meta.name),
                description,
            ));
        }
    }
    for def in agents.list() {
        extras.push(CommandInfo::new(
            format!("/{}", def.name),
            def.description.unwrap_or_else(|| "Custom agent".into()),
        ));
    }
    order_commands(slash_commands(), extras)
}

/// Built-in commands first, then skills and agents, each group sorted by
/// name. An extra never replaces a command listed before it.
fn order_commands(mut builtin: Vec<CommandInfo>, extras: Vec<CommandInfo>) -> Vec<CommandInfo> {
    builtin.sort_by(|a, b| a.name.cmp(&b.name));
    let mut added: Vec<CommandInfo> = Vec::new();
    for command in extras {
        let taken = builtin
            .iter()
            .chain(&added)
            .any(|existing| existing.name == command.name);
        if !taken {
            added.push(command);
        }
    }
    added.sort_by(|a, b| a.name.cmp(&b.name));
    builtin.extend(added);
    builtin
}

// ---------------------------------------------------------------------------
// Status bar helpers
// ---------------------------------------------------------------------------

/// Detect the current git branch by shelling out to `git`.
/// Returns `None` if not in a git repo or git is unavailable.
fn current_git_branch(cwd: Option<&Path>) -> Option<String> {
    let mut command = std::process::Command::new("git");
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command
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
    cwd: Option<&Path>,
) {
    repl.set_status(StatusInfo {
        working_dir: cwd.map(Path::to_path_buf),
        model: active.model.name().to_string(),
        thinking: Some(thinking),
        cost: Some(cost),
        context: active
            .context_window
            .and_then(|cw| repl.last_usage().map(|u| (u.prompt_tokens, cw))),
        git_branch: current_git_branch(cwd),
    });
}

/// Reset the status bar for new or cleared sessions.
fn reset_status(repl: &mut Repl<'_>, model: &str, thinking: ReasoningLevel, cwd: Option<&Path>) {
    repl.set_status(StatusInfo {
        working_dir: cwd.map(Path::to_path_buf),
        model: model.to_string(),
        thinking: Some(thinking),
        cost: None,
        context: None,
        git_branch: current_git_branch(cwd),
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
async fn run_shell_escape(
    repl: &mut Repl<'_>,
    cmdline: &str,
    cwd: &Path,
) -> Result<Option<String>> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
    let mut child = match Command::new(shell)
        .current_dir(cwd)
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
    let cwd = agent.working_dir().unwrap_or(Path::new(".")).to_path_buf();
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
        let mut picker = session_picker(
            &flat,
            "Select a session to resume:",
            None,
            all,
            std::env::current_dir().ok().as_deref(),
        );
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

    let model = resume_model(cli, config, session)?;
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

pub(crate) fn configured_providers(config: &Config) -> Vec<(&str, Provider)> {
    let mut providers = Vec::new();
    if config.ollama_url.is_some() || cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        providers.push(("Ollama", Provider::Ollama));
    }
    if config.api_key_for(&Provider::OpenRouter).is_some() {
        providers.push(("OpenRouter", Provider::OpenRouter));
    }
    if config.api_key_for(&Provider::Anthropic).is_some() {
        providers.push(("Anthropic", Provider::Anthropic));
    }
    if config.api_key_for(&Provider::OpenAi).is_some() {
        providers.push(("OpenAI", Provider::OpenAi));
    }
    if config.api_key_for(&Provider::Mistral).is_some() {
        providers.push(("Mistral", Provider::Mistral));
    }
    let elevenlabs = Provider::Custom("elevenlabs".into());
    if config.api_key_for(&elevenlabs).is_some()
        && config.compatible_provider("elevenlabs").is_none()
    {
        providers.push(("ElevenLabs", elevenlabs));
    }
    providers.extend(
        config
            .compatible_providers()
            .filter(|(name, _)| matches!(name.parse::<Provider>(), Ok(Provider::Custom(_))))
            .map(|(name, _)| (name.as_str(), Provider::Custom(name.clone()))),
    );
    providers
}

/// The active model with what the session needs to know about it.
pub(crate) struct ModelChoice {
    pub(crate) model: Model,
    pub(crate) pricing: ModelPricing,
    pub(crate) context_window: Option<u32>,
}

fn resume_model(cli: &crate::Cli, config: &Config, session: &SessionSummary) -> Result<Model> {
    if cli.model.is_some() {
        return crate::provider::resolve_model(cli, config);
    }
    match session.model.as_deref().map(str::parse).transpose()? {
        Some(model) => Ok(model),
        None => crate::provider::resolve_model(cli, config),
    }
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
    let provider = if model.provider == agent.provider().provider() {
        agent.provider_arc().clone()
    } else {
        build_provider(&model.provider, config)?
    };
    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = agent.switch_model(provider, model.clone()).await;
    crate::autocompact::apply_settings(agent, config);

    Ok(ModelChoice {
        model,
        pricing,
        context_window,
    })
}

fn cancellable_compaction<'a>(
    conversation: &'a mut Conversation,
    provider: &'a dyn LlmProvider,
    model: &'a Model,
    cancel: &'a CancellationToken,
) -> impl Stream<Item = AgentEvent> + 'a {
    flashmind_core::compaction::compact_manually(conversation, provider, model)
        .take_until(cancel.cancelled())
}

const MODEL_COMMAND_TIMEOUT: Duration = Duration::from_secs(15);

async fn switch_model_interactively(
    agent: &mut Agent,
    model: &str,
    config: &Config,
    repl: &mut Repl<'_>,
) -> Result<Option<ModelChoice>> {
    let result = repl
        .run_cancellable_task(
            "switching model",
            tokio::time::timeout(MODEL_COMMAND_TIMEOUT, apply_model(agent, model, config)),
        )
        .await?;
    match result {
        Some(Ok(Ok(choice))) => Ok(Some(choice)),
        Some(Ok(Err(error))) => {
            repl.println(Line::from(format!("  Could not switch model: {error}")))?;
            Ok(None)
        }
        Some(Err(_)) => {
            repl.println(Line::from(
                "  Model switch timed out; current model unchanged.",
            ))?;
            Ok(None)
        }
        None => Ok(None),
    }
}

async fn handle_model_command(
    args: &str,
    agent: &mut Agent,
    config: &Config,
    tui: &mut Tui,
    repl: &mut Repl<'_>,
) -> Result<Option<ModelChoice>> {
    let input = args.trim();

    if !input.is_empty() {
        let Some(choice) = switch_model_interactively(agent, input, config, repl).await? else {
            return Ok(None);
        };
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
    providers.retain(
        |(_, provider)| !matches!(provider, Provider::Custom(name) if name == "elevenlabs"),
    );
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
    let result = repl
        .run_cancellable_task(
            "fetching models",
            tokio::time::timeout(MODEL_COMMAND_TIMEOUT, tmp_provider.list_models()),
        )
        .await?;
    let models = match result {
        Some(Ok(Some(models))) => models,
        Some(Ok(None)) => {
            repl.println(Line::from("  Failed to list models."))?;
            return Ok(None);
        }
        Some(Err(_)) => {
            repl.println(Line::from("  Listing models timed out."))?;
            return Ok(None);
        }
        None => return Ok(None),
    };
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
    let Some(choice) = switch_model_interactively(agent, &model.to_string(), config, repl).await?
    else {
        return Ok(None);
    };

    tui.println(&Line::from(Span::styled(
        format!(
            "  Switched to {} {}",
            model.name(),
            ModelPriceSummary(&choice.pricing)
        ),
        S_AGENT,
    )))?;

    Ok(Some(choice))
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

fn apply_metadata(
    metadata: ModelMetadata,
    agent: &mut Agent,
    active: &mut ModelChoice,
    reasoning: &mut ReasoningLevel,
) {
    if metadata.apply(agent) {
        active.context_window = metadata.context_window();
        active.pricing = metadata.pricing;
        *reasoning = agent.llm().reasoning;
    }
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
        crate::tools::skill_instructions(
            &crate::tools::skill_index(&*provider),
            agent.tools().contains("skill_load"),
        )
    } else {
        String::new()
    };
    skill_context.push_str(&state.always_skills.prompt);
    let agent_store = crate::agents::AgentStore::open()?;
    skill_context.push_str(&agent_store.instructions(agent.tools().contains("delegate")));
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

    let history_file = crate::paths::state_dir().ok().map(|d| d.join("history"));
    let cwd = agent.working_dir().unwrap_or(Path::new(".")).to_path_buf();
    let mention_provider = Arc::new(crate::mention::CwdMentionProvider::new(cwd.clone()))
        as Arc<dyn flashmind_tui::MentionProvider>;
    let model_completion = Arc::new(crate::model_completion::ModelCompletion::default());
    let repl_config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: None,
        history_file,
        available_commands: build_available_commands(&state.skill_provider, &agent_store).await,
        mention_provider: Some(mention_provider),
        argument_provider: Some(model_completion.clone()),
        stream_chars_per_second: config
            .stream_chars_per_second
            .unwrap_or_else(|| ReplConfig::default().stream_chars_per_second),
        tool_display_interval: config
            .tool_display_interval_ms
            .map(Duration::from_millis)
            .unwrap_or_else(|| ReplConfig::default().tool_display_interval),
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
    store.update_model(&session_key, &active.model).await?;
    let mut current_reasoning = state.reasoning;
    let mut system_prompt = state.system_prompt;
    let mut title_generated = false;
    let mut turn_count: usize = 0;

    // Default agent state, captured so `/agents reset` can restore it after a
    // custom agent has been switched in.
    let default_system_prompt = system_prompt.clone();
    let default_model_str = active.model.to_string();
    let default_reasoning = current_reasoning;

    repl.set_mode(conversation.interaction_mode());
    conversation.set_interaction_mode(repl.mode());

    // Replay display log if restoring a session
    let mut display_log = state.display_log_path.map(DisplayLog::open);
    replay_session(&mut repl, display_log.as_ref(), conversation)?;
    if !state.skip_banner {
        print_banner(&mut tui, agent.tools().list().len(), &state.always_skills)?;
    }

    // Restore running cost and last token usage from the session state (resume).
    let mut usage = state.usage;
    if let Some(ref u) = state.last_usage {
        repl.set_usage(u.prompt_tokens, u.completion_tokens);
    }
    refresh_status(
        &mut repl,
        &active,
        current_reasoning,
        usage.total_cost,
        agent.working_dir(),
    );
    repl.draw_input()?;

    let mut metadata = MetadataLoader::start(agent);
    let providers = configured_providers(config)
        .into_iter()
        .map(|(_, p)| p)
        .filter(|provider| !matches!(provider, Provider::Custom(name) if name == "elevenlabs"));
    let mut catalog_tasks = model_completion.start(providers, config);

    let mut notifications = Notifications::new(state.subagents);
    loop {
        let cwd = agent.working_dir().unwrap_or(Path::new(".")).to_path_buf();
        repl.set_mention_provider(Arc::new(crate::mention::CwdMentionProvider::new(
            cwd.clone(),
        )));
        let mut completed = Vec::new();
        let input = repl.read_input_until(|repl| {
            metadata.discard_if_changed(agent);
            if let Some(result) = metadata.try_ready() {
                apply_metadata(result, agent, &mut active, &mut current_reasoning);
                refresh_status(
                    repl,
                    &active,
                    current_reasoning,
                    usage.total_cost,
                    agent.working_dir(),
                );
                repl.draw_input()?;
            }
            while let Some(result) = catalog_tasks.try_join_next() {
                if let Err(error) = result {
                    tracing::warn!(%error, "model catalog task failed");
                }
            }
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
        conversation.set_interaction_mode(repl.mode());
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
            if let Some(note) = run_shell_escape(&mut repl, cmdline, &cwd).await? {
                conversation.add(ConversationEntry::system_message(note));
            }
            continue;
        }
        // Slash command dispatch
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
            match cmd {
                "mode" => {
                    if args.trim().is_empty() {
                        repl.println(Line::from(format!(
                            "  {} mode. Shift+Tab switches modes.",
                            repl.mode()
                        )))?;
                    } else {
                        match args.trim().parse::<InteractionMode>() {
                            Ok(mode) => {
                                repl.set_mode(mode);
                                conversation.set_interaction_mode(mode);
                                save_turn(store, &session_key, conversation).await?;
                                repl.println(Line::from(format!("  {mode} mode")))?;
                            }
                            Err(_) => repl.println(Line::from("  Usage: /mode [read|act]"))?,
                        }
                    }
                    continue;
                }
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
                            } else if agent_store.get(rest).is_some() {
                                repl.println(Line::from("  Imported Markdown agents must be edited or removed in their source directory."))?;
                            } else {
                                repl.println(Line::from(format!("  No agent named '{rest}'")))?;
                            }
                        }
                        "reset" => {
                            let Some(choice) = switch_model_interactively(
                                agent,
                                &default_model_str,
                                config,
                                &mut repl,
                            )
                            .await?
                            else {
                                continue;
                            };
                            conversation.set_system(&default_system_prompt);
                            system_prompt = default_system_prompt.clone();
                            active = choice;
                            store.update_model(&session_key, &active.model).await?;
                            agent.llm_mut().reasoning = default_reasoning;
                            current_reasoning = default_reasoning;
                            refresh_status(
                                &mut repl,
                                &active,
                                current_reasoning,
                                usage.total_cost,
                                agent.working_dir(),
                            );
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
                "model" | "models" => {
                    if let Some(choice) =
                        handle_model_command(args, agent, config, &mut tui, &mut repl).await?
                    {
                        active = choice;
                        store.update_model(&session_key, &active.model).await?;
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
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
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
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
                    if conversation.summarizable_entry_count() > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    session_key = new_session_key();
                    agent.llm_mut().session_id = Some(session_key.parse()?);
                    *conversation = Conversation::with_system(&system_prompt);
                    conversation.set_interaction_mode(repl.mode());
                    display_log = Some(DisplayLog::new(display_log_path(&session_key)?));
                    title_generated = false;
                    terminal_title.set("");
                    turn_count = 0;
                    usage = SessionUsage::default();
                    reset_status(
                        &mut repl,
                        active.model.name(),
                        current_reasoning,
                        agent.working_dir(),
                    );
                    repl.println(Line::from(Span::styled("  New session started", S_AGENT)))?;
                    continue;
                }
                "clear" => {
                    repl.clear_transcript()?;
                    *conversation = Conversation::with_system(&system_prompt);
                    conversation.set_interaction_mode(repl.mode());
                    terminal_title.set("");
                    turn_count = 0;
                    usage = SessionUsage::default();
                    if let Some(ref mut log) = display_log {
                        log.clear();
                        save_display_log(log, &mut repl)?;
                    }
                    save_turn(store, &session_key, conversation).await?;
                    reset_status(
                        &mut repl,
                        active.model.name(),
                        current_reasoning,
                        agent.working_dir(),
                    );
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
                        let cancel = CancellationToken::new();
                        let stream = cancellable_compaction(
                            conversation,
                            agent.provider(),
                            &agent.llm().model,
                            &cancel,
                        );
                        let tapped = log_stream(stream, &mut display_log);
                        tokio::pin!(tapped);
                        repl.set_activity("compacting");
                        repl.stream_events(&cancel, tapped).await?;
                        repl.clear_activity();
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
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
                        save_turn(store, &session_key, conversation).await?;
                    }
                    continue;
                }
                "undo" => {
                    if notifications.active_count().await > 0 {
                        repl.println(Line::from("  Wait for subagents to finish before undo."))?;
                        continue;
                    }
                    if undo_conversation(conversation, &mut display_log) {
                        save_turn(store, &session_key, conversation).await?;
                        if let Some(log) = display_log.as_mut() {
                            save_display_log(log, &mut repl)?;
                        }
                        turn_count = conversation
                            .entries()
                            .iter()
                            .filter(|entry| entry.is_user())
                            .count();
                        repl.clear_transcript()?;
                        replay_transcript(&mut repl, display_log.as_ref(), conversation)?;
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
                        repl.println(Line::from("  Last user message and replies removed."))?;
                    } else {
                        repl.println(Line::from("  Nothing to undo."))?;
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
                    if notifications.active_count().await > 0 {
                        repl.println(Line::from(
                            "  Wait for subagents to finish before retrying.",
                        ))?;
                        continue;
                    }
                    if let Some(original) = last_user_msg {
                        undo_conversation(conversation, &mut display_log);

                        let retry_msg = if extra.is_empty() {
                            original
                        } else {
                            format!("{original}\n\n{extra}")
                        };
                        if let Some(log) = display_log.as_mut() {
                            log.log_user(&retry_msg);
                        }
                        repl.clear_transcript()?;
                        replay_transcript(&mut repl, display_log.as_ref(), conversation)?;
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
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
                        save_turn(store, &session_key, conversation).await?;
                    } else {
                        repl.println(Line::from(Span::styled("  Nothing to retry", S_DIM)))?;
                    }
                    continue;
                }
                "sessions" | "resume" => {
                    persist_cost_usage(store, &session_key, &usage, &repl).await?;
                    if conversation.summarizable_entry_count() > 0 {
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
                    let mut picker = session_picker(
                        &flat,
                        "Switch session:",
                        Some(&session_key),
                        false,
                        agent.working_dir(),
                    );
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
                                if Some(target) != agent.working_dir() {
                                    if target.is_dir() {
                                        agent.set_working_dir(target.to_path_buf());
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
                            repl.set_mode(conversation.interaction_mode());
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
                            refresh_status(
                                &mut repl,
                                &active,
                                current_reasoning,
                                usage.total_cost,
                                agent.working_dir(),
                            );
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
                    if let Err(e) = std::fs::write(cwd.join(&path), &output) {
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
                    let cwd = agent
                        .working_dir()
                        .and_then(Path::to_str)
                        .map(str::to_owned);
                    store
                        .save_meta(
                            &new_key,
                            Some(&fork_title),
                            Some(&active.model.to_string()),
                            cwd.as_deref(),
                        )
                        .await?;
                    session_key = new_key;
                    if let Some(log) = display_log.as_mut() {
                        log.path = display_log_path(&session_key)?;
                        log.rewrite_required = true;
                        save_display_log(log, &mut repl)?;
                    }
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
                    rows.push(("mode", repl.mode().to_string()));
                    rows.push(("cost", format!("${}", usage.total_cost)));
                    rows.push(("session", session_key.clone()));
                    rows.push(("turns", format!("{turn_count}")));
                    if let Some(branch) = current_git_branch(agent.working_dir()) {
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
                            ("Shift+Tab", "Switch between Read and Act modes"),
                            ("Ctrl+B", "Background the running command"),
                            ("Ctrl+G", "Open subagents"),
                            ("Ctrl+V", "Paste image from clipboard"),
                            ("Ctrl+L", "Redraw screen"),
                            ("PgUp / PgDn", "Scroll transcript"),
                            ("Ctrl+Home / Ctrl+End", "Jump to first/latest message"),
                            ("Alt+Up / Alt+Down", "Jump to previous/next prompt"),
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
                        let result = repl
                            .run_cancellable_task(
                                "switching agent",
                                tokio::time::timeout(
                                    MODEL_COMMAND_TIMEOUT,
                                    crate::agents::apply_agent(agent, config, &def, &mut tui),
                                ),
                            )
                            .await?;
                        let applied = match result {
                            Some(Ok(result)) => result?,
                            Some(Err(_)) => {
                                repl.println(Line::from(
                                    "  Agent switch timed out; current agent unchanged.",
                                ))?;
                                continue;
                            }
                            None => continue,
                        };
                        system_prompt = format!("{}{skill_context}", def.system_prompt);
                        conversation.set_system(&system_prompt);
                        if let Some(choice) = applied.model {
                            active = choice;
                            store.update_model(&session_key, &active.model).await?;
                        }
                        if let Some(r) = applied.reasoning {
                            current_reasoning = r;
                        }
                        refresh_status(
                            &mut repl,
                            &active,
                            current_reasoning,
                            usage.total_cost,
                            agent.working_dir(),
                        );
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

        metadata.discard_if_changed(agent);
        if metadata.is_pending() {
            let Some(result) = repl
                .run_cancellable_task("loading model", metadata.ready())
                .await?
            else {
                continue;
            };
            if let Some(result) = result {
                apply_metadata(result, agent, &mut active, &mut current_reasoning);
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
        if let Some(path) = agent.working_dir() {
            store.update_working_dir(&session_key, path).await?;
        }
        tool_sync.sync(agent.tools_mut()).await;
        repl.set_available_commands(
            build_available_commands(&state.skill_provider, &agent_store).await,
        );
        turn_count += 1;

        if let Some(ref mut log) = display_log {
            save_display_log(log, &mut repl)?;
        }

        refresh_status(
            &mut repl,
            &active,
            current_reasoning,
            usage.total_cost,
            agent.working_dir(),
        );

        conversation.set_interaction_mode(repl.mode());
        save_turn(store, &session_key, conversation).await?;

        // Generate session title after the first exchange
        if turn_count == 1 && !title_generated {
            title_generated = true;
            let cwd = agent
                .working_dir()
                .and_then(Path::to_str)
                .map(str::to_owned);
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
    if turn_count > 0
        || conversation
            .entries()
            .iter()
            .any(|entry| entry.is_user() || entry.is_summary())
    {
        conversation.set_interaction_mode(repl.mode());
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

fn undo_conversation(conversation: &mut Conversation, display: &mut Option<DisplayLog>) -> bool {
    let Some(index) = conversation
        .entries()
        .iter()
        .rposition(|entry| entry.is_user())
    else {
        return false;
    };
    conversation.entries_mut().truncate(index);
    conversation.mark_turn_start();
    if let Some(log) = display {
        let index = log
            .events
            .iter()
            .rposition(|event| matches!(event, DisplayEvent::User { .. }))
            .unwrap_or(0);
        log.truncate(index);
    }
    true
}

/// Run the agent turn loop: stream one LLM turn, execute tool calls, repeat.
async fn run_turn_loop(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    mut accounting: UsageAccounting<'_>,
    display_log: &mut Option<DisplayLog>,
    notifications: &mut Notifications,
) -> Result<()> {
    let mut cancel = CancellationToken::new();
    let mut run = AgentRun::new(conversation, AgentInput::Resume, None);
    repl.mark_turn_start();

    loop {
        let (cancelled, boundary) = {
            let mut step = run.advance(agent, conversation, &cancel);
            let cancelled = {
                let tapped = log_stream(&mut step, display_log);
                tokio::pin!(tapped);
                stream_accounted_events(repl, &cancel, tapped, &mut accounting).await?
            };
            // The renderer stops on an Error event, including a recoverable
            // compaction failure. Finish the step to release its checkpoint.
            while let Some(event) = step.next().await {
                emit_logged(repl, display_log, &event)?;
            }
            let boundary = step
                .take_result()
                .unwrap_or_else(|| Err(anyhow::anyhow!("Agent step ended without a result")));
            (cancelled, boundary)
        };

        if cancelled || cancel.is_cancelled() {
            run.finish(conversation);
            finish_cancelled_turn(repl, conversation, display_log)?;
            save_turn(accounting.store, accounting.session_key, conversation).await?;
            if resume_queued_turn(repl, conversation, display_log, &mut cancel)? {
                run = AgentRun::new(conversation, AgentInput::Resume, None);
                continue;
            }
            break;
        }

        match boundary {
            Ok(RunBoundary::BeforeModel) => {
                conversation.set_interaction_mode(repl.mode());
                sync_workspace_context(agent, conversation);
                notifications.deliver(conversation);
                steer(repl, conversation, display_log)?;
                save_turn(accounting.store, accounting.session_key, conversation).await?;
            }
            Ok(RunBoundary::Checkpoint(_)) => {
                save_turn(accounting.store, accounting.session_key, conversation).await?;
            }
            Ok(RunBoundary::Tools(tool_calls)) => {
                let interrupted = execute_tools(
                    agent,
                    conversation,
                    repl,
                    &tool_calls,
                    &cancel,
                    display_log,
                    &mut accounting,
                )
                .await?;
                if interrupted {
                    run.finish(conversation);
                    if cancel.is_cancelled() {
                        finish_cancelled_turn(repl, conversation, display_log)?;
                    }
                    save_turn(accounting.store, accounting.session_key, conversation).await?;
                    if cancel.is_cancelled()
                        && resume_queued_turn(repl, conversation, display_log, &mut cancel)?
                    {
                        run = AgentRun::new(conversation, AgentInput::Resume, None);
                        continue;
                    }
                    break;
                }
                steer(repl, conversation, display_log)?;
            }
            Ok(RunBoundary::Done(_)) => {
                repl.finish_turn()?;
                if notifications.deliver(conversation) {
                    repl.mark_turn_start();
                    run = AgentRun::new(conversation, AgentInput::Resume, None);
                    continue;
                }
                break;
            }
            Err(error) => {
                run.finish(conversation);
                emit_logged(repl, display_log, &AgentEvent::Error(error.to_string()))?;
                repl.finish_turn()?;
                break;
            }
        }
    }

    run.finish(conversation);
    Ok(())
}

// ---------------------------------------------------------------------------
// Turn-loop helpers
// ---------------------------------------------------------------------------

fn sync_workspace_context(agent: &Agent, conversation: &mut Conversation) {
    let Some(path) = agent.working_dir() else {
        return;
    };
    let metadata = serde_json::json!({"working_dir": path, "claude": crate::is_claude_model(&agent.llm().model)});
    let previous = conversation
        .entries()
        .iter()
        .rev()
        .find_map(|entry| match &entry.kind {
            EntryKind::Developer { tag, metadata, .. } if tag.as_deref() == Some("workspace") => {
                metadata.as_ref()
            }
            _ => None,
        });
    if previous == Some(&metadata) {
        return;
    }
    let instructions = crate::instructions::project_instructions(
        path,
        crate::instructions::global_level(),
        crate::is_claude_model(&agent.llm().model),
    );
    let mut entry = ConversationEntry::system_message(format!(
        "Current session working directory: {}. Commands and file tools resolve paths here. Project instructions for this directory follow.\n{}",
        path.display(),
        instructions,
    ));
    if let EntryKind::Developer {
        tag,
        metadata: data,
        ..
    } = &mut entry.kind
    {
        *tag = Some("workspace".into());
        *data = Some(metadata);
    }
    conversation.add(entry);
}

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
        let completed =
            run_tool_batch(agent, &batch, repl, cancel, display_log, accounting).await?;
        let mut interrupted = false;
        let mut events = Vec::new();
        for done in completed {
            if let Some(path) = done.result.working_dir() {
                accounting
                    .store
                    .update_working_dir(accounting.session_key, path)
                    .await
                    .map_err(io::Error::other)?;
                repl.set_git_branch(current_git_branch(Some(path)));
                repl.set_working_dir(path);
                repl.set_mention_provider(Arc::new(crate::mention::CwdMentionProvider::new(
                    path.to_path_buf(),
                )));
            }
            if cancel.is_cancelled() {
                record_cancelled_tools(conversation, tool_calls, &answered);
                return Ok(true);
            }
            interrupted |= done.result.is_interrupt();
            answered.push(done.call.id.clone());
            events.extend(done.finish(conversation));
        }
        let tapped = log_stream(futures_util::stream::iter(events), display_log);
        tokio::pin!(tapped);
        repl.stream_events(cancel, tapped).await?;
        if cancel.is_cancelled() {
            record_cancelled_tools(conversation, tool_calls, &answered);
        }
        // Commit completed results before the next batch can have side effects.
        save_turn(accounting.store, accounting.session_key, conversation)
            .await
            .map_err(io::Error::other)?;
        if interrupted || cancel.is_cancelled() {
            repl.finish_turn()?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// Run one batch of tool calls, showing its events as they arrive.
async fn run_tool_batch(
    agent: &mut Agent,
    batch: &ToolBatch<'_>,
    repl: &mut Repl<'_>,
    cancel: &CancellationToken,
    display_log: &mut Option<DisplayLog>,
    accounting: &mut UsageAccounting<'_>,
) -> io::Result<Vec<CompletedToolCall>> {
    let mut execution = agent.execute_tool_batch(batch, cancel);
    {
        let tapped = log_stream(&mut execution, display_log);
        tokio::pin!(tapped);
        stream_accounted_events(repl, cancel, tapped, accounting).await?;
    }
    execution
        .take_result()
        .ok_or_else(|| io::Error::other("Tool batch ended without results"))
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

fn print_banner(
    tui: &mut Tui,
    tool_count: usize,
    always_skills: &AlwaysOnSkills,
) -> io::Result<()> {
    for line in banner_lines(tool_count, always_skills) {
        tui.println(&line)?;
    }
    Ok(())
}

/// Name, version and key hints. The model, thinking level, directory and
/// branch are left to the status bar, which keeps them current.
fn banner_lines(tool_count: usize, always_skills: &AlwaysOnSkills) -> Vec<Line<'static>> {
    use ratatui::style::{Color, Modifier, Style};

    let cyan = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let mut first = vec![
        Span::styled("  flsh", cyan),
        Span::styled(format!(" v{}", env!("CARGO_PKG_VERSION")), S_DIM),
    ];
    if tool_count > 0 {
        first.push(Span::styled(format!("  {tool_count} tools"), S_DIM));
    }
    first.push(Span::styled(
        "  /help for commands, Esc to cancel, Ctrl-D to quit",
        S_DIM,
    ));
    let mut lines = vec![Line::default(), Line::from(first)];
    if !always_skills.loaded.is_empty() || !always_skills.missing.is_empty() {
        lines.extend(always_skill_lines(always_skills));
    }
    // The first prompt brings its own blank line above it.
    lines
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
    cwd: Option<&Path>,
) -> ChoicePicker {
    let cwd = cwd.and_then(Path::to_str);
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
        .filter(|(_, e)| e.summary.working_dir.as_deref() == cwd)
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
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    struct StalledProvider;

    #[async_trait::async_trait]
    impl LlmProvider for StalledProvider {
        fn name(&self) -> &str {
            "stalled"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        async fn list_models(&self) -> Option<Vec<flashmind_types::ModelInfo>> {
            std::future::pending().await
        }
        fn complete(&self, _: CompletionRequest) -> flashmind_types::CompletionStream {
            Box::pin(futures_util::stream::pending())
        }
    }

    #[tokio::test]
    async fn manual_compaction_can_be_cancelled_while_the_provider_is_stalled() {
        let mut conversation = Conversation::with_system("system prompt");
        conversation.add(ConversationEntry::user("question"));
        conversation.add(ConversationEntry::assistant("answer"));
        let before = serde_json::to_value(&conversation).unwrap();
        let cancel = CancellationToken::new();
        let model = "ollama:example".parse().unwrap();
        {
            let stream =
                cancellable_compaction(&mut conversation, &StalledProvider, &model, &cancel);
            tokio::pin!(stream);
            assert!(poll_once(std::pin::pin!(stream.next())).is_pending());
            cancel.cancel();
            assert!(matches!(
                poll_once(std::pin::pin!(stream.next())),
                std::task::Poll::Ready(None)
            ));
        }
        assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
    }

    #[tokio::test]
    async fn cancelling_a_model_switch_preserves_the_active_model() {
        let provider: Arc<dyn LlmProvider> = Arc::new(StalledProvider);
        let mut agent = Agent::builder(provider.clone()).build_sync();
        agent.llm_mut().model = "ollama:original".parse().unwrap();
        let model = agent.llm().model.clone();
        let config = Config::default();
        {
            let switch = apply_model(&mut agent, "ollama:replacement", &config);
            tokio::pin!(switch);
            assert!(poll_once(switch.as_mut()).is_pending());
        }
        assert_eq!(agent.llm().model, model);
        assert!(Arc::ptr_eq(agent.provider_arc(), &provider));
    }

    #[test]
    fn model_picker_includes_named_compatible_providers() {
        let config: Config = toml::from_str(
            r#"
            [openrouter.openai_compat.example]
            url = "https://provider.example/v1"
            api_key = "example-key"
        "#,
        )
        .unwrap();
        assert!(
            configured_providers(&config)
                .iter()
                .any(|(label, provider)| {
                    *label == "example" && *provider == Provider::Custom("example".into())
                })
        );
    }

    #[test]
    fn resume_uses_an_explicit_model_then_the_saved_model_then_the_default() {
        use clap::Parser;
        let config = Config {
            model: Some("ollama:default".into()),
            ..Default::default()
        };
        let mut session = sample_summary();
        session.model = Some("ollama:saved".into());
        let cli =
            crate::Cli::try_parse_from(["flsh", "resume", "--model", "ollama:selected"]).unwrap();
        assert_eq!(
            resume_model(&cli, &config, &session).unwrap().name(),
            "selected"
        );
        let cli = crate::Cli::try_parse_from(["flsh", "resume"]).unwrap();
        assert_eq!(
            resume_model(&cli, &config, &session).unwrap().name(),
            "saved"
        );
        session.model = None;
        assert_eq!(
            resume_model(&cli, &config, &session).unwrap().name(),
            "default"
        );
    }

    #[test]
    fn built_in_commands_come_before_skills_and_agents() {
        let names = |commands: Vec<CommandInfo>| -> Vec<String> {
            commands.into_iter().map(|command| command.name).collect()
        };
        let ordered = order_commands(
            vec![
                CommandInfo::new("/quit", "Exit"),
                CommandInfo::new("/clear", "Clear"),
            ],
            vec![
                CommandInfo::new("/review", "A skill"),
                CommandInfo::new("/clear", "A skill with a taken name"),
                CommandInfo::new("/agents-helper", "Custom agent"),
            ],
        );
        assert_eq!(
            names(ordered),
            ["/clear", "/quit", "/agents-helper", "/review"]
        );
    }

    #[test]
    fn banner_is_one_line_and_leaves_session_state_to_the_status_bar() {
        let text = |lines: Vec<Line<'static>>| -> Vec<String> {
            lines.iter().map(Line::to_string).collect()
        };
        let lines = text(banner_lines(15, &AlwaysOnSkills::default()));
        assert_eq!(lines.len(), 2, "{lines:#?}");
        assert!(lines[1].starts_with("  flsh v"));
        assert!(lines[1].contains("15 tools"));
        assert!(lines[1].contains("/help"));

        let skills = AlwaysOnSkills {
            loaded: vec!["terse".into()],
            ..Default::default()
        };
        let lines = text(banner_lines(15, &skills));
        assert_eq!(lines[2], "  Always-on skills: terse");
    }

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
        let before = log.events.clone();
        log.log_user("second");
        log.save().unwrap();
        assert_eq!(DisplayLog::load(&path).len(), 3);

        log.truncate(before.len());
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
    fn display_log_restore_drops_last_prompt_and_reply() {
        let mut log = DisplayLog::new(PathBuf::from("unused"));
        log.log_user("first");
        log.log_event(&AgentEvent::TextDelta("one".into()));
        let before = log.events.clone();
        log.log_user("second");
        log.log_skill("example", "Read the code.");
        log.log_event(&AgentEvent::TextDelta("two".into()));
        log.truncate(before.len());
        assert_eq!(log.events.len(), 2);
        assert!(matches!(&log.events[0], DisplayEvent::User { text } if text == "first"));
    }

    #[test]
    fn conversation_undo_works_after_resume_without_snapshot_history() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("example.txt");
        std::fs::write(&file, "keep edits").unwrap();
        let mut conversation = Conversation::with_system("instructions");
        conversation.add(ConversationEntry::user("first"));
        conversation.add(ConversationEntry::assistant("one"));
        conversation.add(ConversationEntry::user("second"));
        conversation.add(ConversationEntry::tool("call", "output"));
        conversation.add(ConversationEntry::assistant("two"));
        let saved = serde_json::to_vec(&conversation).unwrap();
        let mut conversation = serde_json::from_slice(&saved).unwrap();
        let path = directory.path().join("display.jsonl");
        let mut log = DisplayLog::new(path.clone());
        log.log_user("first");
        log.log_event(&AgentEvent::TextDelta("one".into()));
        log.log_user("second");
        log.log_skill("example", "instructions");
        log.log_event(&AgentEvent::TextDelta("two".into()));
        log.save().unwrap();
        let mut display = Some(DisplayLog::open(path.clone()));

        assert!(undo_conversation(&mut conversation, &mut display));
        assert_eq!(conversation.entries().len(), 3);
        assert_eq!(conversation.last_assistant_content(), Some("one"));
        display.as_mut().unwrap().save().unwrap();
        assert_eq!(DisplayLog::load(&path).len(), 2);
        assert_eq!(std::fs::read_to_string(file).unwrap(), "keep edits");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 2);

        assert!(undo_conversation(&mut conversation, &mut display));
        assert_eq!(conversation.entries().len(), 1);
        assert_eq!(conversation.entries()[0].content(), "instructions");
        assert!(display.as_ref().unwrap().events.is_empty());
        assert!(!undo_conversation(&mut conversation, &mut display));
    }

    #[test]
    fn conversation_undo_handles_partial_turns_and_compaction() {
        let mut conversation = Conversation::with_system("instructions");
        conversation.add(ConversationEntry::summary("earlier conversation"));
        let mut display = None;
        assert!(!undo_conversation(&mut conversation, &mut display));
        conversation.add(ConversationEntry::user("unfinished"));
        conversation.mark_turn_start();
        conversation.add(ConversationEntry::assistant("partial reply"));
        assert!(undo_conversation(&mut conversation, &mut display));
        assert_eq!(conversation.entries().len(), 2);
        assert!(conversation.entries()[1].is_summary());
        assert!(conversation.last_turn().is_empty());
        assert!(!undo_conversation(&mut conversation, &mut display));
    }

    #[test]
    fn compaction_preserves_the_log_and_restore_starts_at_the_latest_summary() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("display.jsonl");
        let mut log = DisplayLog::new(path.clone());
        log.log_user("Old question.");
        log.log_event(&AgentEvent::TextDelta("Old answer.".into()));
        log.log_event(&AgentEvent::Compacted("Old summary.".into()));
        log.log_user("Another question.");
        log.log_event(&AgentEvent::TextDelta("Another answer.".into()));
        log.save().unwrap();
        let before = std::fs::read(&path).unwrap();
        log.log_event(&AgentEvent::Compacted("Latest summary.".into()));
        log.log_user("Latest question.");
        log.log_skill("example", "Instructions.");
        log.log_event(&AgentEvent::ToolStart {
            id: "example".into(),
            name: "exec".into(),
            humanized: "Example command".into(),
        });
        log.log_event(&AgentEvent::TextDelta("First reply.".into()));
        log.log_user("Follow-up question.");
        log.log_event(&AgentEvent::TextDelta("Latest reply.".into()));
        log.save().unwrap();
        assert!(std::fs::read(&path).unwrap().starts_with(&before));

        let restored = DisplayLog::open(path);
        assert_eq!(restored.events.len(), 12);
        assert!(matches!(restored.restore_events(), [
            DisplayEvent::Agent { event: AgentEvent::Compacted(summary) },
            DisplayEvent::User { text: question },
            DisplayEvent::Skill { .. },
            DisplayEvent::Agent { event: AgentEvent::ToolStart { .. } },
            DisplayEvent::Agent { event: AgentEvent::TextDelta(first_reply) },
            DisplayEvent::User { text: follow_up },
            DisplayEvent::Agent { event: AgentEvent::TextDelta(last_reply) },
        ] if summary == "Latest summary."
            && question == "Latest question."
            && first_reply == "First reply."
            && follow_up == "Follow-up question."
            && last_reply == "Latest reply."));
    }

    #[test]
    fn restore_without_compaction_replays_the_whole_log() {
        let mut log = DisplayLog::new(PathBuf::from("unused"));
        assert!(log.restore_events().is_empty());
        log.log_user("First question.");
        log.log_event(&AgentEvent::TextDelta("First answer.".into()));
        log.log_user("Latest question.");
        log.log_event(&AgentEvent::TextDelta("Latest answer.".into()));
        assert_eq!(log.restore_events().len(), 4);
        assert!(matches!(
            &log.restore_events()[0],
            DisplayEvent::User { text } if text == "First question."
        ));
    }

    #[test]
    fn restore_does_not_replay_events_before_a_clear() {
        let mut log = DisplayLog::new(PathBuf::from("unused"));
        log.log_event(&AgentEvent::Compacted("Old summary.".into()));
        log.events.push(DisplayEvent::Clear);
        log.log_user("New question.");
        log.log_event(&AgentEvent::TextDelta("New answer.".into()));
        assert!(matches!(
            log.restore_events(),
            [
                DisplayEvent::Clear,
                DisplayEvent::User { .. },
                DisplayEvent::Agent {
                    event: AgentEvent::TextDelta(_)
                },
            ]
        ));
        log.clear();
        assert!(matches!(log.restore_events(), [DisplayEvent::Clear]));
    }

    #[test]
    fn restore_without_a_log_replays_the_latest_summary_and_subsequent_entries() {
        let mut conversation = Conversation::new();
        assert!(restore_conversation_entries(&conversation).is_empty());
        conversation.add(ConversationEntry::system("Example instructions."));
        conversation.add(ConversationEntry::user("Old question."));
        conversation.add(ConversationEntry::assistant("Old answer."));
        assert_eq!(restore_conversation_entries(&conversation).len(), 3);
        conversation.add(ConversationEntry::summary("Old summary."));
        conversation.add(ConversationEntry::summary("Latest summary."));
        conversation.add(ConversationEntry::user("Latest question."));
        conversation.add(ConversationEntry::tool("example", "Tool output."));
        conversation.add(ConversationEntry::assistant("First reply."));
        conversation.add(ConversationEntry::user("Follow-up question."));
        conversation.add(ConversationEntry::assistant("Latest reply."));
        let contents: Vec<_> = restore_conversation_entries(&conversation)
            .iter()
            .map(ConversationEntry::content)
            .collect();
        assert_eq!(
            contents,
            [
                "Latest summary.",
                "Latest question.",
                "Tool output.",
                "First reply.",
                "Follow-up question.",
                "Latest reply."
            ]
        );
    }

    #[test]
    fn failed_compaction_preserves_saved_history() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("display.jsonl");
        let mut log = DisplayLog::new(path.clone());
        log.log_user("Original question.");
        log.log_event(&AgentEvent::TextDelta("Original answer.".into()));
        log.save().unwrap();
        let before = std::fs::read(&path).unwrap();
        log.log_event(&AgentEvent::Error(
            "Compaction failed. Conversation unchanged.".into(),
        ));
        log.save().unwrap();
        assert!(std::fs::read(&path).unwrap().starts_with(&before));
        assert_eq!(DisplayLog::load(&path).len(), 3);
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
