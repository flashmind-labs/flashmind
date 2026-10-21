//! REPL loop — reads user input, streams agent responses, renders events.
//!
//! This module provides the [`Repl`] struct, which orchestrates the read-eval-print
//! cycle for an interactive agent CLI:
//!
//! 1. **Read** — collects multi-line user input via a [`TextArea`] widget
//!    with full cursor navigation and word-aware editing.
//! 2. **Eval** — passes the input to the agent and receives a stream of [`AgentEvent`]s.
//! 3. **Print** — renders each event incrementally using [`EventRenderer`],
//!    showing tool progress, diffs, errors, and token usage as they arrive.
//!
//! The REPL enters raw terminal mode while reading input and exits it once the user
//! submits a line or presses Ctrl+D.  Ctrl+C cancels any in-flight agent turn by
//! triggering the associated [`CancellationToken`].
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`Repl`] | Main REPL orchestrator |
//! | [`ReplConfig`] | Prompt string and optional greeting message |
//! | [`ReplEvent`] | Outcome of [`Repl::read_input`] — user text or quit signal |

mod completion;
mod input;
mod render;
pub mod subagents;
mod transcript_view;
mod turn;

use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::selection::ScreenSelection;
use crate::transcript::{TranscriptLayout, TranscriptScroll};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use rust_decimal::Decimal;
use tokio_util::sync::CancellationToken;

use flashmind_types::AgentEvent;
use flashmind_types::llm::TokenUsage;

use super::dropdown::Dropdown;
use super::pet::{Pet, PetState, PetWidget};
use super::spinner::Spinner;
use super::textarea::TextArea;
use crate::event_render::{EventRenderer, apply_to_lines, truncate_display};
use crate::styles;
use crate::term;
use crate::widgets::subagent_progress::SubagentProgress;
use completion::MentionState;
pub(crate) use input::copy_to_clipboard;
use input::load_history;
use transcript_view::TranscriptSelection;

/// RAII guard that enables raw terminal mode on creation and restores normal
/// mode on drop.  Ensures the terminal is always cleaned up, even on panic.
pub struct RawModeGuard;

impl RawModeGuard {
    pub fn enable() -> io::Result<Self> {
        enable_raw_mode()?;
        let _ = execute!(io::stdout(), crossterm::event::EnableBracketedPaste);
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), crossterm::event::DisableBracketedPaste);
        let _ = disable_raw_mode();
    }
}

/// Provides file-path candidates for `@mention` autocompletion.
///
/// Implementations are typically cwd-scoped fuzzy/substring matchers over
/// the project's files.  The REPL calls [`complete`](MentionProvider::complete)
/// with the text typed after the `@` and renders the returned paths as a
/// dropdown.  Returning an empty vector dismisses the dropdown.
pub trait MentionProvider: Send + Sync + std::fmt::Debug {
    /// Return matching file paths (relative to cwd) for the given query.
    ///
    /// The query may be empty (just typed `@`); implementations should return
    /// a reasonable default set (e.g. recently modified files).
    fn complete(&self, query: &str) -> Vec<String>;

    /// Search file contents for lines matching `query`. Returns results in
    /// `"path:line: content"` format. Default: empty (no content search).
    fn search_content(&self, _query: &str) -> Vec<String> {
        Vec::new()
    }
}

/// Provides argument candidates for slash commands, such as model names for
/// `/model`.
///
/// The REPL calls [`complete`](ArgumentProvider::complete) once the input is a
/// known command followed by a space, and shows the returned values in the
/// dropdown. Returning an empty vector hides the dropdown.
pub trait ArgumentProvider: Send + Sync + std::fmt::Debug {
    /// Return candidates for `command` (with its leading slash) given the
    /// argument text typed so far.
    fn complete(&self, command: &str, query: &str) -> Vec<String>;

    /// Optional display text beside a candidate, excluded from the argument.
    fn description(&self, _command: &str, _candidate: &str) -> Option<String> {
        None
    }
}

/// Configuration for the REPL session.
///
/// Control how the prompt appears and whether a welcome banner is printed on
/// startup.
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// The prompt prefix displayed before the input widget title bar.
    /// Default: `"▸"`.
    pub prompt: String,
    /// Optional greeting message printed once when [`Repl::print_greeting`] is called.
    /// Default: `None`.
    pub greeting: Option<String>,
    /// Placeholder text shown in the input widget when empty.
    /// Default: `"Type a message..."`.
    pub placeholder: String,
    /// Maximum height (in rows) the input widget can grow to.
    /// Default: `10`.
    pub max_input_height: u16,
    /// Whether to show tokens and cost on the line that closes each turn.
    /// Default: `true`.
    pub show_usage: bool,
    /// Optional path to a file for persisting input history across sessions.
    /// Each entry is stored as a single line with literal newlines escaped as `\\n`.
    /// Default: `None` (history is not persisted).
    pub history_file: Option<PathBuf>,
    /// Slash commands available for autocomplete and inline ghost-text hints.
    /// When the user types `/`, a dropdown of matching names is shown and a
    /// dim ghost suffix completes the matched command name + args hint.
    /// Default: empty (no autocomplete).
    pub available_commands: Vec<CommandInfo>,
    /// Optional provider for `@mention` file-path autocompletion.
    /// When set, typing `@` (at start of input or after whitespace) shows a
    /// dropdown of matching files from the provider.
    /// Default: `None` (mentions disabled).
    pub mention_provider: Option<std::sync::Arc<dyn MentionProvider>>,
    /// Optional provider for slash-command argument completion.
    /// Default: `None` (arguments are not completed).
    pub argument_provider: Option<std::sync::Arc<dyn ArgumentProvider>>,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "▸".to_string(),
            greeting: None,
            placeholder: "Type a message...".to_string(),
            max_input_height: 10,
            show_usage: true,
            history_file: None,
            available_commands: Vec::new(),
            mention_provider: None,
            argument_provider: None,
        }
    }
}

/// Metadata for a slash command, used for autocomplete and ghost-text hints.
#[derive(Debug, Clone, Default)]
pub struct CommandInfo {
    /// Full command name including the leading slash (e.g. `"/sessions"`).
    pub name: String,
    /// Arguments hint shown after the name (e.g. `"[title]"`). Empty for none.
    pub args_hint: String,
    /// Short human-readable description.
    pub description: String,
    /// Example invocation (e.g. `"/sessions"`). Empty for none.
    pub example: String,
}

impl CommandInfo {
    /// Build a command with no args hint or example.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            args_hint: String::new(),
            example: String::new(),
        }
    }

    /// Set the args hint (e.g. `"[title]"`).
    pub fn with_args(mut self, args_hint: impl Into<String>) -> Self {
        self.args_hint = args_hint.into();
        self
    }

    /// Set the example invocation.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = example.into();
        self
    }
}

/// Structured status info displayed on the right side of the input title bar.
///
/// All fields hold raw values; rendering to styled spans happens internally.
#[derive(Debug, Clone, Default)]
pub struct StatusInfo {
    /// Model name or identifier.
    pub model: String,
    /// Reasoning/thinking level. `None` or `Off` means hidden.
    pub thinking: Option<flashmind_types::ReasoningLevel>,
    /// Cumulative session cost in USD. `None` means no pricing available.
    pub cost: Option<rust_decimal::Decimal>,
    /// Context window usage: `(prompt_tokens, context_window)`. `None` means unknown.
    pub context: Option<(u32, u32)>,
    /// Current git branch (for display). `None` hides it.
    pub git_branch: Option<String>,
}

impl StatusInfo {
    fn to_spans(&self, width: usize) -> Vec<Span<'static>> {
        use ratatui::style::Color;

        if width < 3 {
            return Vec::new();
        }
        let mut fields = Vec::new();
        if !self.model.is_empty() {
            fields.push(Span::styled(self.model.clone(), styles::S_DIM));
        }
        if let Some(level) = self.thinking
            && level.is_on()
        {
            fields.push(Span::styled(format!("{level}"), styles::S_DIM));
        }
        if let Some(cost) = self.cost
            && !cost.is_zero()
        {
            let formatted = if cost < rust_decimal::Decimal::new(1, 2) {
                format!("${:.4}", cost)
            } else if cost < rust_decimal::Decimal::ONE {
                format!("${:.3}", cost)
            } else {
                format!("${:.2}", cost)
            };
            fields.push(Span::styled(formatted, styles::S_DIM));
        }
        if let Some((used, total)) = self.context
            && total > 0
        {
            let pct = (used as u64 * 100) / total as u64;
            let style = if pct >= 90 {
                styles::S_DIM.fg(Color::Red)
            } else if pct >= 70 {
                styles::S_DIM.fg(Color::Yellow)
            } else {
                styles::S_DIM
            };
            fields.push(Span::styled(format!("ctx {pct}%"), style));
        }
        if let Some(branch) = &self.git_branch
            && !branch.is_empty()
        {
            fields.push(Span::styled(branch.clone(), styles::S_DIM));
        }
        // Drop secondary fields before shortening the model name.
        while fields.len() > 1
            && fields.iter().map(Span::width).sum::<usize>() + fields.len() * 2 > width
        {
            fields.pop();
        }
        if let Some(first) = fields.first_mut() {
            first.content = truncate_display(&first.content, width.saturating_sub(2)).into();
        }
        fields
            .into_iter()
            .flat_map(|field| [Span::raw("  "), field])
            .collect()
    }
}

/// A base64-encoded image from the clipboard.
#[derive(Debug, Clone)]
pub struct PastedImage {
    /// MIME type (e.g. `"image/png"`).
    pub media_type: String,
    /// Base64-encoded image data.
    pub data: String,
}

/// Result of calling [`Repl::read_input`].
#[derive(Debug)]
pub enum ReplEvent {
    /// The user submitted text (pressed Enter on a non-empty input).
    /// Optionally carries pasted images from Ctrl+V/Cmd+V.
    UserInput(String, Vec<PastedImage>),
    /// The user pressed Ctrl+D on an empty input line.
    Quit,
}

/// Result of processing a single key event — callers decide how to act on it
/// depending on context (input-wait vs streaming).
enum KeyAction {
    OpenSubagents,
    /// The input widget changed and needs redrawing.
    Redraw,
    /// The user submitted text (Enter on non-empty input).
    Submit {
        text: String,
        images: Vec<PastedImage>,
    },
    /// Ctrl+D on empty input — quit.
    Quit,
    /// Esc with no dropdown visible.
    Escape,
    /// Ctrl+C.
    Interrupt,
    /// Ctrl+L — clear screen.
    ClearScreen,
    /// No visible change needed.
    Nothing,
}

enum ReasoningClick {
    Transcript(usize),
    Live,
}

/// Interactive REPL that reads user input, streams agent responses, and renders
/// events in real time.
///
/// Ratatui owns the alternate screen. The transcript scrolls above a fixed
/// input area, including while responses and tool progress are streaming.
///
/// # Lifecycle
///
/// 1. Create with [`Repl::new`] and a [`ReplConfig`].
/// 2. Optionally call [`Repl::print_greeting`] to show a welcome banner.
/// 3. Loop:
///    - Call [`Repl::read_input`] to get user text (blocks until Enter or Ctrl+D).
///    - Pass the text to your agent, then call [`Repl::stream_response`] with the
///      resulting event stream.
/// 4. Break the loop on [`ReplEvent::Quit`].
///
/// # Cancellation
///
/// The [`CancellationToken`] passed to [`Repl::stream_response`] is stored so
/// callers can wire external cancellation into the in-flight agent turn.
/// Pressing Ctrl+C while typing simply clears the current input.
///
/// # Example
///
/// ```rust,ignore
/// use flashmind_tui::{Repl, ReplConfig, ReplEvent};
/// use tokio_util::sync::CancellationToken;
///
/// let mut repl = Repl::new(ReplConfig::default());
/// repl.print_greeting()?;
///
/// loop {
///     match repl.read_input()? {
///         ReplEvent::UserInput(text, _images) => {
///             let cancel = CancellationToken::new();
///             let stream = agent.start(&mut conversation, cancel.clone(), AgentInput::user(&text), None);
///             repl.stream_response(cancel, stream).await?;
///         }
///         ReplEvent::Quit => break,
///     }
/// }
/// ```
pub struct Repl<'a> {
    config: ReplConfig,
    textarea: TextArea<'a>,
    renderer: EventRenderer,
    spinner: Spinner,
    cancel_token: Option<CancellationToken>,
    tui: term::Tui,
    transcript_layout: TranscriptLayout,
    transcript_scroll: TranscriptScroll,
    pending_scroll_top: Option<usize>,
    transcript_search: Option<String>,
    search_match: Option<usize>,
    search_pending: bool,
    search_backwards: bool,
    transcript_area: Rect,
    selection: Option<TranscriptSelection>,
    screen_selection: ScreenSelection,
    selecting_screen: bool,
    pending_screen_copy: Option<String>,
    pending_reasoning_click: Option<ReasoningClick>,
    copy_pending: bool,
    dragging_scrollbar: bool,
    live_output: Vec<Line<'static>>,
    last_usage: Option<TokenUsage>,
    status: Option<StatusInfo>,
    /// Past user inputs, most recent last.
    history: Vec<String>,
    /// Index into `history` while the user is browsing with Up/Down.
    /// `None` means not currently browsing history.
    history_index: Option<usize>,
    /// Saves the in-progress input when the user starts browsing history,
    /// so it can be restored when they navigate past the end.
    history_draft: String,
    /// The input is a shell command: entered by typing `!` on an empty line.
    shell_mode: bool,
    /// Active slash-command autocomplete dropdown, if any.
    dropdown: Option<Dropdown>,
    /// `@mention` state: byte offset where the current `@` token starts, plus
    /// whether the dropdown is currently driven by mentions (vs slash commands).
    mention: Option<MentionState>,
    /// Byte offset where the slash-command argument starts while the
    /// dropdown shows argument candidates.
    argument_start: Option<usize>,
    /// Images pasted via Ctrl+V, keyed by atomic token id (see [`TextArea::insert_image_token`]).
    pasted_images: HashMap<usize, PastedImage>,
    /// Long pasted texts (>10 lines) keyed by atomic token id
    /// (see [`TextArea::insert_pasted_text_token`]).
    pasted_texts: HashMap<usize, String>,
    /// Stashed prompts (Ctrl+S to stash, Ctrl+P to pop). Session-only.
    stash: Vec<String>,
    /// Animated pet companion rendered beside the textarea, if enabled.
    pet: Option<PetWidget>,
    /// Toast message shown briefly in the input title bar.
    toast: Option<(String, u32)>,
    /// Monotonic tick counter for toast expiry and pet animation.
    tick: u32,
    /// Inputs submitted during streaming, queued for the next model request.
    pending_inputs: VecDeque<(String, Vec<PastedImage>)>,
    /// Activity label shown in the input bar (e.g. "thinking", "file_read").
    activity: Option<String>,
    /// Reverse-i-search state: `(query, match_index)`.
    reverse_search: Option<(String, usize)>,
    /// Structured progress for spawned subagents.
    subagent_progress: SubagentProgress,
    subagent_source: Option<Arc<dyn subagents::SubagentSource>>,
    conversation_title: Option<String>,
    subagent_view: Option<subagents::SubagentView>,
}

impl<'a> Repl<'a> {
    /// Create a new REPL with the given configuration.
    ///
    /// If [`ReplConfig::history_file`] is set and the file exists, previously
    /// saved entries are loaded (up to 1000 most recent).
    pub fn new(config: ReplConfig) -> Self {
        let history = config
            .history_file
            .as_deref()
            .map(load_history)
            .unwrap_or_default();
        let mut renderer = EventRenderer::new();
        renderer.set_show_usage(config.show_usage);
        Self {
            config,
            textarea: TextArea::default(),
            renderer,
            spinner: Spinner::new(),
            cancel_token: None,
            tui: term::Tui::new(),
            transcript_layout: TranscriptLayout::default(),
            transcript_scroll: TranscriptScroll::default(),
            pending_scroll_top: None,
            transcript_search: None,
            search_match: None,
            search_pending: false,
            search_backwards: false,
            transcript_area: Rect::default(),
            selection: None,
            screen_selection: ScreenSelection::default(),
            selecting_screen: false,
            pending_screen_copy: None,
            pending_reasoning_click: None,
            copy_pending: false,
            dragging_scrollbar: false,
            live_output: Vec::new(),
            last_usage: None,
            status: None,
            history,
            history_index: None,
            history_draft: String::new(),
            shell_mode: false,
            dropdown: None,
            mention: None,
            argument_start: None,
            pasted_images: HashMap::new(),
            pasted_texts: HashMap::new(),
            stash: Vec::new(),
            pet: None,
            toast: None,
            tick: 0,
            pending_inputs: VecDeque::new(),
            activity: None,
            reverse_search: None,
            subagent_progress: SubagentProgress::new(),
            subagent_source: None,
            conversation_title: None,
            subagent_view: None,
        }
    }

    pub fn with_tui(mut self, tui: &term::Tui) -> Self {
        self.tui = tui.clone();
        // A fresh lock cannot be poisoned; if it were, the transcript would
        // only start in the wrong mode until the next toggle.
        let _ = self.tui.set_tools_expanded(self.renderer.expand_tools());
        self
    }

    pub fn clear_transcript(&mut self) -> io::Result<()> {
        self.tui.clear_transcript()?;
        self.live_output.clear();
        self.transcript_layout = TranscriptLayout::default();
        self.transcript_scroll = TranscriptScroll::default();
        self.pending_scroll_top = None;
        self.transcript_search = None;
        self.search_match = None;
        self.selection = None;
        self.screen_selection.clear();
        self.selecting_screen = false;
        self.pending_screen_copy = None;
        self.pending_reasoning_click = None;
        self.dragging_scrollbar = false;
        Ok(())
    }

    /// Print the greeting message (if configured) and flush stdout.
    ///
    /// Call this once at startup before entering the main loop.  Does nothing
    /// if [`ReplConfig::greeting`] is `None`.
    pub fn print_greeting(&mut self) -> io::Result<()> {
        if let Some(greeting) = &self.config.greeting {
            self.println(Line::from(Span::styled(greeting.clone(), styles::S_AGENT)))?;
        }
        Ok(())
    }

    /// Record the token usage of the most recent turn.
    pub fn set_usage(&mut self, prompt_tokens: u32, completion_tokens: u32) {
        self.last_usage = Some(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..TokenUsage::default()
        });
    }

    /// Get the last token usage (if any) from the most recent turn.
    pub fn last_usage(&self) -> Option<&TokenUsage> {
        self.last_usage.as_ref()
    }

    /// Update the structured status shown on the right side of the title bar.
    pub fn set_status(&mut self, status: StatusInfo) {
        self.status = Some(status);
    }

    /// Update the session cost without replacing the other status fields.
    pub fn set_cost(&mut self, cost: Decimal) {
        self.status.get_or_insert_default().cost = Some(cost);
    }

    /// Set an activity label shown in the input bar (e.g. "thinking").
    pub fn set_activity(&mut self, label: impl Into<String>) {
        self.activity = Some(label.into());
    }

    /// Clear the activity indicator.
    pub fn clear_activity(&mut self) {
        self.activity = None;
    }

    /// Replace the set of slash commands available for autocomplete and hints.
    pub fn set_available_commands(&mut self, commands: Vec<CommandInfo>) {
        self.config.available_commands = commands;
    }

    /// Set whether reasoning blocks are expanded (full text) or collapsed
    /// (one-line summary), including blocks already in the transcript.
    pub fn set_expand_reasoning(&mut self, expand: bool) -> io::Result<()> {
        self.tui.set_reasoning_expanded(expand)?;
        self.renderer.set_expand_reasoning(expand);
        self.transcript_layout.invalidate();
        self.selection = None;
        self.search_match = None;
        Ok(())
    }

    /// Whether reasoning blocks are currently expanded.
    pub fn expand_reasoning(&self) -> bool {
        self.renderer.expand_reasoning()
    }

    /// Set whether full tool output is shown or collapsed.
    ///
    /// Applies to every tool block already in the transcript, the running
    /// tool, and tools that run later.
    pub fn set_expand_tools(&mut self, expand: bool) -> io::Result<()> {
        self.renderer.set_expand_tools(expand);
        self.tui.set_tools_expanded(expand)?;
        self.transcript_layout = TranscriptLayout::default();
        let actions = self.renderer.tick_tool();
        apply_to_lines(&mut self.live_output, &actions, expand);
        Ok(())
    }

    /// Whether full tool output is currently shown.
    pub fn expand_tools(&self) -> bool {
        self.renderer.expand_tools()
    }

    /// Enable structured subagent progress display.
    ///
    /// When enabled, `SpawnedEvent` rendering is suppressed in the main
    /// output and a compact progress section is rendered above the input bar.
    pub fn enable_subagent_progress(&mut self) {
        self.renderer.set_suppress_spawned(true);
    }

    /// Enable the animated pet companion rendered beside the textarea.
    /// The pet reacts to agent events (Thinking/Coding/Searching/…) and falls
    /// asleep when idle.
    pub fn enable_pet(&mut self, kind: Pet) {
        self.pet = Some(PetWidget::new(kind));
    }

    /// Drain all stored long-paste texts (keyed by atomic token id).
    ///
    /// Long pastes are normally resolved inline at submit time, so callers do
    /// not need this; it is exposed for callers that want to attach the full
    /// text as a separate message part instead.
    pub fn take_pasted_texts(&mut self) -> HashMap<usize, String> {
        std::mem::take(&mut self.pasted_texts)
    }

    /// Map an [`AgentEvent`] to a pet activity state and advance the pet.
    fn update_pet_state(&mut self, event: &AgentEvent) {
        let Some(pet) = self.pet.as_mut() else {
            return;
        };
        let state = match event {
            AgentEvent::ReasoningDelta(_) => Some(PetState::Thinking),
            AgentEvent::TextDelta(_) => Some(PetState::Reading),
            AgentEvent::ToolStart { name, .. } => Some(match name.as_str() {
                "file_read" | "read_lines" | "glob" | "grep" | "list_models"
                | "web_search_read" | "web_crawl" | "web_scrape" | "brave_search"
                | "firecrawl_search" | "sqlite_query" => PetState::Searching,
                "file_write" | "file_delete" | "str_replace" | "str_replace_regex"
                | "image_gen" | "image_edit" | "video_gen" => PetState::Coding,
                "exec" => PetState::Running,
                _ => PetState::Thinking,
            }),
            AgentEvent::Done(_) | AgentEvent::Cancelled | AgentEvent::Error(_) => {
                Some(PetState::Idle)
            }
            _ => None,
        };
        if let Some(s) = state {
            pet.set_state(s);
        }
    }

    /// Advance the pet animation frame, expire any stale toast, and return
    /// whether the input bar needs redrawing because of it.
    fn tick_animations(&mut self) -> bool {
        self.tick = self.tick.saturating_add(1);
        self.spinner.advance();
        if let Some(pet) = self.pet.as_mut() {
            pet.tick();
        }
        // Expire toast after TOAST_TICKS ticks.
        let toast_expired = if let Some((_, t)) = self.toast
            && self.tick.saturating_sub(t) >= TOAST_TICKS
        {
            self.toast = None;
            true
        } else {
            false
        };
        // Redraw if the pet is animating, a toast is still shown, or a toast
        // just expired (so it gets cleared from the bar).
        self.subagent_view.is_some() || self.pet.is_some() || self.toast.is_some() || toast_expired
    }
}

/// Number of animation ticks (~80ms each) a toast remains visible.
const TOAST_TICKS: u32 = 40;
