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
use base64::Engine;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::transcript::{TranscriptLayout, TranscriptScroll};
use futures::{Stream, StreamExt};
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, Event, KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::layout::{Position, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Padding, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use tokio_util::sync::CancellationToken;
use unicode_width::UnicodeWidthStr;

use flashmind_types::AgentEvent;
use flashmind_types::llm::TokenUsage;

use super::dropdown::Dropdown;
use super::pet::{Pet, PetState, PetWidget};
use super::spinner::Spinner;
use super::textarea::{AtomicToken, TextArea};
use crate::event_render::{EventRenderer, RenderAction, fit_tool_line, truncate_display};
use crate::styles;
use crate::term;
use crate::widgets::subagent_progress::SubagentProgress;

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
    /// Whether to show token usage (prompt↑ completion↓) in the input title bar.
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
    transcript_search: Option<String>,
    search_match: Option<usize>,
    search_pending: bool,
    search_backwards: bool,
    transcript_area: Rect,
    footer_row: u16,
    transcript_buffer: Buffer,
    selection: Option<ScreenSelection>,
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
    /// Active slash-command autocomplete dropdown, if any.
    dropdown: Option<Dropdown>,
    /// `@mention` state: byte offset where the current `@` token starts, plus
    /// whether the dropdown is currently driven by mentions (vs slash commands).
    mention: Option<MentionState>,
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
    /// Inputs submitted during streaming, queued for injection between turns.
    pending_inputs: VecDeque<(String, Vec<PastedImage>)>,
    /// Activity label shown in the input bar (e.g. "thinking", "file_read").
    activity: Option<String>,
    /// Reverse-i-search state: `(query, match_index)`.
    reverse_search: Option<(String, usize)>,
    /// Structured progress for spawned subagents.
    subagent_progress: SubagentProgress,
}

/// `@mention` autocompletion state.
#[derive(Clone, Debug)]
struct MentionState {
    /// Byte offset in the full text where the `@` token begins.
    token_start: usize,
    /// Byte offset of the cursor when the mention was last evaluated.
    cursor: usize,
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
        Self {
            config,
            textarea: TextArea::default(),
            renderer: EventRenderer::new(),
            spinner: Spinner::new(),
            cancel_token: None,
            tui: term::Tui::new(),
            transcript_layout: TranscriptLayout::default(),
            transcript_scroll: TranscriptScroll::default(),
            transcript_search: None,
            search_match: None,
            search_pending: false,
            search_backwards: false,
            transcript_area: Rect::default(),
            footer_row: 0,
            transcript_buffer: Buffer::empty(Rect::default()),
            selection: None,
            dragging_scrollbar: false,
            live_output: Vec::new(),
            last_usage: None,
            status: None,
            history,
            history_index: None,
            history_draft: String::new(),
            dropdown: None,
            mention: None,
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
        }
    }

    pub fn with_tui(mut self, tui: &term::Tui) -> Self {
        self.tui = tui.clone();
        self
    }

    pub fn clear_transcript(&mut self) -> io::Result<()> {
        self.tui.clear_transcript()?;
        self.renderer.clear_lines();
        self.live_output.clear();
        self.transcript_layout = TranscriptLayout::default();
        self.transcript_scroll = TranscriptScroll::default();
        self.transcript_search = None;
        self.search_match = None;
        self.selection = None;
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

    /// Set token usage to display in the input title bar.
    ///
    /// When [`ReplConfig::show_usage`] is `true`, the prompt tokens and completion
    /// tokens are shown as `N↑ M↓` next to the prompt prefix.
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
    /// (one-line summary).  Affects subsequent reasoning blocks.
    pub fn set_expand_reasoning(&mut self, expand: bool) {
        self.renderer.set_expand_reasoning(expand);
    }

    /// Whether reasoning blocks are currently expanded.
    pub fn expand_reasoning(&self) -> bool {
        self.renderer.expand_reasoning()
    }

    /// Set whether tool output previews are shown or collapsed.
    pub fn set_expand_tools(&mut self, expand: bool) {
        self.renderer.set_expand_tools(expand);
    }

    /// Whether tool output previews are currently shown.
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
                "exec" | "process" => PetState::Running,
                _ => PetState::Thinking,
            }),
            AgentEvent::Done(_) | AgentEvent::Error(_) => Some(PetState::Idle),
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
        self.pet.is_some() || self.toast.is_some() || toast_expired
    }

    /// Read a line of input from the user.
    ///
    /// Enters raw terminal mode and renders a [`TextArea`] widget
    /// that supports multi-line editing, cursor navigation, and word-aware commands
    /// (Alt+Left/Right for word movement, Ctrl+W for delete-word, etc.).
    ///
    /// Blocks until the user submits text (Enter) or quits (Ctrl+D on empty input).
    /// Ctrl+C cancels any in-flight agent turn if one is active, otherwise clears
    /// the current input.
    ///
    /// # Key bindings
    ///
    /// | Key | Action |
    /// |-----|--------|
    /// | Enter | Submit input |
    /// | Shift+Enter | Insert newline |
    /// | Ctrl+D (empty) | Quit |
    /// | Ctrl+C | Cancel agent turn / clear input |
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the terminal cannot be switched to raw mode or
    /// if reading keyboard events fails.
    pub fn read_input(&mut self) -> io::Result<ReplEvent> {
        if let Some((text, images)) = self.pending_inputs.pop_front() {
            self.echo_input(&text)?;
            self.draw_input()?;
            return Ok(ReplEvent::UserInput(text, images));
        }
        self.draw_input()?;
        let mut last_tick = std::time::Instant::now();
        loop {
            let timeout = Duration::from_millis(80).saturating_sub(last_tick.elapsed());
            let got_event = event::poll(timeout)?;
            if last_tick.elapsed() >= Duration::from_millis(80) {
                last_tick = std::time::Instant::now();
                if self.tick_animations() {
                    self.draw_input()?;
                }
            }
            if !got_event {
                continue;
            }
            match event::read()? {
                Event::Mouse(mouse) => self.handle_mouse(mouse),
                Event::Paste(text) => self.handle_paste(&text),
                Event::Resize(width, _) => {
                    self.renderer.set_width(width.saturating_sub(1) as usize)
                }
                Event::Key(key) => match self.process_key(key) {
                    KeyAction::Submit { text, images } => {
                        self.echo_input(&text)?;
                        self.draw_input()?;
                        return Ok(ReplEvent::UserInput(text, images));
                    }
                    KeyAction::Quit => {
                        self.clear_input_bar()?;
                        return Ok(ReplEvent::Quit);
                    }
                    KeyAction::Interrupt => {
                        if let Some(token) = self.cancel_token.take() {
                            token.cancel();
                        }
                        self.textarea.clear();
                        self.history_index = None;
                        self.history_draft.clear();
                        self.dropdown = None;
                    }
                    KeyAction::ClearScreen => self.clear_screen()?,
                    _ => {}
                },
                _ => continue,
            }
            self.draw_input()?;
        }
    }

    /// Stream agent events and render them to the terminal in real time.
    ///
    /// Drains the given stream of [`AgentEvent`]s, rendering each one through
    /// the [`EventRenderer`].  The stream typically comes from a single
    /// `Agent::run_turn` call (text/reasoning deltas) or a full
    /// `Agent::start` call (includes tool events and Done/Error).
    ///
    /// Progress is shown via the activity indicator in the input bar when
    /// set by the caller (e.g. during tool execution).
    ///
    /// The `cancel_token` is used to cancel the turn when the user presses
    /// Ctrl+C or Esc.
    ///
    /// Returns `Ok(true)` if the stream was cancelled by the user, `Ok(false)`
    /// if it completed naturally.
    pub async fn stream_events(
        &mut self,
        cancel_token: &CancellationToken,
        mut stream: impl Stream<Item = AgentEvent> + Unpin,
    ) -> io::Result<bool> {
        self.cancel_token = Some(cancel_token.clone());
        self.activity = Some(String::new());
        self.renderer.clear_running_tool();
        self.apply_actions(&[])?;
        self.draw_input()?;
        let mut tick = tokio::time::interval(Duration::from_millis(80));
        let mut cancelled = false;
        loop {
            tokio::select! {
                event = next_terminal_event() => {
                    cancelled |= self.handle_live_event(event?, cancel_token)?;
                }
                event = stream.next() => {
                    let Some(event) = event else {
                        self.activity = None;
                        self.draw_input()?;
                        break;
                    };
                    let done = matches!(event, AgentEvent::Done(_) | AgentEvent::Error(_));
                    self.emit_event(&event)?;
                    if done {
                        self.cancel_token = None;
                        self.activity = None;
                        self.subagent_progress.clear_finished();
                        self.draw_input()?;
                        break;
                    }
                }
                _ = tick.tick() => {
                    self.tick_animations();
                    let actions = self.renderer.tick_tool();
                    self.apply_actions(&actions)?;
                    self.draw_input()?;
                }
            }
        }
        Ok(cancelled)
    }

    /// Render a single event through the renderer and apply it to the terminal.
    ///
    /// Completed output enters the transcript. Running tool progress stays
    /// in the live region until its result arrives.
    pub fn emit_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        if let AgentEvent::Usage(usage) = event {
            self.last_usage = Some(usage.clone());
            if let Some(status) = self.status.as_mut()
                && let Some((_, total)) = status.context
                && total > 0
            {
                status.context = Some((usage.prompt_tokens, total));
            }
        }
        self.update_pet_state(event);
        if matches!(event, AgentEvent::SpawnedEvent { .. }) {
            self.subagent_progress.handle_event(event);
        }
        let actions = self.renderer.render(event);
        self.apply_actions(&actions)?;
        self.draw_input()
    }

    /// Finish a turn: flush remaining text, show elapsed time, and redraw.
    pub fn finish_turn(&mut self) -> io::Result<()> {
        self.renderer.clear_running_tool();
        let actions = self.renderer.render(&AgentEvent::Done(String::new()));
        self.apply_actions(&actions)?;
        self.activity = None;
        self.cancel_token = None;
        self.draw_input()
    }

    /// Release the chat screen before handing the terminal to a menu.
    pub fn clear_input_bar(&mut self) -> io::Result<()> {
        self.tui.suspend()
    }

    /// Run the UI event loop while a tool executes in the background.
    ///
    /// Keeps the input bar visible, animates the tool spinner, and handles
    /// key events (Ctrl+C/Esc to cancel, type-ahead input) until `fut`
    /// completes.  Returns the future's output.
    pub async fn run_tool_ui<T, F>(&mut self, cancel: &CancellationToken, fut: F) -> io::Result<T>
    where
        F: Future<Output = T>,
    {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        drop(sender);
        self.run_tool_ui_with_progress(cancel, String::new(), fut, receiver)
            .await
    }

    /// Like `run_tool_ui` but also drains a progress channel, emitting
    /// `ToolProgress` events so the renderer can show live output lines.
    pub async fn run_tool_ui_with_progress<T, F>(
        &mut self,
        cancel: &CancellationToken,
        tool_id: String,
        fut: F,
        progress_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    ) -> io::Result<T>
    where
        F: Future<Output = T>,
    {
        self.run_progress_ui(cancel, Some(tool_id), fut, progress_rx)
            .await
    }

    /// Run a task while preserving every output line in the transcript.
    pub async fn run_task_with_output<T, F>(
        &mut self,
        cancel: &CancellationToken,
        fut: F,
        output: tokio::sync::mpsc::UnboundedReceiver<String>,
    ) -> io::Result<T>
    where
        F: Future<Output = T>,
    {
        self.run_progress_ui(cancel, None, fut, output).await
    }

    fn render_progress_line(&mut self, tool_id: Option<&str>, line: String) -> io::Result<()> {
        match tool_id {
            Some(id) => self.emit_event(&AgentEvent::ToolProgress {
                id: id.to_string(),
                line,
            }),
            None => self.println(Line::from(line)),
        }
    }

    async fn run_progress_ui<T, F>(
        &mut self,
        cancel: &CancellationToken,
        tool_id: Option<String>,
        fut: F,
        mut progress_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    ) -> io::Result<T>
    where
        F: Future<Output = T>,
    {
        self.cancel_token = Some(cancel.clone());
        tokio::pin!(fut);
        self.draw_input()?;
        let mut tick = tokio::time::interval(Duration::from_millis(80));
        loop {
            tokio::select! {
                event = next_terminal_event() => { self.handle_live_event(event?, cancel)?; }
                result = &mut fut => {
                    progress_rx.close();
                    while let Some(line) = progress_rx.recv().await {
                        self.render_progress_line(tool_id.as_deref(), line)?;
                    }
                    self.cancel_token = None;
                    return Ok(result);
                }
                Some(line) = progress_rx.recv() => {
                    self.render_progress_line(tool_id.as_deref(), line)?;
                }
                _ = tick.tick() => {
                    self.tick_animations();
                    let actions = self.renderer.tick_tool();
                    self.apply_actions(&actions)?;
                    self.draw_input()?;
                }
            }
        }
    }

    /// Add a styled line to the transcript and redraw the visible region.
    pub fn println(&mut self, line: Line<'static>) -> io::Result<()> {
        self.renderer.push_line(line.clone());
        self.commit_lines(&[line])?;
        self.draw_input()
    }

    /// Set the renderer width for right-aligned elapsed times and separators.
    pub fn set_renderer_width(&mut self, width: usize) {
        self.renderer.set_width(width);
    }

    /// Mark the start of a new turn for elapsed time tracking.
    pub fn mark_turn_start(&mut self) {
        self.renderer.mark_turn_start();
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let width = chat_area(Rect::new(0, 0, term_w, 1))
            .width
            .saturating_sub(1);
        self.renderer.set_width(width as usize);
    }

    /// Replay a user message (echo styled input without the input bar).
    pub fn replay_user_input(&mut self, text: &str) -> io::Result<()> {
        self.echo_input(text)
    }

    /// Replay an agent event without the input bar (for session restore).
    pub fn replay_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        let actions = self.renderer.render(event);
        self.apply_actions(&actions)
    }

    /// Finish replaying a turn (flush remaining text buffer).
    pub fn replay_finish_turn(&mut self) -> io::Result<()> {
        let actions = self.renderer.render(&AgentEvent::Done(String::new()));
        self.apply_actions(&actions)
    }

    /// Whether the user submitted input during streaming.
    pub fn has_pending_input(&self) -> bool {
        !self.pending_inputs.is_empty()
    }

    /// Take the next pending input submitted during streaming.
    pub fn take_pending_input(&mut self) -> Option<(String, Vec<PastedImage>)> {
        self.pending_inputs.pop_front()
    }

    /// Stream agent response events and render them to the terminal in real time.
    ///
    /// This is a convenience wrapper around `stream_events` for callers using
    /// `Agent::start` which includes `Done`/`Error` events in the stream.
    ///
    /// Returns `Ok(())` when the stream completes normally or is exhausted.
    #[deprecated(note = "Use stream_events + turn loop instead")]
    pub async fn stream_response(
        &mut self,
        cancel_token: CancellationToken,
        stream: impl Stream<Item = AgentEvent> + Unpin,
    ) -> io::Result<()> {
        self.stream_events(&cancel_token, stream).await?;
        Ok(())
    }

    fn apply_actions(&mut self, actions: &[RenderAction]) -> io::Result<()> {
        self.renderer.record_actions(actions);
        update_live_output(&mut self.live_output, actions);
        if !self.renderer.tool_running() && !self.live_output.is_empty() {
            let lines = std::mem::take(&mut self.live_output);
            self.commit_lines(&lines)?;
        }
        Ok(())
    }

    fn commit_lines(&mut self, lines: &[Line<'static>]) -> io::Result<()> {
        self.tui.append_lines(lines)
    }

    fn handle_live_event(&mut self, event: Event, cancel: &CancellationToken) -> io::Result<bool> {
        let mut cancelled = false;
        match event {
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            Event::Paste(text) => self.handle_paste(&text),
            Event::Resize(width, _) => self.renderer.set_width(width.saturating_sub(1) as usize),
            Event::Key(key) => match self.process_key(key) {
                KeyAction::Submit { text, images } => self.pending_inputs.push_back((text, images)),
                KeyAction::Escape | KeyAction::Interrupt => {
                    cancel.cancel();
                    self.cancel_token = None;
                    cancelled = true;
                }
                KeyAction::ClearScreen => self.clear_screen()?,
                _ => {}
            },
            _ => {}
        }
        self.draw_input()?;
        Ok(cancelled)
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if matches!(
            mouse.kind,
            MouseEventKind::Down(_) | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            self.dragging_scrollbar = false;
        }
        let position = Position::new(mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left)
                if mouse.column == self.transcript_area.right()
                    && mouse.row >= self.transcript_area.y
                    && mouse.row < self.transcript_area.bottom() =>
            {
                self.dragging_scrollbar = true;
                self.selection = None;
                self.scrollbar_to(mouse.row);
            }
            MouseEventKind::Drag(MouseButton::Left) if self.dragging_scrollbar => {
                self.scrollbar_to(mouse.row)
            }
            MouseEventKind::Up(MouseButton::Left) if self.dragging_scrollbar => {
                self.dragging_scrollbar = false
            }
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                self.selection = None;
                self.transcript_scroll.mouse(mouse.kind);
            }
            MouseEventKind::Down(MouseButton::Left) if mouse.row == self.footer_row => {
                self.selection = None;
                self.transcript_scroll.latest();
            }
            MouseEventKind::Down(MouseButton::Left) if self.transcript_area.contains(position) => {
                self.transcript_scroll.following = false;
                self.selection = Some(ScreenSelection {
                    start: position,
                    end: position,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(selection) = self.selection.as_mut() {
                    selection.end = Position::new(
                        mouse.column.clamp(
                            self.transcript_area.x,
                            self.transcript_area.right().saturating_sub(1),
                        ),
                        mouse.row.clamp(
                            self.transcript_area.y,
                            self.transcript_area.bottom().saturating_sub(1),
                        ),
                    );
                }
            }
            MouseEventKind::Up(MouseButton::Left) => self.copy_selection(),
            _ => {}
        }
    }

    fn scrollbar_to(&mut self, row: u16) {
        let height = self.transcript_area.height.saturating_sub(1).max(1) as usize;
        let offset = (row.saturating_sub(self.transcript_area.y) as usize).min(height);
        self.transcript_scroll
            .jump_to(offset.saturating_mul(self.transcript_scroll.max) / height);
        if offset == height {
            self.transcript_scroll.latest();
        }
    }

    fn copy_selection(&mut self) {
        if let Some(selection) = &self.selection {
            let text = selection.text(&self.transcript_buffer);
            if !text.is_empty() {
                let toast = match copy_to_clipboard(&text) {
                    Ok(()) => "Copied selection".to_string(),
                    Err(error) => format!("Copy failed: {error}"),
                };
                self.toast = Some((toast, self.tick));
            }
        }
    }

    fn clear_screen(&mut self) -> io::Result<()> {
        self.tui.suspend()?;
        self.draw_input()
    }

    // -----------------------------------------------------------------------
    // Shared key processing

    /// Process a key event and return the resulting action. Handles all common
    /// key bindings (history, autocomplete, editing) so both `read_input` and
    /// `stream_response` share identical behaviour.
    fn process_key(&mut self, key: event::KeyEvent) -> KeyAction {
        self.dragging_scrollbar = false;
        if self.selection.is_some() && key.code == KeyCode::Esc {
            self.selection = None;
            return KeyAction::Redraw;
        }
        if key.code == KeyCode::Char('c')
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && self.selection.is_some()
        {
            self.copy_selection();
            return KeyAction::Redraw;
        }
        self.selection = None;
        if key.code == KeyCode::Char('f') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.transcript_search = Some(String::new());
            self.search_match = None;
            return KeyAction::Redraw;
        }
        if let Some(query) = self.transcript_search.as_mut() {
            match key.code {
                KeyCode::Esc => self.transcript_search = None,
                KeyCode::Enter => {
                    self.search_backwards = key.modifiers.contains(KeyModifiers::SHIFT);
                    self.search_pending = true;
                }
                KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    query.push(character);
                    self.search_match = None;
                    self.search_pending = true;
                }
                KeyCode::Backspace => {
                    query.pop();
                    self.search_match = None;
                    self.search_pending = true;
                }
                KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End => {
                    self.transcript_scroll.key(key);
                }
                _ => {}
            }
            return KeyAction::Redraw;
        }
        if self.transcript_scroll.key(key) {
            return KeyAction::Redraw;
        }
        // Reverse-i-search mode
        if self.reverse_search.is_some() {
            match key {
                // Ctrl+R — cycle to next match
                event::KeyEvent {
                    code: KeyCode::Char('r'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    let can_advance = self
                        .reverse_search
                        .as_ref()
                        .is_some_and(|(q, s)| self.find_reverse_match(q, s + 1).is_some());
                    if can_advance {
                        self.reverse_search.as_mut().unwrap().1 += 1;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Enter — accept match
                event::KeyEvent {
                    code: KeyCode::Enter,
                    ..
                } => {
                    self.reverse_search = None;
                    return KeyAction::Redraw;
                }
                // Esc / Ctrl+C — cancel, restore draft
                event::KeyEvent {
                    code: KeyCode::Esc, ..
                }
                | event::KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    self.reverse_search = None;
                    self.textarea.set_text(&self.history_draft);
                    return KeyAction::Redraw;
                }
                // Backspace — shrink query
                event::KeyEvent {
                    code: KeyCode::Backspace,
                    ..
                } => {
                    if let Some((ref mut query, ref mut skip)) = self.reverse_search {
                        query.pop();
                        *skip = 0;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Printable char — extend query
                event::KeyEvent {
                    code: KeyCode::Char(ch),
                    modifiers,
                    ..
                } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                    if let Some((ref mut query, ref mut skip)) = self.reverse_search {
                        query.push(ch);
                        *skip = 0;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Anything else exits search and processes normally
                _ => {
                    self.reverse_search = None;
                }
            }
        }

        match key {
            // Ctrl+R — enter reverse search
            event::KeyEvent {
                code: KeyCode::Char('r'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if !self.history.is_empty() => {
                self.history_draft = self.textarea.text();
                self.reverse_search = Some((String::new(), 0));
                self.apply_reverse_search();
                KeyAction::Redraw
            }

            // Ctrl+D on empty — quit
            event::KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if self.textarea.is_empty() => KeyAction::Quit,

            // Ctrl+C — interrupt
            event::KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => KeyAction::Interrupt,

            // Enter
            event::KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } => {
                // Accept dropdown selection
                if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                    && let Some(dropdown) = &self.dropdown
                    && !dropdown.is_empty()
                    && let Some(value) = dropdown.selected_value()
                {
                    let cmd = format!("{value} ");
                    self.textarea.set_text(&cmd);
                    self.dropdown = None;
                    return KeyAction::Redraw;
                }
                // Shift/Alt+Enter — insert newline
                if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
                    self.textarea.insert_newline();
                    self.dropdown = None;
                    return KeyAction::Redraw;
                }
                if self.textarea.is_empty() {
                    return KeyAction::Nothing;
                }
                // Submit
                let (text, images) = self.resolve_tokens();
                self.history.push(text.clone());
                self.history_index = None;
                self.history_draft.clear();
                if let Some(path) = &self.config.history_file {
                    append_history(path, &text);
                }
                self.textarea.clear();
                self.dropdown = None;
                self.mention = None;
                KeyAction::Submit { text, images }
            }

            // Tab — accept dropdown (slash command or @mention)
            event::KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                if let Some(value) = self
                    .dropdown
                    .as_ref()
                    .filter(|d| !d.is_empty())
                    .and_then(|d| d.selected_value())
                {
                    if let Some(state) = self.mention.take() {
                        let text = self.textarea.text();
                        let cursor = self.textarea.cursor_byte_offset();
                        let token_end = cursor.min(text.len());
                        let trigger = text.as_bytes().get(state.token_start).copied();
                        let replacement = if trigger == Some(b'#') {
                            // #search result: "file:line: content" → @file:line
                            let file_line = value.split(": ").next().unwrap_or(value);
                            format!("@{file_line} ")
                        } else {
                            format!("@{value} ")
                        };
                        let new_text = format!(
                            "{}{}{}",
                            &text[..state.token_start],
                            replacement,
                            &text[token_end..]
                        );
                        self.textarea.set_text(&new_text);
                        let new_cursor = state.token_start + replacement.len();
                        self.textarea.set_cursor_byte_offset(new_cursor);
                    } else {
                        // Slash command: replace the whole input.
                        let cmd = format!("{value} ");
                        self.textarea.set_text(&cmd);
                    }
                    self.dropdown = None;
                    KeyAction::Redraw
                } else {
                    KeyAction::Nothing
                }
            }

            // Esc — dismiss dropdown or signal escape
            event::KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                if self.dropdown.is_some() {
                    self.dropdown = None;
                    self.mention = None;
                    KeyAction::Redraw
                } else {
                    KeyAction::Escape
                }
            }

            // Up — dropdown nav / history / textarea
            event::KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if let Some(ref mut dropdown) = self.dropdown
                    && !dropdown.is_empty()
                {
                    dropdown.handle_key(key);
                    return KeyAction::Redraw;
                }
                if self.textarea.cursor().0 == 0 && !self.history.is_empty() {
                    match self.history_index {
                        None => {
                            self.history_draft = self.textarea.text();
                            let idx = self.history.len() - 1;
                            self.history_index = Some(idx);
                            self.textarea.set_text(&self.history[idx]);
                        }
                        Some(idx) if idx > 0 => {
                            let new_idx = idx - 1;
                            self.history_index = Some(new_idx);
                            self.textarea.set_text(&self.history[new_idx]);
                        }
                        _ => {}
                    }
                } else {
                    self.textarea.input(key);
                }
                KeyAction::Redraw
            }

            // Down — dropdown nav / history / textarea
            event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if let Some(ref mut dropdown) = self.dropdown
                    && !dropdown.is_empty()
                {
                    dropdown.handle_key(key);
                    return KeyAction::Redraw;
                }
                if let Some(idx) = self.history_index {
                    if idx + 1 >= self.history.len() {
                        self.history_index = None;
                        let draft = self.history_draft.clone();
                        self.textarea.set_text(&draft);
                    } else {
                        let new_idx = idx + 1;
                        self.history_index = Some(new_idx);
                        self.textarea.set_text(&self.history[new_idx]);
                    }
                } else {
                    self.textarea.input(key);
                }
                KeyAction::Redraw
            }

            // Ctrl+L — clear screen
            event::KeyEvent {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => KeyAction::ClearScreen,

            // Ctrl+V — paste image (or fall through to textarea for text paste)
            event::KeyEvent {
                code: KeyCode::Char('v'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                if let Some(img) = grab_clipboard_image() {
                    let id = self.textarea.insert_image_token();
                    self.pasted_images.insert(id, img);
                } else {
                    self.textarea.input(key);
                    self.update_autocomplete();
                }
                KeyAction::Redraw
            }

            // Ctrl+S — stash current input
            event::KeyEvent {
                code: KeyCode::Char('s'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let text = self.textarea.text();
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    self.stash.push(text);
                    self.textarea.clear();
                    self.prune_removed_tokens();
                    let n = self.stash.len();
                    self.toast = Some((format!("Stashed {n}"), self.tick));
                }
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Ctrl+P — pop stashed input into the textarea
            event::KeyEvent {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                match self.stash.pop() {
                    Some(text) => {
                        let remaining = self.stash.len();
                        self.textarea.set_text(&text);
                        self.prune_removed_tokens();
                        self.toast = Some((format!("Popped ({remaining} left)"), self.tick));
                    }
                    None => {
                        self.toast = Some(("stash empty".to_string(), self.tick));
                    }
                }
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Ctrl+K — copy textarea contents to the system clipboard
            event::KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let text = self.textarea.text();
                if !text.is_empty() {
                    match copy_to_clipboard(&text) {
                        Ok(()) => {
                            self.toast =
                                Some((format!("Copied {} chars", text.chars().count()), self.tick));
                        }
                        Err(e) => {
                            self.toast = Some((format!("Copy failed: {e}"), self.tick));
                        }
                    }
                }
                KeyAction::Redraw
            }

            // Ctrl+J — insert newline
            event::KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.textarea.insert_newline();
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Everything else — textarea
            other => {
                self.textarea.input(other);
                self.update_autocomplete();
                KeyAction::Redraw
            }
        }
    }

    /// Find the nth history entry (searching backwards) containing `query`.
    /// Returns the index into `self.history`.
    fn find_reverse_match(&self, query: &str, skip: usize) -> Option<usize> {
        if query.is_empty() {
            return self.history.len().checked_sub(skip + 1);
        }
        let query_lower = query.to_lowercase();
        self.history
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.to_lowercase().contains(&query_lower))
            .nth(skip)
            .map(|(i, _)| i)
    }

    /// Apply the current reverse search state to the textarea.
    fn apply_reverse_search(&mut self) {
        if let Some((ref query, skip)) = self.reverse_search
            && let Some(idx) = self.find_reverse_match(query, skip)
        {
            self.textarea.set_text(&self.history[idx]);
        }
    }

    /// Prune image/paste backing stores for atomic tokens removed by edits.
    /// Called after every textarea edit via [`update_autocomplete`].
    fn prune_removed_tokens(&mut self) {
        for id in self.textarea.drain_removed_token_ids() {
            self.pasted_images.remove(&id);
            self.pasted_texts.remove(&id);
        }
    }

    /// Resolve atomic tokens at submit time: expand long-paste placeholders to
    /// their full text, strip `[image #N]` labels, and collect the referenced
    /// images in textual order. Clears the backing maps.
    fn resolve_tokens(&mut self) -> (String, Vec<PastedImage>) {
        let lines = self.textarea.lines();
        let tokens: Vec<AtomicToken> = self.textarea.atomic_tokens().to_vec();

        let mut out = String::new();
        let mut image_ids_in_order: Vec<usize> = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            if li > 0 {
                out.push('\n');
            }
            let mut line_toks: Vec<&AtomicToken> = tokens.iter().filter(|t| t.line == li).collect();
            line_toks.sort_by_key(|t| t.start);
            let mut pos = 0;
            for tok in line_toks {
                out.push_str(&line[pos..tok.start]);
                if let Some(full) = self.pasted_texts.get(&tok.id) {
                    out.push_str(full);
                } else if self.pasted_images.contains_key(&tok.id) {
                    image_ids_in_order.push(tok.id);
                    // strip the `[image #N]` label from the text
                } else {
                    // Orphaned label with no backing data — keep verbatim.
                    out.push_str(&line[tok.start..tok.end]);
                }
                pos = tok.end;
            }
            out.push_str(&line[pos..]);
        }

        let images: Vec<PastedImage> = image_ids_in_order
            .iter()
            .filter_map(|id| self.pasted_images.get(id).cloned())
            .collect();

        let cleaned = out.trim().to_string();
        let labeled = if images.is_empty() {
            cleaned
        } else {
            let tags: Vec<String> = (1..=images.len())
                .map(|i| format!("[image #{i}]"))
                .collect();
            let suffix = tags.join(" ");
            if cleaned.is_empty() {
                suffix
            } else {
                format!("{cleaned}\n{suffix}")
            }
        };

        self.pasted_images.clear();
        self.pasted_texts.clear();
        (labeled, images)
    }

    /// Handle a bracketed-paste event: prefer a clipboard image, otherwise
    /// insert the pasted text — collapsing long pastes (>10 lines) into an
    /// atomic placeholder backed by `pasted_texts`.
    fn handle_paste(&mut self, text: &str) {
        if let Some(query) = self.transcript_search.as_mut() {
            query.push_str(text);
            self.search_match = None;
            self.search_pending = true;
            return;
        }
        if let Some(img) = grab_clipboard_image() {
            let id = self.textarea.insert_image_token();
            self.pasted_images.insert(id, img);
        } else {
            let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
            let line_count = normalized.lines().count();
            if line_count > 10 {
                let id = self.textarea.insert_pasted_text_token(line_count);
                self.pasted_texts.insert(id, normalized);
            } else {
                self.textarea.insert_str(&normalized);
                self.update_autocomplete();
            }
        }
        self.prune_removed_tokens();
    }

    /// Update the autocomplete dropdown based on current textarea content.
    ///
    /// Shows matching slash commands when the input starts with `/` and is a
    /// single line.  Hides the dropdown otherwise.
    fn update_autocomplete(&mut self) {
        self.prune_removed_tokens();
        let text = self.textarea.text();
        let cursor = self.textarea.cursor_byte_offset();

        // --- @mention file completion (takes priority when active) ---
        if let Some(provider) = self.config.mention_provider.clone()
            && let Some(state) = detect_mention(&text, cursor)
        {
            let query = &text[state.token_start + 1..state.cursor.min(text.len())];
            let candidates = provider.complete(query);
            if candidates.is_empty() {
                self.dropdown = None;
                self.mention = None;
            } else {
                match &mut self.dropdown {
                    Some(dd) => dd.set_candidates(candidates),
                    None => self.dropdown = Some(Dropdown::new("", candidates)),
                }
                self.mention = Some(state);
            }
            return;
        }

        // --- #content search ---
        if let Some(provider) = self.config.mention_provider.clone()
            && let Some(state) = detect_content_search(&text, cursor)
        {
            let query = &text[state.token_start + 1..state.cursor.min(text.len())];
            if query.len() >= 3 {
                let candidates = provider.search_content(query);
                if candidates.is_empty() {
                    self.dropdown = None;
                    self.mention = None;
                } else {
                    match &mut self.dropdown {
                        Some(dd) => dd.set_candidates(candidates),
                        None => self.dropdown = Some(Dropdown::new("", candidates)),
                    }
                    self.mention = Some(state);
                }
            } else {
                self.dropdown = None;
                self.mention = None;
            }
            return;
        }

        // Not in a mention or search token: clear any lingering state.
        self.mention = None;

        // --- slash-command completion ---
        if !text.starts_with('/')
            || text.contains('\n')
            || self.config.available_commands.is_empty()
        {
            self.dropdown = None;
            self.update_command_hint();
            return;
        }
        let prefix = text.trim_end();
        let candidates: Vec<String> = self
            .config
            .available_commands
            .iter()
            .filter(|cmd| cmd.name.starts_with(prefix) && cmd.name != prefix)
            .map(|cmd| cmd.name.clone())
            .collect();
        if candidates.is_empty() {
            self.dropdown = None;
        } else {
            match &mut self.dropdown {
                Some(dd) => dd.set_candidates(candidates),
                None => self.dropdown = Some(Dropdown::new("", candidates)),
            }
        }
        self.update_command_hint();
    }

    /// Refresh the dim ghost-text suffix and input block title based on the
    /// current input. When the first line starts with `/`, matches it against
    /// [`available_commands`](ReplConfig::available_commands) and shows the
    /// remaining name + args hint as a ghost suffix, with the command
    /// description in the block title. Clears the suffix otherwise.
    fn update_command_hint(&mut self) {
        let first = self.textarea.lines().first().cloned().unwrap_or_default();
        let multi_line = self.textarea.lines().len() > 1;

        let Some(rest) = first.strip_prefix('/') else {
            self.textarea.set_ghost_suffix("");
            return;
        };

        let (name, after_name) = match rest.split_once(' ') {
            Some((n, a)) => (n, Some(a)),
            None => (rest, None),
        };

        let matched = self
            .config
            .available_commands
            .iter()
            .find(|c| c.name == format!("/{name}"))
            .or_else(|| {
                if name.is_empty() {
                    None
                } else {
                    let prefix = format!("/{name}");
                    self.config
                        .available_commands
                        .iter()
                        .find(|c| c.name.starts_with(&prefix))
                }
            });

        let Some(entry) = matched else {
            self.textarea.set_ghost_suffix("");
            return;
        };

        let full_name = &entry.name;
        let args_hint = &entry.args_hint;

        // Ghost text: remaining name chars + args hint. Suppress on multi-line.
        let ghost = if multi_line {
            String::new()
        } else if after_name.is_none() {
            let remaining = &full_name[name.len() + 1..]; // +1 for '/'
            match (remaining.is_empty(), args_hint.is_empty()) {
                (true, true) => String::new(),
                (true, false) => format!(" {args_hint}"),
                (false, true) => remaining.to_string(),
                (false, false) => format!("{remaining} {args_hint}"),
            }
        } else if after_name == Some("") && !args_hint.is_empty() {
            args_hint.to_string()
        } else {
            String::new()
        };
        self.textarea.set_ghost_suffix(&ghost);
    }

    // -----------------------------------------------------------------------
    // Drawing helpers

    fn input_title_spans(&mut self) -> Vec<Span<'static>> {
        if let Some((query, _)) = &self.reverse_search {
            return vec![
                Span::styled(" reverse-i-search: ", styles::S_DIM),
                Span::styled(format!("{query} "), styles::S_USER),
            ];
        }
        let mut spans = vec![Span::styled(
            format!(" {} ", self.config.prompt),
            styles::S_ACCENT,
        )];
        if let Some(label) = &self.activity {
            let ch = self.spinner.tick();
            if label.is_empty() {
                spans.push(Span::styled(format!("{ch} "), styles::S_AGENT));
            } else {
                spans.push(Span::styled(format!("{ch} {label} "), styles::S_AGENT));
            }
        }
        if !self.pasted_images.is_empty() {
            let attached: Vec<usize> = self
                .textarea
                .atomic_tokens()
                .iter()
                .filter(|t| self.pasted_images.contains_key(&t.id))
                .map(|t| t.id)
                .collect();
            if !attached.is_empty() {
                let label = attached
                    .iter()
                    .map(|i| format!("#{i}"))
                    .collect::<Vec<_>>()
                    .join(",");
                spans.push(Span::styled(
                    format!("[{label}] "),
                    ratatui::style::Style::default().fg(ratatui::style::Color::Magenta),
                ));
            }
        }
        if let Some((toast, _)) = &self.toast {
            spans.push(Span::styled(
                format!("{toast} "),
                ratatui::style::Style::default().fg(ratatui::style::Color::Green),
            ));
        }

        spans
    }

    fn input_block(&mut self, width: u16) -> Block<'static> {
        let title = Line::from(self.input_title_spans());
        let available = (width as usize).saturating_sub(title.width() + 4);
        let mut block = Block::default()
            .borders(Borders::TOP)
            .border_style(styles::S_DIMMER)
            .title(title)
            .padding(Padding::new(1, 1, 0, 0));
        if let Some(status) = &self.status {
            let spans = status.to_spans(available);
            if !spans.is_empty() {
                block = block.title(Line::from(spans).right_aligned());
            }
        }
        block
    }

    fn input_height(&self, width: u16) -> u16 {
        // The textarea shares the terminal width with an optional pet column.
        // Compute the inner width the way `draw_input` does so wrapping agrees
        // with what is actually rendered (mismatch undercounts wrapped rows
        // when a pet is enabled, clipping the bottom of the input).
        let pet_enabled = self.pet.is_some() && width > 34;
        let pet_cols: u16 = if pet_enabled { 14 } else { 0 };
        let ta_width = width.saturating_sub(pet_cols).max(1);
        let inner_w = ta_width.saturating_sub(2).max(1); // left+right padding
        // +1 for top border
        let content = self.textarea.visual_line_count(inner_w) as u16 + 1;
        // The pet companion is 3 rows tall; when enabled, reserve at least
        // 3 rows + 1 border so it isn't clipped vertically.
        let min = if pet_enabled { 4 } else { 3 };
        let max = self.config.max_input_height.saturating_add(1).max(1);
        content.clamp(min.min(max), max)
    }

    fn echo_input(&mut self, text: &str) -> io::Result<()> {
        let lines = user_message_lines(text, &self.config.prompt);
        for line in &lines {
            self.renderer.push_line(line.clone());
        }
        self.commit_lines(&lines)
    }

    fn dynamic_lines(&self, width: u16) -> Vec<Line<'static>> {
        let mut lines = self.live_output.clone();
        let reasoning = self.renderer.pending_reasoning();
        if !reasoning.is_empty() {
            #[cfg(feature = "markdown")]
            let preview = crate::markdown_render::render_preview_to_lines(reasoning, width);
            #[cfg(not(feature = "markdown"))]
            let preview: Vec<Line<'static>> = reasoning
                .lines()
                .map(|line| Line::from(line.to_owned()))
                .collect();
            lines.extend(preview.into_iter().map(|mut line| {
                for span in &mut line.spans {
                    span.style = span.style.patch(styles::S_DIM);
                }
                line
            }));
        }
        let partial = self.renderer.pending_text();
        if !partial.is_empty() {
            #[cfg(feature = "markdown")]
            lines.extend(crate::markdown_render::render_preview_to_lines(
                partial, width,
            ));
            #[cfg(not(feature = "markdown"))]
            lines.extend(partial.lines().map(|line| Line::from(line.to_owned())));
        }
        for (text, images) in &self.pending_inputs {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_owned(),
                    styles::S_USER_ECHO.add_modifier(ratatui::style::Modifier::DIM),
                )));
            }
            if !images.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("[{} images]", images.len()),
                    styles::S_DIM,
                )));
            }
            lines.push(Line::default());
        }
        lines.extend(self.subagent_progress.lines(width));
        lines
    }

    fn draw_input(&mut self) -> io::Result<()> {
        let tui = self.tui.clone();
        tui.draw(|frame, history| self.render_input_frame(frame, history))
    }

    fn render_input_frame(&mut self, frame: &mut Frame<'_>, history: &[Line<'static>]) {
        let area = chat_area(frame.area());
        if area.is_empty() {
            return;
        }
        self.footer_row = area.bottom().saturating_sub(1);
        self.transcript_area = Rect::default();
        self.renderer
            .set_width(area.width.saturating_sub(1) as usize);
        self.textarea.set_placeholder_text(&self.config.placeholder);
        let pet_cols = if self.pet.is_some() && area.width > 34 {
            14
        } else {
            0
        };
        let block = self.input_block(area.width.saturating_sub(pet_cols));
        self.textarea.set_block(block);
        let dropdown_lines = self
            .dropdown
            .as_ref()
            .map(|dropdown| {
                dropdown.lines_for_height(area.width, (area.height.saturating_sub(3) / 2) as usize)
            })
            .unwrap_or_default();
        let dropdown_height = dropdown_lines.len() as u16;
        let input_height = self
            .input_height(area.width)
            .min(area.height.saturating_sub(dropdown_height))
            .max(1);
        let extra_height = area
            .height
            .saturating_sub(input_height + dropdown_height + 1);
        if extra_height > 0 {
            let body = Rect::new(area.x, area.y, area.width.saturating_sub(1), extra_height);
            let width = body.width.max(1);
            let anchor = (!self.transcript_scroll.following
                && self.transcript_layout.width() != width)
                .then(|| self.transcript_layout.window(self.transcript_scroll.top, 1));
            self.transcript_layout.measure(history, width);
            let dynamic: Vec<Line<'static>> = self
                .dynamic_lines(width)
                .iter()
                .map(|line| fit_tool_line(line, width as usize))
                .collect();
            let dynamic_rows = if dynamic.is_empty() {
                0
            } else {
                Paragraph::new(dynamic.clone())
                    .wrap(Wrap { trim: false })
                    .line_count(width)
            };
            self.transcript_scroll
                .update(self.transcript_layout.rows() + dynamic_rows, body.height);
            if let Some(anchor) = anchor
                && anchor.entries.start < history.len()
            {
                self.transcript_scroll.jump_to(
                    self.transcript_layout.entry_row(anchor.entries.start)
                        + anchor.row_offset as usize,
                );
            }
            if self.search_pending {
                self.search_pending = false;
                if let Some(query) = self
                    .transcript_search
                    .as_deref()
                    .filter(|query| !query.is_empty())
                {
                    if let Some(index) = find_transcript_match(
                        history,
                        query,
                        self.search_match,
                        self.search_backwards,
                    ) {
                        self.search_match = Some(index);
                        self.transcript_scroll
                            .jump_to(self.transcript_layout.entry_row(index));
                    } else {
                        self.toast = Some(("No matches".into(), self.tick));
                    }
                }
            }
            let window = self
                .transcript_layout
                .window(self.transcript_scroll.top, body.height as usize);
            let mut visible: Vec<Line<'static>> = history[window.entries.clone()]
                .iter()
                .map(|line| fit_tool_line(line, width as usize))
                .collect();
            if let Some(index) = self
                .search_match
                .filter(|index| self.transcript_search.is_some() && window.entries.contains(index))
            {
                for span in &mut visible[index - window.entries.start].spans {
                    span.style = span.style.patch(styles::S_ACCENT);
                }
            }
            if self.transcript_scroll.top + body.height as usize > self.transcript_layout.rows() {
                visible.extend(dynamic);
            }
            frame.render_widget(
                Paragraph::new(visible)
                    .wrap(Wrap { trim: false })
                    .scroll((window.row_offset, 0)),
                body,
            );
            self.transcript_area = body;
            self.transcript_buffer = Buffer::empty(body);
            for y in body.y..body.bottom() {
                for x in body.x..body.right() {
                    self.transcript_buffer[(x, y)] = frame.buffer_mut()[(x, y)].clone();
                    if self
                        .selection
                        .as_ref()
                        .is_some_and(|selection| selection.contains(Position::new(x, y)))
                    {
                        frame.buffer_mut()[(x, y)].set_style(styles::S_ACCENT);
                    }
                }
            }
            // Ratatui adds the viewport length to the number of scroll positions.
            let mut scrollbar = ScrollbarState::new(self.transcript_scroll.max + 1)
                .position(self.transcript_scroll.top)
                .viewport_content_length(body.height as usize);
            if self.transcript_scroll.max > 0 {
                frame.render_stateful_widget(
                    Scrollbar::new(ScrollbarOrientation::VerticalRight)
                        .begin_symbol(None)
                        .end_symbol(None),
                    Rect::new(area.x, area.y, area.width, extra_height),
                    &mut scrollbar,
                );
            }
        }
        let input_area = Rect::new(
            area.x,
            area.y + extra_height,
            area.width.saturating_sub(pet_cols),
            input_height,
        );
        self.textarea.ensure_cursor_visible(
            input_area.width.saturating_sub(2).max(1),
            input_area.height.saturating_sub(1).max(1),
        );
        frame.render_widget(&self.textarea, input_area);
        if let Some(pet) = &self.pet
            && pet_cols > 0
        {
            let pet_area = Rect::new(input_area.right(), input_area.y, pet_cols, input_height);
            pet.render(frame.buffer_mut(), pet_area);
        }
        if dropdown_height > 0 {
            frame.render_widget(
                Paragraph::new(dropdown_lines),
                Rect::new(area.x, input_area.bottom(), area.width, dropdown_height),
            );
        }
        if area.height > 2 {
            let hint = if let Some(query) = &self.transcript_search {
                format!("Find: {query} | Enter: next | Shift+Enter: previous | Esc: close")
            } else if self.transcript_scroll.following {
                "Enter send  |  /help  |  PgUp scroll  |  Ctrl+F find".into()
            } else {
                "Reading history | Ctrl+End: follow latest | Ctrl+F: find".into()
            };
            let hint = truncate_display(&hint, area.width as usize);
            let mut footer = vec![Span::styled(hint.clone(), styles::S_DIM)];
            if self.config.show_usage
                && let Some(usage) = &self.last_usage
            {
                let tokens = format!(
                    "in {}  out {}",
                    usage.prompt_tokens, usage.completion_tokens
                );
                let used = hint.width() + tokens.width();
                if used + 2 <= area.width as usize {
                    footer.push(Span::raw(" ".repeat(area.width as usize - used)));
                    footer.push(Span::styled(tokens, styles::S_DIM));
                }
            }
            frame.render_widget(
                Line::from(footer),
                Rect::new(area.x, area.bottom() - 1, area.width, 1),
            );
        }
        if let Some(cursor) = self.textarea.cursor_screen_pos(input_area) {
            frame.set_cursor_position(cursor);
        }
    }
}

fn chat_area(area: Rect) -> Rect {
    let margin = if area.width >= 60 { 2 } else { 0 };
    Rect::new(
        area.x + margin,
        area.y,
        area.width.saturating_sub(margin * 2),
        area.height,
    )
}

fn user_message_lines(text: &str, prompt: &str) -> Vec<Line<'static>> {
    let mut lines = vec![Line::default()];
    for (index, text) in text.split('\n').enumerate() {
        let prefix = if index == 0 {
            format!("{prompt} ")
        } else {
            " ".repeat(prompt.width() + 1)
        };
        lines.push(Line::from(vec![
            Span::styled(prefix, styles::S_ACCENT),
            Span::raw(text.to_owned()),
        ]));
    }
    lines.push(Line::default());
    lines
}

// Poll input without holding a background reader across menu transitions.
async fn next_terminal_event() -> io::Result<Event> {
    loop {
        if event::poll(Duration::ZERO)? {
            return event::read();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

struct ScreenSelection {
    start: Position,
    end: Position,
}

impl ScreenSelection {
    fn bounds(&self) -> (Position, Position) {
        if (self.start.y, self.start.x) <= (self.end.y, self.end.x) {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    fn contains(&self, position: Position) -> bool {
        let (start, end) = self.bounds();
        (start.y, start.x) <= (position.y, position.x) && (position.y, position.x) <= (end.y, end.x)
    }

    fn text(&self, buffer: &Buffer) -> String {
        let (start, end) = self.bounds();
        let mut lines = Vec::new();
        for y in start.y..=end.y {
            let mut line = String::new();
            let mut x = buffer.area.x;
            while x < buffer.area.right() {
                let position = Position::new(x, y);
                if !buffer.area.contains(position) {
                    break;
                }
                let symbol = buffer[(x, y)].symbol();
                let footprint = (symbol.width() as u16).max(1);
                if self.contains(position) || self.contains(Position::new(x + footprint - 1, y)) {
                    line.push_str(symbol);
                }
                x += footprint;
            }
            lines.push(line.trim_end().to_string());
        }
        lines.join("\n")
    }
}

fn find_transcript_match(
    lines: &[Line<'static>],
    query: &str,
    current: Option<usize>,
    backwards: bool,
) -> Option<usize> {
    if lines.is_empty() || query.is_empty() {
        return None;
    }
    let query = query.to_lowercase();
    for step in 0..lines.len() {
        let index = match current {
            Some(index) if backwards => {
                (index + lines.len() - (step + 1) % lines.len()) % lines.len()
            }
            Some(index) => (index + step + 1) % lines.len(),
            None if backwards => lines.len() - step - 1,
            None => step,
        };
        let text: String = lines[index]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        if text.to_lowercase().contains(&query) {
            return Some(index);
        }
    }
    None
}

fn update_live_output(lines: &mut Vec<Line<'static>>, actions: &[RenderAction]) {
    for action in actions {
        match action {
            RenderAction::Append(line) => lines.push(line.clone()),
            RenderAction::ReplaceTool {
                erase_count,
                lines: replacement,
            } => {
                lines.truncate(lines.len().saturating_sub(*erase_count));
                lines.extend(replacement.iter().cloned());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// History persistence helpers

/// Maximum number of history entries kept in memory and on disk.
const MAX_HISTORY: usize = 1000;

/// Number of animation ticks (~80ms each) a toast remains visible.
const TOAST_TICKS: u32 = 40;

/// Load history entries from a file.
///
/// Each line in the file represents one entry.  Literal newlines within an entry
/// are stored as the two-character sequence `\n`.  Returns at most the
/// [`MAX_HISTORY`] most recent entries.  Returns an empty `Vec` if the file does
/// not exist or cannot be read.
fn load_history(path: &Path) -> Vec<String> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let entries: Vec<String> = contents
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.replace("\\n", "\n"))
        .collect();
    if entries.len() > MAX_HISTORY {
        entries[entries.len() - MAX_HISTORY..].to_vec()
    } else {
        entries
    }
}

/// Append a single history entry to the file.
///
/// Literal newlines in `entry` are escaped as `\\n` so each entry occupies
/// exactly one file line.  The file (and parent directories) are created if they
/// do not exist.  Errors are silently ignored — history persistence is
/// best-effort.
///
/// To keep the file from growing without bound, once it exceeds
/// `2 * MAX_HISTORY` lines it is compacted in place to the most recent
/// [`MAX_HISTORY`] entries.  Compaction is amortized — it runs roughly once
/// every `MAX_HISTORY` appends rather than on every call.
fn append_history(path: &Path, entry: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let escaped = entry.replace('\n', "\\n");
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = writeln!(file, "{escaped}");
    drop(file);

    compact_history(path);
}

/// Rewrite the history file to the most recent [`MAX_HISTORY`] entries when it
/// has grown past `2 * MAX_HISTORY` lines.  Best-effort: any IO error leaves the
/// existing file untouched.
fn compact_history(path: &Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
    if lines.len() <= 2 * MAX_HISTORY {
        return;
    }
    let kept = &lines[lines.len() - MAX_HISTORY..];
    // Write to a sibling temp file then rename, so a crash mid-write can't
    // truncate the live history.
    let tmp = path.with_extension("tmp");
    let mut body = kept.join("\n");
    body.push('\n');
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

// ---------------------------------------------------------------------------
// Clipboard helpers

/// Copy text to the system clipboard. Returns an error message on failure.
///
/// Falls back to OSC 52 for remote terminals without a local clipboard.
pub(crate) fn copy_to_clipboard(text: &str) -> Result<(), String> {
    #[cfg(not(target_os = "android"))]
    if arboard::Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text))
        .is_ok()
    {
        return Ok(());
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let mut stdout = io::stdout();
    write!(stdout, "\x1b]52;c;{encoded}\x07")
        .and_then(|_| stdout.flush())
        .map_err(|error| error.to_string())
}

/// Attempt to grab image data from the system clipboard.
///
/// On macOS, uses `osascript` to check for PNG data, then `pbpaste` isn't
/// suitable for binary data so we write an AppleScript that outputs base64.
/// Returns `None` if no image is present or on non-macOS platforms.
fn grab_clipboard_image() -> Option<PastedImage> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let tmp = std::env::temp_dir().join("flashmind_clip.png");
        let tmp_path = tmp.to_string_lossy().to_string();

        // Write clipboard PNG to a temp file, then base64-encode it.
        // Accessing the clipboard and writing the file both happen in the
        // outer osascript process, avoiding nested-subprocess permission
        // issues and fragile Unicode piping through sed/xxd.
        let script = format!(
            r#"try
    set imgData to the clipboard as «class PNGf»
    set fref to open for access POSIX file "{tmp_path}" with write permission
    set eof of fref to 0
    write imgData to fref
    close access fref
    set b64 to do shell script "base64 -i " & quoted form of "{tmp_path}" & " && rm -f " & quoted form of "{tmp_path}"
    return b64
on error
    return ""
end try"#
        );

        let output = Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .output()
            .ok()?;

        // Clean up in case the script errored after creating the file.
        let _ = std::fs::remove_file(&tmp);

        if !output.status.success() {
            return None;
        }

        let b64 = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if b64.is_empty() {
            return None;
        }

        Some(PastedImage {
            media_type: "image/png".to_string(),
            data: b64,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// @mention token detection (free function)

/// Detect an `@mention` token ending at `cursor` in `text`.
///
/// Returns the byte offset of the `@` if there is one at the start of the text
/// or preceded by ASCII whitespace, with no whitespace between it and the
/// cursor.  This keeps the mention "live" while the user types a path but
/// dismisses it once they space away.
fn detect_mention(text: &str, cursor: usize) -> Option<MentionState> {
    detect_token(text, cursor, '@')
}

fn detect_content_search(text: &str, cursor: usize) -> Option<MentionState> {
    detect_token(text, cursor, '#')
}

fn detect_token(text: &str, cursor: usize, trigger: char) -> Option<MentionState> {
    if cursor == 0 || cursor > text.len() {
        return None;
    }
    let before = &text[..cursor];
    let pos = before.rfind(trigger)?;
    if pos > 0 && !before.as_bytes()[pos - 1].is_ascii_whitespace() {
        return None;
    }
    if text[pos + 1..cursor].contains(char::is_whitespace) {
        return None;
    }
    Some(MentionState {
        token_start: pos,
        cursor,
    })
}

#[cfg(test)]
mod history_tests {
    use super::{MAX_HISTORY, append_history, load_history};
    use std::path::PathBuf;

    /// A unique temp file path that is removed on drop.
    struct TempHistory(PathBuf);

    impl TempHistory {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("flashmind-hist-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempHistory {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let _ = std::fs::remove_file(self.0.with_extension("tmp"));
        }
    }

    #[test]
    fn append_then_load_roundtrips_with_escaped_newlines() {
        let h = TempHistory::new("roundtrip");
        append_history(h.path(), "first");
        append_history(h.path(), "multi\nline");
        let loaded = load_history(h.path());
        assert_eq!(loaded, vec!["first".to_string(), "multi\nline".to_string()]);
    }

    #[test]
    fn append_compacts_when_file_exceeds_threshold() {
        let h = TempHistory::new("compact");
        // Append more than 2 * MAX_HISTORY entries; the file should be
        // compacted down to the most recent MAX_HISTORY.
        let total = 2 * MAX_HISTORY + 5;
        for i in 0..total {
            append_history(h.path(), &format!("entry {i}"));
        }
        let contents = std::fs::read_to_string(h.path()).unwrap();
        let line_count = contents.lines().filter(|l| !l.is_empty()).count();
        assert!(
            line_count <= 2 * MAX_HISTORY,
            "file should be compacted, has {line_count} lines"
        );
        let loaded = load_history(h.path());
        assert_eq!(loaded.len(), MAX_HISTORY);
        // The most-recent entry survives.
        assert_eq!(loaded.last().unwrap(), &format!("entry {}", total - 1));
    }
}

#[cfg(test)]
mod viewport_tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::{Terminal, TerminalOptions, Viewport};

    #[test]
    fn completed_tool_times_stay_at_the_right_edge_after_resizing() {
        let mut repl = Repl::new(ReplConfig::default());
        let label =
            "Run example checks with a description that needs truncation in a small terminal";
        repl.renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: label.into(),
        });
        let actions = repl.renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            success: true,
            elapsed_ms: 13,
            output: String::new(),
            sources: vec![],
        });
        let history = match &actions[0] {
            RenderAction::ReplaceTool { lines, .. } => lines.clone(),
            _ => panic!("a completed tool replaces its live row"),
        };
        assert!(history[0].to_string().contains(label));
        let mut terminal = Terminal::new(TestBackend::new(200, 12)).unwrap();
        for width in [200, 40, 100, 20, 200] {
            terminal.backend_mut().resize(width, 12);
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            assert_eq!(repl.transcript_layout.rows(), 1);
            let body = repl.transcript_area;
            let buffer = terminal.backend().buffer();
            let duration: String = (body.right() - 5..body.right())
                .map(|x| buffer[(x, body.y)].symbol())
                .collect();
            assert_eq!(duration, "13 ms");
            let row: String = (body.x..body.right())
                .map(|x| buffer[(x, body.y)].symbol())
                .collect();
            if width == 200 {
                assert!(row.contains(label));
            } else if width == 20 {
                assert!(row.contains('\u{2026}'));
            }
        }
    }

    #[test]
    fn chat_uses_the_terminal_width_and_footer_fields_do_not_overlap() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_status(StatusInfo {
            model: "provider/example-model".into(),
            thinking: Some(flashmind_types::ReasoningLevel::Medium),
            cost: Some(rust_decimal::Decimal::new(123, 4)),
            context: Some((25, 100)),
            git_branch: Some("example-branch".into()),
        });
        repl.set_usage(1234, 56);
        let history = vec![Line::from("example text ".repeat(35))];
        for width in [24, 40, 80, 200] {
            let mut terminal = Terminal::new(TestBackend::new(width, 20)).unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            let margin = if width >= 60 { 2 } else { 0 };
            assert_eq!(repl.transcript_area.width, width - margin * 2 - 1);
            let buffer = terminal.backend().buffer();
            let footer: String = (0..width).map(|x| buffer[(x, 19)].symbol()).collect();
            if width == 200 {
                assert!(footer.contains("Enter send"));
                assert!(footer.contains("in 1234  out 56"));
                for y in 0..20 {
                    for x in width - margin..width {
                        assert_eq!(buffer[(x, y)].symbol(), " ");
                    }
                }
            }
            let title = repl.input_title_spans();
            let status = repl
                .status
                .as_ref()
                .unwrap()
                .to_spans((width as usize).saturating_sub(Line::from(title).width() + 4));
            assert!(Line::from(status).width() <= width as usize);
        }
    }

    #[test]
    fn user_messages_use_the_prompt_without_a_label_or_background() {
        let lines = user_message_lines("First line\nSecond line", ">");
        assert_eq!(
            lines.iter().map(Line::to_string).collect::<Vec<_>>(),
            ["", "> First line", "  Second line", ""]
        );
        assert!(
            lines
                .iter()
                .flat_map(|line| &line.spans)
                .all(|span| span.style.bg.is_none())
        );
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn completed_reply_reflows_when_the_terminal_shrinks_and_grows() {
        let mut repl = Repl::new(ReplConfig::default());
        let text = "A completed response should reflow to the current terminal width. ".repeat(8);
        repl.renderer.set_width(30);
        let actions = repl.renderer.render(&AgentEvent::TextDelta(text.clone()));
        repl.renderer.record_actions(&actions);
        let actions = repl.renderer.flush();
        repl.renderer.record_actions(&actions);
        let history = repl.renderer.lines().to_vec();
        assert_eq!(
            history
                .iter()
                .filter(|line| !line.to_string().is_empty())
                .count(),
            1
        );
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let narrow_rows = repl.transcript_layout.rows();
        terminal.backend_mut().resize(200, 12);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.transcript_area.width, 195);
        assert!(repl.transcript_layout.rows() < narrow_rows);
        terminal.backend_mut().resize(40, 12);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.transcript_layout.rows(), narrow_rows);
        let cursor = terminal.get_cursor_position().unwrap();
        assert!(terminal.backend().buffer().area.contains(cursor));
    }

    #[test]
    fn live_preview_keeps_all_lines_of_an_open_paragraph() {
        let mut repl = Repl::new(ReplConfig::default());
        let actions = repl
            .renderer
            .render(&AgentEvent::TextDelta("First line\nsecond line".into()));
        assert!(actions.is_empty());
        let text = repl
            .dynamic_lines(40)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("First line"));
        assert!(text.contains("second line"));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn live_preview_shows_unclosed_code_and_expanded_reasoning() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.renderer.render(&AgentEvent::TextDelta(
            "```rust\nlet first = 1;\nlet second = 2;".into(),
        ));
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("let first = 1;"));
        assert!(text.contains("let second = 2;"));
        repl.renderer
            .render(&AgentEvent::ReasoningDelta("considering options".into()));
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("considering options"));
        repl.set_expand_reasoning(false);
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("considering options"));
    }

    #[test]
    fn input_cursor_stays_inside_frame_with_progress_and_wrapped_input() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str(&"example input ".repeat(80));
        repl.enable_pet(Pet::Cat);
        repl.live_output = (0..40)
            .map(|index| Line::from(format!("progress {index}")))
            .collect();
        for (width, height) in [(80, 12), (40, 8), (16, 5)] {
            let area = Rect::new(0, 4, width, height);
            let mut terminal = Terminal::with_options(
                TestBackend::new(width, height + 4),
                TerminalOptions {
                    viewport: Viewport::Fixed(area),
                },
            )
            .unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let cursor = terminal.get_cursor_position().unwrap();
            assert!(area.contains(cursor), "cursor {cursor:?} outside {area:?}");
            let screen: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(screen.contains("example"));
        }
    }

    #[test]
    fn history_scroll_stays_put_during_output_and_can_search() {
        let mut repl = Repl::new(ReplConfig::default());
        let mut history: Vec<Line<'static>> = (0..120)
            .map(|index| Line::from(format!("message {index}")))
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert!(repl.transcript_scroll.following);
        repl.process_key(event::KeyEvent::from(KeyCode::PageUp));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let top = repl.transcript_scroll.top;
        history.push(Line::from("new output"));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.transcript_scroll.top, top);
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(!screen.contains("new output"));
        repl.process_key(event::KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert!(repl.transcript_scroll.following);
        repl.process_key(event::KeyEvent::new(
            KeyCode::Char('f'),
            KeyModifiers::CONTROL,
        ));
        repl.handle_paste("message 10");
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.search_match, Some(10));
        assert!(!repl.transcript_scroll.following);
        assert!(repl.textarea.text().is_empty());
    }

    #[test]
    fn scrollbar_thumb_reaches_both_ends_and_moves_with_history() {
        use ratatui::symbols::scrollbar::DOUBLE_VERTICAL;

        let mut repl = Repl::new(ReplConfig::default());
        let history: Vec<Line<'static>> = (0..40)
            .map(|index| Line::from(format!("message {index}")))
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let body = repl.transcript_area;
        let thumb_rows = |terminal: &Terminal<TestBackend>| -> Vec<u16> {
            (body.y..body.bottom())
                .filter(|&row| {
                    terminal.backend().buffer()[(body.right(), row)].symbol()
                        == DOUBLE_VERTICAL.thumb
                })
                .collect()
        };
        assert_eq!(repl.transcript_scroll.top, repl.transcript_scroll.max);
        let bottom = thumb_rows(&terminal);
        assert_eq!(bottom.last().copied(), Some(body.bottom() - 1));

        repl.process_key(event::KeyEvent::from(KeyCode::PageUp));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let middle = thumb_rows(&terminal);
        assert!(middle[0] < bottom[0]);

        repl.process_key(event::KeyEvent::new(KeyCode::Home, KeyModifiers::CONTROL));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(thumb_rows(&terminal).first().copied(), Some(body.y));

        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: body.right(),
            row: body.bottom() - 1,
            modifiers: KeyModifiers::NONE,
        });
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert!(repl.transcript_scroll.following);
        assert_eq!(
            thumb_rows(&terminal).last().copied(),
            Some(body.bottom() - 1)
        );
    }

    #[test]
    fn scrollbar_drag_can_pause_and_resume_following() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.transcript_area = Rect::new(0, 0, 79, 20);
        repl.transcript_scroll.update(100, 20);
        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 79,
            row: 10,
            modifiers: KeyModifiers::NONE,
        });
        assert!(!repl.transcript_scroll.following);
        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 79,
            row: 19,
            modifiers: KeyModifiers::NONE,
        });
        assert!(repl.transcript_scroll.following);
        assert_eq!(repl.transcript_scroll.top, 80);
    }

    #[test]
    fn resize_keeps_the_same_history_entry_in_view() {
        let mut repl = Repl::new(ReplConfig::default());
        let history: Vec<Line<'static>> = (0..80)
            .map(|index| Line::from(format!("entry {index}: {}", "example ".repeat(30))))
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        repl.transcript_scroll
            .jump_to(repl.transcript_layout.entry_row(30));
        terminal.backend_mut().resize(40, 16);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let window = repl.transcript_layout.window(repl.transcript_scroll.top, 1);
        assert_eq!(window.entries.start, 30);
        assert!(!repl.transcript_scroll.following);
    }

    #[test]
    fn search_cycles_both_ways_and_selection_preserves_unicode() {
        let history = vec![
            Line::from("Example one"),
            Line::from("other"),
            Line::from("example two"),
        ];
        assert_eq!(
            find_transcript_match(&history, "EXAMPLE", None, false),
            Some(0)
        );
        assert_eq!(
            find_transcript_match(&history, "example", Some(0), false),
            Some(2)
        );
        assert_eq!(
            find_transcript_match(&history, "example", Some(0), true),
            Some(2)
        );
        let buffer = Buffer::with_lines(["one example", "two界text"]);
        let selection = ScreenSelection {
            start: Position::new(4, 0),
            end: Position::new(4, 1),
        };
        assert_eq!(selection.text(&buffer), "example\ntwo界");
        let reverse = ScreenSelection {
            start: selection.end,
            end: selection.start,
        };
        assert_eq!(reverse.text(&buffer), selection.text(&buffer));
    }

    #[test]
    fn tool_progress_replaces_live_lines_and_preserves_preceding_output() {
        let mut renderer = EventRenderer::new();
        let mut live = vec![Line::from("preceding answer")];
        let start = AgentEvent::ToolStart {
            name: "exec".into(),
            id: "call-1".into(),
            humanized: "Example command".into(),
        };
        update_live_output(&mut live, &renderer.render(&start));
        for index in 0..100 {
            let event = AgentEvent::ToolProgress {
                id: "call-1".into(),
                line: format!("progress {index}"),
            };
            update_live_output(&mut live, &renderer.render(&event));
            assert!(live.len() <= 13);
        }
        let result = AgentEvent::ToolResult {
            name: "exec".into(),
            id: "call-1".into(),
            output: "command completed".into(),
            success: true,
            elapsed_ms: 1,
            sources: Vec::new(),
        };
        update_live_output(&mut live, &renderer.render(&result));
        let text: String = live
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect();
        assert!(text.contains("preceding answer"));
        assert!(text.contains("command completed"));
        assert!(!text.contains("progress 99"));
        assert!(!renderer.tool_running());
    }
}
