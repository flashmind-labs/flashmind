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
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::EventStream;
use futures::{Stream, StreamExt};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::crossterm::{
    cursor::{MoveToColumn, MoveUp, Show},
    execute, queue,
    style::Print,
    terminal::{Clear, ClearType},
};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding};
use tokio_util::sync::CancellationToken;

use flashmind_types::AgentEvent;
use flashmind_types::llm::TokenUsage;

use super::dropdown::Dropdown;
use super::pet::{Pet, PetState, PetWidget};
use super::spinner::Spinner;
use super::textarea::{AtomicToken, TextArea};
use crate::event_render::{EventRenderer, RenderAction};
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
    fn to_spans(&self) -> Vec<Span<'static>> {
        use ratatui::style::{Color, Style};

        let mut spans = Vec::new();
        if !self.model.is_empty() {
            spans.push(Span::styled(format!(" {} ", self.model), styles::S_DIM));
        }
        if let Some(level) = self.thinking
            && level.is_on()
        {
            spans.push(Span::styled(
                format!("{level}  "),
                Style::default().fg(Color::Yellow),
            ));
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
            spans.push(Span::styled(
                format!("{formatted}  "),
                Style::default().fg(Color::Green),
            ));
        }
        if let Some((used, total)) = self.context
            && total > 0
        {
            let pct = (used as u64 * 100) / total as u64;
            let color = if pct >= 90 {
                Color::Red
            } else if pct >= 70 {
                Color::Yellow
            } else {
                Color::Reset
            };
            spans.push(Span::styled(
                format!("ctx: {pct}%  "),
                Style::default().fg(color),
            ));
        }
        if let Some(branch) = &self.git_branch
            && !branch.is_empty()
        {
            spans.push(Span::styled(
                format!(" {branch}  "),
                Style::default().fg(Color::Magenta),
            ));
        }
        spans
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
/// The REPL uses raw terminal mode while reading input and append-mode output
/// for agent responses.  It tracks the height of the last-drawn input widget so
/// it can erase and redraw it on each keystroke.
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
    /// Absolute terminal row of the widget top (for erase positioning).
    widget_top_row: Option<u16>,
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
    /// Visual row count of the last-drawn dynamic region (partial streaming
    /// text + queued inputs + subagent progress + input bar, and dropdown when
    /// applicable).  Used by cursor-relative erase on resize.
    last_dynamic_height: u16,
    /// Cursor row offset from the dynamic-region top at the last draw.
    /// `MoveUp` by this on resize reaches the reflowed region top.
    last_cursor_offset: u16,
    /// Terminal width used for the last draw.  The input bar's full-width top
    /// border reflows to `ceil(last_draw_width / new_width)` rows on shrink, so
    /// the erase must account for the extra rows to avoid ghost borders.
    last_draw_width: u16,
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
            widget_top_row: None,
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
            last_dynamic_height: 0,
            last_cursor_offset: 0,
            last_draw_width: 80,
        }
    }

    /// Print the greeting message (if configured) and flush stdout.
    ///
    /// Call this once at startup before entering the main loop.  Does nothing
    /// if [`ReplConfig::greeting`] is `None`.
    pub fn print_greeting(&mut self) -> io::Result<()> {
        if let Some(greeting) = &self.config.greeting {
            let mut stdout = io::stdout();
            let line = Line::from(Span::styled(greeting.clone(), styles::S_AGENT));
            term::print_line(&mut stdout, &line)?;
            self.renderer.push_line(line);
            stdout.flush()?;
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
        // Return queued input from a submission during streaming.
        if let Some((text, images)) = self.pending_inputs.pop_front() {
            let mut stdout = io::stdout();
            self.echo_input(&mut stdout, &text)?;
            return Ok(ReplEvent::UserInput(text, images));
        }

        let mut stdout = io::stdout();

        let _raw = RawModeGuard::enable()?;
        self.draw_input(&mut stdout)?;

        // Animation cadence: advance the pet frame and expire toasts while the
        // user is typing (not just during streaming). Without this the pet
        // never animates and toasts set by Ctrl+S/Ctrl+K never expire.
        let mut last_tick = std::time::Instant::now();

        loop {
            let timeout = Duration::from_millis(80).saturating_sub(last_tick.elapsed());
            let got_event = event::poll(timeout)?;
            if last_tick.elapsed() >= Duration::from_millis(80) {
                last_tick = std::time::Instant::now();
                if self.tick_animations() {
                    self.draw_input(&mut stdout)?;
                }
            }
            if !got_event {
                continue;
            }
            let ev = event::read()?;

            if let Event::Paste(text) = &ev {
                self.handle_paste(text);
                self.draw_input(&mut stdout)?;
                continue;
            }

            if let Event::Resize(w, _) = ev {
                self.renderer.set_width(w.saturating_sub(1) as usize);
                self.erase_dynamic_region(&mut stdout, w, false)?;
                // Force `draw_input` to query the cursor (rather than reuse the
                // now-stale absolute top row) so the bar re-anchors right after
                // the committed output at the new width.
                self.widget_top_row = None;
                self.draw_input(&mut stdout)?;
                continue;
            }

            let Event::Key(key) = ev else {
                continue;
            };

            match self.process_key(key) {
                KeyAction::Redraw => self.draw_input(&mut stdout)?,
                KeyAction::Submit { text, images } => {
                    self.erase_widget(&mut stdout)?;
                    self.echo_input(&mut stdout, &text)?;
                    return Ok(ReplEvent::UserInput(text, images));
                }
                KeyAction::Quit => {
                    self.erase_widget(&mut stdout)?;
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
                    self.draw_input(&mut stdout)?;
                }
                KeyAction::ClearScreen => {
                    execute!(
                        stdout,
                        crossterm::terminal::Clear(ClearType::All),
                        crossterm::cursor::MoveTo(0, 0)
                    )?;
                    self.widget_top_row = None;
                    self.draw_input(&mut stdout)?;
                }
                KeyAction::Escape | KeyAction::Nothing => {}
            }
        }
    }

    /// Stream agent events and render them to the terminal in real time.
    ///
    /// Drains the given stream of [`AgentEvent`]s, rendering each one through
    /// the [`EventRenderer`].  The stream typically comes from a single
    /// [`Agent::run_turn`] call (text/reasoning deltas) or a full
    /// [`Agent::start`] call (includes tool events and Done/Error).
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
        // A tool is never executing during an LLM stream (tools run separately,
        // between turns).  If the renderer still thinks one is running, it is
        // leaked state from a tool whose ToolResult was skipped (e.g. an inline
        // interrupt that resolved out-of-band).  Clear it so the spinner tick
        // can't re-print a phantom running-tool line against content that has
        // since scrolled away.
        self.renderer.clear_running_tool();
        let mut stdout = io::stdout();
        let mut tick_interval = tokio::time::interval(Duration::from_millis(80));
        let mut cancelled = false;

        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        self.renderer.set_width(term_w.saturating_sub(1) as usize);

        let _raw = RawModeGuard::enable()?;

        // Draw the input bar before starting EventStream (draw_input can query
        // cursor position; EventStream would conflict with that query).
        self.draw_input(&mut stdout)?;
        let mut input_bar_row = self.widget_top_row.unwrap_or(0);

        let mut key_stream = EventStream::new();

        loop {
            tokio::select! {
                biased;
                maybe_key = key_stream.next() => {
                    if let Some(Ok(ev)) = maybe_key {
                        match ev {
                            crossterm::event::Event::Key(key) => {
                                match self.process_key(key) {
                                    KeyAction::Redraw => {
                                        input_bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            input_bar_row,
                                        )?;
                                    }
                                    KeyAction::Submit { text, images } => {
                                        self.pending_inputs.push_back((text, images));
                                        input_bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            input_bar_row,
                                        )?;
                                    }
                                    KeyAction::Escape | KeyAction::Interrupt => {
                                        cancel_token.cancel();
                                        self.cancel_token = None;
                                        cancelled = true;
                                    }
                                    KeyAction::ClearScreen => {
                                        execute!(
                                            stdout,
                                            crossterm::terminal::Clear(ClearType::All),
                                            crossterm::cursor::MoveTo(0, 0)
                                        )?;
                                        input_bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            0,
                                        )?;
                                    }
                                    _ => {}
                                }
                            }
                            crossterm::event::Event::Resize(w, _) => {
                                input_bar_row =
                                    self.redraw_streaming_resize(&mut stdout, w)?;
                            }
                            crossterm::event::Event::Paste(text) => {
                                self.handle_paste(&text);
                                input_bar_row =
                                    self.draw_input_at_row(&mut stdout, input_bar_row)?;
                            }
                            _ => {}
                        }
                    }
                }
                maybe_event = stream.next() => {
                    match maybe_event {
                        Some(event) => {
                            if let AgentEvent::Usage(u) = &event {
                                self.last_usage = Some(u.clone());
                                // Live-update the context-usage % in the status
                                // bar as prompt tokens arrive mid-stream.
                                if let Some(status) = self.status.as_mut()
                                    && let Some((_, total)) = status.context
                                    && total > 0
                                {
                                    status.context = Some((u.prompt_tokens, total));
                                }
                            }
                            self.update_pet_state(&event);

                            if matches!(&event, AgentEvent::SpawnedEvent { .. }) {
                                self.subagent_progress.handle_event(&event);
                            }

                            let is_done =
                                matches!(&event, AgentEvent::Done(_) | AgentEvent::Error(_));
                            let is_text =
                                matches!(&event, AgentEvent::TextDelta(_));
                            let is_reasoning =
                                matches!(&event, AgentEvent::ReasoningDelta(_));
                            let is_spawned =
                                matches!(&event, AgentEvent::SpawnedEvent { .. });
                            // Show a “thinking” activity indicator while the
                            // model streams text or reasoning and no tool is
                            // running.  Cleared on Done/Error (below) and when a
                            // tool starts (execute_tools sets its own activity).
                            let activity_started = (is_text || is_reasoning)
                                && self.activity.is_none();
                            if activity_started {
                                self.activity = Some("thinking".to_string());
                            }
                            let actions = self.renderer.render(&event);
                            let is_usage = matches!(&event, AgentEvent::Usage(_));
                            let needs_update =
                                !actions.is_empty() || is_done || is_text
                                || activity_started || is_usage || is_spawned;

                            if needs_update {
                                Self::erase_at_row(&mut stdout, input_bar_row)?;

                                let (tw, _) = ratatui::crossterm::terminal::size()
                                    .unwrap_or((80, 24));
                                let delta = Self::actions_cursor_delta(&actions, tw);
                                self.renderer.record_actions(&actions);
                                Self::apply_actions(&mut stdout, &actions)?;
                                stdout.flush()?;

                                let new_row =
                                    (input_bar_row as i32 + delta).max(0) as u16;
                                let (_, th) = ratatui::crossterm::terminal::size()
                                    .unwrap_or((80, 24));
                                input_bar_row = new_row.min(th.saturating_sub(1));
                            }

                            if is_done {
                                self.cancel_token = None;
                                self.widget_top_row = None;
                                self.activity = None;
                                self.subagent_progress.clear_finished();
                                break;
                            }

                            if needs_update {
                                input_bar_row =
                                    self.draw_input_at_row(&mut stdout, input_bar_row)?;
                            }
                        }
                        None => {
                            self.activity = None;
                            Self::erase_at_row(&mut stdout, input_bar_row)?;
                            let _ =
                                self.draw_input_at_row(&mut stdout, input_bar_row)?;
                            stdout.flush()?;
                            break;
                        }
                    }
                }
                _ = tick_interval.tick() => {
                    let anim_changed = self.tick_animations();
                    if self.renderer.tool_running() {
                        Self::erase_at_row(&mut stdout, input_bar_row)?;
                        let actions = self.renderer.tick_tool();
                        let (tw, _) = ratatui::crossterm::terminal::size()
                            .unwrap_or((80, 24));
                        let delta = Self::actions_cursor_delta(&actions, tw);
                        self.renderer.record_actions(&actions);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        let new_row = (input_bar_row as i32 + delta).max(0) as u16;
                        input_bar_row =
                            self.draw_input_at_row(&mut stdout, new_row)?;
                    } else if self.activity.is_some()
                        || !self.subagent_progress.is_empty()
                        || anim_changed
                    {
                        Self::erase_at_row(&mut stdout, input_bar_row)?;
                        input_bar_row =
                            self.draw_input_at_row(&mut stdout, input_bar_row)?;
                    }
                }
            }
        }

        Ok(cancelled)
    }

    /// Render a single event through the renderer and apply it to the terminal.
    ///
    /// When the input bar is active (`widget_top_row` is set — i.e. during a
    /// turn loop), the bar is erased before rendering and redrawn afterwards so
    /// tool events appear above the bar rather than replacing it.
    pub fn emit_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        self.update_pet_state(event);
        let actions = self.renderer.render(event);
        if actions.is_empty() {
            return Ok(());
        }
        let mut stdout = io::stdout();

        if let Some(bar_row) = self.widget_top_row {
            Self::erase_at_row(&mut stdout, bar_row)?;
            let (tw, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            let delta = Self::actions_cursor_delta(&actions, tw);
            self.renderer.record_actions(&actions);
            Self::apply_actions(&mut stdout, &actions)?;
            stdout.flush()?;
            let new_row = (bar_row as i32 + delta).max(0) as u16;
            let (_, th) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            let clamped = new_row.min(th.saturating_sub(1));
            self.draw_input_at_row(&mut stdout, clamped)?;
        } else {
            self.renderer.record_actions(&actions);
            Self::apply_actions(&mut stdout, &actions)?;
            stdout.flush()?;
        }

        Ok(())
    }

    /// Finish a turn: erase the input bar if active, flush remaining text
    /// buffer, show elapsed time, and add a trailing blank line.
    pub fn finish_turn(&mut self) -> io::Result<()> {
        let done_event = AgentEvent::Done(String::new());
        let actions = self.renderer.render(&done_event);
        let mut stdout = io::stdout();

        if let Some(bar_row) = self.widget_top_row.take() {
            Self::erase_at_row(&mut stdout, bar_row)?;
        }

        if !actions.is_empty() {
            self.renderer.record_actions(&actions);
            Self::apply_actions(&mut stdout, &actions)?;
        }

        stdout.flush()
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
        tokio::pin!(fut);

        let _raw = RawModeGuard::enable()?;
        let mut stdout = io::stdout();
        let mut tick_interval = tokio::time::interval(Duration::from_millis(80));
        let mut key_stream = EventStream::new();

        let start_row = self.widget_top_row.unwrap_or_else(|| {
            let (_, h) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            h.saturating_sub(3)
        });
        let mut bar_row = self.draw_input_at_row(&mut stdout, start_row)?;

        loop {
            tokio::select! {
                biased;
                maybe_key = key_stream.next() => {
                    if let Some(Ok(ev)) = maybe_key {
                        match ev {
                            crossterm::event::Event::Key(key) => {
                                match self.process_key(key) {
                                    KeyAction::Submit { text, images } => {
                                        self.pending_inputs
                                            .push_back((text, images));
                                        bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            bar_row,
                                        )?;
                                    }
                                    KeyAction::Escape | KeyAction::Interrupt => {
                                        cancel.cancel();
                                    }
                                    KeyAction::Redraw => {
                                        bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            bar_row,
                                        )?;
                                    }
                                    KeyAction::ClearScreen => {
                                        execute!(
                                            stdout,
                                            Clear(ClearType::All),
                                            crossterm::cursor::MoveTo(0, 0)
                                        )?;
                                        bar_row = self.draw_input_at_row(
                                            &mut stdout,
                                            0,
                                        )?;
                                    }
                                    _ => {}
                                }
                            }
                            crossterm::event::Event::Resize(w, _) => {
                                bar_row = self.redraw_streaming_resize(&mut stdout, w)?;
                            }
                            crossterm::event::Event::Paste(text) => {
                                self.handle_paste(&text);
                                bar_row = self.draw_input_at_row(
                                    &mut stdout,
                                    bar_row,
                                )?;
                            }
                            _ => {}
                        }
                    }
                }
                result = &mut fut => {
                    return Ok(result);
                }
                _ = tick_interval.tick() => {
                    let anim_changed = self.tick_animations();
                    if self.renderer.tool_running() {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        let actions = self.renderer.tick_tool();
                        let (tw, _) =
                            ratatui::crossterm::terminal::size()
                                .unwrap_or((80, 24));
                        let delta =
                            Self::actions_cursor_delta(&actions, tw);
                        self.renderer.record_actions(&actions);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        let new_row =
                            (bar_row as i32 + delta).max(0) as u16;
                        bar_row = self.draw_input_at_row(
                            &mut stdout,
                            new_row,
                        )?;
                    } else if self.activity.is_some() || anim_changed {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        bar_row = self.draw_input_at_row(
                            &mut stdout,
                            bar_row,
                        )?;
                    }
                }
            }
        }
    }

    /// Like [`run_tool_ui`] but also drains a progress channel, emitting
    /// `ToolProgress` events so the renderer can show live output lines.
    pub async fn run_tool_ui_with_progress<T, F>(
        &mut self,
        cancel: &CancellationToken,
        tool_id: String,
        fut: F,
        mut progress_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    ) -> io::Result<T>
    where
        F: Future<Output = T>,
    {
        tokio::pin!(fut);

        let _raw = RawModeGuard::enable()?;
        let mut stdout = io::stdout();
        let mut tick_interval = tokio::time::interval(Duration::from_millis(80));
        let mut key_stream = EventStream::new();

        let start_row = self.widget_top_row.unwrap_or_else(|| {
            let (_, h) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            h.saturating_sub(3)
        });
        let mut bar_row = self.draw_input_at_row(&mut stdout, start_row)?;

        loop {
            tokio::select! {
                biased;
                maybe_key = key_stream.next() => {
                    if let Some(Ok(ev)) = maybe_key {
                        match ev {
                            crossterm::event::Event::Key(key) => {
                                match self.process_key(key) {
                                    KeyAction::Submit { text, images } => {
                                        self.pending_inputs.push_back((text, images));
                                        bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                                    }
                                    KeyAction::Escape | KeyAction::Interrupt => {
                                        cancel.cancel();
                                    }
                                    KeyAction::Redraw => {
                                        bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                                    }
                                    KeyAction::ClearScreen => {
                                        execute!(
                                            stdout,
                                            Clear(ClearType::All),
                                            crossterm::cursor::MoveTo(0, 0)
                                        )?;
                                        bar_row = self.draw_input_at_row(&mut stdout, 0)?;
                                    }
                                    _ => {}
                                }
                            }
                            crossterm::event::Event::Resize(w, _) => {
                                bar_row = self.redraw_streaming_resize(&mut stdout, w)?;
                            }
                            crossterm::event::Event::Paste(text) => {
                                self.handle_paste(&text);
                                bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                            }
                            _ => {}
                        }
                    }
                }
                result = &mut fut => {
                    progress_rx.close();
                    while let Some(line) = progress_rx.recv().await {
                        let event = AgentEvent::ToolProgress { id: tool_id.clone(), line };
                        let actions = self.renderer.render(&event);
                        if !actions.is_empty() {
                            Self::erase_at_row(&mut stdout, bar_row)?;
                            let (tw, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
                            let delta = Self::actions_cursor_delta(&actions, tw);
                            self.renderer.record_actions(&actions);
                            Self::apply_actions(&mut stdout, &actions)?;
                            stdout.flush()?;
                            bar_row = (bar_row as i32 + delta).max(0) as u16;
                            bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                        }
                    }
                    return Ok(result);
                }
                Some(line) = progress_rx.recv() => {
                    let event = AgentEvent::ToolProgress { id: tool_id.clone(), line };
                    let actions = self.renderer.render(&event);
                    if !actions.is_empty() {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        let (tw, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
                        let delta = Self::actions_cursor_delta(&actions, tw);
                        self.renderer.record_actions(&actions);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        bar_row = (bar_row as i32 + delta).max(0) as u16;
                        bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                    }
                }
                _ = tick_interval.tick() => {
                    let anim_changed = self.tick_animations();
                    if self.renderer.tool_running() {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        let actions = self.renderer.tick_tool();
                        let (tw, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
                        let delta = Self::actions_cursor_delta(&actions, tw);
                        self.renderer.record_actions(&actions);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        let new_row = (bar_row as i32 + delta).max(0) as u16;
                        bar_row = self.draw_input_at_row(&mut stdout, new_row)?;
                    } else if self.activity.is_some() || anim_changed {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        bar_row = self.draw_input_at_row(&mut stdout, bar_row)?;
                    }
                }
            }
        }
    }

    /// Print a styled line to stdout and record it for resize redraw.
    pub fn println(&mut self, line: Line<'static>) -> io::Result<()> {
        let mut stdout = io::stdout();
        term::print_line(&mut stdout, &line)?;
        self.renderer.push_line(line);
        stdout.flush()
    }

    /// Set the renderer width for right-aligned elapsed times and separators.
    pub fn set_renderer_width(&mut self, width: usize) {
        self.renderer.set_width(width);
    }

    /// Mark the start of a new turn for elapsed time tracking.
    pub fn mark_turn_start(&mut self) {
        self.renderer.mark_turn_start();
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        self.renderer.set_width(term_w.saturating_sub(1) as usize);
    }

    /// Replay a user message (echo styled input without the input bar).
    pub fn replay_user_input(&mut self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout();
        self.echo_input(&mut stdout, text)
    }

    /// Replay an agent event without the input bar (for session restore).
    pub fn replay_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        let actions = self.renderer.render(event);
        if actions.is_empty() {
            return Ok(());
        }
        self.renderer.record_actions(&actions);
        let mut stdout = io::stdout();
        Self::apply_actions(&mut stdout, &actions)?;
        stdout.flush()
    }

    /// Finish replaying a turn (flush remaining text buffer).
    pub fn replay_finish_turn(&mut self) -> io::Result<()> {
        let done_event = AgentEvent::Done(String::new());
        let actions = self.renderer.render(&done_event);
        if !actions.is_empty() {
            self.renderer.record_actions(&actions);
            let mut stdout = io::stdout();
            Self::apply_actions(&mut stdout, &actions)?;
            stdout.flush()?;
        }
        Ok(())
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
    /// This is a convenience wrapper around [`stream_events`] for callers using
    /// [`Agent::start`] which includes `Done`/`Error` events in the stream.
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

    /// Apply render actions — handles both appending and in-place tool replacement.
    fn apply_actions(stdout: &mut io::Stdout, actions: &[RenderAction]) -> io::Result<()> {
        for action in actions {
            match action {
                RenderAction::Append(line) => {
                    term::print_line(stdout, line)?;
                }
                RenderAction::ReplaceTool { erase_count, lines } => {
                    for _ in 0..*erase_count {
                        execute!(
                            stdout,
                            MoveUp(1),
                            MoveToColumn(0),
                            Clear(ClearType::CurrentLine),
                        )?;
                    }
                    for line in lines {
                        term::print_line(stdout, line)?;
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Streaming helpers

    /// Faithful viewport repaint for streaming-loop resize handlers.
    ///
    /// `erase_dynamic_region` + a bottom-anchored redraw (`draw_input_at_row`
    /// at `th - region_height`) avoids a flash but is fragile: when the
    /// conversation is taller than the viewport, the reflowed committed output
    /// extends below that bottom anchor, and `draw_input_at_row`'s
    /// `Clear(FromCursorDown)` then overwrites the most recent committed lines
    /// with the input bar — silently deleting streamed content (observed as a
    /// block of blank lines where text used to be).
    ///
    /// Streaming loops cannot query the cursor to re-anchor cursor-relatively
    /// the way [`read_input`](Self::read_input) does: the `EventStream` is
    /// concurrently reading stdin, so an `ESC[6n` probe would race with it.
    /// We therefore fall back to a full repaint: clear the viewport, reprint
    /// the visible tail of the recorded committed output
    /// ([`renderer.lines()`](EventRenderer::lines)), then draw the dynamic
    /// region (partial + queued inputs + subagent progress + input bar) flush
    /// against it at the bottom.  This re-renders a screenful of history (a
    /// brief flash) but the recorded line buffer is the source of truth, so no
    /// content is ever lost on resize.
    ///
    /// Returns the new top row of the input bar.
    fn redraw_streaming_resize(&mut self, stdout: &mut io::Stdout, width: u16) -> io::Result<u16> {
        self.renderer.set_width(width.saturating_sub(1) as usize);

        // Dynamic-region heights at the new width.  These must match what
        // `draw_input_at_row` is about to emit.  `extra` counts *visual*
        // (wrapped) rows so the reserved space matches the screen footprint of
        // the in-flight partial text.
        let partial = self.renderer.partial_text().to_owned();
        let extra: u16 = if partial.is_empty() {
            0
        } else {
            #[cfg(feature = "markdown")]
            {
                crate::markdown_render::render_to_lines(&partial)
                    .iter()
                    .map(|l| term::visual_height(l, width))
                    .sum()
            }
            #[cfg(not(feature = "markdown"))]
            {
                term::visual_height(&Line::from(partial.clone()), width)
            }
        };
        let queued: u16 = self
            .pending_inputs
            .iter()
            .map(|(text, imgs)| {
                let lines = text.split('\n').count() as u16;
                let img_lines = if imgs.is_empty() { 0 } else { 1 };
                lines + img_lines + 1
            })
            .sum();
        let progress: u16 = if self.subagent_progress.is_empty() {
            0
        } else {
            self.subagent_progress.lines(width).len() as u16
        };
        let input_h = self.input_height(width);
        let total = extra + queued + progress + input_h;

        let (_, th) = ratatui::crossterm::terminal::size().unwrap_or((width, 24));
        let avail = th.saturating_sub(total);

        // Clear the viewport, then bottom-align the committed tail against the
        // dynamic region so it sits flush above the input bar — no gap, no
        // overlap with the in-flight partial text.
        queue!(
            stdout,
            ratatui::crossterm::cursor::MoveTo(0, 0),
            Clear(ClearType::FromCursorDown),
        )?;

        let lines = self.renderer.lines();
        let mut rows_left = avail;
        let mut start = lines.len();
        for i in (0..lines.len()).rev() {
            let lh = term::visual_height(&lines[i], width);
            if lh > rows_left {
                break;
            }
            rows_left -= lh;
            start = i;
        }
        // `rows_left` is now the blank space left above the reprinted tail.
        queue!(stdout, ratatui::crossterm::cursor::MoveTo(0, rows_left))?;
        for line in &lines[start..] {
            term::print_line(stdout, line)?;
        }
        stdout.flush()?;

        let new_top = th.saturating_sub(total);
        self.draw_input_at_row(stdout, new_top)
    }

    /// Cursor-relative erase of the last-drawn dynamic region on resize.
    ///
    /// The terminal reflows scrollback on resize, invalidating the absolute
    /// row positions we track (`widget_top_row` / `input_bar_row`).  But the
    /// dynamic region's *content* is unchanged, so its post-reflow cursor
    /// offset (rows from region top to the textarea cursor) equals the offset
    /// computed at the new width.  `MoveUp` by that offset lands exactly on the
    /// reflowed region top; `Clear(FromCursorDown)` then wipes the old region
    /// in place — no full-screen clear, no history repaint, no flash.
    ///
    /// `border_extra` accounts for the input bar's full-width top border: a
    /// line of `last_draw_width` box-drawing chars reflows to
    /// `ceil(last_draw_width / new_width)` rows on shrink, while the offset
    /// computation counts it as one.  Adding the difference erases the smeared
    /// border remnant that would otherwise linger as a ghost bar.
    ///
    /// `streaming` selects which layout to measure: the streaming draw renders
    /// partial text + queued inputs + subagent progress + the input bar (no
    /// dropdown); the read-input draw renders the input bar + dropdown.  The
    /// measurement must match what the subsequent redraw will emit.
    ///
    /// Returns the new total dynamic height (for bottom-anchored repositioning
    /// in the streaming loops).  The caller redraws the region afterwards.
    fn erase_dynamic_region(
        &mut self,
        stdout: &mut io::Stdout,
        new_width: u16,
        streaming: bool,
    ) -> io::Result<u16> {
        let width = new_width.max(1);

        // Partial streaming text (streaming draw only).
        let extra_new: u16 = if streaming {
            let partial = self.renderer.partial_text();
            if partial.is_empty() {
                0
            } else {
                #[cfg(feature = "markdown")]
                {
                    crate::markdown_render::render_to_lines(partial)
                        .iter()
                        .map(|l| term::visual_height(l, width))
                        .sum()
                }
                #[cfg(not(feature = "markdown"))]
                {
                    term::visual_height(&Line::from(partial.to_owned()), width)
                }
            }
        } else {
            0
        };

        // Queued inputs (streaming draw only).
        let queued_new: u16 = if streaming {
            self.pending_inputs
                .iter()
                .map(|(text, imgs)| {
                    let lines = text.split('\n').count() as u16;
                    let img_lines = if imgs.is_empty() { 0 } else { 1 };
                    lines + img_lines + 1
                })
                .sum()
        } else {
            0
        };

        // Subagent progress (streaming draw only).
        let progress_new: u16 = if streaming && !self.subagent_progress.is_empty() {
            self.subagent_progress.lines(width).len() as u16
        } else {
            0
        };

        // Configure the textarea so input_height / cursor_screen_pos match the
        // subsequent redraw.
        self.textarea.set_placeholder_text(&self.config.placeholder);
        let block = self.input_block();
        self.textarea.set_block(block);
        let input_h_new = self.input_height(width);
        // Match the pet-aware textarea width used by draw_input / draw_input_at_row
        // so cursor wrapping agrees with how the bar is actually rendered.
        let pet_enabled = self.pet.is_some() && width > 34;
        let ta_w = if pet_enabled {
            width.saturating_sub(14).max(1)
        } else {
            width
        };
        let cy_new = self
            .textarea
            .cursor_screen_pos(ratatui::layout::Rect::new(0, 0, ta_w, input_h_new))
            .map(|(_, cy)| cy)
            .unwrap_or(0);

        // Dropdown (read-input draw only; rendered below the bar so it
        // contributes to the total height but not the cursor offset).
        let dropdown_part: u16 = if !streaming {
            if let Some(d) = &self.dropdown {
                1 + d.lines(width).len() as u16
            } else {
                0
            }
        } else {
            0
        };

        let total_new = extra_new + queued_new + progress_new + input_h_new + dropdown_part;
        let cursor_offset_new = extra_new + queued_new + progress_new + cy_new;

        // The full-width top border reflows to ceil(old_width / new_width)
        // rows on shrink; the offset counts it as one.  Erase the difference.
        let border_extra = self.last_draw_width.div_ceil(width).saturating_sub(1);
        let erase_up = cursor_offset_new.saturating_add(border_extra);

        if erase_up > 0 {
            queue!(
                stdout,
                MoveUp(erase_up),
                MoveToColumn(0),
                Clear(ClearType::FromCursorDown),
            )?;
            stdout.flush()?;
        }

        Ok(total_new)
    }

    /// Erase everything from the given row down without modifying `widget_top_row`.
    fn erase_at_row(stdout: &mut io::Stdout, row: u16) -> io::Result<()> {
        queue!(
            stdout,
            ratatui::crossterm::cursor::MoveTo(0, row),
            Clear(ClearType::FromCursorDown)
        )?;
        stdout.flush()
    }

    /// Draw the input bar at a known row (avoids cursor-position query, safe
    /// to call while `EventStream` is active). Returns the actual top row
    /// after accounting for terminal scroll.
    fn draw_input_at_row(&mut self, stdout: &mut io::Stdout, row: u16) -> io::Result<u16> {
        let (width, term_h) = ratatui::crossterm::terminal::size()?;

        queue!(
            stdout,
            ratatui::crossterm::cursor::MoveTo(0, row),
            Clear(ClearType::FromCursorDown)
        )?;

        // Render partial (uncommitted) streaming text above everything,
        // with inline markdown applied so formatting appears immediately.
        let partial = self.renderer.partial_text();
        let mut extra_lines: u16 = 0;
        if !partial.is_empty() {
            #[cfg(feature = "markdown")]
            let rendered = crate::markdown_render::render_to_lines(partial);
            #[cfg(not(feature = "markdown"))]
            let rendered = vec![Line::from(partial.to_owned())];
            for line in &rendered {
                term::print_line(stdout, line)?;
                // Count wrapped terminal rows, not logical lines: a long
                // streamed line (e.g. an emoji-prefixed note) wraps to
                // multiple rows. Undercounting here makes `total_height`
                // too small, so printing the dynamic region can scroll the
                // terminal past the tracked `input_bar_row` anchor — the
                // subsequent erase then misses the old partial and leaves
                // duplicated lines on screen.
                extra_lines = extra_lines.saturating_add(term::visual_height(line, width));
            }
        }

        // Render queued messages (dim) above the input bar.
        let mut queued_lines: u16 = 0;
        for (text, images) in &self.pending_inputs {
            for part in text.split('\n') {
                let line = Line::from(Span::styled(
                    part.to_string(),
                    styles::S_USER_ECHO.add_modifier(ratatui::style::Modifier::DIM),
                ));
                term::print_line(stdout, &line)?;
                queued_lines = queued_lines.saturating_add(term::visual_height(&line, width));
            }
            if !images.is_empty() {
                let n = images.len();
                let label = if n == 1 {
                    "1 image".to_string()
                } else {
                    format!("{n} images")
                };
                let line = Line::from(Span::styled(format!("     [{label}]"), styles::S_DIM));
                term::print_line(stdout, &line)?;
                queued_lines = queued_lines.saturating_add(term::visual_height(&line, width));
            }
            term::print_line(stdout, &Line::default())?;
            queued_lines = queued_lines.saturating_add(1);
        }

        // Render subagent progress above the input bar.
        let mut progress_lines: u16 = 0;
        if !self.subagent_progress.is_empty() {
            for line in self.subagent_progress.lines(width) {
                term::print_line(stdout, &line)?;
                progress_lines = progress_lines.saturating_add(term::visual_height(&line, width));
            }
        }

        self.textarea.set_placeholder_text(&self.config.placeholder);
        let block = self.input_block();
        self.textarea.set_block(block);

        let height = self.input_height(width);

        let spacing = 1u16;
        queue!(stdout, Print("\r\n"))?;

        // Compose the textarea with an optional pet column on the right.
        let pet_enabled = self.pet.is_some() && width > 34;
        let pet_cols: u16 = if pet_enabled { 14 } else { 0 };
        let ta_width = width.saturating_sub(pet_cols).max(1);
        // Keep the cursor in view when content overflows the (clamped) height.
        let inner_w = ta_width.saturating_sub(2).max(1);
        let inner_h = height.saturating_sub(1).max(1);
        self.textarea.ensure_cursor_visible(inner_w, inner_h);
        let ta_area = ratatui::layout::Rect::new(0, 0, ta_width, height);
        if pet_enabled {
            // Render into a buffer: textarea on the left, pet on the right.
            let pet_area = ratatui::layout::Rect::new(ta_width, 0, pet_cols, height);
            let ta_ref = &self.textarea;
            let pet_ref = self.pet.as_ref();
            term::render_composited_to_stdout(stdout, width, height, |buf| {
                ta_ref.render(ta_area, buf);
                if let Some(pet) = pet_ref {
                    pet.render(buf, pet_area);
                }
            })?;
        } else {
            term::render_widget_to_stdout(stdout, &self.textarea, width, height)?;
        }

        let bottom_pad = 1u16;
        for _ in 0..bottom_pad {
            queue!(stdout, Print("\r\n"))?;
        }

        let total_height =
            extra_lines + queued_lines + progress_lines + spacing + height + bottom_pad;
        let actual_top = row.min(term_h.saturating_sub(total_height));

        // Cursor position is relative to the textarea's inner area; shift it
        // right by the pet column offset when the pet is rendered beside it.
        let cursor_pos = self.textarea.cursor_screen_pos(ratatui::layout::Rect::new(
            0,
            0,
            ta_width.max(1),
            height,
        ));
        let cy = cursor_pos.map(|(_, cy)| cy).unwrap_or(0);
        if let Some((cx, _)) = cursor_pos {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(
                    cx,
                    actual_top + extra_lines + queued_lines + progress_lines + spacing + cy,
                ),
                Show,
            )?;
        }

        // Track the layout we just drew so the next resize can erase the
        // dynamic region cursor-relatively (see `erase_dynamic_region`).
        self.last_dynamic_height = total_height;
        self.last_cursor_offset = extra_lines + queued_lines + progress_lines + spacing + cy;
        self.last_draw_width = width;

        stdout.flush()?;
        self.widget_top_row = Some(actual_top);
        Ok(actual_top)
    }

    /// Compute net cursor row displacement from a set of render actions.
    ///
    /// Uses raw character-width wrapping (matching terminal behaviour) instead
    /// of ratatui's `Paragraph::wrap` which can disagree with the terminal.
    fn actions_cursor_delta(actions: &[RenderAction], term_width: u16) -> i32 {
        use unicode_width::UnicodeWidthStr;
        let tw = term_width.max(1) as usize;
        let line_rows = |line: &Line<'_>| -> i32 {
            let w: usize = line
                .spans
                .iter()
                .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                .sum();
            if w == 0 { 1 } else { w.div_ceil(tw) as i32 }
        };
        let mut delta: i32 = 0;
        for action in actions {
            match action {
                RenderAction::Append(line) => {
                    delta += line_rows(line);
                }
                RenderAction::ReplaceTool { erase_count, lines } => {
                    delta -= *erase_count as i32;
                    for line in lines {
                        delta += line_rows(line);
                    }
                }
            }
        }
        delta
    }

    // -----------------------------------------------------------------------
    // Shared key processing

    /// Process a key event and return the resulting action. Handles all common
    /// key bindings (history, autocomplete, editing) so both `read_input` and
    /// `stream_response` share identical behaviour.
    fn process_key(&mut self, key: event::KeyEvent) -> KeyAction {
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
            styles::S_USER,
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
        if self.config.show_usage
            && let Some(u) = &self.last_usage
        {
            spans.push(Span::styled(
                format!("{}↑ {}↓ ", u.prompt_tokens, u.completion_tokens),
                styles::S_DIM,
            ));
        }
        spans
    }

    fn input_block(&mut self) -> Block<'static> {
        let mut block = Block::default()
            .borders(Borders::TOP)
            .title(Line::from(self.input_title_spans()))
            .padding(Padding::new(1, 1, 0, 0));
        if let Some(status) = &self.status {
            let spans = status.to_spans();
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
        let min = if self.pet.is_some() { 4 } else { 2 };
        content.clamp(min, self.config.max_input_height + 1)
    }

    fn echo_input(&mut self, stdout: &mut io::Stdout, text: &str) -> io::Result<()> {
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let full_pad = " ".repeat(term_w.saturating_sub(1) as usize);

        let top = Line::from(Span::styled(full_pad.clone(), styles::S_USER_ECHO));
        term::print_line(stdout, &top)?;
        self.renderer.push_line(top);
        for line in text.split('\n') {
            let pad = " ".repeat(
                (term_w as usize)
                    .saturating_sub(line.len())
                    .saturating_sub(1),
            );
            let l = Line::from(Span::styled(format!("{line}{pad}"), styles::S_USER_ECHO));
            term::print_line(stdout, &l)?;
            self.renderer.push_line(l);
        }
        let bot = Line::from(Span::styled(full_pad, styles::S_USER_ECHO));
        term::print_line(stdout, &bot)?;
        self.renderer.push_line(bot);
        let blank = Line::default();
        term::print_line(stdout, &blank)?;
        self.renderer.push_line(blank);
        stdout.flush()
    }

    /// Erase the last-drawn widget from the screen.
    /// Moves cursor back to anchor, clears downward, resets tracking state.
    fn erase_widget(&mut self, stdout: &mut io::Stdout) -> io::Result<()> {
        if let Some(top_row) = self.widget_top_row.take() {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(0, top_row),
                Clear(ClearType::FromCursorDown),
            )?;
            stdout.flush()?;
        }
        Ok(())
    }

    fn draw_input(&mut self, stdout: &mut io::Stdout) -> io::Result<()> {
        let (width, term_h) = ratatui::crossterm::terminal::size()?;

        if let Some(top_row) = self.widget_top_row {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(0, top_row),
                Clear(ClearType::FromCursorDown),
            )?;
        }

        self.textarea.set_placeholder_text(&self.config.placeholder);
        let block = self.input_block();
        self.textarea.set_block(block);

        let height = self.input_height(width);

        // Spacing above the input bar so it doesn't hug the content above.
        queue!(stdout, Print("\r\n"))?;

        // Compose the textarea with an optional pet column on the right.
        let pet_enabled = self.pet.is_some() && width > 34;
        let pet_cols: u16 = if pet_enabled { 14 } else { 0 };
        let ta_width = width.saturating_sub(pet_cols).max(1);
        // Keep the cursor in view when content overflows the (clamped) height.
        // Inner dims account for the TOP border (1 row) and left+right padding
        // (1 col each) so wrapping + cursor rows match the actual render.
        let inner_w = ta_width.saturating_sub(2).max(1);
        let inner_h = height.saturating_sub(1).max(1);
        self.textarea.ensure_cursor_visible(inner_w, inner_h);
        let ta_area = ratatui::layout::Rect::new(0, 0, ta_width, height);
        if pet_enabled {
            let pet_area = ratatui::layout::Rect::new(ta_width, 0, pet_cols, height);
            let ta_ref = &self.textarea;
            let pet_ref = self.pet.as_ref();
            term::render_composited_to_stdout(stdout, width, height, |buf| {
                ta_ref.render(ta_area, buf);
                if let Some(pet) = pet_ref {
                    pet.render(buf, pet_area);
                }
            })?;
        } else {
            term::render_widget_to_stdout(stdout, &self.textarea, width, height)?;
        }

        let dropdown_lines = if let Some(ref dropdown) = self.dropdown {
            queue!(stdout, Print("\r\n"))?;
            let lines = dropdown.lines(width);
            for line in &lines {
                term::print_line(stdout, line)?;
            }
            lines.len() as u16
        } else {
            0
        };

        // Bottom padding so the input bar doesn't sit flush against the
        // terminal's bottom edge.
        let bottom_pad = 1u16;
        for _ in 0..bottom_pad {
            queue!(stdout, Print("\r\n"))?;
        }

        // Compute widget top row. On the first draw we don't know where the
        // cursor started, so we query the terminal once.  On redraws we know
        // exactly where we are because we MoveTo'd the stored top_row, and
        // render_widget_to_stdout cancels pending-wrap (via EL) so row counts
        // are deterministic.
        let top_row = if let Some(top) = self.widget_top_row {
            // Account for any scrolling caused by the dropdown extending past
            // the terminal bottom.
            let separator = if dropdown_lines > 0 { 1 } else { 0 };
            let spacing = 1u16;
            let total = spacing + height + separator + dropdown_lines + bottom_pad;
            let end_row = top + total;
            if end_row > term_h {
                top.saturating_sub(end_row - term_h)
            } else {
                top
            }
        } else {
            stdout.flush()?;
            let (_, after_row) = ratatui::crossterm::cursor::position()?;
            // The cursor is past the dropdown (if any). Walk back to find the
            // widget top, accounting for the extra \r\n separator before the
            // dropdown.
            let rows_after_widget = if dropdown_lines > 0 {
                dropdown_lines + 1
            } else {
                0
            };
            after_row.saturating_sub(height - 1 + rows_after_widget + 1 + bottom_pad)
        };
        self.widget_top_row = Some(top_row);

        // Position cursor at the textarea cursor using absolute coordinates.
        // +1 accounts for the spacing line above the input bar.  The cursor
        // rect uses `ta_width` (not `width`) so wrapping matches how the
        // textarea was actually rendered beside the pet column.
        let cursor_pos = self.textarea.cursor_screen_pos(ratatui::layout::Rect::new(
            0,
            0,
            ta_width.max(1),
            height,
        ));
        let cy = cursor_pos.map(|(_, cy)| cy).unwrap_or(0);
        if let Some((cx, _)) = cursor_pos {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(cx, top_row + 1 + cy),
                Show,
            )?;
        }

        // Track the layout we just drew so the next resize can erase the
        // dynamic region cursor-relatively (see `erase_dynamic_region`).
        let separator = if dropdown_lines > 0 { 1 } else { 0 };
        self.last_dynamic_height = 1 + height + separator + dropdown_lines + bottom_pad;
        self.last_cursor_offset = cy;
        self.last_draw_width = width;

        stdout.flush()?;
        Ok(())
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
/// On Android (no clipboard support in `arboard`), always returns `Err`.
pub(crate) fn copy_to_clipboard(text: &str) -> Result<(), String> {
    #[cfg(not(target_os = "android"))]
    {
        arboard::Clipboard::new()
            .and_then(|mut cb| cb.set_text(text))
            .map_err(|e| e.to_string())
    }
    #[cfg(target_os = "android")]
    {
        let _ = text;
        Err("Clipboard not available on Android".into())
    }
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
