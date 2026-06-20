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
use std::collections::VecDeque;
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
use super::spinner::Spinner;
use super::textarea::TextArea;
use crate::event_render::{EventRenderer, RenderAction};
use crate::styles;
use crate::term;

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
    /// Slash commands available for autocomplete (e.g. `["/help", "/model", "/quit"]`).
    /// When the user types `/`, a dropdown of matching commands is shown.
    /// Default: empty (no autocomplete).
    pub available_commands: Vec<String>,
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
    /// Images pasted via Ctrl+V, pending attachment to the next message.
    pending_images: Vec<PastedImage>,
    /// Inputs submitted during streaming, queued for injection between turns.
    pending_inputs: VecDeque<(String, Vec<PastedImage>)>,
    /// Activity label shown in the input bar (e.g. "thinking", "file_read").
    activity: Option<String>,
    /// Reverse-i-search state: `(query, match_index)`.
    reverse_search: Option<(String, usize)>,
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
            pending_images: Vec::new(),
            pending_inputs: VecDeque::new(),
            activity: None,
            reverse_search: None,
        }
    }

    /// Print the greeting message (if configured) and flush stdout.
    ///
    /// Call this once at startup before entering the main loop.  Does nothing
    /// if [`ReplConfig::greeting`] is `None`.
    pub fn print_greeting(&self) -> io::Result<()> {
        if let Some(greeting) = &self.config.greeting {
            let mut stdout = io::stdout();
            term::print_line(
                &mut stdout,
                &Line::from(Span::styled(greeting.clone(), styles::S_AGENT)),
            )?;
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

    /// Set whether reasoning blocks are expanded (full text) or collapsed
    /// (one-line summary).  Affects subsequent reasoning blocks.
    pub fn set_expand_reasoning(&mut self, expand: bool) {
        self.renderer.set_expand_reasoning(expand);
    }

    /// Whether reasoning blocks are currently expanded.
    pub fn expand_reasoning(&self) -> bool {
        self.renderer.expand_reasoning()
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

        loop {
            if !event::poll(Duration::from_millis(100))? {
                continue;
            }
            let ev = event::read()?;

            if let Event::Paste(text) = &ev {
                if let Some(img) = grab_clipboard_image() {
                    self.pending_images.push(img);
                    let n = self.pending_images.len();
                    self.textarea.insert_str(&format!("[image #{n}]"));
                } else {
                    self.textarea.insert_str(text);
                    self.update_autocomplete();
                }
                self.draw_input(&mut stdout)?;
                continue;
            }

            if matches!(ev, Event::Resize(..)) {
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
                            crossterm::event::Event::Resize(w, h) => {
                                self.renderer.set_width(w.saturating_sub(1) as usize);
                                Self::erase_at_row(&mut stdout, 0)?;
                                input_bar_row = self.draw_input_at_row(
                                    &mut stdout,
                                    h.saturating_sub(1),
                                )?;
                            }
                            crossterm::event::Event::Paste(text) => {
                                if let Some(img) = grab_clipboard_image() {
                                    self.pending_images.push(img);
                                    let n = self.pending_images.len();
                                    self.textarea
                                        .insert_str(&format!("[image #{n}]"));
                                } else {
                                    self.textarea.insert_str(&text);
                                    self.update_autocomplete();
                                }
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
                            }

                            let is_done =
                                matches!(&event, AgentEvent::Done(_) | AgentEvent::Error(_));
                            let is_text =
                                matches!(&event, AgentEvent::TextDelta(_));
                            let actions = self.renderer.render(&event);
                            let needs_update =
                                !actions.is_empty() || is_done || is_text;

                            if needs_update {
                                Self::erase_at_row(&mut stdout, input_bar_row)?;

                                let (tw, _) = ratatui::crossterm::terminal::size()
                                    .unwrap_or((80, 24));
                                let delta = Self::actions_cursor_delta(&actions, tw);
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
                    if self.renderer.tool_running() {
                        Self::erase_at_row(&mut stdout, input_bar_row)?;
                        let actions = self.renderer.tick_tool();
                        let (tw, _) = ratatui::crossterm::terminal::size()
                            .unwrap_or((80, 24));
                        let delta = Self::actions_cursor_delta(&actions, tw);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        let new_row = (input_bar_row as i32 + delta).max(0) as u16;
                        input_bar_row =
                            self.draw_input_at_row(&mut stdout, new_row)?;
                    } else if self.activity.is_some() {
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
        let actions = self.renderer.render(event);
        if actions.is_empty() {
            return Ok(());
        }
        let mut stdout = io::stdout();

        if let Some(bar_row) = self.widget_top_row {
            Self::erase_at_row(&mut stdout, bar_row)?;
            let (tw, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            let delta = Self::actions_cursor_delta(&actions, tw);
            Self::apply_actions(&mut stdout, &actions)?;
            stdout.flush()?;
            let new_row = (bar_row as i32 + delta).max(0) as u16;
            let (_, th) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
            let clamped = new_row.min(th.saturating_sub(1));
            self.draw_input_at_row(&mut stdout, clamped)?;
        } else {
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
                            crossterm::event::Event::Resize(w, h) => {
                                self.renderer
                                    .set_width(w.saturating_sub(1) as usize);
                                Self::erase_at_row(&mut stdout, 0)?;
                                bar_row = self.draw_input_at_row(
                                    &mut stdout,
                                    h.saturating_sub(1),
                                )?;
                            }
                            crossterm::event::Event::Paste(text) => {
                                if let Some(img) = grab_clipboard_image() {
                                    self.pending_images.push(img);
                                    let n = self.pending_images.len();
                                    self.textarea
                                        .insert_str(&format!("[image #{n}]"));
                                } else {
                                    self.textarea.insert_str(&text);
                                    self.update_autocomplete();
                                }
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
                    if self.renderer.tool_running() {
                        Self::erase_at_row(&mut stdout, bar_row)?;
                        let actions = self.renderer.tick_tool();
                        let (tw, _) =
                            ratatui::crossterm::terminal::size()
                                .unwrap_or((80, 24));
                        let delta =
                            Self::actions_cursor_delta(&actions, tw);
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                        let new_row =
                            (bar_row as i32 + delta).max(0) as u16;
                        bar_row = self.draw_input_at_row(
                            &mut stdout,
                            new_row,
                        )?;
                    } else if self.activity.is_some() {
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

    /// Mark the start of a new turn for elapsed time tracking.
    pub fn mark_turn_start(&mut self) {
        self.renderer.mark_turn_start();
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        self.renderer.set_width(term_w.saturating_sub(1) as usize);
    }

    /// Replay a user message (echo styled input without the input bar).
    pub fn replay_user_input(&self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout();
        self.echo_input(&mut stdout, text)
    }

    /// Replay an agent event without the input bar (for session restore).
    pub fn replay_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        let actions = self.renderer.render(event);
        if actions.is_empty() {
            return Ok(());
        }
        let mut stdout = io::stdout();
        Self::apply_actions(&mut stdout, &actions)?;
        stdout.flush()
    }

    /// Finish replaying a turn (flush remaining text buffer).
    pub fn replay_finish_turn(&mut self) -> io::Result<()> {
        let done_event = AgentEvent::Done(String::new());
        let actions = self.renderer.render(&done_event);
        if !actions.is_empty() {
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
                extra_lines += 1;
            }
        }

        // Render queued messages (dim) above the input bar.
        let mut queued_lines: u16 = 0;
        for (text, images) in &self.pending_inputs {
            for part in text.split('\n') {
                term::print_line(
                    stdout,
                    &Line::from(Span::styled(
                        part.to_string(),
                        styles::S_USER_ECHO.add_modifier(ratatui::style::Modifier::DIM),
                    )),
                )?;
                queued_lines += 1;
            }
            if !images.is_empty() {
                let n = images.len();
                let label = if n == 1 {
                    "1 image".to_string()
                } else {
                    format!("{n} images")
                };
                term::print_line(
                    stdout,
                    &Line::from(Span::styled(format!("     [{label}]"), styles::S_DIM)),
                )?;
                queued_lines += 1;
            }
            term::print_line(stdout, &Line::default())?;
            queued_lines += 1;
        }

        self.textarea.set_placeholder_text(&self.config.placeholder);
        let block = self.input_block();
        self.textarea.set_block(block);

        let height = self.input_height(width);
        term::render_widget_to_stdout(stdout, &self.textarea, width, height)?;

        let total_height = extra_lines + queued_lines + height;
        let actual_top = row.min(term_h.saturating_sub(total_height));

        if let Some((cx, cy)) = self
            .textarea
            .cursor_screen_pos(ratatui::layout::Rect::new(0, 0, width, height))
        {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(
                    cx,
                    actual_top + extra_lines + queued_lines + cy,
                ),
                Show,
            )?;
        }

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
                let text = self.textarea.text();
                self.history.push(text.clone());
                self.history_index = None;
                self.history_draft.clear();
                if let Some(path) = &self.config.history_file {
                    append_history(path, &text);
                }
                self.textarea.clear();
                self.dropdown = None;
                self.mention = None;
                let all_images = std::mem::take(&mut self.pending_images);
                let (text, images) = resolve_image_labels(&text, all_images);
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
                        // @mention: splice only the `@query` token.
                        let text = self.textarea.text();
                        let cursor = self.textarea.cursor_byte_offset();
                        let token_end = cursor.min(text.len());
                        let replacement = format!("@{value} ");
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

            // Ctrl+V — paste image
            event::KeyEvent {
                code: KeyCode::Char('v'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                if let Some(img) = grab_clipboard_image() {
                    self.pending_images.push(img);
                    let n = self.pending_images.len();
                    self.textarea.insert_str(&format!("[image #{n}]"));
                } else {
                    self.textarea.input(key);
                    self.update_autocomplete();
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
                self.renumber_image_labels();
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

    /// After a text edit, check whether any `[image #N]` labels were removed
    /// and renumber the survivors so they stay sequential (1, 2, 3, …).
    /// Also drops the corresponding entries from `pending_images`.
    fn renumber_image_labels(&mut self) {
        if self.pending_images.is_empty() {
            return;
        }
        let text = self.textarea.text();
        let mut present: Vec<bool> = Vec::with_capacity(self.pending_images.len());
        for i in 1..=self.pending_images.len() {
            present.push(text.contains(&format!("[image #{i}]")));
        }
        if present.iter().all(|&p| p) {
            return;
        }

        let mut new_images: Vec<PastedImage> = Vec::new();
        let mut new_text = text.clone();
        let mut next_num = 1usize;
        for (i, &is_present) in present.iter().enumerate() {
            let old_label = format!("[image #{}]", i + 1);
            if is_present {
                let new_label = format!("[image #{next_num}]");
                if old_label != new_label {
                    new_text = new_text.replacen(&old_label, &new_label, 1);
                }
                new_images.push(self.pending_images[i].clone());
                next_num += 1;
            } else {
                new_text = new_text.replace(&old_label, "");
            }
        }

        self.pending_images = new_images;
        if new_text != text {
            self.textarea.set_text_preserve_cursor(&new_text);
        }
    }

    /// Update the autocomplete dropdown based on current textarea content.
    ///
    /// Shows matching slash commands when the input starts with `/` and is a
    /// single line.  Hides the dropdown otherwise.
    fn update_autocomplete(&mut self) {
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
        // Not in a mention token: clear any lingering mention state.
        self.mention = None;

        // --- slash-command completion ---
        if !text.starts_with('/')
            || text.contains('\n')
            || self.config.available_commands.is_empty()
        {
            self.dropdown = None;
            return;
        }
        let prefix = text.trim_end();
        let candidates: Vec<String> = self
            .config
            .available_commands
            .iter()
            .filter(|cmd| cmd.starts_with(prefix) && *cmd != prefix)
            .cloned()
            .collect();
        if candidates.is_empty() {
            self.dropdown = None;
        } else {
            match &mut self.dropdown {
                Some(dd) => dd.set_candidates(candidates),
                None => self.dropdown = Some(Dropdown::new("", candidates)),
            }
        }
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
        if !self.pending_images.is_empty() {
            let text = self.textarea.text();
            let attached: Vec<usize> = (1..=self.pending_images.len())
                .filter(|i| text.contains(&format!("[image #{i}]")))
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
        // +1 for top border
        (self.textarea.visual_line_count(width.saturating_sub(2)) as u16 + 1)
            .clamp(2, self.config.max_input_height + 1)
    }

    fn echo_input(&self, stdout: &mut io::Stdout, text: &str) -> io::Result<()> {
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let full_pad = " ".repeat(term_w.saturating_sub(1) as usize);

        term::print_line(
            stdout,
            &Line::from(Span::styled(full_pad.clone(), styles::S_USER_ECHO)),
        )?;
        for line in text.split('\n') {
            let pad = " ".repeat(
                (term_w as usize)
                    .saturating_sub(line.len())
                    .saturating_sub(1),
            );
            term::print_line(
                stdout,
                &Line::from(Span::styled(format!("{line}{pad}"), styles::S_USER_ECHO)),
            )?;
        }
        term::print_line(
            stdout,
            &Line::from(Span::styled(full_pad, styles::S_USER_ECHO)),
        )?;
        term::print_line(stdout, &Line::default())?;
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

        term::render_widget_to_stdout(stdout, &self.textarea, width, height)?;

        let dropdown_lines = if let Some(ref dropdown) = self.dropdown {
            queue!(stdout, Print("\r\n"))?;
            let lines = dropdown.lines(8, width);
            for line in &lines {
                term::print_line(stdout, line)?;
            }
            lines.len() as u16
        } else {
            0
        };

        // Compute widget top row. On the first draw we don't know where the
        // cursor started, so we query the terminal once.  On redraws we know
        // exactly where we are because we MoveTo'd the stored top_row, and
        // render_widget_to_stdout cancels pending-wrap (via EL) so row counts
        // are deterministic.
        let top_row = if let Some(top) = self.widget_top_row {
            // Account for any scrolling caused by the dropdown extending past
            // the terminal bottom.
            let separator = if dropdown_lines > 0 { 1 } else { 0 };
            let total = height + separator + dropdown_lines;
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
            after_row.saturating_sub(height - 1 + rows_after_widget)
        };
        self.widget_top_row = Some(top_row);

        // Position cursor at the textarea cursor using absolute coordinates.
        if let Some((cx, cy)) = self
            .textarea
            .cursor_screen_pos(ratatui::layout::Rect::new(0, 0, width, height))
        {
            queue!(
                stdout,
                ratatui::crossterm::cursor::MoveTo(cx, top_row + cy),
                Show,
            )?;
        }

        stdout.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// History persistence helpers

/// Maximum number of history entries kept in memory and on disk.
const MAX_HISTORY: usize = 1000;

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
}

// ---------------------------------------------------------------------------
// Clipboard image helpers

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

/// Resolve image labels: keep only images whose `[image #N]` label is still in
/// the text, strip all labels from the text, and renumber the survivors
/// sequentially so `[image #1]`, `[image #2]`, ... map 1:1 to the returned vec.
fn resolve_image_labels(text: &str, images: Vec<PastedImage>) -> (String, Vec<PastedImage>) {
    if images.is_empty() {
        return (text.to_string(), images);
    }

    let mut kept: Vec<PastedImage> = Vec::new();
    for i in 1..=images.len() {
        if text.contains(&format!("[image #{i}]")) {
            kept.push(images[i - 1].clone());
        }
    }

    let mut cleaned = text.to_string();
    for i in 1..=images.len() {
        cleaned = cleaned.replace(&format!("[image #{i}]"), "");
    }
    let cleaned = cleaned.trim().to_string();

    if kept.is_empty() {
        return (cleaned, Vec::new());
    }

    let tags: Vec<String> = (1..=kept.len()).map(|i| format!("[image #{i}]")).collect();
    let suffix = tags.join(" ");
    let labeled = if cleaned.is_empty() {
        suffix
    } else {
        format!("{cleaned}\n{suffix}")
    };

    (labeled, kept)
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
    if cursor == 0 || cursor > text.len() {
        return None;
    }
    let before = &text[..cursor];
    let at_pos = before.rfind('@')?;
    // `@` must be at start of text or preceded by whitespace.
    if at_pos > 0 && !before.as_bytes()[at_pos - 1].is_ascii_whitespace() {
        return None;
    }
    // No whitespace allowed between `@` and the cursor.
    if text[at_pos + 1..cursor].contains(char::is_whitespace) {
        return None;
    }
    Some(MentionState {
        token_start: at_pos,
        cursor,
    })
}
