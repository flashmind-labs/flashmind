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
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crossterm::event::EventStream;
use futures::{Stream, StreamExt};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::crossterm::{
    cursor::{MoveToColumn, MoveUp, Show},
    execute, queue,
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
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
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
    /// Slash commands available for autocomplete (e.g. `["/help", "/model", "/quit"]`).
    /// When the user types `/`, a dropdown of matching commands is shown.
    /// Default: empty (no autocomplete).
    pub available_commands: Vec<String>,
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
    /// Height of the last-drawn input widget (0 = nothing drawn yet).
    last_input_height: u16,
    /// Cursor offset from the anchor (top of widget). Used to return to anchor
    /// before redrawing.
    cursor_rows_from_anchor: u16,
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
    /// Images pasted via Ctrl+V, pending attachment to the next message.
    pending_images: Vec<PastedImage>,
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
            last_input_height: 0,
            cursor_rows_from_anchor: 0,
            last_usage: None,
            status: None,
            history,
            history_index: None,
            history_draft: String::new(),
            dropdown: None,
            pending_images: Vec::new(),
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
        let mut stdout = io::stdout();

        let _raw = RawModeGuard::enable()?;
        self.draw_input(&mut stdout)?;

        loop {
            if !event::poll(std::time::Duration::from_millis(100))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };

            match key {
                event::KeyEvent {
                    code: KeyCode::Char('d'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } if self.textarea.is_empty() => {
                    return self.handle_quit(&mut stdout);
                }

                event::KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    self.handle_cancel(&mut stdout)?;
                }

                event::KeyEvent {
                    code: KeyCode::Enter,
                    modifiers,
                    ..
                } => {
                    // If dropdown is visible with candidates, accept the selection
                    if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                        && let Some(dropdown) = &self.dropdown
                        && !dropdown.is_empty()
                        && let Some(value) = dropdown.selected_value()
                    {
                        let cmd = format!("{value} ");
                        self.textarea.set_text(&cmd);
                        self.dropdown = None;
                        self.draw_input(&mut stdout)?;
                        continue;
                    }
                    if let Some(ev) = self.handle_submit(&mut stdout, modifiers)? {
                        return Ok(ev);
                    }
                }

                // Tab: accept dropdown selection if visible
                event::KeyEvent {
                    code: KeyCode::Tab, ..
                } => {
                    if let Some(dropdown) = &self.dropdown
                        && !dropdown.is_empty()
                        && let Some(value) = dropdown.selected_value()
                    {
                        let cmd = format!("{value} ");
                        self.textarea.set_text(&cmd);
                        self.dropdown = None;
                        self.draw_input(&mut stdout)?;
                        continue;
                    }
                }

                // Esc: dismiss dropdown if visible
                event::KeyEvent {
                    code: KeyCode::Esc, ..
                } => {
                    if self.dropdown.is_some() {
                        self.dropdown = None;
                        self.draw_input(&mut stdout)?;
                    }
                }

                // Up: history browsing when cursor is on first line, otherwise
                // navigate dropdown or pass through to textarea
                event::KeyEvent {
                    code: KeyCode::Up,
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    if let Some(ref mut dropdown) = self.dropdown
                        && !dropdown.is_empty()
                    {
                        dropdown.handle_key(key);
                        self.draw_input(&mut stdout)?;
                        continue;
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
                        self.draw_input(&mut stdout)?;
                    } else {
                        self.textarea.input(key);
                        self.draw_input(&mut stdout)?;
                    }
                }

                // Down: navigate history forward or pass through to textarea
                event::KeyEvent {
                    code: KeyCode::Down,
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    if let Some(ref mut dropdown) = self.dropdown
                        && !dropdown.is_empty()
                    {
                        dropdown.handle_key(key);
                        self.draw_input(&mut stdout)?;
                        continue;
                    }
                    if let Some(idx) = self.history_index {
                        if idx + 1 >= self.history.len() {
                            // Past the end — restore draft
                            self.history_index = None;
                            let draft = self.history_draft.clone();
                            self.textarea.set_text(&draft);
                        } else {
                            let new_idx = idx + 1;
                            self.history_index = Some(new_idx);
                            self.textarea.set_text(&self.history[new_idx]);
                        }
                        self.draw_input(&mut stdout)?;
                    } else {
                        self.textarea.input(key);
                        self.draw_input(&mut stdout)?;
                    }
                }

                // Ctrl+L: clear screen
                event::KeyEvent {
                    code: KeyCode::Char('l'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    execute!(
                        stdout,
                        crossterm::terminal::Clear(ClearType::All),
                        crossterm::cursor::MoveTo(0, 0)
                    )?;
                    self.last_input_height = 0;
                    self.cursor_rows_from_anchor = 0;
                    self.draw_input(&mut stdout)?;
                }

                // Ctrl+V: paste image from clipboard (macOS)
                event::KeyEvent {
                    code: KeyCode::Char('v'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    if let Some(img) = grab_clipboard_image() {
                        self.pending_images.push(img);
                        self.draw_input(&mut stdout)?;
                    } else {
                        // No image — fall through to normal paste handling
                        self.textarea.input(key);
                        self.update_autocomplete();
                        self.draw_input(&mut stdout)?;
                    }
                }

                // Ctrl+J: insert newline (fallback for terminals that don't support Shift+Enter)
                event::KeyEvent {
                    code: KeyCode::Char('j'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    self.textarea.insert_newline();
                    self.update_autocomplete();
                    self.draw_input(&mut stdout)?;
                }

                other => {
                    self.textarea.input(other);
                    self.update_autocomplete();
                    self.draw_input(&mut stdout)?;
                }
            }
        }
    }

    /// Stream agent response events and render them to the terminal in real time.
    ///
    /// Drains the given stream of [`AgentEvent`]s, rendering each one
    /// through the [`EventRenderer`].  While waiting for the first event, displays
    /// an animated spinner with a "thinking..." label.
    ///
    /// The `cancel_token` is used to cancel the turn when the user presses Ctrl+C.
    ///
    /// # Event handling
    ///
    /// - **TextDelta** — buffered by the `EventRenderer`; complete lines are flushed
    ///   as they arrive; remaining partial text is flushed on `Done`/`Error`.
    /// - **ToolStart / ToolResult** — rendered immediately with status icons (▶, ✓, ✗).
    /// - **FileDiff** — shown with green/red coloring for added/removed lines.
    /// - **SpawnedEvent** — prefixed with the task name in magenta.
    /// - **Done / Error** — flushes any buffered text and breaks the loop.
    ///
    /// Returns `Ok(())` when the stream completes normally or is exhausted.
    pub async fn stream_response(
        &mut self,
        cancel_token: CancellationToken,
        mut stream: impl Stream<Item = AgentEvent> + Unpin,
    ) -> io::Result<()> {
        self.cancel_token = Some(cancel_token.clone());
        let mut stdout = io::stdout();
        let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));
        let mut thinking = true;

        // Update renderer with current terminal width for right-aligned elapsed times.
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        self.renderer.set_width(term_w.saturating_sub(1) as usize);

        let _raw = RawModeGuard::enable()?;
        let mut key_stream = EventStream::new();

        loop {
            tokio::select! {
                biased;
                maybe_key = key_stream.next() => {
                    if let Some(Ok(crossterm::event::Event::Key(key))) = maybe_key
                        && (key.code == crossterm::event::KeyCode::Esc
                            || (key.code == crossterm::event::KeyCode::Char('c')
                                && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)))
                    {
                        cancel_token.cancel();
                        self.cancel_token = None;
                    }
                }
                maybe_event = stream.next() => {
                    match maybe_event {
                        Some(event) => {
                            if thinking {
                                Self::clear_spinner(&mut stdout)?;
                                thinking = false;
                            }
                            match &event {
                                AgentEvent::Done(_) | AgentEvent::Error(_) => {
                                    let actions = self.renderer.render(&event);
                                    Self::apply_actions(&mut stdout, &actions)?;
                                    if let AgentEvent::Usage(u) = &event {
                                        self.last_usage = Some(u.clone());
                                    }
                                    stdout.flush()?;
                                    self.cancel_token = None;
                                    break;
                                }
                                _ => {
                                    if let AgentEvent::Usage(u) = &event {
                                        self.last_usage = Some(u.clone());
                                    }
                                    let actions = self.renderer.render(&event);
                                    Self::apply_actions(&mut stdout, &actions)?;
                                    stdout.flush()?;
                                }
                            }
                        }
                        None => {
                            if thinking {
                                Self::clear_spinner(&mut stdout)?;
                            }
                            let actions: Vec<_> = self
                                .renderer
                                .flush()
                                .into_iter()
                                .map(crate::event_render::RenderAction::Append)
                                .collect();
                            Self::apply_actions(&mut stdout, &actions)?;
                            stdout.flush()?;
                            self.cancel_token = None;
                            break;
                        }
                    }
                }
                _ = tick_interval.tick() => {
                    if self.renderer.tool_running() {
                        let actions = self.renderer.tick_tool();
                        Self::apply_actions(&mut stdout, &actions)?;
                        stdout.flush()?;
                    } else if thinking {
                        let line = self.spinner.line("thinking...");
                        execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
                        term::print_line(&mut stdout, &line)?;
                        execute!(stdout, MoveUp(1))?;
                        stdout.flush()?;
                    }
                }
            }
        }

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
    // Key handlers

    fn handle_quit(&mut self, stdout: &mut io::Stdout) -> io::Result<ReplEvent> {
        self.erase_widget(stdout)?;
        Ok(ReplEvent::Quit)
    }

    fn handle_cancel(&mut self, stdout: &mut io::Stdout) -> io::Result<()> {
        if let Some(token) = self.cancel_token.take() {
            token.cancel();
        }
        self.textarea.clear();
        self.history_index = None;
        self.history_draft.clear();
        self.dropdown = None;
        self.draw_input(stdout)
    }

    fn handle_submit(
        &mut self,
        stdout: &mut io::Stdout,
        modifiers: KeyModifiers,
    ) -> io::Result<Option<ReplEvent>> {
        if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
            self.textarea.insert_newline();
            self.dropdown = None;
            self.draw_input(stdout)?;
            return Ok(None);
        }
        if self.textarea.is_empty() {
            return Ok(None);
        }
        let text = self.textarea.text();
        self.history.push(text.clone());
        self.history_index = None;
        self.history_draft.clear();
        if let Some(path) = &self.config.history_file {
            append_history(path, &text);
        }
        self.textarea.clear();
        self.dropdown = None;
        let images = std::mem::take(&mut self.pending_images);
        self.erase_widget(stdout)?;
        self.echo_input(stdout, &text)?;
        Ok(Some(ReplEvent::UserInput(text, images)))
    }

    /// Update the autocomplete dropdown based on current textarea content.
    ///
    /// Shows matching slash commands when the input starts with `/` and is a
    /// single line.  Hides the dropdown otherwise.
    fn update_autocomplete(&mut self) {
        let text = self.textarea.text();
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

    fn input_title_spans(&self) -> Vec<Span<'static>> {
        let mut spans = vec![Span::styled(
            format!(" {} ", self.config.prompt),
            styles::S_USER,
        )];
        if !self.pending_images.is_empty() {
            let n = self.pending_images.len();
            let label = if n == 1 {
                "1 image".to_string()
            } else {
                format!("{n} images")
            };
            spans.push(Span::styled(
                format!("[{label}] "),
                ratatui::style::Style::default().fg(ratatui::style::Color::Magenta),
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

    fn input_block(&self) -> Block<'static> {
        let mut block = Block::default()
            .borders(Borders::TOP)
            .title(Line::from(self.input_title_spans()))
            .padding(Padding::new(1, 1, 0, 1));
        if let Some(status) = &self.status {
            let spans = status.to_spans();
            if !spans.is_empty() {
                block = block.title(Line::from(spans).right_aligned());
            }
        }
        block
    }

    fn input_height(&self, width: u16) -> u16 {
        // +1 for top border, +1 for bottom padding
        (self.textarea.visual_line_count(width.saturating_sub(2)) as u16 + 2)
            .clamp(3, self.config.max_input_height + 1)
    }

    fn echo_input(&self, stdout: &mut io::Stdout, text: &str) -> io::Result<()> {
        let prefix = "you> ";
        let indent = "     ";
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let w = (term_w as usize)
            .saturating_sub(1)
            .saturating_sub(prefix.len());

        for (i, line) in text.split('\n').enumerate() {
            let tag = if i == 0 { prefix } else { indent };
            if line.is_empty() {
                term::print_line(
                    stdout,
                    &Line::from(Span::styled(tag.to_string(), styles::S_USER)),
                )?;
                continue;
            }
            term::print_line(
                stdout,
                &Line::from(vec![
                    Span::styled(tag.to_string(), styles::S_USER),
                    Span::styled(line.to_string(), styles::S_TEXT),
                ]),
            )?;
        }
        let _ = w; // reserved for future word-wrapping
        stdout.flush()
    }

    /// Erase the last-drawn widget from the screen.
    /// Moves cursor back to anchor, clears downward, resets tracking state.
    fn erase_widget(&mut self, stdout: &mut io::Stdout) -> io::Result<()> {
        if self.last_input_height == 0 {
            return Ok(());
        }
        if self.cursor_rows_from_anchor > 0 {
            queue!(stdout, MoveUp(self.cursor_rows_from_anchor))?;
        }
        queue!(stdout, MoveToColumn(0))?;
        queue!(stdout, Clear(ClearType::FromCursorDown))?;
        stdout.flush()?;
        self.last_input_height = 0;
        self.cursor_rows_from_anchor = 0;
        Ok(())
    }

    fn draw_input(&mut self, stdout: &mut io::Stdout) -> io::Result<()> {
        if self.last_input_height > 0 {
            if self.cursor_rows_from_anchor > 0 {
                queue!(stdout, MoveUp(self.cursor_rows_from_anchor))?;
            }
            queue!(stdout, MoveToColumn(0))?;
            queue!(stdout, Clear(ClearType::FromCursorDown))?;
        }

        let (width, _) = ratatui::crossterm::terminal::size()?;
        self.textarea.set_placeholder_text(&self.config.placeholder);
        self.textarea.set_block(self.input_block());

        let height = self.input_height(width);

        term::render_widget_to_stdout(stdout, &self.textarea, width, height)?;

        // Render autocomplete dropdown below the input widget.
        let dropdown_lines = if let Some(ref dropdown) = self.dropdown {
            let lines = dropdown.lines(8, width);
            for line in &lines {
                term::print_line(stdout, line)?;
            }
            lines.len() as u16
        } else {
            0
        };

        let total_height = height + dropdown_lines;

        if total_height > 1 {
            queue!(stdout, MoveUp(total_height - 1))?;
        }
        queue!(stdout, MoveToColumn(0))?;

        let anchor_offset = if let Some((cx, cy)) = self
            .textarea
            .cursor_screen_pos(ratatui::layout::Rect::new(0, 0, width, height))
        {
            if cy > 0 {
                queue!(stdout, ratatui::crossterm::cursor::MoveDown(cy))?;
            }
            queue!(stdout, MoveToColumn(cx), Show)?;
            cy
        } else {
            0
        };

        stdout.flush()?;
        self.last_input_height = total_height;
        self.cursor_rows_from_anchor = anchor_offset;
        Ok(())
    }

    fn clear_spinner(stdout: &mut io::Stdout) -> io::Result<()> {
        execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine),)?;
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

        // AppleScript that checks for image data and outputs base64
        let script = r#"
            try
                set imgData to the clipboard as «class PNGf»
                set b64 to do shell script "osascript -e 'the clipboard as «class PNGf»' | sed 's/«data PNGf//;s/»//' | xxd -r -p | base64"
                return b64
            end try
            return ""
        "#;

        let output = Command::new("osascript")
            .arg("-e")
            .arg(script)
            .output()
            .ok()?;

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
