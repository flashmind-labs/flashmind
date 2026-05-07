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

use super::spinner::Spinner;
use super::textarea::TextArea;
use crate::event_render::EventRenderer;
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
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "▸".to_string(),
            greeting: None,
            placeholder: "Type a message...".to_string(),
            max_input_height: 10,
            show_usage: true,
        }
    }
}

/// Result of calling [`Repl::read_input`].
#[derive(Debug)]
pub enum ReplEvent {
    /// The user submitted text (pressed Enter on a non-empty input).
    UserInput(String),
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
/// Pressing Ctrl+C during [`Repl::stream_response`] cancels the in-flight agent
/// turn by triggering the [`CancellationToken`] received in the
/// [`AgentEvent::Started`] event.  Pressing Ctrl+C while typing simply clears
/// the current input.
///
/// # Example
///
/// ```rust,ignore
/// use flashmind_tui::{Repl, ReplConfig, ReplEvent};
///
/// let mut repl = Repl::new(ReplConfig::default());
/// repl.print_greeting()?;
///
/// loop {
///     match repl.read_input()? {
///         ReplEvent::UserInput(text) => {
///             let stream = agent.start(&mut conversation, AgentInput::user(&text));
///             repl.stream_response(stream).await?;
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
}

impl<'a> Repl<'a> {
    /// Create a new REPL with the given configuration.
    pub fn new(config: ReplConfig) -> Self {
        Self {
            config,
            textarea: TextArea::default(),
            renderer: EventRenderer::new(),
            spinner: Spinner::new(),
            cancel_token: None,
            last_input_height: 0,
            cursor_rows_from_anchor: 0,
            last_usage: None,
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
        });
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
                    if let Some(ev) = self.handle_submit(&mut stdout, modifiers)? {
                        return Ok(ev);
                    }
                }

                // Ctrl+J: insert newline (fallback for terminals that don't support Shift+Enter)
                event::KeyEvent {
                    code: KeyCode::Char('j'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    self.textarea.insert_newline();
                    self.draw_input(&mut stdout)?;
                }

                other => {
                    self.textarea.input(other);
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
    /// The method captures the [`CancellationToken`] from the first
    /// [`AgentEvent::Started`] event so that Ctrl+C can cancel the turn.
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
    pub async fn stream_response(&mut self, mut stream: impl Stream<Item = AgentEvent> + Unpin) -> io::Result<()> {
        let mut stdout = io::stdout();
        let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));
        let mut thinking = true;

        loop {
            tokio::select! {
                maybe_event = stream.next() => {
                    match maybe_event {
                        Some(event) => {
                            if thinking {
                                Self::clear_spinner(&mut stdout)?;
                                thinking = false;
                            }
                            match &event {
                                AgentEvent::Started { cancel_token, .. } => {
                                    self.cancel_token = Some(cancel_token.clone());
                                    thinking = true;
                                }
                                AgentEvent::Usage(u) => {
                                    self.last_usage = Some(u.clone());
                                }
                                AgentEvent::Done(_) | AgentEvent::Error(_) => {
                                    let lines = self.renderer.render(&event);
                                    for line in &lines {
                                        term::print_line(&mut stdout, line)?;
                                    }
                                    stdout.flush()?;
                                    self.cancel_token = None;
                                    break;
                                }
                                _ => {
                                    let lines = self.renderer.render(&event);
                                    for line in &lines {
                                        term::print_line(&mut stdout, line)?;
                                    }
                                    stdout.flush()?;
                                }
                            }
                        }
                        None => break,
                    }
                }
                _ = tick_interval.tick(), if thinking => {
                    let line = self.spinner.line("thinking...");
                    execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
                    term::print_line(&mut stdout, &line)?;
                    execute!(stdout, MoveUp(1))?;
                    stdout.flush()?;
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
        self.draw_input(stdout)
    }

    fn handle_submit(
        &mut self,
        stdout: &mut io::Stdout,
        modifiers: KeyModifiers,
    ) -> io::Result<Option<ReplEvent>> {
        if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
            self.textarea.insert_newline();
            self.draw_input(stdout)?;
            return Ok(None);
        }
        if self.textarea.is_empty() {
            return Ok(None);
        }
        let text = self.textarea.text();
        self.textarea.clear();
        self.erase_widget(stdout)?;
        self.echo_input(stdout, &text)?;
        Ok(Some(ReplEvent::UserInput(text)))
    }

    // -----------------------------------------------------------------------
    // Drawing helpers

    fn input_title_spans(&self) -> Vec<Span<'static>> {
        let mut spans = vec![Span::styled(
            format!(" {} ", self.config.prompt),
            styles::S_USER,
        )];
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
        Block::default()
            .borders(Borders::TOP)
            .title(Line::from(self.input_title_spans()))
            .padding(Padding::horizontal(1))
    }

    fn input_height(&self, width: u16) -> u16 {
        (self.textarea.visual_line_count(width.saturating_sub(2)) as u16 + 1)
            .clamp(2, self.config.max_input_height)
    }

    fn echo_input(&self, stdout: &mut io::Stdout, text: &str) -> io::Result<()> {
        term::print_line(
            stdout,
            &Line::from(vec![
                Span::styled(format!("{} ", self.config.prompt), styles::S_USER),
                Span::styled(text.to_string(), styles::S_TEXT),
            ]),
        )?;
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

        if height > 1 {
            queue!(stdout, MoveUp(height - 1))?;
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
        self.last_input_height = height;
        self.cursor_rows_from_anchor = anchor_offset;
        Ok(())
    }

    fn clear_spinner(stdout: &mut io::Stdout) -> io::Result<()> {
        execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine),)?;
        Ok(())
    }
}
