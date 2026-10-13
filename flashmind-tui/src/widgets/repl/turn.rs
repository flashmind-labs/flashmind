//! Turn handling: the input loop, streamed events and tool progress.

use std::future::Future;
use std::io;
use std::time::Duration;

use futures::{Stream, StreamExt};
use ratatui::crossterm::event::{self, Event};
use ratatui::layout::Rect;
use ratatui::text::Line;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use flashmind_types::AgentEvent;
use flashmind_types::llm::{ModelPricing, TokenUsage};

use super::render::chat_area;
use super::{KeyAction, PastedImage, Repl, ReplEvent};
use crate::event_render::{RenderAction, apply_to_lines};

impl<'a> Repl<'a> {
    /// Read a line of input from the user.
    ///
    /// Enters raw terminal mode and renders a [`TextArea`](super::TextArea) widget
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
                Event::Mouse(mouse) => {
                    self.handle_mouse(mouse);
                    self.draw_input()?;
                }
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
                        self.shell_mode = false;
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
    /// the [`EventRenderer`](crate::event_render::EventRenderer).  The stream typically comes from a single
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
        // Stream events arrive per token; draw at most once per frame.
        let mut frame = tokio::time::interval(FRAME);
        frame.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut dirty = false;
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
                    self.apply_event(&event)?;
                    dirty = true;
                    if done {
                        self.cancel_token = None;
                        self.activity = None;
                        self.subagent_progress.clear_finished();
                        self.draw_input()?;
                        break;
                    }
                }
                _ = frame.tick(), if dirty => {
                    dirty = false;
                    self.draw_input()?;
                }
                _ = tick.tick() => {
                    self.tick_animations();
                    let actions = self.renderer.tick_tool();
                    self.apply_actions(&actions)?;
                    dirty = false;
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
        self.apply_event(event)?;
        self.draw_input()
    }

    /// Apply an event to the transcript without drawing.
    fn apply_event(&mut self, event: &AgentEvent) -> io::Result<()> {
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
        self.apply_actions(&actions)
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

    /// Add a progress line without drawing; the caller redraws once per frame.
    fn render_progress_line(&mut self, tool_id: Option<&str>, line: String) -> io::Result<()> {
        match tool_id {
            Some(id) => self.apply_event(&AgentEvent::ToolProgress {
                id: id.to_string(),
                line,
            }),
            None => self.commit_lines(&[Line::from(line)]),
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
        let mut frame = tokio::time::interval(FRAME);
        frame.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut dirty = false;
        loop {
            tokio::select! {
                event = next_terminal_event() => { self.handle_live_event(event?, cancel)?; }
                result = &mut fut => {
                    progress_rx.close();
                    while let Some(line) = progress_rx.recv().await {
                        self.render_progress_line(tool_id.as_deref(), line)?;
                    }
                    self.cancel_token = None;
                    self.draw_input()?;
                    return Ok(result);
                }
                Some(line) = progress_rx.recv() => {
                    self.render_progress_line(tool_id.as_deref(), line)?;
                    dirty = true;
                }
                _ = frame.tick(), if dirty => {
                    dirty = false;
                    self.draw_input()?;
                }
                _ = tick.tick() => {
                    self.tick_animations();
                    let actions = self.renderer.tick_tool();
                    self.apply_actions(&actions)?;
                    dirty = false;
                    self.draw_input()?;
                }
            }
        }
    }

    /// Add a styled line to the transcript and redraw the visible region.
    pub fn println(&mut self, line: Line<'static>) -> io::Result<()> {
        self.commit_lines(&[line])?;
        self.draw_input()
    }

    /// Set the renderer width for right-aligned elapsed times and separators.
    pub fn set_renderer_width(&mut self, width: usize) {
        self.renderer.set_width(width);
    }

    /// Add one request's usage to the summary shown when the turn ends.
    pub fn add_usage(&mut self, usage: &TokenUsage, pricing: Option<&ModelPricing>) {
        self.renderer.add_usage(usage, pricing);
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
        let expand = self.renderer.expand_tools();
        for action in actions {
            if let RenderAction::FinishTool {
                erase_count,
                output,
            } = action
            {
                let keep = self.live_output.len().saturating_sub(*erase_count);
                self.live_output.truncate(keep);
                let earlier = std::mem::take(&mut self.live_output);
                self.commit_lines(&earlier)?;
                self.tui.append_tool_output(output.clone())?;
            } else {
                apply_to_lines(&mut self.live_output, std::slice::from_ref(action), expand);
            }
        }
        if !self.renderer.tool_running() && !self.live_output.is_empty() {
            let lines = std::mem::take(&mut self.live_output);
            self.commit_lines(&lines)?;
        }
        Ok(())
    }

    pub(super) fn commit_lines(&mut self, lines: &[Line<'static>]) -> io::Result<()> {
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

    fn clear_screen(&mut self) -> io::Result<()> {
        self.tui.suspend()?;
        self.draw_input()
    }
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

/// Shortest gap between streamed redraws (about 60 frames per second).
const FRAME: Duration = Duration::from_millis(16);
