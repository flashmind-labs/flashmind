//! Turn handling: the input loop, streamed events and tool progress.

use std::future::Future;
use std::io;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use ratatui::crossterm::event::{self, Event, KeyEvent};
use ratatui::layout::Rect;
use ratatui::text::Line;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use flashmind_types::AgentEvent;
use flashmind_types::llm::{ModelPricing, TokenUsage};

use super::render::chat_area;
use super::stream_output::StreamOutput;
use super::{ForegroundCommand, KeyAction, PastedImage, Repl, ReplEvent};
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
        self.read_input_with(|_| Ok(()))
    }

    /// Read input while polling application events between terminal updates.
    pub fn read_input_with(
        &mut self,
        mut on_tick: impl FnMut(&mut Self) -> io::Result<()>,
    ) -> io::Result<ReplEvent> {
        self.read_input_until(|repl| {
            on_tick(repl)?;
            Ok(false)
        })?
        .ok_or_else(|| io::Error::other("Input polling stopped without an event."))
    }

    /// Read input until submission, quit, or `on_tick` returns true.
    /// Returning `None` preserves the draft and queued user input.
    pub fn read_input_until(
        &mut self,
        mut on_tick: impl FnMut(&mut Self) -> io::Result<bool>,
    ) -> io::Result<Option<ReplEvent>> {
        if self.quit_requested {
            self.clear_input_bar()?;
            return Ok(Some(ReplEvent::Quit));
        }
        if on_tick(self)? {
            return Ok(None);
        }
        if let Some((text, images)) = self.pending_inputs.pop_front() {
            self.echo_input(&text)?;
            self.draw_input()?;
            return Ok(Some(ReplEvent::UserInput(text, images)));
        }
        self.draw_input()?;
        let mut last_tick = std::time::Instant::now();
        loop {
            let timeout = Duration::from_millis(80).saturating_sub(last_tick.elapsed());
            let got_event = event::poll(timeout)?;
            if last_tick.elapsed() >= Duration::from_millis(80) {
                last_tick = std::time::Instant::now();
                if on_tick(self)? {
                    return Ok(None);
                }
                if self.tick_animations() {
                    self.draw_input()?;
                }
            }
            if !got_event {
                continue;
            }
            let event = event::read()?;
            if self.handle_subagent_event(&event)? {
                self.draw_input()?;
                continue;
            }
            match event {
                Event::Mouse(mouse) => {
                    self.handle_mouse(mouse);
                    self.draw_input()?;
                }
                Event::Paste(text) => self.handle_paste(&text),
                Event::Resize(width, _) => {
                    self.renderer.set_width(width.saturating_sub(1) as usize)
                }
                Event::Key(key) => match self.process_key(key) {
                    KeyAction::BackgroundCommand => self.background_command(),
                    KeyAction::OpenSubagents => self.open_subagents()?,
                    KeyAction::Submit { text, images } => {
                        self.echo_input(&text)?;
                        self.draw_input()?;
                        return Ok(Some(ReplEvent::UserInput(text, images)));
                    }
                    KeyAction::Quit => {
                        self.clear_input_bar()?;
                        return Ok(Some(ReplEvent::Quit));
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
        stream: impl Stream<Item = AgentEvent> + Unpin,
    ) -> io::Result<bool> {
        self.stream_events_with(cancel_token, stream, |_, _| {})
            .await
    }

    /// Stream events, updating application state before each event is rendered.
    pub async fn stream_events_with(
        &mut self,
        cancel_token: &CancellationToken,
        mut stream: impl Stream<Item = AgentEvent> + Unpin,
        mut on_event: impl FnMut(&mut Self, &AgentEvent),
    ) -> io::Result<bool> {
        self.cancel_token = Some(cancel_token.clone());
        self.activity.get_or_insert_with(String::new);
        // Tool starts and results can arrive in separate streams.
        self.apply_actions(&[])?;
        self.draw_input()?;
        let mut tick = tokio::time::interval(Duration::from_millis(80));
        // Stream events arrive per token; draw at most once per frame.
        let mut frame = tokio::time::interval(FRAME);
        frame.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut last_frame = tokio::time::Instant::now();
        let mut output = StreamOutput::new(self.config.stream_chars_per_second).with_tool_display(
            self.config.tool_display_interval,
            self.last_tool_display.clone(),
        );
        let mut stream_ended = false;
        let mut dirty = false;
        let mut cancelled = false;
        loop {
            if cancelled || cancel_token.is_cancelled() {
                output.skip_delay();
            }
            dirty |= self.apply_stream_output(&mut output)?;
            if stream_ended && output.is_empty() {
                self.activity = None;
                self.draw_input()?;
                break;
            }
            tokio::select! {
                event = next_terminal_event() => {
                    cancelled |= self.handle_live_event(event?, cancel_token)?;
                }
                event = stream.next(), if !stream_ended && output.can_receive() => {
                    let Some(event) = event else {
                        stream_ended = true;
                        continue;
                    };
                    stream_ended = matches!(event, AgentEvent::Done(_) | AgentEvent::Cancelled | AgentEvent::Error(_));
                    cancelled |= matches!(event, AgentEvent::Cancelled);
                    if matches!(event, AgentEvent::Cancelled | AgentEvent::Error(_)) {
                        output.skip_delay();
                    }
                    on_event(self, &event);
                    output.push(event);
                }
                _ = frame.tick(), if dirty || !output.is_empty() => {
                    let now = tokio::time::Instant::now();
                    output.advance(now.duration_since(last_frame).min(FRAME));
                    last_frame = now;
                    self.apply_stream_output(&mut output)?;
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
        self.cancel_token = None;
        Ok(cancelled)
    }

    fn apply_stream_output(&mut self, output: &mut StreamOutput) -> io::Result<bool> {
        let mut changed = false;
        while let Some(event) = output.next_ready() {
            if let AgentEvent::ToolStart { name, .. } = &event {
                self.activity = Some(name.clone());
            }
            if matches!(
                event,
                AgentEvent::ToolStart { .. }
                    | AgentEvent::ToolResult { .. }
                    | AgentEvent::Interrupted { .. }
            ) {
                self.last_tool_display = output.last_tool_display().cloned();
            }
            self.apply_event(&event)?;
            changed = true;
        }
        Ok(changed)
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
    pub(super) fn apply_event(&mut self, event: &AgentEvent) -> io::Result<()> {
        let response = matches!(event, AgentEvent::TextDelta(text) | AgentEvent::ReasoningDelta(text) if !text.is_empty())
            || matches!(event, AgentEvent::Compacted(_));
        if (response || matches!(event, AgentEvent::ToolStart { .. }))
            && self.tui.hide_tool_preview()?
        {
            self.transcript_layout.invalidate();
        }
        if response {
            self.tui.end_tool_batch()?;
        }
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
        match event {
            AgentEvent::ToolBackgroundReady {
                id,
                request: Some(request),
            } => {
                self.foreground_command = Some(ForegroundCommand {
                    id: id.clone(),
                    request: request.clone(),
                });
            }
            AgentEvent::ToolResult { id, .. }
                if self
                    .foreground_command
                    .as_ref()
                    .is_some_and(|command| command.id == *id) =>
            {
                self.foreground_command = None;
            }
            AgentEvent::Interrupted { tool_call_id, .. }
                if self
                    .foreground_command
                    .as_ref()
                    .is_some_and(|command| command.id == *tool_call_id) =>
            {
                self.foreground_command = None;
            }
            AgentEvent::Done(_) | AgentEvent::Cancelled | AgentEvent::Error(_) => {
                self.foreground_command = None;
            }
            _ => {}
        }
        if matches!(event, AgentEvent::SpawnedEvent { .. }) {
            self.subagent_progress.handle_event(event);
        }
        let actions = self.renderer.render(event);
        self.apply_actions(&actions)
    }

    /// Finish a turn: flush remaining text, show elapsed time, and redraw.
    pub fn finish_turn(&mut self) -> io::Result<()> {
        self.foreground_command = None;
        self.renderer.clear_running_tool();
        let actions = self.renderer.render(&AgentEvent::Done(String::new()));
        self.apply_actions(&actions)?;
        self.activity = None;
        self.cancel_token = None;
        self.draw_input()
    }

    /// Erase a running tool's block so the same call can run again.
    pub fn discard_tool(&mut self, id: &str) -> io::Result<()> {
        let actions = self.renderer.discard_tool(id);
        self.apply_actions(&actions)
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

    /// Run an operation that can safely be dropped on cancellation.
    /// Returns `None` when interrupted, preserving any queued input.
    pub async fn run_cancellable_task<T>(
        &mut self,
        label: &str,
        fut: impl Future<Output = T>,
    ) -> io::Result<Option<T>> {
        let cancel = CancellationToken::new();
        self.set_activity(label);
        let result = self
            .run_tool_ui(&cancel, async {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => None,
                    result = fut => Some(result),
                }
            })
            .await;
        self.clear_activity();
        result
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
        self.print_lines(&[line])
    }

    /// Add unwrapped lines to the transcript with one redraw.
    pub fn print_lines(&mut self, lines: &[Line<'static>]) -> io::Result<()> {
        self.commit_lines(lines)?;
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
        self.apply_event(event)
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

    /// Take a queued message for the next model request. Slash commands and
    /// shell escapes stay queued for `read_input`, along with later inputs.
    pub fn take_pending_message(&mut self) -> Option<(String, Vec<PastedImage>)> {
        if self.quit_requested {
            return None;
        }
        let (text, _) = self.pending_inputs.front()?;
        if text.starts_with(['/', '!']) {
            return None;
        }
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

    pub(super) fn apply_actions(&mut self, actions: &[RenderAction]) -> io::Result<()> {
        let expand = self.renderer.expand_tools();
        for action in actions {
            match action {
                RenderAction::FinishTool {
                    erase_count,
                    output,
                } => {
                    let keep = self.live_output.len().saturating_sub(*erase_count);
                    self.live_output.truncate(keep);
                    let earlier = std::mem::take(&mut self.live_output);
                    self.commit_lines(&earlier)?;
                    self.tui.append_tool_output(output.clone())?;
                    self.transcript_layout.invalidate();
                }
                RenderAction::Reasoning { output, expanded } => {
                    let earlier = std::mem::take(&mut self.live_output);
                    self.commit_lines(&earlier)?;
                    self.tui
                        .append_reasoning_output(output.clone(), *expanded)?;
                }
                _ => apply_to_lines(&mut self.live_output, std::slice::from_ref(action), expand),
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
        if self.handle_subagent_event(&event)? {
            self.draw_input()?;
            return Ok(false);
        }
        let mut cancelled = false;
        match event {
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            Event::Paste(text) => self.handle_paste(&text),
            Event::Resize(width, _) => self.renderer.set_width(width.saturating_sub(1) as usize),
            Event::Key(key) => cancelled = self.handle_live_key(key, cancel)?,
            _ => {}
        }
        self.draw_input()?;
        Ok(cancelled)
    }

    fn handle_live_key(&mut self, key: KeyEvent, cancel: &CancellationToken) -> io::Result<bool> {
        let mut cancelled = false;
        match self.process_key(key) {
            KeyAction::BackgroundCommand => self.background_command(),
            KeyAction::OpenSubagents => self.open_subagents()?,
            KeyAction::Submit { text, images } => self.pending_inputs.push_back((text, images)),
            KeyAction::Quit => {
                self.quit_requested = true;
                cancel.cancel();
                self.cancel_token = None;
                cancelled = true;
            }
            KeyAction::Escape | KeyAction::Interrupt => {
                cancel.cancel();
                self.cancel_token = None;
                cancelled = true;
            }
            KeyAction::ClearScreen => self.clear_screen()?,
            _ => {}
        }
        Ok(cancelled)
    }

    pub(super) fn background_command(&mut self) {
        if let Some(command) = self.foreground_command.take() {
            command.request.request();
            self.toast = Some(("Moving command to background".into(), self.tick));
        }
    }

    pub(super) fn clear_screen(&mut self) -> io::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReplConfig;

    #[test]
    fn fast_completion_stays_partial_until_the_display_catches_up() {
        let mut repl = Repl::new(ReplConfig::default());
        let mut output = StreamOutput::new(100);
        let reply = "A complete answer.";
        output.push(AgentEvent::TextDelta(reply.into()));
        output.push(AgentEvent::Done(reply.into()));

        output.advance(Duration::from_millis(20));
        assert!(repl.apply_stream_output(&mut output).unwrap());
        assert_eq!(repl.renderer.pending_text(), "A ");
        assert!(repl.tui.transcript_lines().is_empty());
        assert!(!output.is_empty());

        output.advance(Duration::from_millis(160));
        repl.apply_stream_output(&mut output).unwrap();
        assert!(output.is_empty());
        assert!(repl.renderer.pending_text().is_empty());
        let transcript = repl
            .tui
            .transcript_lines()
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(transcript.contains(reply), "{transcript}");
    }

    #[test]
    fn a_completed_command_stays_visible_until_the_next_tool_starts() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.apply_event(&AgentEvent::ToolStart {
            id: "first".into(),
            name: "exec".into(),
            humanized: "Running `printf example`".into(),
        })
        .unwrap();
        repl.apply_event(&AgentEvent::ToolResult {
            id: "first".into(),
            name: "exec".into(),
            output: "[exit code: 0]\nexample output".into(),
            success: true,
            elapsed_ms: 20,
            sources: Vec::new(),
        })
        .unwrap();
        repl.replay_finish_turn().unwrap();
        let text = |repl: &Repl<'_>| {
            repl.tui
                .transcript_lines()
                .iter()
                .map(Line::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        };
        let completed = text(&repl);
        assert!(completed.contains("printf example"), "{completed}");
        assert!(completed.contains("example output"), "{completed}");
        assert!(!repl.renderer.tool_running());

        repl.apply_event(&AgentEvent::ToolStart {
            id: "second".into(),
            name: "exec".into(),
            humanized: "Running `printf next`".into(),
        })
        .unwrap();
        let next = text(&repl);
        assert!(!next.contains("printf example"), "{next}");
        assert!(!next.contains("example output"), "{next}");
        assert!(next.contains("Ran 1 command"), "{next}");
        repl.set_expand_tools(true).unwrap();
        assert!(text(&repl).contains("printf example"));
    }

    #[test]
    fn thinking_and_reply_text_collapse_the_previous_command_preview() {
        for event in [
            AgentEvent::ReasoningDelta("Considering the result.".into()),
            AgentEvent::TextDelta("Here is the result.".into()),
        ] {
            let mut repl = Repl::new(ReplConfig::default());
            repl.apply_event(&AgentEvent::ToolResult {
                id: "command".into(),
                name: "exec".into(),
                output: "Example command output".into(),
                success: true,
                elapsed_ms: 20,
                sources: Vec::new(),
            })
            .unwrap();
            let text = |repl: &Repl<'_>| {
                repl.tui
                    .transcript_lines()
                    .iter()
                    .map(Line::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            assert!(text(&repl).contains("Example command output"));
            repl.apply_event(&AgentEvent::ReasoningDelta(String::new()))
                .unwrap();
            assert!(text(&repl).contains("Example command output"));
            repl.apply_event(&event).unwrap();
            repl.replay_finish_turn().unwrap();
            assert!(!text(&repl).contains("Example command output"));
            assert!(text(&repl).contains("Ran 1 command"));
            repl.set_expand_tools(true).unwrap();
            assert!(text(&repl).contains("Example command output"));
        }
    }

    #[test]
    fn ctrl_d_while_busy_cancels_and_quits_before_queued_input_or_notifications() {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};

        let mut repl = Repl::new(ReplConfig::default());
        repl.pending_inputs
            .push_back(("queued message".into(), Vec::new()));
        let cancel = CancellationToken::new();
        assert!(
            repl.handle_live_key(
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
                &cancel,
            )
            .unwrap()
        );
        assert!(cancel.is_cancelled());
        assert!(repl.take_pending_message().is_none());
        assert!(matches!(
            repl.read_input_until(|_| panic!("quit must precede notifications"))
                .unwrap(),
            Some(ReplEvent::Quit)
        ));
    }

    #[test]
    fn escape_while_busy_cancels_without_discarding_queued_input() {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};

        let mut repl = Repl::new(ReplConfig::default());
        repl.pending_inputs
            .push_back(("queued message".into(), Vec::new()));
        let cancel = CancellationToken::new();
        assert!(
            repl.handle_live_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &cancel)
                .unwrap()
        );
        assert!(cancel.is_cancelled());
        assert_eq!(repl.take_pending_message().unwrap().0, "queued message");
    }

    #[test]
    fn escape_while_scrolled_up_follows_latest_before_cancelling() {
        use ratatui::crossterm::event::KeyCode;

        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("Draft message");
        repl.transcript_scroll.update(100, 20);
        repl.process_key(KeyEvent::from(KeyCode::PageUp));
        assert!(!repl.transcript_scroll.following);
        let cancel = CancellationToken::new();

        assert!(
            !repl
                .handle_live_key(KeyEvent::from(KeyCode::Esc), &cancel)
                .unwrap()
        );
        assert!(repl.transcript_scroll.following);
        assert_eq!(repl.transcript_scroll.top, repl.transcript_scroll.max);
        assert!(!cancel.is_cancelled());
        assert_eq!(repl.textarea.text(), "Draft message");

        repl.transcript_scroll.update(110, 20);
        assert_eq!(repl.transcript_scroll.top, repl.transcript_scroll.max);
        assert!(
            repl.handle_live_key(KeyEvent::from(KeyCode::Esc), &cancel)
                .unwrap()
        );
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn ctrl_b_backgrounds_the_command_without_cancelling_or_changing_input() {
        use flashmind_types::tool::BackgroundRequest;
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("Draft message");
        repl.pending_inputs
            .push_back(("Queued message".into(), Vec::new()));
        let request = BackgroundRequest::default();
        repl.apply_event(&AgentEvent::ToolBackgroundReady {
            id: "command".into(),
            request: Some(request.clone()),
        })
        .unwrap();
        let cancel = CancellationToken::new();
        repl.cancel_token = Some(cancel.clone());
        assert!(matches!(
            repl.process_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL)),
            KeyAction::BackgroundCommand
        ));
        repl.background_command();
        tokio::time::timeout(Duration::from_secs(5), request.requested())
            .await
            .unwrap();
        assert!(!cancel.is_cancelled());
        assert!(repl.cancel_token.is_some());
        assert!(repl.foreground_command.is_none());
        assert_eq!(repl.textarea.text(), "Draft message");
        assert_eq!(repl.take_pending_input().unwrap().0, "Queued message");
    }

    #[test]
    fn completed_and_interrupted_commands_clear_the_background_shortcut() {
        use flashmind_types::tool::BackgroundRequest;

        for event in [
            AgentEvent::ToolResult {
                name: "exec".into(),
                id: "command".into(),
                output: "done".into(),
                success: true,
                elapsed_ms: 0,
                sources: Vec::new(),
            },
            AgentEvent::Interrupted {
                tool_call_id: "command".into(),
                tool_name: "exec".into(),
                output: "approval required".into(),
                payload: None,
            },
            AgentEvent::Cancelled,
        ] {
            let mut repl = Repl::new(ReplConfig::default());
            repl.apply_event(&AgentEvent::ToolBackgroundReady {
                id: "command".into(),
                request: Some(BackgroundRequest::default()),
            })
            .unwrap();
            repl.apply_event(&event).unwrap();
            assert!(repl.foreground_command.is_none());
        }
    }

    #[test]
    fn background_wake_preserves_draft_and_queued_input() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("Unsubmitted draft");
        repl.pending_inputs
            .push_back(("Queued message".into(), Vec::new()));
        assert!(repl.read_input_until(|_| Ok(true)).unwrap().is_none());
        assert_eq!(repl.textarea.text(), "Unsubmitted draft");
        assert_eq!(repl.take_pending_input().unwrap().0, "Queued message");
    }

    #[test]
    fn queued_commands_wait_for_dispatch_and_preserve_input_order() {
        for command in [
            "/autocompact 300000",
            "/compact",
            "/example args",
            "!echo hello",
        ] {
            let mut repl = Repl::new(ReplConfig::default());
            repl.pending_inputs.push_back(("first".into(), Vec::new()));
            repl.pending_inputs.push_back((command.into(), Vec::new()));
            repl.pending_inputs.push_back((
                "later".into(),
                vec![PastedImage {
                    media_type: "image/png".into(),
                    data: "cG5n".into(),
                }],
            ));

            assert_eq!(repl.take_pending_message().unwrap().0, "first");
            assert!(repl.take_pending_message().is_none());
            assert!(repl.take_pending_message().is_none());
            assert!(repl.has_pending_input());
            assert_eq!(repl.take_pending_input().unwrap().0, command);

            let (text, images) = repl.take_pending_message().unwrap();
            assert_eq!(text, "later");
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].data, "cG5n");
            assert!(!repl.has_pending_input());
        }
    }
}
