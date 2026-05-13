//! Full-featured TUI application for the CLI.
//!
//! Copied from the agent's TUI with agent-specific features stripped.
//! Stays in raw mode, maintains a scrollback buffer with in-place tool
//! updates, and redraws the input widget on every tick for live elapsed
//! times and spinner animation.

mod events;
mod input;

pub use events::TuiState;

use ratatui::crossterm::event::{DisableBracketedPaste, EnableBracketedPaste, Event};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use std::io::{self, Write};

use flashmind_tui::styles::*;
use flashmind_tui::term;
use flashmind_tui::TextArea;
use tokio::sync::mpsc;
use unicode_width::UnicodeWidthStr;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Action returned by the TUI event handler.
pub enum TuiAction {
    Submit(String),
    Cancel,
    Quit,
    None,
}

/// Data from an interrupted tool call (tool requested user approval).
pub struct StreamInterrupt {
    pub tool_call_id: String,
    pub output: String,
}

/// Braille spinner frames.
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

// ---------------------------------------------------------------------------
// Sub-state structs
// ---------------------------------------------------------------------------

/// A single history record.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct HistRecord {
    pub content: String,
}

/// Command history state.
#[derive(Default)]
pub(super) struct History {
    pub entries: Vec<HistRecord>,
    pub pos: Option<usize>,
    pub saved_input: String,
    pub path: Option<std::path::PathBuf>,
}

/// Reverse search state (Ctrl+R).
#[derive(Default)]
pub(super) struct Search {
    pub active: bool,
    pub query: String,
    pub match_idx: Option<usize>,
    pub saved_input: String,
}

/// Tab completion state for /commands.
#[derive(Default)]
pub(super) struct Completion {
    pub candidates: Vec<String>,
    pub idx: usize,
}

/// Status bar state.
#[derive(Default)]
pub(super) struct StatusBar {
    pub base: String,
    pub extra: String,
    pub spinner_active: bool,
    pub spinner_tick: usize,
    pub toast: Option<(String, usize)>,
    pub tick_count: usize,
}

impl StatusBar {
    fn display_text(&self) -> String {
        match (self.base.is_empty(), self.extra.is_empty()) {
            (true, true) => String::new(),
            (false, true) => self.base.clone(),
            (true, false) => self.extra.clone(),
            (false, false) => format!("{} | {}", self.base, self.extra),
        }
    }
}

// ---------------------------------------------------------------------------
// TuiApp
// ---------------------------------------------------------------------------

pub struct TuiApp<'a> {
    /// All output lines (scrollback buffer).
    pub(crate) lines: Vec<Line<'static>>,
    /// Input text area.
    textarea: TextArea<'a>,
    /// Terminal width at the last draw.
    cached_width: u16,
    /// Lines already emitted to the terminal.
    printed: usize,
    needs_full_redraw: bool,
    last_input_height: u16,
    /// Rows the cursor was moved down from anchor.
    cursor_rows_from_anchor: u16,
    /// Rows to move up before rewriting.
    rewrite_up_rows: u16,

    pub(super) history: History,
    pub(super) search: Search,
    pub(super) completion: Completion,
    pub(super) status: StatusBar,
    /// Known slash commands for tab completion.
    pub(super) slash_commands: Vec<String>,
    /// Last token usage from the most recent turn (total, prompt, completion).
    last_usage: Option<(u32, u32, u32)>,
    /// Context window size in tokens (for percentage display).
    pub(crate) context_window: u32,
}

impl<'a> TuiApp<'a> {
    pub fn new() -> io::Result<Self> {
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(
                io::stdout(),
                DisableBracketedPaste,
                ratatui::crossterm::cursor::Show,
            );
            original_hook(info);
        }));

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnableBracketedPaste)?;

        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("Type your message...");
        textarea.set_cursor_line_style(Style::default());
        textarea.set_block(
            ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::TOP)
                .border_style(S_TEXT)
                .padding(ratatui::widgets::Padding::horizontal(1)),
        );

        Ok(Self {
            lines: Vec::new(),
            textarea,
            cached_width: 0,
            printed: 0,
            needs_full_redraw: false,
            last_input_height: 0,
            cursor_rows_from_anchor: 0,
            rewrite_up_rows: 0,
            history: History::default(),
            search: Search::default(),
            completion: Completion::default(),
            status: StatusBar::default(),
            slash_commands: crate::commands::command_names(),
            last_usage: None,
            context_window: 0,
        })
    }

    fn invalidate(&mut self) {
        self.printed = 0;
        self.needs_full_redraw = true;
    }

    fn request_redraw(&mut self) {
        self.needs_full_redraw = true;
    }

    // ========================================================================
    // Public API
    // ========================================================================

    pub fn last_usage(&self) -> Option<(u32, u32, u32)> {
        self.last_usage
    }

    pub fn set_status(&mut self, status: impl ToString) {
        self.status.base = status.to_string();
    }

    pub fn set_status_extra(&mut self, extra: impl ToString) {
        self.status.extra = extra.to_string();
    }

    pub fn start_spinner(&mut self) {
        self.status.spinner_active = true;
        self.status.spinner_tick = 0;
    }

    pub fn stop_spinner(&mut self) {
        self.status.spinner_active = false;
    }

    pub fn tick(&mut self) {
        self.status.tick_count = self.status.tick_count.wrapping_add(1);
        if self.status.spinner_active {
            self.status.spinner_tick = self.status.spinner_tick.wrapping_add(1);
        }
        if let Some((_, shown_at)) = &self.status.toast
            && self.status.tick_count.wrapping_sub(*shown_at) > 63
        {
            self.status.toast = None;
        }
    }

    /// Update the in-progress tool line with live elapsed time.
    pub fn update_running_tool(&mut self, state: &TuiState) {
        let Some(idx) = state.tool_line else { return };
        let Some((name, args)) = state.tool_info.as_ref() else {
            return;
        };
        let Some(start) = state.tool_start_time else {
            return;
        };
        if idx >= self.lines.len() {
            return;
        }

        let elapsed = format_duration(start.elapsed().as_millis() as u64);
        let w = self.width();
        let max_text = w.saturating_sub(12);
        let display = if args.is_empty() {
            format!("  \u{25cc} {}", name)
        } else {
            let full = format!("  \u{25cc} {}", args);
            if UnicodeWidthStr::width(full.as_str()) > max_text {
                format!(
                    "{}\u{2026}",
                    truncate_display(&full, max_text.saturating_sub(1))
                )
            } else {
                full
            }
        };

        let display_cols = UnicodeWidthStr::width(display.as_str());
        let padded_elapsed = format!("{:>w$}", elapsed, w = w.saturating_sub(display_cols));

        self.update_line(
            idx,
            Line::from(vec![
                Span::styled(display, S_TOOL_RUN),
                Span::styled(padded_elapsed, S_DIM),
            ]),
        );
    }

    pub fn load_history(&mut self, path: &std::path::Path) {
        self.history.path = Some(path.to_path_buf());
        if let Ok(content) = std::fs::read_to_string(path) {
            self.history.entries = content
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    serde_json::from_str::<HistRecord>(line).ok().or_else(|| {
                        Some(HistRecord {
                            content: line.to_string(),
                        })
                    })
                })
                .collect();
        }
    }

    pub(super) fn append_history(&self, record: &HistRecord) {
        let Some(path) = &self.history.path else {
            return;
        };
        if let Ok(json) = serde_json::to_string(record)
            && let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
        {
            let _ = writeln!(file, "{}", json);
        }
    }

    pub fn handle_event(&mut self, event: Event) -> TuiAction {
        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Paste(text) => {
                self.textarea.insert_str(&text);
                TuiAction::None
            }
            Event::Resize(w, h) => {
                self.handle_resize(w, h);
                TuiAction::None
            }
            _ => TuiAction::None,
        }
    }

    /// Drain all pending events, returning actionable TuiActions.
    pub fn drain_events(
        &mut self,
        rx: &mut mpsc::UnboundedReceiver<Event>,
        first: Event,
    ) -> Vec<TuiAction> {
        let mut actions = Vec::new();

        let a = self.handle_event(first);
        if !matches!(a, TuiAction::None) {
            actions.push(a);
        }

        while let Ok(ev) = rx.try_recv() {
            let a = self.handle_event(ev);
            if !matches!(a, TuiAction::None) {
                actions.push(a);
            }
        }

        actions
    }

    // ========================================================================
    // Draw
    // ========================================================================

    pub fn draw(&mut self, state: Option<&TuiState>) -> io::Result<()> {
        use ratatui::crossterm::{
            cursor::{Hide, MoveToColumn, MoveUp},
            queue,
            terminal::{Clear, ClearType},
        };

        // Update input block style based on content prefix.
        let first_line = self
            .textarea
            .lines()
            .first()
            .cloned()
            .unwrap_or_default();
        if !self.search.active {
            if first_line.starts_with('/') {
                self.textarea.set_block(self.command_input_block());
            } else {
                self.textarea.set_block(self.default_input_block());
            }
        }

        self.textarea.set_placeholder_text("Type your message...");

        let (term_w, term_h) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let width_changed = term_w != self.cached_width;
        let full_redraw = width_changed || self.needs_full_redraw;

        if full_redraw {
            let new_w = (term_w as usize).saturating_sub(1);
            for line in &mut self.lines {
                if is_separator_line(line) {
                    *line = make_separator(new_w);
                } else {
                    repad_tool_line(line, new_w);
                }
            }
            self.cached_width = term_w;
        }

        let spinner = if self.status.spinner_active {
            Some(SPINNER[self.status.spinner_tick % SPINNER.len()])
        } else {
            None
        };

        let subagent_lines = state
            .map(|s| events::render_subagent_progress(s, term_w as usize))
            .unwrap_or_default();

        let subagent_h = subagent_lines.len() as u16;
        let elapsed_ms = state.and_then(|s| s.start_time.map(|t| t.elapsed().as_millis() as u64));

        let textarea_w_for_height = term_w.saturating_sub(2);
        let visual_rows_for_input = self.textarea.visual_line_count(textarea_w_for_height);

        let companion_rows = subagent_h + 1 + 1; // sep + status
        let max_textarea_h = term_h.saturating_sub(companion_rows).max(3);
        let textarea_height = (visual_rows_for_input as u16 + 1).clamp(3, max_textarea_h);
        let textarea_inner_height = textarea_height.saturating_sub(1); // TOP border
        let textarea_inner_width = term_w.saturating_sub(2);
        self.textarea
            .ensure_cursor_visible(textarea_inner_width, textarea_inner_height);

        let stdout = io::stdout();
        let mut out = stdout.lock();

        queue!(&mut out, Hide)?;

        if self.cursor_rows_from_anchor > 0 {
            queue!(
                &mut out,
                MoveUp(self.cursor_rows_from_anchor),
                MoveToColumn(0)
            )?;
            self.cursor_rows_from_anchor = 0;
        }

        self.emit_lines(&mut out, full_redraw, width_changed, term_w, term_h)?;

        let status_text = self.status.display_text();
        let toast_msg = self.status.toast.as_ref().map(|(msg, _)| msg.clone());
        let input_total_h = subagent_h + 1 + 1 + textarea_height;

        let widget = InputAreaWidget {
            subagent_lines: &subagent_lines,
            status: &status_text,
            spinner,
            toast: toast_msg.as_deref(),
            elapsed_ms,
            textarea: &self.textarea,
            textarea_height,
        };
        term::render_widget_to_stdout(&mut out, widget, term_w, input_total_h)?;

        if self.last_input_height > input_total_h {
            queue!(
                &mut out,
                ratatui::crossterm::cursor::MoveDown(1),
                MoveToColumn(0),
                Clear(ClearType::FromCursorDown),
            )?;
            queue!(&mut out, MoveUp(1))?;
        }
        self.last_input_height = input_total_h;

        // Return cursor to the anchor (top-left of input area).
        if input_total_h > 1 {
            queue!(&mut out, MoveUp(input_total_h - 1))?;
        }
        queue!(&mut out, MoveToColumn(0))?;

        // Position cursor inside textarea.
        let ta_y = subagent_h + 1 + 1; // sep + status
        let textarea_area = ratatui::layout::Rect::new(0, ta_y, term_w, textarea_height);
        if let Some((cx, cy)) = self.textarea.cursor_screen_pos(textarea_area) {
            if cy > 0 {
                queue!(&mut out, ratatui::crossterm::cursor::MoveDown(cy))?;
            }
            queue!(
                &mut out,
                MoveToColumn(cx),
                ratatui::crossterm::cursor::Show,
            )?;
            self.cursor_rows_from_anchor = cy;
        }

        out.flush()?;
        Ok(())
    }

    fn emit_lines<W: io::Write>(
        &mut self,
        out: &mut W,
        full_redraw: bool,
        width_changed: bool,
        term_w: u16,
        term_h: u16,
    ) -> io::Result<()> {
        use ratatui::crossterm::{
            cursor::{MoveTo, MoveToColumn, MoveUp},
            queue,
            terminal::{Clear, ClearType},
        };

        if full_redraw {
            self.needs_full_redraw = false;

            if self.printed == 0 {
                queue!(out, Clear(ClearType::All), MoveTo(0, 0))?;
                for line in &self.lines {
                    term::print_line(out, line)?;
                }
            } else if width_changed {
                let visible_rows = term_h.saturating_sub(self.last_input_height);
                if visible_rows > 0 {
                    queue!(out, MoveUp(visible_rows), MoveToColumn(0))?;
                } else {
                    queue!(out, MoveToColumn(0))?;
                }
                queue!(out, Clear(ClearType::FromCursorDown))?;

                let mut remaining = visible_rows;
                let mut start_idx = self.printed;
                for i in (0..self.printed).rev() {
                    let h = term::visual_height(&self.lines[i], term_w);
                    if h > remaining {
                        break;
                    }
                    remaining = remaining.saturating_sub(h);
                    start_idx = i;
                }
                for line in &self.lines[start_idx..] {
                    term::print_line(out, line)?;
                }
            } else {
                if self.rewrite_up_rows > 0 {
                    queue!(
                        out,
                        MoveUp(self.rewrite_up_rows),
                        MoveToColumn(0),
                        Clear(ClearType::FromCursorDown)
                    )?;
                    self.rewrite_up_rows = 0;
                } else {
                    queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
                }
                for line in &self.lines[self.printed..] {
                    term::print_line(out, line)?;
                }
            }
        } else if self.printed < self.lines.len() {
            queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
            for line in &self.lines[self.printed..] {
                term::print_line(out, line)?;
            }
        } else {
            queue!(out, MoveToColumn(0))?;
        }

        self.printed = self.lines.len();
        Ok(())
    }

    pub fn update_line(&mut self, idx: usize, new_line: Line<'static>) {
        if idx >= self.lines.len() {
            return;
        }
        if self.printed > self.lines.len() {
            self.lines[idx] = new_line;
            self.invalidate();
            return;
        }
        if idx >= self.printed {
            self.lines[idx] = new_line;
            return;
        }

        use ratatui::crossterm::{
            cursor::{MoveDown, MoveToColumn, MoveUp, RestorePosition, SavePosition},
            queue,
            terminal::{Clear, ClearType},
        };

        let w = if self.cached_width == 0 {
            ratatui::crossterm::terminal::size()
                .map(|(w, _)| w)
                .unwrap_or(80)
                .max(1)
        } else {
            self.cached_width
        };
        let h_old = term::visual_height(&self.lines[idx], w);
        let h_new = term::visual_height(&new_line, w);

        let rows_below: u16 = self.lines[idx + 1..self.printed]
            .iter()
            .map(|l| term::visual_height(l, w))
            .sum();
        let rows_up = rows_below + h_old;

        let term_h = ratatui::crossterm::terminal::size()
            .map(|(_, h)| h)
            .unwrap_or(24);
        if rows_up >= term_h {
            self.lines[idx] = new_line;
            self.request_redraw();
            return;
        }

        self.lines[idx] = new_line;

        let stdout = io::stdout();
        let mut out = stdout.lock();

        if self.cursor_rows_from_anchor > 0 {
            let _ = queue!(
                &mut out,
                MoveUp(self.cursor_rows_from_anchor),
                MoveToColumn(0)
            );
            self.cursor_rows_from_anchor = 0;
        }

        let res: io::Result<()> = (|| {
            if h_new != h_old {
                if rows_up > 0 {
                    queue!(&mut out, MoveUp(rows_up), MoveToColumn(0))?;
                } else {
                    queue!(&mut out, MoveToColumn(0))?;
                }
                queue!(&mut out, Clear(ClearType::FromCursorDown))?;
                for line in &self.lines[idx..self.printed] {
                    term::print_line(&mut out, line)?;
                }
                return out.flush();
            }

            queue!(&mut out, SavePosition)?;
            if rows_up > 0 {
                queue!(&mut out, MoveUp(rows_up), MoveToColumn(0))?;
            } else {
                queue!(&mut out, MoveToColumn(0))?;
            }
            for _ in 0..h_old {
                queue!(&mut out, Clear(ClearType::CurrentLine), MoveDown(1))?;
            }
            queue!(&mut out, MoveUp(h_old), MoveToColumn(0))?;
            term::print_line(&mut out, &self.lines[idx])?;
            queue!(&mut out, RestorePosition)?;
            out.flush()
        })();
        if res.is_err() {
            self.request_redraw();
        }
    }

    fn handle_resize(&mut self, _w: u16, _h: u16) {
        self.request_redraw();
        self.cached_width = 0;
    }

    // ========================================================================
    // Output
    // ========================================================================

    pub(crate) fn width(&self) -> usize {
        ratatui::crossterm::terminal::size()
            .map(|(w, _)| (w as usize).saturating_sub(1))
            .unwrap_or(79)
    }

    pub fn push_styled(&mut self, text: &str, style: Style) {
        if text.is_empty() {
            return;
        }
        for (i, chunk) in text.split('\n').enumerate() {
            if i > 0 {
                self.lines.push(Line::default());
            }
            if !chunk.is_empty() {
                if let Some(last) = self.lines.last_mut() {
                    last.spans.push(Span::styled(chunk.to_string(), style));
                } else {
                    self.lines
                        .push(Line::from(Span::styled(chunk.to_string(), style)));
                }
            }
        }
    }

    pub fn push_line(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }

    pub fn newline(&mut self) {
        self.lines.push(Line::default());
    }

    pub(crate) fn separator(&mut self) {
        self.push_line(make_separator(self.width()));
    }

    pub fn add_user_message(&mut self, text: &str) {
        let prefix = "you> ";
        let indent = "     ";
        let w = self.width().saturating_sub(prefix.len());

        for (i, line) in text.split('\n').enumerate() {
            if line.is_empty() {
                let tag = if i == 0 { prefix } else { indent };
                self.push_line(Line::from(Span::styled(tag.to_string(), S_USER)));
                continue;
            }

            let mut remaining = line;
            let mut first = true;
            while !remaining.is_empty() {
                let chunk = if w == 0 {
                    remaining
                } else {
                    let end = remaining
                        .char_indices()
                        .nth(w)
                        .map(|(idx, _)| idx)
                        .unwrap_or(remaining.len());
                    let head = &remaining[..end];
                    if end < remaining.len() {
                        head.rfind(' ').map(|p| &head[..p + 1]).unwrap_or(head)
                    } else {
                        head
                    }
                };
                let rest = &remaining[chunk.len()..];
                let tag = if first && i == 0 { prefix } else { indent };
                self.push_line(Line::from(vec![
                    Span::styled(tag.to_string(), S_USER),
                    Span::styled(chunk.to_string(), S_TEXT),
                ]));
                remaining = rest;
                first = false;
            }
        }
    }

    #[allow(dead_code)]
    pub fn clear_lines(&mut self) {
        self.lines.clear();
        self.invalidate();
    }

    pub fn add_system_message(&mut self, text: &str) {
        self.newline();
        self.push_styled(text, S_DIM);
    }

    // ========================================================================
    // Stream response (event loop)
    // ========================================================================

    pub async fn stream_response<S, F>(
        &mut self,
        mut stream: std::pin::Pin<Box<S>>,
        state: &mut TuiState,
        key_rx: &mut mpsc::UnboundedReceiver<Event>,
        mut on_event: F,
    ) -> io::Result<Option<StreamInterrupt>>
    where
        S: futures::Stream<Item = flashmind_types::AgentEvent> + ?Sized,
        F: FnMut(&flashmind_types::AgentEvent),
    {
        self.start_spinner();
        self.newline();

        let tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));
        tokio::pin!(tick_interval);

        let mut interrupted = None;

        loop {
            tokio::select! {
                biased;
                Some(ev) = key_rx.recv() => {
                    for action in self.drain_events(key_rx, ev) {
                        match action {
                            TuiAction::Cancel | TuiAction::Quit => {
                                self.stop_spinner();
                                self.add_system_message("[cancelled]");
                                self.draw(Some(state))?;
                                return Ok(None);
                            }
                            _ => {}
                        }
                    }
                    self.draw(Some(state))?;
                }
                event = futures::StreamExt::next(&mut stream) => {
                    match event {
                        Some(ev) => {
                            on_event(&ev);
                            let is_done = matches!(ev, flashmind_types::AgentEvent::Done(_));
                            if let flashmind_types::AgentEvent::Interrupted {
                                ref tool_call_id,
                                ref output,
                                ..
                            } = ev
                            {
                                interrupted = Some(StreamInterrupt {
                                    tool_call_id: tool_call_id.clone(),
                                    output: output.clone(),
                                });
                            }
                            self.handle_agent_event(&ev, state);
                            self.draw(Some(state))?;
                            if is_done || interrupted.is_some() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
                _ = tick_interval.tick() => {
                    self.tick();
                    self.update_running_tool(state);
                    self.draw(Some(state))?;
                }
            }
        }

        self.stop_spinner();
        self.newline();
        self.draw(None)?;
        Ok(interrupted)
    }

    /// Replay display events into the line buffer (for session restore).
    pub fn load_display_log(&mut self, events: &[crate::display::DisplayEvent]) {
        let mut state = TuiState::new();
        for event in events {
            match event {
                crate::display::DisplayEvent::Server { msg } => {
                    if let Some(agent_event) =
                        crate::display::server_msg_to_event(msg.clone())
                    {
                        self.handle_agent_event(&agent_event, &mut state);
                    }
                }
                crate::display::DisplayEvent::User { text } => {
                    self.add_user_message(text);
                }
                crate::display::DisplayEvent::System { text } => {
                    self.add_system_message(text);
                }
                crate::display::DisplayEvent::Clear => {
                    self.clear_lines();
                }
            }
        }
    }

    // ========================================================================
    // Read input (event loop)
    // ========================================================================

    pub async fn read_input(
        &mut self,
        key_rx: &mut mpsc::UnboundedReceiver<Event>,
    ) -> io::Result<TuiAction> {
        self.draw(None)?;

        let tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));
        tokio::pin!(tick_interval);

        loop {
            tokio::select! {
                biased;
                ev = key_rx.recv() => {
                    match ev {
                        Some(event) => {
                            let actions = self.drain_events(key_rx, event);
                            for action in actions {
                                match action {
                                    TuiAction::None => {}
                                    other => {
                                        self.draw(None)?;
                                        return Ok(other);
                                    }
                                }
                            }
                            self.draw(None)?;
                        }
                        None => return Ok(TuiAction::Quit),
                    }
                }
                _ = tick_interval.tick() => {
                    self.tick();
                    self.draw(None)?;
                }
            }
        }
    }
}

impl Drop for TuiApp<'_> {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableBracketedPaste,
            ratatui::crossterm::cursor::Show,
        );
    }
}

// ---------------------------------------------------------------------------
// Input area widget
// ---------------------------------------------------------------------------

struct InputAreaWidget<'a> {
    subagent_lines: &'a [Line<'a>],
    status: &'a str,
    spinner: Option<char>,
    toast: Option<&'a str>,
    elapsed_ms: Option<u64>,
    textarea: &'a TextArea<'a>,
    textarea_height: u16,
}

impl<'a> ratatui::widgets::Widget for InputAreaWidget<'a> {
    fn render(self, area: ratatui::layout::Rect, buf: &mut ratatui::buffer::Buffer) {
        use ratatui::layout::{Constraint, Direction, Layout};
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;

        if area.height == 0 || area.width == 0 {
            return;
        }

        let subagent_h = self.subagent_lines.len() as u16;
        let sep_h: u16 = 1;
        let status_h: u16 = 1;
        let textarea_h = self.textarea_height;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(subagent_h),
                Constraint::Length(sep_h),
                Constraint::Length(status_h),
                Constraint::Length(textarea_h),
            ])
            .split(area);
        let subagent_chunk = chunks[0];
        let sep_chunk = chunks[1];
        let status_chunk = chunks[2];
        let textarea_chunk = chunks[3];

        // Subagent progress
        if subagent_h > 0 {
            Paragraph::new(self.subagent_lines.to_vec()).render(subagent_chunk, buf);
        }

        // Horizontal separator
        let sep_line = "\u{2500}".repeat(sep_chunk.width as usize);
        Paragraph::new(Line::from(Span::styled(sep_line, S_TEXT))).render(sep_chunk, buf);

        // Status bar
        let status_content = match self.spinner {
            Some(ch) => format!(" {} {}", ch, self.status),
            None => format!(" {}", self.status),
        };
        let status_style = if self.spinner.is_some() {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default()
        };

        let mut spans = vec![Span::styled(status_content, status_style)];
        if let Some(ms) = self.elapsed_ms {
            spans.push(Span::styled(
                format!(" \u{00b7} {}", format_duration(ms)),
                S_DIM,
            ));
        }
        if let Some(msg) = self.toast {
            let toast_text = format!(" {} ", msg);
            let total_len: usize = spans.iter().map(|s| s.content.len()).sum();
            let gap = (status_chunk.width as usize).saturating_sub(total_len + toast_text.len());
            spans.push(Span::styled(" ".repeat(gap), S_DIM));
            spans.push(Span::styled(toast_text, Style::default().fg(Color::Green)));
        }
        Paragraph::new(Line::from(spans)).render(status_chunk, buf);

        // Textarea
        self.textarea.render(textarea_chunk, buf);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_separator(width: usize) -> Line<'static> {
    let w = width.saturating_sub(2);
    Line::from(Span::styled("─".repeat(w), S_DIM))
}

fn is_separator_line(line: &Line) -> bool {
    line.spans.len() == 1
        && line.spans[0].style == S_DIM
        && !line.spans[0].content.is_empty()
        && line.spans[0].content.chars().all(|c| c == '─')
}

fn repad_tool_line(line: &mut Line<'static>, width: usize) {
    if line.spans.len() != 2 {
        return;
    }
    if line.spans[1].style != S_DIM {
        return;
    }
    let elapsed = line.spans[1].content.trim_start();
    if elapsed.is_empty()
        || !elapsed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.')
    {
        return;
    }
    let tool_len = UnicodeWidthStr::width(line.spans[0].content.as_ref());
    let pad = width.saturating_sub(tool_len + UnicodeWidthStr::width(elapsed));
    let mut padded = String::with_capacity(pad + elapsed.len());
    padded.extend(std::iter::repeat_n(' ', pad));
    padded.push_str(elapsed);
    line.spans[1].content = padded.into();
}

pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let mins = ms / 60_000;
        let secs = (ms % 60_000) / 1000;
        format!("{}m{}s", mins, secs)
    }
}

/// Truncate a string to fit within `max_width` display columns.
pub(crate) fn truncate_display(s: &str, max_width: usize) -> &str {
    use unicode_width::UnicodeWidthChar;
    let mut width = 0;
    for (i, c) in s.char_indices() {
        let cw = c.width().unwrap_or(0);
        if width + cw > max_width {
            return &s[..i];
        }
        width += cw;
    }
    s
}

// ---------------------------------------------------------------------------
// Background key reader
// ---------------------------------------------------------------------------

pub fn spawn_key_reader() -> mpsc::UnboundedReceiver<Event> {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(ev) = ratatui::crossterm::event::read() {
            if tx.send(ev).is_err() {
                break;
            }
        }
    });
    rx
}
