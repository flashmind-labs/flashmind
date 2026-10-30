//! Transcript scrolling, mouse handling, selection and search.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{Event, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Position, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget, Wrap};

use super::input::copy_to_clipboard;
use super::{FoldClick, FooterClick, Repl};
use crate::selection::SelectionAction;
pub(super) use crate::selection::{TextPoint, TextSelection as TranscriptSelection};
use crate::styles;
use crate::transcript::layout_line;

impl<'a> Repl<'a> {
    pub(super) fn handle_mouse(&mut self, mouse: MouseEvent) {
        if matches!(
            mouse.kind,
            MouseEventKind::Down(_) | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            self.dragging_scrollbar = false;
        }
        let position = Position::new(mouse.column, mouse.row);
        // A footer hint fires on release without movement, so dragging across
        // the footer still selects its text.
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.pending_footer_click = self
                    .footer_buttons
                    .iter()
                    .find(|button| button.area.contains(position))
                    .map(|button| FooterClick {
                        key: button.key,
                        at: position,
                    });
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some(click) = self.pending_footer_click.take()
                    && click.at == position
                {
                    self.screen_selection.clear();
                    self.selecting_screen = false;
                    // Footer hints only toggle views or scroll, so the action
                    // needs no handling beyond the redraw after every mouse event.
                    self.process_key(click.key);
                    return;
                }
            }
            MouseEventKind::Drag(_)
                if self
                    .pending_footer_click
                    .is_some_and(|click| click.at != position) =>
            {
                self.pending_footer_click = None;
            }
            _ => {}
        }
        if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
            self.pending_fold_click = None;
            self.selecting_screen = false;
            self.screen_selection.handle_event(&Event::Mouse(mouse));
            let on_scrollbar = mouse.column == self.transcript_area.right()
                && mouse.row >= self.transcript_area.y
                && mouse.row < self.transcript_area.bottom();
            if on_scrollbar {
                self.screen_selection.clear();
            } else if !self.transcript_area.contains(position) {
                self.selection = None;
                self.copy_pending = false;
                self.selecting_screen = true;
                return;
            }
        }
        if matches!(
            mouse.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            self.screen_selection.clear();
            self.selecting_screen = false;
        }
        if mouse.kind == MouseEventKind::Drag(MouseButton::Left)
            && !self.dragging_scrollbar
            && (self.selecting_screen
                || ((mouse.row < self.transcript_area.y
                    || mouse.row >= self.transcript_area.bottom())
                    && self.screen_selection.contains(position)
                    && self.screen_selection.is_dragging()))
        {
            self.selecting_screen = true;
            self.selection = None;
            self.copy_pending = false;
            self.pending_fold_click = None;
        }
        if self.selecting_screen {
            if let Some(action) = self.screen_selection.handle_event(&Event::Mouse(mouse)) {
                self.apply_screen_action(action);
            }
            return;
        }
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left)
                if self
                    .latest_button
                    .is_some_and(|button| button.contains(position)) =>
            {
                self.selection = None;
                self.copy_pending = false;
                self.pending_fold_click = None;
                self.screen_selection.clear();
                self.transcript_scroll.latest();
            }
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
                self.transcript_scroll.mouse(mouse.kind);
            }
            MouseEventKind::Down(MouseButton::Left) if self.transcript_area.contains(position) => {
                let row = self.transcript_scroll.top
                    + mouse.row.saturating_sub(self.transcript_area.y) as usize;
                let window = self.transcript_layout.window(row, 1);
                if self.live_reasoning_row() == Some(row) {
                    self.pending_fold_click = Some(FoldClick::Live);
                } else if window.row_offset == 0 {
                    self.pending_fold_click = Some(FoldClick::Transcript(window.entries.start));
                }
                self.transcript_scroll.following = false;
                let point = self.text_point(position);
                self.selection = Some(TranscriptSelection {
                    start: point,
                    end: point,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) if self.selection.is_some() => {
                self.pending_fold_click = None;
                // Dragging past the top or bottom edge scrolls one row per event.
                let area = self.transcript_area;
                let top = self.transcript_scroll.top;
                if mouse.row < area.y {
                    self.transcript_scroll.jump_to(top.saturating_sub(1));
                } else if mouse.row >= area.bottom() {
                    self.transcript_scroll.jump_to(top + 1);
                }
                let point = self.text_point(Position::new(
                    mouse.column.clamp(area.x, area.right().saturating_sub(1)),
                    mouse.row.clamp(area.y, area.bottom().saturating_sub(1)),
                ));
                if let Some(selection) = self.selection.as_mut() {
                    selection.end = point;
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.screen_selection.clear();
                if let Some(target) = self.pending_fold_click.take() {
                    let toggled = match target {
                        FoldClick::Transcript(entry) => self.tui.toggle_fold_at(entry),
                        FoldClick::Live => {
                            self.renderer
                                .set_expand_reasoning(!self.renderer.expand_reasoning());
                            Ok(true)
                        }
                    };
                    match toggled {
                        Ok(true) => {
                            self.pending_scroll_top = Some(self.transcript_scroll.top);
                            self.transcript_layout.invalidate();
                            self.selection = None;
                            self.copy_pending = false;
                            self.search_match = None;
                            return;
                        }
                        Err(error) => {
                            self.toast = Some((format!("Toggle failed: {error}"), self.tick));
                        }
                        Ok(false) => {}
                    }
                }
                self.copy_selection();
            }
            _ => {}
        }
    }

    pub(super) fn apply_screen_action(&mut self, action: SelectionAction) {
        if let SelectionAction::Copy(text) = action {
            self.pending_screen_copy = Some(text);
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

    /// Transcript rows from `top` laid out for `area`, with the history entry
    /// `highlight` drawn in the accent style.
    pub(super) fn transcript_paragraph(
        &mut self,
        history: &[Line<'static>],
        dynamic: &[Line<'static>],
        top: usize,
        area: Rect,
        highlight: Option<usize>,
    ) -> Paragraph<'static> {
        let window = self.transcript_layout.window(top, area.height as usize);
        let mut visible: Vec<Line<'static>> = Vec::new();
        let mut append = |line: &Line<'static>, highlighted: bool| {
            let mut rows = layout_line(line, area.width.max(1));
            for (position, target) in crate::hyperlinks::layout_links(line, &rows) {
                let row = visible.len() + usize::from(position.y);
                if let Some(row) = row.checked_sub(usize::from(window.row_offset))
                    && row < usize::from(area.height)
                {
                    self.screen_links.insert(
                        Position::new(area.x + position.x, area.y + row as u16),
                        target,
                    );
                }
            }
            if highlighted {
                for row in &mut rows {
                    for span in &mut row.spans {
                        span.style = span.style.patch(styles::S_ACCENT);
                    }
                }
            }
            visible.extend(rows);
        };
        for (index, line) in history[window.entries.clone()].iter().enumerate() {
            append(line, highlight == Some(window.entries.start + index));
        }
        if top + area.height as usize > self.transcript_layout.rows() {
            for line in dynamic {
                append(line, false);
            }
        }
        Paragraph::new(visible)
            .wrap(Wrap { trim: false })
            .scroll((window.row_offset, 0))
    }

    /// Transcript position under a screen cell inside the transcript area.
    fn text_point(&self, position: Position) -> TextPoint {
        TextPoint {
            row: self.transcript_scroll.top
                + position.y.saturating_sub(self.transcript_area.y) as usize,
            column: position.x,
        }
    }

    /// Copy the selection on the next draw, which has the transcript lines.
    pub(super) fn copy_selection(&mut self) {
        // A click without a drag selects nothing.
        if self
            .selection
            .as_ref()
            .is_some_and(|selection| selection.start == selection.end)
        {
            self.selection = None;
        }
        self.copy_pending = self.selection.is_some();
    }

    /// Copy logical lines while using the layout to locate the selected cells.
    fn selected_text(
        &self,
        history: &[Line<'static>],
        dynamic: &[Line<'static>],
        area: Rect,
    ) -> Option<String> {
        let selection = self.selection.as_ref()?;
        let (start, end) = selection.bounds();
        let window = self
            .transcript_layout
            .window(start.row, end.row - start.row + 1);
        let mut first_row = self.transcript_layout.entry_row(window.entries.start);
        let dynamic = if end.row >= self.transcript_layout.rows() {
            dynamic
        } else {
            &[]
        };
        let mut selected = Vec::new();
        for line in history[window.entries].iter().chain(dynamic) {
            if first_row > end.row {
                break;
            }
            let paragraph =
                Paragraph::new(layout_line(line, area.width.max(1))).wrap(Wrap { trim: false });
            let rows = paragraph.line_count(area.width.max(1));
            if first_row + rows > start.row {
                let height = rows.min(u16::MAX as usize) as u16;
                let area = Rect::new(area.x, 0, area.width, height);
                let mut buffer = Buffer::empty(area);
                paragraph.render(area, &mut buffer);
                selected.push(
                    selection
                        .text_from_source(&buffer, first_row, &line.to_string())
                        .unwrap_or_else(|| selection.text(&buffer, first_row)),
                );
            }
            first_row += rows;
        }
        Some(selected.join("\n"))
    }

    pub(super) fn copy_selected_text(
        &mut self,
        history: &[Line<'static>],
        dynamic: &[Line<'static>],
        area: Rect,
    ) {
        if let Some(text) = self.selected_text(history, dynamic, area)
            && !text.is_empty()
        {
            let toast = match copy_to_clipboard(&text) {
                Ok(()) => "Copied selection".to_string(),
                Err(error) => format!("Copy failed: {error}"),
            };
            self.toast = Some((toast, self.tick));
        }
    }
}

/// Direction for Alt+Up and Alt+Down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PromptJump {
    Previous,
    Next,
}

/// History lines that start a user prompt: the first line of an echoed input.
pub(super) fn prompt_lines(lines: &[Line<'static>]) -> impl Iterator<Item = usize> {
    lines.iter().enumerate().filter_map(|(index, line)| {
        let starts_prompt = line.style == styles::S_USER_ECHO
            && line
                .spans
                .first()
                .is_some_and(|span| !span.content.starts_with(' '));
        starts_prompt.then_some(index)
    })
}

/// A search match's 1-based place among every matching transcript line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SearchPosition {
    pub(super) current: usize,
    pub(super) total: usize,
}

/// Where the match at line `index` falls among all lines matching `query`.
pub(super) fn transcript_match_position(
    lines: &[Line<'static>],
    query: &str,
    index: usize,
) -> SearchPosition {
    let query = query.to_lowercase();
    let mut position = SearchPosition {
        current: 0,
        total: 0,
    };
    for (line_index, line) in lines.iter().enumerate() {
        if line_matches(line, &query) {
            position.total += 1;
            if line_index <= index {
                position.current = position.total;
            }
        }
    }
    position
}

fn line_matches(line: &Line<'static>, lowercase_query: &str) -> bool {
    let text: String = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    text.to_lowercase().contains(lowercase_query)
}

pub(super) fn find_transcript_match(
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
        if line_matches(&lines[index], &query) {
            return Some(index);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::repl::{KeyAction, ReplConfig};
    use flashmind_types::AgentEvent;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{self, KeyCode, KeyModifiers};

    fn text_position(buffer: &Buffer, text: &str) -> Position {
        for y in buffer.area.y..buffer.area.bottom() {
            let mut line = String::new();
            let mut offsets = Vec::new();
            for x in buffer.area.x..buffer.area.right() {
                offsets.push(line.len());
                line.push_str(buffer[(x, y)].symbol());
            }
            if let Some(offset) = line.find(text) {
                let column = offsets.partition_point(|&start| start <= offset) - 1;
                return Position::new(buffer.area.x + column as u16, y);
            }
        }
        panic!("the rendered screen contains {text}");
    }

    #[test]
    fn text_positions_use_cells_after_multibyte_and_combining_characters() {
        let buffer = Buffer::with_lines(["\u{e9} e\u{301} target"]);
        assert_eq!(text_position(&buffer, "target"), Position::new(4, 0));
    }

    fn mouse_at(kind: MouseEventKind, position: Position) -> MouseEvent {
        MouseEvent {
            kind,
            column: position.x,
            row: position.y,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn tool_batches_keep_live_output_order_and_expand_after_replay_and_resize() {
        use flashmind_types::tool::DiffLine;

        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("draft");
        let start = |id: &str| AgentEvent::ToolStart {
            id: id.into(),
            name: "exec".into(),
            humanized: format!("command {id}"),
        };
        let result = |id: &str| AgentEvent::ToolResult {
            id: id.into(),
            name: "exec".into(),
            output: format!("output {id}"),
            success: true,
            elapsed_ms: 1,
            sources: Vec::new(),
        };
        repl.replay_event(&start("a")).unwrap();
        repl.replay_event(&start("b")).unwrap();
        repl.replay_event(&result("b")).unwrap();
        assert!(
            repl.live_output
                .iter()
                .any(|line| line.to_string().contains("command a"))
        );
        repl.replay_event(&result("a")).unwrap();
        repl.replay_event(&start("c")).unwrap();
        repl.replay_event(&AgentEvent::ToolProgress {
            id: "c".into(),
            line: "streaming".into(),
        })
        .unwrap();
        assert!(
            repl.tui
                .transcript_lines()
                .iter()
                .any(|line| line.to_string() == "> Ran 2 commands")
        );
        assert!(
            repl.live_output
                .iter()
                .any(|line| line.to_string().contains("streaming"))
        );
        repl.replay_event(&result("c")).unwrap();
        // Empty deltas do not split a batch; actual assistant text does.
        repl.replay_event(&AgentEvent::TextDelta(String::new()))
            .unwrap();
        repl.replay_event(&start("d")).unwrap();
        repl.replay_event(&result("d")).unwrap();
        repl.replay_event(&AgentEvent::TextDelta("assistant reply".into()))
            .unwrap();
        repl.replay_event(&AgentEvent::ToolStart {
            id: "edit".into(),
            name: "file_write".into(),
            humanized: "write example.txt".into(),
        })
        .unwrap();
        repl.replay_event(&AgentEvent::FileDiff {
            path: "example.txt".into(),
            diff: vec![DiffLine::Added {
                line: 1,
                content: "new content".into(),
            }],
        })
        .unwrap();
        repl.replay_event(&AgentEvent::ToolResult {
            id: "edit".into(),
            name: "file_write".into(),
            output: "edit failed".into(),
            success: false,
            elapsed_ms: 1,
            sources: Vec::new(),
        })
        .unwrap();
        let collapsed = repl.tui.transcript_lines();
        let headers: Vec<_> = collapsed
            .iter()
            .map(Line::to_string)
            .filter(|line| line.contains("Ran "))
            .collect();
        assert_eq!(headers, ["> Ran 4 commands", "> Ran 1 tool (1 failed)"]);
        assert!(
            !collapsed
                .iter()
                .any(|line| line.to_string().contains("new content"))
        );

        let mut terminal = Terminal::new(TestBackend::new(45, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &collapsed))
            .unwrap();
        let position = text_position(terminal.backend().buffer(), "Ran 4 commands");
        repl.handle_mouse(mouse_at(MouseEventKind::Down(MouseButton::Left), position));
        repl.handle_mouse(mouse_at(MouseEventKind::Up(MouseButton::Left), position));
        let expanded = repl.tui.transcript_lines();
        let output_lines: Vec<_> = expanded
            .iter()
            .map(Line::to_string)
            .filter(|line| line.trim().starts_with("output "))
            .collect();
        assert_eq!(
            output_lines,
            [
                "    output a",
                "    output b",
                "    output c",
                "    output d"
            ]
        );
        assert!(find_transcript_match(&expanded, "output a", None, false).is_some());
        terminal.backend_mut().resize(25, 24);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &expanded))
            .unwrap();
        repl.process_key(event::KeyEvent::new(
            KeyCode::Char('o'),
            KeyModifiers::CONTROL,
        ));
        repl.process_key(event::KeyEvent::new(
            KeyCode::Char('o'),
            KeyModifiers::CONTROL,
        ));
        assert_eq!(repl.tui.transcript_lines(), collapsed);
        assert!(repl.tui.reveal_tool_matches("edit failed").unwrap());
        let revealed = repl.tui.transcript_lines();
        assert!(
            revealed
                .iter()
                .any(|line| line.to_string().contains("+new content"))
        );
        assert!(
            revealed
                .iter()
                .any(|line| line.to_string().contains("edit failed"))
        );
        assert_eq!(repl.input_text(), "draft");
    }

    #[test]
    fn input_and_footer_text_can_be_selected_without_changing_the_draft() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("draft input");
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &[]))
            .unwrap();
        for text in ["draft", "thinking"] {
            let start = text_position(terminal.backend().buffer(), text);
            let end = Position::new(start.x + text.len() as u16 - 1, start.y);
            repl.handle_mouse(mouse_at(MouseEventKind::Down(MouseButton::Left), start));
            repl.handle_mouse(mouse_at(MouseEventKind::Drag(MouseButton::Left), end));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            assert_eq!(
                terminal.backend().buffer()[(start.x, start.y)].bg,
                styles::S_SELECTION.bg.unwrap()
            );
            repl.handle_mouse(mouse_at(MouseEventKind::Up(MouseButton::Left), end));
            assert_eq!(repl.pending_screen_copy.take().as_deref(), Some(text));
            assert!(matches!(
                repl.process_key(event::KeyEvent::new(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL
                )),
                KeyAction::Redraw
            ));
            assert_eq!(repl.pending_screen_copy.take().as_deref(), Some(text));
            repl.process_key(event::KeyEvent::from(KeyCode::Esc));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            assert_eq!(repl.input_text(), "draft input");
        }
    }

    #[test]
    fn a_drag_can_cross_from_transcript_to_input() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str("draft input");
        let history = vec![Line::from("reply text")];
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let start = text_position(terminal.backend().buffer(), "reply text");
        let end = text_position(terminal.backend().buffer(), "input");
        repl.handle_mouse(mouse_at(MouseEventKind::Down(MouseButton::Left), start));
        repl.handle_mouse(mouse_at(
            MouseEventKind::Drag(MouseButton::Left),
            Position::new(end.x + 4, end.y),
        ));
        repl.handle_mouse(mouse_at(MouseEventKind::Up(MouseButton::Left), end));
        let text = repl.pending_screen_copy.take().unwrap();
        assert!(text.contains("reply text"));
        assert!(text.contains("draft input"));
        assert!(repl.selection.is_none());
    }

    #[test]
    fn ctrl_t_toggles_existing_and_streaming_reasoning_without_changing_input() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_expand_reasoning(false).unwrap();
        repl.replay_event(&AgentEvent::ReasoningDelta("saved thought".into()))
            .unwrap();
        repl.replay_event(&AgentEvent::TextDelta("answer".into()))
            .unwrap();
        repl.replay_finish_turn().unwrap();
        repl.replay_event(&AgentEvent::ReasoningDelta("live thought".into()))
            .unwrap();
        repl.textarea.insert_str("draft input");
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        for expanded in [true, false, true] {
            repl.process_key(event::KeyEvent::new(
                KeyCode::Char('t'),
                KeyModifiers::CONTROL,
            ));
            let history = repl.tui.transcript_lines();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            let screen: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert_eq!(repl.expand_reasoning(), expanded);
            assert_eq!(screen.contains("saved thought"), expanded);
            assert!(screen.contains("live thought"));
            assert_eq!(repl.input_text(), "draft input");
        }
    }

    #[test]
    fn live_thinking_expands_on_click_and_collapses_when_finished() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.commit_lines(
            &(0..30)
                .map(|i| Line::from(format!("earlier {i}")))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        repl.replay_event(&AgentEvent::ReasoningDelta(
            "first thought\nsecond thought\nthird thought\nfourth thought".into(),
        ))
        .unwrap();
        repl.live_output = vec![Line::from("progress ".repeat(12))];
        let mut terminal = Terminal::new(TestBackend::new(60, 24)).unwrap();
        let history = repl.tui.transcript_lines();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let header_row = repl.live_reasoning_row().unwrap();
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: repl.transcript_area.x,
            row: repl.transcript_area.y + (header_row - repl.transcript_scroll.top) as u16,
            modifiers: KeyModifiers::NONE,
        };
        repl.handle_mouse(click);
        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..click
        });
        assert!(repl.expand_reasoning());
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(screen.contains("first thought"));

        repl.replay_event(&AgentEvent::TextDelta("answer".into()))
            .unwrap();
        assert!(repl.live_reasoning_row().is_none());
        assert!(!repl.expand_reasoning());
        let history = repl.tui.transcript_lines();
        assert!(
            history
                .iter()
                .any(|line| line.to_string().contains("> thinking, 4 lines"))
        );
        assert!(
            !history
                .iter()
                .any(|line| line.to_string().contains("first thought"))
        );
    }

    #[test]
    fn replayed_reasoning_expands_on_click_after_scroll_and_resize() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_expand_reasoning(false).unwrap();
        let before: Vec<_> = (0..30)
            .map(|index| Line::from(format!("before {index}")))
            .collect();
        repl.commit_lines(&before).unwrap();
        repl.replay_event(&AgentEvent::ReasoningDelta(
            "first thought\nsecond thought".into(),
        ))
        .unwrap();
        repl.replay_event(&AgentEvent::TextDelta("answer".into()))
            .unwrap();
        repl.replay_finish_turn().unwrap();
        let after: Vec<_> = (0..30)
            .map(|index| Line::from(format!("after {index}")))
            .collect();
        repl.commit_lines(&after).unwrap();
        let collapsed = repl.tui.transcript_lines();
        let header = collapsed
            .iter()
            .position(|line| line.to_string().contains("thinking, "))
            .unwrap();
        assert!(
            !collapsed
                .iter()
                .any(|line| line.to_string().contains("first thought"))
        );
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &collapsed))
            .unwrap();
        repl.transcript_scroll
            .jump_to(repl.transcript_layout.entry_row(header) - 2);
        terminal.backend_mut().resize(40, 24);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &collapsed))
            .unwrap();
        let top = repl.transcript_scroll.top;
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: repl.transcript_area.x,
            row: repl.transcript_area.y + (repl.transcript_layout.entry_row(header) - top) as u16,
            modifiers: KeyModifiers::NONE,
        };
        repl.handle_mouse(click);
        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..click
        });
        let expanded = repl.tui.transcript_lines();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &expanded))
            .unwrap();
        assert_eq!(repl.transcript_scroll.top, top);
        assert!(repl.selection.is_none());
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(screen.contains("first thought"));
        assert!(screen.contains("second thought"));
        repl.handle_mouse(click);
        repl.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            ..click
        });
        assert_eq!(repl.tui.transcript_lines(), collapsed);

        repl.set_expand_reasoning(true).unwrap();
        assert_eq!(repl.tui.transcript_lines(), expanded);
        repl.set_expand_reasoning(false).unwrap();
        assert_eq!(repl.tui.transcript_lines(), collapsed);
    }

    #[test]
    fn compaction_preserves_scrollback_and_appends_the_summary() {
        for replay in [false, true] {
            let mut repl = Repl::new(ReplConfig::default());
            for index in 0..60 {
                repl.replay_user_input(&format!("message {index}")).unwrap();
                repl.replay_event(&AgentEvent::TextDelta(format!("answer {index}")))
                    .unwrap();
                repl.replay_finish_turn().unwrap();
            }
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &repl.tui.transcript_lines()))
                .unwrap();
            assert!(repl.transcript_scroll.top > 0);
            repl.process_key(event::KeyEvent::from(KeyCode::PageUp));
            let scroll_top = repl.transcript_scroll.top;
            let event =
                AgentEvent::Compacted("Conversation summary.\nKeep the current task.".into());
            if replay {
                repl.replay_event(&event).unwrap();
            } else {
                repl.apply_event(&event).unwrap();
            }
            let history = repl.tui.transcript_lines();
            let text = history
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("message 0"));
            assert!(text.contains("answer 58"));
            assert!(text.contains("answer 59"));
            assert!(text.contains("Context compacted."));
            assert!(text.contains("Conversation summary."));
            assert!(text.contains("Keep the current task."));
            let marker = text.find("Context compacted.").unwrap();
            let summary = text.find("Conversation summary.").unwrap();
            let last_reply = text.find("answer 59").unwrap();
            assert!(last_reply < marker && marker < summary);
            assert_eq!(text.matches("answer 59").count(), 1);
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            assert_eq!(repl.transcript_scroll.top, scroll_top);
            repl.process_key(event::KeyEvent::new(KeyCode::Home, KeyModifiers::CONTROL));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            assert_eq!(repl.transcript_scroll.top, 0);
            text_position(terminal.backend().buffer(), "message 0");
            repl.process_key(event::KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            text_position(terminal.backend().buffer(), "Keep the current task.");
        }
    }

    #[test]
    fn compaction_flushes_pending_output_without_discarding_previous_output() {
        for event in [
            AgentEvent::TextDelta("Unfinished reply.".into()),
            AgentEvent::ReasoningDelta("Unfinished thought.".into()),
            AgentEvent::ToolStart {
                id: "call-example".into(),
                name: "exec".into(),
                humanized: "Example command".into(),
            },
        ] {
            let mut repl = Repl::new(ReplConfig::default());
            repl.replay_event(&AgentEvent::TextDelta("Previous answer.".into()))
                .unwrap();
            repl.replay_finish_turn().unwrap();
            repl.apply_event(&event).unwrap();
            repl.apply_event(&AgentEvent::Compacted("Conversation summary.".into()))
                .unwrap();
            repl.replay_finish_turn().unwrap();
            assert!(repl.renderer.pending_text().is_empty());
            assert!(repl.renderer.pending_reasoning().is_empty());
            assert!(repl.renderer.tick_tool().is_empty());
            let text = repl
                .tui
                .transcript_lines()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("Context compacted."));
            assert!(text.contains("Previous answer."));
            match event {
                AgentEvent::TextDelta(text_before) => assert!(text.contains(&text_before)),
                AgentEvent::ReasoningDelta(_) => {
                    repl.set_expand_reasoning(true).unwrap();
                    let expanded = repl
                        .tui
                        .transcript_lines()
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("\n");
                    assert!(expanded.contains("Unfinished thought."));
                    assert!(expanded.contains("Previous answer."));
                }
                AgentEvent::ToolStart { .. } => {}
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn repeated_compaction_preserves_history_without_duplicating_replies() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.replay_user_input("Old question.").unwrap();
        repl.replay_event(&AgentEvent::TextDelta("Old answer.".into()))
            .unwrap();
        repl.replay_user_input("Latest question.").unwrap();
        for chunk in [
            "First paragraph.\n\n",
            "```text\n",
            "example\n",
            "```\n",
            "Last paragraph.",
        ] {
            repl.replay_event(&AgentEvent::TextDelta(chunk.into()))
                .unwrap();
        }
        for count in 1..=3 {
            repl.apply_event(&AgentEvent::Compacted("summary".into()))
                .unwrap();
            let text = repl
                .tui
                .transcript_lines()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("First paragraph."));
            assert!(text.contains("example"));
            assert!(text.contains("Last paragraph."));
            assert!(text.contains("Old answer."));
            assert!(text.contains("Latest question."));
            assert_eq!(text.matches("Context compacted.").count(), count);
            assert_eq!(text.matches("summary").count(), count);
            assert_eq!(text.matches("Last paragraph.").count(), 1);
        }
        repl.clear_transcript().unwrap();
        repl.apply_event(&AgentEvent::Compacted("summary".into()))
            .unwrap();
        assert!(
            !repl
                .tui
                .transcript_lines()
                .iter()
                .any(|line| line.to_string().contains("Last paragraph."))
        );
    }

    #[test]
    fn failed_compaction_preserves_the_displayed_history() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.replay_user_input("Original question.").unwrap();
        repl.replay_event(&AgentEvent::TextDelta("Original answer.".into()))
            .unwrap();
        repl.replay_finish_turn().unwrap();
        repl.apply_event(&AgentEvent::Error(
            "Compaction failed. Conversation unchanged.".into(),
        ))
        .unwrap();
        let text = repl
            .tui
            .transcript_lines()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Original question."));
        assert!(text.contains("Original answer."));
        assert!(!text.contains("Context compacted."));
    }

    #[test]
    fn latest_button_shows_while_scrolled_up_and_returns_to_the_bottom() {
        let mut repl = Repl::new(ReplConfig::default());
        let history: Vec<Line<'static>> = (0..120)
            .map(|index| Line::from(format!("message {index}")))
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert!(repl.latest_button.is_none());
        repl.process_key(event::KeyEvent::from(KeyCode::PageUp));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let mut history = history;
        history.extend((0..3).map(|index| Line::from(format!("new {index}"))));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let button = text_position(terminal.backend().buffer(), "Jump to latest (3 new lines)");
        repl.handle_mouse(mouse_at(MouseEventKind::Down(MouseButton::Left), button));
        repl.handle_mouse(mouse_at(MouseEventKind::Up(MouseButton::Left), button));
        assert!(repl.transcript_scroll.following);
        assert!(repl.selection.is_none());
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert!(repl.latest_button.is_none());
        assert_eq!(repl.transcript_scroll.top, repl.transcript_scroll.max);
    }

    #[test]
    fn clicking_footer_hints_acts_like_their_keys() {
        let mut repl = Repl::new(ReplConfig::default());
        let history: Vec<Line<'static>> = (0..120)
            .map(|index| Line::from(format!("message {index}")))
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(140, 24)).unwrap();
        let mut click = |repl: &mut Repl, text: &str| {
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            // Search only the footer row, since hints can also show elsewhere.
            let screen = terminal.backend().buffer();
            let row = screen.area.bottom() - 1;
            let mut footer = Buffer::empty(Rect::new(0, row, screen.area.width, 1));
            for x in 0..screen.area.width {
                footer[(x, row)] = screen[(x, row)].clone();
            }
            let position = text_position(&footer, text);
            repl.handle_mouse(mouse_at(MouseEventKind::Down(MouseButton::Left), position));
            repl.handle_mouse(mouse_at(MouseEventKind::Up(MouseButton::Left), position));
        };
        let mode = repl.mode();
        click(&mut repl, "(Shift+Tab)");
        assert_ne!(repl.mode(), mode);
        let tools = repl.renderer.expand_tools();
        click(&mut repl, "Ctrl+O tools");
        assert_ne!(repl.renderer.expand_tools(), tools);
        click(&mut repl, "PgUp scroll");
        assert!(!repl.transcript_scroll.following);
        click(&mut repl, "Ctrl+End: follow latest");
        assert!(repl.transcript_scroll.following);
        click(&mut repl, "Ctrl+F find");
        assert!(repl.transcript_search.is_some());
        click(&mut repl, "Esc: close");
        assert!(repl.transcript_search.is_none());
    }

    #[test]
    fn alt_up_and_down_move_between_prompts() {
        let mut repl = Repl::new(ReplConfig::default());
        let mut history = Vec::new();
        let mut prompt_rows = Vec::new();
        for prompt in 0..4 {
            prompt_rows.push(history.len());
            history.push(Line::from(format!("> prompt {prompt}")).style(styles::S_USER_ECHO));
            history.push(Line::from("  continued").style(styles::S_USER_ECHO));
            history.extend((0..30).map(|index| Line::from(format!("reply {index}"))));
        }
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let mut press = |repl: &mut Repl, code| {
            repl.process_key(event::KeyEvent::new(code, KeyModifiers::ALT));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
        };
        press(&mut repl, KeyCode::Down);
        assert!(repl.transcript_scroll.following);
        press(&mut repl, KeyCode::Up);
        assert_eq!(repl.transcript_scroll.top, prompt_rows[3]);
        press(&mut repl, KeyCode::Up);
        assert_eq!(repl.transcript_scroll.top, prompt_rows[2]);
        press(&mut repl, KeyCode::Down);
        assert_eq!(repl.transcript_scroll.top, prompt_rows[3]);
        press(&mut repl, KeyCode::Down);
        assert!(repl.transcript_scroll.following);
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
        // "message 10" also matches messages 100 to 109.
        text_position(terminal.backend().buffer(), "(1 of 11)");
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
        let selection = TranscriptSelection {
            start: TextPoint { row: 5, column: 4 },
            end: TextPoint { row: 6, column: 4 },
        };
        assert_eq!(selection.text(&buffer, 5), "example\ntwo界");
        let reverse = TranscriptSelection {
            start: selection.end,
            end: selection.start,
        };
        assert_eq!(reverse.text(&buffer, 5), selection.text(&buffer, 5));
    }

    #[test]
    fn selection_stays_on_its_text_while_scrolling() {
        let mut repl = Repl::new(ReplConfig::default());
        let history: Vec<Line<'static>> =
            (0..40).map(|n| Line::from(format!("line {n}"))).collect();
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        let draw = |repl: &mut Repl, terminal: &mut Terminal<TestBackend>| {
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
        };
        draw(&mut repl, &mut terminal);
        let body = repl.transcript_area;
        let mouse = |kind, column, row| MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };
        let row_text = |terminal: &Terminal<TestBackend>, y| {
            let buffer = terminal.backend().buffer();
            (body.x..body.right())
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        };
        let selected_rows = |terminal: &Terminal<TestBackend>| {
            let buffer = terminal.backend().buffer();
            (body.y..body.bottom())
                .filter(|&y| buffer[(body.x, y)].bg == styles::S_SELECTION.bg.unwrap())
                .collect::<Vec<_>>()
        };
        let y = body.y + 2;
        let text = row_text(&terminal, y);
        repl.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), body.x, y));
        repl.handle_mouse(mouse(
            MouseEventKind::Drag(MouseButton::Left),
            body.right(),
            y,
        ));
        draw(&mut repl, &mut terminal);
        assert_eq!(selected_rows(&terminal), vec![y]);

        repl.handle_mouse(mouse(MouseEventKind::ScrollUp, body.x, y));
        draw(&mut repl, &mut terminal);
        assert_eq!(selected_rows(&terminal), vec![y + 3]);
        assert_eq!(row_text(&terminal, y + 3), text);
        assert_eq!(
            repl.selected_text(&history, &[], body).as_deref(),
            Some(text.trim_end())
        );

        // Dragging below the transcript scrolls and extends the selection.
        repl.handle_mouse(mouse(
            MouseEventKind::Drag(MouseButton::Left),
            body.right(),
            body.bottom() + 1,
        ));
        draw(&mut repl, &mut terminal);
        assert_eq!(
            selected_rows(&terminal),
            (y + 2..body.bottom()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copying_wrapped_transcript_preserves_logical_lines() {
        use ratatui::text::Span;

        let history = [
            Line::from(vec![
                Span::raw("* "),
                Span::raw("A paragraph with   several words and averylongunbrokenword.  "),
            ]),
            Line::default(),
            Line::from("   "),
            Line::from("    indented code with spaces and \u{754c}e\u{301} characters"),
            Line::from("A separate line.   "),
        ];
        let expected = history
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        for (history, dynamic) in [(&history[..], &[][..]), (&history[..2], &history[2..])] {
            for width in [12, 24, 80] {
                let mut repl = Repl::new(ReplConfig::default());
                repl.transcript_layout.measure(history, width);
                let dynamic_rows: usize = dynamic
                    .iter()
                    .map(|line| {
                        Paragraph::new(layout_line(line, width))
                            .wrap(Wrap { trim: false })
                            .line_count(width)
                    })
                    .sum();
                let start = TextPoint { row: 0, column: 3 };
                let end = TextPoint {
                    row: repl.transcript_layout.rows() + dynamic_rows - 1,
                    column: width + 2,
                };
                for (start, end) in [(start, end), (end, start)] {
                    repl.selection = Some(TranscriptSelection { start, end });
                    assert_eq!(
                        repl.selected_text(history, dynamic, Rect::new(3, 0, width, 20)),
                        Some(expected.clone()),
                        "viewport width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn copying_part_of_a_wrapped_paragraph_omits_hanging_indents() {
        use ratatui::text::Span;

        let history = vec![Line::from(vec![
            Span::raw("* "),
            Span::raw("alpha beta gamma delta"),
        ])];
        let mut repl = Repl::new(ReplConfig::default());
        repl.transcript_layout.measure(&history, 12);
        let start = TextPoint { row: 0, column: 8 };
        let end = TextPoint { row: 2, column: 7 };
        for (start, end) in [(start, end), (end, start)] {
            repl.selection = Some(TranscriptSelection { start, end });
            assert_eq!(
                repl.selected_text(&history, &[], Rect::new(3, 0, 12, 20)),
                Some("ha beta gamma del".to_owned())
            );
        }
    }

    #[test]
    fn dragging_highlights_the_selection_and_a_click_selects_nothing() {
        let mut repl = Repl::new(ReplConfig::default());
        let history = vec![Line::from("first example line"), Line::from("second")];
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let body = repl.transcript_area;
        let mouse = |kind, column| MouseEvent {
            kind,
            column,
            row: body.y,
            modifiers: KeyModifiers::NONE,
        };
        repl.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), body.x));
        repl.handle_mouse(mouse(MouseEventKind::Drag(MouseButton::Left), body.x + 4));
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let selected = |x| buffer[(x, body.y)].bg == styles::S_SELECTION.bg.unwrap();
        assert!((body.x..=body.x + 4).all(selected));
        assert!(!selected(body.x + 5));
        assert_eq!(
            repl.selected_text(&history, &[], body).as_deref(),
            Some("first")
        );

        repl.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), body.x));
        repl.handle_mouse(mouse(MouseEventKind::Up(MouseButton::Left), body.x));
        assert!(repl.selection.is_none());
        assert!(repl.toast.is_none());
    }
}
