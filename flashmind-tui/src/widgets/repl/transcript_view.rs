//! Transcript scrolling, mouse handling, selection and search.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Position, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget, Wrap};
use unicode_width::UnicodeWidthStr;

use super::Repl;
use super::input::copy_to_clipboard;
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
                self.transcript_scroll.mouse(mouse.kind);
            }
            MouseEventKind::Down(MouseButton::Left) if mouse.row == self.footer_row => {
                self.selection = None;
                self.transcript_scroll.latest();
            }
            MouseEventKind::Down(MouseButton::Left) if self.transcript_area.contains(position) => {
                self.transcript_scroll.following = false;
                let point = self.text_point(position);
                self.selection = Some(TranscriptSelection {
                    start: point,
                    end: point,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) if self.selection.is_some() => {
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

    /// Transcript rows from `top` laid out for `area`, with the history entry
    /// `highlight` drawn in the accent style.
    pub(super) fn transcript_paragraph(
        &self,
        history: &[Line<'static>],
        dynamic: &[Line<'static>],
        top: usize,
        area: Rect,
        highlight: Option<usize>,
    ) -> Paragraph<'static> {
        let window = self.transcript_layout.window(top, area.height as usize);
        let mut visible: Vec<Line<'static>> = Vec::new();
        for (index, line) in history[window.entries.clone()].iter().enumerate() {
            let mut line = line.clone();
            if highlight == Some(window.entries.start + index) {
                for span in &mut line.spans {
                    span.style = span.style.patch(styles::S_ACCENT);
                }
            }
            visible.extend(layout_line(&line, area.width.max(1)));
        }
        if top + area.height as usize > self.transcript_layout.rows() {
            visible.extend(dynamic.iter().cloned());
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

    /// Selected text, laid out at the transcript area's width.
    fn selected_text(
        &self,
        history: &[Line<'static>],
        dynamic: &[Line<'static>],
        area: Rect,
    ) -> Option<String> {
        let selection = self.selection.as_ref()?;
        let (start, end) = selection.bounds();
        let height = (end.row - start.row + 1).min(u16::MAX as usize) as u16;
        let area = Rect::new(area.x, 0, area.width, height);
        let mut buffer = Buffer::empty(area);
        self.transcript_paragraph(history, dynamic, start.row, area, None)
            .render(area, &mut buffer);
        Some(selection.text(&buffer, start.row))
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

/// A cell in the laid-out transcript: `row` counts from the first transcript
/// row, so a selection stays on its text while the view scrolls.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TextPoint {
    pub(super) row: usize,
    pub(super) column: u16,
}

pub(super) struct TranscriptSelection {
    start: TextPoint,
    end: TextPoint,
}

impl TranscriptSelection {
    fn bounds(&self) -> (TextPoint, TextPoint) {
        (self.start.min(self.end), self.start.max(self.end))
    }

    pub(super) fn contains(&self, point: TextPoint) -> bool {
        let (start, end) = self.bounds();
        start <= point && point <= end
    }

    /// Selected text in `buffer`, whose first row is transcript row `first_row`.
    fn text(&self, buffer: &Buffer, first_row: usize) -> String {
        let area = buffer.area;
        let mut lines = Vec::new();
        for y in area.y..area.bottom() {
            let row = first_row + (y - area.y) as usize;
            let mut line = String::new();
            let mut x = area.x;
            while x < area.right() {
                let symbol = buffer[(x, y)].symbol();
                let footprint = (symbol.width() as u16).max(1);
                if self.contains(TextPoint { row, column: x })
                    || self.contains(TextPoint {
                        row,
                        column: x + footprint - 1,
                    })
                {
                    line.push_str(symbol);
                }
                x += footprint;
            }
            lines.push(line.trim_end().to_string());
        }
        lines.join("\n")
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::repl::ReplConfig;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{self, KeyCode, KeyModifiers};

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
