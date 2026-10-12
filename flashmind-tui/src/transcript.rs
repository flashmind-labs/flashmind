use std::ops::Range;

use crate::event_render::fit_tool_line;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

/// Cached row offsets for append-only transcript entries at one terminal width.
#[derive(Default)]
pub(crate) struct TranscriptLayout {
    width: u16,
    offsets: Vec<usize>,
}

pub(crate) struct TranscriptWindow {
    pub entries: Range<usize>,
    pub row_offset: u16,
}

impl TranscriptLayout {
    pub fn width(&self) -> u16 {
        self.width
    }
    pub fn measure(&mut self, lines: &[Line<'static>], width: u16) {
        if width != self.width || self.offsets.len() > lines.len() + 1 || self.offsets.is_empty() {
            self.width = width;
            self.offsets = vec![0];
        }
        for line in &lines[self.offsets.len() - 1..] {
            let rows = Paragraph::new(fit_tool_line(line, width as usize))
                .wrap(Wrap { trim: false })
                .line_count(width.max(1));
            self.offsets.push(self.rows() + rows);
        }
    }

    pub fn rows(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    pub fn entry_row(&self, index: usize) -> usize {
        self.offsets.get(index).copied().unwrap_or(self.rows())
    }

    pub fn window(&self, top: usize, height: usize) -> TranscriptWindow {
        let entries = self.offsets.len().saturating_sub(1);
        if top >= self.rows() {
            return TranscriptWindow {
                entries: entries..entries,
                row_offset: top.saturating_sub(self.rows()).min(u16::MAX as usize) as u16,
            };
        }
        let start = self
            .offsets
            .partition_point(|&offset| offset <= top)
            .saturating_sub(1)
            .min(entries);
        let end = self
            .offsets
            .partition_point(|&offset| offset < top + height)
            .min(entries);
        TranscriptWindow {
            entries: start..end,
            row_offset: top
                .saturating_sub(self.offsets[start])
                .min(u16::MAX as usize) as u16,
        }
    }
}

pub(crate) struct TranscriptScroll {
    pub top: usize,
    pub max: usize,
    pub following: bool,
    visible: usize,
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            top: 0,
            max: 0,
            following: true,
            visible: 0,
        }
    }
}

impl TranscriptScroll {
    pub fn jump_to(&mut self, row: usize) {
        self.top = row.min(self.max);
        self.following = false;
    }

    pub fn latest(&mut self) {
        self.top = self.max;
        self.following = true;
    }
    pub fn update(&mut self, rows: usize, height: u16) {
        let was_at_bottom = self.top == self.max;
        self.visible = height as usize;
        self.max = rows.saturating_sub(self.visible);
        self.top = if self.following || was_at_bottom {
            self.max
        } else {
            self.top.min(self.max)
        };
        self.following |= was_at_bottom;
    }

    fn up(&mut self, rows: usize) {
        if self.max > 0 {
            self.following = false;
        }
        self.top = self.top.saturating_sub(rows);
    }

    fn down(&mut self, rows: usize) {
        self.top = (self.top + rows).min(self.max);
        self.following = self.top == self.max;
    }

    pub fn key(&mut self, key: KeyEvent) -> bool {
        let page = (self.visible / 2).max(1);
        match key.code {
            KeyCode::PageUp => self.up(page),
            KeyCode::PageDown => self.down(page),
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => self.up(self.max),
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => self.down(self.max),
            _ => return false,
        }
        true
    }

    pub fn mouse(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => self.up(3),
            MouseEventKind::ScrollDown => self.down(3),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_pauses_follow_and_keeps_position_as_output_arrives() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(100, 20);
        assert_eq!(scroll.top, 80);
        scroll.key(KeyEvent::from(KeyCode::PageUp));
        assert_eq!(scroll.top, 70);
        scroll.update(150, 15);
        assert_eq!(scroll.top, 70);
        assert!(!scroll.following);
        scroll.key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        assert_eq!(scroll.top, 135);
        assert!(scroll.following);
    }

    #[test]
    fn mouse_can_read_history_and_return_to_following() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(30, 10);
        scroll.mouse(MouseEventKind::ScrollUp);
        assert_eq!(scroll.top, 17);
        scroll.mouse(MouseEventKind::ScrollDown);
        assert!(scroll.following);
    }

    #[test]
    fn new_output_follows_when_position_is_already_at_bottom() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(30, 10);
        scroll.jump_to(scroll.max);
        assert!(!scroll.following);

        scroll.update(35, 10);
        assert_eq!(scroll.top, 25);
        assert!(scroll.following);

        scroll.key(KeyEvent::from(KeyCode::PageUp));
        scroll.update(40, 10);
        assert_eq!(scroll.top, 20);
        assert!(!scroll.following);
    }

    #[test]
    fn visible_window_handles_wrapping_growth_and_resize() {
        let mut lines = vec![
            Line::from("界".repeat(10)),
            Line::from("second"),
            Line::from("third"),
        ];
        let mut layout = TranscriptLayout::default();
        layout.measure(&lines, 10);
        assert_eq!(layout.rows(), 4);
        let window = layout.window(1, 2);
        assert_eq!(window.entries, 0..2);
        assert_eq!(window.row_offset, 1);
        lines.push(Line::from("fourth"));
        layout.measure(&lines, 10);
        assert_eq!(layout.rows(), 5);
        layout.measure(&lines, 20);
        assert_eq!(layout.rows(), 4);
    }
}
