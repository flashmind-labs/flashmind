//! Scrollable dropdown widget.
//!
//! Renders a list of candidates with selection highlighting and scroll support.
//! Useful for autocomplete overlays (file mentions, commands, etc.).
//!
//! The widget is a state machine — call [`Dropdown::handle_key`] with keyboard
//! events and [`Dropdown::lines`] to render the current state.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::styles::S_DIM;

// ---------------------------------------------------------------------------
// Public types

#[derive(Debug)]
pub enum DropdownAction {
    Select(String),
    Cancel,
}

// ---------------------------------------------------------------------------
// Widget

pub struct Dropdown {
    pub title: String,
    pub candidates: Vec<String>,
    pub selected: usize,
    scroll_offset: usize,
}

impl Dropdown {
    /// Maximum number of candidate rows shown at once before scrolling.
    pub const VISIBLE_ROWS: usize = 8;

    pub fn new(title: impl Into<String>, candidates: Vec<String>) -> Self {
        Self {
            title: title.into(),
            candidates,
            selected: 0,
            scroll_offset: 0,
        }
    }

    pub fn set_candidates(&mut self, candidates: Vec<String>) {
        self.candidates = candidates;
        self.selected = 0;
        self.scroll_offset = 0;
    }

    pub fn selected_value(&self) -> Option<&str> {
        self.candidates.get(self.selected).map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<DropdownAction> {
        match key.code {
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                self.adjust_scroll();
                None
            }
            KeyCode::Down => {
                let max = self.candidates.len().saturating_sub(1);
                self.selected = self.selected.saturating_add(1).min(max);
                self.adjust_scroll();
                None
            }
            KeyCode::Enter | KeyCode::Tab => {
                let value = self
                    .candidates
                    .get(self.selected)
                    .cloned()
                    .unwrap_or_default();
                Some(DropdownAction::Select(value))
            }
            KeyCode::Esc => Some(DropdownAction::Cancel),
            _ => None,
        }
    }

    pub fn lines(&self, max_width: u16) -> Vec<Line<'static>> {
        if self.candidates.is_empty() {
            return Vec::new();
        }

        let visible_count = Self::VISIBLE_ROWS.min(self.candidates.len()).max(1);
        let mut scroll_offset = self
            .scroll_offset
            .min(self.candidates.len().saturating_sub(1));
        if self.selected < scroll_offset {
            scroll_offset = self.selected;
        } else if self.selected >= scroll_offset + visible_count {
            scroll_offset = self.selected + 1 - visible_count;
        }
        let visible = &self.candidates
            [scroll_offset..self.candidates.len().min(scroll_offset + visible_count)];
        let inner_w = (max_width as usize).saturating_sub(4);

        let mut lines = Vec::with_capacity(visible_count);

        for (i, name) in visible.iter().enumerate() {
            let abs_idx = i + scroll_offset;
            let is_selected = abs_idx == self.selected;

            let (indicator, style) = if is_selected {
                (
                    "\u{25b8} ",
                    Style::default().fg(Color::Black).bg(Color::Cyan),
                )
            } else {
                ("  ", S_DIM)
            };

            let display = if name.len() > inner_w {
                let truncated: String = name.chars().take(inner_w.saturating_sub(1)).collect();
                format!("{truncated}\u{2026}")
            } else {
                format!("{:<width$}", name, width = inner_w)
            };

            lines.push(Line::from(vec![
                Span::styled(indicator, style),
                Span::styled(display, style),
            ]));
        }

        let has_more_above = scroll_offset > 0;
        let has_more_below = scroll_offset + visible_count < self.candidates.len();
        if has_more_above || has_more_below {
            let indicator = match (has_more_above, has_more_below) {
                (true, true) => format!(
                    "  \u{2191}\u{2193} {}/{}",
                    self.selected + 1,
                    self.candidates.len()
                ),
                (true, false) => {
                    format!("  \u{2191} {}/{}", self.selected + 1, self.candidates.len())
                }
                (false, true) => {
                    format!("  \u{2193} {}/{}", self.selected + 1, self.candidates.len())
                }
                _ => String::new(),
            };
            lines.push(Line::from(Span::styled(
                indicator,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            )));
        }

        lines
    }

    /// Keep `scroll_offset` consistent so the selected row stays within the
    /// visible window `[scroll_offset, scroll_offset + VISIBLE_ROWS)` after
    /// navigating in either direction.
    fn adjust_scroll(&mut self) {
        let window = Self::VISIBLE_ROWS.min(self.candidates.len()).max(1);
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + window {
            self.scroll_offset = self.selected + 1 - window;
        }
    }

    #[cfg(test)]
    fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn down(d: &mut Dropdown) {
        d.handle_key(KeyEvent::from(KeyCode::Down));
    }
    fn up(d: &mut Dropdown) {
        d.handle_key(KeyEvent::from(KeyCode::Up));
    }

    #[test]
    fn scroll_offset_follows_selection_downward() {
        let items: Vec<String> = (0..20).map(|i| format!("item {i}")).collect();
        let mut d = Dropdown::new("t", items);
        // Move selection past the visible window.
        for _ in 0..Dropdown::VISIBLE_ROWS + 2 {
            down(&mut d);
        }
        // scroll_offset must keep the selection inside the window.
        let off = d.scroll_offset();
        assert!(d.selected >= off, "selected {} < offset {off}", d.selected);
        assert!(
            d.selected < off + Dropdown::VISIBLE_ROWS,
            "selected {} outside window starting at {off}",
            d.selected
        );
    }

    #[test]
    fn scroll_offset_returns_to_top() {
        let items: Vec<String> = (0..20).map(|i| format!("item {i}")).collect();
        let mut d = Dropdown::new("t", items);
        for _ in 0..15 {
            down(&mut d);
        }
        for _ in 0..15 {
            up(&mut d);
        }
        assert_eq!(d.selected, 0);
        assert_eq!(d.scroll_offset(), 0);
    }
}
