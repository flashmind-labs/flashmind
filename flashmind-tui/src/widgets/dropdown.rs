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

    pub fn lines(&self, max_visible: usize, max_width: u16) -> Vec<Line<'static>> {
        if self.candidates.is_empty() {
            return Vec::new();
        }

        let visible_count = max_visible.min(self.candidates.len());
        let visible = &self.candidates
            [self.scroll_offset..self.candidates.len().min(self.scroll_offset + visible_count)];
        let inner_w = (max_width as usize).saturating_sub(4);

        let mut lines = Vec::with_capacity(visible_count);

        for (i, name) in visible.iter().enumerate() {
            let abs_idx = i + self.scroll_offset;
            let is_selected = abs_idx == self.selected;

            let (indicator, style) = if is_selected {
                (
                    "\u{25b8} ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan),
                )
            } else {
                ("  ", S_DIM)
            };

            let display = if name.len() > inner_w {
                format!("{}\u{2026}", &name[..inner_w.saturating_sub(1)])
            } else {
                format!("{:<width$}", name, width = inner_w)
            };

            lines.push(Line::from(vec![
                Span::styled(indicator, style),
                Span::styled(display, style),
            ]));
        }

        let has_more_above = self.scroll_offset > 0;
        let has_more_below = self.scroll_offset + visible_count < self.candidates.len();
        if has_more_above || has_more_below {
            let indicator = match (has_more_above, has_more_below) {
                (true, true) => format!(
                    "  \u{2191}\u{2193} {}/{}",
                    self.selected + 1,
                    self.candidates.len()
                ),
                (true, false) => format!(
                    "  \u{2191} {}/{}",
                    self.selected + 1,
                    self.candidates.len()
                ),
                (false, true) => format!(
                    "  \u{2193} {}/{}",
                    self.selected + 1,
                    self.candidates.len()
                ),
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

    fn adjust_scroll(&mut self) {
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
    }
}
