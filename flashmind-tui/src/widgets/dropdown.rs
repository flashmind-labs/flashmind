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
    descriptions: Vec<Option<String>>,
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
            descriptions: Vec::new(),
            scroll_offset: 0,
        }
    }

    pub fn set_candidates(&mut self, candidates: Vec<String>) {
        self.candidates = candidates;
        self.descriptions.clear();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Set display-only descriptions in candidate order.
    pub fn set_descriptions(&mut self, descriptions: Vec<Option<String>>) {
        self.descriptions = descriptions;
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
        self.lines_for_height(max_width, Self::VISIBLE_ROWS + 1)
    }

    pub fn lines_for_height(&self, max_width: u16, max_height: usize) -> Vec<Line<'static>> {
        if self.candidates.is_empty() || max_height == 0 {
            return Vec::new();
        }

        let visible_count = Self::VISIBLE_ROWS
            .min(self.candidates.len())
            .min(max_height)
            .max(1);
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

            let description = self
                .descriptions
                .get(abs_idx)
                .and_then(Option::as_deref)
                .map(|text| format!("  {text}"))
                .unwrap_or_default();
            let name_w = inner_w.saturating_sub(description.chars().count());
            let display_name = if name.chars().count() > name_w {
                let truncated: String = name.chars().take(name_w.saturating_sub(1)).collect();
                format!("{truncated}\u{2026}")
            } else {
                format!("{name:<name_w$}")
            };
            let display: String = format!("{display_name}{description}")
                .chars()
                .take(inner_w)
                .collect();

            lines.push(Line::from(vec![
                Span::styled(indicator, style),
                Span::styled(display, style),
            ]));
        }

        let has_more_above = scroll_offset > 0;
        let has_more_below = scroll_offset + visible_count < self.candidates.len();
        if (has_more_above || has_more_below) && lines.len() < max_height {
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
    fn description_stays_visible_when_the_candidate_is_truncated() {
        let name = "provider:example/a-model-with-a-long-name";
        let mut dropdown = Dropdown::new("", vec![name.to_string()]);
        dropdown.set_descriptions(vec![Some("$0.25/m in, $1.5/m out".to_string())]);
        let line = dropdown.lines(40).remove(0).to_string();
        assert!(line.contains("$0.25/m in, $1.5/m out"));
        assert!(!line.contains(name));
        assert_eq!(dropdown.selected_value(), Some(name));
        dropdown.set_candidates(vec!["other".to_string()]);
        assert!(!dropdown.lines(40)[0].to_string().contains("$0.25/m"));
    }

    #[test]
    fn short_viewport_keeps_selected_candidate_visible() {
        let mut dropdown = Dropdown::new(
            "files",
            (0..30).map(|index| format!("item {index}")).collect(),
        );
        for _ in 0..20 {
            down(&mut dropdown);
        }
        let lines = dropdown.lines_for_height(40, 3);
        assert!(lines.len() <= 3);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect();
        assert!(text.contains("item 20"));
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
