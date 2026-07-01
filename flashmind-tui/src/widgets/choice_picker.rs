//! Interactive choice selection widget.
//!
//! Presents a list of options that the user can navigate and select.  Some
//! options may accept free-form text input.  The widget is a standalone state
//! machine — call [`ChoicePicker::handle_key`] with keyboard events and
//! [`ChoicePicker::lines`] to render the current state.
//!
//! Long lists are rendered through a bounded viewport so the output never
//! exceeds the terminal height (which would otherwise break in-place erase and
//! leave duplicated headers on screen). Pickers created with
//! [`ChoicePicker::searchable`] additionally accept typed characters to filter
//! the visible options.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::styles::S_DIM;

// ---------------------------------------------------------------------------
// Public types

#[derive(Debug, Clone)]
pub struct ChoiceOption {
    pub label: String,
    pub accepts_input: bool,
}

#[derive(Debug, Clone)]
pub struct ChoiceResponse {
    pub selected: usize,
    pub label: String,
    pub input: String,
}

#[derive(Debug)]
pub enum ChoicePickerAction {
    Select(ChoiceResponse),
    Cancel,
    Delete(usize),
}

// ---------------------------------------------------------------------------
// Widget

#[derive(Debug, Clone)]
struct TabDef {
    label: String,
    members: Vec<usize>,
}

#[derive(Debug)]
pub struct ChoicePicker {
    pub title: String,
    pub options: Vec<ChoiceOption>,
    /// Index into `options` of the highlighted entry.
    pub selected: usize,
    pub input_buffer: String,
    pub editing_input: bool,
    previews: Vec<String>,
    /// When enabled, typed characters filter the list and `^D` deletes.
    searchable: bool,
    query: String,
    /// Indices into `options` that match the current query, in original order.
    filtered: Vec<usize>,
    /// Optional tab groups. Empty means no tabs (single flat list).
    tabs: Vec<TabDef>,
    active_tab: usize,
}

impl ChoicePicker {
    pub fn new(title: String, options: Vec<ChoiceOption>) -> Self {
        let filtered = (0..options.len()).collect();
        Self {
            title,
            options,
            selected: 0,
            input_buffer: String::new(),
            editing_input: false,
            previews: Vec::new(),
            searchable: false,
            query: String::new(),
            filtered,
            tabs: Vec::new(),
            active_tab: 0,
        }
    }

    pub fn with_previews(mut self, previews: Vec<String>) -> Self {
        self.previews = previews;
        self
    }

    /// Enable type-to-search filtering. In this mode printable keys edit a
    /// query that filters the list, and `Ctrl+D` deletes the highlighted entry.
    pub fn searchable(mut self, on: bool) -> Self {
        self.searchable = on;
        self
    }

    /// Attach tab groups. Each tab names a subset of `options` by index. The
    /// active tab's members are intersected with the search query. Empty tabs
    /// leave the picker as a single flat list.
    pub fn with_tabs(mut self, tabs: Vec<(String, Vec<usize>)>) -> Self {
        self.tabs = tabs
            .into_iter()
            .map(|(label, members)| TabDef { label, members })
            .collect();
        self.active_tab = 0;
        self.refilter();
        self
    }

    pub fn active_tab(&self) -> usize {
        self.active_tab
    }

    /// Cycle to the next (`forward`) or previous tab, wrapping. No-op without tabs.
    fn switch_tab(&mut self, forward: bool) {
        if self.tabs.is_empty() {
            return;
        }
        let n = self.tabs.len();
        self.active_tab = if forward {
            (self.active_tab + 1) % n
        } else {
            (self.active_tab + n - 1) % n
        };
        self.refilter();
        self.selected = self.filtered.first().copied().unwrap_or(0);
    }

    pub fn respond(&self) -> ChoiceResponse {
        let Some(option) = self.options.get(self.selected) else {
            return ChoiceResponse {
                selected: 0,
                label: String::new(),
                input: String::new(),
            };
        };
        ChoiceResponse {
            selected: self.selected,
            label: option.label.clone(),
            input: if option.accepts_input {
                self.input_buffer.clone()
            } else {
                String::new()
            },
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ChoicePickerAction> {
        if self.options.is_empty() {
            return match key.code {
                KeyCode::Esc => Some(ChoicePickerAction::Cancel),
                _ => None,
            };
        }

        if self.editing_input {
            match key.code {
                KeyCode::Enter => return Some(ChoicePickerAction::Select(self.respond())),
                KeyCode::Esc => self.editing_input = false,
                KeyCode::Up => self.up(),
                KeyCode::Char(c) => self.input_buffer.push(c),
                KeyCode::Backspace => {
                    self.input_buffer.pop();
                }
                _ => {}
            }
            return None;
        }

        if self.searchable {
            return self.handle_key_search(key);
        }

        // Ctrl+D deletes the selected option, matching searchable mode and the
        // PlanPicker.  (Esc cancels in every mode.)
        if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(ChoicePickerAction::Delete(self.selected));
        }

        match key.code {
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
            KeyCode::Tab => self.switch_tab(true),
            KeyCode::BackTab => self.switch_tab(false),
            KeyCode::Enter => {
                if self.options[self.selected].accepts_input {
                    self.editing_input = true;
                    return None;
                }
                return Some(ChoicePickerAction::Select(self.respond()));
            }
            KeyCode::Esc => return Some(ChoicePickerAction::Cancel),
            KeyCode::Char('d') | KeyCode::Delete => {
                return Some(ChoicePickerAction::Delete(self.selected));
            }
            _ => {}
        }
        None
    }

    /// Key handling for searchable pickers: printable chars edit the query.
    fn handle_key_search(&mut self, key: KeyEvent) -> Option<ChoicePickerAction> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        if ctrl {
            return match key.code {
                KeyCode::Char('d') => {
                    if self.filtered.is_empty() {
                        None
                    } else {
                        Some(ChoicePickerAction::Delete(self.selected))
                    }
                }
                KeyCode::Char('u') => {
                    self.query.clear();
                    self.refilter();
                    None
                }
                _ => None,
            };
        }

        match key.code {
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
            KeyCode::Tab => self.switch_tab(true),
            KeyCode::BackTab => self.switch_tab(false),
            KeyCode::Enter => {
                if self.filtered.is_empty() {
                    return None;
                }
                return Some(ChoicePickerAction::Select(self.respond()));
            }
            KeyCode::Esc => return Some(ChoicePickerAction::Cancel),
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.refilter();
            }
            _ => {}
        }
        None
    }

    pub fn lines(&self) -> Vec<Line<'static>> {
        let has_previews = !self.previews.is_empty();
        let (term_w, term_h) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let term_w = term_w as usize;

        let left_col_w = if has_previews {
            self.options
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    // "▸ " or "  " (2) + "N. " (digits + 2) + label
                    let num_w = format!("{}. ", i + 1).len();
                    2 + num_w + o.label.len()
                })
                .max()
                .unwrap_or(20)
                .min(term_w / 2)
        } else {
            0
        };
        let separator = " │ ";
        let right_col_w = if has_previews && term_w > left_col_w + separator.len() + 20 {
            term_w - left_col_w - separator.len()
        } else {
            0
        };

        // How many extra rows the selected entry's wrapped preview occupies
        // below its own line. Capped so the viewport math stays bounded.
        const PREVIEW_EXTRA: usize = 5;
        let selected_preview: Vec<String> = if right_col_w > 0 {
            self.previews
                .get(self.selected)
                .map(|p| wrap_preview(p, right_col_w))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Reserve rows for the chrome: title + blank, optional query line,
        // blank + hint, the selected preview overflow, and scroll indicators.
        let query_rows = usize::from(self.searchable);
        let reserved = 2 + query_rows + 2 + PREVIEW_EXTRA + 2;
        let window = (term_h as usize).saturating_sub(reserved).max(3);

        // Center the cursor within the viewport when the list overflows.
        let total = self.filtered.len();
        let cursor = self.cursor_pos();
        let scroll = if total <= window {
            0
        } else {
            cursor.saturating_sub(window / 2).min(total - window)
        };
        let end = (scroll + window).min(total);

        let mut lines = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("  {}", self.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        if !self.tabs.is_empty() {
            let mut spans = vec![Span::styled("  ", S_DIM)];
            for (i, tab) in self.tabs.iter().enumerate() {
                if i > 0 {
                    spans.push(Span::styled(" · ", S_DIM));
                }
                let text = format!("{} ({})", tab.label, tab.members.len());
                let style = if i == self.active_tab {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    S_DIM
                };
                spans.push(Span::styled(text, style));
            }
            lines.push(Line::from(spans));
        }
        if self.searchable {
            let q = if self.query.is_empty() {
                Span::styled("(type to search)", S_DIM)
            } else {
                Span::styled(self.query.clone(), Style::default().fg(Color::Yellow))
            };
            lines.push(Line::from(vec![Span::styled("  search: ", S_DIM), q]));
        }
        lines.push(Line::from(""));

        if total == 0 {
            lines.push(Line::from(Span::styled("  (no matches)", S_DIM)));
        }

        if scroll > 0 {
            lines.push(Line::from(Span::styled(
                format!("  \u{2191} {scroll} more"),
                S_DIM,
            )));
        }

        for &i in &self.filtered[scroll..end] {
            let option = &self.options[i];
            let is_selected = self.selected == i;

            let mut spans = vec![];

            if is_selected {
                spans.push(Span::styled("▸ ", Style::default().fg(Color::Blue)));
            } else {
                spans.push(Span::raw("  "));
            }

            spans.push(Span::styled(
                format!("{}. ", i + 1),
                Style::default().fg(Color::DarkGray),
            ));

            let label_style = if option.accepts_input && !is_selected {
                Style::default().add_modifier(Modifier::DIM)
            } else if is_selected {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            spans.push(Span::styled(option.label.clone(), label_style));

            if right_col_w > 0 {
                let left_len = spans.iter().map(|s| s.content.width()).sum::<usize>();
                let pad = left_col_w.saturating_sub(left_len);
                spans.push(Span::raw(" ".repeat(pad)));
                spans.push(Span::styled(separator.to_string(), S_DIM));
                if is_selected {
                    if let Some(pline) = selected_preview.first() {
                        spans.push(Span::styled(pline.clone(), S_DIM));
                    }
                } else if let Some(text) = self.previews.get(i) {
                    let first = text.lines().next().unwrap_or("").trim();
                    if !first.is_empty() {
                        let truncated: String = first.chars().take(right_col_w).collect();
                        spans.push(Span::styled(truncated, S_DIM));
                    }
                }
            }

            lines.push(Line::from(spans));

            if right_col_w > 0 && is_selected && selected_preview.len() > 1 {
                for pline in selected_preview.iter().skip(1).take(PREVIEW_EXTRA) {
                    let pad = left_col_w + separator.len();
                    lines.push(Line::from(Span::styled(
                        format!("{:pad$}{pline}", "", pad = pad),
                        S_DIM,
                    )));
                }
            }

            if option.accepts_input && is_selected && self.editing_input {
                lines.push(Line::from(Span::styled(
                    format!("    {}_", self.input_buffer),
                    Style::default().fg(Color::Yellow),
                )));
            }
        }

        if end < total {
            lines.push(Line::from(Span::styled(
                format!("  \u{2193} {} more", total - end),
                S_DIM,
            )));
        }

        lines.push(Line::from(""));

        let hint = if self.searchable {
            "  type: search  \u{2191}/\u{2193}: navigate  Enter: select  ^D: delete  Esc: cancel"
        } else if has_previews {
            "  \u{2191}/\u{2193}: navigate  Enter: select  d: delete  Esc: cancel"
        } else {
            "  \u{2191}/\u{2193}: navigate  Enter: select  Esc: cancel"
        };
        let hint = if !self.tabs.is_empty() {
            format!("{hint}  Tab: switch")
        } else {
            hint.to_string()
        };
        lines.push(Line::from(Span::styled(hint, S_DIM)));

        lines
    }

    pub fn set_previews(&mut self, previews: Vec<String>) {
        self.previews = previews;
    }

    pub fn remove(&mut self, index: usize) {
        if index >= self.options.len() {
            return;
        }
        self.options.remove(index);
        if index < self.previews.len() {
            self.previews.remove(index);
        }
        for tab in &mut self.tabs {
            tab.members.retain(|&m| m != index);
            for m in &mut tab.members {
                if *m > index {
                    *m -= 1;
                }
            }
        }
        if self.selected >= self.options.len() && self.selected > 0 {
            self.selected -= 1;
        }
        self.refilter();
    }

    /// Position of the highlighted entry within the filtered view.
    fn cursor_pos(&self) -> usize {
        self.filtered
            .iter()
            .position(|&i| i == self.selected)
            .unwrap_or(0)
    }

    /// Recompute `filtered` from the current query and keep `selected` valid.
    fn refilter(&mut self) {
        let q = self.query.to_lowercase();
        let base: Vec<usize> = if self.tabs.is_empty() {
            (0..self.options.len()).collect()
        } else {
            self.tabs
                .get(self.active_tab)
                .map(|t| t.members.clone())
                .unwrap_or_default()
        };
        self.filtered = base
            .into_iter()
            .filter(|&i| {
                let Some(o) = self.options.get(i) else {
                    return false;
                };
                if q.is_empty() {
                    return true;
                }
                if o.label.to_lowercase().contains(&q) {
                    return true;
                }
                self.previews
                    .get(i)
                    .map(|p| p.to_lowercase().contains(&q))
                    .unwrap_or(false)
            })
            .collect();

        // Keep the highlight on a visible entry.
        if !self.filtered.contains(&self.selected) {
            self.selected = self.filtered.first().copied().unwrap_or(0);
        }
    }

    fn up(&mut self) {
        if self.editing_input {
            self.editing_input = false;
        }
        let cursor = self.cursor_pos();
        if cursor > 0 {
            self.selected = self.filtered[cursor - 1];
        }
    }

    fn down(&mut self) {
        let cursor = self.cursor_pos();
        if cursor + 1 < self.filtered.len() {
            self.selected = self.filtered[cursor + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn opts(labels: &[&str]) -> Vec<ChoiceOption> {
        labels
            .iter()
            .map(|l| ChoiceOption {
                label: (*l).to_string(),
                accepts_input: false,
            })
            .collect()
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn tabs_filter_to_active_members() {
        // options: a b c d ; tab0 = {0,2}, tab1 = {0,1,2,3}
        let p = ChoicePicker::new("t".into(), opts(&["a", "b", "c", "d"])).with_tabs(vec![
            ("even".into(), vec![0, 2]),
            ("all".into(), vec![0, 1, 2, 3]),
        ]);
        assert_eq!(p.active_tab(), 0);
        assert_eq!(p.filtered, vec![0, 2]);
    }

    #[test]
    fn tab_key_cycles_and_wraps() {
        let mut p = ChoicePicker::new("t".into(), opts(&["a", "b", "c"]))
            .with_tabs(vec![("x".into(), vec![0]), ("y".into(), vec![1, 2])]);
        assert!(p.handle_key(key(KeyCode::Tab)).is_none());
        assert_eq!(p.active_tab(), 1);
        assert_eq!(p.filtered, vec![1, 2]);
        // wrap forward back to 0
        assert!(p.handle_key(key(KeyCode::Tab)).is_none());
        assert_eq!(p.active_tab(), 0);
        // back-tab wraps to last
        assert!(p.handle_key(key(KeyCode::BackTab)).is_none());
        assert_eq!(p.active_tab(), 1);
    }

    #[test]
    fn query_intersects_active_tab() {
        let mut p = ChoicePicker::new("t".into(), opts(&["apple", "apricot", "banana"]))
            .with_tabs(vec![("all".into(), vec![0, 1, 2])])
            .searchable(true);
        // type "ap" -> apple, apricot
        p.handle_key(key(KeyCode::Char('a')));
        p.handle_key(key(KeyCode::Char('p')));
        assert_eq!(p.filtered, vec![0, 1]);
    }

    #[test]
    fn remove_keeps_tab_members_valid() {
        let mut p = ChoicePicker::new("t".into(), opts(&["a", "b", "c"]))
            .with_tabs(vec![("all".into(), vec![0, 1, 2])]);
        // remove option index 1 ("b"); members should become {0,1} pointing at a,c
        p.remove(1);
        assert_eq!(p.options.len(), 2);
        assert_eq!(p.tabs[0].members, vec![0, 1]);
        assert_eq!(p.filtered, vec![0, 1]);
    }
}

fn wrap_preview(text: &str, width: usize) -> Vec<String> {
    let max_lines = 6;
    let mut lines = Vec::new();
    for src_line in text.lines() {
        let trimmed = src_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut pos = 0;
        let chars: Vec<char> = trimmed.chars().collect();
        while pos < chars.len() && lines.len() < max_lines {
            let end = (pos + width).min(chars.len());
            let chunk: String = chars[pos..end].iter().collect();
            lines.push(chunk);
            pos = end;
        }
        if lines.len() >= max_lines {
            break;
        }
    }
    lines
}
