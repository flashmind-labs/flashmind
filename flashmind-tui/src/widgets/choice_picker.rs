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

        if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(ChoicePickerAction::Cancel);
        }

        match key.code {
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
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
            cursor
                .saturating_sub(window / 2)
                .min(total - window)
        };
        let end = (scroll + window).min(total);

        let mut lines = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("  {}", self.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        if self.searchable {
            let q = if self.query.is_empty() {
                Span::styled("(type to search)", S_DIM)
            } else {
                Span::styled(self.query.clone(), Style::default().fg(Color::Yellow))
            };
            lines.push(Line::from(vec![
                Span::styled("  search: ", S_DIM),
                q,
            ]));
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
                let left_len = spans.iter().map(|s| s.content.len()).sum::<usize>();
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
        lines.push(Line::from(Span::styled(hint, S_DIM)));

        lines
    }

    pub fn set_previews(&mut self, previews: Vec<String>) {
        self.previews = previews;
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.options.len() {
            self.options.remove(index);
            if index < self.previews.len() {
                self.previews.remove(index);
            }
            if self.selected >= self.options.len() && self.selected > 0 {
                self.selected -= 1;
            }
            self.refilter();
        }
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
        self.filtered = self
            .options
            .iter()
            .enumerate()
            .filter(|(i, o)| {
                if q.is_empty() {
                    return true;
                }
                if o.label.to_lowercase().contains(&q) {
                    return true;
                }
                self.previews
                    .get(*i)
                    .map(|p| p.to_lowercase().contains(&q))
                    .unwrap_or(false)
            })
            .map(|(i, _)| i)
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
