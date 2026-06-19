//! Interactive choice selection widget.
//!
//! Presents a list of options that the user can navigate and select.  Some
//! options may accept free-form text input.  The widget is a standalone state
//! machine — call [`ChoicePicker::handle_key`] with keyboard events and
//! [`ChoicePicker::lines`] to render the current state.

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
    pub selected: usize,
    pub input_buffer: String,
    pub editing_input: bool,
    previews: Vec<String>,
}

impl ChoicePicker {
    pub fn new(title: String, options: Vec<ChoiceOption>) -> Self {
        Self {
            title,
            options,
            selected: 0,
            input_buffer: String::new(),
            editing_input: false,
            previews: Vec::new(),
        }
    }

    pub fn with_previews(mut self, previews: Vec<String>) -> Self {
        self.previews = previews;
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

    pub fn lines(&self) -> Vec<Line<'static>> {
        let has_previews = !self.previews.is_empty();
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
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

        let selected_preview: Vec<String> = if right_col_w > 0 {
            self.previews
                .get(self.selected)
                .map(|p| wrap_preview(p, right_col_w))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut lines = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("  {}", self.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        for (i, option) in self.options.iter().enumerate() {
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
                for pline in &selected_preview[1..] {
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

        lines.push(Line::from(""));

        let hint = if has_previews {
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
        }
    }

    fn up(&mut self) {
        if self.editing_input {
            self.editing_input = false;
            if self.selected > 0 {
                self.selected -= 1;
            }
        } else if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn down(&mut self) {
        if self.selected + 1 < self.options.len() {
            self.selected += 1;
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
