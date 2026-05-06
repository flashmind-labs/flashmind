//! Interactive choice selection widget.
//!
//! Presents a list of options that the user can navigate and select.  Some
//! options may accept free-form text input.  The widget is a standalone state
//! machine — call [`ChoicePicker::handle_key`] with keyboard events and
//! [`ChoicePicker::lines`] to render the current state.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
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
}

// ---------------------------------------------------------------------------
// Widget

pub struct ChoicePicker {
    pub title: String,
    pub options: Vec<ChoiceOption>,
    pub selected: usize,
    pub input_buffer: String,
    pub editing_input: bool,
}

impl ChoicePicker {
    pub fn new(title: String, options: Vec<ChoiceOption>) -> Self {
        Self {
            title,
            options,
            selected: 0,
            input_buffer: String::new(),
            editing_input: false,
        }
    }

    pub fn respond(&self) -> ChoiceResponse {
        let option = &self.options[self.selected];
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
            _ => {}
        }
        None
    }

    pub fn lines(&self) -> Vec<Line<'static>> {
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

            lines.push(Line::from(spans));

            if option.accepts_input && is_selected && self.editing_input {
                lines.push(Line::from(Span::styled(
                    format!("    {}_", self.input_buffer),
                    Style::default().fg(Color::Yellow),
                )));
            }
        }

        lines.push(Line::from(""));

        lines.push(Line::from(Span::styled(
            "  \u{2191}/\u{2193}: navigate  Enter: select  Esc: cancel",
            S_DIM,
        )));

        lines
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
