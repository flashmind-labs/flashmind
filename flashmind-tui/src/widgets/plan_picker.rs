//! Interactive plan approval widget.
//!
//! Presents a list of plan steps with checkboxes that the user can toggle,
//! expand, reorder, inline-edit, and approve or reject.  The widget is a
//! standalone state machine — call [`PlanPicker::handle_key`] with keyboard
//! events and [`PlanPicker::lines`] to render the current state.

use std::collections::HashSet;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::styles::S_DIM;

// ---------------------------------------------------------------------------
// Public types

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct PlanResponse {
    pub approved: bool,
    pub steps: Vec<PlanStepResponse>,
    pub feedback: String,
}

#[derive(Debug, Clone)]
pub struct PlanStepResponse {
    pub id: String,
    pub description: String,
    pub accepted: bool,
}

#[derive(Debug)]
pub enum PlanPickerAction {
    Approve(PlanResponse),
    Reject(PlanResponse),
}

// ---------------------------------------------------------------------------
// Widget

#[derive(Debug)]
pub struct PlanPicker {
    pub title: String,
    pub steps: Vec<PlanPickerStep>,
    pub selected: usize,
    pub feedback: String,
    pub editing_feedback: bool,
    pub editing_step: Option<usize>,
    pub edit_buffer: String,
    pub expanded: HashSet<usize>,
}

#[derive(Debug)]
pub struct PlanPickerStep {
    pub id: String,
    pub description: String,
    pub accepted: bool,
}

impl PlanPicker {
    pub fn new(title: String, steps: Vec<PlanStep>) -> Self {
        let picker_steps = steps
            .into_iter()
            .map(|s| PlanPickerStep {
                id: s.id,
                description: s.description,
                accepted: true,
            })
            .collect();

        Self {
            title,
            steps: picker_steps,
            selected: 0,
            feedback: String::new(),
            editing_feedback: false,
            editing_step: None,
            edit_buffer: String::new(),
            expanded: HashSet::new(),
        }
    }

    pub fn approve(&self) -> PlanResponse {
        PlanResponse {
            approved: true,
            steps: self
                .steps
                .iter()
                .map(|s| PlanStepResponse {
                    id: s.id.clone(),
                    description: s.description.clone(),
                    accepted: s.accepted,
                })
                .collect(),
            feedback: self.feedback.clone(),
        }
    }

    pub fn reject(&self) -> PlanResponse {
        PlanResponse {
            approved: false,
            steps: Vec::new(),
            feedback: self.feedback.clone(),
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<PlanPickerAction> {
        if let Some(idx) = self.editing_step {
            match key.code {
                KeyCode::Enter => {
                    if let Some(step) = self.steps.get_mut(idx) {
                        step.description = self.edit_buffer.clone();
                    }
                    self.editing_step = None;
                    self.edit_buffer.clear();
                }
                KeyCode::Esc => {
                    self.editing_step = None;
                    self.edit_buffer.clear();
                }
                KeyCode::Char(c) => self.edit_buffer.push(c),
                KeyCode::Backspace => {
                    self.edit_buffer.pop();
                }
                _ => {}
            }
            return None;
        }

        if self.editing_feedback {
            match key.code {
                KeyCode::Enter => return Some(PlanPickerAction::Approve(self.approve())),
                KeyCode::Esc => return Some(PlanPickerAction::Reject(self.reject())),
                KeyCode::Up => self.up(),
                KeyCode::Char(c) => self.feedback.push(c),
                KeyCode::Backspace => {
                    self.feedback.pop();
                }
                _ => {}
            }
            return None;
        }

        match key.code {
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
            KeyCode::Char(' ') => self.toggle(),
            KeyCode::Char('e') => self.toggle_expand(),
            KeyCode::Char('E') => {
                if let Some(step) = self.steps.get(self.selected) {
                    self.edit_buffer = step.description.clone();
                    self.editing_step = Some(self.selected);
                }
            }
            KeyCode::Char('a') => {
                let new_id = format!("{}", self.steps.len() + 1);
                let insert_at = (self.selected + 1).min(self.steps.len());
                self.steps.insert(
                    insert_at,
                    PlanPickerStep {
                        id: new_id,
                        description: String::new(),
                        accepted: true,
                    },
                );
                self.selected = insert_at;
                self.edit_buffer.clear();
                self.editing_step = Some(self.selected);
            }
            KeyCode::Char('d') if self.steps.len() > 1 => {
                self.steps.remove(self.selected);
                if self.selected >= self.steps.len() {
                    self.selected = self.steps.len() - 1;
                }
            }
            KeyCode::Enter => return Some(PlanPickerAction::Approve(self.approve())),
            KeyCode::Esc => return Some(PlanPickerAction::Reject(self.reject())),
            _ => {}
        }
        None
    }

    pub fn lines(&self, max_width: u16) -> Vec<Line<'static>> {
        let mut lines = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("  Plan: {}", self.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        let max_desc = max_width.saturating_sub(8) as usize;

        for (i, step) in self.steps.iter().enumerate() {
            let checkbox = if step.accepted { "[x]" } else { "[ ]" };
            let is_selected = !self.editing_feedback && self.selected == i;
            let is_editing = self.editing_step == Some(i);
            let is_expanded = self.expanded.contains(&i);

            let prefix = if is_selected { "▸ " } else { "  " };
            let prefix_style = if is_selected {
                Style::default().fg(Color::Blue)
            } else {
                Style::default()
            };

            let style = if is_selected {
                Style::default().add_modifier(Modifier::BOLD)
            } else if !step.accepted {
                Style::default().add_modifier(Modifier::DIM)
            } else {
                Style::default()
            };

            if is_editing {
                lines.push(Line::from(vec![
                    Span::styled(prefix, prefix_style),
                    Span::styled(format!("{checkbox} {}_", self.edit_buffer), style),
                ]));
            } else if is_expanded {
                let mut desc_lines = step.description.lines();
                if let Some(first) = desc_lines.next() {
                    lines.push(Line::from(vec![
                        Span::styled(prefix, prefix_style),
                        Span::styled(format!("{checkbox} {first}"), style),
                    ]));
                }
                let cont_style = if !step.accepted {
                    Style::default().add_modifier(Modifier::DIM)
                } else {
                    Style::default()
                };
                for line in desc_lines {
                    lines.push(Line::from(Span::styled(
                        format!("      {line}"),
                        cont_style,
                    )));
                }
            } else {
                let desc = step.description.lines().next().unwrap_or("");
                let content = if desc.len() > max_desc {
                    let truncated = &desc[..desc
                        .char_indices()
                        .take(max_desc.saturating_sub(1))
                        .last()
                        .map_or(0, |(i, c)| i + c.len_utf8())];
                    format!("{checkbox} {truncated}\u{2026}")
                } else {
                    format!("{checkbox} {desc}")
                };
                lines.push(Line::from(vec![
                    Span::styled(prefix, prefix_style),
                    Span::styled(content, style),
                ]));
            }
        }

        lines.push(Line::from(""));

        let feedback_selected = self.editing_feedback;
        let feedback_prefix = if feedback_selected { "▸ " } else { "  " };
        let feedback_style = if feedback_selected {
            Style::default()
        } else {
            Style::default().add_modifier(Modifier::DIM)
        };

        let feedback_text = if self.feedback.is_empty() && !feedback_selected {
            "Tell the agent what to change...".to_string()
        } else {
            format!("{}_", self.feedback)
        };

        lines.push(Line::from(vec![
            Span::styled(feedback_prefix, Style::default().fg(Color::Blue)),
            Span::styled(format!("> {feedback_text}"), feedback_style),
        ]));

        lines.push(Line::from(""));

        lines.push(Line::from(Span::styled(
            "  Space: toggle  e: expand  a: add  d: delete  Enter: approve  Esc: reject",
            S_DIM,
        )));

        lines
    }

    fn up(&mut self) {
        if self.editing_feedback {
            self.editing_feedback = false;
            self.selected = self.steps.len().saturating_sub(1);
        } else if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn down(&mut self) {
        if self.selected + 1 < self.steps.len() {
            self.selected += 1;
        } else {
            self.editing_feedback = true;
        }
    }

    fn toggle(&mut self) {
        if !self.editing_feedback
            && let Some(step) = self.steps.get_mut(self.selected)
        {
            step.accepted = !step.accepted;
        }
    }

    fn toggle_expand(&mut self) {
        if !self.editing_feedback {
            if self.expanded.contains(&self.selected) {
                self.expanded.remove(&self.selected);
            } else {
                self.expanded.insert(self.selected);
            }
        }
    }
}
