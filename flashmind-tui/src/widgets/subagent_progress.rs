//! Structured progress display for spawned subagents.
//!
//! Tracks per-subagent state (running tools, completion) and renders a compact
//! progress section showing active and finished agents.

use std::time::Instant;

use flashmind_types::AgentEvent;
use ratatui::text::{Line, Span};

use crate::styles::*;

const MAX_DISPLAY: usize = 10;
const MAX_COMPLETED_TOOLS_SHOWN: usize = 3;

/// State of a single spawned subagent.
#[derive(Debug)]
struct SubagentEntry {
    id: String,
    task: String,
    completed_tools: Vec<String>,
    current_tool: Option<String>,
    started_at: Instant,
    finished: Option<(bool, Instant)>,
}

/// Tracks and renders progress for all spawned subagents.
#[derive(Debug, Default)]
pub struct SubagentProgress {
    entries: Vec<SubagentEntry>,
}

impl SubagentProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a `SpawnedEvent` and update the corresponding subagent state.
    pub fn handle_event(&mut self, event: &AgentEvent) {
        let AgentEvent::SpawnedEvent {
            id, task, event, ..
        } = event
        else {
            return;
        };

        let entry = match self.entries.iter_mut().find(|e| e.id == *id) {
            Some(e) => e,
            None => {
                self.entries.push(SubagentEntry {
                    id: id.clone(),
                    task: task.clone(),
                    completed_tools: Vec::new(),
                    current_tool: None,
                    started_at: Instant::now(),
                    finished: None,
                });
                self.entries.last_mut().unwrap()
            }
        };

        match event.as_ref() {
            AgentEvent::ToolStart { humanized, .. } => {
                entry.current_tool = Some(if humanized.is_empty() {
                    "running tool".to_string()
                } else {
                    humanized.clone()
                });
            }
            AgentEvent::ToolResult {
                name, success, ..
            } => {
                if *success {
                    entry.completed_tools.push(name.clone());
                }
                entry.current_tool = None;
            }
            AgentEvent::Done(_) => {
                entry.current_tool = None;
                entry.finished = Some((true, Instant::now()));
            }
            AgentEvent::Error(_) => {
                entry.current_tool = None;
                entry.finished = Some((false, Instant::now()));
            }
            _ => {}
        }
    }

    /// Whether there are any tracked subagents.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of lines the progress section will render.
    pub fn height(&self) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        let count = self.entries.len().min(MAX_DISPLAY);
        let mut h = 1; // separator
        for entry in self.entries.iter().rev().take(count) {
            h += 1; // header line
            if entry.finished.is_none() {
                if !entry.completed_tools.is_empty() {
                    h += 1;
                }
                if entry.current_tool.is_some() {
                    h += 1;
                }
            }
        }
        h
    }

    /// Render the progress section as styled lines.
    pub fn lines(&self, width: u16) -> Vec<Line<'static>> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let w = width as usize;
        let mut lines = Vec::new();

        // Separator
        lines.push(Line::from(Span::styled(
            "\u{2500}".repeat(w.saturating_sub(1)),
            S_DIM,
        )));

        let count = self.entries.len().min(MAX_DISPLAY);
        for entry in self.entries.iter().rev().take(count) {
            let elapsed = format_elapsed_since(entry.started_at);
            let short_id = &entry.id[..entry.id.len().min(8)];
            let task_max = w.saturating_sub(16 + elapsed.len());
            let task_display: String = entry.task.chars().take(task_max).collect();

            if let Some((success, _)) = entry.finished {
                let (icon, style) = if success {
                    ("\u{2714}", S_TOOL_OK) // ✔
                } else {
                    ("\u{2718}", S_TOOL_FAIL) // ✘
                };
                let text = format!("  {icon} {short_id}  {task_display}");
                let text_w = unicode_width::UnicodeWidthStr::width(text.as_str());
                let pad = format!("{:>w$}", elapsed, w = w.saturating_sub(text_w));
                lines.push(Line::from(vec![
                    Span::styled(text, style),
                    Span::styled(pad, S_DIM),
                ]));
            } else {
                // Active agent header
                let text = format!("  \u{25b8} {short_id}  {task_display}");
                let text_w = unicode_width::UnicodeWidthStr::width(text.as_str());
                let pad = format!("{:>w$}", elapsed, w = w.saturating_sub(text_w));
                lines.push(Line::from(vec![
                    Span::styled(text, S_SUBAGENT),
                    Span::styled(pad, S_DIM),
                ]));

                // Completed tools
                if !entry.completed_tools.is_empty() {
                    let total = entry.completed_tools.len();
                    let shown: Vec<&str> = entry
                        .completed_tools
                        .iter()
                        .rev()
                        .take(MAX_COMPLETED_TOOLS_SHOWN)
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    let mut tools_text = format!("      \u{25cf} {}", shown.join(", "));
                    if total > MAX_COMPLETED_TOOLS_SHOWN {
                        tools_text
                            .push_str(&format!(", +{} others", total - MAX_COMPLETED_TOOLS_SHOWN));
                    }
                    lines.push(Line::from(Span::styled(tools_text, S_DIM)));
                }

                // Current running tool
                if let Some(ref tool) = entry.current_tool {
                    let tool_text = format!("      \u{228c} {tool}");
                    lines.push(Line::from(Span::styled(tool_text, S_TOOL_RUN)));
                }
            }
        }

        lines
    }

    /// Remove all finished subagent entries.
    pub fn clear_finished(&mut self) {
        self.entries.retain(|e| e.finished.is_none());
    }
}

fn format_elapsed_since(start: Instant) -> String {
    let secs = start.elapsed().as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m{:02}s", secs / 60, secs % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_event(id: &str, task: &str, inner: AgentEvent) -> AgentEvent {
        AgentEvent::SpawnedEvent {
            id: id.to_string(),
            task: task.to_string(),
            model: None,
            event: Box::new(inner),
        }
    }

    #[test]
    fn tracks_new_subagent() {
        let mut progress = SubagentProgress::new();
        assert!(progress.is_empty());

        progress.handle_event(&spawn_event(
            "abc12345",
            "find bugs",
            AgentEvent::ToolStart {
                name: "grep".into(),
                id: "t1".into(),
                humanized: "Searching files".into(),
            },
        ));

        assert!(!progress.is_empty());
        assert_eq!(progress.entries.len(), 1);
        assert_eq!(progress.entries[0].current_tool.as_deref(), Some("Searching files"));
    }

    #[test]
    fn tool_result_moves_to_completed() {
        let mut progress = SubagentProgress::new();
        progress.handle_event(&spawn_event(
            "abc",
            "task",
            AgentEvent::ToolStart {
                name: "exec".into(),
                id: "t1".into(),
                humanized: "Running tests".into(),
            },
        ));
        progress.handle_event(&spawn_event(
            "abc",
            "task",
            AgentEvent::ToolResult {
                name: "exec".into(),
                id: "t1".into(),
                output: String::new(),
                success: true,
                elapsed_ms: 100,
                sources: Vec::new(),
            },
        ));

        assert!(progress.entries[0].current_tool.is_none());
        assert_eq!(progress.entries[0].completed_tools.len(), 1);
    }

    #[test]
    fn done_marks_finished() {
        let mut progress = SubagentProgress::new();
        progress.handle_event(&spawn_event(
            "abc",
            "task",
            AgentEvent::Done("result".into()),
        ));

        assert!(progress.entries[0].finished.is_some());
        assert!(progress.entries[0].finished.unwrap().0);
    }

    #[test]
    fn renders_lines() {
        let mut progress = SubagentProgress::new();
        progress.handle_event(&spawn_event(
            "abcdef12",
            "find bugs",
            AgentEvent::ToolStart {
                name: "grep".into(),
                id: "t1".into(),
                humanized: "Searching".into(),
            },
        ));

        let lines = progress.lines(80);
        assert!(lines.len() >= 3); // separator + header + current tool
    }

    #[test]
    fn clear_finished_removes_done() {
        let mut progress = SubagentProgress::new();
        progress.handle_event(&spawn_event(
            "abc",
            "task1",
            AgentEvent::Done("done".into()),
        ));
        progress.handle_event(&spawn_event(
            "def",
            "task2",
            AgentEvent::ToolStart {
                name: "exec".into(),
                id: "t1".into(),
                humanized: "running".into(),
            },
        ));
        assert_eq!(progress.entries.len(), 2);
        progress.clear_finished();
        assert_eq!(progress.entries.len(), 1);
        assert_eq!(progress.entries[0].id, "def");
    }
}
