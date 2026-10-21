//! Structured progress display for spawned subagents.
//!
//! Tracks running tools and renders a compact progress section for active agents.

use std::time::Instant;

use flashmind_types::AgentEvent;
use flashmind_types::utils::truncate_utf8;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::event_render::{format_secs, truncate_display};
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

        if matches!(
            event.as_ref(),
            AgentEvent::Done(_) | AgentEvent::Error(_) | AgentEvent::Cancelled
        ) {
            self.entries.retain(|entry| entry.id != *id);
            return;
        }
        if !matches!(
            event.as_ref(),
            AgentEvent::TextDelta(_)
                | AgentEvent::ReasoningDelta(_)
                | AgentEvent::ToolStart { .. }
                | AgentEvent::ToolResult { .. }
                | AgentEvent::Status(_)
        ) {
            return;
        }

        let entry = match self.entries.iter_mut().find(|e| e.id == *id) {
            Some(e) => e,
            None => {
                self.entries.push(SubagentEntry {
                    id: id.clone(),
                    task: task.clone(),
                    completed_tools: Vec::new(),
                    current_tool: None,
                    started_at: Instant::now(),
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
            AgentEvent::ToolResult { name, success, .. } => {
                if *success {
                    entry.completed_tools.push(name.clone());
                }
                entry.current_tool = None;
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
            if !entry.completed_tools.is_empty() {
                h += 1;
            }
            if entry.current_tool.is_some() {
                h += 1;
            }
        }
        h
    }

    /// Render the progress section as styled lines.
    pub fn lines(&self, width: u16) -> Vec<Line<'static>> {
        self.lines_at(width, Instant::now())
    }

    fn lines_at(&self, width: u16, now: Instant) -> Vec<Line<'static>> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let w = width as usize;
        let mut lines = Vec::new();

        // Separator
        let heading = truncate_display(" Subagents | Ctrl+B: open ", w.saturating_sub(1));
        lines.push(Line::from(Span::styled(
            format!(
                "{heading}{}",
                "\u{2500}".repeat(w.saturating_sub(heading.width() + 1))
            ),
            S_DIM,
        )));

        let count = self.entries.len().min(MAX_DISPLAY);
        for entry in self.entries.iter().rev().take(count) {
            let duration = now.duration_since(entry.started_at);
            let elapsed = truncate_display(&format_secs(duration.as_secs()), w);
            let short_id = truncate_utf8(&entry.id, 8);
            let text = truncate_display(
                &format!("  \u{25b8} {short_id}  {}", entry.task),
                w.saturating_sub(elapsed.width() + 1),
            );
            let padding = " ".repeat(w.saturating_sub(text.width() + elapsed.width()));
            lines.push(Line::from(vec![
                Span::styled(text, S_SUBAGENT),
                Span::styled(format!("{padding}{elapsed}"), S_DIM),
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
                lines.push(Line::from(Span::styled(
                    truncate_display(&tools_text, w),
                    S_DIM,
                )));
            }

            // Current running tool
            if let Some(ref tool) = entry.current_tool {
                let tool_text = format!("      \u{228c} {tool}");
                lines.push(Line::from(Span::styled(
                    truncate_display(&tool_text, w),
                    S_TOOL_RUN,
                )));
            }
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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
        assert_eq!(
            progress.entries[0].current_tool.as_deref(),
            Some("Searching files")
        );
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
    fn finished_agents_disappear_while_other_agents_stay_visible() {
        for outcome in [
            AgentEvent::Done("result".into()),
            AgentEvent::Error("request failed".into()),
            AgentEvent::Cancelled,
        ] {
            let mut progress = SubagentProgress::new();
            progress.handle_event(&spawn_event(
                "first",
                "first task",
                AgentEvent::TextDelta(String::new()),
            ));
            progress.handle_event(&spawn_event(
                "second",
                "second task",
                AgentEvent::TextDelta(String::new()),
            ));
            progress.handle_event(&spawn_event("first", "first task", outcome));
            let lines = progress.lines(80);
            assert!(
                lines
                    .iter()
                    .all(|line| !line.to_string().contains("first task"))
            );
            assert!(
                lines
                    .iter()
                    .any(|line| line.to_string().contains("second task"))
            );
            progress.handle_event(&spawn_event(
                "second",
                "second task",
                AgentEvent::Done("result".into()),
            ));
            assert!(progress.is_empty());
            assert_eq!(progress.height(), 0);
            assert!(progress.lines(80).is_empty());
            progress.handle_event(&spawn_event(
                "second",
                "second task",
                AgentEvent::Usage(Default::default()),
            ));
            assert!(progress.is_empty());
        }
    }

    #[test]
    fn follow_up_restarts_the_elapsed_timer() {
        let mut progress = SubagentProgress::new();
        progress.handle_event(&spawn_event(
            "worker",
            "task",
            AgentEvent::Done("result".into()),
        ));
        progress.handle_event(&spawn_event(
            "worker",
            "task",
            AgentEvent::TextDelta("follow-up".into()),
        ));
        let now = Instant::now();
        assert_ne!(
            progress.lines_at(80, now),
            progress.lines_at(80, now + Duration::from_secs(60)),
        );
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
    fn headers_fit_viewport_and_keep_elapsed_time() {
        for task in ["review changes ".repeat(20), "界".repeat(100)] {
            let mut progress = SubagentProgress::new();
            progress.handle_event(&spawn_event(
                "abcdef12",
                &task,
                AgentEvent::TextDelta(String::new()),
            ));

            for width in 0..100 {
                let lines = progress.lines(width);
                let header = &lines[1];
                assert!(header.width() <= width as usize, "{width}: {header:?}");
                if width >= 20 {
                    assert_eq!(header.width(), width as usize);
                    assert!(header.to_string().ends_with(" 0s"));
                }
            }
        }
    }

    #[test]
    fn tool_rows_fit_viewport() {
        let mut progress = SubagentProgress::new();
        let tool = "界".repeat(100);
        for _ in 0..4 {
            progress.handle_event(&spawn_event(
                "abcdef12",
                "review changes",
                AgentEvent::ToolResult {
                    name: tool.clone(),
                    id: "t1".into(),
                    output: String::new(),
                    success: true,
                    elapsed_ms: 100,
                    sources: Vec::new(),
                },
            ));
        }
        progress.handle_event(&spawn_event(
            "abcdef12",
            "review changes",
            AgentEvent::ToolStart {
                name: "exec".into(),
                id: "t2".into(),
                humanized: tool,
            },
        ));

        for width in 0..100 {
            let lines = progress.lines(width);
            assert_eq!(lines.len(), progress.height());
            for line in &lines[2..] {
                assert!(line.width() <= width as usize, "{width}: {line:?}");
            }
        }
    }
}
