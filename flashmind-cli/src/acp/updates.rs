//! flsh's agent events as ACP session updates.

use std::collections::HashMap;
use std::path::PathBuf;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use flashmind_types::AgentEvent;

/// The most output of a running tool kept for its live view, in bytes.
const MAX_PROGRESS: usize = 8 * 1024;

/// How a turn ended, from its terminal event.
#[derive(Debug, PartialEq)]
pub(super) enum Ending {
    Done,
    Cancelled,
    Failed(String),
}

impl Ending {
    /// The ending `event` marks, or `None` when the turn goes on.
    pub(super) fn of(event: &AgentEvent) -> Option<Self> {
        match event {
            AgentEvent::Done(_) => Some(Self::Done),
            AgentEvent::Cancelled => Some(Self::Cancelled),
            AgentEvent::Error(error) => Some(Self::Failed(error.clone())),
            _ => None,
        }
    }
}

/// Turns one turn's events into updates. It remembers what a later event
/// needs: the files changed by the tool whose result comes next, and the
/// output of each running tool so far.
#[derive(Default)]
pub(super) struct Updates {
    changed: Vec<PathBuf>,
    progress: HashMap<String, String>,
}

impl Updates {
    /// The updates `event` sends; none for an event the client has no use for.
    pub(super) fn updates(&mut self, event: &AgentEvent) -> Vec<SessionUpdate> {
        match event {
            AgentEvent::TextDelta(text) if !text.is_empty() => {
                vec![SessionUpdate::AgentMessageChunk(chunk(text))]
            }
            AgentEvent::ReasoningDelta(text) if !text.is_empty() => {
                vec![SessionUpdate::AgentThoughtChunk(chunk(text))]
            }
            AgentEvent::ToolStart {
                name,
                id,
                humanized,
            } => {
                let call = ToolCall::new(id.clone(), title(name, humanized))
                    .kind(kind(name))
                    .status(ToolCallStatus::InProgress);
                vec![SessionUpdate::ToolCall(call)]
            }
            AgentEvent::ToolProgress { id, line } => {
                let output = self.progress.entry(id.clone()).or_default();
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(line);
                keep_tail(output, MAX_PROGRESS);
                let fields = ToolCallUpdateFields::new().content(vec![text_content(output)]);
                vec![update(id, fields)]
            }
            AgentEvent::FileDiff { path, .. } => {
                self.changed.push(PathBuf::from(path));
                Vec::new()
            }
            AgentEvent::ToolResult {
                id,
                output,
                success,
                ..
            } => {
                self.progress.remove(id);
                let status = if *success {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                };
                let mut fields = ToolCallUpdateFields::new()
                    .status(status)
                    .content(vec![text_content(output)]);
                if !self.changed.is_empty() {
                    let locations: Vec<ToolCallLocation> =
                        self.changed.drain(..).map(ToolCallLocation::new).collect();
                    fields = fields.locations(locations);
                }
                vec![update(id, fields)]
            }
            AgentEvent::Interrupted {
                tool_call_id,
                output,
                ..
            } => {
                self.progress.remove(tool_call_id);
                let fields = ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .content(vec![text_content(output)]);
                vec![update(tool_call_id, fields)]
            }
            _ => Vec::new(),
        }
    }
}

fn chunk(text: &str) -> ContentChunk {
    ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
}

fn text_content(text: &str) -> ToolCallContent {
    ToolCallContent::from(ContentBlock::Text(TextContent::new(text)))
}

fn update(id: &str, fields: ToolCallUpdateFields) -> SessionUpdate {
    SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id.to_owned(), fields))
}

/// A tool call's title: the tool and a summary of its arguments.
fn title(name: &str, humanized: &str) -> String {
    if humanized.is_empty() || humanized == name {
        name.to_owned()
    } else {
        format!("{name}: {humanized}")
    }
}

/// The ACP kind of the flsh tool `name`.
fn kind(name: &str) -> ToolKind {
    match name {
        "read" | "image_read" => ToolKind::Read,
        "edit" | "write" => ToolKind::Edit,
        "exec" | "process" => ToolKind::Execute,
        "find" | "glob" | "grep" | "lsp" => ToolKind::Search,
        "http_request" | "brave_search" | "firecrawl_search" | "web_search_read" => ToolKind::Fetch,
        _ if name.starts_with("web_") => ToolKind::Fetch,
        _ => ToolKind::Other,
    }
}

/// Cuts `text` to its last `max` bytes at most, on a character boundary.
fn keep_tail(text: &mut String, max: usize) {
    if text.len() <= max {
        return;
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text.drain(..start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_call_reports_its_changed_files_with_its_result() {
        let mut updates = Updates::default();
        let started = updates.updates(&AgentEvent::ToolStart {
            name: "edit".into(),
            id: "call-1".into(),
            humanized: "src/lib.rs".into(),
        });
        let [SessionUpdate::ToolCall(call)] = started.as_slice() else {
            panic!("expected a tool call, got {started:?}");
        };
        assert_eq!(call.title, "edit: src/lib.rs");
        assert_eq!(call.kind, ToolKind::Edit);
        assert_eq!(call.status, ToolCallStatus::InProgress);

        let diff = AgentEvent::FileDiff {
            path: "src/lib.rs".into(),
            diff: Vec::new(),
        };
        assert!(updates.updates(&diff).is_empty());
        let finished = updates.updates(&AgentEvent::ToolResult {
            name: "edit".into(),
            id: "call-1".into(),
            output: "edited".into(),
            success: true,
            elapsed_ms: 1,
            sources: Vec::new(),
        });
        let [SessionUpdate::ToolCallUpdate(update)] = finished.as_slice() else {
            panic!("expected a tool call update, got {finished:?}");
        };
        assert_eq!(update.tool_call_id.to_string(), "call-1");
        assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));
        let locations = update.fields.locations.as_deref().unwrap_or_default();
        assert_eq!(locations[0].path, PathBuf::from("src/lib.rs"));
    }

    #[test]
    fn progress_keeps_only_the_tail_of_long_output() {
        let mut text = "ab€cd".to_owned();
        keep_tail(&mut text, 4);
        assert_eq!(text, "cd");
        let mut short = "abc".to_owned();
        keep_tail(&mut short, 4);
        assert_eq!(short, "abc");
    }
}
