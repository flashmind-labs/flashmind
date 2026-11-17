//! flsh's agent events as ACP session updates.

use std::collections::HashMap;
use std::path::PathBuf;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use flashmind_types::AgentEvent;
use serde_json::json;

use super::media::{ToolMedia, reply_image};

/// The most output of a running tool kept for its live view, in bytes.
const MAX_PROGRESS: usize = 8 * 1024;

/// Turns one turn's events into updates. It remembers what a later event
/// needs: the files changed by the tool whose result comes next, the output
/// of each running tool so far, and the media of each finished tool whose
/// result comes next.
#[derive(Default)]
pub(super) struct Updates {
    children: HashMap<String, Updates>,
    text: String,
    changed: Vec<PathBuf>,
    progress: HashMap<String, String>,
    media: HashMap<String, ToolMedia>,
}

impl Updates {
    /// The updates `event` sends; none for an event the client has no use for.
    pub(super) fn updates(&mut self, event: &AgentEvent) -> Vec<SessionUpdate> {
        match event {
            AgentEvent::SpawnedEvent {
                id,
                task,
                model,
                event,
            } => {
                let owner = format!("flsh-agent:{id}");
                let mut updates = Vec::new();
                if !self.children.contains_key(id) {
                    updates.push(SessionUpdate::ToolCall(
                        ToolCall::new(owner.clone(), task.clone())
                            .name("flsh_agent")
                            .raw_input(json!({"flshAgent": true, "name": id, "model": model.as_ref().map(ToString::to_string)}))
                            .status(ToolCallStatus::InProgress),
                    ));
                }
                let child = self.children.entry(id.clone()).or_default();
                match event.as_ref() {
                    AgentEvent::TextDelta(text) => {
                        child.text.push_str(text);
                        keep_tail(&mut child.text, MAX_PROGRESS);
                        updates.push(update(
                            &owner,
                            ToolCallUpdateFields::new().content(vec![text_content(&child.text)]),
                        ));
                    }
                    AgentEvent::Done(result) | AgentEvent::Error(result) => {
                        let status = if matches!(event.as_ref(), AgentEvent::Done(_)) {
                            ToolCallStatus::Completed
                        } else {
                            ToolCallStatus::Failed
                        };
                        updates.push(update(
                            &owner,
                            ToolCallUpdateFields::new()
                                .status(status)
                                .content(vec![text_content(result)]),
                        ));
                    }
                    AgentEvent::Cancelled => {
                        updates.push(update(
                            &owner,
                            ToolCallUpdateFields::new().status(ToolCallStatus::Failed),
                        ));
                    }
                    _ => {
                        for mut update in child.updates(event) {
                            // Child messages and thoughts must not enter the parent's reply.
                            match &mut update {
                                SessionUpdate::ToolCall(call) => {
                                    call.tool_call_id =
                                        format!("{owner}:{}", call.tool_call_id).into();
                                    call.meta
                                        .get_or_insert_with(Default::default)
                                        .insert("flsh".into(), json!({"parentToolUseId": owner}));
                                }
                                SessionUpdate::ToolCallUpdate(call) => {
                                    call.tool_call_id =
                                        format!("{owner}:{}", call.tool_call_id).into();
                                }
                                _ => continue,
                            }
                            updates.push(update);
                        }
                    }
                }
                updates
            }
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
            AgentEvent::Attachment {
                media_type, data, ..
            } => reply_image(media_type, data)
                .map(|image| SessionUpdate::AgentMessageChunk(ContentChunk::new(image)))
                .into_iter()
                .collect(),
            AgentEvent::ToolMedia { id, parts, files } => {
                self.media.insert(id.clone(), ToolMedia::new(parts, files));
                Vec::new()
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
                let media = self.media.remove(id).unwrap_or_default();
                let mut content = vec![text_content(output)];
                content.extend(media.content);
                let mut fields = ToolCallUpdateFields::new().status(status).content(content);
                let mut locations: Vec<ToolCallLocation> =
                    self.changed.drain(..).map(ToolCallLocation::new).collect();
                locations.extend(media.locations);
                if !locations.is_empty() {
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
        "read" | "image_read" | "attach" => ToolKind::Read,
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

    fn spawned(event: AgentEvent) -> AgentEvent {
        AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "Check files".into(),
            model: None,
            event: Box::new(event),
        }
    }

    #[test]
    fn spawned_tools_share_one_card_and_child_text_stays_out_of_parent_reply() {
        let mut updates = Updates::default();
        let started = updates.updates(&spawned(AgentEvent::ToolStart {
            name: "read".into(),
            id: "call-1".into(),
            humanized: "src/lib.rs".into(),
        }));
        assert_eq!(started.len(), 2);
        let SessionUpdate::ToolCall(card) = &started[0] else {
            panic!("expected card")
        };
        assert_eq!(card.tool_call_id.to_string(), "flsh-agent:worker");
        assert_eq!(card.raw_input.as_ref().unwrap()["flshAgent"], true);
        let SessionUpdate::ToolCall(step) = &started[1] else {
            panic!("expected step")
        };
        assert_eq!(step.tool_call_id.to_string(), "flsh-agent:worker:call-1");
        assert_eq!(
            step.meta.as_ref().unwrap()["flsh"]["parentToolUseId"],
            "flsh-agent:worker"
        );
        let text = updates.updates(&spawned(AgentEvent::TextDelta("Checking".into())));
        assert!(matches!(
            text.as_slice(),
            [SessionUpdate::ToolCallUpdate(_)]
        ));
        assert!(
            updates
                .updates(&spawned(AgentEvent::ReasoningDelta(
                    "Private thought".into()
                )))
                .is_empty()
        );
        let finished = updates.updates(&spawned(AgentEvent::Done("Checked.".into())));
        let [SessionUpdate::ToolCallUpdate(card)] = finished.as_slice() else {
            panic!("expected completion")
        };
        assert_eq!(card.fields.status, Some(ToolCallStatus::Completed));
    }

    #[test]
    fn child_failure_and_cancellation_close_the_card() {
        for event in [AgentEvent::Error("failed".into()), AgentEvent::Cancelled] {
            let mut updates = Updates::default();
            let result = updates.updates(&spawned(event));
            let SessionUpdate::ToolCallUpdate(card) = &result[1] else {
                panic!("expected failure")
            };
            assert_eq!(card.fields.status, Some(ToolCallStatus::Failed));
        }
    }

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
