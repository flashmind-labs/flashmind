//! Replay the stored transcript before answering session/load.

use agent_client_protocol::schema::v1::{
    Content, ContentBlock, ContentChunk, ImageContent, SessionId, SessionNotification,
    SessionUpdate, TextContent, ToolCall, ToolCallContent, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields,
};
use agent_client_protocol::{Client, ConnectionTo, Error};
use flashmind_core::{ConversationEntry, EntryKind};
use flashmind_types::ContentPart;

pub(super) fn replay(
    cx: &ConnectionTo<Client>,
    id: &SessionId,
    entries: &[ConversationEntry],
) -> Result<(), Error> {
    for entry in entries {
        for update in updates(entry) {
            cx.send_notification(SessionNotification::new(id.clone(), update))?;
        }
    }
    Ok(())
}

fn chunk(text: &str) -> ContentChunk {
    ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
}

fn updates(entry: &ConversationEntry) -> Vec<SessionUpdate> {
    let mut updates = Vec::new();
    match &entry.kind {
        EntryKind::User { content, parts } => {
            if !content.is_empty() {
                updates.push(SessionUpdate::UserMessageChunk(chunk(content)));
            }
            for part in parts.iter().flatten() {
                let block = match part {
                    ContentPart::Text { text } => ContentBlock::Text(TextContent::new(text)),
                    ContentPart::Image { media_type, data } => {
                        ContentBlock::Image(ImageContent::new(data.clone(), media_type.clone()))
                    }
                    _ => continue,
                };
                updates.push(SessionUpdate::UserMessageChunk(ContentChunk::new(block)));
            }
        }
        EntryKind::Assistant {
            content,
            reasoning,
            tool_calls,
        } => {
            if let Some(reasoning) = reasoning.as_ref().filter(|text| !text.is_empty()) {
                updates.push(SessionUpdate::AgentThoughtChunk(chunk(reasoning)));
            }
            if !content.is_empty() {
                updates.push(SessionUpdate::AgentMessageChunk(chunk(content)));
            }
            for call in tool_calls.iter().flatten() {
                updates.push(SessionUpdate::ToolCall(
                    ToolCall::new(call.id.clone(), call.name.clone())
                        .raw_input(call.arguments.clone())
                        .status(ToolCallStatus::InProgress),
                ));
            }
        }
        EntryKind::Tool {
            call_id, output, ..
        } => {
            let fields = ToolCallUpdateFields::new()
                .status(ToolCallStatus::Completed)
                .content(vec![ToolCallContent::Content(Content::new(
                    ContentBlock::Text(TextContent::new(output)),
                ))]);
            updates.push(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                call_id.clone(),
                fields,
            )));
        }
        _ => {}
    }
    updates
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ToolCall as StoredToolCall;
    use serde_json::json;

    #[test]
    fn replay_preserves_message_thought_and_tool_order() {
        let entries = [
            ConversationEntry::system("private instructions"),
            ConversationEntry::user("question"),
            ConversationEntry::assistant_with_tool_calls(
                "checking",
                vec![StoredToolCall {
                    id: "call-1".into(),
                    name: "read_file".into(),
                    arguments: json!({"path": "example.txt"}),
                }],
                Some("thought".into()),
            ),
            ConversationEntry::tool("call-1", "result"),
            ConversationEntry::assistant("answer"),
        ];
        let replayed = entries.iter().flat_map(updates).collect::<Vec<_>>();
        assert_eq!(replayed.len(), 6);
        assert!(matches!(replayed[0], SessionUpdate::UserMessageChunk(_)));
        assert!(matches!(replayed[1], SessionUpdate::AgentThoughtChunk(_)));
        assert!(matches!(replayed[2], SessionUpdate::AgentMessageChunk(_)));
        assert!(
            matches!(&replayed[3], SessionUpdate::ToolCall(call) if call.tool_call_id.0.as_ref() == "call-1" && call.raw_input == Some(json!({"path":"example.txt"})))
        );
        assert!(
            matches!(&replayed[4], SessionUpdate::ToolCallUpdate(call) if call.tool_call_id.0.as_ref() == "call-1")
        );
        assert!(matches!(replayed[5], SessionUpdate::AgentMessageChunk(_)));
    }
}
