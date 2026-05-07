//! Data types for session entries.

use serde::{Deserialize, Serialize};

/// Kind of session entry, determining how it maps to conversation messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntryKind {
    /// System prompt message.
    SystemPrompt,
    /// Developer-injected message with optional tag.
    Developer {
        /// Optional tag for categorizing the developer message.
        tag: Option<String>,
    },
    /// User message.
    User,
    /// Assistant response.
    Assistant,
    /// Tool result message.
    Tool,
}

/// A single entry in a persisted session (conversation history).
#[derive(Debug, Clone)]
pub struct SessionEntry {
    /// Row ID (0 for unsaved entries).
    pub id: i64,
    /// Conversation key grouping related entries.
    pub chat_key: String,
    /// The kind of this entry (system, user, assistant, etc.).
    pub entry_kind: SessionEntryKind,
    /// Text content of the entry.
    pub content: String,
    /// Serialized tool calls, if this is an assistant message with tool use.
    pub tool_calls: Option<serde_json::Value>,
    /// Tool call ID, if this is a tool result message.
    pub tool_call_id: Option<String>,
    /// Tool name, if this is a tool result message.
    pub tool_name: Option<String>,
    /// Arbitrary metadata attached to this entry.
    pub metadata: Option<serde_json::Value>,
    /// Position within the conversation (0-based).
    pub turn_index: i64,
    /// Unix timestamp when this entry was created.
    pub created_at: i64,
}
