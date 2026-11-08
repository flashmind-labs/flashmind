//! Data types for session entries.

use flashmind_types::SessionUsage;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Entry ID (0 for entries saved through a conversation rewrite).
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

/// Summary of a stored session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionSummary {
    /// Conversation key.
    pub chat_key: String,
    /// Number of entries in the session.
    pub entry_count: i64,
    /// Unix timestamp of the most recent entry.
    pub last_updated: i64,
    /// Optional human-readable title for the session.
    pub title: Option<String>,
    /// Model used for this session (e.g. `openrouter:anthropic/claude-sonnet-4`).
    pub model: Option<String>,
    /// Content of the first user message (for previews).
    pub first_message: Option<String>,
    /// Content of the most recent user/assistant message (for previews).
    pub last_message: Option<String>,
    /// Current working directory, restored when the session resumes.
    pub working_dir: Option<String>,
    /// Accumulated session cost as a decimal string (e.g. `"0.0123"`).
    pub total_cost: Option<String>,
    /// JSON-serialized `TokenUsage` from the last turn, for restoring the
    /// status-bar usage display on resume.
    pub last_usage: Option<String>,
    /// Cumulative session totals, including input and output tokens.
    pub total_usage: Option<SessionUsage>,
}
