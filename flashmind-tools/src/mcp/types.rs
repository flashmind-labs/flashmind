use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, RwLock};

use rmcp::model::CallToolResult;
use serde::{Deserialize, Serialize};

/// Tool definition received from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

impl PartialEq for McpToolDef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.description == other.description
            && self.input_schema == other.input_schema
    }
}

/// Result of calling a tool on an MCP server.
pub struct McpToolCallResult {
    /// Content items returned by the tool (text, images, etc.).
    pub content: Vec<McpContent>,
    /// Whether the MCP server flagged this result as an error.
    pub is_error: bool,
}

/// A content item from an MCP tool call result.
///
/// MCP servers can return multiple content blocks per tool call. Most tools
/// return plain text, but arbitrary content types are represented as `Other`
/// with a debug string.
pub enum McpContent {
    /// Plain text content.
    Text(String),
    /// Non-text content (debug representation of the original).
    Other(String),
}

impl McpContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text(s) | McpContent::Other(s) => Some(s),
        }
    }
}

/// Queued operation for the caller to apply to its `ToolRegistry`.
///
/// The MCP registry queues these operations when servers connect or disconnect.
/// The caller (agent loop) should drain them each turn via
/// [`McpRegistry::drain_pending_ops`](super::McpRegistry::drain_pending_ops)
/// and register/unregister wrapper tools.
pub enum McpToolOp {
    /// A server connected and its tools should be registered.
    Register {
        server_name: String,
        tool_defs: Vec<McpToolDef>,
        /// Tool names that require approval before execution.
        restricted: Arc<HashSet<String>>,
        /// Tools approved for the current session (shared across all wrappers
        /// for this server).
        session_approved: Arc<RwLock<HashSet<String>>>,
    },
    /// A server was removed and its tools should be unregistered.
    Unregister { server_name: String },
}

#[derive(Debug)]
pub struct McpAuthRequired {
    pub server: String,
}

impl fmt::Display for McpAuthRequired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MCP server '{}' requires authentication", self.server)
    }
}

impl std::error::Error for McpAuthRequired {}

pub(crate) fn tool_def_from_rmcp(t: &rmcp::model::Tool) -> McpToolDef {
    McpToolDef {
        name: t.name.to_string(),
        description: t.description.as_ref().map(|d| d.to_string()),
        input_schema: serde_json::to_value(&*t.input_schema).unwrap_or_default(),
    }
}

pub(crate) fn call_result_from_rmcp(r: CallToolResult) -> McpToolCallResult {
    let content = r
        .content
        .into_iter()
        .map(|c| match c.as_text() {
            Some(text) => McpContent::Text(text.text.clone()),
            None => McpContent::Other(format!("{c:?}")),
        })
        .collect();
    McpToolCallResult {
        content,
        is_error: r.is_error.unwrap_or(false),
    }
}
