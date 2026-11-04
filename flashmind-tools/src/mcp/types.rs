use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, RwLock};

use flashmind_types::ContentPart;
use flashmind_types::tool::ToolResult;
use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};
use serde::{Deserialize, Serialize};

/// The largest file an MCP tool result hands the model.
pub const MAX_BLOB_BYTES: usize = 512 * 1024;

/// Tool definition received from an MCP server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Result of calling a tool on an MCP server.
pub struct McpToolCallResult {
    /// Content items returned by the tool (text, images, etc.).
    pub content: Vec<McpContent>,
    /// Whether the MCP server flagged this result as an error.
    pub is_error: bool,
}

impl McpToolCallResult {
    /// The result the model gets: the text, with files as attachments. A file
    /// the model cannot take fails the call with a one-line reason, so its
    /// data never enters the history.
    pub fn into_tool_result(self, tool_call_id: &str) -> ToolResult {
        let mut text = Vec::new();
        let mut parts = Vec::new();
        for content in self.content {
            match content {
                McpContent::Blob(blob) => match blob.into_part() {
                    Ok(part) => parts.push(part),
                    Err(reason) => return ToolResult::failure(tool_call_id, reason),
                },
                McpContent::Text(s) | McpContent::Other(s) => text.push(s),
            }
        }
        let output = text.join("\n");
        if self.is_error {
            ToolResult::failure(tool_call_id, output)
        } else {
            ToolResult::success(tool_call_id, output).with_parts(parts)
        }
    }
}

/// A content item from an MCP tool call result.
///
/// Text and text resources are `Text`; images, audio and binary resources
/// are `Blob`; anything else is `Other` with a debug string.
pub enum McpContent {
    /// Plain text content.
    Text(String),
    /// A file: an image, audio, or a binary resource.
    Blob(McpBlob),
    /// Other non-text content (debug representation of the original).
    Other(String),
}

impl McpContent {
    /// The content as text the model can read; `None` for a file.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text(s) | McpContent::Other(s) => Some(s),
            McpContent::Blob(_) => None,
        }
    }
}

/// A file an MCP tool returned, still base64 encoded.
pub struct McpBlob {
    /// The resource URI, when the server gave one.
    pub uri: Option<String>,
    pub mime_type: Option<String>,
    /// The file, base64 encoded.
    pub data: String,
}

impl McpBlob {
    /// The file's size once decoded.
    pub fn size(&self) -> usize {
        let padding = self.data.bytes().rev().take_while(|b| *b == b'=').count();
        (self.data.len() / 4 * 3).saturating_sub(padding)
    }

    /// The name to call it by: the last segment of its URI.
    pub fn name(&self) -> &str {
        self.uri
            .as_deref()
            .and_then(|uri| uri.rsplit(['/', ':']).find(|segment| !segment.is_empty()))
            .unwrap_or("The file")
    }

    /// The file as a part the model can take, or why it cannot.
    pub fn into_part(self) -> Result<ContentPart, String> {
        let name = self.name().to_owned();
        let size = self.size();
        if size > MAX_BLOB_BYTES {
            return Err(format!(
                "{name} is {} KB, over the {} KB the model can take; save it to a file and \
                 read it with a tool instead",
                size.div_ceil(1024),
                MAX_BLOB_BYTES / 1024
            ));
        }
        let mime = self.mime_type.unwrap_or_default();
        match mime.as_str() {
            "image/png" | "image/jpeg" | "image/gif" | "image/webp" => Ok(ContentPart::Image {
                media_type: mime,
                data: self.data,
            }),
            "application/pdf" => Ok(ContentPart::Document {
                media_type: mime,
                filename: name,
                data: self.data,
            }),
            "" => Err(format!(
                "{name} is of an unknown type, which the model cannot read; save it to a file \
                 and read it with a tool instead"
            )),
            _ => Err(format!(
                "{name} is {mime}, which the model cannot read; save it to a file and read it \
                 with a tool instead"
            )),
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

/// Files keep their bytes out of any text, so the model never reads base64.
fn content_from_rmcp(content: ContentBlock) -> McpContent {
    let blob = |uri, mime_type, data| {
        McpContent::Blob(McpBlob {
            uri,
            mime_type,
            data,
        })
    };
    match content {
        ContentBlock::Text(text) => McpContent::Text(text.text),
        ContentBlock::Image(image) => blob(None, Some(image.mime_type), image.data),
        ContentBlock::Audio(audio) => blob(None, Some(audio.mime_type), audio.data),
        ContentBlock::Resource(resource) => match resource.resource {
            ResourceContents::TextResourceContents { text, .. } => McpContent::Text(text),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob: data,
                ..
            } => blob(Some(uri), mime_type, data),
            other => McpContent::Other(format!("{other:?}")),
        },
        other => McpContent::Other(format!("{other:?}")),
    }
}

pub(crate) fn call_result_from_rmcp(r: CallToolResult) -> McpToolCallResult {
    McpToolCallResult {
        content: r.content.into_iter().map(content_from_rmcp).collect(),
        is_error: r.is_error.unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pdf(bytes: usize) -> McpToolCallResult {
        let block = ContentBlock::resource(ResourceContents::BlobResourceContents {
            uri: "fastmail:blob/Statement.pdf".into(),
            mime_type: Some("application/pdf".into()),
            blob: "QUJD".repeat(bytes / 3),
            meta: None,
        });
        McpToolCallResult {
            content: vec![content_from_rmcp(block)],
            is_error: false,
        }
    }

    fn text(result: &ToolResult) -> String {
        result
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_small_pdf_is_an_attachment_not_text() {
        let result = pdf(3000).into_tool_result("call");
        assert!(result.is_success());
        assert!(!text(&result).contains("QUJD"));
        assert!(matches!(
            result.content().last(),
            Some(ContentPart::Document { filename, .. }) if filename == "Statement.pdf"
        ));
    }

    #[test]
    fn a_large_pdf_fails_the_call_with_its_size() {
        let result = pdf(916 * 1024).into_tool_result("call");
        assert!(!result.is_success());
        assert_eq!(
            text(&result),
            "Statement.pdf is 916 KB, over the 512 KB the model can take; save it to a \
             file and read it with a tool instead"
        );
    }

    #[test]
    fn an_unsupported_type_fails_the_call() {
        let block = ContentBlock::resource(ResourceContents::BlobResourceContents {
            uri: "x:report.zip".into(),
            mime_type: Some("application/zip".into()),
            blob: "QUJD".into(),
            meta: None,
        });
        let result = McpToolCallResult {
            content: vec![content_from_rmcp(block)],
            is_error: false,
        }
        .into_tool_result("call");
        assert!(!result.is_success());
        assert!(text(&result).starts_with("report.zip is application/zip, which the model"));
    }
}
