//! LLM wire-format types — [`Role`], [`Message`], [`ToolCall`], and [`ContentPart`].
//!
//! These are the types serialized into JSON for API calls and received in responses.
//! They are distinct from the richer intermediate representation in `flashmind_core`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible message role.
///
/// Serialises to lowercase strings (`"system"`, `"user"`, `"assistant"`, `"developer"`, `"tool"`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, strum::Display, strum::EnumString,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Developer,
    Tool,
}

/// A tool call requested by the assistant in its response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Server-assigned ID that must be echoed back in the corresponding tool result.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// JSON object of argument values.
    pub arguments: serde_json::Value,
}

/// Result returned by a tool, attached as a `role: "tool"` message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
    pub success: bool,
}

/// Multimodal content block within a user message.
///
/// Serialised with a `type` discriminant matching the OpenAI `ContentPart` schema:
/// `"text"`, `"image"`, `"document"`, `"video"`, `"audio"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
    #[serde(rename = "document")]
    Document {
        media_type: String,
        filename: String,
        data: String,
    },
    #[serde(rename = "video")]
    Video {
        media_type: String,
        filename: String,
        data: String,
    },
    #[serde(rename = "audio")]
    Audio {
        media_type: String,
        filename: String,
        data: String,
    },
}

/// A single message in the LLM API request / response payload.
///
/// For tool results set `tool_call_id` instead of `content`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<ContentPart>>,
    #[serde(default = "chrono::Utc::now")]
    pub timestamp: DateTime<Utc>,
}

impl Message {
    /// Create a plain-text user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant text message (no tool calls).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant message that includes tool call requests.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a system prompt message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a developer message (system-level instructions using the `developer` role).
    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: Role::Developer,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a tool-result message linked to a prior `ToolCall.id`.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            parts: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a multimodal user message with inline attachments.
    pub fn user_with_parts(content: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            parts: if parts.is_empty() { None } else { Some(parts) },
            timestamp: Utc::now(),
        }
    }

    /// Replace binary content parts (image, document, video, audio) with
    /// text placeholders so the message can be sent to models that don't
    /// support those modalities.
    ///
    /// Example:
    /// - `Image { media_type: "image/jpeg", .. }` → `Text { text: "[Image: image/jpeg]" }`
    pub fn strip_binary_data(&self) -> Self {
        let parts = self.parts.as_ref().map(|parts| {
            parts
                .iter()
                .map(|p| match p {
                    ContentPart::Image { media_type, .. } => ContentPart::Text {
                        text: format!("[Image: {}]", media_type),
                    },
                    ContentPart::Document {
                        media_type,
                        filename,
                        ..
                    } => ContentPart::Text {
                        text: format!("[Document: {} ({})]", filename, media_type),
                    },
                    ContentPart::Video {
                        media_type,
                        filename,
                        ..
                    } => ContentPart::Text {
                        text: format!("[Video: {} ({})]", filename, media_type),
                    },
                    ContentPart::Audio {
                        media_type,
                        filename,
                        ..
                    } => ContentPart::Text {
                        text: format!("[Audio: {} ({})]", filename, media_type),
                    },
                    other => other.clone(),
                })
                .collect()
        });

        Self {
            parts,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_message_constructors() {
        let user = Message::user("hi");
        assert_eq!(user.role, Role::User);
        assert_eq!(user.content, "hi");

        let assistant = Message::assistant("hello");
        assert_eq!(assistant.role, Role::Assistant);

        let system = Message::system("you are helpful");
        assert_eq!(system.role, Role::System);

        let developer = Message::developer("injected context");
        assert_eq!(developer.role, Role::Developer);
        assert_eq!(developer.content, "injected context");

        let tool = Message::tool_result("call-1", "result data");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn test_content_part_serde() {
        let text = ContentPart::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        let roundtrip: ContentPart = serde_json::from_str(&json).unwrap();
        match roundtrip {
            ContentPart::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text"),
        }

        let img = ContentPart::Image {
            media_type: "image/jpeg".into(),
            data: "abc123".into(),
        };
        let json = serde_json::to_string(&img).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn test_strip_binary_data() {
        let msg = Message::user_with_parts(
            "photo",
            vec![
                ContentPart::Text {
                    text: "a cat".into(),
                },
                ContentPart::Image {
                    media_type: "image/jpeg".into(),
                    data: "bigdata".into(),
                },
            ],
        );
        let stripped = msg.strip_binary_data();
        let parts = stripped.parts.unwrap();
        assert_eq!(parts.len(), 2);
        match &parts[1] {
            ContentPart::Text { text } => assert!(text.contains("image/jpeg")),
            _ => panic!("expected text placeholder for image"),
        }
    }

    #[test]
    fn test_role_serde() {
        let role = Role::Assistant;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"assistant\"");

        let parsed: Role = serde_json::from_str("\"tool\"").unwrap();
        assert_eq!(parsed, Role::Tool);
    }
}
