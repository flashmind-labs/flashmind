//! Type conversion and request building for Ollama's native format.
//!
//! <https://github.com/ollama/ollama/blob/main/docs/api.md>
//!
//! Handles conversion from internal types ([`Message`], [`ToolDefinition`],
//! [`CompletionRequest`]) to Ollama wire types.

use flashmind_types::message::{ContentPart, Message};
use flashmind_types::model::ReasoningLevel;
use flashmind_types::{CompletionRequest, ToolDefinition};

use super::wire_types::*;

// ============================================================================
// Type Conversions
// ============================================================================

impl From<&Message> for NativeMessage {
    fn from(msg: &Message) -> Self {
        let tool_calls = msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| NativeToolCall {
                    function: NativeFunction {
                        name: tc.name.clone(),
                        // Ollama requires arguments to be a JSON object.
                        // Guard against malformed values (strings, arrays, null).
                        arguments: ensure_object(&tc.arguments),
                    },
                })
                .collect()
        });

        let (content, images) = if let Some(ref parts) = msg.parts {
            let mut text_parts: Vec<String> = Vec::new();
            let mut image_data = Vec::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => text_parts.push(text.clone()),
                    ContentPart::Image { data, .. } => image_data.push(data.clone()),
                    ContentPart::Document {
                        filename,
                        media_type,
                        ..
                    } => {
                        text_parts.push(format!("[Document: {} ({})]", filename, media_type));
                    }
                    ContentPart::Video {
                        filename,
                        media_type,
                        ..
                    } => {
                        text_parts.push(format!("[Video: {} ({})]", filename, media_type));
                    }
                    ContentPart::Audio {
                        filename,
                        media_type,
                        ..
                    } => {
                        text_parts.push(format!("[Audio: {} ({})]", filename, media_type));
                    }
                }
            }
            // Combine msg.content with any text parts from attachments.
            // msg.content carries timestamp/context that must not be dropped.
            let content = if text_parts.is_empty() {
                msg.content.clone()
            } else if msg.content.is_empty() {
                text_parts.join("\n")
            } else {
                let mut combined = msg.content.clone();
                combined.push('\n');
                combined.push_str(&text_parts.join("\n"));
                combined
            };
            let images = if image_data.is_empty() {
                None
            } else {
                Some(image_data)
            };
            (content, images)
        } else {
            (msg.content.clone(), None)
        };

        let (role, content) = match msg.role {
            flashmind_types::message::Role::Developer => {
                ("user".to_string(), format!("<system>\n{content}\n</system>"))
            }
            other => (other.to_string(), content),
        };

        Self {
            role,
            content,
            tool_calls,
            images,
        }
    }
}

/// Ensure a JSON value is an object. Ollama requires tool call arguments
/// to be `map[string]interface{}` — arrays, strings, and null are invalid.
pub(super) fn ensure_object(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(_) => value.clone(),
        serde_json::Value::String(s) => {
            // Try parsing stringified JSON back to an object
            serde_json::from_str(s)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()))
        }
        serde_json::Value::Array(arr) => {
            // Some models wrap arguments in a single-element array — unwrap it
            if arr.len() == 1
                && let Some(obj @ serde_json::Value::Object(_)) = arr.first()
            {
                return obj.clone();
            }
            tracing::debug!("Coercing non-object tool arguments to empty object: {value}");
            serde_json::Value::Object(Default::default())
        }
        other => {
            tracing::debug!("Coercing non-object tool arguments to empty object: {other}");
            serde_json::Value::Object(Default::default())
        }
    }
}

pub(super) fn convert_tools(tools: Vec<ToolDefinition>) -> Option<Vec<NativeTool>> {
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .into_iter()
            .map(|t| NativeTool {
                tool_type: "function".into(),
                function: NativeToolFunction {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
            })
            .collect(),
    )
}

// ============================================================================
// Request Building
// ============================================================================

pub(super) fn build_native_request(
    request: CompletionRequest,
    num_ctx: Option<u32>,
) -> NativeRequest {
    let think = match request.reasoning {
        ReasoningLevel::On => true,
        ReasoningLevel::Off => false,
    };

    NativeRequest {
        model: request.model.name().to_string(),
        messages: request.messages.iter().map(NativeMessage::from).collect(),
        stream: true,
        tools: convert_tools(request.tools),
        think,
        options: NativeOptions {
            temperature: request.temperature,
            num_predict: request.max_tokens,
            num_ctx,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            min_p: request.sampling.min_p,
            presence_penalty: request.sampling.presence_penalty,
            repetition_penalty: request.sampling.repetition_penalty,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ToolCall;
    use flashmind_types::model::{AliasedModel, Model, Provider};
    use std::str::FromStr;

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::user("Hello");
        let native_msg = NativeMessage::from(&msg);
        assert_eq!(native_msg.role, "user");
        assert_eq!(native_msg.content, "Hello");
        assert!(native_msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_conversion_system() {
        let msg = Message::system("You are helpful");
        let native_msg = NativeMessage::from(&msg);
        assert_eq!(native_msg.role, "system");
        assert_eq!(native_msg.content, "You are helpful");
    }

    #[test]
    fn test_message_conversion_assistant() {
        let msg = Message::assistant("Sure, I can help");
        let native_msg = NativeMessage::from(&msg);
        assert_eq!(native_msg.role, "assistant");
        assert_eq!(native_msg.content, "Sure, I can help");
    }

    #[test]
    fn test_message_conversion_tool_result() {
        let msg = Message::tool_result("call-1", "file contents here");
        let native_msg = NativeMessage::from(&msg);
        assert_eq!(native_msg.role, "tool");
        assert_eq!(native_msg.content, "file contents here");
    }

    #[test]
    fn test_message_conversion_with_tool_calls() {
        let msg = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "tc-1".into(),
                name: "file_read".into(),
                arguments: serde_json::json!({"path": "/tmp/test"}),
            }],
        );
        let native_msg = NativeMessage::from(&msg);
        assert_eq!(native_msg.role, "assistant");
        let calls = native_msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "file_read");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({"path": "/tmp/test"})
        );
    }

    #[test]
    fn test_tool_conversion() {
        let tools = vec![
            ToolDefinition {
                name: "file_read".into(),
                description: "Read a file".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }),
            },
            ToolDefinition {
                name: "bash".into(),
                description: "Run a command".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    }
                }),
            },
        ];

        let native_tools = convert_tools(tools).unwrap();
        assert_eq!(native_tools.len(), 2);
        assert_eq!(native_tools[0].tool_type, "function");
        assert_eq!(native_tools[0].function.name, "file_read");
        assert_eq!(native_tools[1].function.name, "bash");
    }

    #[test]
    fn test_tool_conversion_empty() {
        let result = convert_tools(vec![]);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_native_request_reasoning_on() {
        let request = CompletionRequest {
            model: Model {
                provider: Provider::Ollama,
                model: AliasedModel {
                    name: "qwq".into(),
                    real_name: None,
                },
            },
            messages: vec![Message::user("Think about this")],
            tools: vec![],
            temperature: rust_decimal::Decimal::from_str("0.5").unwrap(),
            max_tokens: Some(8192),
            reasoning: ReasoningLevel::On,
            sampling: Default::default(),
        };

        let native = build_native_request(request, None);
        assert!(native.think);
        assert_eq!(native.options.num_predict, Some(8192));
        assert!(native.options.num_ctx.is_none());
    }

    #[test]
    fn test_build_native_request_reasoning_off() {
        let request = CompletionRequest {
            model: Model {
                provider: Provider::Ollama,
                model: AliasedModel {
                    name: "llama3.2".into(),
                    real_name: None,
                },
            },
            messages: vec![Message::user("Hello")],
            tools: vec![],
            temperature: rust_decimal::Decimal::from_str("0.7").unwrap(),
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: Default::default(),
        };

        let native = build_native_request(request, None);
        assert!(!native.think);
        assert!(native.options.num_predict.is_none());
    }

    #[test]
    fn test_build_native_request_with_num_ctx() {
        let request = CompletionRequest {
            model: Model {
                provider: Provider::Ollama,
                model: AliasedModel {
                    name: "llama3.2".into(),
                    real_name: None,
                },
            },
            messages: vec![Message::user("Hello")],
            tools: vec![],
            temperature: rust_decimal::Decimal::from_str("0.7").unwrap(),
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: Default::default(),
        };

        let native = build_native_request(request, Some(128000));
        assert_eq!(native.options.num_ctx, Some(128000));
    }

    #[test]
    fn test_message_conversion_multimodal() {
        // Image-only parts: content comes from msg.content
        let msg = Message::user_with_parts(
            "What is this?",
            vec![ContentPart::Image {
                media_type: "image/jpeg".into(),
                data: "base64img".into(),
            }],
        );
        let native = NativeMessage::from(&msg);
        assert_eq!(native.content, "What is this?");
        let images = native.images.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0], "base64img");
    }

    #[test]
    fn test_message_conversion_multimodal_preserves_content() {
        // When parts contain text, msg.content (timestamp/context) must not be dropped
        let msg = Message::user_with_parts(
            "[2026-04-08]\n\ncontext info",
            vec![
                ContentPart::Text {
                    text: "user caption".into(),
                },
                ContentPart::Image {
                    media_type: "image/jpeg".into(),
                    data: "base64img".into(),
                },
            ],
        );
        let native = NativeMessage::from(&msg);
        assert!(
            native.content.contains("[2026-04-08]"),
            "msg.content must be preserved"
        );
        assert!(
            native.content.contains("user caption"),
            "text parts must be included"
        );
        assert!(native.images.unwrap().len() == 1);
    }
}
