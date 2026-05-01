//! Shared OpenAI-compatible wire types used by multiple providers.
//!
//! Defines request/response structures for the OpenAI `/v1/chat/completions` protocol,
//! including [`ApiMessage`], [`StreamChunk`], and helper functions for converting
//! between Flashmind domain types (`flashmind_types::message::Message`) and wire format.
//!
//! Provider-specific extensions (reasoning parameters, Anthropic-native fields) live
//! in their respective modules; this crate only contains the shared base types.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize, Serializer};

use flashmind_types::ToolDefinition;
use flashmind_types::message::{ContentPart, Message};

// ============================================================================
// Request Wire Types
// ============================================================================

/// A single message in OpenAI-compatible wire format.
#[derive(Serialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: ApiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content in OpenAI-compatible wire format — either a plain string or an array of parts.
#[derive(Serialize)]
#[serde(untagged)]
pub enum ApiContent {
    Text(String),
    Parts(Vec<ApiContentPart>),
}

/// A multimodal content part in OpenAI-compatible wire format (text, image, file, video, audio).
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ApiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ApiImageUrl },
    #[serde(rename = "file")]
    File { file: ApiFile },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: ApiVideoUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: ApiInputAudio },
}

/// An inline image URL for OpenAI-compatible vision models.
#[derive(Serialize)]
pub struct ApiImageUrl {
    pub url: String,
}

/// An inline file attachment (document) for OpenAI-compatible models.
#[derive(Serialize)]
pub struct ApiFile {
    pub filename: String,
    pub file_data: String,
}

/// A video URL reference for OpenAI-compatible vision models.
#[derive(Serialize)]
pub struct ApiVideoUrl {
    pub url: String,
}

/// An inline audio input for OpenAI-compatible speech models.
#[derive(Serialize)]
pub struct ApiInputAudio {
    pub data: String,
    pub format: String,
}

/// A tool call from the assistant in OpenAI-compatible wire format.
#[derive(Serialize, Deserialize, Clone)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ApiFunctionCall,
}

/// Function call inside an [`ApiToolCall`] — name and JSON string of arguments.
#[derive(Serialize, Deserialize, Clone)]
pub struct ApiFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A tool definition sent to the provider in OpenAI-compatible wire format.
#[derive(Serialize)]
pub struct ApiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ApiToolFunction,
}

/// Function definition inside an [`ApiTool`] — name, description, and JSON schema parameters.
#[derive(Serialize)]
pub struct ApiToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ============================================================================
// Response / Streaming Wire Types
// ============================================================================

/// Token usage from an OpenAI-compatible API response.
#[derive(Deserialize)]
pub struct ApiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// An SSE chunk from an OpenAI-compatible streaming response.
#[derive(Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<StreamChoice>,
    pub usage: Option<ApiUsage>,
}

/// A single choice in a streaming SSE chunk.
#[derive(Deserialize)]
pub struct StreamChoice {
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

/// Incremental content delta within a [`StreamChoice`].
#[derive(Deserialize)]
pub struct StreamDelta {
    pub content: Option<String>,
    /// Reasoning/thinking content from models with extended thinking.
    /// Present in OpenRouter responses, absent in Ollama.
    #[serde(default)]
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
}

/// A tool-call delta within a [`StreamDelta`] — fields arrive piecemeal.
#[derive(Deserialize)]
pub struct StreamToolCallDelta {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub function: Option<StreamFunctionDelta>,
}

/// Function name/arguments delta within a [`StreamToolCallDelta`].
#[derive(Deserialize)]
pub struct StreamFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ============================================================================
// Type Conversions
// ============================================================================

impl From<&Message> for ApiMessage {
    fn from(msg: &Message) -> Self {
        let tool_calls = msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| ApiToolCall {
                    id: tc.id.clone(),
                    call_type: "function".into(),
                    function: ApiFunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect()
        });

        let content = if let Some(ref parts) = msg.parts {
            // Prepend the text content as a Text part so the LLM sees both
            // the user's prompt and the multimodal attachments.
            let mut api_parts = Vec::with_capacity(parts.len() + 1);
            if !msg.content.is_empty() {
                api_parts.push(ApiContentPart::Text {
                    text: msg.content.clone(),
                });
            }
            api_parts.extend(parts.iter().map(|p| match p {
                ContentPart::Text { text } => ApiContentPart::Text { text: text.clone() },
                ContentPart::Image { media_type, data } => ApiContentPart::ImageUrl {
                    image_url: ApiImageUrl {
                        url: format!("data:{};base64,{}", media_type, data),
                    },
                },
                ContentPart::Document {
                    media_type,
                    filename,
                    data,
                } => ApiContentPart::File {
                    file: ApiFile {
                        filename: filename.clone(),
                        file_data: format!("data:{};base64,{}", media_type, data),
                    },
                },
                ContentPart::Video {
                    media_type, data, ..
                } => ApiContentPart::VideoUrl {
                    video_url: ApiVideoUrl {
                        url: format!("data:{};base64,{}", media_type, data),
                    },
                },
                ContentPart::Audio {
                    media_type, data, ..
                } => {
                    // Extract format from MIME type (e.g. "audio/ogg" → "ogg")
                    let format = media_type
                        .split('/')
                        .next_back()
                        .unwrap_or("wav")
                        .to_string();
                    ApiContentPart::InputAudio {
                        input_audio: ApiInputAudio {
                            data: data.clone(),
                            format,
                        },
                    }
                }
            }));
            ApiContent::Parts(api_parts)
        } else {
            ApiContent::Text(msg.content.clone())
        };

        Self {
            role: msg.role.to_string(),
            content,
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

/// Convert tool definitions to API tool format.
pub fn to_api_tools(tools: Vec<ToolDefinition>) -> Vec<ApiTool> {
    tools
        .into_iter()
        .map(|t| ApiTool {
            tool_type: "function".into(),
            function: ApiToolFunction {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
            },
        })
        .collect()
}

/// Build API messages from internal messages.
pub fn to_api_messages(messages: &[Message]) -> Vec<ApiMessage> {
    messages.iter().map(ApiMessage::from).collect()
}

// ============================================================================
// Shared Request Base
// ============================================================================

/// Fields shared by all OpenAI-compatible request bodies.
/// Provider-specific request types embed this via serde flatten.
///
/// Sampling parameters beyond temperature are included as top-level fields.
/// OpenAI-compatible APIs accept or silently ignore unknown fields, so we send
/// all of them (top_p, top_k, min_p, presence_penalty, repetition_penalty)
/// regardless of provider — the API picks up what it supports.
#[derive(Serialize)]
pub struct ApiRequestBase {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ApiTool>,
    /// Tool choice mode - "auto" lets the model decide whether to use tools.
    /// Required by some models to continue after tool results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(with = "rust_decimal::serde::float")]
    pub temperature: Decimal,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub min_p: Option<Decimal>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub presence_penalty: Option<Decimal>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub repetition_penalty: Option<Decimal>,
    pub stream_options: StreamOptions,
    pub stream: bool,
    /// Skip special tokens in output (required for gemma4 with thinking).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    pub chat_template_kwargs: ChatTemplateKwargs,
    pub parallel_tool_calls: bool,
}

/// Chat template kwargs for models that support thinking mode via the API.
#[derive(Serialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: bool,
}

/// Streaming options for OpenAI-compatible requests — always requests usage in the final chunk.
#[derive(Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
        }
    }
}

/// Serialize `Option<Decimal>` as a JSON float (not a string).
fn serialize_optional_decimal<S>(value: &Option<Decimal>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(d) => {
            let f: f64 = (*d).try_into().map_err(serde::ser::Error::custom)?;
            serializer.serialize_f64(f)
        }
        None => serializer.serialize_none(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ToolCall;

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::user("Hello");
        let api_msg = ApiMessage::from(&msg);
        assert_eq!(api_msg.role, "user");
        match &api_msg.content {
            ApiContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn test_message_conversion_tool() {
        let msg = Message::tool_result("call-1", "result");
        let api_msg = ApiMessage::from(&msg);
        assert_eq!(api_msg.role, "tool");
        assert_eq!(api_msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let msg = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "tc-1".into(),
                name: "file_read".into(),
                arguments: serde_json::json!({"path": "/tmp/test"}),
            }],
        );
        let api_msg = ApiMessage::from(&msg);
        assert_eq!(api_msg.role, "assistant");
        match &api_msg.content {
            ApiContent::Text(t) => assert_eq!(t, ""),
            _ => panic!("expected text content"),
        }
        let calls = api_msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "file_read");
    }

    #[test]
    fn test_message_conversion_multimodal() {
        let msg = Message::user_with_parts(
            "describe",
            vec![
                ContentPart::Text {
                    text: "describe".into(),
                },
                ContentPart::Image {
                    media_type: "image/png".into(),
                    data: "abc".into(),
                },
            ],
        );
        let api_msg = ApiMessage::from(&msg);
        match &api_msg.content {
            ApiContent::Parts(parts) => {
                // msg.content produces one Text part, then parts add Text + Image = 3 total
                assert_eq!(parts.len(), 3);
                let json = serde_json::to_value(&api_msg).unwrap();
                let content = &json["content"];
                assert!(content.is_array());
                assert_eq!(content[2]["type"], "image_url");
                assert!(
                    content[2]["image_url"]["url"]
                        .as_str()
                        .unwrap()
                        .starts_with("data:image/png;base64,")
                );
            }
            _ => panic!("expected parts"),
        }
    }
}
