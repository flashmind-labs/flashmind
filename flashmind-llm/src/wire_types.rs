//! Shared OpenAI-compatible wire types used by multiple providers.
//!
//! Defines request/response structures for the OpenAI `/v1/chat/completions` protocol,
//! including [`ApiMessage`], [`StreamChunk`], and helper functions for converting
//! between Flashmind domain types (`flashmind_types::message::Message`) and wire format.
//!
//! Provider-specific extensions (reasoning parameters, Anthropic-native fields) live
//! in their respective modules; this crate only contains the shared base types.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`ApiMessage`] | Request message in OpenAI wire format |
//! | [`ApiContent`] / [`ApiContentPart`] | Text or multimodal content (image, file, video, audio) |
//! | [`ApiToolCall`] / [`ApiToolFunction`] | Tool invocation and definition in wire format |
//! | [`StreamChunk`] | SSE chunk from streaming responses |
//! | [`StreamDelta`] | Incremental content delta within a stream choice |
//! | [`StreamImage`] | Generated image in streaming response (base64 data URL) |
//! | [`ApiSamplingParams`] | Flattened sampling params for request bodies |
//!
//! # Conversion
//!
//! - [`to_api_messages`] — converts `&[Message]` to `Vec<ApiMessage>`
//! - [`to_api_tools`] — converts `Vec<ToolDefinition>` to `Vec<ApiTool>`
//! - [`From<&Message>`] on [`ApiMessage`] — per-message conversion

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use flashmind_types::ToolDefinition;
use flashmind_types::message::{ContentPart, Message};

// ============================================================================
// Request Wire Types
// ============================================================================

/// A single message in OpenAI-compatible wire format.
///
/// Corresponds to the `messages` array in the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages).
#[derive(Serialize)]
pub struct ApiMessage {
    /// Message role (`"system"`, `"user"`, `"assistant"`, `"tool"`, `"developer"`).
    pub role: String,
    /// Text or multimodal content parts.
    pub content: ApiContent,
    /// Tool calls requested by the assistant (only present on `role = "assistant"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    /// ID of the tool call this is a result for (only present on `role = "tool"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content in OpenAI-compatible wire format — either a plain string or an array of parts.
/// Serialises as a scalar string for simple text, or a JSON array for multimodal content.
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
/// Used in the `image_url` content part format. See [OpenAI vision docs](https://platform.openai.com/docs/guides/vision).
#[derive(Serialize)]
pub struct ApiImageUrl {
    /// Data URL (`data:image/...;base64,...`) or remote URL.
    pub url: String,
}

/// An inline file attachment (document) for OpenAI-compatible models.
#[derive(Serialize)]
pub struct ApiFile {
    /// File name (e.g. `"report.pdf"`).
    pub filename: String,
    /// Base64-encoded file data as a data URL.
    pub file_data: String,
}

/// A video URL reference for OpenAI-compatible vision models.
#[derive(Serialize)]
pub struct ApiVideoUrl {
    /// Remote URL or data URL of the video.
    pub url: String,
}

/// An inline audio input for OpenAI-compatible speech models.
#[derive(Serialize)]
pub struct ApiInputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    /// Audio format (e.g. `"wav"`, `"mp3"`, `"opus"`).
    pub format: String,
}

/// Audio output configuration for OpenAI-compatible requests.
/// Corresponds to the `audio` field in the [OpenAI audio output API](https://platform.openai.com/docs/guides/voice-generation).
#[derive(Serialize)]
pub struct ApiAudioConfig {
    /// Voice preset ID (e.g. `"alloy"`, `"echo"`).
    pub voice: String,
    /// Output format (e.g. `"mp3"`, `"wav"`).
    pub format: String,
}

/// Image generation configuration for OpenAI-compatible requests.
/// Used with image-generating models (DALL-E, etc.).
#[derive(Serialize)]
pub struct ApiImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Reference images for super-resolution guidance.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub super_resolution_references: Vec<String>,
}

/// A tool call from the assistant in OpenAI-compatible wire format.
/// Corresponds to the `tool_calls` array in [OpenAI assistant messages](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages).
#[derive(Serialize, Deserialize, Clone)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ApiFunctionCall,
}

/// Function call inside an [`ApiToolCall`] — name and JSON string of arguments.
/// Note: `arguments` is a raw JSON *string* (not a parsed object), per the OpenAI spec.
#[derive(Serialize, Deserialize, Clone)]
pub struct ApiFunctionCall {
    /// Tool name to invoke.
    pub name: String,
    /// Arguments as a JSON-encoded string (e.g. `"{\"path\":\"Cargo.toml\"}"`).
    pub arguments: String,
}

/// A tool definition sent to the provider in OpenAI-compatible wire format.
/// Wraps a [`ApiToolFunction`] with `type = "function"`.
#[derive(Serialize)]
pub struct ApiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ApiToolFunction,
}

/// Function definition inside an [`ApiTool`] — name, description, and JSON schema parameters.
/// Maps directly to the `functions` field in the [OpenAI tools parameter](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools).
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
/// Appears in the final SSE chunk when `stream_options.include_usage = true`.
#[derive(Deserialize)]
pub struct ApiUsage {
    /// Tokens consumed by the prompt (input).
    pub prompt_tokens: u32,
    /// Tokens generated by the model (output).
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
    /// Breakdown of prompt token sources (OpenRouter / OpenAI).
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Prompt token breakdown returned by OpenRouter and OpenAI.
#[derive(Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens served from the provider's prompt cache.
    #[serde(default)]
    pub cached_tokens: u32,
}

/// An SSE chunk from an OpenAI-compatible streaming response.
///
/// Each line prefixed with `data:` contains a JSON-encoded instance of this struct.
/// The stream terminates with `data: [DONE]` (signalled as `None` from the SSE parser).
#[derive(Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<StreamChoice>,
    /// Present only in the final chunk (when `include_usage` is requested).
    pub usage: Option<ApiUsage>,
    /// Top-level generated images (some providers emit them here instead of in the delta).
    #[serde(default)]
    pub images: Option<Vec<StreamImage>>,
}

/// A single choice in a streaming SSE chunk.
/// Most responses have exactly one choice (index 0).
#[derive(Deserialize)]
pub struct StreamChoice {
    pub delta: StreamDelta,
    /// Set on the final chunk to indicate why generation stopped.
    pub finish_reason: Option<String>,
}

/// Incremental content delta within a [`StreamChoice`].
///
/// Each field is optional — different chunks may carry content, tool calls,
/// reasoning tokens, or audio. The model emits partial deltas that must be
/// accumulated across chunks to reconstruct the full response.
#[derive(Deserialize)]
pub struct StreamDelta {
    /// Text content delta. Accumulate across chunks to form the full message.
    pub content: Option<String>,
    /// Reasoning/thinking content from models with extended thinking.
    /// Present in OpenRouter responses, absent in Ollama.
    #[serde(default)]
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
    #[serde(default)]
    pub audio: Option<StreamAudioDelta>,
    /// Generated images (OpenAI image models return these inside the delta).
    #[serde(default)]
    pub images: Option<Vec<StreamImage>>,
}

/// A tool-call delta within a [`StreamDelta`] — fields arrive piecemeal.
///
/// In the OpenAI streaming protocol, `id`, `name`, and `arguments` may come in
/// separate chunks. Use [`crate::sse::ToolCallTracker`] to accumulate them.
#[derive(Deserialize)]
pub struct StreamToolCallDelta {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub function: Option<StreamFunctionDelta>,
}

/// Function name/arguments delta within a [`StreamToolCallDelta`].
#[derive(Deserialize)]
pub struct StreamFunctionDelta {
    /// Tool name — usually arrives in the first delta for this call.
    pub name: Option<String>,
    /// Arguments JSON string — arrives incrementally across multiple deltas.
    pub arguments: Option<String>,
}

/// Audio output delta in a streaming response.
/// Base64-encoded audio chunk for real-time playback.
#[derive(Deserialize)]
pub struct StreamAudioDelta {
    /// Base64-encoded audio data.
    pub data: Option<String>,
    /// Audio format (e.g. `"pcm16"`, `"mp3"`).
    pub format: Option<String>,
}

/// A generated image in a streaming response (base64 data URL).
///
/// Supports both `{"url": "data:..."}` (top-level `images`) and
/// `{"type": "image_url", "image_url": {"url": "data:..."}}` (delta `images`).
#[derive(Deserialize)]
pub struct StreamImage {
    /// Direct URL (top-level images field).
    #[serde(default)]
    pub url: Option<String>,
    /// Nested URL (delta images field from OpenAI models).
    #[serde(default)]
    pub image_url: Option<StreamImageUrl>,
}

/// Nested URL wrapper inside a [`StreamImage`] (delta images field from OpenAI models).
#[derive(Deserialize)]
pub struct StreamImageUrl {
    pub url: String,
}

impl StreamImage {
    /// Extract the data URL, checking both the direct `url` field and the nested
    /// `image_url.url` variant. Returns the first non-empty value found.
    pub fn data_url(&self) -> Option<&str> {
        self.url
            .as_deref()
            .or(self.image_url.as_ref().map(|u| u.url.as_str()))
    }
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
                ContentPart::ImageUrl { url } => ApiContentPart::ImageUrl {
                    image_url: ApiImageUrl { url: url.clone() },
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
                ContentPart::VideoUrl { url } => ApiContentPart::VideoUrl {
                    video_url: ApiVideoUrl { url: url.clone() },
                },
                ContentPart::Audio {
                    media_type, data, ..
                } => {
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
///
/// Wraps each [`ToolDefinition`] in an [`ApiTool`] with `type = "function"`,
/// which is the format expected by the [OpenAI tools parameter](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools).
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
///
/// Converts each [`Message`] to an [`ApiMessage`], handling multimodal content
/// parts and tool calls. The user's text content is prepended as a `Text` part
/// when attachments are present, so the LLM sees both the prompt and the media.
pub fn to_api_messages(messages: &[Message]) -> Vec<ApiMessage> {
    messages.iter().map(ApiMessage::from).collect()
}

/// Request wire type for OpenAI-compatible APIs.
///
/// Carries sampling params flattened into the top-level JSON via `#[serde(flatten)]`.
/// All fields are optional so they can be omitted for models that reject them
/// (e.g. OpenAI image generation models reject `temperature`).
#[derive(Serialize, Default)]
pub struct ApiSamplingParams {
    /// Temperature for sampling (0.0–2.0). Lower = more deterministic.
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::arbitrary_precision_option"
    )]
    pub temperature: Option<Decimal>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold (0.0–1.0).
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::arbitrary_precision_option"
    )]
    pub top_p: Option<Decimal>,
    /// Hard cutoff: sample from K most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Minimum token probability filter (0.0–1.0).
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::arbitrary_precision_option"
    )]
    pub min_p: Option<Decimal>,
    /// Additive penalty for tokens already in output (-2.0 to 2.0).
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::arbitrary_precision_option"
    )]
    pub presence_penalty: Option<Decimal>,
    /// Multiplicative penalty on repeated tokens (1.0–2.0).
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::arbitrary_precision_option"
    )]
    pub repetition_penalty: Option<Decimal>,
}

/// Chat template kwargs for models that support thinking mode via the API.
/// Used by OpenRouter to enable extended reasoning in compatible models.
///
/// Corresponds to the `chat_template_kwargs` field in the [OpenRouter API](https://openrouter.ai/docs/requests).
#[derive(Serialize)]
pub struct ChatTemplateKwargs {
    /// Whether to enable thinking/reasoning mode.
    pub enable_thinking: bool,
}

/// Streaming options for OpenAI-compatible requests — always requests usage in the final chunk.
///
/// Corresponds to the `stream_options` field in the [OpenAI streaming API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream-options).
#[derive(Serialize)]
pub struct StreamOptions {
    /// If `true`, the provider includes token usage in the final SSE chunk.
    pub include_usage: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
        }
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
