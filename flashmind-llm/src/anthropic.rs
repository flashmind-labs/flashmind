//! Anthropic direct API provider.
//!
//! Uses the Anthropic Messages API with SSE streaming. Supports tool calling,
//! reasoning (extended thinking), and image/document inputs.
//!
//! See <https://docs.anthropic.com/en/api/messages> for the API reference.

use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize, Serializer};
use tokio_stream::StreamExt;

use crate::http::{http_client_builder, send_with_retry, wait_for_rate_limit};
use flashmind_types::message::{ContentPart, Message};
use flashmind_types::model::ReasoningLevel;
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities, ModelInfo,
    ModelPricing, StreamEvent, TokenUsage,
};
use metrics;
use ratelimit::Ratelimiter;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
///
/// Direct integration with Claude models using SSE streaming. Supports tool calling,
/// reasoning (extended thinking), and image/document inputs.
///
/// # Example
///
/// ```rust,ignore
/// let provider = AnthropicProvider::new(api_key, rate_limiter);
/// ```
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    /// Request rate limiter.
    rate_limiter: Arc<Ratelimiter>,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given API key.
    pub fn new(api_key: String, rate_limiter: Arc<Ratelimiter>) -> Self {
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            api_key,
            rate_limiter,
        }
    }
}

// Anthropic API types

/// Request body for the Anthropic Messages API.
///
/// Serialised as the JSON body of `POST /v1/messages`.
/// See <https://docs.anthropic.com/en/api/messages-examples> for examples.
#[derive(Serialize)]
struct AnthropicRequest {
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    model: String,
    /// Maximum output tokens — required by Anthropic (we use context window as default).
    max_tokens: u32,
    /// System prompt (top-level in Anthropic API, not a message).
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    /// Message history.
    messages: Vec<AnthropicMessage>,
    stream: bool,
    /// Tool definitions (Anthropic format).
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    /// Extended thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    /// Temperature for sampling.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    temperature: Option<Decimal>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
}

#[derive(Serialize)]
struct AnthropicMetadata {
    user_id: String,
}

/// Extended thinking configuration for Anthropic models.
///
/// When set, Claude emits reasoning tokens before the final answer.
/// See <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>.
#[derive(Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    thinking_type: String,
    /// Maximum tokens allocated for reasoning output.
    budget_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // Variants used for serialization completeness
enum AnthropicBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "document")]
    Document { source: AnthropicDocumentSource },
}

#[derive(Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

#[derive(Serialize)]
struct AnthropicDocumentSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// SSE Event types
#[derive(Deserialize)]
struct ContentBlockStart {
    index: usize,
    content_block: ContentBlock,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlockDelta {
    index: usize,
    delta: DeltaBlock,
}

#[derive(Deserialize)]
struct DeltaBlock {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}

#[derive(Deserialize)]
struct MessageDelta {
    delta: MessageDeltaInner,
    usage: Option<DeltaUsage>,
}

#[derive(Deserialize)]
struct MessageDeltaInner {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct DeltaUsage {
    output_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct MessageStart {
    message: MessageStartInner,
}

#[derive(Deserialize)]
struct MessageStartInner {
    usage: Option<MessageUsage>,
}

#[derive(Deserialize)]
struct MessageUsage {
    input_tokens: Option<u32>,
    #[allow(dead_code)] // Deserialized but only input_tokens used from message_start
    output_tokens: Option<u32>,
}

fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system = None;
    let mut api_messages = Vec::new();

    for msg in messages {
        match msg.role.to_string().as_str() {
            "system" => {
                system = Some(msg.content.clone());
            }
            "assistant" => {
                if let Some(ref tool_calls) = msg.tool_calls {
                    let mut blocks = Vec::new();
                    if !msg.content.is_empty() {
                        blocks.push(AnthropicBlock::Text {
                            text: msg.content.clone(),
                        });
                    }
                    for tc in tool_calls {
                        blocks.push(AnthropicBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.arguments.clone(),
                        });
                    }
                    api_messages.push(AnthropicMessage {
                        role: "assistant".into(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                } else {
                    api_messages.push(AnthropicMessage {
                        role: "assistant".into(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
            }
            "tool" => {
                // Tool results go as user messages in Anthropic format
                api_messages.push(AnthropicMessage {
                    role: "user".into(),
                    content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: msg.content.clone(),
                    }]),
                });
            }
            "developer" => {
                let content = format!("<system>\n{}\n</system>", msg.content);
                api_messages.push(AnthropicMessage {
                    role: "user".into(),
                    content: AnthropicContent::Text(content),
                });
            }
            _ => {
                // user
                if let Some(ref parts) = msg.parts {
                    let mut blocks: Vec<AnthropicBlock> = Vec::with_capacity(parts.len() + 1);
                    if !msg.content.is_empty() {
                        blocks.push(AnthropicBlock::Text {
                            text: msg.content.clone(),
                        });
                    }
                    blocks.extend(parts.iter().map(|p| match p {
                        ContentPart::Text { text } => AnthropicBlock::Text { text: text.clone() },
                        ContentPart::Image { media_type, data } => AnthropicBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "base64".into(),
                                media_type: media_type.clone(),
                                data: Some(data.clone()),
                                url: None,
                            },
                        },
                        ContentPart::ImageUrl { url } => AnthropicBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "url".into(),
                                media_type: String::new(),
                                data: None,
                                url: Some(url.clone()),
                            },
                        },
                        ContentPart::Document {
                            media_type, data, ..
                        } => AnthropicBlock::Document {
                            source: AnthropicDocumentSource {
                                source_type: "base64".into(),
                                media_type: media_type.clone(),
                                data: data.clone(),
                            },
                        },
                        ContentPart::Video {
                            media_type,
                            filename,
                            ..
                        } => AnthropicBlock::Text {
                            text: format!("[Video: {} ({})]", filename, media_type),
                        },
                        ContentPart::VideoUrl { url } => AnthropicBlock::Text {
                            text: format!("[Video: {}]", url),
                        },
                        ContentPart::Audio {
                            media_type,
                            filename,
                            ..
                        } => AnthropicBlock::Text {
                            text: format!("[Audio: {} ({})]", filename, media_type),
                        },
                    }));

                    api_messages.push(AnthropicMessage {
                        role: "user".into(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                } else {
                    api_messages.push(AnthropicMessage {
                        role: "user".into(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
            }
        }
    }

    (system, api_messages)
}

fn convert_tools(tools: &[flashmind_types::ToolDefinition]) -> Option<Vec<AnthropicTool>> {
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect(),
    )
}

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

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn context_window(&self, _model: &flashmind_types::Model) -> Option<u32> {
        // All Claude models have 200k context window
        Some(200_000)
    }

    async fn capabilities(&self, _model: &flashmind_types::Model) -> ModelCapabilities {
        ModelCapabilities::all()
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let rate_limiter = self.rate_limiter.clone();
        let provider_str = self.provider().to_string();

        Box::pin(stream! {
            let start = std::time::Instant::now();
            metrics::counter!("llm.requests.started").increment(1);
            let (system, messages) = convert_messages(&request.messages);
            let tools = convert_tools(&request.tools);

            let thinking = match request.reasoning {
                ReasoningLevel::Low => Some(AnthropicThinking {
                    thinking_type: "enabled".into(),
                    budget_tokens: 4000,
                }),
                ReasoningLevel::Medium => Some(AnthropicThinking {
                    thinking_type: "enabled".into(),
                    budget_tokens: 16000,
                }),
                ReasoningLevel::High => Some(AnthropicThinking {
                    thinking_type: "enabled".into(),
                    budget_tokens: 64000,
                }),
                ReasoningLevel::Off => None,
            };

            let default_max_tokens = if thinking.is_some() { 16000 } else { 4096 };

            let api_request = AnthropicRequest {
                model: request.model.name().to_string(),
                max_tokens: request.max_tokens.unwrap_or(default_max_tokens),
                system,
                messages,
                stream: true,
                tools,
                thinking,
                temperature: request.sampling.temperature,
                top_p: request.sampling.top_p,
                top_k: request.sampling.top_k,
                metadata: request.user.as_ref().map(|id| AnthropicMetadata {
                    user_id: id.clone(),
                }),
            };

            tracing::debug!(model = %request.model, "Sending Anthropic completion request");

            // Wait for rate limiter before sending
            wait_for_rate_limit(&rate_limiter).await;

            let response = match send_with_retry(|| {
                client
                    .post(ANTHROPIC_API_URL)
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .json(&api_request)
            })
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());

                    yield Err(anyhow::anyhow!("{} error: Anthropic request failed: {}", provider_str, e));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let msg = format!(
                    "{} error: Anthropic API error {}: {}",
                    provider_str, status, body
                );
                yield Err(flashmind_types::LlmError::classify(msg).into());
                metrics::counter!("llm.requests.errors").increment(1);
                metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                return;
            }

            let mut stream = response.bytes_stream().eventsource();
            let mut finish_reason = FinishReason::Stop;
            let mut prompt_tokens = 0u32;
            let mut completion_tokens = 0u32;
            // Track which content blocks are tool_use blocks
            let mut tool_blocks: std::collections::HashMap<usize, (String, String)> = std::collections::HashMap::new();

            while let Some(event) = stream.next().await {
                let event = match event {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::debug!(error = %e, "Anthropic SSE error");
                        continue;
                    }
                };

                let event_type = event.event.as_str();
                let data = event.data.as_str();

                match event_type {
                    "message_start" => {
                        if let Ok(msg) = serde_json::from_str::<MessageStart>(data)
                            && let Some(usage) = msg.message.usage {
                                prompt_tokens = usage.input_tokens.unwrap_or(0);
                            }
                    }
                    "content_block_start" => {
                        if let Ok(block) = serde_json::from_str::<ContentBlockStart>(data)
                            && block.content_block.block_type == "tool_use" {
                                let id = block.content_block.id.unwrap_or_default();
                                let name = block.content_block.name.unwrap_or_default();
                                tool_blocks.insert(block.index, (id.clone(), name.clone()));
                                yield Ok(StreamEvent::ToolCallStart {
                                    index: block.index,
                                    id,
                                    name,
                                });
                            }
                    }
                    "content_block_delta" => {
                        if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(data) {
                            match delta.delta.delta_type.as_str() {
                                "text_delta" => {
                                    if let Some(text) = delta.delta.text {
                                        yield Ok(StreamEvent::ContentDelta(text));
                                    }
                                }
                                "thinking_delta" => {
                                    if let Some(thinking) = delta.delta.thinking {
                                        yield Ok(StreamEvent::ReasoningDelta(thinking));
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(json) = delta.delta.partial_json {
                                        yield Ok(StreamEvent::ToolCallDelta {
                                            index: delta.index,
                                            arguments: json,
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "message_delta" => {
                        if let Ok(msg) = serde_json::from_str::<MessageDelta>(data) {
                            if let Some(reason) = msg.delta.stop_reason {
                                finish_reason = match reason.as_str() {
                                    "end_turn" | "stop" => FinishReason::Stop,
                                    "tool_use" => FinishReason::ToolCalls,
                                    "max_tokens" => FinishReason::Length,
                                    _ => FinishReason::Stop,
                                };
                            }
                            if let Some(usage) = msg.usage {
                                completion_tokens = usage.output_tokens.unwrap_or(0);
                            }
                        }
                    }
                    "message_stop" => {
                        break;
                    }
                    _ => {}
                }
            }

            if prompt_tokens > 0 || completion_tokens > 0 {
                metrics::histogram!("llm.tokens.prompt").record(prompt_tokens as f64);
                metrics::histogram!("llm.tokens.completion").record(completion_tokens as f64);
                metrics::histogram!("llm.tokens.total").record((prompt_tokens + completion_tokens) as f64);
            }
            if prompt_tokens > 0 || completion_tokens > 0 {
                yield Ok(StreamEvent::Usage(TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                }));
            }

            metrics::counter!("llm.finish_reason", "reason" => finish_reason.to_string()).increment(1);
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());

            yield Ok(StreamEvent::Finished(finish_reason));
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn provider(&self) -> flashmind_types::Provider {
        flashmind_types::Provider::Anthropic
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        #[derive(Deserialize)]
        struct ListResponse {
            data: Vec<AnthropicModelEntry>,
        }
        #[derive(Deserialize)]
        struct AnthropicModelEntry {
            id: String,
            display_name: String,
            #[serde(rename = "type")]
            _type: String,
        }

        let mut url = reqwest::Url::parse("https://api.anthropic.com/v1/models").ok()?;
        url.query_pairs_mut().append_pair("limit", "100");

        let resp = self
            .client
            .get(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let list: ListResponse = resp.json().await.ok()?;

        let models: Vec<ModelInfo> = list
            .data
            .into_iter()
            .filter(|e| e._type == "model")
            .map(|e| {
                let is_reasoning = e.id.contains("think");
                let capabilities = ModelCapabilities {
                    tool_calling: true,
                    images: true,
                    documents: true,
                    reasoning: is_reasoning,
                    ..Default::default()
                };
                let categories = capabilities.categories();

                // Pricing per million tokens (USD) — convert to per-token
                let pricing = anthropic_pricing(&e.id);

                ModelInfo {
                    id: e.id,
                    name: Some(e.display_name),
                    context_length: Some(200_000),
                    max_completion_tokens: None,
                    capabilities,
                    categories,
                    pricing,
                }
            })
            .collect();

        Some(models)
    }
}

fn anthropic_pricing(model_id: &str) -> ModelPricing {
    let per_m = |input: f64, output: f64, cache: f64| ModelPricing {
        prompt: Some(input / 1_000_000.0),
        completion: Some(output / 1_000_000.0),
        image: None,
        cache_read: Some(cache / 1_000_000.0),
    };

    if model_id.starts_with("claude-opus-4")
        || model_id.starts_with("claude-3-opus")
        || model_id.starts_with("claude-3.0-opus")
    {
        per_m(15.0, 75.0, 1.5)
    } else if model_id.starts_with("claude-sonnet-4")
        || model_id.starts_with("claude-3-7-sonnet")
        || model_id.starts_with("claude-3.7-sonnet")
        || model_id.starts_with("claude-3-5-sonnet")
        || model_id.starts_with("claude-3.5-sonnet")
    {
        per_m(3.0, 15.0, 0.3)
    } else if model_id.starts_with("claude-3-5-haiku") || model_id.starts_with("claude-3.5-haiku") {
        per_m(0.80, 4.0, 0.08)
    } else if model_id.starts_with("claude-3-haiku") || model_id.starts_with("claude-3.0-haiku") {
        per_m(0.25, 1.25, 0.03)
    } else {
        ModelPricing::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::message::{ContentPart, ToolCall};

    #[test]
    fn test_provider_name() {
        let provider =
            AnthropicProvider::new("test-key".into(), crate::http::create_rate_limiter(50));
        assert_eq!(provider.name(), "anthropic");
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let (system, api_msgs) = convert_messages(&messages);
        assert_eq!(system.as_deref(), Some("You are helpful"));
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0].role, "user");
    }

    #[test]
    fn test_convert_messages_tool_calls() {
        let messages = vec![Message::assistant_with_tool_calls(
            "Let me check",
            vec![ToolCall {
                id: "tc-1".into(),
                name: "file_read".into(),
                arguments: serde_json::json!({"path": "/tmp"}),
            }],
        )];
        let (_, api_msgs) = convert_messages(&messages);
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0].role, "assistant");
    }

    #[test]
    fn test_convert_messages_multimodal() {
        let messages = vec![Message::user_with_parts(
            "What's in this image?",
            vec![
                ContentPart::Text {
                    text: "What's in this image?".into(),
                },
                ContentPart::Image {
                    media_type: "image/jpeg".into(),
                    data: "abc123base64".into(),
                },
            ],
        )];
        let (_, api_msgs) = convert_messages(&messages);
        assert_eq!(api_msgs.len(), 1);
        match &api_msgs[0].content {
            AnthropicContent::Blocks(blocks) => {
                // msg.content produces one Text block, then parts add Text + Image = 3 total
                assert_eq!(blocks.len(), 3);
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_convert_tools_empty() {
        assert!(convert_tools(&[]).is_none());
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![flashmind_types::ToolDefinition {
            name: "bash".into(),
            description: "Run command".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let result = convert_tools(&tools).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "bash");
    }
}
