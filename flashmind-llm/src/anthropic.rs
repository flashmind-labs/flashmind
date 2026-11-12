//! Anthropic direct API provider.
//!
//! Uses the Anthropic Messages API with SSE streaming. Supports tool calling,
//! reasoning (extended thinking), and image/document inputs.
//!
//! See <https://docs.anthropic.com/en/api/messages> for the API reference.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::{Stream, StreamExt};
use reqwest::{Client, Response};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::http::{http_client_builder, send_with_retry, wait_for_rate_limit};
use crate::sse::{content_filter_error, stream_error_kind};
use flashmind_types::message::{ContentPart, Message, Role};
use flashmind_types::model::ReasoningLevel;
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmError, LlmErrorKind, LlmProvider,
    ModelCapabilities, ModelInfo, ModelPricing, NativeCompaction, NativeReasoningFormat,
    ReasoningDetails, StreamEvent, TokenUsage,
};
use metrics;
use ratelimit::Ratelimiter;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
const COMPACTION_BETA: &str = "compact-2026-09-04";
const COMPACTION_REPLY_TOKENS: u32 = 4096;

/// Default rate limit for Anthropic (requests per minute).
const DEFAULT_RPM: u32 = 60;

/// Anthropic Messages API provider.
///
/// Direct integration with Claude models using SSE streaming. Supports tool calling,
/// reasoning (extended thinking), and image/document inputs.
///
/// # Example
///
/// ```rust,ignore
/// let provider = AnthropicProvider::new(api_key);
/// ```
pub struct AnthropicProvider {
    client: Client,
    api_url: String,
    api_key: String,
    /// Request rate limiter.
    rate_limiter: Arc<Ratelimiter>,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given API key and a default rate limit.
    pub fn new(api_key: String) -> Self {
        Self::with_rate_limit(api_key, DEFAULT_RPM)
    }

    /// Create a new Anthropic provider with a custom rate limit (requests per minute).
    pub fn with_rate_limit(api_key: String, rpm: u32) -> Self {
        Self::with_rate_limiter(api_key, crate::http::create_rate_limiter(rpm))
    }

    /// Create a new Anthropic provider with a pre-built rate limiter.
    pub fn with_rate_limiter(api_key: String, rate_limiter: Arc<Ratelimiter>) -> Self {
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            api_url: ANTHROPIC_API_URL.into(),
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
    /// Automatically cache the growing prompt with the default five-minute TTL.
    cache_control: AnthropicCacheControl,
    /// Tool definitions (Anthropic format).
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    /// Extended thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
    /// Temperature for sampling.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::wire_types::serialize_optional_decimal"
    )]
    temperature: Option<Decimal>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::wire_types::serialize_optional_decimal"
    )]
    top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicCacheControl {
    Ephemeral,
}

#[derive(Serialize)]
struct AnthropicMetadata {
    user_id: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicThinking {
    Enabled { budget_tokens: u32 },
    Adaptive,
}

#[derive(Serialize)]
struct AnthropicOutputConfig {
    effort: ReasoningLevel,
}

fn supports_adaptive_thinking(model: &str) -> bool {
    // OpenRouter slugs look like `~anthropic/claude-opus-5.5`.
    let model = model.trim_start_matches('~');
    let model = model.strip_prefix("anthropic/").unwrap_or(model);
    let Some(model) = model.strip_prefix("claude-") else {
        return false;
    };
    if model.starts_with("fable-") || model.starts_with("mythos-") {
        return true;
    }
    let Some(version) = ["opus-", "sonnet-", "haiku-"]
        .iter()
        .find_map(|family| model.strip_prefix(family))
    else {
        return false;
    };
    let mut components = version.split(['-', '.']);
    let major = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    let minor = components
        .next()
        .filter(|value| value.len() <= 2)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    major.is_some_and(|major| major > 4 || (major == 4 && minor >= 6))
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: AnthropicContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
    NativeBlocks(Vec<Value>),
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
    content_block: Map<String, Value>,
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
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    citation: Option<Value>,
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
    #[allow(dead_code)]
    output_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system = None;
    let mut api_messages = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                system = Some(msg.content.clone());
            }
            Role::Assistant => {
                let content = if let Some(blocks) = msg
                    .reasoning_details
                    .native_items(NativeReasoningFormat::Anthropic)
                {
                    AnthropicContent::NativeBlocks(blocks.to_vec())
                } else if let Some(ref tool_calls) = msg.tool_calls {
                    let mut blocks = Vec::with_capacity(tool_calls.len() + 1);
                    if !msg.content.is_empty() {
                        blocks.push(AnthropicBlock::Text {
                            text: msg.content.clone(),
                        });
                    }
                    blocks.extend(tool_calls.iter().map(|tc| AnthropicBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input: tc.arguments.clone(),
                    }));
                    AnthropicContent::Blocks(blocks)
                } else {
                    AnthropicContent::Text(msg.content.clone())
                };
                api_messages.push(AnthropicMessage {
                    role: "assistant",
                    content,
                });
            }
            Role::Tool => {
                // Tool results go as user messages in Anthropic format
                api_messages.push(AnthropicMessage {
                    role: "user",
                    content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: msg.content.clone(),
                    }]),
                });
            }
            Role::Developer => {
                let content = format!("<system>\n{}\n</system>", msg.content);
                api_messages.push(AnthropicMessage {
                    role: "user",
                    content: AnthropicContent::Text(content),
                });
            }
            Role::User => {
                let content = if let Some(ref parts) = msg.parts {
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
                    AnthropicContent::Blocks(blocks)
                } else {
                    AnthropicContent::Text(msg.content.clone())
                };
                api_messages.push(AnthropicMessage {
                    role: "user",
                    content,
                });
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

fn build_api_request(request: &CompletionRequest) -> AnthropicRequest {
    let (system, messages) = convert_messages(&request.messages);
    let tools = convert_tools(&request.tools);

    let adaptive = supports_adaptive_thinking(request.model.capability_name());
    let thinking = match request.reasoning {
        ReasoningLevel::Off => None,
        _ if adaptive => Some(AnthropicThinking::Adaptive),
        ReasoningLevel::Low => Some(AnthropicThinking::Enabled {
            budget_tokens: 4000,
        }),
        ReasoningLevel::Medium => Some(AnthropicThinking::Enabled {
            budget_tokens: 16000,
        }),
        ReasoningLevel::High => Some(AnthropicThinking::Enabled {
            budget_tokens: 64000,
        }),
    };
    let output_config = (adaptive && request.reasoning.is_on()).then_some(AnthropicOutputConfig {
        effort: request.reasoning,
    });

    let default_max_tokens = if thinking.is_some() { 16000 } else { 4096 };

    AnthropicRequest {
        model: request.model.name().to_string(),
        max_tokens: request.max_tokens.unwrap_or(default_max_tokens),
        system,
        messages,
        stream: true,
        cache_control: AnthropicCacheControl::Ephemeral,
        tools,
        thinking,
        output_config,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        metadata: request.user.as_ref().map(|id| AnthropicMetadata {
            user_id: id.clone(),
        }),
    }
}

/// Beta that OpenRouter's Messages endpoint accepts for compaction blocks.
pub(crate) const OPENROUTER_COMPACTION_BETA: &str = "compact-2026-01-12";
/// The smallest input that Anthropic's `compact_20260112` edit will compact.
pub(crate) const MIN_COMPACTION_TRIGGER_TOKENS: u32 = 50_000;
/// Keeps the edit present for replay without compacting during a normal turn.
const INERT_COMPACTION_TRIGGER_TOKENS: u32 = 10_000_000;
/// Closes a history that ends with an assistant turn before compaction.
const COMPACTION_CHECKPOINT_REQUEST: &str =
    "Save a context checkpoint before my next request. Do not reply.";

/// Build a Messages request for OpenRouter's Anthropic-compatible endpoint.
///
/// Compaction blocks are only accepted while the `compact_20260112` edit is
/// configured, so every request carries it. With `compaction` set, the edit
/// compacts at the minimum trigger and pauses before generating a reply.
pub(crate) fn openrouter_request(
    request: &CompletionRequest,
    compaction: Option<&str>,
) -> anyhow::Result<Value> {
    let mut api_request = build_api_request(request);
    api_request.model = request.model.capability_name().to_owned();
    if compaction.is_some() {
        // Thinking would require signed blocks that Chat Completions history lacks.
        api_request.thinking = None;
        api_request.output_config = None;
        api_request.stream = false;
    }
    if compaction.is_some()
        && api_request
            .messages
            .last()
            .is_some_and(|message| message.role == "assistant")
    {
        // Current models reject prefill; compaction pauses before any reply.
        api_request.messages.push(AnthropicMessage {
            role: "user",
            content: AnthropicContent::Text(COMPACTION_CHECKPOINT_REQUEST.into()),
        });
    }
    let mut body = serde_json::to_value(api_request)?;
    let edit = match compaction {
        Some(instructions) => serde_json::json!({
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": MIN_COMPACTION_TRIGGER_TOKENS},
            "pause_after_compaction": true,
            "instructions": instructions,
        }),
        None => serde_json::json!({
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": INERT_COMPACTION_TRIGGER_TOKENS},
        }),
    };
    body["context_management"] = serde_json::json!({"edits": [edit]});
    Ok(body)
}

/// Read an OpenRouter compaction, whose summary block carries no signature.
pub(crate) fn parse_openrouter_compaction(body: Value) -> anyhow::Result<Option<NativeCompaction>> {
    parse_compaction_block(body, false)
}

/// Stream a Messages response from any Anthropic-compatible endpoint.
pub(crate) fn stream_messages(response: Response) -> CompletionStream {
    Box::pin(process_sse_stream(response, Instant::now()))
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn compact(
        &self,
        request: CompletionRequest,
        instructions: &str,
    ) -> anyhow::Result<Option<NativeCompaction>> {
        let mut api_request = build_api_request(&request);
        if let Some(AnthropicThinking::Enabled { budget_tokens }) = &api_request.thinking {
            api_request.max_tokens = api_request.max_tokens.max(
                budget_tokens
                    .checked_add(COMPACTION_REPLY_TOKENS)
                    .ok_or_else(|| anyhow::anyhow!("Compaction output budget overflow"))?,
            );
        }
        let mut body = serde_json::to_value(api_request)?;
        body["stream"] = Value::Bool(false);
        body["compaction"] = serde_json::json!({"type":"summarize","instructions":instructions});
        tracing::debug!(model = %request.model, messages = request.messages.len(), "Requesting native Anthropic compaction");
        wait_for_rate_limit(&self.rate_limiter).await;
        let response = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", COMPACTION_BETA)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if matches!(status.as_u16(), 400 | 404 | 405 | 422 | 501) {
            tracing::warn!(model = %request.model, status = status.as_u16(), "Native Anthropic compaction unavailable; using local summarization");
            return Ok(None);
        }
        let body: Value = response.error_for_status()?.json().await?;
        parse_compaction(body)
    }

    async fn context_window(&self, _model: &flashmind_types::Model) -> Option<u32> {
        // All Claude models have 200k context window
        Some(200_000)
    }

    async fn capabilities(&self, _model: &flashmind_types::Model) -> ModelCapabilities {
        ModelCapabilities {
            audio: false,
            ..ModelCapabilities::all()
        }
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let rate_limiter = self.rate_limiter.clone();
        let provider_str = self.provider().to_string();
        let api_url = self.api_url.clone();
        let has_compaction = request.messages.iter().any(|message| {
            message
                .reasoning_details
                .native_items(NativeReasoningFormat::Anthropic)
                .is_some_and(|blocks| {
                    blocks.iter().any(|block| {
                        block.get("type").and_then(Value::as_str) == Some("compaction")
                    })
                })
        });

        Box::pin(stream! {
            let start = Instant::now();
            metrics::counter!("llm.requests.started").increment(1);
            let api_request = build_api_request(&request);

            tracing::debug!(model = %request.model, "Sending Anthropic completion request");

            // Wait for rate limiter before sending
            wait_for_rate_limit(&rate_limiter).await;

            let response = match send_with_retry(|| {
                let request = client
                    .post(&api_url)
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .json(&api_request);
                if has_compaction { request.header("anthropic-beta", COMPACTION_BETA) } else { request }
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
                yield Err(LlmError::classify(msg).into());
                metrics::counter!("llm.requests.errors").increment(1);
                metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                return;
            }

            for await event in process_sse_stream(response, start) {
                yield event;
            }
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
            #[serde(default)]
            created_at: Option<DateTime<Utc>>,
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
                    image: Default::default(),
                    id: e.id,
                    name: Some(e.display_name),
                    created_at: e.created_at,
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
    let million = Decimal::from(1_000_000);
    let per_m =
        |input: Decimal, output: Decimal, cache_read: Decimal, cache_write: Decimal| ModelPricing {
            prompt: Some(input / million),
            completion: Some(output / million),
            image: None,
            cache_read: Some(cache_read / million),
            cache_write: Some(cache_write / million),
            reference: None,
        };

    if model_id.starts_with("claude-opus-4")
        || model_id.starts_with("claude-3-opus")
        || model_id.starts_with("claude-3.0-opus")
    {
        per_m(
            Decimal::from(15),
            Decimal::from(75),
            Decimal::new(15, 1),
            Decimal::new(1875, 2),
        )
    } else if model_id.starts_with("claude-sonnet-4")
        || model_id.starts_with("claude-3-7-sonnet")
        || model_id.starts_with("claude-3.7-sonnet")
        || model_id.starts_with("claude-3-5-sonnet")
        || model_id.starts_with("claude-3.5-sonnet")
    {
        per_m(
            Decimal::from(3),
            Decimal::from(15),
            Decimal::new(3, 1),
            Decimal::new(375, 2),
        )
    } else if model_id.starts_with("claude-3-5-haiku") || model_id.starts_with("claude-3.5-haiku") {
        per_m(
            Decimal::new(80, 2),
            Decimal::from(4),
            Decimal::new(8, 2),
            Decimal::from(1),
        )
    } else if model_id.starts_with("claude-3-haiku") || model_id.starts_with("claude-3.0-haiku") {
        per_m(
            Decimal::new(25, 2),
            Decimal::new(125, 2),
            Decimal::new(3, 2),
            Decimal::new(3125, 4),
        )
    } else {
        ModelPricing::default()
    }
}

fn parse_compaction(body: Value) -> anyhow::Result<Option<NativeCompaction>> {
    parse_compaction_block(body, true)
}

fn parse_compaction_block(
    body: Value,
    require_signature: bool,
) -> anyhow::Result<Option<NativeCompaction>> {
    if body.get("stop_reason").and_then(Value::as_str) != Some("compaction") {
        return Ok(None);
    }
    let blocks = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Native Anthropic compaction returned no content"))?;
    anyhow::ensure!(
        blocks.len() == 1 && blocks[0].get("type").and_then(Value::as_str) == Some("compaction"),
        "Native Anthropic compaction returned an invalid checkpoint"
    );
    let block = &blocks[0];
    let summary = block
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Native Anthropic compaction returned an empty summary"))?;
    anyhow::ensure!(
        !require_signature
            || block
                .get("signature")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty()),
        "Native Anthropic compaction returned no signature"
    );
    let mut usage = TokenUsage::default();
    let rows = body
        .pointer("/usage/iterations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| body.get("usage").cloned().into_iter().collect());
    for row in rows {
        let count = |key: &str| -> anyhow::Result<u32> {
            Ok(u32::try_from(
                row.get(key).and_then(Value::as_u64).unwrap_or(0),
            )?)
        };
        let read = count("cache_read_input_tokens")?;
        let written = count("cache_creation_input_tokens")?;
        usage.prompt_tokens = usage
            .prompt_tokens
            .checked_add(count("input_tokens")?)
            .and_then(|count| count.checked_add(read))
            .and_then(|count| count.checked_add(written))
            .ok_or_else(|| anyhow::anyhow!("Compaction token count overflow"))?;
        usage.completion_tokens = usage
            .completion_tokens
            .checked_add(count("output_tokens")?)
            .ok_or_else(|| anyhow::anyhow!("Compaction token count overflow"))?;
        usage.cache_read_tokens = usage
            .cache_read_tokens
            .checked_add(read)
            .ok_or_else(|| anyhow::anyhow!("Compaction token count overflow"))?;
        usage.cache_creation_tokens = usage
            .cache_creation_tokens
            .checked_add(written)
            .ok_or_else(|| anyhow::anyhow!("Compaction token count overflow"))?;
    }
    usage.total_tokens = usage
        .prompt_tokens
        .checked_add(usage.completion_tokens)
        .ok_or_else(|| anyhow::anyhow!("Compaction token count overflow"))?;
    Ok(Some(NativeCompaction {
        context: ReasoningDetails::native(NativeReasoningFormat::Anthropic, blocks.clone()),
        summary: summary.to_owned(),
        usage,
    }))
}

fn process_sse_stream(
    response: Response,
    start: Instant,
) -> impl Stream<Item = anyhow::Result<StreamEvent>> {
    stream! {
        let mut stream = response.bytes_stream().eventsource();
        let mut finish_reason = FinishReason::Stop;
        let mut got_stop = false;
        let mut transport_error = None;
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut cache_read_tokens = 0u32;
        let mut cache_creation_tokens = 0u32;
        let mut blocks = BTreeMap::<usize, Map<String, Value>>::new();
        let mut tool_inputs = HashMap::<usize, String>::new();
        let mut open_blocks = HashSet::new();
        let mut native_complete = true;

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(kind = stream_error_kind(&e), elapsed_ms = start.elapsed().as_millis(), "Anthropic SSE error");
                    if let EventStreamError::Transport(error) = e {
                        transport_error = Some(anyhow::Error::new(error));
                        break;
                    }
                    continue;
                }
            };

            let event_type = event.event.as_str();
            let data = event.data.as_str();

            match event_type {
                "message_start" => {
                    if let Ok(msg) = serde_json::from_str::<MessageStart>(data)
                        && let Some(usage) = msg.message.usage {
                            cache_read_tokens = usage.cache_read_input_tokens.unwrap_or(0);
                            cache_creation_tokens = usage.cache_creation_input_tokens.unwrap_or(0);
                            // Anthropic reports uncached input separately from cache reads and writes.
                            prompt_tokens = usage.input_tokens.unwrap_or(0)
                                + cache_read_tokens + cache_creation_tokens;
                        }
                }
                "content_block_start" => {
                    if let Ok(block) = serde_json::from_str::<ContentBlockStart>(data) {
                        let kind = block.content_block.get("type").and_then(Value::as_str).unwrap_or("");
                        if kind == "tool_use" {
                            yield Ok(StreamEvent::ToolCallStart {
                                index: block.index,
                                id: block.content_block.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
                                name: block.content_block.get("name").and_then(Value::as_str).unwrap_or_default().to_owned(),
                            });
                        }
                        if let Some(text) = block.content_block.get(if kind == "thinking" {"thinking"} else {"text"}).and_then(Value::as_str).filter(|text| !text.is_empty()) {
                            yield Ok(if kind == "thinking" { StreamEvent::ReasoningDelta(text.to_owned()) } else { StreamEvent::ContentDelta(text.to_owned()) });
                        }
                        blocks.insert(block.index, block.content_block);
                        open_blocks.insert(block.index);
                    }
                }
                "content_block_delta" => {
                    if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(data) {
                        match delta.delta.delta_type.as_str() {
                            "text_delta" => {
                                if let Some(text) = delta.delta.text {
                                    append_block_text(&mut blocks, delta.index, "text", &text);
                                    yield Ok(StreamEvent::ContentDelta(text));
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) = delta.delta.thinking {
                                    append_block_text(&mut blocks, delta.index, "thinking", &thinking);
                                    yield Ok(StreamEvent::ReasoningDelta(thinking));
                                }
                            }
                            "signature_delta" => {
                                if let Some(signature) = delta.delta.signature {
                                    append_block_text(&mut blocks, delta.index, "signature", &signature);
                                }
                            }
                            "citations_delta" => {
                                if let Some(citation) = delta.delta.citation
                                    && let Some(block) = blocks.get_mut(&delta.index)
                                {
                                    let citations = block.entry("citations").or_insert_with(|| Value::Array(Vec::new()));
                                    if let Value::Array(citations) = citations { citations.push(citation); }
                                }
                            }
                            "input_json_delta" => {
                                if let Some(json) = delta.delta.partial_json {
                                    tool_inputs.entry(delta.index).or_default().push_str(&json);
                                    yield Ok(StreamEvent::ToolCallDelta {
                                        index: delta.index,
                                        arguments: json,
                                    });
                                }
                            }
                            // Preserve citation and future provider deltas only
                            // when their complete block can be reconstructed.
                            _ => { native_complete = false; }
                        }
                    }
                }
                "content_block_stop" => {
                    if let Ok(event) = serde_json::from_str::<Value>(data)
                        && let Some(index) = event.get("index").and_then(Value::as_u64).and_then(|index| usize::try_from(index).ok())
                    {
                        open_blocks.remove(&index);
                        if let Some(input) = tool_inputs.remove(&index) {
                            match serde_json::from_str::<Value>(&input) {
                                Ok(input) => { if let Some(block) = blocks.get_mut(&index) { block.insert("input".into(), input); } }
                                Err(_) => native_complete = false,
                            }
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
                                "refusal" | "content_filter" => FinishReason::ContentFilter,
                                _ => FinishReason::Stop,
                            };
                        }
                        if let Some(usage) = msg.usage {
                            completion_tokens = usage.output_tokens.unwrap_or(0);
                        }
                    }
                }
                "message_stop" => {
                    got_stop = true;
                    break;
                }
                _ => {}
            }
        }

        if prompt_tokens > 0 || completion_tokens > 0 {
            metrics::histogram!("llm.tokens.prompt").record(prompt_tokens as f64);
            metrics::histogram!("llm.tokens.completion").record(completion_tokens as f64);
            metrics::histogram!("llm.tokens.total").record((prompt_tokens + completion_tokens) as f64);
            yield Ok(StreamEvent::Usage(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            }));
        }

        if !got_stop {
            metrics::counter!("llm.requests.errors").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
            let error = LlmError::new(
                LlmErrorKind::StreamError,
                "Anthropic stream ended before message_stop",
            );
            yield Err(match transport_error {
                Some(source) => source.context(error),
                None => error.into(),
            });
            return;
        }

        metrics::counter!("llm.finish_reason", "reason" => finish_reason.to_string()).increment(1);
        metrics::counter!("llm.requests.completed").increment(1);
        metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());

        if let Some(error) = content_filter_error(finish_reason) {
            yield Err(error.into());
            return;
        }
        if native_complete && tool_inputs.is_empty() && open_blocks.is_empty()
            && blocks.values().all(|block| block.get("type").and_then(Value::as_str) != Some("thinking")
                || block.get("signature").and_then(Value::as_str).is_some_and(|signature| !signature.is_empty()))
            && blocks.values().any(|block| matches!(block.get("type").and_then(Value::as_str), Some("thinking" | "redacted_thinking")))
        {
            yield Ok(StreamEvent::ReasoningDetailsDelta(ReasoningDetails::native(
                NativeReasoningFormat::Anthropic, blocks.into_values().map(Value::Object).collect(),
            )));
        }
        yield Ok(StreamEvent::Finished(finish_reason));
    }
}

fn append_block_text(
    blocks: &mut BTreeMap<usize, Map<String, Value>>,
    index: usize,
    key: &str,
    text: &str,
) {
    if let Some(block) = blocks.get_mut(&index) {
        let value = block
            .entry(key.to_owned())
            .or_insert_with(|| Value::String(String::new()));
        if let Value::String(value) = value {
            value.push_str(text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ToolDefinition;
    use flashmind_types::message::{ContentPart, ToolCall};

    const CONTENT: &str = "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n";
    const FINISH: &str = "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":3}}\n\n";

    #[test]
    fn openrouter_slugs_use_the_same_adaptive_thinking_rule() {
        assert!(supports_adaptive_thinking("anthropic/claude-opus-5.5"));
        assert!(supports_adaptive_thinking("~anthropic/claude-sonnet-4.6"));
        assert!(!supports_adaptive_thinking("anthropic/claude-sonnet-4.5"));
        assert!(!supports_adaptive_thinking("openai/claude-opus-5.5"));
    }

    #[test]
    fn current_claude_models_use_adaptive_thinking_and_effort() {
        for model in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-opus-5-5",
            "claude-fable-5-2",
            "claude-mythos-preview",
        ] {
            for level in [
                ReasoningLevel::Low,
                ReasoningLevel::Medium,
                ReasoningLevel::High,
            ] {
                let mut request = CompletionRequest::new(
                    format!("anthropic:{model}").parse().unwrap(),
                    vec![Message::user("task")],
                );
                request.reasoning = level;
                let body = serde_json::to_value(build_api_request(&request)).unwrap();
                assert_eq!(
                    body["thinking"],
                    serde_json::json!({"type":"adaptive"}),
                    "{model}"
                );
                assert_eq!(
                    body["output_config"]["effort"],
                    serde_json::to_value(level).unwrap()
                );
            }
        }
    }

    #[test]
    fn older_claude_models_keep_manual_thinking() {
        for model in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-opus-4-5",
            "claude-haiku-4-5",
            "claude-3-7-sonnet-20250219",
        ] {
            let mut request = CompletionRequest::new(
                format!("anthropic:{model}").parse().unwrap(),
                vec![Message::user("task")],
            );
            request.reasoning = ReasoningLevel::Low;
            let body = serde_json::to_value(build_api_request(&request)).unwrap();
            assert_eq!(
                body["thinking"],
                serde_json::json!({"type":"enabled","budget_tokens":4000})
            );
            assert!(body.get("output_config").is_none());
        }
    }

    #[tokio::test]
    async fn native_compaction_uses_adaptive_thinking_on_current_claude() {
        let blocks = serde_json::json!([{"type":"compaction","content":"Retain task.","signature":"signed"}]);
        let endpoint = crate::sse::tests::http_endpoint(
            200,
            "application/json",
            serde_json::json!({"stop_reason":"compaction","content":blocks} ).to_string(),
        )
        .await;
        let mut provider = AnthropicProvider::new("example-key".into());
        provider.api_url = endpoint.url;
        let mut request = CompletionRequest::new(
            "anthropic:claude-opus-4-7".parse().unwrap(),
            vec![Message::user("task")],
        );
        request.reasoning = ReasoningLevel::High;
        request.max_tokens = Some(COMPACTION_REPLY_TOKENS);
        assert!(
            provider
                .compact(request, "retain work")
                .await
                .unwrap()
                .is_some()
        );
        let captured = endpoint.request.await.unwrap();
        assert_eq!(
            captured.body["thinking"],
            serde_json::json!({"type":"adaptive"})
        );
        assert_eq!(captured.body["output_config"]["effort"], "high");
        assert_eq!(captured.body["max_tokens"], COMPACTION_REPLY_TOKENS);
    }

    #[tokio::test]
    async fn native_compaction_preserves_system_tools_thinking_and_the_signed_result() {
        use serde_json::json;
        let blocks = json!([{"type":"compaction","content":"Retain task and evidence.","signature":"signed-summary","extra":"keep"}]);
        let endpoint = crate::sse::tests::http_endpoint(200, "application/json", json!({
            "stop_reason":"compaction","content":blocks,
            "usage":{"input_tokens":0,"output_tokens":0,"iterations":[
                {"type":"compaction","input_tokens":10,"output_tokens":5,"cache_read_input_tokens":4,"cache_creation_input_tokens":2},
                {"type":"compaction","input_tokens":3,"output_tokens":2}
            ]}
        }).to_string()).await;
        let mut provider = AnthropicProvider::new("example-key".into());
        provider.api_url = endpoint.url;
        let original_blocks = vec![
            json!({"type":"thinking","thinking":"Plan","signature":"original-signature"}),
            json!({"type":"tool_use","id":"call_example","name":"lookup","input":{}}),
        ];
        let mut assistant = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_example".into(),
                name: "lookup".into(),
                arguments: json!({}),
            }],
        );
        assistant.reasoning_details =
            ReasoningDetails::native(NativeReasoningFormat::Anthropic, original_blocks.clone());
        let mut request = CompletionRequest::new(
            "anthropic:example-model".parse().unwrap(),
            vec![
                Message::system("original instructions"),
                Message::user("task"),
                assistant,
                Message::tool_result("call_example", "evidence"),
            ],
        );
        request.reasoning = ReasoningLevel::High;
        request.max_tokens = Some(16_384);
        request.tools.push(ToolDefinition {
            name: "lookup".into(),
            description: "Find evidence".into(),
            parameters: json!({"type":"object"}),
        });
        let compacted = provider
            .compact(request, "retain pending work")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(compacted.usage.prompt_tokens, 19);
        assert_eq!(compacted.usage.total_tokens, 26);
        let captured = endpoint.request.await.unwrap();
        assert!(
            captured
                .headers
                .to_ascii_lowercase()
                .contains(&format!("anthropic-beta: {COMPACTION_BETA}"))
        );
        assert_eq!(captured.body["system"], "original instructions");
        assert_eq!(captured.body["tools"][0]["name"], "lookup");
        assert_eq!(captured.body["thinking"]["budget_tokens"], 64_000);
        assert_eq!(captured.body["max_tokens"], 68_096);
        assert_eq!(
            captured.body["messages"][1]["content"],
            json!(original_blocks)
        );
        assert_eq!(
            captured.body["compaction"],
            json!({"type":"summarize","instructions":"retain pending work"})
        );
        assert_eq!(captured.body["stream"], false);
        let stream = concat!(
            "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"continued\"}}\n\n",
            "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\ndata: {}\n\n"
        );
        let endpoint =
            crate::sse::tests::http_endpoint(200, "text/event-stream", stream.into()).await;
        provider.api_url = endpoint.url;
        let mut checkpoint = Message::assistant("");
        checkpoint.reasoning_details = compacted.context;
        let request = CompletionRequest::new(
            "anthropic:example-model".parse().unwrap(),
            vec![
                Message::system("original instructions"),
                checkpoint,
                Message::user("next"),
            ],
        );
        let events = provider.complete(request).collect::<Vec<_>>().await;
        assert!(matches!(
            events.last(),
            Some(Ok(StreamEvent::Finished(FinishReason::Stop)))
        ));
        let captured = endpoint.request.await.unwrap();
        assert_eq!(captured.body["messages"][0]["content"], blocks);
        assert!(
            captured
                .headers
                .to_ascii_lowercase()
                .contains(&format!("anthropic-beta: {COMPACTION_BETA}"))
        );
    }

    #[tokio::test]
    async fn native_compaction_declines_unsupported_models_without_hiding_failures() {
        for status in [400, 404, 501, 401, 500] {
            let endpoint =
                crate::sse::tests::http_endpoint(status, "application/json", "{}".into()).await;
            let mut provider = AnthropicProvider::new("example-key".into());
            provider.api_url = endpoint.url;
            let result = provider
                .compact(
                    CompletionRequest::new(
                        "anthropic:example-model".parse().unwrap(),
                        vec![Message::user("task")],
                    ),
                    "retain state",
                )
                .await;
            if matches!(status, 400 | 404 | 501) {
                assert!(result.unwrap().is_none());
            } else {
                assert!(result.is_err());
            }
            endpoint.request.await.unwrap();
        }
        assert!(
            parse_compaction(serde_json::json!({"stop_reason":"refusal","content":[]}))
                .unwrap()
                .is_none()
        );
        assert!(parse_compaction(serde_json::json!({"stop_reason":"compaction","content":[{"type":"compaction","content":"summary"}]})).is_err());
    }

    #[tokio::test]
    async fn signed_and_redacted_thinking_round_trip_with_native_block_order() {
        use serde_json::json;
        let mut body = String::new();
        let mut event =
            |name: &str, data: Value| body.push_str(&format!("event: {name}\ndata: {data}\n\n"));
        event(
            "content_block_start",
            json!({"index":0,"content_block":{"type":"thinking","thinking":"","signature":"","extra":"keep"}}),
        );
        event(
            "content_block_delta",
            json!({"index":0,"delta":{"type":"thinking_delta","thinking":"Plan"}}),
        );
        for signature in ["sig-", "end"] {
            event(
                "content_block_delta",
                json!({"index":0,"delta":{"type":"signature_delta","signature":signature}}),
            );
        }
        event("content_block_stop", json!({"index":0}));
        event(
            "content_block_start",
            json!({"index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}),
        );
        event("content_block_stop", json!({"index":1}));
        event(
            "content_block_start",
            json!({"index":2,"content_block":{"type":"text","text":""}}),
        );
        event(
            "content_block_delta",
            json!({"index":2,"delta":{"type":"text_delta","text":"Checking"}}),
        );
        event(
            "content_block_delta",
            json!({"index":2,"delta":{"type":"citations_delta","citation":{
                "type":"char_location","document_index":0,"document_title":"Example","start_char_index":0,"end_char_index":1,"cited_text":"x"
            }}}),
        );
        event("content_block_stop", json!({"index":2}));
        event(
            "content_block_start",
            json!({"index":3,"content_block":{"type":"tool_use","id":"call_example","name":"lookup","input":{}}}),
        );
        event(
            "content_block_delta",
            json!({"index":3,"delta":{"type":"input_json_delta","partial_json":"{\"key\":\"value\"}"}}),
        );
        event("content_block_stop", json!({"index":3}));
        event(
            "message_delta",
            json!({"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}),
        );
        event("message_stop", json!({}));
        let events = completion_events(body, false).await;
        let mut details = ReasoningDetails::default();
        for event in events {
            if let StreamEvent::ReasoningDetailsDelta(delta) = event.unwrap() {
                details.append(delta);
            }
        }
        let expected = json!([
            {"type":"thinking","thinking":"Plan","signature":"sig-end","extra":"keep"},
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"text","text":"Checking","citations":[{"type":"char_location","document_index":0,"document_title":"Example","start_char_index":0,"end_char_index":1,"cited_text":"x"}]},
            {"type":"tool_use","id":"call_example","name":"lookup","input":{"key":"value"}}
        ]);
        assert_eq!(
            details
                .native_items(NativeReasoningFormat::Anthropic)
                .unwrap(),
            expected.as_array().unwrap()
        );
        let mut assistant = Message::assistant_with_tool_calls(
            "Checking",
            vec![ToolCall {
                id: "call_example".into(),
                name: "lookup".into(),
                arguments: json!({"key":"value"}),
            }],
        );
        assistant.reasoning_details = details;
        let (_, messages) =
            convert_messages(&[assistant, Message::tool_result("call_example", "evidence")]);
        let messages = serde_json::to_value(messages).unwrap();
        assert_eq!(messages[0]["content"], expected);
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "call_example");
    }

    #[tokio::test]
    async fn thinking_without_a_complete_signature_is_never_replayed() {
        let body = concat!(
            "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"partial\"}}\n\n",
            "event: content_block_stop\ndata: {\"index\":0}\n\n",
            "event: message_stop\ndata: {}\n\n",
        );
        let events = completion_events(body.into(), false).await;
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Ok(StreamEvent::ReasoningDetailsDelta(_))))
        );
    }

    async fn completion_events(
        body: String,
        transport_failure: bool,
    ) -> Vec<anyhow::Result<StreamEvent>> {
        let response = crate::sse::tests::response(body, transport_failure).await;
        process_sse_stream(response, Instant::now()).collect().await
    }

    #[tokio::test]
    async fn truncated_completion_is_a_stream_error() {
        for body in [String::new(), CONTENT.into(), format!("{CONTENT}{FINISH}")] {
            let events = completion_events(body, false).await;
            crate::sse::tests::assert_stream_error(&events);
        }
    }

    #[tokio::test]
    async fn transport_failure_after_content_is_a_stream_error() {
        use reqwest::Error as RequestError;
        let events = completion_events(CONTENT.into(), true).await;
        assert!(
            matches!(events.first(), Some(Ok(StreamEvent::ContentDelta(content))) if content == "partial")
        );
        crate::sse::tests::assert_stream_error(&events);
        assert!(
            events
                .iter()
                .filter_map(|event| event.as_ref().err())
                .any(|error| { error.chain().any(|cause| cause.is::<RequestError>()) })
        );
    }

    #[tokio::test]
    async fn terminal_message_preserves_usage_and_finish_reason() {
        let body = format!(
            "event: message_start\ndata: {{\"message\":{{\"usage\":{{\"input_tokens\":7}}}}}}\n\n{CONTENT}{FINISH}event: message_stop\ndata: {{}}\n\n"
        );
        let events = completion_events(body, false).await;
        assert_eq!(events.len(), 3);
        assert!(
            matches!(&events[1], Ok(StreamEvent::Usage(usage)) if usage.prompt_tokens == 7 && usage.completion_tokens == 3)
        );
        assert!(matches!(
            events.last(),
            Some(Ok(StreamEvent::Finished(FinishReason::Length)))
        ));
    }

    #[tokio::test]
    async fn cached_input_is_included_in_token_totals_and_cost() {
        let body = format!(
            "event: message_start\ndata: {{\"message\":{{\"usage\":{{\"input_tokens\":7,\"cache_read_input_tokens\":100,\"cache_creation_input_tokens\":20}}}}}}\n\n{FINISH}event: message_stop\ndata: {{}}\n\n"
        );
        let events = completion_events(body, false).await;
        let usage = match &events[0] {
            Ok(StreamEvent::Usage(usage)) => usage,
            event => panic!("expected usage, got {event:?}"),
        };
        assert_eq!(usage.prompt_tokens, 127);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 130);
        assert_eq!(usage.cache_read_tokens, 100);
        assert_eq!(usage.cache_creation_tokens, 20);

        let pricing = ModelPricing {
            prompt: Some(Decimal::ONE),
            completion: Some(Decimal::from(2)),
            cache_read: Some(Decimal::new(1, 1)),
            cache_write: Some(Decimal::new(125, 2)),
            ..Default::default()
        };
        assert_eq!(usage.cost(&pricing), Some(Decimal::from(48)));
    }

    #[tokio::test]
    async fn entirely_cached_input_still_emits_usage_without_output() {
        for (read, written) in [(100, 0), (0, 100)] {
            let start = serde_json::json!({
                "message": {
                    "usage": {
                        "input_tokens": 0,
                        "cache_read_input_tokens": read,
                        "cache_creation_input_tokens": written,
                    }
                }
            });
            let body = format!(
                "event: message_start\ndata: {start}\n\nevent: message_stop\ndata: {{}}\n\n"
            );
            let events = completion_events(body, false).await;
            assert_eq!(events.len(), 2);
            assert!(matches!(
                &events[0], Ok(StreamEvent::Usage(usage))
                    if usage.prompt_tokens == 100 && usage.total_tokens == 100
                        && usage.completion_tokens == 0 && usage.cache_read_tokens == read
                        && usage.cache_creation_tokens == written
            ));
        }
    }

    #[test]
    fn automatic_caching_preserves_the_prefix_across_tool_turns() {
        let mut request = CompletionRequest::new(
            "anthropic:claude-sonnet-4".parse().unwrap(),
            vec![
                Message::system("You are helpful"),
                Message::user("Read the file"),
            ],
        );
        request.tools = vec![ToolDefinition {
            name: "file_read".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let first = serde_json::to_value(build_api_request(&request)).unwrap();
        assert_eq!(
            first["cache_control"],
            serde_json::json!({"type": "ephemeral"})
        );
        assert_eq!(first["system"], "You are helpful");
        assert_eq!(first["messages"][0]["content"], "Read the file");
        assert_eq!(first["tools"][0]["name"], "file_read");

        request.messages.push(Message::assistant_with_tool_calls(
            "Reading the file",
            vec![ToolCall {
                id: "call-1".into(),
                name: "file_read".into(),
                arguments: serde_json::json!({"path": "/path/to/file"}),
            }],
        ));
        request
            .messages
            .push(Message::tool_result("call-1", "File contents"));
        let next = serde_json::to_value(build_api_request(&request)).unwrap();
        assert_eq!(next["cache_control"], first["cache_control"]);
        assert_eq!(next["system"], first["system"]);
        assert_eq!(next["tools"], first["tools"]);
        assert_eq!(next["messages"][0], first["messages"][0]);
        assert_eq!(next["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(next["messages"][2]["content"][0]["tool_use_id"], "call-1");
    }

    #[test]
    fn image_after_tool_results_is_sent_as_image_block() {
        let request = CompletionRequest::new(
            "anthropic:example-model".parse().unwrap(),
            vec![
                Message::assistant_with_tool_calls(
                    "",
                    vec![
                        ToolCall {
                            id: "call-1".into(),
                            name: "image_read".into(),
                            arguments: serde_json::json!({}),
                        },
                        ToolCall {
                            id: "call-2".into(),
                            name: "read".into(),
                            arguments: serde_json::json!({}),
                        },
                    ],
                ),
                Message::tool_result("call-1", "Image loaded"),
                Message::tool_result("call-2", "File contents"),
                Message::user_with_parts(
                    "Images returned by the preceding tools:",
                    vec![ContentPart::Image {
                        media_type: "image/png".into(),
                        data: "aW1hZ2U=".into(),
                    }],
                ),
            ],
        );
        let body = serde_json::to_value(build_api_request(&request)).unwrap();
        assert_eq!(body["messages"][1]["content"][0]["tool_use_id"], "call-1");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call-2");
        assert_eq!(body["messages"][3]["role"], "user");
        assert_eq!(
            body["messages"][3]["content"][1],
            serde_json::json!({
                "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=" }
            })
        );
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::with_rate_limit("test-key".into(), 50);
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
