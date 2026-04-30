//! OpenRouter LLM provider implementation.
//! Supports streaming responses via SSE and model-aware reasoning configuration.
//!
//! <https://openrouter.ai/docs/api-reference>

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

use crate::http::{http_client_builder, send_with_retry, wait_for_rate_limit};
use crate::sse::{ToolCallTracker, process_chunk};
use crate::wire_types::{
    ApiRequestBase, ChatTemplateKwargs, StreamChunk, StreamOptions, to_api_messages, to_api_tools,
};
use crate::{ContextWindowCache, oss_capabilities};
use flashmind_types::model::Provider;
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities, ModelInfo,
    StreamEvent,
};
use metrics;
use ratelimit::Ratelimiter;

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// OpenRouter API provider. Routes requests to various LLM backends
/// (OpenAI, Anthropic, Google, etc.) via a unified API.
///
/// Supports SSE streaming, tool calling, reasoning tokens, and TTS.
/// Automatically fetches model capabilities from the `/api/v1/models` endpoint.
///
/// # Example
///
/// ```rust,ignore
/// let provider = OpenRouterProvider::new(api_key, rate_limiter);
/// ```
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    ctx_cache: Arc<ContextWindowCache>,
    /// Cached per-model capabilities from /api/v1/models.
    model_caps: Arc<Mutex<HashMap<String, ModelCapabilities>>>,
    /// Whether we've fetched model data yet.
    models_fetched: Arc<AtomicBool>,
    /// Request rate limiter.
    rate_limiter: Arc<Ratelimiter>,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider with the given API key.
    ///
    /// The API key can also be set via the `OPENROUTER_API_KEY` environment variable.
    pub fn new(api_key: String, rate_limiter: Arc<Ratelimiter>) -> Self {
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            api_key,
            ctx_cache: Arc::new(ContextWindowCache::new()),
            model_caps: Arc::new(Mutex::new(HashMap::new())),
            models_fetched: Arc::new(AtomicBool::new(false)),
            rate_limiter,
        }
    }
}

/// Entry from OpenRouter /api/v1/models response.
#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    context_length: Option<u32>,
    architecture: Option<ModelArchitecture>,
    #[serde(default)]
    supported_parameters: Vec<String>,
}

#[derive(Deserialize)]
struct ModelArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

impl OpenRouterProvider {
    /// Fetch all model entries from the OpenRouter models API.
    async fn fetch_model_entries(&self) -> Option<Vec<ModelEntry>> {
        let resp = match self.client.get(OPENROUTER_MODELS_URL).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Failed to fetch OpenRouter models: {}", e);
                return None;
            }
        };

        if !resp.status().is_success() {
            tracing::warn!("OpenRouter models API returned status {}", resp.status());
            return None;
        }

        match resp.json::<ModelsResponse>().await {
            Ok(m) => Some(m.data),
            Err(e) => {
                tracing::warn!("Failed to parse OpenRouter models response: {}", e);
                None
            }
        }
    }

    /// Build capabilities for a model entry.
    fn entry_capabilities(entry: &ModelEntry) -> ModelCapabilities {
        let modalities = entry
            .architecture
            .as_ref()
            .map(|a| &a.input_modalities[..])
            .unwrap_or_default();

        let reasoning = entry
            .supported_parameters
            .iter()
            .any(|p| p == "include_reasoning" || p == "reasoning");

        ModelCapabilities {
            tool_calling: true,
            images: modalities.iter().any(|m| m == "image"),
            documents: modalities.iter().any(|m| m == "file"),
            video: modalities.iter().any(|m| m == "video"),
            audio: modalities.iter().any(|m| m == "audio"),
            reasoning,
        }
    }

    /// Fetch model capabilities from the OpenRouter models API.
    /// Returns a map of model ID to capabilities. On error, returns empty map.
    async fn fetch_model_capabilities(&self) -> HashMap<String, ModelCapabilities> {
        let Some(entries) = self.fetch_model_entries().await else {
            return HashMap::new();
        };

        let mut caps = HashMap::new();
        for entry in &entries {
            caps.insert(entry.id.clone(), Self::entry_capabilities(entry));

            if let Some(ctx_len) = entry.context_length {
                self.ctx_cache.set(&entry.id, ctx_len);
            }
        }

        caps
    }
}

// ============================================================================
// OpenRouter-specific types (reasoning config)
// ============================================================================

/// Reasoning configuration for extended thinking models.
#[derive(Serialize)]
struct ApiReasoning {
    effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// OpenRouter request — extends the shared base with reasoning.
#[derive(Serialize)]
struct ApiRequest {
    #[serde(flatten)]
    base: ApiRequestBase,
    /// Reasoning configuration - only included when reasoning is enabled
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    /// Controls whether reasoning/chain-of-thought is included in the response.
    /// When true, models like Qwen/DeepSeek stream their reasoning via the `reasoning` delta field.
    include_reasoning: bool,
}

// ============================================================================
// Request Building
// ============================================================================

fn build_api_request(request: CompletionRequest) -> ApiRequest {
    let include_reasoning = request.reasoning.is_on();
    let tools = to_api_tools(request.tools);

    // Use capability_name for model lookup (handles aliases via real_name)
    let model_name = request.model.capability_name();
    // Gemma4 models require skip_special_tokens=false when thinking is enabled.
    let is_gemma4 = model_name.contains("gemma4");
    let include_special_tokens = is_gemma4 && request.reasoning.is_on();

    // Kimi/Moonshot models only accept presence_penalty=0
    let is_kimi = model_name.contains("kimi");
    let presence_penalty = if is_kimi {
        None
    } else {
        request.sampling.presence_penalty
    };

    ApiRequest {
        base: ApiRequestBase {
            model: request.model.name().to_string(),
            messages: to_api_messages(&request.messages),
            tool_choice: if tools.is_empty() {
                None
            } else {
                Some("auto".into())
            },
            tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            min_p: request.sampling.min_p,
            presence_penalty,
            repetition_penalty: request.sampling.repetition_penalty,
            stream: true,
            stream_options: StreamOptions::default(),
            skip_special_tokens: if include_special_tokens {
                Some(false)
            } else {
                None
            },
            chat_template_kwargs: ChatTemplateKwargs {
                enable_thinking: request.reasoning.is_on(),
            },
        },
        reasoning: if include_reasoning {
            Some(ApiReasoning {
                effort: "low".to_string(),
                max_tokens: None,
            })
        } else {
            None
        },
        include_reasoning,
    }
}

// ============================================================================
// LlmProvider Implementation
// ============================================================================

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn provider(&self) -> Provider {
        Provider::OpenRouter
    }

    async fn context_window(&self, model: &flashmind_types::Model) -> Option<u32> {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        // Check cache first
        if let Some(size) = self.ctx_cache.get(lookup_name) {
            return Some(size);
        }

        // Query OpenRouter models API
        let resp = self
            .client
            .get(OPENROUTER_MODELS_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let models: ModelsResponse = resp.json().await.ok()?;

        // Cache all models from the response
        for entry in &models.data {
            if let Some(ctx_len) = entry.context_length {
                self.ctx_cache.set(&entry.id, ctx_len);
            }
        }

        self.ctx_cache.get(lookup_name)
    }

    async fn capabilities(&self, model: &flashmind_types::Model) -> ModelCapabilities {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        // Lazy-fetch on first call
        if !self.models_fetched.load(Ordering::Relaxed) {
            let caps = self.fetch_model_capabilities().await;
            let mut cache = self.model_caps.lock().unwrap();
            *cache = caps;
            self.models_fetched.store(true, Ordering::Relaxed);
        }

        let cache = self.model_caps.lock().unwrap();

        cache
            .get(lookup_name)
            .copied()
            .or_else(|| oss_capabilities::get_oss_capabilities(lookup_name))
            .unwrap_or(ModelCapabilities {
                tool_calling: true,
                images: false,
                documents: false,
                video: false,
                audio: false,
                reasoning: false,
            })
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        let entries = self.fetch_model_entries().await?;

        let mut models: Vec<ModelInfo> = entries
            .iter()
            .map(|entry| ModelInfo {
                id: entry.id.clone(),
                context_length: entry.context_length,
                capabilities: Self::entry_capabilities(entry),
            })
            .collect();

        models.sort_by(|a, b| a.id.cmp(&b.id));
        Some(models)
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let rate_limiter = self.rate_limiter.clone();
        let provider_str = self.provider().to_string();

        Box::pin(stream! {
            let start = std::time::Instant::now();
            let request_id = format!("{:08x}", rand::random::<u32>());
            tracing::debug!(
                model = %request.model,
                messages = request.messages.len(),
                tools = request.tools.len(),
                request_id = %request_id,
                "Sending completion request"
            );

            let api_request = build_api_request(request);

            // Dump request to disk for debugging
            // if let Ok(body) = serde_json::to_string_pretty(&api_request) {
            //     let dir = crate::config::Config::base_dir().join("logs/requests");
            //     let _ = std::fs::create_dir_all(&dir);
            //     let path = dir.join(format!("{}.json", request_id));
            //     let _ = std::fs::write(&path, &body);
            // }

            // Wait for rate limiter before sending
            wait_for_rate_limit(&rate_limiter).await;

            // Use shared retry logic for 429 handling
            let response = match send_with_retry(|| {
                client
                    .post(OPENROUTER_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("HTTP-Referer", "https://github.com/flashmind-labs/agent")
                    .header("X-Title", "Flash Agent")
                    .json(&api_request)
            })
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    yield Err(anyhow::anyhow!("{} error: Request failed: {}", provider_str, e));
                    return;
                }
            };

            // Check for non-success status
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                let data = serde_json::to_string(&api_request).unwrap();
                tracing::debug!("API error response body: {text}. Sent:\n{data}\n");
                let reason = status.canonical_reason().unwrap_or("Unknown");
                yield Err(anyhow::anyhow!(
                    "{} error: API error {} {}",
                    provider_str,
                    status.as_u16(),
                    reason,
                ));
                metrics::counter!("llm.requests.errors").increment(1);
                metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                return;
            }

            for await event in process_sse_stream(response, request_id) {
                yield event;
            }
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
        })
    }
}

/// Process SSE stream and yield events.
fn process_sse_stream(
    response: reqwest::Response,
    _request_id: String,
) -> impl tokio_stream::Stream<Item = anyhow::Result<StreamEvent>> {
    stream! {
        tracing::debug!(
            status = %response.status(),
            content_type = ?response.headers().get("content-type"),
            "Starting SSE stream processing"
        );
        let mut sse_stream = response.bytes_stream().eventsource();
        let mut tracker = ToolCallTracker::default();
        let mut finish_reason = FinishReason::Stop;
        let mut event_count: u32 = 0;
        let mut error_count: u32 = 0;
        let mut got_done = false;
        let mut _raw_chunks: Vec<String> = Vec::new();

        while let Some(event_result) = sse_stream.next().await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    error_count += 1;
                    // Classify error type for better debugging
                    let (error_kind, is_fatal) = match &e {
                        eventsource_stream::EventStreamError::Transport(_) => ("transport", true),
                        eventsource_stream::EventStreamError::Parser(_) => ("parser", false),
                        eventsource_stream::EventStreamError::Utf8(_) => ("utf8", false),
                    };
                    tracing::warn!(
                        error = %e,
                        kind = error_kind,
                        event_count,
                        error_count,
                        "SSE stream error"
                    );
                    // Transport errors mean the connection is dead — stop reading
                    if is_fatal {
                        break;
                    }
                    continue;
                }
            };

            event_count += 1;
            // debug!(
            //     event_count,
            //     event_type = %event.event,
            //     data_len = event.data.len(),
            //     data_preview = %event.data.chars().take(100).collect::<String>(),
            //     "SSE event received"
            // );

            // Collect raw chunk for debugging
            // _raw_chunks.push(event.data.clone());

            if event.data == "[DONE]" {
                got_done = true;
                tracing::debug!(event_count, "SSE stream completed normally");
                break;
            }

            let chunk: StreamChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        data = %event.data,
                        event_count,
                        "Failed to parse SSE chunk JSON"
                    );
                    continue;
                }
            };

            let (events, finish) = process_chunk(&chunk, &mut tracker);
            if let Some(reason) = finish {
                finish_reason = reason;
            }
            for ev in events {
                yield Ok(ev);
            }
        }

        if !got_done {
            tracing::warn!(
                event_count,
                error_count,
                finish_reason = ?finish_reason,
                "SSE stream ended without [DONE] marker (connection dropped?)"
            );
        } else {
            tracing::debug!(event_count, error_count, "SSE stream completed successfully");
        }

        // Dump response chunks to disk for debugging
        // let dir = crate::config::Config::base_dir().join("logs/responses");
        // let _ = std::fs::create_dir_all(&dir);
        // let path = dir.join(format!("{}.jsonl", request_id));
        // let content = _raw_chunks.join("\n");
        // let _ = std::fs::write(&path, &content);

        // Emit SSE-level metrics before yielding the final event
        metrics::counter!("llm.stream.events.total").increment(event_count.into());
        metrics::counter!("llm.stream.parser_errors.total").increment(error_count.into());
        metrics::counter!("llm.finish_reason", "reason" => finish_reason.to_string()).increment(1);

        yield Ok(StreamEvent::Finished(finish_reason));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire_types::{ApiContent, ApiMessage};
    use flashmind_types::ToolCall;

    #[test]
    fn test_openrouter_new() {
        let provider =
            OpenRouterProvider::new("test-key".into(), crate::http::create_rate_limiter(200));
        assert_eq!(provider.name(), "openrouter");
    }

    #[test]
    fn test_message_conversion() {
        use flashmind_types::Message;
        let msg = Message::user("Hello");
        let api_msg = ApiMessage::from(&msg);
        assert_eq!(api_msg.role, "user");
        match &api_msg.content {
            ApiContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn test_tool_message_conversion() {
        use flashmind_types::Message;
        let msg = Message::tool_result("call-1", "result");
        let api_msg = ApiMessage::from(&msg);
        assert_eq!(api_msg.role, "tool");
        assert_eq!(api_msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn test_assistant_with_tool_calls_conversion() {
        use flashmind_types::Message;
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
}
