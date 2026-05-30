//! Ollama LLM provider for running local models.
//!
//! Uses Ollama's native `/api/chat` endpoint with NDJSON streaming. Supports the
//! `think` parameter for reasoning/thinking mode and native tool call format.
//!
//! See <https://github.com/ollama/ollama/blob/main/docs/api.md> for the API reference.

mod convert;
mod wire_types;

use anyhow::Context;
use async_stream::stream;
use async_trait::async_trait;
use futures::TryStreamExt;
use metrics;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;
use url::Url;

use serde::Deserialize;

use crate::http::{http_client_builder, send_with_retry};
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities, ModelInfo,
    ModelPricing, StreamEvent, TokenUsage,
};

use convert::*;
use wire_types::*;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434/";

/// Ollama provider using native `/api/chat` with NDJSON streaming.
///
/// Connects to a local or remote Ollama instance for running open-source models.
/// Supports thinking/reasoning mode via the `think` parameter and native tool calling.
///
/// # Example
///
/// ```rust,ignore
/// // Default: localhost:11434
/// let provider = OllamaProvider::new(None, None)?;
///
/// // Custom URL with larger context window
/// let provider = OllamaProvider::new(Some("http://remote:11434".into()), Some(128_000))?;
/// ```
pub struct OllamaProvider {
    client: Client,
    base_url: Url,
    /// Cache of model context window sizes, populated from /api/show.
    context_windows: Arc<Mutex<HashMap<String, u32>>>,
    /// User-specified context window size override (num_ctx).
    /// Ollama caps context to save memory by default; set this to use larger contexts.
    num_ctx: Option<u32>,
}

impl OllamaProvider {
    /// Create a new Ollama provider.
    ///
    /// - **`base_url`** — Ollama API URL (default: `http://localhost:11434`)
    /// - **`num_ctx`** — optional context window override. If not set, context
    ///   windows are auto-discovered from each model's metadata via `/api/show`.
    pub fn new(base_url: Option<String>, num_ctx: Option<u32>) -> anyhow::Result<Self> {
        let raw = base_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.into());
        let base_url = Url::parse(&raw).with_context(|| format!("invalid Ollama URL '{raw}'"))?;
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            client,
            base_url,
            context_windows: Arc::new(Mutex::new(HashMap::new())),
            num_ctx,
        })
    }

    fn url(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url
    }
}

// ============================================================================
// Retry Logic
// ============================================================================

/// Send request with retry logic using the shared implementation.
async fn send_with_retry_helper(
    client: &Client,
    url: &Url,
    request: &NativeRequest,
) -> anyhow::Result<reqwest::Response> {
    let url_clone = url.clone();

    let response = send_with_retry(|| client.post(url_clone.clone()).json(request)).await?;

    let status = response.status();

    // Handle non-success status codes
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(flashmind_types::LlmError::classify(format!(
            "Ollama native API error {}: {}",
            status, text
        ))
        .into());
    }

    Ok(response)
}

// ============================================================================
// LlmProvider Implementation
// ============================================================================

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn provider(&self) -> flashmind_types::Provider {
        flashmind_types::Provider::Ollama
    }

    async fn context_window(&self, model: &flashmind_types::Model) -> Option<u32> {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        // Check cache first
        if let Ok(cache) = self.context_windows.lock()
            && let Some(&size) = cache.get(lookup_name)
        {
            return Some(size);
        }

        // Query /api/show asynchronously
        let show = query_show(&self.client, self.url("/api/show"), lookup_name).await?;
        let ctx_size = extract_context_window(&show)?;

        // Cache the result
        if let Ok(mut cache) = self.context_windows.lock() {
            cache.insert(lookup_name.to_string(), ctx_size);
        }

        Some(ctx_size)
    }

    async fn capabilities(&self, model: &flashmind_types::Model) -> ModelCapabilities {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        let Some(show) = query_show(&self.client, self.url("/api/show"), lookup_name).await else {
            // Can't reach Ollama — assume capable (matches old supports_tools behavior)
            return ModelCapabilities {
                tool_calling: true,
                images: false,
                documents: false,
                video: false,
                audio: false,
                reasoning: false,
                ..Default::default()
            };
        };

        let tool_calling = show.capabilities.iter().any(|c| c == "tools");
        let has_reasoning = show.capabilities.iter().any(|c| c == "thinking");

        // Vision: check capabilities array first, then fall back to model_info metadata.
        // Some models (custom Modelfiles, community quants, older Ollama versions) may not
        // report "vision" in capabilities even though they have a vision projector/encoder.
        let images = show.capabilities.iter().any(|c| c == "vision") || has_vision_metadata(&show);

        ModelCapabilities {
            tool_calling,
            images,
            documents: false,
            video: false,
            audio: false,
            reasoning: has_reasoning,
            ..Default::default()
        }
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = self.client.clone();
        let chat_url = self.url("/api/chat");
        let num_ctx = self.num_ctx;

        Box::pin(stream! {
            let start = std::time::Instant::now();
            metrics::counter!("llm.requests.started").increment(1);
            tracing::debug!(
                model = %request.model,
                messages = request.messages.len(),
                tools = request.tools.len(),
                "Sending Ollama native completion request"
            );

            let native_request = build_native_request(request, num_ctx);

            if let Ok(json) = serde_json::to_string(&native_request) {
                let body_bytes = json.len();
                if body_bytes > 100_000 {
                    tracing::warn!(
                        body_kb = body_bytes / 1024,
                        messages = native_request.messages.len(),
                        tools = native_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                        "Large Ollama request payload — may cause parse errors on small models"
                    );
                }
            }

            let response = match send_with_retry_helper(&client, &chat_url, &native_request).await {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    yield Err(e);
                    return;
                }
            };

            // Process NDJSON stream: each line is a complete JSON object
            let byte_stream = response
                .bytes_stream()
                .map_err(std::io::Error::other);
            let reader = StreamReader::new(byte_stream);
            let mut lines = reader.lines();

            let mut finish_reason = FinishReason::Stop;
            let mut tool_call_index: usize = 0;

            loop {
                let line = match lines.next_line().await {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        tracing::debug!(error = %e, "NDJSON read error");
                        break;
                    }
                };

                if line.trim().is_empty() {
                    continue;
                }

                // Check for Ollama error responses (no `done` field)
                if let Ok(err_obj) = serde_json::from_str::<serde_json::Value>(&line)
                    && let Some(error) = err_obj.get("error").and_then(|e| e.as_str()) {
                        tracing::error!(
                            ollama_error = %error,
                            message_count = native_request.messages.len(),
                            "Ollama returned error"
                        );
                        for (i, msg) in native_request.messages.iter().enumerate() {
                            let tc_summary = msg.tool_calls.as_ref().map(|tcs| {
                                tcs.iter()
                                    .map(|tc| format!("{}({})", tc.function.name, tc.function.arguments))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            });
                            let preview = crate::truncate_utf8_line(&msg.content, 200);
                            tracing::error!(
                                idx = i,
                                role = %msg.role,
                                content_preview = %preview,
                                tool_calls = ?tc_summary,
                                "Request message dump"
                            );
                        }
                        if let Ok(request_json) = serde_json::to_string(&native_request) {
                            tracing::error!(
                                request_size = request_json.len(),
                                request_body = crate::truncate_utf8_line(&request_json, 120),
                                "Full Ollama request body"
                            );
                        }
                        yield Err(flashmind_types::LlmError::classify(
                            format!("Ollama error: {error}"),
                        ).into());
                        return;
                    }

                let chunk: NativeChunk = match serde_json::from_str(&line) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!(error = %e, data = %line, "Failed to parse NDJSON chunk");
                        continue;
                    }
                };

                // Process message content from this chunk
                if let Some(ref message) = chunk.message {
                    if let Some(ref thinking) = message.thinking
                        && !thinking.is_empty() {
                            yield Ok(StreamEvent::ReasoningDelta(thinking.clone()));
                        }

                    if let Some(ref content) = message.content
                        && !content.is_empty() {
                            yield Ok(StreamEvent::ContentDelta(content.clone()));
                        }

                    if let Some(ref tool_calls) = message.tool_calls {
                        for tc in tool_calls {
                            let id = format!("call_{}", uuid::Uuid::new_v4());
                            let args_str = tc.function.arguments.to_string();

                            yield Ok(StreamEvent::ToolCallStart {
                                index: tool_call_index,
                                id,
                                name: tc.function.name.clone(),
                            });
                            yield Ok(StreamEvent::ToolCallDelta {
                                index: tool_call_index,
                                arguments: args_str,
                            });
                            tool_call_index += 1;
                            finish_reason = FinishReason::ToolCalls;
                        }
                    }
                }

                if chunk.done {
                    let prompt_tokens = chunk.prompt_eval_count.unwrap_or(0);
                    let completion_tokens = chunk.eval_count.unwrap_or(0);

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
                    break;
                }
            }

            metrics::counter!("llm.finish_reason", "reason" => finish_reason.to_string()).increment(1);
            yield Ok(StreamEvent::Finished(finish_reason));
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
        })
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<TagEntry>,
        }
        #[derive(Deserialize)]
        struct TagEntry {
            #[serde(alias = "model")]
            name: String,
        }

        let url = self.url("/api/tags");
        let resp = self.client.get(url.as_str()).send().await.ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let tags: TagsResponse = resp.json().await.ok()?;

        let mut models = Vec::with_capacity(tags.models.len());
        for entry in tags.models {
            let show = query_show(&self.client, self.url("/api/show"), &entry.name).await;

            let (context_length, capabilities) = match &show {
                Some(s) => {
                    let ctx = extract_context_window(s);
                    let tool_calling = s.capabilities.iter().any(|c| c == "tools");
                    let has_reasoning = s.capabilities.iter().any(|c| c == "thinking");
                    let images =
                        s.capabilities.iter().any(|c| c == "vision") || has_vision_metadata(s);
                    (
                        ctx,
                        ModelCapabilities {
                            tool_calling,
                            images,
                            reasoning: has_reasoning,
                            ..Default::default()
                        },
                    )
                }
                None => (None, ModelCapabilities::default()),
            };

            let categories = capabilities.categories();

            models.push(ModelInfo {
                id: entry.name,
                name: None,
                context_length,
                max_completion_tokens: None,
                capabilities,
                categories,
                pricing: ModelPricing::default(),
            });
        }

        Some(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_default_url() {
        let provider = OllamaProvider::new(None, None).unwrap();
        assert_eq!(provider.base_url.as_str(), "http://localhost:11434/");
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_ollama_custom_url() {
        let provider =
            OllamaProvider::new(Some("http://192.168.1.100:11434".into()), None).unwrap();
        assert_eq!(provider.base_url.as_str(), "http://192.168.1.100:11434/");
    }

    #[test]
    fn test_ollama_with_num_ctx() {
        let provider = OllamaProvider::new(None, Some(128000)).unwrap();
        assert_eq!(provider.num_ctx, Some(128000));
    }
}
