//! Ollama LLM provider for running local models.
//!
//! Uses Ollama's native `/api/chat` endpoint with NDJSON streaming.
//!
//! <https://github.com/ollama/ollama/blob/main/docs/api.md>
//! Supports the `think` parameter for reasoning/thinking mode and native
//! tool call format.

mod convert;
mod wire_types;

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

use crate::http::{http_client_builder, send_with_retry};
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities, StreamEvent,
    TokenUsage,
};

use convert::*;
use wire_types::*;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434/";

/// Ollama provider using native `/api/chat` with NDJSON streaming.
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
    pub fn new(base_url: Option<String>, num_ctx: Option<u32>) -> Self {
        let raw = base_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.into());
        let base_url =
            Url::parse(&raw).unwrap_or_else(|e| panic!("Invalid Ollama URL '{}': {}", raw, e));
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            base_url,
            context_windows: Arc::new(Mutex::new(HashMap::new())),
            num_ctx,
        }
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
        anyhow::bail!("Ollama native API error {}: {}", status, text);
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
                        yield Err(anyhow::anyhow!("Ollama error: {error}"));
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

            metrics::counter!("llm.finish_reason", "reason" => match finish_reason {
                FinishReason::Stop => "stop",
                FinishReason::ToolCalls => "tool_calls",
                FinishReason::Length => "length",
                FinishReason::ContentFilter => "content_filter",
            }).increment(1);
            yield Ok(StreamEvent::Finished(finish_reason));
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_default_url() {
        let provider = OllamaProvider::new(None, None);
        assert_eq!(provider.base_url.as_str(), "http://localhost:11434/");
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_ollama_custom_url() {
        let provider = OllamaProvider::new(Some("http://192.168.1.100:11434".into()), None);
        assert_eq!(provider.base_url.as_str(), "http://192.168.1.100:11434/");
    }

    #[test]
    fn test_ollama_with_num_ctx() {
        let provider = OllamaProvider::new(None, Some(128000));
        assert_eq!(provider.num_ctx, Some(128000));
    }
}
