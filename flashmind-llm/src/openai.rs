//! OpenAI-compatible provider — works with OpenAI, vLLM, and any API
//! following the OpenAI chat completions format.
//!
//! <https://platform.openai.com/docs/api-reference/chat>
//!
//! Reuses wire_types and SSE parsing from the OpenRouter/OpenAI-compatible stack.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use flate2::Compression;
use flate2::write::GzEncoder;
use metrics;
use reqwest::Client;
use serde::Deserialize;
use tokio_stream::StreamExt;
use url::Url;

use crate::http::{http_client_builder, send_with_retry, wait_for_rate_limit};
use crate::sse::{ToolCallTracker, process_chunk};
use crate::wire_types::{
    ApiContent, ApiRequestBase, ChatTemplateKwargs, StreamChunk, StreamOptions, to_api_messages,
    to_api_tools,
};
use crate::{ContextWindowCache, oss_capabilities};
use flashmind_types::model::Provider;
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities, StreamEvent,
};
use ratelimit::Ratelimiter;

const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/";

pub type RoutingTable = Arc<RwLock<HashMap<String, Url>>>;

/// OpenAI-compatible provider with SSE streaming.
///
/// Works with the OpenAI API, vLLM, and any service implementing the
/// OpenAI `/v1/chat/completions` and `/v1/models` endpoints.
pub struct OpenAiProvider {
    client: Client,
    base_url: Url,
    api_key: Option<String>,
    ctx_cache: Arc<ContextWindowCache>,
    /// Per-model URL overrides. Checked per-request; falls back to `base_url`.
    routing: RoutingTable,
    /// Whether to gzip request bodies.
    compression: bool,
    /// Request rate limiter.
    rate_limiter: Option<Arc<Ratelimiter>>,
}

/// Entry from /v1/models response (OpenAI-compatible).
#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    max_model_len: Option<u32>,
}

impl OpenAiProvider {
    pub fn new(
        base_url: Option<String>,
        api_key: Option<String>,
        routing: RoutingTable,
        compression: bool,
        rate_limiter: Option<Arc<Ratelimiter>>,
    ) -> Self {
        let raw = base_url.unwrap_or_else(|| DEFAULT_OPENAI_URL.into());
        let base_url =
            Url::parse(&raw).unwrap_or_else(|e| panic!("Invalid OpenAI URL '{}': {}", raw, e));
        let mut client_builder = http_client_builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120));

        if compression {
            client_builder = client_builder.gzip(true);
        }

        let client = client_builder.build().expect("Failed to build HTTP client");
        Self {
            client,
            base_url,
            api_key,
            ctx_cache: Arc::new(ContextWindowCache::new()),
            routing,
            compression,
            rate_limiter,
        }
    }

    /// Build a full URL by appending a path to the base URL.
    /// Uses the routing table to resolve model-specific URLs, falling back to `base_url`.
    fn url_for_model(&self, path: &str, model: &str) -> Url {
        let base = {
            let table = self.routing.read().unwrap();
            table.get(model).cloned()
        }
        .unwrap_or_else(|| self.base_url.clone());

        let mut url = base;
        url.set_path(path);
        url
    }

    /// Build a full URL by appending a path to the base URL.
    fn url(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url
    }

    /// Build a request with optional Bearer auth.
    fn authed_request(&self, method: reqwest::Method, url: Url) -> reqwest::RequestBuilder {
        let mut req = self.client.request(method, url.as_str());
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        req
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    async fn context_window(&self, model: &flashmind_types::Model) -> Option<u32> {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        if let Some(size) = self.ctx_cache.get(lookup_name) {
            return Some(size);
        }

        // Use routing-aware URL so routed models query the correct server
        let url = self.url_for_model("v1/models", model.name());
        let resp = match send_with_retry(|| self.authed_request(reqwest::Method::GET, url.clone()))
            .await
        {
            Ok(r) => r,
            Err(_) => return None,
        };
        if !resp.status().is_success() {
            return None;
        }

        let models: ModelsResponse = resp.json().await.ok()?;
        for entry in &models.data {
            if let Some(len) = entry.max_model_len {
                self.ctx_cache.set(&entry.id, len);
            }
        }

        self.ctx_cache
            .get(lookup_name)
            .or_else(|| self.ctx_cache.get(model.name()))
    }

    async fn capabilities(&self, model: &flashmind_types::Model) -> ModelCapabilities {
        // Use real_name for capability lookups if alias is set
        let lookup_name = model.capability_name();

        // First check our registry of known open-source models
        if let Some(caps) = oss_capabilities::get_oss_capabilities(lookup_name) {
            return caps;
        }

        // Fallback for unknown models: assume no vision/media capabilities.
        // Most custom models on OpenAI-compatible endpoints (vLLM, etc.) are text-only.
        // Known vision models should be in the OSS capabilities registry.
        ModelCapabilities {
            tool_calling: true,
            images: false,
            reasoning: false,
            documents: false,
            video: false,
            audio: false,
            ..Default::default()
        }
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = self.client.clone();
        let url = self.url_for_model("v1/chat/completions", request.model.name());
        let api_key = self.api_key.clone();
        let rate_limiter = self.rate_limiter.clone();
        let provider_str = self.provider().to_string();
        let compression = self.compression;

        Box::pin(stream! {
            let start = std::time::Instant::now();
            metrics::counter!("llm.requests.started").increment(1);
            let mut messages = to_api_messages(&request.messages);
            let tools = to_api_tools(request.tools.clone());

            // OpenAI's API rejects unknown fields (top_k, min_p, repetition_penalty).
            // Only strip them when targeting api.openai.com — compatible endpoints
            // like vLLM accept all fields.
            let is_openai = url.host_str() == Some("api.openai.com");

            // vLLM and other compatible endpoints may not support the developer role;
            // fall back to user wrapped in <system> tags for non-OpenAI targets.
            if !is_openai {
                for msg in &mut messages {
                    if msg.role == "developer" {
                        msg.role = "user".into();
                        if let ApiContent::Text(ref mut text) = msg.content {
                            *text = format!("<system>\n{text}\n</system>");
                        }
                    }
                }
            }

            // Use capability_name for model lookup (handles aliases via real_name)
            let model_name = request.model.capability_name();
            // gemma-4 models require skip_special_tokens=false when thinking is enabled.
            let is_gemma_4 = model_name.contains("gemma-4");
            let include_special_tokens = is_gemma_4 && request.reasoning.is_on();

            let api_request = ApiRequestBase {
                model: request.model.name().to_string(),
                messages,
                tool_choice: if tools.is_empty() { None } else { Some("auto".into()) },
                tools,
                temperature: request.temperature,
                max_tokens: request.max_tokens,
                top_p: request.sampling.top_p,
                top_k: if is_openai { None } else { request.sampling.top_k },
                min_p: if is_openai { None } else { request.sampling.min_p },
                presence_penalty: request.sampling.presence_penalty,
                repetition_penalty: if is_openai { None } else { request.sampling.repetition_penalty },
                stream: true,
                stream_options: StreamOptions::default(),
                skip_special_tokens: if include_special_tokens { Some(false) } else { None },
                chat_template_kwargs: ChatTemplateKwargs { enable_thinking: request.reasoning.is_on() },
                parallel_tool_calls: true,
                modalities: request.modalities.map(|ms| {
                    ms.iter().map(|m| match m {
                        flashmind_types::Modality::Text => "text".into(),
                        flashmind_types::Modality::Audio => "audio".into(),
                        flashmind_types::Modality::Image => "image".into(),
                    }).collect()
                }),
                audio: request.audio_config.map(|c| crate::wire_types::ApiAudioConfig {
                    voice: c.voice,
                    format: c.format.to_string(),
                }),
                image_config: request.image_config.map(|c| crate::wire_types::ApiImageConfig {
                    aspect_ratio: c.aspect_ratio,
                    size: c.size,
                    super_resolution_references: c.reference_images,
                }),
            };

            tracing::debug!(model = %request.model, url = %url, "Sending OpenAI completion request");

            let mut req = if compression {
                let json_bytes = match serde_json::to_vec(&api_request) {
                    Ok(b) => b,
                    Err(e) => {
                        metrics::counter!("llm.requests.errors").increment(1);
                        metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                        yield Err(anyhow::anyhow!("{} error: Failed to serialize request: {}", provider_str, e));
                        return;
                    }
                };

                let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
                if let Err(e) = encoder.write_all(&json_bytes) {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    yield Err(anyhow::anyhow!("{} error: Failed to compress request: {}", provider_str, e));
                    return;
                }
                let compressed = match encoder.finish() {
                    Ok(b) => b,
                    Err(e) => {
                        metrics::counter!("llm.requests.errors").increment(1);
                        metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                        yield Err(anyhow::anyhow!("{} error: Failed to finish compression: {}", provider_str, e));
                        return;
                    }
                };

                tracing::debug!(
                    original = json_bytes.len(),
                    compressed = compressed.len(),
                    ratio = format!("{:.1}%", (compressed.len() as f64 / json_bytes.len() as f64) * 100.0),
                    "Compressed request body"
                );

                client
                    .post(url.as_str())
                    .header("Content-Encoding", "gzip")
                    .header("Content-Type", "application/json")
                    .body(compressed)
            } else {
                client
                    .post(url.as_str())
                    .json(&api_request)
            };

            if let Some(ref key) = api_key {
                req = req.bearer_auth(key);
            }

            // Wait for rate limiter before sending (if configured)
            if let Some(ref limiter) = rate_limiter {
                wait_for_rate_limit(limiter).await;
            }

            let response = match send_with_retry(|| req.try_clone().expect("cloneable request")).await {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    yield Err(anyhow::anyhow!("{} error: OpenAI request failed: {}", provider_str, e));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!(
                    "{} error: OpenAI API error {}: {}",
                    provider_str, status, body
                ));
                metrics::counter!("llm.requests.errors").increment(1);
                metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                return;
            }

            let mut stream = response.bytes_stream().eventsource();
            let mut tracker = ToolCallTracker::default();
            let mut finish_reason = FinishReason::Stop;
            let mut consecutive_errors: u32 = 0;

            while let Some(event) = stream.next().await {
                let event = match event {
                    Ok(e) => {
                        consecutive_errors = 0;
                        e
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        tracing::debug!(error = %e, consecutive_errors, "OpenAI SSE error");

                        if consecutive_errors >= 5 {
                            tracing::warn!("Too many consecutive SSE errors, aborting stream");
                            break;
                        }

                        continue;
                    }
                };

                if event.data == "[DONE]" {
                    break;
                }

                let chunk: StreamChunk = match serde_json::from_str(&event.data) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!(error = %e, data = %event.data, "Failed to parse OpenAI chunk");
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

            metrics::counter!("llm.finish_reason", "reason" => finish_reason.to_string()).increment(1);
            yield Ok(StreamEvent::Finished(finish_reason));
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
        })
    }

    fn update_routing(&self, routing: &HashMap<String, String>) {
        let new_map: HashMap<String, Url> = routing
            .iter()
            .filter_map(|(model, url_str)| {
                Url::parse(url_str)
                    .map(|url| (model.clone(), url))
                    .map_err(|e| tracing::warn!("Invalid routing URL for model '{}': {}", model, e))
                    .ok()
            })
            .collect();

        let mut table = self.routing.write().unwrap();
        *table = new_map;
    }

    async fn list_voices(&self, _model: &str) -> Option<Vec<flashmind_types::Voice>> {
        let url = self.url("/v1/audio/voices");

        let mut req = self.client.get(url.as_str());
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        #[derive(Deserialize)]
        struct VoicesResponse {
            voices: Vec<VoiceEntry>,
        }
        #[derive(Deserialize)]
        struct VoiceEntry {
            #[serde(alias = "voice_id")]
            id: String,
            name: Option<String>,
        }

        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(parsed) = resp.json::<VoicesResponse>().await {
                    return Some(
                        parsed
                            .voices
                            .into_iter()
                            .map(|v| flashmind_types::Voice {
                                name: v.name.unwrap_or_else(|| v.id.clone()),
                                id: v.id,
                            })
                            .collect(),
                    );
                }
            }
            _ => {}
        }

        // Fallback: static OpenAI voices
        Some(
            [
                "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage",
                "shimmer",
            ]
            .into_iter()
            .map(|v| flashmind_types::Voice {
                id: v.to_string(),
                name: v.to_string(),
            })
            .collect(),
        )
    }

    async fn text_to_speech(
        &self,
        request: flashmind_types::TtsRequest,
    ) -> anyhow::Result<Vec<u8>> {
        let url = self.url("/v1/audio/speech");

        let body = serde_json::json!({
            "model": request.model,
            "input": request.input,
            "voice": request.voice,
            "response_format": request.response_format.to_string(),
        });

        let mut req = self.client.post(url.as_str()).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("TTS request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("TTS API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read TTS response: {}", e))?;

        Ok(bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = OpenAiProvider::new(None, None, RoutingTable::default(), false, None);
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_default_url() {
        let provider = OpenAiProvider::new(None, None, RoutingTable::default(), false, None);
        assert_eq!(provider.base_url.as_str(), DEFAULT_OPENAI_URL);
    }

    #[test]
    fn test_custom_url() {
        let provider = OpenAiProvider::new(
            Some("http://myhost:9000/".into()),
            None,
            RoutingTable::default(),
            false,
            None,
        );
        assert_eq!(provider.base_url.as_str(), "http://myhost:9000/");
    }

    #[test]
    fn test_api_key_stored() {
        let provider = OpenAiProvider::new(
            None,
            Some("sk-test-key".into()),
            RoutingTable::default(),
            false,
            None,
        );
        assert_eq!(provider.api_key.as_deref(), Some("sk-test-key"));
    }
}
