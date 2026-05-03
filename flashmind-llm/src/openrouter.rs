//! OpenRouter LLM provider implementation.
//!
//! Routes requests to various LLM backends (OpenAI, Anthropic, Google, etc.) via a unified API.
//! Supports SSE streaming, tool calling, reasoning tokens, TTS, and auto-discovery of model
//! capabilities from the `/api/v1/models` endpoint.
//!
//! See <https://openrouter.ai/docs/api-reference> for the API reference.

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
    ApiAudioConfig, ApiImageConfig, ApiImageUrl, ApiMessage, ApiSamplingParams, ApiTool,
    ChatTemplateKwargs, StreamChunk, StreamOptions, to_api_messages, to_api_tools,
};
use crate::{ContextWindowCache, oss_capabilities};
use flashmind_types::model::Provider;
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities,
    ModelCategory, ModelInfo, ModelPricing, StreamEvent,
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
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    architecture: Option<ModelArchitecture>,
    #[serde(default)]
    supported_parameters: Vec<String>,
    #[serde(default)]
    pricing: Option<ModelPricingEntry>,
    #[serde(default)]
    top_provider: Option<TopProvider>,
}

#[derive(Deserialize)]
struct ModelArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
}

#[derive(Deserialize, Default)]
struct ModelPricingEntry {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
}

#[derive(Deserialize)]
struct TopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u32>,
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
        let input = entry
            .architecture
            .as_ref()
            .map(|a| &a.input_modalities[..])
            .unwrap_or_default();

        let output = entry
            .architecture
            .as_ref()
            .map(|a| &a.output_modalities[..])
            .unwrap_or_default();

        let reasoning = entry
            .supported_parameters
            .iter()
            .any(|p| p == "include_reasoning" || p == "reasoning");

        ModelCapabilities {
            tool_calling: true,
            images: input.iter().any(|m| m == "image"),
            documents: input.iter().any(|m| m == "file"),
            video: input.iter().any(|m| m == "video"),
            audio: input.iter().any(|m| m == "audio"),
            reasoning,
            audio_output: output.iter().any(|m| m == "audio"),
            image_generation: output.iter().any(|m| m == "image"),
            video_generation: output.iter().any(|m| m == "video"),
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
///
/// Sent in the `reasoning` field of the [OpenRouter API](https://openrouter.ai/docs/requests).
#[derive(Serialize)]
struct ApiReasoning {
    effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// OpenRouter-specific request body.
///
/// Wraps the standard OpenAI-compatible fields with OpenRouter extensions:
/// reasoning config, chat template kwargs, image/audio modality support.
/// Serialised as the JSON body of `POST /v1/chat/completions`.
///
/// See <https://openrouter.ai/docs/requests> for the full schema.
#[derive(Serialize)]
struct ApiRequest {
    /// Model identifier (e.g. `"anthropic/claude-sonnet-4"`).
    model: String,
    /// Message history in wire format.
    messages: Vec<ApiMessage>,
    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    /// Sampling params flattened into the top-level object.
    #[serde(flatten)]
    sampling: ApiSamplingParams,
    stream: bool,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_special_tokens: Option<bool>,
    chat_template_kwargs: ChatTemplateKwargs,
    parallel_tool_calls: bool,
    /// Requested output modalities (`text`, `audio`, `image`).
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<ApiAudioConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_config: Option<ApiImageConfig>,
    /// Extended reasoning/thinking mode config (OpenRouter extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
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

    let has_image_output = request
        .modalities
        .contains(&flashmind_types::Modality::Image);

    let sampling = if has_image_output {
        ApiSamplingParams::default()
    } else {
        ApiSamplingParams {
            temperature: request.sampling.temperature,
            max_tokens: request.max_tokens,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            min_p: request.sampling.min_p,
            presence_penalty,
            repetition_penalty: request.sampling.repetition_penalty,
        }
    };

    ApiRequest {
        model: request.model.name().to_string(),
        messages: to_api_messages(&request.messages),
        tool_choice: if tools.is_empty() {
            None
        } else {
            Some("auto".into())
        },
        tools,
        sampling,
        stream: true,
        stream_options: StreamOptions::default(),
        skip_special_tokens: if include_special_tokens && !has_image_output {
            Some(false)
        } else {
            None
        },
        chat_template_kwargs: ChatTemplateKwargs {
            enable_thinking: if has_image_output {
                false
            } else {
                request.reasoning.is_on()
            },
        },
        parallel_tool_calls: !has_image_output,
        modalities: if request.modalities.is_empty() {
            None
        } else {
            Some(request.modalities.iter().map(|m| m.to_string()).collect())
        },
        audio: request.audio_config.map(|c| ApiAudioConfig {
            voice: c.voice,
            format: c.format.to_string(),
        }),
        image_config: request.image_config.map(|c| ApiImageConfig {
            aspect_ratio: c.aspect_ratio,
            size: c.size,
            super_resolution_references: vec![],
        }),
        reasoning: if include_reasoning && !has_image_output {
            Some(ApiReasoning {
                effort: "low".to_string(),
                max_tokens: None,
            })
        } else {
            None
        },
        include_reasoning: include_reasoning && !has_image_output,
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
                ..Default::default()
            })
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        let entries = self.fetch_model_entries().await?;

        let mut models: Vec<ModelInfo> = entries
            .iter()
            .map(|entry| {
                let capabilities = Self::entry_capabilities(entry);
                let categories = capabilities.categories();
                let pricing = entry
                    .pricing
                    .as_ref()
                    .map(|p| ModelPricing {
                        prompt: p.prompt.as_deref().and_then(|s| s.parse().ok()),
                        completion: p.completion.as_deref().and_then(|s| s.parse().ok()),
                        image: p.image.as_deref().and_then(|s| s.parse().ok()),
                        cache_read: p.input_cache_read.as_deref().and_then(|s| s.parse().ok()),
                    })
                    .unwrap_or_default();
                let max_completion_tokens = entry
                    .top_provider
                    .as_ref()
                    .and_then(|tp| tp.max_completion_tokens);

                ModelInfo {
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                    context_length: entry.context_length,
                    max_completion_tokens,
                    capabilities,
                    categories,
                    pricing,
                }
            })
            .collect();

        let chat_ids: std::collections::HashSet<String> =
            models.iter().map(|m| m.id.clone()).collect();

        if let Ok(image_models) = self.list_image_models().await {
            for m in image_models {
                if !chat_ids.contains(&m.id) {
                    models.push(ModelInfo {
                        id: m.id,
                        name: m.name,
                        context_length: None,
                        max_completion_tokens: None,
                        capabilities: ModelCapabilities {
                            image_generation: true,
                            ..Default::default()
                        },
                        categories: vec![ModelCategory::ImageGeneration],
                        pricing: ModelPricing::default(),
                    });
                }
            }
        }

        if let Ok(video_models) = self.list_video_models().await {
            for m in video_models {
                if !chat_ids.contains(&m.id) {
                    models.push(ModelInfo {
                        id: m.id,
                        name: m.name,
                        context_length: None,
                        max_completion_tokens: None,
                        capabilities: ModelCapabilities {
                            video_generation: true,
                            ..Default::default()
                        },
                        categories: vec![ModelCategory::VideoGeneration],
                        pricing: ModelPricing::default(),
                    });
                }
            }
        }

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
                    .header("X-Title", "Flash")
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

    async fn list_voices(&self, _model: &str) -> Option<Vec<flashmind_types::Voice>> {
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

        let resp = self
            .client
            .get("https://openrouter.ai/api/v1/audio/voices")
            .bearer_auth(&self.api_key)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let parsed: VoicesResponse = resp.json().await.ok()?;
        Some(
            parsed
                .voices
                .into_iter()
                .map(|v| flashmind_types::Voice {
                    name: v.name.unwrap_or_else(|| v.id.clone()),
                    id: v.id,
                })
                .collect(),
        )
    }

    fn text_to_speech(&self, request: flashmind_types::TtsRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let format = request.response_format.to_string();

        Box::pin(stream! {
            let body = serde_json::json!({
                "model": request.model,
                "input": request.input,
                "voice": request.voice,
                "response_format": &format,
            });

            let response = match client
                .post("https://openrouter.ai/api/v1/audio/speech")
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("TTS request failed: {e}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("TTS API error ({status}): {text}"));
                return;
            }

            use base64::Engine;
            let mut byte_stream = response.bytes_stream();
            while let Some(chunk) = tokio_stream::StreamExt::next(&mut byte_stream).await {
                match chunk {
                    Ok(bytes) => {
                        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        yield Ok(StreamEvent::AudioDelta { data, format: format.clone() });
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("TTS stream error: {e}"));
                        return;
                    }
                }
            }
            yield Ok(StreamEvent::Finished(FinishReason::Stop));
        })
    }

    fn transcribe(&self, request: flashmind_types::SttRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();

        Box::pin(stream! {
            let ext = request.media_type.split('/').next_back().unwrap_or("mp3");
            let filename = format!("audio.{ext}");

            let part = reqwest::multipart::Part::bytes(request.audio)
                .file_name(filename)
                .mime_str(&request.media_type)
                .unwrap();

            let mut form = reqwest::multipart::Form::new()
                .text("model", request.model)
                .part("file", part);

            if let Some(lang) = request.language {
                form = form.text("language", lang);
            }

            let response = match client
                .post("https://openrouter.ai/api/v1/audio/transcriptions")
                .bearer_auth(&api_key)
                .multipart(form)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("STT request failed: {e}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("STT API error ({status}): {text}"));
                return;
            }

            #[derive(Deserialize)]
            struct TranscriptionResponse {
                text: String,
            }

            match response.json::<TranscriptionResponse>().await {
                Ok(resp) => {
                    yield Ok(StreamEvent::ContentDelta(resp.text));
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to parse transcription response: {e}"));
                    return;
                }
            }
            yield Ok(StreamEvent::Finished(FinishReason::Stop));
        })
    }

    fn generate_image(&self, request: flashmind_types::ImageGenRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let rate_limiter = self.rate_limiter.clone();

        Box::pin(async_stream::stream! {
            #[derive(Serialize)]
            struct ApiImageGenRequest {
                model: String,
                prompt: String,
                #[serde(skip_serializing_if = "Option::is_none")]
                size: Option<String>,
                #[serde(skip_serializing_if = "Option::is_none")]
                quality: Option<String>,
                #[serde(skip_serializing_if = "Option::is_none")]
                style: Option<String>,
                #[serde(skip_serializing_if = "Option::is_none")]
                n: Option<u32>,
                response_format: String,
            }

            let api_request = ApiImageGenRequest {
                model: request.model,
                prompt: request.prompt,
                size: request.size,
                quality: request.quality,
                style: request.style,
                n: request.n,
                response_format: "b64_json".into(),
            };

            let mut req = client
                .post("https://openrouter.ai/api/v1/images/generations")
                .json(&api_request)
                .header("HTTP-Referer", "https://github.com/flashmind-labs/agent")
                .header("X-Title", "Flash");
            req = req.bearer_auth(&api_key);

            wait_for_rate_limit(&rate_limiter).await;

            let response = match send_with_retry(|| req.try_clone().expect("cloneable request")).await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Image generation request failed: {e}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("Image generation failed ({status}): {body}"));
                return;
            }

            #[derive(Deserialize)]
            struct ImageResponse {
                data: Vec<ImageData>,
            }
            #[derive(Deserialize)]
            struct ImageData {
                b64_json: String,
                #[serde(default)]
                revised_prompt: Option<String>,
            }

            let parsed: ImageResponse = match response.json().await {
                Ok(p) => p,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to parse image response: {e}"));
                    return;
                }
            };

            for img in parsed.data {
                if let Some(prompt) = &img.revised_prompt {
                    yield Ok(StreamEvent::ContentDelta(format!("Revised prompt: {prompt}\n")));
                }
                use base64::Engine;
                let bytes = match base64::engine::general_purpose::STANDARD.decode(&img.b64_json) {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Failed to decode image data: {e}"));
                        return;
                    }
                };
                yield Ok(StreamEvent::FileAttachment {
                    filename: String::new(),
                    media_type: "image/png".into(),
                    data: bytes,
                });
            }
            yield Ok(StreamEvent::Finished(FinishReason::Stop));
        })
    }

    fn generate_video(&self, request: flashmind_types::VideoGenRequest) -> CompletionStream {
        self.generate_video_impl(request)
    }
}

// ============================================================================
// Video Generation — wire types for the OpenRouter videos API
// ============================================================================

#[derive(Serialize)]
struct ApiVideoGenRequest {
    model: String,
    #[serde(rename = "prompt")]
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    frame_images: Vec<ApiFrameImage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_references: Vec<ApiInputReference>,
}

#[derive(Serialize)]
struct ApiFrameImage {
    #[serde(rename = "type")]
    content_type: String,
    image_url: ApiImageUrl,
    frame_type: String,
}

#[derive(Serialize)]
struct ApiInputReference {
    #[serde(rename = "type")]
    content_type: String,
    image_url: ApiImageUrl,
}

impl From<flashmind_types::VideoGenRequest> for ApiVideoGenRequest {
    fn from(r: flashmind_types::VideoGenRequest) -> Self {
        Self {
            model: r.model,
            description: r.description,
            resolution: r.resolution,
            aspect_ratio: r.aspect_ratio,
            duration: r.duration,
            generate_audio: r.generate_audio,
            frame_images: r
                .frame_images
                .into_iter()
                .map(|f| ApiFrameImage {
                    content_type: "image_url".into(),
                    image_url: ApiImageUrl { url: f.url },
                    frame_type: f.frame_type,
                })
                .collect(),
            input_references: r
                .input_references
                .into_iter()
                .map(|url| ApiInputReference {
                    content_type: "image_url".into(),
                    image_url: ApiImageUrl { url },
                })
                .collect(),
        }
    }
}

const OPENROUTER_VIDEOS_URL: &str = "https://openrouter.ai/api/v1/videos";

impl OpenRouterProvider {
    fn generate_video_impl(&self, request: flashmind_types::VideoGenRequest) -> CompletionStream {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let body = ApiVideoGenRequest::from(request);

        Box::pin(stream! {

            yield Ok(StreamEvent::ContentDelta("Submitting video generation job...\n".into()));

            let resp = match client
                .post(OPENROUTER_VIDEOS_URL)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Video generation request failed: {e}"));
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("Video generation API error ({status}): {text}"));
                return;
            }

            #[derive(Deserialize)]
            struct SubmitResponse {
                id: Option<String>,
                error: Option<serde_json::Value>,
            }

            let text = resp.text().await.unwrap_or_default();
            tracing::debug!("Video submit response body: {text}");
            let job: SubmitResponse = match serde_json::from_str(&text) {
                Ok(j) => j,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to parse video job response: {e}\nBody: {text}"));
                    return;
                }
            };

            if let Some(err) = job.error {
                let msg = err.get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                yield Err(anyhow::anyhow!("Video generation API error: {msg}"));
                return;
            }

            let job_id = match job.id {
                Some(id) => id,
                None => {
                    yield Err(anyhow::anyhow!("Video generation response missing job ID. Body: {text}"));
                    return;
                }
            };

            yield Ok(StreamEvent::ContentDelta(format!("Video job submitted: {}\n", job_id)));

            // Poll with exponential backoff
            let mut delay = Duration::from_secs(5);
            let max_delay = Duration::from_secs(30);
            let deadline = tokio::time::Instant::now() + Duration::from_secs(600);

            loop {
                tokio::time::sleep(delay).await;

                if tokio::time::Instant::now() > deadline {
                    yield Err(anyhow::anyhow!("Video generation timed out after 10 minutes"));
                    return;
                }

                let poll_url = format!("{}/{}", OPENROUTER_VIDEOS_URL, job_id);
                let poll_resp = match client
                    .get(&poll_url)
                    .bearer_auth(&api_key)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Video poll failed: {e}");
                        delay = (delay * 2).min(max_delay);
                        continue;
                    }
                };

                if !poll_resp.status().is_success() {
                    tracing::warn!("Video poll returned status {}", poll_resp.status());
                    delay = (delay * 2).min(max_delay);
                    continue;
                }

                let poll_body = match poll_resp.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("Failed to read poll response: {e}");
                        delay = (delay * 2).min(max_delay);
                        continue;
                    }
                };

                tracing::debug!("Video poll response: {poll_body}");

                #[derive(Deserialize)]
                struct PollResponse {
                    id: String,
                    status: String,
                    #[serde(default)]
                    unsigned_urls: Vec<String>,
                    #[serde(default)]
                    error: Option<String>,
                }

                let poll: PollResponse = match serde_json::from_str(&poll_body) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("Failed to parse poll response: {e}");
                        delay = (delay * 2).min(max_delay);
                        continue;
                    }
                };

                match poll.status.as_str() {
                    "pending" => {
                        yield Ok(StreamEvent::ContentDelta("Video generation pending...\n".into()));
                    }
                    "in_progress" => {
                        yield Ok(StreamEvent::ContentDelta("Video generation in progress...\n".into()));
                    }
                    "completed" => {
                        let video_url = poll.unsigned_urls.into_iter().next()
                            .unwrap_or_else(|| format!("{}/{}/content", OPENROUTER_VIDEOS_URL, poll.id));

                        yield Ok(StreamEvent::ContentDelta(format!("Downloading video...\n")));

                        match client.get(&video_url)
                            .bearer_auth(&api_key)
                            .send()
                            .await
                        {
                            Ok(dl) if dl.status().is_success() => {
                                let media_type = dl
                                    .headers()
                                    .get("content-type")
                                    .and_then(|v| v.to_str().ok())
                                    .unwrap_or("video/mp4")
                                    .to_string();
                                let ext = media_type.split('/').next_back().unwrap_or("mp4");
                                match dl.bytes().await {
                                    Ok(bytes) => {
                                        yield Ok(StreamEvent::FileAttachment {
                                            filename: format!("generated_video.{ext}"),
                                            media_type,
                                            data: bytes.to_vec(),
                                        });
                                    }
                                    Err(e) => {
                                        yield Err(anyhow::anyhow!("Failed to download video: {e}"));
                                    }
                                }
                            }
                            Ok(dl) => {
                                yield Err(anyhow::anyhow!("Video download failed: {}", dl.status()));
                            }
                            Err(e) => {
                                yield Err(anyhow::anyhow!("Video download failed: {e}"));
                            }
                        }
                        break;
                    }
                    "failed" => {
                        let err = poll.error.unwrap_or_else(|| "Unknown error".into());
                        yield Err(anyhow::anyhow!("Video generation failed: {err}"));
                        return;
                    }
                    other => {
                        yield Ok(StreamEvent::ContentDelta(format!("Video status: {other}\n")));
                    }
                }

                delay = (delay * 2).min(max_delay);
            }

            yield Ok(StreamEvent::Finished(FinishReason::Stop));
        })
    }

    pub async fn list_video_models(&self) -> anyhow::Result<Vec<VideoModelInfo>> {
        let url = format!("{}/models", OPENROUTER_VIDEOS_URL);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Video models API error: {}", resp.status());
        }

        #[derive(Deserialize)]
        struct ModelsResp {
            data: Vec<VideoModelInfo>,
        }

        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed.data)
    }

    pub async fn list_image_models(&self) -> anyhow::Result<Vec<ImageModelInfo>> {
        let resp = self
            .client
            .get("https://openrouter.ai/api/v1/images/models")
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Image models API error: {}", resp.status());
        }

        #[derive(Deserialize)]
        struct ModelsResp {
            data: Vec<ImageModelInfo>,
        }

        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed.data)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoModelInfo {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageModelInfo {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
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
