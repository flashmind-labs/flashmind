//! [`MistralProvider`]: chat, model discovery and batch transcription.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use async_stream::stream;
use async_trait::async_trait;
use reqwest::{Client, Method, RequestBuilder};
use secrecy::{ExposeSecret, SecretString};
use url::Url;

use super::models::{self, ModelCache, ModelCard, ModelList};
use super::realtime::{self, RealtimeTranscription, RealtimeTranscriptionRequest};
use super::request::{build_chat_request, to_json};
use crate::http::{
    http_client, send_stream_with_retry, send_with_retry, streaming_http_client, stt_stream,
};
use crate::sse::{
    DEFAULT_STREAM_IDLE_TIMEOUT, DEFAULT_STREAM_START_TIMEOUT, completion_request_error,
    process_sse_stream,
};
use flashmind_types::model::Provider;
use flashmind_types::{
    CompletionRequest, CompletionStream, LlmError, LlmErrorKind, LlmProvider, Model,
    ModelCapabilities, ModelInfo, SttRequest, TtsRequest,
};

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai/";

/// Provider for Mistral's API.
///
/// Chat goes to `/v1/chat/completions` (OpenAI-compatible SSE), model
/// capabilities come from `/v1/models`, [`LlmProvider::transcribe`] posts to
/// `/v1/audio/transcriptions`, and [`MistralProvider::realtime_transcription`]
/// opens a live transcription session over a WebSocket.
///
/// ```no_run
/// use std::env;
///
/// use flashmind_llm::MistralProvider;
///
/// # fn example() -> anyhow::Result<()> {
/// let provider = MistralProvider::new(env::var("MISTRAL_API_KEY")?);
/// # Ok(())
/// # }
/// ```
pub struct MistralProvider {
    api_key: SecretString,
    base_url: Url,
    client: Client,
    models: Arc<ModelCache>,
    stream_start_timeout: Duration,
    stream_idle_timeout: Duration,
}

impl MistralProvider {
    /// Create a provider for `https://api.mistral.ai`.
    pub fn new(api_key: impl Into<SecretString>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(DEFAULT_BASE_URL).expect("the default Mistral URL is valid"),
            client: http_client(),
            models: Arc::default(),
            stream_start_timeout: DEFAULT_STREAM_START_TIMEOUT,
            stream_idle_timeout: DEFAULT_STREAM_IDLE_TIMEOUT,
        }
    }

    /// Send requests to another base URL, such as a proxy or a test server.
    /// The realtime WebSocket uses the same host with `ws`/`wss`.
    pub fn with_base_url(mut self, base_url: &str) -> anyhow::Result<Self> {
        self.base_url =
            Url::parse(base_url).with_context(|| format!("invalid Mistral URL '{base_url}'"))?;
        Ok(self)
    }

    /// Set how long a chat stream may wait for its first output.
    pub fn with_stream_start_timeout(mut self, timeout: Duration) -> Self {
        self.stream_start_timeout = timeout;
        self
    }

    /// Set how long a chat stream may stay silent after output has started.
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Open a realtime transcription session.
    ///
    /// Returns once Mistral has created the session. Push 16 kHz mono
    /// `pcm_s16le` audio into the [`AudioSink`](super::AudioSink) and read
    /// [`TranscriptionEvent`](super::TranscriptionEvent)s from the event
    /// stream; see [`RealtimeTranscription`].
    pub async fn realtime_transcription(
        &self,
        request: RealtimeTranscriptionRequest,
    ) -> anyhow::Result<RealtimeTranscription> {
        realtime::connect(&self.base_url, &self.api_key, request).await
    }

    /// Process a document with Mistral's native OCR API.
    pub async fn ocr(&self, request: &super::OcrRequest) -> anyhow::Result<serde_json::Value> {
        super::ocr::process(self.authed(&self.client, Method::POST, "v1/ocr"), request).await
    }

    fn url(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url
    }

    fn authed(&self, client: &Client, method: Method, path: &str) -> RequestBuilder {
        client
            .request(method, self.url(path))
            .bearer_auth(self.api_key.expose_secret())
    }

    /// Model cards, fetched at most once an hour.
    async fn model_cards(&self) -> Option<Arc<[ModelCard]>> {
        if let Some(cards) = self.models.fresh() {
            return Some(cards);
        }
        let response = send_with_retry(|| self.authed(&self.client, Method::GET, "v1/models"))
            .await
            .inspect_err(|error| tracing::warn!(%error, "Mistral model list request failed"))
            .ok()?;
        if !response.status().is_success() {
            tracing::warn!(status = %response.status(), "Mistral model list request failed");
            return None;
        }
        let list: ModelList = response
            .json()
            .await
            .inspect_err(|error| tracing::warn!(%error, "Mistral model list did not parse"))
            .ok()?;
        Some(self.models.store(list.data))
    }
}

#[async_trait]
impl LlmProvider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }

    fn provider(&self) -> Provider {
        Provider::Mistral
    }

    async fn context_window(&self, model: &Model) -> Option<u32> {
        let cards = self.model_cards().await?;
        models::find(&cards, model.capability_name())?.context_length()
    }

    async fn capabilities(&self, model: &Model) -> ModelCapabilities {
        let Some(cards) = self.model_cards().await else {
            return models::fallback_capabilities();
        };
        models::find(&cards, model.capability_name())
            .map_or_else(models::fallback_capabilities, ModelCard::capabilities)
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        let cards = self.model_cards().await?;
        Some(cards.iter().map(ModelCard::to_model_info).collect())
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let client = streaming_http_client();
        let url = self.url("v1/chat/completions");
        let api_key = self.api_key.clone();
        let model_reasons = self.models.reasons(request.model.capability_name());
        let start_timeout = self.stream_start_timeout;
        let idle_timeout = self.stream_idle_timeout;
        let provider = self.provider().to_string();

        Box::pin(stream! {
            let start = Instant::now();
            metrics::counter!("llm.requests.started").increment(1);
            let body = match to_json(&build_chat_request(&request, model_reasons)) {
                Ok(body) => body,
                Err(error) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    yield Err(anyhow::anyhow!("{provider} error: Failed to serialize request: {error}"));
                    return;
                }
            };
            tracing::debug!(model = %request.model, messages = request.messages.len(), tools = request.tools.len(), "Sending Mistral completion request");

            let response = match send_stream_with_retry(|| {
                client
                    .post(url.clone())
                    .bearer_auth(api_key.expose_secret())
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream")
                    .body(body.clone())
            })
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    yield Err(completion_request_error(&provider, error));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = match tokio::time::timeout(idle_timeout, response.text()).await {
                    Ok(text) => text.unwrap_or_default(),
                    Err(_) => {
                        yield Err(LlmError::new(LlmErrorKind::StreamError, "LLM error response body timed out").into());
                        return;
                    }
                };
                metrics::counter!("llm.requests.errors").increment(1);
                metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                let reason = status.canonical_reason().unwrap_or("Unknown");
                yield Err(LlmError::classify(format!("{provider} error: API error {} {reason}: {text}", status.as_u16())).into());
                return;
            }

            for await event in process_sse_stream(response, start_timeout, idle_timeout) {
                let failed = event.is_err();
                yield event;
                if failed {
                    metrics::counter!("llm.requests.errors").increment(1);
                    metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
                    return;
                }
            }
            metrics::counter!("llm.requests.completed").increment(1);
            metrics::histogram!("llm.request.duration_seconds").record(start.elapsed().as_secs_f64());
        })
    }

    fn text_to_speech(&self, request: TtsRequest) -> CompletionStream {
        super::speech::generate(
            self.authed(&self.client, Method::POST, "v1/audio/speech"),
            request,
        )
    }

    /// Batch transcription with a Voxtral model. The language hint is sent
    /// when set; the transcript arrives as one content delta.
    fn transcribe(&self, request: SttRequest) -> CompletionStream {
        stt_stream(
            self.authed(&self.client, Method::POST, "v1/audio/transcriptions"),
            request,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::tests::response_url;
    use flashmind_types::{FinishReason, Message, StreamEvent};
    use futures_util::StreamExt;

    /// Chunks in the shape Mistral streams for a reasoning model: thinking
    /// as a chunk list, the switch to text in one delta, then plain strings,
    /// with usage on the last chunk.
    const REASONING_STREAM: &str = concat!(
        "data: {\"id\":\"a1\",\"object\":\"chat.completion.chunk\",\"created\":1773842038,\"model\":\"mistral-medium-3-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"17 * 23 is 391.\"}]}]},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"a1\",\"object\":\"chat.completion.chunk\",\"created\":1773842038,\"model\":\"mistral-medium-3-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[],\"closed\":true},{\"type\":\"text\",\"text\":\"17 * 23\"}]},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"a1\",\"object\":\"chat.completion.chunk\",\"created\":1773842038,\"model\":\"mistral-medium-3-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" = 391.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":25,\"total_tokens\":212,\"completion_tokens\":187,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n",
        "data: [DONE]\n\n",
    );

    /// A streamed tool call as Mistral sends it: the whole call in one delta.
    const TOOL_STREAM: &str = concat!(
        "data: {\"id\":\"b2\",\"object\":\"chat.completion.chunk\",\"model\":\"mistral-small-2603\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"b2\",\"object\":\"chat.completion.chunk\",\"model\":\"mistral-small-2603\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"D681PevKs\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"},\"index\":0}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":90,\"total_tokens\":112,\"completion_tokens\":22}}\n\n",
        "data: [DONE]\n\n",
    );

    async fn events(body: &str) -> Vec<StreamEvent> {
        let url = response_url(body.to_string(), false).await;
        let provider = MistralProvider::new("test-key")
            .with_base_url(&url)
            .unwrap();
        let request = CompletionRequest::new(
            "mistral:mistral-medium-3-5".parse().unwrap(),
            vec![Message::user("What is 17 * 23?")],
        );
        provider
            .complete(request)
            .map(|event| event.unwrap())
            .collect()
            .await
    }

    #[tokio::test]
    async fn reasoning_chunks_stream_as_reasoning_then_text() {
        let events = events(REASONING_STREAM).await;
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "17 * 23 is 391.");
        assert_eq!(content, "17 * 23 = 391.");
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::Usage(usage) if usage.prompt_tokens == 25 && usage.completion_tokens == 187
        )));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Finished(FinishReason::Stop))
        ));
    }

    #[tokio::test]
    async fn tool_calls_stream_as_start_and_arguments() {
        let events: Vec<StreamEvent> = events(TOOL_STREAM)
            .await
            .into_iter()
            .filter(|event| !matches!(event, StreamEvent::Usage(_)))
            .collect();
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { index: 0, id, name } if id == "D681PevKs" && name == "get_weather"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::ToolCallDelta { index: 0, arguments } if arguments == "{\"city\": \"Paris\"}"
        ));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Finished(FinishReason::ToolCalls))
        ));
    }

    /// Needs `MISTRAL_API_KEY`. Run with `cargo test -p flashmind-llm -- --ignored mistral_live`.
    #[tokio::test]
    #[ignore = "calls the Mistral API"]
    async fn mistral_live_chat_streams_an_answer() {
        let key =
            std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY must be set for this test");
        let provider = MistralProvider::new(key);
        let model: Model = "mistral:mistral-small-2603".parse().unwrap();
        assert!(provider.capabilities(&model).await.tool_calling);
        let request = CompletionRequest::new(model, vec![Message::user("Say hi in one word.")]);
        let events: Vec<StreamEvent> = provider
            .complete(request)
            .map(|event| event.unwrap())
            .collect()
            .await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::ContentDelta(_)))
        );
        assert!(matches!(events.last(), Some(StreamEvent::Finished(_))));
    }

    #[test]
    fn debug_output_never_contains_the_key() {
        let provider = MistralProvider::new("very-secret-key");
        assert!(!format!("{:?}", provider.api_key).contains("very-secret-key"));
    }

    #[test]
    fn create_provider_builds_mistral() {
        let provider = crate::create_provider(Provider::Mistral, Some("key")).unwrap();
        assert_eq!(provider.provider(), Provider::Mistral);
        assert_eq!(provider.name(), "mistral");
        assert!(crate::create_provider(Provider::Mistral, None).is_err());
    }
}
