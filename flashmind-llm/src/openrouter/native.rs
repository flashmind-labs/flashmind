//! Provider-native context compaction through OpenRouter.
//!
//! Chat Completions cannot carry a native checkpoint. Compaction, and every
//! request that replays a checkpoint or the turns after it, use OpenRouter's
//! Anthropic Messages or OpenAI Responses endpoint instead.

use anyhow::Context;
use flashmind_types::{
    CompletionRequest, CompletionStream, LlmError, LlmProvider, Model, NativeCompaction,
    NativeReasoningFormat, ProviderPreferences,
};
use serde_json::{Value, json};

use super::{AppAttribution, OpenRouterProvider, model_routing_preferences};
use crate::http::{send_stream_with_retry, wait_for_rate_limit};
use crate::sse::completion_request_error;
use crate::{anthropic, responses};

pub(super) const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// A low characters-per-token bound. Dense text such as numbers and code can
/// approach two, so this only skips histories that are clearly too short.
/// Overestimating costs one ordinary reply before local summarization.
const MIN_CHARS_PER_TOKEN: usize = 2;

/// An OpenRouter endpoint that speaks a vendor's native format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeApi {
    Messages,
    Responses,
}

impl NativeApi {
    /// The native endpoint that serves this model's vendor, if it compacts.
    pub(super) fn for_model(model: &Model) -> Option<Self> {
        let slug = model.capability_name().trim_start_matches('~');
        if slug.starts_with("anthropic/") {
            Some(Self::Messages)
        } else if slug.starts_with("openai/") {
            Some(Self::Responses)
        } else {
            None
        }
    }

    /// The native endpoint whose blocks this request replays.
    pub(super) fn replayed_by(request: &CompletionRequest) -> Option<Self> {
        request.messages.iter().find_map(|message| {
            let details = &message.reasoning_details;
            if details
                .native_items(NativeReasoningFormat::Anthropic)
                .is_some()
            {
                Some(Self::Messages)
            } else if details
                .native_items(NativeReasoningFormat::Responses)
                .is_some()
            {
                Some(Self::Responses)
            } else {
                None
            }
        })
    }

    /// Whether this request can be sent to the native endpoint.
    pub(super) fn accepts(self, request: &CompletionRequest) -> bool {
        self == Self::Messages || !responses::requires_chat_completions(request)
    }

    fn path(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::Responses => "responses",
        }
    }
}

/// Everything needed to rebuild a native request for each retry.
struct NativeTarget {
    client: reqwest::Client,
    url: String,
    api_key: String,
    app: AppAttribution,
    api: NativeApi,
}

impl NativeTarget {
    fn post(&self, body: &Value) -> reqwest::RequestBuilder {
        let request = self
            .client
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(body);
        let request = match self.api {
            NativeApi::Messages => request
                .header("anthropic-version", anthropic::ANTHROPIC_VERSION)
                .header("anthropic-beta", anthropic::OPENROUTER_COMPACTION_BETA),
            NativeApi::Responses => request,
        };
        self.app.apply(request)
    }
}

impl OpenRouterProvider {
    fn routing(&self, request: &CompletionRequest) -> Option<ProviderPreferences> {
        request.provider_preferences.clone().or_else(|| {
            model_routing_preferences(
                self.provider_preferences.as_ref(),
                &self.model_quantizations,
                request.model.capability_name(),
            )
        })
    }

    fn add_routing(&self, body: &mut Value, request: &CompletionRequest) -> anyhow::Result<()> {
        if let Some(routing) = self.routing(request) {
            body["provider"] = serde_json::to_value(routing)?;
        }
        if let Some(session) = &request.session_id {
            body["session_id"] = serde_json::to_value(session)?;
        }
        Ok(())
    }

    fn native_target(&self, api: NativeApi) -> NativeTarget {
        NativeTarget {
            client: self.client.clone(),
            url: format!("{}/{}", self.native_base_url, api.path()),
            api_key: self.api_key.clone(),
            app: self.app.clone(),
            api,
        }
    }

    /// Compact through the vendor's native endpoint. `None` selects local summarization.
    pub(super) async fn compact_natively(
        &self,
        request: CompletionRequest,
        instructions: &str,
    ) -> anyhow::Result<Option<NativeCompaction>> {
        let Some(api) = NativeApi::for_model(&request.model).filter(|api| api.accepts(&request))
        else {
            return Ok(None);
        };
        let mut body = match api {
            NativeApi::Messages => {
                let body = anthropic::openrouter_request(&request, Some(instructions))?;
                let estimated =
                    serde_json::to_string(&body["messages"])?.len() / MIN_CHARS_PER_TOKEN;
                if estimated < anthropic::MIN_COMPACTION_TRIGGER_TOKENS as usize {
                    tracing::debug!(model = %request.model, estimated, "History too short for native Anthropic compaction");
                    return Ok(None);
                }
                body
            }
            NativeApi::Responses => {
                let mut input = responses::build_request(&request)?["input"].take();
                input
                    .as_array_mut()
                    .context("Responses input is not a list")?
                    .push(json!({"type": "compaction_trigger"}));
                json!({
                    "model": request.model.capability_name(),
                    "input": input,
                    "instructions": instructions,
                    "store": false,
                })
            }
        };
        self.add_routing(&mut body, &request)?;
        tracing::debug!(model = %request.model, ?api, "Requesting native compaction through OpenRouter");
        wait_for_rate_limit(&self.rate_limiter).await;
        let response = self
            .native_target(api)
            .post(&body)
            .send()
            .await
            .context("OpenRouter native compaction request failed")?;
        let status = response.status();
        if matches!(status.as_u16(), 400 | 404 | 405 | 422 | 501) {
            let text = response.text().await.unwrap_or_default();
            tracing::warn!(model = %request.model, status = status.as_u16(), body = %text, "OpenRouter native compaction unavailable; using local summarization");
            return Ok(None);
        }
        let body: Value = response
            .error_for_status()
            .context("OpenRouter native compaction failed")?
            .json()
            .await?;
        match api {
            NativeApi::Messages => anthropic::parse_openrouter_compaction(body),
            NativeApi::Responses => responses::parse_compaction(body).map(Some),
        }
    }

    /// Stream a completion that replays native blocks.
    pub(super) fn complete_natively(
        &self,
        api: NativeApi,
        request: CompletionRequest,
    ) -> CompletionStream {
        let body = match api {
            NativeApi::Messages => anthropic::openrouter_request(&request, None),
            NativeApi::Responses => responses::build_request(&request).map(|mut body| {
                responses::apply_api_options(&mut body, &request);
                body
            }),
        }
        .and_then(|mut body| {
            self.add_routing(&mut body, &request)?;
            Ok(body)
        });
        let body = match body {
            Ok(body) => body,
            Err(error) => return Box::pin(futures_util::stream::once(async { Err(error) })),
        };
        let target = self.native_target(api);
        let rate_limiter = self.rate_limiter.clone();
        let provider = self.provider().to_string();
        tracing::debug!(model = %request.model, ?api, "Sending native completion through OpenRouter");
        Box::pin(async_stream::stream! {
            wait_for_rate_limit(&rate_limiter).await;
            let response = match send_stream_with_retry(|| target.post(&body)).await
            {
                Ok(response) => response,
                Err(error) => {
                    yield Err(completion_request_error(&provider, error));
                    return;
                }
            };
            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                yield Err(LlmError::classify(format!("{provider} error: API error {}: {text}", status.as_u16())).into());
                return;
            }
            let mut events = match api {
                NativeApi::Messages => anthropic::stream_messages(response),
                NativeApi::Responses => responses::process_response(response),
            };
            while let Some(event) = futures_util::StreamExt::next(&mut events).await {
                yield event;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::tests::http_endpoint;
    use flashmind_types::model::ReasoningLevel;
    use flashmind_types::{FinishReason, Message, ReasoningDetails, SessionId, StreamEvent};
    use futures_util::StreamExt;

    fn provider(url: &str) -> OpenRouterProvider {
        let mut provider = OpenRouterProvider::new("example-key".into());
        provider.native_base_url = url.into();
        provider
    }

    fn request(model: &str, messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest::new(format!("openrouter:{model}").parse().unwrap(), messages)
    }

    fn long_history() -> Vec<Message> {
        vec![
            Message::system("original instructions"),
            Message::user("x".repeat(250_000)),
            Message::assistant("noted"),
        ]
    }

    #[test]
    fn only_vendors_with_native_compaction_get_a_native_route() {
        let provider = OpenRouterProvider::new("example-key".into());
        for (model, native) in [
            ("anthropic/claude-opus-5.5", true),
            ("~anthropic/claude-sonnet-5", true),
            ("openai/gpt-5.5", true),
            ("z-ai/glm-5", false),
        ] {
            let request = request(model, vec![Message::user("task")]);
            assert_eq!(
                provider.supports_native_context(&request),
                native,
                "{model}"
            );
        }
    }

    #[tokio::test]
    async fn other_vendors_and_short_histories_compact_locally() {
        let provider = provider("http://127.0.0.1:9");
        let other = request("z-ai/glm-5", long_history());
        assert!(provider.compact(other, "retain").await.unwrap().is_none());
        let short = request("anthropic/claude-opus-5.5", vec![Message::user("task")]);
        assert!(provider.compact(short, "retain").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn anthropic_compaction_pauses_and_replays_through_messages() {
        let block = json!({"type":"compaction","content":"summary of the task"});
        let endpoint = http_endpoint(
            200,
            "application/json",
            json!({
                "stop_reason":"compaction",
                "content":[block],
                "usage":{"input_tokens":0,"output_tokens":0,"iterations":[
                    {"type":"compaction","input_tokens":60000,"output_tokens":400}
                ]}
            })
            .to_string(),
        )
        .await;
        let provider = provider(&endpoint.url);
        let mut compaction = request("anthropic/claude-opus-5.5", long_history());
        compaction.reasoning = ReasoningLevel::High;
        compaction.session_id = Some(SessionId::new());
        let compacted = provider
            .compact(compaction, "retain pending work")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(compacted.summary, "summary of the task");
        assert_eq!(compacted.usage.prompt_tokens, 60_000);

        let captured = endpoint.request.await.unwrap();
        let headers = captured.headers.to_lowercase();
        assert!(headers.starts_with("post /messages "));
        assert!(headers.contains("authorization: bearer example-key"));
        assert!(headers.contains("anthropic-beta: compact-2026-01-12"));
        let body = captured.body;
        assert_eq!(body["model"], "anthropic/claude-opus-5.5");
        assert_eq!(body["stream"], false);
        assert!(body.get("thinking").is_none());
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(
            messages.last().unwrap()["role"],
            "user",
            "no assistant prefill"
        );
        assert!(body["session_id"].is_string());
        assert_eq!(
            body["context_management"]["edits"][0],
            json!({
                "type":"compact_20260112",
                "trigger":{"type":"input_tokens","value":50_000},
                "pause_after_compaction":true,
                "instructions":"retain pending work",
            })
        );

        let stream = "event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":7}}}\n\n\
            event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n\
            event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n\
            event: message_stop\ndata: {}\n\n";
        let endpoint = http_endpoint(200, "text/event-stream", stream.into()).await;
        let provider = super::tests::provider(&endpoint.url);
        let mut checkpoint = Message::assistant("");
        checkpoint.reasoning_details = compacted.context;
        let mut replay = request(
            "anthropic/claude-opus-5.5",
            vec![checkpoint, Message::user("next task")],
        );
        replay.reasoning = ReasoningLevel::High;
        let events = provider.complete(replay).collect::<Vec<_>>().await;
        assert!(
            events.iter().any(
                |event| matches!(event, Ok(StreamEvent::ContentDelta(text)) if text == "answer")
            )
        );
        let body = endpoint.request.await.unwrap().body;
        assert_eq!(body["messages"][0]["content"][0], block);
        assert_eq!(body["thinking"], json!({"type":"adaptive"}));
        assert_eq!(
            body["context_management"]["edits"][0]["trigger"]["value"],
            10_000_000
        );
        assert!(
            body["context_management"]["edits"][0]
                .get("pause_after_compaction")
                .is_none()
        );
    }

    #[tokio::test]
    async fn responses_compaction_appends_a_trigger_and_replays_through_responses() {
        let item = json!({"type":"compaction","id":"cmp_example","encrypted_content":"opaque"});
        let endpoint = http_endpoint(
            200,
            "application/json",
            json!({"output":[item],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}})
                .to_string(),
        )
        .await;
        let provider = provider(&endpoint.url);
        let compacted = provider
            .compact(
                request("openai/gpt-5.5", long_history()),
                "retain pending work",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(compacted.usage.total_tokens, 15);
        let captured = endpoint.request.await.unwrap();
        assert!(
            captured
                .headers
                .to_lowercase()
                .starts_with("post /responses ")
        );
        let input = captured.body["input"].as_array().unwrap();
        assert_eq!(input.last().unwrap(), &json!({"type":"compaction_trigger"}));
        assert_eq!(captured.body["instructions"], "retain pending work");
        assert_eq!(captured.body["store"], false);

        let stream = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n\
            data: {\"type\":\"response.completed\",\"response\":{\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n";
        let endpoint = http_endpoint(200, "text/event-stream", stream.into()).await;
        let provider = super::tests::provider(&endpoint.url);
        let mut checkpoint = Message::assistant("");
        checkpoint.reasoning_details = compacted.context;
        let mut replay = request("openai/gpt-5.5", vec![checkpoint, Message::user("next")]);
        replay.max_tokens = Some(100);
        let events = provider.complete(replay).collect::<Vec<_>>().await;
        assert!(matches!(
            events.last(),
            Some(Ok(StreamEvent::Finished(FinishReason::Stop)))
        ));
        let body = endpoint.request.await.unwrap().body;
        assert_eq!(body["input"][0], item);
        assert_eq!(body["model"], "openai/gpt-5.5");
        assert_eq!(body["max_output_tokens"], 100);
    }

    #[test]
    fn chat_completions_history_does_not_select_a_native_route() {
        let mut assistant = Message::assistant("answer");
        assistant.reasoning_details =
            ReasoningDetails::from_blocks([serde_json::Map::from_iter([
                ("type".into(), json!("reasoning.text")),
                ("text".into(), json!("thinking")),
            ])]);
        let request = request("anthropic/claude-opus-5.5", vec![assistant]);
        assert_eq!(NativeApi::replayed_by(&request), None);
    }
}
