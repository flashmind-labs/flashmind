//! Responses API support for API keys and ChatGPT plan usage.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use flashmind_types::llm::ImageModelCapabilities;
use flashmind_types::{
    CompletionRequest, CompletionStream, ContentPart, FinishReason, ImageEditRequest,
    ImageGenRequest, LlmProvider, Model, ModelCapabilities, ModelInfo, ModelPricing,
    NativeCompaction, NativeReasoningFormat, Provider, ReasoningDetails, Role, StreamEvent,
    SttRequest, TokenUsage, TtsRequest, VideoGenRequest, Voice,
};
use futures_util::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};

/// Supplies an authorization token, refreshing it when necessary.
#[async_trait]
pub trait AccessTokenSource: Send + Sync {
    async fn access_token(&self) -> Result<SecretString>;
}

/// OpenAI Responses API provider for API keys and ChatGPT plans.
pub struct ResponsesProvider {
    client: reqwest::Client,
    tokens: Arc<dyn AccessTokenSource>,
    base_url: String,
    api_provider: Option<Arc<dyn LlmProvider>>,
}

struct ApiKeyToken(SecretString);

#[async_trait]
impl AccessTokenSource for ApiKeyToken {
    async fn access_token(&self) -> Result<SecretString> {
        Ok(self.0.clone())
    }
}

impl ResponsesProvider {
    /// Create a provider using a source of plan-authorized OAuth tokens.
    pub fn new(tokens: Arc<dyn AccessTokenSource>) -> Self {
        Self {
            client: crate::http::streaming_http_client(),
            tokens,
            base_url: "https://api.openai.com/v1".into(),
            api_provider: None,
        }
    }

    /// Use Responses for API-key chat and retain the existing provider for other operations.
    pub fn with_api_key(key: impl Into<SecretString>, provider: Arc<dyn LlmProvider>) -> Self {
        let mut result = Self::new(Arc::new(ApiKeyToken(key.into())));
        result.api_provider = Some(provider);
        result
    }
}

#[async_trait]
impl LlmProvider for ResponsesProvider {
    fn name(&self) -> &str {
        "openai"
    }
    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    async fn context_window(&self, model: &Model) -> Option<u32> {
        match &self.api_provider {
            Some(provider) => provider.context_window(model).await,
            None => None,
        }
    }

    async fn capabilities(&self, model: &Model) -> ModelCapabilities {
        match &self.api_provider {
            Some(provider) => provider.capabilities(model).await,
            None => ModelCapabilities {
                tool_calling: true,
                images: true,
                documents: true,
                reasoning: true,
                ..Default::default()
            },
        }
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        if let Some(provider) = &self.api_provider {
            return provider.list_models().await;
        }
        let token = self.tokens.access_token().await.ok()?;
        let response: Value = self
            .client
            .get(format!("{}/models", self.base_url))
            .bearer_auth(token.expose_secret())
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        let capabilities = ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            reasoning: true,
            ..Default::default()
        };
        Some(
            response
                .get("models")?
                .as_array()?
                .iter()
                .filter_map(|entry| {
                    if entry.get("visibility")?.as_str()? != "list" {
                        return None;
                    }
                    Some(ModelInfo {
                        id: entry.get("slug")?.as_str()?.to_owned(),
                        name: entry
                            .get("display_name")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        created_at: None,
                        context_length: None,
                        max_completion_tokens: None,
                        capabilities,
                        categories: capabilities.categories(),
                        pricing: ModelPricing::default(),
                        image: ImageModelCapabilities::default(),
                    })
                })
                .collect(),
        )
    }

    fn supports_native_context(&self, request: &CompletionRequest) -> bool {
        self.api_provider.is_none() || !requires_chat_completions(request)
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        if let Some(provider) = &self.api_provider
            && requires_chat_completions(&request)
        {
            if request.messages.iter().any(|message| {
                message
                    .reasoning_details
                    .native_items(NativeReasoningFormat::Responses)
                    .is_some_and(|items| items.iter().any(|item| item["type"] == "compaction"))
            }) {
                return Box::pin(futures_util::stream::once(async {
                    Err(anyhow::anyhow!(
                        "Chat Completions cannot replay a Responses checkpoint; restore the original transcript"
                    ))
                }));
            }
            return provider.complete(request);
        }
        let client = self.client.clone();
        let tokens = self.tokens.clone();
        let url = format!("{}/responses", self.base_url);
        let api_key_mode = self.api_provider.is_some();
        Box::pin(async_stream::try_stream! {
            let mut body = build_request(&request)?;
            if api_key_mode {
                apply_api_options(&mut body, &request);
            }
            let token = tokens.access_token().await?;
            let response = client.post(url)
                .bearer_auth(token.expose_secret()).json(&body).send().await?
                .error_for_status().context("OpenAI Responses request failed")?;
            let mut stream = process_response(response);
            while let Some(event) = stream.next().await { yield event?; }
        })
    }

    async fn compact(
        &self,
        mut request: CompletionRequest,
        instructions: &str,
    ) -> Result<Option<NativeCompaction>> {
        if requires_chat_completions(&request) {
            return Ok(None);
        }
        let system = request
            .messages
            .iter()
            .filter(|message| message.role == Role::System)
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        request
            .messages
            .retain(|message| message.role != Role::System);
        let body = build_request(&request)?;
        let instructions = if system.is_empty() {
            instructions.to_owned()
        } else {
            format!("{system}\n\n{instructions}")
        };
        let compact_body = json!({"model":request.model.capability_name(),"input":body["input"],"instructions":instructions});
        tracing::debug!(model = %request.model, input_items = body["input"].as_array().map_or(0, Vec::len), "Requesting native Responses compaction");
        let token = self.tokens.access_token().await?;
        let response = self
            .client
            .post(format!("{}/responses/compact", self.base_url))
            .bearer_auth(token.expose_secret())
            .json(&compact_body)
            .send()
            .await
            .context("Native Responses compaction request failed")?;
        let status = response.status();
        if matches!(status.as_u16(), 400 | 404 | 405 | 422 | 501) {
            tracing::warn!(model = %request.model, status = status.as_u16(), "Native Responses compaction unavailable; using local summarization");
            return Ok(None);
        }
        let body: Value = response
            .error_for_status()
            .context("Native Responses compaction failed")?
            .json()
            .await?;
        parse_compaction(body).map(Some)
    }

    fn text_to_speech(&self, request: TtsRequest) -> CompletionStream {
        self.api_provider.as_ref().map_or_else(
            || unsupported("text-to-speech"),
            |provider| provider.text_to_speech(request),
        )
    }

    fn transcribe(&self, request: SttRequest) -> CompletionStream {
        self.api_provider.as_ref().map_or_else(
            || unsupported("transcription"),
            |provider| provider.transcribe(request),
        )
    }

    async fn list_voices(&self, model: &str) -> Option<Vec<Voice>> {
        match &self.api_provider {
            Some(provider) => provider.list_voices(model).await,
            None => None,
        }
    }

    fn generate_image(&self, request: ImageGenRequest) -> CompletionStream {
        self.api_provider.as_ref().map_or_else(
            || unsupported("image generation"),
            |provider| provider.generate_image(request),
        )
    }

    fn edit_image(&self, request: ImageEditRequest) -> CompletionStream {
        self.api_provider.as_ref().map_or_else(
            || unsupported("image editing"),
            |provider| provider.edit_image(request),
        )
    }

    fn generate_video(&self, request: VideoGenRequest) -> CompletionStream {
        self.api_provider.as_ref().map_or_else(
            || unsupported("video generation"),
            |provider| provider.generate_video(request),
        )
    }
}

pub(crate) fn parse_compaction(body: Value) -> Result<NativeCompaction> {
    let output = body
        .get("output")
        .and_then(Value::as_array)
        .context("Native Responses compaction returned no output window")?;
    anyhow::ensure!(
        output.iter().any(
            |item| item.get("type").and_then(Value::as_str) == Some("compaction")
                && item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.is_empty())
        ),
        "Native Responses compaction returned no encrypted checkpoint"
    );
    let context = ReasoningDetails::native(NativeReasoningFormat::Responses, output.clone());
    let usage = response_usage(&json!({"response":body}))?.unwrap_or_default();
    Ok(NativeCompaction {
        context,
        summary: "Context compacted with native OpenAI state.".into(),
        usage,
    })
}

pub(crate) fn process_response(response: reqwest::Response) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        let mut stream = response.bytes_stream().eventsource();
        let mut complete = false;
        let mut called_tool = false;
        let mut tool_indices = std::collections::HashMap::new();
        let mut output_items = BTreeMap::new();
        while let Some(event) = stream.next().await {
            let event = event.context("OpenAI Responses stream failed")?;
            let event: Value = serde_json::from_str(&event.data).context("invalid Responses event")?;
            match event.get("type").and_then(Value::as_str).unwrap_or("") {
                "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        yield StreamEvent::ReasoningDelta(delta.to_owned());
                    }
                }
                "response.output_item.done" => {
                    let index = event.get("output_index").and_then(Value::as_u64).context("missing output item index")?;
                    let item = event.get("item").context("missing completed output item")?;
                    output_items.insert(index, item.clone());
                }
                "response.output_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        yield StreamEvent::ContentDelta(delta.to_owned());
                    }
                }
                "response.output_item.added" if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") => {
                    let output_index = event.get("output_index").and_then(Value::as_u64).context("missing tool index")?;
                    let index = tool_indices.len();
                    if tool_indices.insert(output_index, index).is_some() {
                        Err(anyhow::anyhow!("duplicate tool index"))?;
                    }
                    let id = event.pointer("/item/call_id").and_then(Value::as_str).context("missing tool call ID")?;
                    let name = event.pointer("/item/name").and_then(Value::as_str).context("missing tool name")?;
                    called_tool = true;
                    yield StreamEvent::ToolCallStart { index, id: id.to_owned(), name: name.to_owned() };
                }
                "response.function_call_arguments.delta" => {
                    let output_index = event.get("output_index").and_then(Value::as_u64).context("missing tool index")?;
                    let index = *tool_indices.get(&output_index).context("unannounced tool call")?;
                    let arguments = event.get("delta").and_then(Value::as_str).context("missing tool arguments")?;
                    yield StreamEvent::ToolCallDelta { index, arguments: arguments.to_owned() };
                }
                "response.completed" => {
                    complete = true;
                    if let Some(details) = completed_reasoning(&event, &output_items) {
                        yield StreamEvent::ReasoningDetailsDelta(details);
                    }
                    if let Some(usage) = response_usage(&event)? { yield StreamEvent::Usage(usage); }
                    yield StreamEvent::Finished(if called_tool { FinishReason::ToolCalls } else { FinishReason::Stop });
                    break;
                }
                "response.incomplete" if event.pointer("/response/incomplete_details/reason").and_then(Value::as_str) == Some("max_output_tokens") => {
                    complete = true;
                    if let Some(details) = completed_reasoning(&event, &output_items) {
                        yield StreamEvent::ReasoningDetailsDelta(details);
                    }
                    if let Some(usage) = response_usage(&event)? { yield StreamEvent::Usage(usage); }
                    yield StreamEvent::Finished(FinishReason::Length);
                    break;
                }
                "response.failed" | "response.incomplete" | "error" => {
                    Err(anyhow::anyhow!("OpenAI Responses request did not complete: {}", event.pointer("/response/error/code").or_else(|| event.get("code")).and_then(Value::as_str).unwrap_or("unknown error")))?;
                }
                _ => {}
            }
        }
        if !complete { Err(anyhow::anyhow!("OpenAI Responses stream ended before completion"))?; }
    })
}

fn response_usage(event: &Value) -> Result<Option<TokenUsage>> {
    let Some(usage) = event.pointer("/response/usage") else {
        return Ok(None);
    };
    let count = |key: &str| -> Result<u32> {
        Ok(u32::try_from(
            usage.get(key).and_then(Value::as_u64).unwrap_or(0),
        )?)
    };
    Ok(Some(TokenUsage {
        prompt_tokens: count("input_tokens")?,
        completion_tokens: count("output_tokens")?,
        total_tokens: count("total_tokens")?,
        cache_read_tokens: u32::try_from(
            usage
                .pointer("/input_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )?,
        cache_creation_tokens: 0,
    }))
}

fn completed_reasoning(event: &Value, items: &BTreeMap<u64, Value>) -> Option<ReasoningDetails> {
    let output = event
        .pointer("/response/output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| items.values().cloned().collect());
    output
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .then(|| ReasoningDetails::native(NativeReasoningFormat::Responses, output))
}

fn unsupported(operation: &str) -> CompletionStream {
    let message = format!("ChatGPT plan provider does not support {operation}");
    Box::pin(futures_util::stream::once(async move {
        Err(anyhow::anyhow!(message))
    }))
}

/// Sampling and limit fields that API-key routes accept but plan routes reject.
pub(crate) fn apply_api_options(body: &mut Value, request: &CompletionRequest) {
    if let Some(max_tokens) = request.max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = request.sampling.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.sampling.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(user) = &request.user {
        body["user"] = json!(user);
    }
}

pub(crate) fn requires_chat_completions(request: &CompletionRequest) -> bool {
    !request.modalities.is_empty()
        || request.sampling.top_k.is_some()
        || request.sampling.min_p.is_some()
        || request.sampling.presence_penalty.is_some()
        || request.sampling.repetition_penalty.is_some()
        || request
            .messages
            .iter()
            .flat_map(|message| message.parts.iter().flatten())
            .any(|part| {
                matches!(
                    part,
                    ContentPart::Audio { .. }
                        | ContentPart::Video { .. }
                        | ContentPart::VideoUrl { .. }
                )
            })
}

/// Convert a completion request to a streaming Responses API request.
pub fn build_request(request: &CompletionRequest) -> Result<Value> {
    let mut input = Vec::new();
    for message in &request.messages {
        if message.role == Role::Assistant
            && let Some(items) = message
                .reasoning_details
                .native_items(NativeReasoningFormat::Responses)
        {
            input.extend_from_slice(items);
            continue;
        }
        if message.role == Role::Tool {
            let call_id = message
                .tool_call_id
                .as_deref()
                .context("tool result has no call ID")?;
            input.push(
                json!({"type":"function_call_output", "call_id":call_id, "output":message.content}),
            );
            continue;
        }
        let mut content = Vec::new();
        if !message.content.is_empty() {
            let kind = if message.role == Role::Assistant {
                "output_text"
            } else {
                "input_text"
            };
            content.push(json!({"type":kind,"text":message.content}));
        }
        if let Some(parts) = &message.parts {
            for part in parts {
                content.push(match part {
                    ContentPart::Text { text } => json!({"type":"input_text","text":text}),
                    ContentPart::Image { media_type, data } => json!({"type":"input_image","image_url":format!("data:{media_type};base64,{data}")}),
                    ContentPart::ImageUrl { url } => json!({"type":"input_image","image_url":url}),
                    ContentPart::Document { filename, media_type, data } => json!({"type":"input_file","filename":filename,"file_data":format!("data:{media_type};base64,{data}")}),
                    _ => bail!("This attachment type is not supported by the Responses API provider"),
                });
            }
        }
        if !content.is_empty() {
            let role = if message.role == Role::System {
                "developer".to_owned()
            } else {
                message.role.to_string()
            };
            input.push(json!({"role":role,"content":content}));
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                input.push(json!({"type":"function_call","namespace":"flsh","call_id":call.id,"name":call.name,"arguments":serde_json::to_string(&call.arguments)?}));
            }
        }
    }
    let tools: Vec<_> = request.tools.iter().map(|tool| json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.parameters,"strict":false})).collect();
    let mut body =
        json!({"model":request.model.capability_name(),"input":input,"stream":true,"store":false});
    body["include"] = json!(["reasoning.encrypted_content"]);
    if !tools.is_empty() {
        body["tools"] = json!([{"type":"namespace","name":"flsh","description":"Tools available to the assistant.","tools":tools}]);
    }
    if request.reasoning.is_on() {
        body["reasoning"] = json!({"effort":request.reasoning.to_string(), "summary":"auto"});
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::{CompletionRequest, Message, ToolCall, ToolDefinition};
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct TestToken;

    struct Fallback;

    #[async_trait]
    impl LlmProvider for Fallback {
        fn name(&self) -> &str {
            "openai"
        }
        fn provider(&self) -> Provider {
            Provider::OpenAi
        }
        fn complete(&self, _request: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamEvent::ContentDelta("fallback".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
        fn generate_image(&self, _request: ImageGenRequest) -> CompletionStream {
            Box::pin(futures_util::stream::once(async {
                Ok(StreamEvent::ContentDelta("image".into()))
            }))
        }
    }

    #[async_trait]
    impl AccessTokenSource for TestToken {
        async fn access_token(&self) -> Result<SecretString> {
            Ok("example-token".into())
        }
    }

    #[tokio::test]
    async fn chat_fallback_rejects_opaque_responses_context() {
        let provider = ResponsesProvider::with_api_key("example-key", Arc::new(Fallback));
        let mut snapshot = Message::assistant("");
        snapshot.reasoning_details = ReasoningDetails::native(
            NativeReasoningFormat::Responses,
            vec![json!({"type":"compaction","encrypted_content":"opaque"})],
        );
        let mut request =
            CompletionRequest::new("openai:example-model".parse().unwrap(), vec![snapshot]);
        request.sampling.top_k = Some(5);
        let events = provider.complete(request).collect::<Vec<_>>().await;
        assert!(events[0].is_err());
    }

    #[tokio::test]
    async fn native_compaction_replays_the_entire_canonical_window() {
        let output = json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":"original task"}]},
            {"type":"compaction","id":"cmp_example","encrypted_content":"opaque","extra":"keep"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"retained answer","annotations":[]}]}
        ]);
        let endpoint = crate::sse::tests::http_endpoint(
            200,
            "application/json",
            json!({
                "output":output,"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}
            })
            .to_string(),
        )
        .await;
        let mut provider = ResponsesProvider::new(Arc::new(TestToken));
        provider.base_url = endpoint.url;
        let request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![
                Message::system("original instructions"),
                Message::user("original task"),
                Message::assistant("original answer"),
            ],
        );
        let compacted = provider
            .compact(request, "retain pending work")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(compacted.usage.total_tokens, 15);
        let captured = endpoint.request.await.unwrap();
        assert!(captured.headers.starts_with("POST /responses/compact "));
        assert_eq!(
            captured.body["instructions"],
            "original instructions\n\nretain pending work"
        );
        assert!(captured.body.get("stream").is_none());
        assert_eq!(captured.body["input"].as_array().unwrap().len(), 2);
        let mut checkpoint = Message::assistant("");
        checkpoint.reasoning_details = compacted.context;
        let request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![
                Message::system("original instructions"),
                checkpoint,
                Message::user("next task"),
            ],
        );
        let body = build_request(&request).unwrap();
        assert_eq!(
            &body["input"].as_array().unwrap()[1..4],
            output.as_array().unwrap()
        );
        assert_eq!(body["input"][4]["content"][0]["text"], "next task");
    }

    #[tokio::test]
    async fn native_compaction_declines_unsupported_routes_and_preserves_real_errors() {
        for status in [400, 404, 501, 401, 500] {
            let endpoint =
                crate::sse::tests::http_endpoint(status, "application/json", "{}".into()).await;
            let mut provider = ResponsesProvider::new(Arc::new(TestToken));
            provider.base_url = endpoint.url;
            let request = CompletionRequest::new(
                "openai:example-model".parse().unwrap(),
                vec![Message::user("task")],
            );
            let result = provider.compact(request, "retain state").await;
            if matches!(status, 400 | 404 | 501) {
                assert!(result.unwrap().is_none());
            } else {
                assert!(result.is_err());
            }
            endpoint.request.await.unwrap();
        }
        assert!(
            parse_compaction(json!({"output":[{"type":"compaction","encrypted_content":""}]}))
                .is_err()
        );
    }

    #[tokio::test]
    async fn encrypted_reasoning_uses_completed_items_and_replays_the_exact_output_order() {
        let output = json!([
            {"type":"reasoning","id":"rs_first","summary":[{"type":"summary_text","text":"Plan"}],"encrypted_content":"opaque-final","extra":"keep"},
            {"type":"message","id":"msg_example","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Checking","annotations":[]}]},
            {"type":"reasoning","id":"rs_second","summary":[],"encrypted_content":"opaque-second"},
            {"type":"function_call","id":"fc_example","call_id":"call_example","namespace":"flsh","name":"lookup","arguments":"{}","status":"completed"}
        ]);
        for snapshot in [false, true] {
            let mut body = String::new();
            let mut event = |data: Value| body.push_str(&format!("data: {data}\n\n"));
            event(
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_first","encrypted_content":"incomplete"}}),
            );
            event(json!({"type":"response.reasoning_summary_text.delta","delta":"Plan"}));
            for index in [0, 1, 2] {
                event(
                    json!({"type":"response.output_item.done","output_index":index,"item":output[index]}),
                );
            }
            event(json!({"type":"response.output_item.added","output_index":3,"item":output[3]}));
            event(
                json!({"type":"response.function_call_arguments.delta","output_index":3,"delta":"{}"}),
            );
            event(json!({"type":"response.output_item.done","output_index":3,"item":output[3]}));
            let mut completed = json!({"type":"response.completed","response":{}});
            if snapshot {
                completed["response"]["output"] = output.clone();
            }
            event(completed);
            let response = crate::sse::tests::response(body, false).await;
            let events = process_response(response).collect::<Vec<_>>().await;
            let mut details = ReasoningDetails::default();
            let mut reasoning = String::new();
            for event in events {
                match event.unwrap() {
                    StreamEvent::ReasoningDetailsDelta(delta) => details.append(delta),
                    StreamEvent::ReasoningDelta(text) => reasoning.push_str(&text),
                    _ => {}
                }
            }
            assert_eq!(reasoning, "Plan");
            assert_eq!(
                details
                    .native_items(NativeReasoningFormat::Responses)
                    .unwrap(),
                output.as_array().unwrap()
            );
            let mut assistant = Message::assistant_with_tool_calls(
                "Checking",
                vec![ToolCall {
                    id: "call_example".into(),
                    name: "lookup".into(),
                    arguments: json!({}),
                }],
            );
            assistant.reasoning_details = details;
            let request = CompletionRequest::new(
                "openai:example-model".parse().unwrap(),
                vec![
                    Message::user("question"),
                    assistant,
                    Message::tool_result("call_example", "evidence"),
                ],
            );
            let body = build_request(&request).unwrap();
            assert_eq!(
                &body["input"].as_array().unwrap()[1..5],
                output.as_array().unwrap()
            );
            assert_eq!(body["input"][5]["call_id"], "call_example");
            assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        }
    }

    #[tokio::test]
    async fn output_limit_keeps_reasoning_and_usage_when_thinking_consumes_the_budget() {
        let output = json!([{ "type":"reasoning","id":"rs_example","summary":[],"encrypted_content":"opaque" }]);
        let event = json!({"type":"response.incomplete","response":{
            "output":output,"incomplete_details":{"reason":"max_output_tokens"},
            "usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}
        }});
        let response = crate::sse::tests::response(format!("data: {event}\n\n"), false).await;
        let events = process_response(response).collect::<Vec<_>>().await;
        assert!(events.iter().any(|event| matches!(event, Ok(StreamEvent::ReasoningDetailsDelta(details)) if details.native_items(NativeReasoningFormat::Responses).unwrap() == output.as_array().unwrap())));
        assert!(events.iter().any(|event| matches!(event, Ok(StreamEvent::Usage(usage)) if usage.prompt_tokens == 10 && usage.completion_tokens == 20)));
        assert!(matches!(
            events.last(),
            Some(Ok(StreamEvent::Finished(FinishReason::Length)))
        ));
    }

    #[tokio::test]
    async fn interrupted_responses_do_not_return_partial_replay_state() {
        let body = "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_example\",\"encrypted_content\":\"opaque\"}}\n\n";
        let response = crate::sse::tests::response(body.into(), false).await;
        let events = process_response(response).collect::<Vec<_>>().await;
        assert!(events.last().unwrap().is_err());
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Ok(StreamEvent::ReasoningDetailsDelta(_))))
        );
    }

    #[test]
    fn request_preserves_tool_calls_and_results() {
        let mut request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![
                Message::user("hello"),
                Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "call-1".into(),
                        name: "search".into(),
                        arguments: json!({"q":"test"}),
                    }],
                ),
                Message::tool_result("call-1", "found"),
            ],
        );
        request.tools.push(ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: json!({"type":"object"}),
        });
        let body = build_request(&request).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["input"][1]["call_id"], "call-1");
        assert_eq!(body["input"][2]["output"], "found");
        assert_eq!(body["tools"][0]["tools"][0]["name"], "search");
    }

    #[test]
    fn image_after_tool_results_is_sent_as_image_input() {
        let request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![
                Message::assistant_with_tool_calls(
                    "",
                    vec![
                        ToolCall {
                            id: "call-1".into(),
                            name: "image_read".into(),
                            arguments: json!({}),
                        },
                        ToolCall {
                            id: "call-2".into(),
                            name: "read".into(),
                            arguments: json!({}),
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
        let body = build_request(&request).unwrap();
        assert_eq!(body["input"][2]["call_id"], "call-1");
        assert_eq!(body["input"][3]["call_id"], "call-2");
        assert_eq!(body["input"][4]["role"], "user");
        assert_eq!(
            body["input"][4]["content"][1],
            json!({
                "type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="
            })
        );
    }

    #[tokio::test]
    async fn lists_visible_models_for_the_plan() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let mut provider = ResponsesProvider::new(Arc::new(TestToken));
        provider.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap();
            assert!(headers.starts_with("GET /v1/models "));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: bearer example-token")
            );
            let body = r#"{"models":[{"slug":"example-model","display_name":"Example","visibility":"list"},{"slug":"hidden-model","visibility":"hide"}]}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        };
        let (models, ()) = tokio::join!(provider.list_models(), server);
        let models = models.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "example-model");
    }

    #[tokio::test]
    async fn api_key_chat_uses_responses_and_preserves_generation_options() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let mut provider = ResponsesProvider::with_api_key("example-api-key", Arc::new(Fallback));
        provider.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let mut request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![Message::user("hello")],
        );
        request.max_tokens = Some(64);
        request.sampling.temperature = Some(rust_decimal::Decimal::new(5, 1));
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap();
            assert!(headers.starts_with("POST /v1/responses "));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: bearer example-api-key")
            );
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(str::to_owned)
                })
                .unwrap()
                .parse()
                .unwrap();
            let mut bytes = vec![0; length];
            socket.read_exact(&mut bytes).await.unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["max_output_tokens"], 64);
            assert_eq!(body["temperature"], 0.5);
            let data = "data: {\"type\":\"response.completed\"}\n\n";
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len()).as_bytes()).await.unwrap();
        };
        let (events, ()) = tokio::join!(provider.complete(request).collect::<Vec<_>>(), server);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::Finished(FinishReason::Stop))
        ));
    }

    #[tokio::test]
    async fn unsupported_chat_options_use_existing_provider() {
        let provider = ResponsesProvider::with_api_key("example-api-key", Arc::new(Fallback));
        let mut request = CompletionRequest::new(
            "openai:example-model".parse().unwrap(),
            vec![Message::user("hello")],
        );
        request.sampling.top_k = Some(5);
        let events = provider.complete(request).collect::<Vec<_>>().await;
        assert!(
            matches!(&events[0], Ok(StreamEvent::ContentDelta(content)) if content == "fallback")
        );
        let image = ImageGenRequest {
            model: "example-image".into(),
            prompt: "draw".into(),
            size: None,
            aspect_ratio: None,
            quality: None,
            style: None,
            n: None,
            input_references: Vec::new(),
            output_format: None,
        };
        let events = provider.generate_image(image).collect::<Vec<_>>().await;
        assert!(matches!(&events[0], Ok(StreamEvent::ContentDelta(content)) if content == "image"));
    }

    #[tokio::test]
    async fn streaming_requires_completed_event() {
        for completed in [false, true] {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let mut provider = ResponsesProvider::new(Arc::new(TestToken));
            provider.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
            let request = CompletionRequest::new(
                "openai:example-model".parse().unwrap(),
                vec![Message::system("instructions"), Message::user("hello")],
            );
            let server = async {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    headers.push(socket.read_u8().await.unwrap());
                }
                let headers = String::from_utf8(headers).unwrap();
                assert!(headers.starts_with("POST /v1/responses "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer example-token")
                );
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_owned)
                    })
                    .unwrap()
                    .parse()
                    .unwrap();
                let mut bytes = vec![0; length];
                socket.read_exact(&mut bytes).await.unwrap();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["input"][0]["role"], "developer");
                assert!(body.get("max_output_tokens").is_none());
                let mut data =
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n"
                        .to_string();
                if completed {
                    data.push_str("data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"search\"}}\n\n");
                    data.push_str("data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"delta\":\"{}\"}\n\n");
                    data.push_str("data: {\"type\":\"response.completed\"}\n\n");
                }
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len()).as_bytes()).await.unwrap();
            };
            let (events, ()) = tokio::join!(provider.complete(request).collect::<Vec<_>>(), server);
            assert!(matches!(&events[0], Ok(StreamEvent::ContentDelta(text)) if text == "hello"));
            assert_eq!(events.last().unwrap().is_ok(), completed);
            if completed {
                assert!(matches!(
                    &events[1],
                    Ok(StreamEvent::ToolCallStart { index: 0, .. })
                ));
                assert!(matches!(
                    &events[2],
                    Ok(StreamEvent::ToolCallDelta { index: 0, .. })
                ));
            }
        }
    }
}
