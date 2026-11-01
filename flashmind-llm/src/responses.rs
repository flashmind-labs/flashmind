//! Responses API support for API keys and ChatGPT plan usage.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use flashmind_types::llm::ImageModelCapabilities;
use flashmind_types::{
    CompletionRequest, CompletionStream, ContentPart, FinishReason, ImageEditRequest,
    ImageGenRequest, LlmProvider, Model, ModelCapabilities, ModelInfo, ModelPricing, Provider,
    Role, StreamEvent, SttRequest, TokenUsage, TtsRequest, VideoGenRequest, Voice,
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

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        if let Some(provider) = &self.api_provider
            && requires_chat_completions(&request)
        {
            return provider.complete(request);
        }
        let client = self.client.clone();
        let tokens = self.tokens.clone();
        let url = format!("{}/responses", self.base_url);
        let api_key_mode = self.api_provider.is_some();
        Box::pin(async_stream::try_stream! {
            let mut body = build_request(&request)?;
            if api_key_mode {
                if let Some(max_tokens) = request.max_tokens { body["max_output_tokens"] = json!(max_tokens); }
                if let Some(temperature) = request.sampling.temperature { body["temperature"] = json!(temperature); }
                if let Some(top_p) = request.sampling.top_p { body["top_p"] = json!(top_p); }
                if let Some(user) = &request.user { body["user"] = json!(user); }
            }
            let token = tokens.access_token().await?;
            let response = client.post(url)
                .bearer_auth(token.expose_secret()).json(&body).send().await?
                .error_for_status().context("OpenAI Responses request failed")?;
            let mut stream = response.bytes_stream().eventsource();
            let mut complete = false;
            let mut called_tool = false;
            let mut tool_indices = std::collections::HashMap::new();
            while let Some(event) = stream.next().await {
                let event = event.context("OpenAI Responses stream failed")?;
                let event: Value = serde_json::from_str(&event.data).context("invalid Responses event")?;
                match event.get("type").and_then(Value::as_str).unwrap_or("") {
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
                        if let Some(usage) = event.pointer("/response/usage") {
                            let count = |key: &str| -> Result<u32> {
                                Ok(u32::try_from(usage.get(key).and_then(Value::as_u64).unwrap_or(0))?)
                            };
                            yield StreamEvent::Usage(TokenUsage {
                                prompt_tokens: count("input_tokens")?,
                                completion_tokens: count("output_tokens")?,
                                total_tokens: count("total_tokens")?,
                                cache_read_tokens: u32::try_from(usage.pointer("/input_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0))?,
                                cache_creation_tokens: 0,
                            });
                        }
                        yield StreamEvent::Finished(if called_tool { FinishReason::ToolCalls } else { FinishReason::Stop });
                        break;
                    }
                    "response.incomplete" if event.pointer("/response/incomplete_details/reason").and_then(Value::as_str) == Some("max_output_tokens") => {
                        complete = true;
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

fn unsupported(operation: &str) -> CompletionStream {
    let message = format!("ChatGPT plan provider does not support {operation}");
    Box::pin(futures_util::stream::once(async move {
        Err(anyhow::anyhow!(message))
    }))
}

fn requires_chat_completions(request: &CompletionRequest) -> bool {
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
    if !tools.is_empty() {
        body["tools"] = json!([{"type":"namespace","name":"flsh","description":"Tools available to the assistant.","tools":tools}]);
    }
    if request.reasoning.is_on() {
        body["reasoning"] = json!({"effort":request.reasoning.to_string()});
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
