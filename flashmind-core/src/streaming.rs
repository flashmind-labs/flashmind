//! LLM response streaming and tool-call assembly.
//!
//! Wraps the raw provider stream into an [`AgentEvent`] stream and assembles
//! incremental text deltas / reasoning tokens into complete [`LlmResponse`] values.
//!
//! # Flow
//!
//! 1. Converts conversation entries to wire-format messages via `Conversation::to_messages()`
//! 2. Sends a [`CompletionRequest`] to the provider
//! 3. Drives the completion stream event loop, handling cancellation
//! 4. Emits `TextDelta`, `ReasoningDelta`, `Usage` events as they arrive
//! 5. Assembles partial tool call deltas into complete [`ToolCall`] objects
//! 6. Returns the final [`LlmResponse`] with content, tool calls, token counts, and finish reason
//!
//! # Tool call assembly
//!
//! Streaming tool calls arrive fragmented across multiple SSE chunks. The internal
//! `PendingToolCall` struct accumulates id/name/arguments until the stream finishes,
//! then `finalize_tool_calls` parses the JSON and produces structured `ToolCall` values.

use std::io::{Error as IoError, ErrorKind};

use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use flashmind_types::{
    AgentEvent, AgentLlmConfig, AgentStream, CompletionRequest, FinishReason, LlmError,
    LlmErrorKind, LlmProvider, Outcome, StreamEvent, ToolCall, ToolDefinition,
};

use crate::conversation::Conversation;

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

/// Fully-assembled response from the LLM after the stream completes.
///
/// Produced by [`stream_llm_response`] after consuming all events from the provider's
/// completion stream. Contains the assembled text content, any tool call requests,
/// token usage, and the reason generation stopped.
#[derive(Debug)]
pub struct LlmResponse {
    /// Plain text content of the assistant message.
    pub content: String,
    /// Reasoning/thinking trace (if the model supports it).
    pub reasoning: Option<String>,
    /// Tool call requests (empty if no tools were requested).
    pub tool_calls: Vec<ToolCall>,
    /// Token counts reported by the provider.
    pub prompt_tokens: u32,
    /// Completion tokens consumed during generation.
    pub completion_tokens: u32,
    /// Prompt tokens served from cache.
    pub cache_read_tokens: u32,
    /// Prompt tokens written to cache.
    pub cache_creation_tokens: u32,
    /// Why the stream ended (stop, length limit, etc.).
    pub finish_reason: FinishReason,
}

/// Stream LLM completions for the current conversation, emitting [`AgentEvent`]s
/// incrementally and returning a `oneshot` with the assembled [`LlmResponse`].
///
/// Handles:
/// - Text and reasoning token deltas → `TextDelta` / `ReasoningDelta` events
/// - Tool call streaming (`ToolCallStart` / `ToolCallDelta`) → assembled `LlmResponse.tool_calls`
/// - Usage telemetry → `AgentEvent::Usage`
/// - Cancellation via `cancel_token`
pub fn stream_llm_response<'a>(
    provider: &'a dyn LlmProvider,
    llm: &'a AgentLlmConfig,
    cancel_token: &'a CancellationToken,
    conversation: &mut Conversation,
    tool_definitions: &'a [ToolDefinition],
) -> AgentStream<'a, AgentEvent, anyhow::Result<LlmResponse>> {
    conversation.sanitize();
    let messages = conversation.to_messages();

    AgentStream::new(async_stream::stream! {
        use futures_util::StreamExt;

        let mut request = CompletionRequest::from_config(llm, messages);
        request.tools = tool_definitions.to_vec();

        let mut llm_stream = provider.complete(request);

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut pending_calls: Vec<PendingToolCall> = Vec::new();
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut cache_read_tokens: u32 = 0;
        let mut cache_creation_tokens: u32 = 0;
        let mut finish_reason = None;

        let outcome: anyhow::Result<FinishReason> = loop {
            let event = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => {
                    break Err(anyhow::anyhow!("Cancelled"));
                }
                event = llm_stream.next() => event,
            };

            match event {
                Some(Ok(StreamEvent::ReasoningDelta(delta))) => {
                    reasoning.push_str(&delta);
                    yield Outcome::Item(AgentEvent::ReasoningDelta(delta));
                }
                Some(Ok(StreamEvent::ContentDelta(delta))) => {
                    content.push_str(&delta);
                    yield Outcome::Item(AgentEvent::TextDelta(delta));
                }
                Some(Ok(StreamEvent::ToolCallStart { index, name, .. })) => {
                    while pending_calls.len() <= index {
                        pending_calls.push(PendingToolCall::default());
                    }
                    pending_calls[index].id = format!("{:08x}", Uuid::new_v4().as_fields().0);
                    pending_calls[index].name = name;
                }
                Some(Ok(StreamEvent::ToolCallDelta { index, arguments })) => {
                    if index < pending_calls.len() {
                        pending_calls[index].arguments_json.push_str(&arguments);
                    } else {
                        tracing::warn!(
                            pending_calls = pending_calls.len(),
                            current = index,
                            "Agent calling more tools than announced",
                        );
                    }
                }
                Some(Ok(StreamEvent::AudioDelta { data, format })) => {
                    yield Outcome::Item(AgentEvent::AudioChunk { data, format });
                }
                Some(Ok(StreamEvent::FileAttachment { filename, media_type, .. })) => {
                    tracing::debug!(filename = %filename, media_type = %media_type, "Received file attachment (not handled by agent)");
                }
                Some(Ok(StreamEvent::Usage(usage))) => {
                    prompt_tokens = usage.prompt_tokens;
                    completion_tokens = usage.completion_tokens;
                    cache_read_tokens = usage.cache_read_tokens;
                    cache_creation_tokens = usage.cache_creation_tokens;
                    yield Outcome::Item(AgentEvent::Usage(usage));
                }
                Some(Ok(StreamEvent::Finished(reason))) => {
                    tracing::info!(
                        "LLM response: finish_reason={reason:?}, content_len={}, tool_calls={}",
                        content.len(),
                        pending_calls.len()
                    );
                    finish_reason = Some(reason);
                }
                Some(Err(error)) => {
                    let connection_failed = error.chain().any(|cause| {
                        cause.downcast_ref::<IoError>().is_some_and(|error| matches!(
                            error.kind(),
                            ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted
                                | ErrorKind::BrokenPipe | ErrorKind::UnexpectedEof
                                | ErrorKind::TimedOut | ErrorKind::NotConnected
                        ))
                    });
                    let error = if error.downcast_ref::<LlmError>().is_none() && connection_failed {
                        let message = format!("LLM stream connection interrupted: {error}");
                        error.context(LlmError::new(LlmErrorKind::StreamError, message))
                    } else {
                        error
                    };
                    break Err(error);
                }
                None => break finish_reason.ok_or_else(|| {
                    LlmError::new(
                        LlmErrorKind::StreamError,
                        "LLM stream ended without a completion signal",
                    ).into()
                }),
            }
        };

        if let Err(error) = &outcome
            && !cancel_token.is_cancelled()
        {
            tracing::warn!(
                kind = error.downcast_ref::<LlmError>().map(|error| error.kind.as_str()),
                content_bytes = content.len(),
                reasoning_bytes = reasoning.len(),
                pending_tool_calls = pending_calls.len(),
                finish_received = finish_reason.is_some(),
                "LLM response stream failed"
            );
        }

        yield Outcome::Done(outcome.map(|finish_reason| {
            metrics::counter!("llm.finish_reasons").increment(1);
            metrics::histogram!("llm.prompt_tokens").record(prompt_tokens as f64);
            metrics::histogram!("llm.completion_tokens").record(completion_tokens as f64);
            LlmResponse {
                content,
                reasoning: if reasoning.is_empty() { None } else { Some(reasoning) },
                tool_calls: finalize_tool_calls(pending_calls),
                prompt_tokens,
                completion_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                finish_reason,
            }
        }));
    })
}

fn finalize_tool_calls(pending: Vec<PendingToolCall>) -> Vec<ToolCall> {
    pending
        .into_iter()
        .filter(|tc| !tc.id.is_empty() && !tc.name.is_empty())
        .map(|tc| {
            let arguments = match serde_json::from_str(&tc.arguments_json) {
                Ok(args) => args,
                Err(e) => {
                    tracing::warn!(tool = tc.name, "Malformed tool call arguments: {e}");
                    serde_json::json!({ "__malformed__": tc.arguments_json })
                }
            };
            ToolCall {
                id: tc.id,
                name: tc.name,
                arguments,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use flashmind_types::{CompletionStream, Provider, TokenUsage};

    use super::*;

    struct MockProvider {
        events: Mutex<Vec<anyhow::Result<StreamEvent>>>,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            let events = self.events.lock().unwrap().drain(..).collect::<Vec<_>>();
            Box::pin(futures_util::stream::iter(events))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn provider(&self) -> Provider {
            Provider::OpenAi
        }
    }

    async fn response(events: Vec<anyhow::Result<StreamEvent>>) -> anyhow::Result<LlmResponse> {
        let provider = MockProvider {
            events: Mutex::new(events),
        };
        let llm = AgentLlmConfig::new("openai:test".parse().unwrap());
        let cancel = CancellationToken::new();
        let mut conversation = Conversation::new();
        let mut stream = stream_llm_response(&provider, &llm, &cancel, &mut conversation, &[]);
        while stream.next().await.is_some() {}
        stream.take_result().unwrap()
    }

    #[tokio::test]
    async fn tool_call_ids_are_eight_lowercase_hex_digits() {
        let response = response(vec![
            Ok(StreamEvent::ToolCallStart {
                index: 0,
                id: String::new(),
                name: "example".into(),
            }),
            Ok(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            }),
            Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
        ])
        .await
        .unwrap();

        assert_eq!(response.tool_calls.len(), 1);
        let id = &response.tool_calls[0].id;
        assert_eq!(id.len(), 8);
        assert!(
            id.bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[tokio::test]
    async fn provider_eof_without_finish_is_a_stream_error() {
        let error = response(vec![Ok(StreamEvent::ContentDelta("partial".into()))])
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<LlmError>().unwrap().kind,
            LlmErrorKind::StreamError
        );
    }

    #[tokio::test]
    async fn connection_failure_after_content_is_retryable_and_keeps_its_cause() {
        for kind in [
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionAborted,
            ErrorKind::BrokenPipe,
            ErrorKind::UnexpectedEof,
            ErrorKind::TimedOut,
            ErrorKind::NotConnected,
        ] {
            let error = response(vec![
                Ok(StreamEvent::ContentDelta("partial".into())),
                Err(
                    anyhow::Error::new(IoError::new(kind, "connection interrupted"))
                        .context("Reading provider response"),
                ),
            ])
            .await
            .unwrap_err();
            assert_eq!(
                error.downcast_ref::<LlmError>().map(|error| error.kind),
                Some(LlmErrorKind::StreamError)
            );
            assert_eq!(error.downcast_ref::<IoError>().unwrap().kind(), kind);
        }
    }

    #[tokio::test]
    async fn non_connection_errors_keep_their_classification() {
        for error in [
            anyhow::Error::new(LlmError::new(LlmErrorKind::Other, "Invalid credentials")),
            IoError::new(ErrorKind::PermissionDenied, "Access denied").into(),
            anyhow::Error::new(IoError::new(ErrorKind::ConnectionReset, "connection reset"))
                .context(LlmError::new(LlmErrorKind::Other, "Invalid request")),
        ] {
            let message = error.to_string();
            let error = response(vec![
                Ok(StreamEvent::ContentDelta("partial".into())),
                Err(error),
            ])
            .await
            .unwrap_err();
            assert_eq!(error.to_string(), message);
            assert_ne!(
                error.downcast_ref::<LlmError>().map(|error| error.kind),
                Some(LlmErrorKind::StreamError)
            );
        }
    }

    #[tokio::test]
    async fn usage_after_finish_is_retained() {
        let response = response(vec![
            Ok(StreamEvent::ContentDelta("complete".into())),
            Ok(StreamEvent::Finished(FinishReason::Length)),
            Ok(StreamEvent::Usage(TokenUsage {
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
                cache_read_tokens: 2,
                cache_creation_tokens: 1,
            })),
        ])
        .await
        .unwrap();
        assert_eq!(response.content, "complete");
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.prompt_tokens, 7);
        assert_eq!(response.completion_tokens, 3);
        assert_eq!(response.cache_read_tokens, 2);
        assert_eq!(response.cache_creation_tokens, 1);
    }

    #[tokio::test]
    async fn provider_error_after_finish_remains_authoritative() {
        let error = response(vec![
            Ok(StreamEvent::Finished(FinishReason::Stop)),
            Err(LlmError::new(LlmErrorKind::StreamError, "connection dropped").into()),
        ])
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "connection dropped");
    }
}
