//! Shared SSE (Server-Sent Events) stream processing logic for OpenAI-compatible
//! providers, including **OpenRouter**, **OpenAI**, **vLLM**, and **LiteLLM**.
//!
//! These providers all speak the same streaming wire protocol: each SSE event carries
//! a JSON [`StreamChunk`] whose fields may be partial deltas rather than complete
//! values. This module parses those chunks into provider-agnostic [`StreamEvent`]
//! enums that downstream code can handle uniformly.
//!
//! **Note:** Anthropic and Ollama use entirely different streaming formats and each
//! has its own dedicated parser; this module is *only* for the OpenAI-compatible
//! protocol family.

use crate::wire_types::{StreamChunk, StreamToolCallDelta};
use eventsource_stream::EventStreamError;
use flashmind_types::{FinishReason, LlmError, LlmErrorKind, StreamEvent, TokenUsage};

pub(crate) fn chunk_error(chunk: &StreamChunk) -> Option<LlmError> {
    let Some(error) = chunk.error.as_ref() else {
        return chunk
            .choices
            .iter()
            .any(|choice| choice.finish_reason.as_deref() == Some("error"))
            .then(|| {
                LlmError::new(
                    LlmErrorKind::StreamError,
                    "Provider ended stream with finish_reason=error",
                )
            });
    };
    let mut classified = LlmError::classify(format!(
        "LLM provider stream error ({}): {}",
        error.code.as_ref().unwrap_or(&serde_json::Value::Null),
        error.message,
    ));
    if classified.kind == LlmErrorKind::Other {
        classified.kind = match error.code.as_ref().and_then(serde_json::Value::as_u64) {
            Some(429) => LlmErrorKind::StreamError,
            Some(400..=499) => LlmErrorKind::Other,
            _ => LlmErrorKind::StreamError,
        };
    } else if classified.kind == LlmErrorKind::RateLimited {
        // HTTP retries cannot see rate limits delivered inside a successful response.
        classified.kind = LlmErrorKind::StreamError;
    }
    Some(classified)
}

pub(crate) fn stream_error_code(chunk: &StreamChunk) -> Option<&str> {
    let code = chunk.error.as_ref()?.code.as_ref()?.as_str()?;
    // Provider messages and arbitrary metadata may contain request data or credentials.
    Some(match code {
        "server_error"
        | "rate_limit_exceeded"
        | "rate_limit_error"
        | "context_length_exceeded"
        | "invalid_request_error"
        | "insufficient_quota" => code,
        _ => "other",
    })
}

pub(crate) fn stream_error_kind<E>(error: &EventStreamError<E>) -> &'static str {
    match error {
        EventStreamError::Transport(_) => "transport",
        EventStreamError::Parser(_) => "parser",
        EventStreamError::Utf8(_) => "utf8",
    }
}

/// Tracks streaming tool call state across multiple SSE deltas so that the caller
/// receives structured, deduplicated events.
///
/// In the OpenAI streaming protocol, a single logical tool call is delivered across
/// several chunks: one delta may carry the `id`, another the `name`, and subsequent
/// ones carry incremental `arguments`.  This tracker accumulates those fragments so
/// we can emit a proper [`StreamEvent::ToolCallStart`] exactly once, followed by
/// argument-delta events as they arrive.
///
/// # Internal representation
///
/// The `calls` vector holds one entry per tool-call index, where each entry is the
/// tuple `(id: String, name: String, start_emitted: bool)`.  The `start_emitted`
/// flag guards against duplicate [`StreamEvent::ToolCallStart`] events — it is set
/// to `true` immediately after the first such event is emitted for that index.
#[derive(Default)]
pub struct ToolCallTracker {
    /// (id, name, start_emitted) for each tool call index.
    calls: Vec<(String, String, bool)>,
}

impl ToolCallTracker {
    /// Process a single tool-call delta and return the [`StreamEvent`]s to yield.
    ///
    /// The `delta` argument is one entry from `choice.delta.tool_calls` inside an SSE
    /// chunk.  Because fields arrive piecemeal, this method accumulates `id`, `name`,
    /// and `arguments` across calls before emitting events.
    ///
    /// # Returns
    ///
    /// A (possibly empty) list of [`StreamEvent`] variants:
    /// - **0** — if no new arguments arrived and the start event was already emitted
    /// - **1** — typically just a [`StreamEvent::ToolCallDelta`] with arguments, or
    ///   only a [`StreamEvent::ToolCallStart`] if we just received both `id` and
    ///   `name` but no arguments yet
    /// - **2+** — a start event *and* one or more argument deltas in the same call
    ///
    /// # Index handling
    ///
    /// The delta's `index` field defaults to `0` when absent (single-tool-call
    /// requests often omit it).
    pub fn process_delta(&mut self, delta: &StreamToolCallDelta) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let idx = delta.index.unwrap_or(0);

        // Grow state tracker as needed
        while self.calls.len() <= idx {
            self.calls.push((String::new(), String::new(), false));
        }

        // Update id and name
        if let Some(ref id) = delta.id {
            self.calls[idx].0.clone_from(id);
        }
        if let Some(ref func) = delta.function
            && let Some(ref name) = func.name
        {
            self.calls[idx].1.clone_from(name);
        }

        // Emit start event once we have both id and name
        let (ref id, ref name, ref mut emitted) = self.calls[idx];
        if !*emitted && !id.is_empty() && !name.is_empty() {
            events.push(StreamEvent::ToolCallStart {
                index: idx,
                id: id.clone(),
                name: name.clone(),
            });
            *emitted = true;
        }

        // Emit argument deltas
        if let Some(ref func) = delta.function
            && let Some(ref args) = func.arguments
            && !args.is_empty()
        {
            events.push(StreamEvent::ToolCallDelta {
                index: idx,
                arguments: args.clone(),
            });
        }

        events
    }
}

/// Process a single SSE JSON chunk into provider-agnostic [`StreamEvent`]s.
///
/// Parses every field in the [`StreamChunk`] — content deltas, reasoning deltas,
/// tool-call deltas, and usage information — and converts them into a uniform list
/// of [`StreamEvent`] variants that downstream code can handle regardless of which
/// OpenAI-compatible provider produced the stream.
///
/// # Arguments
///
/// * `chunk` — The deserialized SSE chunk from the wire.
/// * `tracker` — A **mutable** reference to a [`ToolCallTracker`].  It is mutable
///   because tool-call state must persist across chunks (the `id`, `name`, and
///   incremental arguments arrive separately), so the tracker mutates its internal
///   buffers as each new delta flows through.
///
/// # Returns
///
/// `(events, finish_reason)` where:
/// - `events` is a `Vec<StreamEvent>` containing everything emitted from this chunk
/// - `finish_reason` is `Some` only on the final chunk; the raw string from the API
///   is parsed via [`str::parse`] with a fallback to [`FinishReason::Stop`] for any
///   unrecognized value
pub fn process_chunk(
    chunk: &StreamChunk,
    tracker: &mut ToolCallTracker,
) -> (Vec<StreamEvent>, Option<FinishReason>) {
    let mut events = Vec::new();
    let mut finish = None;

    // Yield usage if present
    if let Some(ref u) = chunk.usage {
        tracing::debug!(
            prompt_tokens = u.prompt_tokens,
            completion_tokens = u.completion_tokens,
            "SSE chunk: usage"
        );
        let details = u.prompt_tokens_details.as_ref();
        let cached = details.map_or(0, |d| d.cached_tokens);
        let written = details.map_or(0, |d| d.cache_write_tokens);
        events.push(StreamEvent::Usage(TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cache_read_tokens: cached,
            cache_creation_tokens: written,
        }));
    }

    for choice in chunk.choices.iter() {
        // Reasoning delta
        if let Some(ref reasoning) = choice.delta.reasoning
            && !reasoning.is_empty()
        {
            events.push(StreamEvent::ReasoningDelta(reasoning.clone()));
        }

        // Content delta
        if let Some(ref content) = choice.delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::ContentDelta(content.clone()));
        }

        // Tool call deltas
        if let Some(ref tc_deltas) = choice.delta.tool_calls {
            for tc_delta in tc_deltas {
                events.extend(tracker.process_delta(tc_delta));
            }
        }

        // Audio output delta
        if let Some(ref audio) = choice.delta.audio
            && let Some(ref data) = audio.data
            && !data.is_empty()
        {
            use base64::Engine;
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data) {
                events.push(StreamEvent::AudioDelta {
                    data: bytes,
                    format: audio.format.clone().unwrap_or_else(|| "pcm16".into()),
                });
            }
        }

        // Generated images inside delta (OpenAI image models)
        if let Some(ref images) = choice.delta.images {
            extract_images(images, &mut events);
        }

        // Capture finish reason
        if let Some(ref reason) = choice.finish_reason {
            finish = Some(reason.parse().unwrap_or(FinishReason::Stop));
        }
    }

    // Generated images at top level
    if let Some(ref images) = chunk.images {
        extract_images(images, &mut events);
    }

    (events, finish)
}

/// Extract generated images from a slice of [`StreamImage`]s, pushing
/// [`StreamEvent::FileAttachment`] events for each valid data URL found.
fn extract_images(images: &[crate::wire_types::StreamImage], events: &mut Vec<StreamEvent>) {
    for img in images {
        if let Some(url) = img.data_url()
            && let Some((media_type, data)) = parse_data_url(url)
        {
            events.push(StreamEvent::FileAttachment {
                filename: String::new(),
                media_type,
                data,
            });
        }
    }
}

/// Parse a `data:<media_type>;base64,<data>` URL into `(media_type, decoded_bytes)`.
fn parse_data_url(url: &str) -> Option<(String, Vec<u8>)> {
    use base64::Engine;
    let url = url.strip_prefix("data:")?;
    let (meta, b64) = url.split_once(',')?;
    let media_type = meta.strip_suffix(";base64")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    Some((media_type.to_string(), bytes))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{chunk_error, stream_error_code};
    use crate::wire_types::StreamChunk;
    use flashmind_types::{LlmError, LlmErrorKind, StreamEvent};
    use reqwest::{Client, Response};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    pub(crate) const CONTENT: &str =
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
    pub(crate) const FINISH: &str =
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n";
    pub(crate) const USAGE: &str = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n";

    pub(crate) async fn response_url(body: String, transport_failure: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                let headers = String::from_utf8_lossy(&request);
                if let Some(end) = headers.find("\r\n\r\n") {
                    let length = headers[..end]
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.eq_ignore_ascii_case("content-length") {
                                value.trim().parse::<usize>().ok()
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let declared_length = body.len() + if transport_failure { 64 } else { 0 };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n{body}"
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        url
    }

    pub(crate) async fn response(body: String, transport_failure: bool) -> Response {
        Client::new()
            .get(response_url(body, transport_failure).await)
            .send()
            .await
            .unwrap()
    }

    pub(crate) fn assert_stream_error(events: &[anyhow::Result<StreamEvent>]) {
        assert!(events.iter().any(|event| {
            event.as_ref().err().is_some_and(|error| {
                error
                    .downcast_ref::<LlmError>()
                    .is_some_and(|error| error.kind == LlmErrorKind::StreamError)
            })
        }));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(StreamEvent::Finished(_))))
        );
    }

    #[test]
    fn provider_stream_errors_keep_permanent_and_context_failures_distinct() {
        for (code, expected) in [
            (serde_json::json!(502), LlmErrorKind::StreamError),
            (serde_json::json!(429), LlmErrorKind::StreamError),
            (
                serde_json::json!("rate_limit_exceeded"),
                LlmErrorKind::StreamError,
            ),
            (serde_json::json!(401), LlmErrorKind::Other),
            (
                serde_json::json!("context_length_exceeded"),
                LlmErrorKind::ContextLengthExceeded,
            ),
        ] {
            let chunk: StreamChunk = serde_json::from_value(serde_json::json!({
                "error": {"code": code, "message": "Request failed"}
            }))
            .unwrap();
            assert_eq!(chunk_error(&chunk).unwrap().kind, expected);
        }
    }

    #[test]
    fn finish_reason_error_without_details_is_a_failure() {
        let chunk: StreamChunk = serde_json::from_value(serde_json::json!({
            "choices": [{"delta": {}, "finish_reason": "error"}]
        }))
        .unwrap();
        assert_eq!(chunk_error(&chunk).unwrap().kind, LlmErrorKind::StreamError);
    }

    #[test]
    fn arbitrary_provider_error_codes_are_not_logged() {
        let chunk: StreamChunk = serde_json::from_value(serde_json::json!({
            "error": {"code": "private-request-data", "message": "Request failed"}
        }))
        .unwrap();
        assert_eq!(stream_error_code(&chunk), Some("other"));
    }
}
