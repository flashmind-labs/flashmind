//! LLM response streaming and tool-call assembly.
//!
//! Wraps the raw provider stream into an [`AgentEvent`] stream and assembles
//! incremental text deltas / reasoning tokens into complete [`LlmResponse`] values.
//!
//! # Flow
//!
//! 1. Converts conversation entries to wire-format messages via `Conversation::to_messages()`
//! 2. Sends a [`CompletionRequest`] to the provider
//! 3. Drives the [`CompletionStream`] event loop, handling cancellation
//! 4. Emits `TextDelta`, `ReasoningDelta`, `Usage` events as they arrive
//! 5. Assembles partial tool call deltas into complete [`ToolCall`] objects
//! 6. Returns the final [`LlmResponse`] with content, tool calls, token counts, and finish reason
//!
//! # Tool call assembly
//!
//! Streaming tool calls arrive fragmented across multiple SSE chunks. The internal
//! [`PendingToolCall`] struct accumulates id/name/arguments until the stream finishes,
//! then [`finalize_tool_calls`] parses the JSON and produces structured `ToolCall` values.

use std::path::Path;

use tokio_util::sync::CancellationToken;

use flashmind_types::{
    AgentEvent, AgentLlmConfig, AgentStream, CompletionRequest, FinishReason, LlmProvider, Outcome,
    StreamEvent, ToolCall, ToolDefinition,
};

use crate::conversation::Conversation;

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

/// Fully-assembled response from the LLM after the stream completes.
pub struct LlmResponse {
    /// Plain text content of the assistant message.
    pub content: String,
    /// Tool call requests (empty if no tools were requested).
    pub tool_calls: Vec<ToolCall>,
    /// Token counts reported by the provider.
    pub prompt_tokens: u32,
    /// Completion tokens consumed during generation.
    pub completion_tokens: u32,
    /// Why the stream ended (stop, length limit, etc.).
    pub finish_reason: FinishReason,
}

/// Stream LLM completions for the current conversation, emitting [`AgentEvent`]s
/// incrementally and returning a `oneshot` with the assembled [`LlmResponse`].
///
/// Handles:
/// - Text and reasoning token deltas → `TextDelta` / `ReasoningDelta` events
/// - Tool call streaming (`ToolCallStart` / `ToolCallDelta`) → assembled `LlmResponse.tool_calls`
/// - File attachments → saved to `~/.flashagent/downloads/` as `AgentEvent::Status`
/// - Usage telemetry → `AgentEvent::Usage`
/// - Cancellation via `cancel_token`
pub fn stream_llm_response<'a>(
    provider: &'a dyn LlmProvider,
    llm: &'a AgentLlmConfig,
    cancel_token: &'a CancellationToken,
    conversation: &mut Conversation,
    tool_definitions: &'a [ToolDefinition],
    downloads_dir: Option<&'a Path>,
) -> AgentStream<'a, AgentEvent, anyhow::Result<LlmResponse>> {
    conversation.sanitize();
    let messages = conversation.to_messages();

    AgentStream::new(async_stream::stream! {
        use futures::StreamExt;

        let request = CompletionRequest {
            model: llm.model.clone(),
            messages,
            tools: tool_definitions.to_vec(),
            max_tokens: llm.max_tokens,
            reasoning: llm.reasoning.clone(),
            sampling: llm.sampling.clone(),
            modalities: vec![],
            audio_config: None,
            image_config: None,
        };

        let mut llm_stream = provider.complete(request);

        let mut content = String::new();
        let mut pending_calls: Vec<PendingToolCall> = Vec::new();
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut finish_reason = FinishReason::Stop;

        let outcome: anyhow::Result<()> = loop {
            let event = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => {
                    break Err(anyhow::anyhow!("Cancelled"));
                }
                event = llm_stream.next() => event,
            };

            match event {
                Some(Ok(StreamEvent::ReasoningDelta(delta))) => {
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
                    pending_calls[index].id = format!("{:08x}", rand::random::<u32>());
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
                Some(Ok(StreamEvent::FileAttachment { filename, media_type, data })) => {
                    tracing::debug!(filename = %filename, media_type = %media_type, "Received file attachment from server");
                    if let Some(dir) = downloads_dir
                        && let Some(ev) = save_file_attachment(&filename, &media_type, &data, dir).await
                    {
                        yield Outcome::Item(ev);
                    }
                }
                Some(Ok(StreamEvent::Usage(usage))) => {
                    prompt_tokens = usage.prompt_tokens;
                    completion_tokens = usage.completion_tokens;
                    yield Outcome::Item(AgentEvent::Usage(usage));
                }
                Some(Ok(StreamEvent::Finished(reason))) => {
                    tracing::info!(
                        "LLM response: finish_reason={reason:?}, content_len={}, tool_calls={}",
                        content.len(),
                        pending_calls.len()
                    );
                    finish_reason = reason;
                    metrics::counter!("llm.finish_reasons").increment(1);
                    metrics::histogram!("llm.prompt_tokens").record(prompt_tokens as f64);
                    metrics::histogram!("llm.completion_tokens").record(completion_tokens as f64);
                    break Ok(());
                }
                Some(Err(e)) => break Err(anyhow::anyhow!("LLM stream error: {e}")),
                None => break Ok(()),
            }
        };

        yield Outcome::Done(outcome.map(|_| LlmResponse {
            content,
            tool_calls: finalize_tool_calls(pending_calls),
            prompt_tokens,
            completion_tokens,
            finish_reason,
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
                    tracing::warn!(
                        tool = tc.name,
                        args = tc.arguments_json,
                        "Malformed tool call arguments: {e}"
                    );
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

fn unique_path(dir: &std::path::Path, filename: &str) -> std::path::PathBuf {
    let base = dir.join(filename);

    if !base.exists() {
        return base;
    }

    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|s| s.to_str());

    for i in 1..100 {
        let name = match ext {
            Some(e) => format!("{stem}-{i}.{e}"),
            None => format!("{stem}-{i}"),
        };
        let candidate = dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }

    base
}

async fn save_file_attachment(
    filename: &str,
    _media_type: &str,
    bytes: &[u8],
    save_dir: &Path,
) -> Option<AgentEvent> {
    if let Err(e) = tokio::fs::create_dir_all(save_dir).await {
        tracing::warn!(dir = %save_dir.display(), error = %e, "Failed to create save directory");
        return None;
    }

    let safe_name = std::path::Path::new(filename)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download");
    let path = unique_path(save_dir, safe_name);

    if let Err(e) = tokio::fs::write(&path, &bytes).await {
        tracing::warn!(path = %path.display(), error = %e, "Failed to write file attachment");
        return None;
    }

    tracing::info!(path = %path.display(), bytes = bytes.len(), "Saved file attachment");
    Some(AgentEvent::Status(format!("Saved: {}", path.display())))
}
