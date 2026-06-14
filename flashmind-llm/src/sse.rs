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
use flashmind_types::{FinishReason, StreamEvent, TokenUsage};

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
        let cached = u
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        events.push(StreamEvent::Usage(TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cache_read_tokens: cached,
            cache_creation_tokens: 0,
        }));
    }

    for choice in chunk.choices.iter() {
        // Reasoning delta
        if let Some(ref reasoning) = choice.delta.reasoning
            && !reasoning.is_empty()
        {
            // debug!(
            //     choice = choice_idx,
            //     len = reasoning.len(),
            //     "SSE chunk: reasoning delta"
            // );
            events.push(StreamEvent::ReasoningDelta(reasoning.clone()));
        }

        // Content delta
        if let Some(ref content) = choice.delta.content
            && !content.is_empty()
        {
            // debug!(
            //     choice = choice_idx,
            //     content = %content,
            //     "SSE chunk: content delta"
            // );
            events.push(StreamEvent::ContentDelta(content.clone()));
        }

        // Tool call deltas
        if let Some(ref tc_deltas) = choice.delta.tool_calls {
            for tc_delta in tc_deltas {
                // debug!(
                //     choice = choice_idx,
                //     index = ?tc_delta.index,
                //     id = ?tc_delta.id,
                //     name = ?tc_delta.function.as_ref().and_then(|f| f.name.as_ref()),
                //     args_len = tc_delta.function.as_ref().and_then(|f| f.arguments.as_ref()).map(|a| a.len()),
                //     "SSE chunk: tool call delta"
                // );
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
