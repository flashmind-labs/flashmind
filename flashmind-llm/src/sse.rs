//! Shared SSE stream processing logic for OpenAI-compatible providers.

use tracing::debug;

use crate::wire_types::{StreamChunk, StreamToolCallDelta};
use flashmind_types::{FinishReason, StreamEvent, TokenUsage};

/// Tracks streaming tool call state across multiple SSE deltas.
#[derive(Default)]
pub struct ToolCallTracker {
    /// (id, name, start_emitted) for each tool call index.
    calls: Vec<(String, String, bool)>,
}

impl ToolCallTracker {
    /// Process a tool call delta, returning events to emit.
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

/// Process a single SSE chunk, yielding stream events.
/// Returns `(events, optional_finish_reason)`.
pub fn process_chunk(
    chunk: &StreamChunk,
    tracker: &mut ToolCallTracker,
) -> (Vec<StreamEvent>, Option<FinishReason>) {
    let mut events = Vec::new();
    let mut finish = None;

    // Yield usage if present
    if let Some(ref u) = chunk.usage {
        debug!(
            prompt_tokens = u.prompt_tokens,
            completion_tokens = u.completion_tokens,
            "SSE chunk: usage"
        );
        events.push(StreamEvent::Usage(TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
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

        // Capture finish reason
        if let Some(ref reason) = choice.finish_reason {
            // debug!(
            //     choice = choice_idx,
            //     reason = %reason,
            //     "SSE chunk: finish reason"
            // );
            finish = Some(reason.parse().unwrap_or(FinishReason::Stop));
        }
    }

    (events, finish)
}
