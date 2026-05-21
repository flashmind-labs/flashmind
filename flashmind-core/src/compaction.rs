//! Proactive context compaction when token usage exceeds threshold.
//!
//! The [`try_compact`] function is called after each LLM response.
//! It runs an escalation ladder to reduce conversation size:
//!
//! 1. **Truncate long tool outputs** — cap at 200 bytes, append `[truncated]`
//! 2. **LLM summarization** — send remaining entries to the compaction model with [`COMPACTION_PROMPT`]
//! 3. **Strip tool messages** — remove Tool entries and clear tool_calls from Assistant entries
//! 4. **Last exchange fallback** — keep only system prompt + last user/assistant pair
//!
//! # When compaction triggers
//!
//! - **Proactive**: Prompt tokens exceed 90% of the context window (after any turn)
//! - **Reactive**: Model returns `finish_reason=Length` without explicit `max_tokens` set

use futures::Stream;

use flashmind_types::{AgentEvent, LlmProvider, Model};

use crate::conversation::Conversation;

/// System prompt sent to the compaction model when LLM summarization is triggered.
///
/// Instructs the model to preserve recent context in detail while summarizing
/// older exchanges, retaining key facts (file paths, URLs, IDs) and dropping
/// routine tool call details.
pub const COMPACTION_PROMPT: &str = "\
Summarize this conversation for context continuity. This summary replaces the original messages.\n\n\
Guidelines:\n\
- Preserve ALL recent context in detail (last few exchanges)\n\
- Summarize older exchanges more briefly — key decisions, facts, and outcomes only\n\
- Include timestamps for important events and decisions\n\
- Preserve: file paths, URLs, names, IDs, code snippets, and technical details that may be referenced later\n\
- Preserve: the user's current task, goals, and any pending work\n\
- Drop: routine tool call details, intermediate debugging steps, and verbose outputs";

/// Compact the conversation proactively when prompt tokens exceed 90% of the context window.
///
/// Returns a stream that yields `Compacted` (with the summary text) once done,
/// or `Error` if all fallbacks fail and we resort to truncating to the last exchange.
/// The caller must drive the stream to completion before the next turn begins.
pub fn try_compact<'a>(
    conversation: &'a mut Conversation,
    estimated_tokens: u32,
    context_window: u32,
    compaction_provider: &'a dyn LlmProvider,
    compaction_model: &'a Model,
) -> impl Stream<Item = AgentEvent> + 'a {
    async_stream::stream! {
        let pct = (estimated_tokens as f64 / context_window as f64 * 100.0) as u32;
        let entries_before = conversation.entries().len();
        metrics::counter!("agent.compactions.triggered").increment(1);
        tracing::debug!(
            "Compaction triggered: {pct}% capacity ({estimated_tokens}/{context_window} tokens), {entries_before} entries"
        );

        yield AgentEvent::Status(format!(
            "Compacting conversation ({pct}% context used)..."
        ));

        let truncated = conversation.truncate_long_tool_outputs(200);
        if truncated > 0 {
            tracing::debug!("Truncated {truncated} long tool outputs before summarization");
        }

        // Short conversations don't benefit from LLM summarization — the
        // summary destroys tool call context and rarely saves much space.
        // Apply only mechanical steps (truncate above + strip below).
        const MIN_ENTRIES_FOR_LLM_COMPACT: usize = 15;
        if conversation.summarizable_entry_count() < MIN_ENTRIES_FOR_LLM_COMPACT {
            tracing::info!(
                "Conversation too short for LLM compaction ({} entries < {MIN_ENTRIES_FOR_LLM_COMPACT}), applying mechanical compaction only",
                conversation.summarizable_entry_count(),
            );
            let stripped = conversation.strip_tool_messages();
            if stripped > 0 {
                tracing::debug!("Stripped {stripped} tool messages (short conversation fallback)");
            }
            if truncated > 0 || stripped > 0 {
                metrics::counter!("agent.compactions.succeeded").increment(1);
                yield AgentEvent::Compacted(
                    "[compacted — truncated tool outputs and stripped tool messages]".into(),
                );
            } else {
                metrics::counter!("agent.compactions.failed").increment(1);
                tracing::warn!("Short conversation with nothing to compact mechanically — truncating to last exchange");
                conversation.truncate_to_last_exchange();
                yield AgentEvent::Compacted(
                    "[compacted via fallback — conversation too short for summarization]".into(),
                );
            }
            return;
        }

        let first_attempt = conversation
            .compact_with_llm(compaction_provider, compaction_model)
            .await;

        match first_attempt {
            Ok(Some(s)) => {
                let entries_after = conversation.entries().len();
                metrics::counter!("agent.compactions.succeeded").increment(1);
                metrics::gauge!("agent.compaction.entries_before").set(entries_before as f64);
                metrics::gauge!("agent.compaction.entries_after").set(entries_after as f64);
                tracing::debug!("Compaction complete: {entries_before} → {entries_after} entries");
                yield AgentEvent::Compacted(s);
            }
            other => {
                if let Err(ref e) = other {
                    tracing::warn!("Compaction LLM call errored, falling back: {e}");
                } else {
                    tracing::warn!("LLM summarization returned empty; pruning tool outputs and retrying");
                }

                let stripped = conversation.strip_tool_messages();
                if stripped > 0 {
                    tracing::debug!("Stripped {stripped} tool messages as compaction fallback");
                    match conversation.compact_with_llm(compaction_provider, compaction_model).await {
                        Ok(Some(s)) => {
                            let entries_after = conversation.entries().len();
                            metrics::counter!("agent.compactions.succeeded").increment(1);
                            metrics::gauge!("agent.compaction.entries_before").set(entries_before as f64);
                            metrics::gauge!("agent.compaction.entries_after").set(entries_after as f64);
                            tracing::debug!("Compaction complete after strip: {entries_before} → {entries_after} entries");
                            yield AgentEvent::Compacted(s);
                            return;
                        }
                        Ok(None) => tracing::warn!("LLM compaction returned empty after strip"),
                        Err(e) => tracing::warn!("LLM compaction errored after strip: {e}"),
                    }
                }

                metrics::counter!("agent.compactions.failed").increment(1);
                tracing::warn!("All compaction attempts failed — truncating to last exchange");
                conversation.truncate_to_last_exchange();
                yield AgentEvent::Compacted(
                    "[compacted via fallback — LLM summarization unavailable]".into(),
                );
            }
        }
    }
}
