//! Proactive context compaction when token usage exceeds threshold.
//!
//! The [`try_compact`] function is called after each LLM response.
//! It runs an escalation ladder to reduce conversation size:
//!
//! 1. **Truncate long tool outputs** — cap at 2000 bytes, append `[truncated]`
//! 2. **LLM summarization** — send remaining entries to the compaction model with [`COMPACTION_PROMPT`]
//! 3. **Prune all tool outputs** — replace with `[output pruned]`
//! 4. **Strip tool messages** — remove Tool entries and clear tool_calls from Assistant entries
//! 5. **Last exchange fallback** — keep only system prompt + last user/assistant pair
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
        tracing::info!(
            "Compaction triggered: {pct}% capacity ({estimated_tokens}/{context_window} tokens), {entries_before} entries"
        );

        yield AgentEvent::Status(format!(
            "Compacting conversation ({pct}% context used)..."
        ));

        let truncated = conversation.truncate_long_tool_outputs(2000);
        if truncated > 0 {
            tracing::info!("Truncated {truncated} long tool outputs before summarization");
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
                tracing::info!("Compaction complete: {entries_before} → {entries_after} entries");
                yield AgentEvent::Compacted(s);
            }
            // Err: the LLM call itself failed (auth, network, timeout, etc.).
            // Retrying the same call won't help — fall straight to the
            // non-LLM fallback ladder. Ok(None): nothing produced; retry once
            // after pruning to reduce input size.
            other => {
                if let Err(ref e) = other {
                    tracing::warn!("Compaction LLM call errored, falling back: {e}");
                } else {
                    tracing::warn!("LLM summarization returned empty; pruning tool outputs and retrying");
                }
                let llm_errored = other.is_err();

                let pruned = conversation.prune_tool_outputs(0);
                if pruned > 0 {
                    tracing::info!("Pruned {pruned} tool outputs as compaction fallback");
                }

                let stripped = conversation.strip_tool_messages();
                if stripped > 0 {
                    tracing::info!("Stripped {stripped} tool messages as compaction fallback");
                }

                // Only retry the LLM call when the prior attempt returned
                // Ok(None) AND we just reduced the input. If the prior call
                // errored, the model is unhappy — don't burn another call.
                if !llm_errored && (pruned > 0 || stripped > 0) {
                    match conversation
                        .compact_with_llm(compaction_provider, compaction_model)
                        .await
                    {
                        Ok(Some(s)) => {
                            let entries_after = conversation.entries().len();
                            metrics::counter!("agent.compactions.succeeded").increment(1);
                            metrics::gauge!("agent.compaction.entries_before").set(entries_before as f64);
                            metrics::gauge!("agent.compaction.entries_after").set(entries_after as f64);
                            tracing::info!("Compaction complete on retry: {entries_before} → {entries_after} entries");
                            yield AgentEvent::Compacted(s);
                            return;
                        }
                        Ok(None) => {
                            tracing::warn!("LLM summarization failed on retry — truncating to last exchange");
                        }
                        Err(e) => {
                            tracing::warn!("LLM summarization errored on retry: {e}");
                        }
                    }
                }

                metrics::counter!("agent.compactions.failed").increment(1);
                if pruned == 0 && stripped == 0 {
                    tracing::warn!("No pruning possible — truncating to last exchange");
                }
                conversation.truncate_to_last_exchange();
                yield AgentEvent::Compacted(
                    "[compacted via fallback — LLM summarization unavailable]".into(),
                );
            }
        }
    }
}
