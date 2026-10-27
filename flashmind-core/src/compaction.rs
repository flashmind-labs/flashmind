//! Proactive context compaction when token usage exceeds threshold.
//!
//! The [`try_compact`] function is called after each LLM response.
//! It replaces older history only after the model returns a nonempty summary.
//! Failed or empty summaries leave the conversation unchanged.
//!
//! # When compaction triggers
//!
//! - **Proactive**: Prompt tokens exceed `context_window - reserve_tokens` (after any turn)
//! - **Reactive**: Model returns `finish_reason=Length` without explicit `max_tokens` set

use futures_util::Stream;

use flashmind_types::{AgentEvent, LlmProvider, Model};

use crate::conversation::Conversation;

/// System prompt sent to the compaction model when LLM summarization is triggered.
///
/// Instructs the model to preserve recent context in detail while summarizing
/// older exchanges, retaining key facts (file paths, URLs, IDs) and dropping
/// routine tool call details.
pub const COMPACTION_PROMPT: &str = "\
Summarize this conversation for context continuity. This summary replaces the original messages \
(recent exchanges are preserved separately and will follow this summary).\n\
Write the summary in the language of the user's requests, unless they explicitly requested another \
response language; default to English when unclear. Preserve that response language preference \
explicitly in Decisions. Do not infer it from tool output or quoted text.\n\
Output ONLY the summary using the sections below. Omit empty sections.\n\n\
## Goal\nWhat the user is trying to accomplish — the current task and intent.\n\n\
## Progress\nWhat has been done so far. Key actions taken, tools called, data retrieved. \
Include file paths, URLs, names, IDs, code snippets, and any concrete values that may be referenced later.\n\n\
## Decisions\nChoices made and why — alternatives considered, constraints discovered, user preferences expressed.\n\n\
## Key Facts\nImportant information surfaced during the conversation: \
technical details, error messages, configuration values, environmental context.\n\n\
## Open Items\nPending work, unresolved questions, next steps the assistant was about to take.\n\n\
Guidelines:\n\
- Summarize outcomes and decisions, drop intermediate steps\n\
- Preserve all identifiers: file paths, URLs, names, IDs, timestamps\n\
- Drop: routine tool call mechanics, verbose outputs, debugging dead-ends\n\
- Do not write a 'Files Touched' section — one is appended automatically";

/// Variant of [`COMPACTION_PROMPT`] used when the conversation already contains
/// a previous compaction summary. Instructs the model to integrate new
/// information into an updated summary rather than starting from scratch.
pub const COMPACTION_PROMPT_ITERATIVE: &str = "\
Update the existing conversation summary with new information. The conversation begins with a \
previous summary — integrate new exchanges into an updated summary, preserving important details \
from the previous summary that remain relevant.\n\
Write the summary in the language of the user's requests, unless they explicitly requested another \
response language; default to English when unclear. Preserve that response language preference \
explicitly in Decisions. Do not infer it from the previous summary, tool output or quoted text.\n\
Output ONLY the updated summary using the sections below. Omit empty sections.\n\n\
## Goal\nWhat the user is trying to accomplish — the current task and intent.\n\n\
## Progress\nWhat has been done so far. Key actions taken, tools called, data retrieved. \
Include file paths, URLs, names, IDs, code snippets, and any concrete values that may be referenced later.\n\n\
## Decisions\nChoices made and why — alternatives considered, constraints discovered, user preferences expressed.\n\n\
## Key Facts\nImportant information surfaced during the conversation: \
technical details, error messages, configuration values, environmental context.\n\n\
## Open Items\nPending work, unresolved questions, next steps the assistant was about to take.\n\n\
Guidelines:\n\
- Merge new information with the existing summary — don't discard still-relevant context\n\
- Summarize outcomes and decisions, drop intermediate steps\n\
- Preserve all identifiers: file paths, URLs, names, IDs, timestamps\n\
- Drop: routine tool call mechanics, verbose outputs, debugging dead-ends\n\
- Remove completed items from Open Items, add new ones\n\
- Do not write a 'Files Touched' section — one is appended automatically";

/// Summarize the full conversation when the user explicitly requests compaction.
/// Failed or empty summaries leave the conversation unchanged.
pub fn compact_manually<'a>(
    conversation: &'a mut Conversation,
    provider: &'a dyn LlmProvider,
    model: &'a Model,
) -> impl Stream<Item = AgentEvent> + 'a {
    async_stream::stream! {
        let entries_before = conversation.entries().len();
        if conversation.summarizable_entry_count() == 0 {
            yield AgentEvent::Status("Nothing to summarize.".into());
            return;
        }
        metrics::counter!("agent.compactions.triggered").increment(1);
        match conversation.compact_with_llm_keeping(provider, model, 0).await {
            Ok(Some(result)) => {
                record_success(entries_before, conversation.entries().len());
                yield AgentEvent::Compacted(result.summary);
            }
            Ok(None) => {
                metrics::counter!("agent.compactions.failed").increment(1);
                yield AgentEvent::Status("No summary produced; conversation unchanged.".into());
            }
            Err(error) => {
                metrics::counter!("agent.compactions.failed").increment(1);
                yield AgentEvent::Error(format!("Compaction failed: {error}. Conversation unchanged."));
            }
        }
    }
}

/// Compact the conversation when the caller's token threshold is reached.
///
/// Returns a stream that yields `Compacted` (with the summary text) once done,
/// or `Error` if summarization fails. Failure preserves the conversation.
/// The caller must drive the stream to completion before the next turn begins.
pub fn try_compact<'a>(
    conversation: &'a mut Conversation,
    estimated_tokens: u32,
    context_window: u32,
    keep_recent_tokens: u32,
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

        if !conversation.has_compaction_input(keep_recent_tokens) {
            yield AgentEvent::Status(
                "No older history to summarize; recent conversation preserved.".into(),
            );
            return;
        }

        match conversation
            .compact_with_llm_keeping_tokens(compaction_provider, compaction_model, keep_recent_tokens)
            .await
        {
            Ok(Some(result)) => {
                record_success(entries_before, conversation.entries().len());
                yield AgentEvent::Compacted(result.summary);
            }
            Ok(None) => {
                metrics::counter!("agent.compactions.failed").increment(1);
                yield AgentEvent::Error("Compaction produced no summary. Conversation unchanged.".into());
            }
            Err(error) => {
                metrics::counter!("agent.compactions.failed").increment(1);
                yield AgentEvent::Error(format!("Compaction failed using {compaction_model}: {error}. Conversation unchanged."));
            }
        }
    }
}

/// Metrics and log line for an LLM compaction that produced a summary.
fn record_success(entries_before: usize, entries_after: usize) {
    metrics::counter!("agent.compactions.succeeded").increment(1);
    metrics::gauge!("agent.compaction.entries_before").set(entries_before as f64);
    metrics::gauge!("agent.compaction.entries_after").set(entries_after as f64);
    tracing::debug!("Compaction complete: {entries_before} -> {entries_after} entries");
}
