//! Automatic memory capture after prompt completion.

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc;
use tracing::info;

use flashmind_core::Agent;
use flashmind_core::conversation::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::DbStore;
use flashmind_memory::embeddings::EmbeddingProvider;
use flashmind_types::tool::ToolRegistry;
use flashmind_types::{AgentInput, AgentLlmConfig, LlmProvider};

use super::PostTurnEvent;
use super::prompt::build_capture_prompt;
use crate::config::CaptureConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate tool output entries to keep the capture agent's context small.
fn truncate_tool_entries(entries: &[ConversationEntry]) -> Vec<ConversationEntry> {
    entries
        .iter()
        .map(|entry| match &entry.kind {
            EntryKind::Tool { call_id, output } if output.len() > 200 => {
                let truncated = match output.char_indices().nth(200) {
                    Some((byte_pos, _)) => {
                        format!("{}...[truncated]", &output[..byte_pos])
                    }
                    None => output.clone(),
                };
                ConversationEntry {
                    kind: EntryKind::Tool {
                        call_id: call_id.clone(),
                        output: truncated,
                    },
                    timestamp: entry.timestamp,
                }
            }
            _ => entry.clone(),
        })
        .collect()
}

/// Build a tool registry containing only the memory tools needed for capture:
/// `memory_store`, `memory_recall`, and `memory_forget`.
fn build_capture_tools(db: DbStore, embedder: Arc<dyn EmbeddingProvider>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    crate::memory::register_tools(&mut registry, db, embedder);
    registry.remove("memory_edit");
    registry.remove("memory_list");
    registry
}

// ---------------------------------------------------------------------------
// Capture agent
// ---------------------------------------------------------------------------

/// Spawn a background capture agent that analyzes the last exchange and stores
/// durable facts as long-term memories.
#[allow(clippy::too_many_arguments)]
pub fn spawn_capture_agent(
    capture_config: &CaptureConfig,
    exchange_entries: Vec<ConversationEntry>,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    db: DbStore,
    embedder: Arc<dyn EmbeddingProvider>,
    tx: mpsc::Sender<PostTurnEvent>,
    username: Option<String>,
) {
    if !capture_config.enable {
        return;
    }

    tokio::spawn(async move {
        let tools = build_capture_tools(db, embedder);
        let mut agent = Agent::builder(provider).llm(llm).tools(tools).build();

        let mut conversation = Conversation::new();
        conversation.prepend(ConversationEntry::system(build_capture_prompt(
            username.as_deref(),
        )));
        conversation.add(ConversationEntry::system_message(
            "The following messages are from a PREVIOUS agent turn.",
        ));

        let truncated = truncate_tool_entries(&exchange_entries);
        for entry in truncated {
            conversation.add(entry);
        }

        let input = AgentInput::user(
            "Analyze the conversation above and extract any durable facts into memory.",
        );

        {
            let cancel = flashmind_core::CancellationToken::new();
            let stream = agent.start(&mut conversation, cancel, input, Some(10));
            futures::pin_mut!(stream);
            while let Some(_event) = stream.next().await {}
        }

        let (stored, forgotten) = summarize_and_emit(&conversation, &tx).await;

        let _ = tx
            .send(PostTurnEvent::CaptureComplete { stored, forgotten })
            .await;

        info!(stored, forgotten, "capture agent finished");
    });
}

/// Walk the capture agent's conversation and emit events for each memory
/// store/forget action. Returns `(stored, forgotten)` counts.
async fn summarize_and_emit(
    conversation: &Conversation,
    tx: &mpsc::Sender<PostTurnEvent>,
) -> (usize, usize) {
    let mut stored = 0usize;
    let mut forgotten = 0usize;

    for entry in conversation.entries() {
        if let EntryKind::Assistant {
            tool_calls: Some(ref tcs),
            ..
        } = entry.kind
        {
            for tc in tcs {
                match tc.name.as_str() {
                    "memory_store" => {
                        if let Some(content) = tc.arguments.get("content").and_then(|v| v.as_str())
                        {
                            let _ = tx
                                .send(PostTurnEvent::MemoryStored {
                                    content: content.to_string(),
                                })
                                .await;
                            stored += 1;
                        }
                    }
                    "memory_forget" => {
                        if let Some(id) = tc.arguments.get("id").and_then(|v| v.as_str()) {
                            let _ = tx
                                .send(PostTurnEvent::MemoryForgotten { id: id.to_string() })
                                .await;
                            forgotten += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    (stored, forgotten)
}
