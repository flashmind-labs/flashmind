//! Post-turn orchestrator — spawns background agents after a prompt completes.

use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::warn;

use flashmind_core::Conversation;
use flashmind_core::conversation::ConversationEntry;
use flashmind_memory::DbStore;
use flashmind_memory::embeddings::EmbeddingProvider;
use flashmind_types::model::Model;
use flashmind_types::{AgentLlmConfig, LlmProvider, Message};

use crate::config::AppConfig;
use crate::session::Sessions;

use super::PostTurnEvent;
use super::capture::{extract_exchange, spawn_capture_agent};
use super::enrichment::spawn_title_enrichment;

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Spawn all post-turn background agents and return a receiver for their events.
///
/// Currently spawns:
/// - **Title enrichment** — generates a kebab-case session title.
/// - **Memory capture** — extracts durable facts from the last exchange (if
///   memory is configured and the exchange is non-empty).
#[allow(clippy::too_many_arguments)]
pub fn spawn_post_turn(
    config: &AppConfig,
    session_key: String,
    conversation: &Conversation,
    messages: Vec<Message>,
    model: Model,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    sessions: Sessions,
    db: Option<DbStore>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    username: Option<String>,
) -> mpsc::Receiver<PostTurnEvent> {
    let exchange = extract_exchange(conversation);
    spawn_post_turn_with_entries(
        config,
        session_key,
        exchange,
        messages,
        model,
        provider,
        llm,
        sessions,
        db,
        embedder,
        username,
    )
}

/// Like [`spawn_post_turn`] but takes pre-extracted exchange entries and messages.
///
/// Useful when the conversation is behind a lock and you need to extract data
/// before releasing it (e.g. desktop app with `Mutex<Conversation>`).
#[allow(clippy::too_many_arguments)]
pub fn spawn_post_turn_with_entries(
    config: &AppConfig,
    session_key: String,
    exchange_entries: Vec<ConversationEntry>,
    messages: Vec<Message>,
    model: Model,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    sessions: Sessions,
    db: Option<DbStore>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    username: Option<String>,
) -> mpsc::Receiver<PostTurnEvent> {
    let (tx, rx) = mpsc::channel(32);

    // --- Title enrichment (always) -------------------------------------------
    spawn_title_enrichment(
        session_key,
        messages,
        model.clone(),
        provider.clone(),
        sessions,
        tx.clone(),
    );

    // --- Memory capture (when configured) ------------------------------------
    if let (Some(db), Some(embedder)) = (db, embedder)
        && let Some(capture_config) = config.memory.as_ref().and_then(|m| m.capture.as_ref())
        && !exchange_entries.is_empty()
    {
        let mut capture_llm = llm;
        let mut capture_provider = provider;

        if let Some(ref model_str) = capture_config.model {
            match model_str.parse::<Model>() {
                Ok(parsed_model) => {
                    if parsed_model.provider != capture_provider.provider() {
                        match config.build_provider_for(&parsed_model.provider) {
                            Ok(p) => capture_provider = p,
                            Err(e) => {
                                warn!(
                                    error = %e,
                                    model = %model_str,
                                    "failed to build provider for capture model, \
                                     falling back to main provider"
                                );
                            }
                        }
                    }
                    capture_llm.model = parsed_model;
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        model = %model_str,
                        "failed to parse capture model, using main model"
                    );
                }
            }
        }

        spawn_capture_agent(
            capture_config,
            exchange_entries,
            capture_provider,
            capture_llm,
            db,
            embedder,
            tx,
            username,
        );
    }

    rx
}
