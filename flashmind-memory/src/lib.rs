//! Vector memory store with hybrid search, embeddings, and SQLite storage.
//!
//! This crate provides the long-term memory subsystem for the Flashmind AI agent
//! framework. It stores memories as embeddings in SQLite via `sqlite-vec`, with
//! FTS5 full-text search for hybrid retrieval.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`DbStore`] | Low-level SQLite store (vector + FTS5 + tags + TTL) |
//! | [`VectorMemory`] | [`MemoryProvider`](flashmind_types::MemoryProvider) implementation wrapping `DbStore` + embedder |
//! | [`EmbeddingProvider`] | Trait for embedding backends (OpenAI, Ollama, OpenRouter) |
//! | [`MemoryComponents`] | Startup bundle: `DbStore` + `Arc<dyn EmbeddingProvider>` |
//!
//! # Architecture
//!
//! ```text
//! store/search/forget
//!       │
//!       ▼
//!  VectorMemory ──► EmbeddingProvider (embed text → Vec<f32>)
//!       │
//!       ▼
//!    DbStore ──► SQLite (sqlite-vec cosine + FTS5 BM25)
//! ```
//!
//! Isolated from the main runtime so non-memory changes get fast incremental builds.

pub mod embeddings;
pub mod error;
pub(crate) mod http;
pub mod local_sessions;
pub mod provider;
pub mod schema;
pub mod search;
pub mod session;
pub mod sharing;
pub mod store;
pub mod user_sessions;
pub mod users;

pub use embeddings::{
    EmbeddingProvider, EmbeddingProviderConfig, OllamaEmbedding, OpenAIEmbedding,
    OpenRouterEmbedding, create_embedding_provider,
};
pub use error::{FlashmemError, Result};
pub use local_sessions::{LocalSession, SessionMode};
pub use provider::VectorMemory;
pub use schema::{Scope, Source, Tag};
pub use session::SessionEntry;
pub use sharing::{ShareData, ShareInfo};
pub use store::{DbStore, MemoryRecord, MemorySearchResult};
pub use user_sessions::UserSessionMeta;
pub use users::{ApiKeyInfo, User};

/// Shared memory components for embedding-based memory tools.
/// Created once at startup, cloned to each agent.
#[derive(Clone)]
pub struct MemoryComponents {
    pub vector_memory: DbStore,
    pub embedder: std::sync::Arc<dyn EmbeddingProvider>,
}

// Re-export so the main crate doesn't need direct dependencies.
pub use rusqlite;
pub use sqlite_vec;
pub use tokio_rusqlite;

#[cfg(test)]
pub mod test_util {
    use std::sync::Once;

    #[allow(clippy::missing_transmute_annotations)]
    pub fn register_sqlite_vec() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        });
    }
}
