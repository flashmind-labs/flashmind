//! Vector memory store with hybrid search, embeddings, and SQLite storage.
//!
//! Stores memories as embeddings in SQLite via `sqlite-vec`, with FTS5 full-text
//! search for hybrid retrieval using Reciprocal Rank Fusion (RRF).
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`DbStore`] | Low-level SQLite store (vector + FTS5 + tags + TTL) |
//! | [`VectorMemory`] | [`MemoryProvider`](flashmind_types::MemoryProvider) wrapping `DbStore` + embedder |
//! | [`EmbeddingProvider`] | Trait for embedding backends (OpenAI, Ollama, OpenRouter) |
//!
//! # Creating a memory store
//!
//! ```rust,ignore
//! let embedder = Arc::new(OllamaEmbedding::new(None));
//! let store = DbStore::connect(Path::new("memory.db"), embedder.dimensions()).await?;
//! let memory = VectorMemory::new(store, embedder);
//! ```

pub mod embeddings;
pub mod error;
pub(crate) mod http;
pub mod provider;
pub mod schema;
pub mod search;
pub mod store;

#[cfg(feature = "session")]
pub mod session;

pub use embeddings::{
    EmbeddingProvider, EmbeddingProviderConfig, OllamaEmbedding, OpenAIEmbedding,
    OpenRouterEmbedding, create_embedding_provider,
};
pub use error::{FlashmemError, Result};
pub use provider::VectorMemory;
pub use schema::{Scope, Source, Tag};
pub use store::{DbStore, MemoryRecord, MemorySearchResult};

#[cfg(feature = "session")]
pub use session::{SessionEntry, SessionEntryKind, SessionStore};

// Re-export so downstream crates don't need direct dependencies.
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
