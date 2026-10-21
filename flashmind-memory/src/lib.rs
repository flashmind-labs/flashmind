//! Vector memory store with hybrid search and embeddings.
//!
//! Storage comes from three features, at least one of which must be on:
//!
//! - `sqlite` (default): SQLite with sqlite-vec and FTS5, see
//!   `MemoryStore::connect`.
//! - `postgres`: Postgres with pgvector, see `MemoryStore::connect_postgres`.
//! - `session`: SQLite conversation history without vector storage.
//!
//! `http-embeddings` (default) provides the built-in embedding clients.
//!
//! # Usage
//!
//! ```rust,ignore
//! let embedder = Arc::new(OllamaEmbedding::new(None, "nomic-embed-text".into()));
//! let store = MemoryStore::connect(Path::new("memory.db"), embedder).await?;
//!
//! // Store
//! store.store("user prefers dark mode")
//!     .meta("tag", "preference")
//!     .await?;
//!
//! // Search
//! let results = store.search("preferences").limit(10).await?;
//! ```

#[cfg(not(any(feature = "sqlite", feature = "postgres", feature = "session")))]
compile_error!("flashmind-memory needs the `sqlite`, `postgres`, or `session` feature");

pub mod embeddings;
pub mod error;
#[cfg(feature = "http-embeddings")]
pub(crate) mod http;
#[cfg(feature = "postgres")]
mod pg;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub mod provider;
#[cfg(feature = "sqlite")]
pub mod schema;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub mod search;
#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub mod store;

#[cfg(feature = "session")]
pub mod session;

pub use embeddings::{EmbeddingProvider, EmbeddingProviderConfig};
#[cfg(feature = "http-embeddings")]
pub use embeddings::{
    OllamaEmbedding, OpenAIEmbedding, OpenRouterEmbedding, create_embedding_provider,
};
pub use error::{FlashmemError, Result};
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub use store::{
    ListBuilder, MemoryRecord, MemorySearchResult, MemoryStore, SearchBuilder, SimilarBuilder,
    StoreBuilder,
};

#[cfg(feature = "session")]
pub use session::{SessionEntry, SessionEntryKind, SessionStore, SessionSummary};

#[cfg(any(feature = "sqlite", feature = "session"))]
pub use rusqlite;
#[cfg(feature = "sqlite")]
pub use sqlite_vec;
#[cfg(any(feature = "sqlite", feature = "session"))]
pub use tokio_rusqlite;

/// Register the `sqlite-vec` extension as an auto-extension so every new
/// SQLite connection gets the `vec0` virtual table.
#[cfg(feature = "sqlite")]
#[allow(clippy::missing_transmute_annotations)]
pub fn register_sqlite_vec() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

#[cfg(all(test, feature = "sqlite"))]
pub mod test_util {
    pub fn register_sqlite_vec() {
        crate::register_sqlite_vec();
    }
}
