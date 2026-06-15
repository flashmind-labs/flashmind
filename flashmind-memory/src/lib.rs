//! Vector memory store with hybrid search, embeddings, and SQLite storage.
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
pub use store::{MemoryRecord, MemorySearchResult, MemoryStore};

#[cfg(feature = "session")]
pub use session::{SessionEntry, SessionEntryKind, SessionStore, SessionSummary};

pub use rusqlite;
pub use sqlite_vec;
pub use tokio_rusqlite;

/// Register the `sqlite-vec` extension as an auto-extension so every new
/// SQLite connection gets the `vec0` virtual table.
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

#[cfg(test)]
pub mod test_util {
    pub fn register_sqlite_vec() {
        crate::register_sqlite_vec();
    }
}
