//! [`MemoryProvider`](flashmind_types::memory::MemoryProvider) implementation backed by [`DbStore`] + [`EmbeddingProvider`].
//!
//! The [`MemoryProvider`] trait exposes a simplified three-operation interface
//! (`store`, `search`, `forget`) with no embeddings in the signature. This wrapper
//! holds both the store and an embedding provider, generating embeddings on the fly
//! for store and search operations. Hybrid search combines cosine-similarity vector
//! matching with BM25 keyword scoring via Reciprocal Rank Fusion (RRF).
//!
//! # Thread safety
//!
//! [`VectorMemory`] is not `Clone` itself (it wraps a SQLite connection), but can be
//! shared across async tasks by placing it behind an `Arc`.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::DateTime;
use flashmind_types::memory::{MemoryEntry, MemoryMetadata, MemoryProvider};

use crate::embeddings::EmbeddingProvider;
use crate::schema::{Scope, Source, Tag};
use crate::store::{DbStore, MemorySearchResult};

/// Wrapper around [`DbStore`] + [`EmbeddingProvider`] that implements [`MemoryProvider`].
///
/// Handles embedding generation internally so callers don't need to manage embeddings.
/// Supports hybrid search (vector + full-text), tagging, TTL expiry, and deduplication.
///
/// # Example
///
/// ```rust,ignore
/// let embedder = Arc::new(OllamaEmbedding::new(None));
/// let store = DbStore::open("memory.db", embedder.dimensions()).await?;
/// let memory = VectorMemory::new(store, embedder);
///
/// // Store, search, forget
/// let id = memory.store("user prefers dark mode", metadata).await?;
/// let results = memory.search("preferences", 10).await?;
/// memory.forget(&id).await?;
/// ```
pub struct VectorMemory {
    store: DbStore,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl VectorMemory {
    pub fn new(store: DbStore, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { store, embedder }
    }

    pub fn store(&self) -> &DbStore {
        &self.store
    }

    pub fn embedder(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedder
    }
}

#[async_trait]
impl MemoryProvider for VectorMemory {
    async fn store(&self, content: &str, metadata: MemoryMetadata) -> anyhow::Result<String> {
        let embedding = self
            .embedder
            .embed(content)
            .await
            .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;

        let tags: Vec<Tag> = metadata
            .tags
            .iter()
            .filter_map(|t| t.parse().ok())
            .collect();

        let expires_at = metadata.expires_at.map(|dt| dt.timestamp());

        let id = self
            .store
            .store(
                content,
                embedding,
                Source::Manual,
                None, // chat_key -- not exposed in trait
                None, // identity
                &tags,
                expires_at,
                None, // tool_name
            )
            .await?;

        Ok(id)
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>> {
        let embedding = self
            .embedder
            .embed(query)
            .await
            .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;

        let results = self
            .store
            .search_hybrid(embedding, query, limit, None, Some(Scope::Global))
            .await?;

        Ok(results.into_iter().map(memory_search_to_entry).collect())
    }

    async fn forget(&self, id: &str) -> anyhow::Result<()> {
        self.store.delete(id).await?;
        Ok(())
    }
}

fn memory_search_to_entry(r: MemorySearchResult) -> MemoryEntry {
    MemoryEntry {
        id: r.id.unwrap_or_default(),
        content: r.content,
        score: r.score as f64,
        metadata: MemoryMetadata {
            context: r.chat_key,
            tags: r.tags.iter().map(|t| t.to_string()).collect(),
            expires_at: r.expires_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
        },
        created_at: DateTime::from_timestamp(r.created_at, 0).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantEmbedder;

    #[async_trait]
    impl EmbeddingProvider for ConstantEmbedder {
        async fn embed(&self, _text: &str) -> crate::error::Result<Vec<f32>> {
            Ok(vec![1.0; 64])
        }

        fn dimensions(&self) -> usize {
            64
        }

        fn name(&self) -> &str {
            "constant"
        }
    }

    async fn test_memory() -> VectorMemory {
        crate::test_util::register_sqlite_vec();
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let store = DbStore::connect(&db_path, 64).await.unwrap();
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(ConstantEmbedder);
        // Leak the tempdir so it lives for the test duration
        std::mem::forget(dir);
        VectorMemory::new(store, embedder)
    }

    #[tokio::test]
    async fn store_and_search() {
        let mem = test_memory().await;
        let meta = MemoryMetadata {
            context: Some("test-chat".into()),
            tags: vec!["fact".into()],
            expires_at: None,
        };

        let id = MemoryProvider::store(&mem, "the sky is blue", meta)
            .await
            .unwrap();
        assert!(!id.is_empty());

        let results = MemoryProvider::search(&mem, "sky color", 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("sky is blue"));
    }

    #[tokio::test]
    async fn store_and_forget() {
        let mem = test_memory().await;
        let meta = MemoryMetadata::default();

        let id = MemoryProvider::store(&mem, "temporary fact", meta)
            .await
            .unwrap();
        let result = MemoryProvider::forget(&mem, &id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn search_empty_store() {
        let mem = test_memory().await;
        let results = MemoryProvider::search(&mem, "anything", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn memory_search_to_entry_conversion() {
        let result = MemorySearchResult {
            id: Some("id-1".into()),
            content: "test content".into(),
            source: Source::Manual,
            chat_key: Some("chat:123".into()),
            created_at: 1700000000,
            score: 0.85,
            tags: vec![Tag::Fact],
            expires_at: None,
        };
        let entry = memory_search_to_entry(result);
        assert_eq!(entry.id, "id-1");
        assert_eq!(entry.content, "test content");
        assert!((entry.score - 0.85).abs() < 0.01);
        assert_eq!(entry.metadata.context.as_deref(), Some("chat:123"));
        assert_eq!(entry.metadata.tags, vec!["fact"]);
    }
}
