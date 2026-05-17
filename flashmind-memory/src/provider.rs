//! [`MemoryProvider`] implementation backed by [`MemoryStore`].
//!
//! Thin adapter between the [`MemoryProvider`] trait (from `flashmind-types`)
//! and the [`MemoryStore`] builder API.

use async_trait::async_trait;
use chrono::DateTime;
use flashmind_types::memory::{MemoryEntry, MemoryMetadata, MemoryProvider};

use crate::store::{MemorySearchResult, MemoryStore};

#[async_trait]
impl MemoryProvider for MemoryStore {
    async fn store(&self, content: &str, metadata: MemoryMetadata) -> anyhow::Result<String> {
        let mut builder = MemoryStore::store(self, content);

        for tag in &metadata.tags {
            builder = builder.meta("tag", tag);
        }

        if let Some(ref ctx) = metadata.context {
            builder = builder.meta("context", ctx);
        }

        if let Some(dt) = metadata.expires_at {
            builder = builder.expires_at(dt.timestamp());
        }

        let id = builder.await?;
        Ok(id)
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>> {
        let results = MemoryStore::search(self, query).limit(limit).await?;
        Ok(results.into_iter().map(search_result_to_entry).collect())
    }

    async fn forget(&self, id: &str) -> anyhow::Result<()> {
        self.delete(id).await?;
        Ok(())
    }
}

fn search_result_to_entry(r: MemorySearchResult) -> MemoryEntry {
    let tags: Vec<String> = r
        .meta
        .iter()
        .filter(|(k, _)| k == "tag")
        .map(|(_, v)| v.clone())
        .collect();

    let context = r
        .meta
        .iter()
        .find(|(k, _)| k == "context")
        .map(|(_, v)| v.clone());

    let expires_at = r.expires_at.and_then(|ts| DateTime::from_timestamp(ts, 0));

    MemoryEntry {
        id: r.id.unwrap_or_default(),
        content: r.content,
        score: r.score as f64,
        metadata: MemoryMetadata {
            context,
            tags,
            expires_at,
        },
        created_at: DateTime::from_timestamp(r.created_at, 0).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingProvider;
    use crate::error::Result;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct ConstantEmbedder;

    #[async_trait]
    impl EmbeddingProvider for ConstantEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0; 64])
        }
        fn dimensions(&self) -> usize {
            64
        }
        fn name(&self) -> &str {
            "constant"
        }
    }

    async fn test_memory() -> MemoryStore {
        crate::test_util::register_sqlite_vec();
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(ConstantEmbedder);
        let store = MemoryStore::connect(&db_path, embedder).await.unwrap();
        std::mem::forget(dir);
        store
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
}
