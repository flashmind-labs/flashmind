//! Memory abstraction — [`MemoryProvider`] trait and entry types.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Arbitrary key-value metadata attached to a memory entry.
///
/// Use `tags` for categorization (e.g. `"fact"`, `"preference"`, `"project:foo"`).
/// Set `expires_at` to enable automatic TTL-based eviction.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMetadata {
    /// Free-form human-readable context (e.g. summarises the surrounding conversation).
    pub context: Option<String>,
    /// Flat list of string tags used for filtering and discovery.
    pub tags: Vec<String>,
    /// When set, the entry is considered expired after this instant.
    pub expires_at: Option<DateTime<Utc>>,
}

/// A single stored memory with its relevance score and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier assigned at store time.
    pub id: String,
    /// The stored text content.
    pub content: String,
    /// Relevance score from the last search (1.0 = perfect match).
    pub score: f64,
    /// Attached metadata (tags, expiry, context).
    pub metadata: MemoryMetadata,
    /// UTC timestamp when this entry was first stored.
    pub created_at: DateTime<Utc>,
}

/// Abstraction over long-term memory storage and retrieval.
///
/// Implementations back the `memory_store` / `memory_recall` tools.
/// Returned IDs are opaque strings; callers must store them if they need to call `forget`.
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Persist `content` with `metadata`. Returns the new entry's ID.
    async fn store(&self, content: &str, metadata: MemoryMetadata) -> anyhow::Result<String>;

    /// Semantic search. Returns up to `limit` entries ordered by descending relevance.
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>>;

    /// Remove a specific entry by ID. No-op if the ID does not exist.
    async fn forget(&self, id: &str) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_metadata_default() {
        let meta = MemoryMetadata::default();
        assert!(meta.context.is_none());
        assert!(meta.tags.is_empty());
        assert!(meta.expires_at.is_none());
    }

    #[test]
    fn memory_entry_serde_round_trip() {
        let entry = MemoryEntry {
            id: "mem-123".into(),
            content: "the user prefers dark mode".into(),
            score: 0.95,
            metadata: MemoryMetadata {
                context: Some("preferences".into()),
                tags: vec!["preference".into(), "ui".into()],
                expires_at: None,
            },
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "mem-123");
        assert_eq!(deserialized.content, "the user prefers dark mode");
        assert!((deserialized.score - 0.95).abs() < f64::EPSILON);
        assert_eq!(deserialized.metadata.tags.len(), 2);
    }

    #[test]
    fn memory_metadata_with_expiry() {
        let expiry = Utc::now() + chrono::Duration::hours(24);
        let meta = MemoryMetadata {
            context: None,
            tags: vec!["ephemeral".into()],
            expires_at: Some(expiry),
        };
        assert!(meta.expires_at.unwrap() > Utc::now());
    }
}
