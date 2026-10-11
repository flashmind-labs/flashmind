//! Vector memory store using SQLite + sqlite-vec + FTS5 with the `sqlite`
//! feature, or Postgres + pgvector with the `postgres` feature.
//!
//! # Usage
//!
//! ```rust,ignore
//! let store = MemoryStore::connect(path, embedder).await?;
//!
//! // Simple store
//! store.store("user prefers dark mode").await?;
//!
//! // Store with metadata
//! store.store("user prefers dark mode")
//!     .meta("source", "manual")
//!     .meta("tag", "preference")
//!     .expires_at(some_ts)
//!     .await?;
//!
//! // Simple search
//! let results = store.search("dark mode").await?;
//!
//! // Filtered search
//! let results = store.search("dark mode")
//!     .filter("source", "manual")
//!     .limit(5)
//!     .await?;
//! ```

use std::cmp::Ordering;
use std::collections::HashMap;
use std::future::IntoFuture;
#[cfg(feature = "sqlite")]
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::future::BoxFuture;
use tokio::time::timeout;
use uuid::Uuid;

#[cfg(feature = "postgres")]
use sqlx::PgPool;

use crate::embeddings::EmbeddingProvider;
use crate::error::{FlashmemError, Result};
#[cfg(feature = "postgres")]
use crate::pg;
#[cfg(feature = "sqlite")]
use crate::sqlite;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A search result with relevance score.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub id: Option<String>,
    pub content: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub score: f32,
    pub meta: Vec<(String, String)>,
}

/// A full memory record (for listing/curation).
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub meta: Vec<(String, String)>,
}

/// A memory for a backend to insert.
pub(crate) struct NewMemory {
    pub id: String,
    pub content: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub embedding: Vec<f32>,
    pub meta: Vec<(String, String)>,
}

/// Changes for a backend's `update`. `None` leaves a field as it is.
pub(crate) struct MemoryUpdate {
    pub content: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub meta: Option<Vec<(String, String)>>,
    pub expires_at: Option<Option<i64>>,
}

/// Filters for a backend's `list`.
pub(crate) struct ListQuery {
    pub filters: Vec<(String, String)>,
    pub contains: Option<String>,
    pub cursor: Option<i64>,
    pub after: Option<i64>,
    pub before: Option<i64>,
    pub limit: usize,
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Where a [`MemoryStore`] keeps its rows.
#[derive(Clone)]
enum Backend {
    #[cfg(feature = "sqlite")]
    Sqlite(tokio_rusqlite::Connection),
    #[cfg(feature = "postgres")]
    Postgres(PgPool),
}

/// Embedding width for the schema. Ollama models are probed since their
/// width depends on the model.
async fn embedding_dim(embedder: &dyn EmbeddingProvider) -> Result<usize> {
    Ok(if embedder.name() == "ollama" {
        embedder.embed("dimension probe").await?.len()
    } else {
        embedder.dimensions()
    })
}

/// Vector memory store backed by SQLite with sqlite-vec and FTS5, or by
/// Postgres with pgvector.
///
/// Handles embedding generation internally. Use [`store`](MemoryStore::store)
/// and [`search`](MemoryStore::search) builders for the primary API.
#[derive(Clone)]
pub struct MemoryStore {
    backend: Backend,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl MemoryStore {
    /// Connect to a SQLite memory store.
    ///
    /// Creates parent directories and initializes the schema on first run.
    #[cfg(feature = "sqlite")]
    pub async fn connect(db_path: &Path, embedder: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let embedding_dim = embedding_dim(embedder.as_ref()).await?;
        tracing::info!(path = %db_path.display(), "connecting to memory store");
        let conn = sqlite::open(db_path, embedding_dim).await?;
        tracing::info!("memory store ready");

        Ok(Self {
            backend: Backend::Sqlite(conn),
            embedder,
        })
    }

    /// Use a Postgres pool for the memory store.
    ///
    /// Creates the `vector` extension and the `memories` and `memory_meta`
    /// tables if missing, and touches no other tables, so the pool can be
    /// shared. Fails if `memories` holds embeddings of another dimension.
    #[cfg(feature = "postgres")]
    pub async fn connect_postgres(
        pool: PgPool,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        let embedding_dim = embedding_dim(embedder.as_ref()).await?;
        tracing::info!("connecting to Postgres memory store");
        pg::init_schema(&pool, embedding_dim).await?;
        tracing::info!("memory store ready");

        Ok(Self {
            backend: Backend::Postgres(pool),
            embedder,
        })
    }

    /// Get a reference to the underlying SQLite connection, `None` on Postgres.
    #[cfg(feature = "sqlite")]
    pub fn connection(&self) -> Option<&tokio_rusqlite::Connection> {
        match &self.backend {
            Backend::Sqlite(conn) => Some(conn),
            #[cfg(feature = "postgres")]
            Backend::Postgres(_) => None,
        }
    }

    /// Get a reference to the embedding provider.
    pub fn embedder(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedder
    }

    /// Create a [`SessionStore`](crate::session::SessionStore) sharing this
    /// connection, `None` on Postgres since sessions are SQLite only.
    #[cfg(feature = "session")]
    pub fn session_store(&self) -> Option<crate::session::SessionStore> {
        self.connection()
            .map(|conn| crate::session::SessionStore::new(conn.clone()))
    }

    // -- Builders -------------------------------------------------------------

    /// Store a new memory. Returns a builder — call `.await` to execute.
    ///
    /// ```rust,ignore
    /// store.store("user prefers dark mode")
    ///     .meta("tag", "preference")
    ///     .await?;
    /// ```
    pub fn store<'a>(&'a self, content: &'a str) -> StoreBuilder<'a> {
        StoreBuilder {
            store: self,
            content,
            meta: Vec::new(),
            expires_at: None,
            embedding: None,
        }
    }

    /// Search memories by semantic similarity. Returns a builder — call `.await` to execute.
    ///
    /// ```rust,ignore
    /// let results = store.search("dark mode")
    ///     .filter("source", "manual")
    ///     .limit(10)
    ///     .await?;
    /// ```
    pub fn search<'a>(&'a self, query: &'a str) -> SearchBuilder<'a> {
        SearchBuilder {
            store: self,
            query,
            filters: Vec::new(),
            limit: 20,
        }
    }

    /// Find memories close in meaning to `text`, by vector similarity alone.
    /// Returns a builder — call `.await` to execute.
    ///
    /// ```rust,ignore
    /// let duplicates = store.similar("user prefers dark mode")
    ///     .filter("user", "alice")
    ///     .threshold(0.8)
    ///     .limit(1)
    ///     .await?;
    /// ```
    pub fn similar<'a>(&'a self, text: &'a str) -> SimilarBuilder<'a> {
        SimilarBuilder {
            store: self,
            text,
            filters: Vec::new(),
            threshold: 0.0,
            limit: 20,
        }
    }

    /// List memories newest first. Returns a builder — call `.await` to execute.
    ///
    /// ```rust,ignore
    /// let records = store.records()
    ///     .filter("user", "alice")
    ///     .limit(100)
    ///     .await?;
    /// ```
    pub fn records(&self) -> ListBuilder<'_> {
        ListBuilder {
            store: self,
            filters: Vec::new(),
            contains: None,
            cursor: None,
            after: None,
            before: None,
            limit: 100,
        }
    }

    // -- Direct operations ----------------------------------------------------

    /// Delete a memory by ID (or prefix ≥8 chars).
    pub async fn delete(&self, id: &str) -> Result<()> {
        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::delete(conn, id).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::delete(pool, id).await,
        }?;
        metrics::counter!("memory.deletes").increment(1);
        Ok(())
    }

    /// Delete all expired memories. Returns count of deleted rows.
    pub async fn delete_expired(&self) -> Result<usize> {
        let now = Utc::now().timestamp();
        let count = match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::delete_expired(conn, now).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::delete_expired(pool, now).await,
        }?;
        tracing::debug!(count, "deleted expired memories");
        Ok(count)
    }

    /// Delete all memories with the given metadata scope value.
    pub async fn delete_by_scope(&self, scope: &str) -> Result<usize> {
        self.delete_matching(&[("scope", scope)]).await
    }

    /// Delete every memory carrying all of the given metadata pairs. Returns
    /// how many were deleted. Refuses an empty filter, which would match all.
    pub async fn delete_matching(&self, filters: &[(&str, &str)]) -> Result<usize> {
        if filters.is_empty() {
            return Err(FlashmemError::Memory(
                "delete_matching needs at least one filter".into(),
            ));
        }
        let filters: Vec<(String, String)> = filters
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let count = match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::delete_matching(conn, filters).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::delete_matching(pool, &filters).await,
        }?;
        metrics::counter!("memory.deletes").increment(count as u64);
        Ok(count)
    }

    /// The memory with exactly this ID, if there is one. Unlike the
    /// prefix lookups, a short ID never matches.
    pub async fn get(&self, id: &str) -> Result<Option<MemoryRecord>> {
        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::get(conn, id).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::get(pool, id).await,
        }
    }

    /// Update content of an existing memory by ID (or prefix ≥8 chars).
    pub async fn update(
        &self,
        id: &str,
        new_content: Option<&str>,
        new_meta: Option<&[(&str, &str)]>,
        expires_at: Option<Option<i64>>,
    ) -> Result<String> {
        let embedding = if let Some(content) = new_content {
            Some(self.embedder.embed(content).await?)
        } else {
            None
        };

        let changes = MemoryUpdate {
            content: new_content.map(str::to_string),
            embedding,
            meta: new_meta.map(|m| {
                m.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect()
            }),
            expires_at,
        };

        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::update(conn, id, changes).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::update(pool, id, changes).await,
        }
    }

    /// Find memories similar to the given text above a score threshold.
    pub async fn find_similar(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let results = self.similar(text).threshold(threshold).limit(limit).await?;
        Ok(results
            .into_iter()
            .map(|r| (r.id.unwrap_or_default(), r.score))
            .collect())
    }

    /// List all memories with optional pagination and content filter.
    pub async fn list(
        &self,
        limit: usize,
        cursor: Option<i64>,
        filter: Option<&str>,
    ) -> Result<Vec<MemoryRecord>> {
        self.list_dated(limit, cursor, filter, None, None).await
    }

    /// List memories with date range filtering.
    pub async fn list_dated(
        &self,
        limit: usize,
        cursor: Option<i64>,
        filter: Option<&str>,
        after: Option<i64>,
        before: Option<i64>,
    ) -> Result<Vec<MemoryRecord>> {
        ListBuilder {
            store: self,
            filters: Vec::new(),
            contains: filter.map(str::to_string),
            cursor,
            after,
            before,
            limit,
        }
        .await
    }

    /// Repair corrupted virtual tables. Returns count of memories needing re-embedding.
    /// Postgres has no virtual tables to rebuild, so there it returns 0.
    pub async fn repair(&self) -> Result<usize> {
        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::repair(conn, self.embedder.dimensions()).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(_) => Ok(0),
        }
    }

    /// Re-insert an embedding for an existing memory ID (used after repair).
    pub async fn reindex(&self, id: &str, embedding: Vec<f32>) -> Result<()> {
        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::reindex(conn, id, embedding).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::reindex(pool, id, embedding).await,
        }
    }

    /// List all memory IDs and content (for bulk re-embedding after repair).
    pub async fn list_content_for_reindex(&self) -> Result<Vec<(String, String)>> {
        match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => sqlite::list_content_for_reindex(conn).await,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => pg::list_content_for_reindex(pool).await,
        }
    }

    // -- Internal search methods ----------------------------------------------

    async fn search_vec(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        filters: Vec<(String, String)>,
    ) -> Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let now = Utc::now().timestamp();

        let results = match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => {
                sqlite::search_vec(conn, query_embedding, limit, now, filters).await
            }
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                pg::search_vec(pool, query_embedding, limit, now, &filters).await
            }
        };
        results.inspect(|_| {
            metrics::counter!("memory.searches").increment(1);
            metrics::histogram!("memory.search.duration_seconds")
                .record(start.elapsed().as_secs_f64());
        })
    }

    async fn search_hybrid_internal(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        limit: usize,
        filters: &[(String, String)],
    ) -> Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let now = Utc::now().timestamp();

        let results = match &self.backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(conn) => {
                sqlite::search_hybrid(conn, query_embedding, query_text, limit, now, filters).await
            }
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                pg::search_hybrid(pool, query_embedding, query_text, limit, now, filters).await
            }
        };
        results.inspect(|results| {
            metrics::counter!("memory.hybrid_searches").increment(1);
            metrics::histogram!("memory.search.duration_seconds")
                .record(start.elapsed().as_secs_f64());
            metrics::histogram!("memory.search.results_count").record(results.len() as f64);
        })
    }

    /// Multi-query search: run multiple embeddings, deduplicate by ID.
    pub async fn search_multi_query(
        &self,
        queries: Vec<(Vec<f32>, String)>,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        let futs: Vec<_> = queries
            .into_iter()
            .map(|(emb, text)| {
                let this = self.clone();
                async move {
                    let result = timeout(
                        Duration::from_secs(3),
                        this.search_hybrid_internal(emb, &text, limit, &[]),
                    )
                    .await;
                    match result {
                        Ok(Ok(r)) => Some(r),
                        Ok(Err(e)) => {
                            tracing::warn!("search_hybrid failed: {e}");
                            None
                        }
                        Err(_) => {
                            tracing::warn!("search_hybrid timed out");
                            None
                        }
                    }
                }
            })
            .collect();

        let all_results = futures::future::join_all(futs).await;

        let mut id_to_result: HashMap<String, MemorySearchResult> = HashMap::new();
        for result in all_results.into_iter().flatten() {
            for r in result {
                let id = r.id.clone().unwrap_or_default();
                if let Some(existing) = id_to_result.get(&id) {
                    if r.score > existing.score {
                        id_to_result.insert(id, r);
                    }
                } else {
                    id_to_result.insert(id, r);
                }
            }
        }

        let mut merged: Vec<_> = id_to_result.into_values().collect();
        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        merged.truncate(limit);

        Ok(merged)
    }
}

// ---------------------------------------------------------------------------
// StoreBuilder
// ---------------------------------------------------------------------------

/// Builder for storing a memory. Finish with `.await`.
pub struct StoreBuilder<'a> {
    store: &'a MemoryStore,
    content: &'a str,
    meta: Vec<(String, String)>,
    expires_at: Option<i64>,
    embedding: Option<Vec<f32>>,
}

impl<'a> StoreBuilder<'a> {
    /// Use an embedding already computed for this content instead of asking
    /// the provider, as when importing memories embedded elsewhere.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attach a key-value metadata pair.
    pub fn meta(mut self, key: &str, value: &str) -> Self {
        self.meta.push((key.to_string(), value.to_string()));
        self
    }

    /// Attach a metadata pair only if value is `Some`.
    pub fn meta_opt(self, key: &str, value: Option<&str>) -> Self {
        match value {
            Some(v) => self.meta(key, v),
            None => self,
        }
    }

    /// Set an expiration timestamp (Unix epoch seconds).
    pub fn expires_at(mut self, ts: i64) -> Self {
        self.expires_at = Some(ts);
        self
    }

    /// Set expiration only if `Some`.
    pub fn expires_at_opt(self, ts: Option<i64>) -> Self {
        match ts {
            Some(t) => self.expires_at(t),
            None => self,
        }
    }
}

impl<'a> IntoFuture for StoreBuilder<'a> {
    type Output = Result<String>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            if self.content.trim().is_empty() {
                return Err(FlashmemError::Memory("cannot store empty content".into()));
            }

            let start = Instant::now();

            let embedding = match self.embedding {
                Some(embedding) => embedding,
                None => self.store.embedder.embed(self.content).await?,
            };

            let memory = NewMemory {
                id: Uuid::new_v4().to_string(),
                content: self.content.to_string(),
                created_at: Utc::now().timestamp(),
                expires_at: self.expires_at,
                embedding,
                meta: self.meta,
            };
            let id = match &self.store.backend {
                #[cfg(feature = "sqlite")]
                Backend::Sqlite(conn) => sqlite::insert(conn, memory).await,
                #[cfg(feature = "postgres")]
                Backend::Postgres(pool) => pg::insert(pool, memory).await,
            }?;
            metrics::counter!("memory.stores").increment(1);
            metrics::histogram!("memory.store.duration_seconds")
                .record(start.elapsed().as_secs_f64());
            Ok(id)
        })
    }
}

// ---------------------------------------------------------------------------
// SearchBuilder
// ---------------------------------------------------------------------------

/// Builder for searching memories. Finish with `.await`.
pub struct SearchBuilder<'a> {
    store: &'a MemoryStore,
    query: &'a str,
    filters: Vec<(String, String)>,
    limit: usize,
}

impl<'a> SearchBuilder<'a> {
    /// Filter results to only those with this metadata key-value pair.
    pub fn filter(mut self, key: &str, value: &str) -> Self {
        self.filters.push((key.to_string(), value.to_string()));
        self
    }

    /// Maximum number of results to return (default: 20).
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl<'a> IntoFuture for SearchBuilder<'a> {
    type Output = Result<Vec<MemorySearchResult>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            if self.query.trim().is_empty() {
                return Ok(Vec::new());
            }
            let embedding = self.store.embedder.embed(self.query).await?;
            self.store
                .search_hybrid_internal(embedding, self.query, self.limit, &self.filters)
                .await
        })
    }
}

// ---------------------------------------------------------------------------
// SimilarBuilder
// ---------------------------------------------------------------------------

/// Builder for a vector similarity lookup. Finish with `.await`.
pub struct SimilarBuilder<'a> {
    store: &'a MemoryStore,
    text: &'a str,
    filters: Vec<(String, String)>,
    threshold: f32,
    limit: usize,
}

impl<'a> SimilarBuilder<'a> {
    /// Only consider memories with this metadata key-value pair.
    pub fn filter(mut self, key: &str, value: &str) -> Self {
        self.filters.push((key.to_string(), value.to_string()));
        self
    }

    /// Minimum score a memory needs (default: 0, any).
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Maximum number of results to return (default: 20).
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl<'a> IntoFuture for SimilarBuilder<'a> {
    type Output = Result<Vec<MemorySearchResult>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let embedding = self.store.embedder.embed(self.text).await?;
            let mut results = self
                .store
                .search_vec(embedding, self.limit, self.filters)
                .await?;
            results.retain(|r| r.score >= self.threshold);
            Ok(results)
        })
    }
}

// ---------------------------------------------------------------------------
// ListBuilder
// ---------------------------------------------------------------------------

/// Builder for listing memories newest first. Finish with `.await`.
pub struct ListBuilder<'a> {
    store: &'a MemoryStore,
    filters: Vec<(String, String)>,
    contains: Option<String>,
    cursor: Option<i64>,
    after: Option<i64>,
    before: Option<i64>,
    limit: usize,
}

impl<'a> ListBuilder<'a> {
    /// Only list memories with this metadata key-value pair.
    pub fn filter(mut self, key: &str, value: &str) -> Self {
        self.filters.push((key.to_string(), value.to_string()));
        self
    }

    /// Only list memories whose content contains `text`.
    pub fn contains(mut self, text: &str) -> Self {
        self.contains = Some(text.to_string());
        self
    }

    /// Only list memories created at or after `ts`.
    pub fn after(mut self, ts: i64) -> Self {
        self.after = Some(ts);
        self
    }

    /// Only list memories created before `ts`; pass the last `created_at`
    /// seen to page.
    pub fn before(mut self, ts: i64) -> Self {
        self.before = Some(ts);
        self
    }

    /// Maximum number of records to return (default: 100).
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl<'a> IntoFuture for ListBuilder<'a> {
    type Output = Result<Vec<MemoryRecord>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let query = ListQuery {
            filters: self.filters,
            contains: self.contains,
            cursor: self.cursor,
            after: self.after,
            before: self.before,
            limit: self.limit,
        };
        let store = self.store;
        Box::pin(async move {
            match &store.backend {
                #[cfg(feature = "sqlite")]
                Backend::Sqlite(conn) => sqlite::list(conn, query).await,
                #[cfg(feature = "postgres")]
                Backend::Postgres(pool) => pg::list(pool, query).await,
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingProvider;
    use async_trait::async_trait;
    #[cfg(feature = "sqlite")]
    use tempfile::tempdir;

    struct ConstantEmbedder;

    #[async_trait]
    impl EmbeddingProvider for ConstantEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0, 0.0])
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn name(&self) -> &str {
            "constant"
        }
    }

    #[cfg(feature = "sqlite")]
    async fn sqlite_store(embedder: Arc<dyn EmbeddingProvider>) -> MemoryStore {
        crate::test_util::register_sqlite_vec();
        let dir = tempdir().unwrap();
        let store = MemoryStore::connect(&dir.path().join("test.db"), embedder)
            .await
            .unwrap();
        std::mem::forget(dir);
        store
    }

    /// Run `body` against a store in a fresh Postgres schema, dropped
    /// afterwards. Skips when `FLASHMIND_TEST_DATABASE_URL` is unset.
    #[cfg(feature = "postgres")]
    async fn with_pg_store<F, Fut>(embedder: Arc<dyn EmbeddingProvider>, body: F)
    where
        F: FnOnce(MemoryStore) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        use futures::FutureExt;
        use sqlx::postgres::PgPoolOptions;
        use std::panic::{AssertUnwindSafe, resume_unwind};

        let Ok(url) = std::env::var("FLASHMIND_TEST_DATABASE_URL") else {
            eprintln!("skipping Postgres test: FLASHMIND_TEST_DATABASE_URL is not set");
            return;
        };
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("test database is reachable");
        // Parallel tests race to create the extension; the loser's error is harmless.
        let _ = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&admin)
            .await;
        let schema = format!("test_{}", Uuid::new_v4().simple());
        sqlx::query(&format!("CREATE SCHEMA {schema}"))
            .execute(&admin)
            .await
            .expect("test schema is created");

        let search_path = format!("SET search_path TO {schema}, public");
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .after_connect(move |conn, _meta| {
                let search_path = search_path.clone();
                Box::pin(async move {
                    sqlx::query(&search_path).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await
            .expect("test pool connects");

        let outcome = match MemoryStore::connect_postgres(pool.clone(), embedder).await {
            Ok(store) => AssertUnwindSafe(body(store)).catch_unwind().await,
            Err(e) => Err(Box::new(format!("connect_postgres failed: {e}")) as Box<_>),
        };
        pool.close().await;
        let _ = sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
            .execute(&admin)
            .await;
        admin.close().await;
        if let Err(panic) = outcome {
            resume_unwind(panic);
        }
    }

    /// Each body runs once on SQLite and once on Postgres.
    macro_rules! on_both_backends {
        ($($name:ident($embedder:expr);)*) => {
            #[cfg(feature = "sqlite")]
            mod sqlite {
                use super::*;
                $(
                    #[tokio::test]
                    async fn $name() {
                        super::$name(sqlite_store(Arc::new($embedder)).await).await;
                    }
                )*
            }

            #[cfg(feature = "postgres")]
            mod postgres {
                use super::*;
                $(
                    #[tokio::test]
                    async fn $name() {
                        with_pg_store(Arc::new($embedder), super::$name).await;
                    }
                )*
            }
        };
    }

    on_both_backends! {
        test_store_and_search(ConstantEmbedder);
        test_store_with_meta(ConstantEmbedder);
        test_search_with_filter(ConstantEmbedder);
        test_delete(ConstantEmbedder);
        test_delete_expired(ConstantEmbedder);
        test_store_with_expiry(ConstantEmbedder);
        similar_is_not_crowded_out_by_other_owners(WordEmbedder);
        keyword_search_matches_any_word(WordEmbedder);
        records_get_and_delete_by_meta(WordEmbedder);
        update_by_prefix(WordEmbedder);
        list_pages_and_filters(WordEmbedder);
        reindex_replaces_embedding(WordEmbedder);
    }

    async fn test_store_and_search(store: MemoryStore) {
        store.store("User likes Rust").await.unwrap();

        let results = store.search("Rust").await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    async fn test_store_with_meta(store: MemoryStore) {
        store
            .store("User likes Rust")
            .meta("source", "manual")
            .meta("tag", "preference")
            .await
            .unwrap();

        let results = store.search("Rust").await.unwrap();
        assert!(!results.is_empty());
        assert!(
            results[0]
                .meta
                .contains(&("source".to_string(), "manual".to_string()))
        );
        assert!(
            results[0]
                .meta
                .contains(&("tag".to_string(), "preference".to_string()))
        );
    }

    async fn test_search_with_filter(store: MemoryStore) {
        store
            .store("User likes Rust")
            .meta("source", "manual")
            .await
            .unwrap();
        store
            .store("User likes Python")
            .meta("source", "capture")
            .await
            .unwrap();

        let results = store
            .search("programming")
            .filter("source", "manual")
            .await
            .unwrap();

        assert!(results.iter().all(|r| {
            r.meta
                .contains(&("source".to_string(), "manual".to_string()))
        }));
    }

    async fn test_delete(store: MemoryStore) {
        let id = store.store("temp").await.unwrap();
        store.delete(&id).await.unwrap();
        let results = store.search("temp").await.unwrap();
        assert!(results.is_empty());
    }

    async fn test_delete_expired(store: MemoryStore) {
        store.store("expired").expires_at(1).await.unwrap();
        store.store("valid").await.unwrap();

        let count = store.delete_expired().await.unwrap();
        assert_eq!(count, 1);
    }

    async fn test_store_with_expiry(store: MemoryStore) {
        let future_ts = Utc::now().timestamp() + 3600;
        store
            .store("temporary fact")
            .expires_at(future_ts)
            .await
            .unwrap();

        let results = store.search("temporary").await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].expires_at, Some(future_ts));
    }

    /// Embeds by which of four words a text mentions, so closeness is predictable.
    struct WordEmbedder;

    #[async_trait]
    impl EmbeddingProvider for WordEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let mut v: Vec<f32> = ["rust", "python", "coffee", "tea"]
                .iter()
                .map(|w| if text.contains(w) { 1.0 } else { 0.0 })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
            v.iter_mut().for_each(|x| *x /= norm);
            Ok(v)
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn name(&self) -> &str {
            "words"
        }
    }

    async fn similar_is_not_crowded_out_by_other_owners(store: MemoryStore) {
        for i in 0..20 {
            store
                .store(&format!("bob likes coffee {i}"))
                .meta("user", "bob")
                .await
                .unwrap();
        }
        store
            .store("alice likes coffee and tea")
            .meta("user", "alice")
            .await
            .unwrap();

        let hits = store
            .similar("coffee")
            .filter("user", "alice")
            .limit(1)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "alice likes coffee and tea");

        let none = store
            .similar("coffee")
            .filter("user", "alice")
            .threshold(0.99)
            .await
            .unwrap();
        assert!(none.is_empty());

        let searched = store
            .search("coffee")
            .filter("user", "alice")
            .limit(1)
            .await
            .unwrap();
        assert_eq!(searched.len(), 1);
        assert_eq!(searched[0].content, "alice likes coffee and tea");
    }

    async fn keyword_search_matches_any_word(store: MemoryStore) {
        store.store("walks the dog at noon").await.unwrap();
        let hits = store.search("who walks a cat").await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    async fn records_get_and_delete_by_meta(store: MemoryStore) {
        let alice = store
            .store("alice likes rust")
            .meta("user", "alice")
            .meta("tag", "work")
            .await
            .unwrap();
        store
            .store("alice likes tea")
            .meta("user", "alice")
            .embedding(vec![0.0, 0.0, 0.0, 1.0])
            .await
            .unwrap();
        store
            .store("bob likes rust")
            .meta("user", "bob")
            .await
            .unwrap();

        let records = store.records().filter("user", "alice").await.unwrap();
        assert_eq!(records.len(), 2);
        let tagged = store
            .records()
            .filter("user", "alice")
            .filter("tag", "work")
            .await
            .unwrap();
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].id, alice);

        let record = store.get(&alice).await.unwrap().unwrap();
        assert_eq!(record.content, "alice likes rust");
        assert!(store.get(&alice[..8]).await.unwrap().is_none());

        assert!(store.delete_matching(&[]).await.is_err());
        assert_eq!(
            store.delete_matching(&[("user", "alice")]).await.unwrap(),
            2
        );
        assert!(
            store
                .records()
                .filter("user", "alice")
                .await
                .unwrap()
                .is_empty()
        );
        let rest = store.similar("rust").await.unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].content, "bob likes rust");
    }

    async fn update_by_prefix(store: MemoryStore) {
        let id = store
            .store("likes rust")
            .meta("user", "alice")
            .meta("tag", "work")
            .await
            .unwrap();

        let short = store.update(&id[..4], Some("x"), None, None).await;
        assert!(
            short
                .unwrap_err()
                .to_string()
                .contains("at least 8 characters")
        );
        let missing = store.update("ffffffff", None, None, None).await;
        assert!(
            missing
                .unwrap_err()
                .to_string()
                .contains("No memory found with ID prefix 'ffffffff'")
        );

        let full = store
            .update(
                &id[..8],
                Some("likes coffee"),
                Some(&[("user", "bob")]),
                Some(Some(i64::MAX)),
            )
            .await
            .unwrap();
        assert_eq!(full, id);
        let record = store.get(&id).await.unwrap().unwrap();
        assert_eq!(record.content, "likes coffee");
        assert_eq!(record.meta, vec![("user".to_string(), "bob".to_string())]);
        assert_eq!(record.expires_at, Some(i64::MAX));

        let hits = store.similar("coffee").threshold(0.99).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id.as_deref(), Some(id.as_str()));

        store.update(&id, None, None, Some(None)).await.unwrap();
        let record = store.get(&id).await.unwrap().unwrap();
        assert_eq!(record.expires_at, None);
        assert_eq!(record.meta.len(), 1);

        store.delete(&id[..8]).await.unwrap();
        assert!(store.get(&id).await.unwrap().is_none());
    }

    async fn list_pages_and_filters(store: MemoryStore) {
        for content in ["first rust", "second Rust", "third tea"] {
            store.store(content).meta("user", "alice").await.unwrap();
        }
        let all = store.records().await.unwrap();
        assert_eq!(all.len(), 3);
        assert!(all.iter().all(|r| r.meta.len() == 1));

        let rust = store.records().contains("rust").await.unwrap();
        assert_eq!(rust.len(), 2);
        let listed = store.list(10, None, Some("tea")).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(store.records().limit(1).await.unwrap().len(), 1);

        let created_at = all[0].created_at;
        assert!(store.records().before(created_at).await.unwrap().is_empty());
        assert_eq!(store.records().after(created_at).await.unwrap().len(), 3);
        assert!(
            store
                .records()
                .after(created_at + 1)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .records()
                .filter("user", "bob")
                .await
                .unwrap()
                .is_empty()
        );
    }

    async fn reindex_replaces_embedding(store: MemoryStore) {
        let id = store.store("likes rust").await.unwrap();
        let listed = store.list_content_for_reindex().await.unwrap();
        assert_eq!(listed, vec![(id.clone(), "likes rust".to_string())]);

        // SQLite rebuilds an empty vector table; Postgres has nothing to repair.
        store.repair().await.unwrap();
        store.reindex(&id, vec![0.0, 0.0, 1.0, 0.0]).await.unwrap();
        let hits = store.similar("coffee").threshold(0.99).await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn postgres_rejects_a_different_dimension() {
        struct WideEmbedder;

        #[async_trait]
        impl EmbeddingProvider for WideEmbedder {
            async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0; 8])
            }
            fn dimensions(&self) -> usize {
                8
            }
            fn name(&self) -> &str {
                "wide"
            }
        }

        with_pg_store(Arc::new(WordEmbedder), |store| async move {
            let Some(Backend::Postgres(pool)) = Some(store.backend.clone()) else {
                unreachable!("with_pg_store gives a Postgres store");
            };
            let err = MemoryStore::connect_postgres(pool, Arc::new(WideEmbedder))
                .await
                .err()
                .expect("dimension mismatch is refused");
            assert!(err.to_string().contains("4-dimension"), "{err}");
        })
        .await;
    }
}
