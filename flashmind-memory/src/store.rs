//! Vector memory store using SQLite + sqlite-vec + FTS5.
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
use std::collections::{HashMap, HashSet};
use std::future::IntoFuture;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::future::BoxFuture;
use tokio::time::timeout;
use uuid::Uuid;

use crate::embeddings::EmbeddingProvider;
use crate::error::{FlashmemError, Result};
use crate::schema;

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn fetch_meta(conn: &rusqlite::Connection, memory_id: &str) -> Vec<(String, String)> {
    let Ok(mut stmt) = conn.prepare("SELECT key, value FROM memory_meta WHERE memory_id = ?1")
    else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map(rusqlite::params![memory_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return Vec::new();
    };

    rows.filter_map(|r| r.ok()).collect()
}

fn row_to_record(
    conn: &rusqlite::Connection,
    row: &rusqlite::Row,
) -> rusqlite::Result<MemoryRecord> {
    let id: String = row.get(0)?;
    let meta = fetch_meta(conn, &id);

    Ok(MemoryRecord {
        id,
        content: row.get(1)?,
        created_at: row.get(2)?,
        expires_at: row.get(3)?,
        meta,
    })
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Vector memory store backed by SQLite with sqlite-vec and FTS5.
///
/// Handles embedding generation internally. Use [`store`](MemoryStore::store)
/// and [`search`](MemoryStore::search) builders for the primary API.
#[derive(Clone)]
pub struct MemoryStore {
    conn: tokio_rusqlite::Connection,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl MemoryStore {
    /// Connect to a SQLite memory store.
    ///
    /// Creates parent directories and initializes the schema on first run.
    pub async fn connect(db_path: &Path, embedder: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let embedding_dim = embedder.dimensions();
        tracing::info!(path = %db_path.display(), "connecting to memory store");

        let conn = tokio_rusqlite::Connection::open(db_path).await?;

        conn.call(move |conn| {
            schema::init_schema(conn, embedding_dim)?;
            #[cfg(feature = "session")]
            crate::session::schema::init_session_schema(conn)?;
            Ok(())
        })
        .await?;

        tracing::info!("memory store ready");

        Ok(Self { conn, embedder })
    }

    /// Get a reference to the underlying connection.
    pub fn connection(&self) -> &tokio_rusqlite::Connection {
        &self.conn
    }

    /// Get a reference to the embedding provider.
    pub fn embedder(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedder
    }

    /// Create a [`SessionStore`](crate::session::SessionStore) sharing this connection.
    #[cfg(feature = "session")]
    pub fn session_store(&self) -> crate::session::SessionStore {
        crate::session::SessionStore::new(self.conn.clone())
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

    // -- Direct operations ----------------------------------------------------

    /// Delete a memory by ID (or prefix ≥8 chars).
    pub async fn delete(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        let id_for_err = id.clone();

        self.conn
            .call(move |conn| {
                let full_id: String = conn
                    .query_row(
                        "SELECT id FROM memories WHERE id LIKE ?1 || '%' LIMIT 1",
                        rusqlite::params![id],
                        |row| row.get(0),
                    )
                    .map_err(|_| rusqlite::Error::QueryReturnedNoRows)?;

                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM memories_vec WHERE id = ?1",
                    rusqlite::params![full_id],
                )?;
                tx.execute(
                    "DELETE FROM memories WHERE id = ?1",
                    rusqlite::params![full_id],
                )?;
                tx.commit()?;
                Ok(())
            })
            .await
            .map(|()| {
                metrics::counter!("memory.deletes").increment(1);
            })
            .map_err(|e| {
                if matches!(&e, tokio_rusqlite::Error::Error(re) if *re == rusqlite::Error::QueryReturnedNoRows) {
                    FlashmemError::Memory(format!("No memory found with ID prefix '{id_for_err}'"))
                } else {
                    e.into()
                }
            })
    }

    /// Delete all expired memories. Returns count of deleted rows.
    pub async fn delete_expired(&self) -> Result<usize> {
        let now = Utc::now().timestamp();
        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM memories_vec WHERE id IN \
                     (SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?1)",
                    rusqlite::params![now],
                )?;
                let count = tx.execute(
                    "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?1",
                    rusqlite::params![now],
                )?;
                tx.commit()?;
                tracing::debug!(count, "deleted expired memories");
                Ok(count)
            })
            .await
            .map_err(Into::into)
    }

    /// Delete all memories with the given metadata scope value.
    pub async fn delete_by_scope(&self, scope: &str) -> Result<usize> {
        let scope = scope.to_string();
        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM memories_vec WHERE id IN \
                     (SELECT memory_id FROM memory_meta WHERE key = 'scope' AND value = ?1)",
                    rusqlite::params![scope],
                )?;
                let count = tx.execute(
                    "DELETE FROM memories WHERE id IN \
                     (SELECT memory_id FROM memory_meta WHERE key = 'scope' AND value = ?1)",
                    rusqlite::params![scope],
                )?;
                tx.commit()?;
                Ok(count)
            })
            .await
            .map_err(Into::into)
    }

    /// Update content of an existing memory by ID (or prefix ≥8 chars).
    pub async fn update(
        &self,
        id: &str,
        new_content: Option<&str>,
        new_meta: Option<&[(&str, &str)]>,
        expires_at: Option<Option<i64>>,
    ) -> Result<String> {
        let new_embedding = if let Some(content) = new_content {
            Some(self.embedder.embed(content).await?)
        } else {
            None
        };

        let id = id.to_string();
        let id_for_err = id.clone();
        let content = new_content.map(|s| s.to_string());
        let meta: Option<Vec<(String, String)>> = new_meta.map(|m| {
            m.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        });

        self.conn
            .call(move |conn| {
                let full_id: String = conn
                    .query_row(
                        "SELECT id FROM memories WHERE id LIKE ?1 || '%' LIMIT 1",
                        rusqlite::params![id],
                        |row| row.get(0),
                    )
                    .map_err(|_| rusqlite::Error::QueryReturnedNoRows)?;

                let tx = conn.transaction()?;

                if let Some(ref c) = content {
                    tx.execute(
                        "UPDATE memories SET content = ?1 WHERE id = ?2",
                        rusqlite::params![c, full_id],
                    )?;
                }

                if let Some(exp) = expires_at {
                    tx.execute(
                        "UPDATE memories SET expires_at = ?1 WHERE id = ?2",
                        rusqlite::params![exp, full_id],
                    )?;
                }

                if let Some(emb) = new_embedding {
                    let bytes = embedding_to_bytes(&emb);
                    tx.execute(
                        "UPDATE memories_vec SET embedding = ?1 WHERE id = ?2",
                        rusqlite::params![bytes, full_id],
                    )?;
                }

                if let Some(ref m) = meta {
                    tx.execute(
                        "DELETE FROM memory_meta WHERE memory_id = ?1",
                        [&full_id],
                    )?;
                    for (k, v) in m {
                        tx.execute(
                            "INSERT INTO memory_meta (memory_id, key, value) VALUES (?1, ?2, ?3)",
                            rusqlite::params![full_id, k, v],
                        )?;
                    }
                }

                tx.commit()?;
                Ok(full_id)
            })
            .await
            .map_err(|e| {
                if matches!(&e, tokio_rusqlite::Error::Error(re) if *re == rusqlite::Error::QueryReturnedNoRows) {
                    FlashmemError::Memory(format!("No memory found with ID prefix '{id_for_err}'"))
                } else {
                    e.into()
                }
            })
    }

    /// Find memories similar to the given text above a score threshold.
    pub async fn find_similar(
        &self,
        text: &str,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let embedding = self.embedder.embed(text).await?;
        let results = self.search_vec(embedding, limit * 3).await?;

        let filtered: Vec<(String, f32)> = results
            .into_iter()
            .filter(|r| r.score >= threshold)
            .take(limit)
            .map(|r| (r.id.unwrap_or_default(), r.score))
            .collect();

        Ok(filtered)
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
        let filter = filter.map(|s| s.to_string());

        self.conn
            .call(move |conn| {
                let mut conditions = vec!["1=1".to_string()];
                let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

                if let Some(cursor_val) = cursor {
                    conditions.push(format!("m.created_at < ?{}", params.len() + 1));
                    params.push(Box::new(cursor_val));
                }

                if let Some(ref f) = filter {
                    conditions.push(format!("m.content LIKE ?{}", params.len() + 1));
                    params.push(Box::new(format!("%{f}%")));
                }

                if let Some(after_val) = after {
                    conditions.push(format!("m.created_at >= ?{}", params.len() + 1));
                    params.push(Box::new(after_val));
                }

                if let Some(before_val) = before {
                    conditions.push(format!("m.created_at < ?{}", params.len() + 1));
                    params.push(Box::new(before_val));
                }

                let where_clause = conditions.join(" AND ");
                let sql = format!(
                    "SELECT m.id, m.content, m.created_at, m.expires_at
                     FROM memories m
                     WHERE {where_clause}
                     ORDER BY m.created_at DESC
                     LIMIT ?{}",
                    params.len() + 1
                );
                params.push(Box::new(limit as i64));

                let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();

                let mut stmt = conn.prepare(&sql)?;
                let records: Vec<_> = stmt
                    .query_map(param_refs.as_slice(), |row| row_to_record(conn, row))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                Ok(records)
            })
            .await
            .map_err(Into::into)
    }

    /// Repair corrupted virtual tables. Returns count of memories needing re-embedding.
    pub async fn repair(&self) -> Result<usize> {
        let embedding_dim = self.embedder.dimensions();

        self.conn
            .call(move |conn| {
                conn.execute_batch(
                    "DROP TRIGGER IF EXISTS memories_ai;
                     DROP TRIGGER IF EXISTS memories_ad;
                     DROP TRIGGER IF EXISTS memories_au;
                     DROP TABLE IF EXISTS memories_vec;
                     DROP TABLE IF EXISTS memories_fts;",
                )?;

                schema::init_schema(conn, embedding_dim)?;
                conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")?;

                let count: usize =
                    conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

                tracing::info!(count, "repair complete — FTS rebuilt, vec index empty");
                Ok(count)
            })
            .await
            .map_err(Into::into)
    }

    /// Re-insert an embedding for an existing memory ID (used after repair).
    pub async fn reindex(&self, id: &str, embedding: Vec<f32>) -> Result<()> {
        let id = id.to_string();

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&embedding);
                conn.execute(
                    "INSERT OR REPLACE INTO memories_vec (id, embedding) VALUES (?1, ?2)",
                    rusqlite::params![id, bytes],
                )?;
                Ok(())
            })
            .await
            .map_err(Into::into)
    }

    /// List all memory IDs and content (for bulk re-embedding after repair).
    pub async fn list_content_for_reindex(&self) -> Result<Vec<(String, String)>> {
        self.conn
            .call(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, content FROM memories ORDER BY created_at")?;
                let rows = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            })
            .await
            .map_err(Into::into)
    }

    // -- Internal search methods ----------------------------------------------

    async fn search_vec(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let now = Utc::now().timestamp();

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                let mut stmt = conn.prepare(
                    "SELECT v.id, v.distance, m.content, m.created_at, m.expires_at
                     FROM memories_vec v
                     JOIN memories m ON m.id = v.id
                     WHERE v.embedding MATCH ?1 AND k = ?2
                       AND (m.expires_at IS NULL OR m.expires_at > ?3)
                     ORDER BY v.distance",
                )?;

                let results: Vec<_> = stmt
                    .query_map(rusqlite::params![bytes, limit, now], |row| {
                        let id: String = row.get(0)?;
                        let distance: f32 = row.get(1)?;
                        let content: String = row.get(2)?;
                        let created_at: i64 = row.get(3)?;
                        let expires_at: Option<i64> = row.get(4)?;
                        Ok((id, distance, content, created_at, expires_at))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?
                    .into_iter()
                    .map(|(id, distance, content, created_at, expires_at)| {
                        let score = 1.0 - (distance / 2.0);
                        let meta = fetch_meta(conn, &id);
                        MemorySearchResult {
                            id: Some(id),
                            content,
                            created_at,
                            expires_at,
                            score,
                            meta,
                        }
                    })
                    .collect();

                Ok(results)
            })
            .await
            .inspect(|_| {
                metrics::counter!("memory.searches").increment(1);
                metrics::histogram!("memory.search.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
            })
            .map_err(Into::into)
    }

    async fn search_hybrid_internal(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        limit: usize,
        filters: &[(String, String)],
    ) -> Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let over_limit = limit * 3;
        let now = Utc::now().timestamp();
        let query_text_owned = query_text.to_string();
        let filters_owned: Vec<(String, String)> = filters.to_vec();

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                type MemRow = (String, i64, Option<i64>);

                let mut vec_scores: HashMap<String, f32> = HashMap::new();
                let mut row_cache: HashMap<String, MemRow> = HashMap::new();
                {
                    let mut stmt = conn.prepare(
                        "SELECT v.id, v.distance, m.content, m.created_at, m.expires_at
                         FROM memories_vec v
                         JOIN memories m ON m.id = v.id
                         WHERE v.embedding MATCH ?1 AND k = ?2
                           AND (m.expires_at IS NULL OR m.expires_at > ?3)
                         ORDER BY v.distance",
                    )?;

                    let rows: Vec<(String, f32, String, i64, Option<i64>)> =
                        stmt.query_map(rusqlite::params![bytes, over_limit, now], |row| {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;

                    for (id, dist, content, created_at, expires_at) in rows {
                        vec_scores.insert(id.clone(), 1.0 - (dist / 2.0));
                        // Store content in a separate cache for later
                        row_cache.insert(id, (content, created_at, expires_at));
                    }
                }

                let fts_scores: HashMap<String, f32> = {
                    let clean_query: String = query_text_owned
                        .chars()
                        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
                        .collect::<String>()
                        .trim()
                        .to_string();

                    if clean_query.is_empty() {
                        HashMap::new()
                    } else {
                        let mut stmt = conn.prepare(
                            "SELECT m.id, bm25(memories_fts) as score FROM memories m
                             JOIN memories_fts f ON f.rowid = m.rowid
                             WHERE memories_fts MATCH ?1 AND (m.expires_at IS NULL OR m.expires_at > ?2)
                             ORDER BY score LIMIT ?3",
                        )?;
                        let rows: Vec<(String, f64)> = stmt
                            .query_map(rusqlite::params![clean_query, now, over_limit], |row| {
                                Ok((row.get(0)?, row.get(1)?))
                            })?
                            .collect::<rusqlite::Result<Vec<_>>>()?;

                        if rows.is_empty() {
                            HashMap::new()
                        } else {
                            let min_bm25 = rows.iter().map(|(_, s)| *s).fold(0.0f64, f64::min);
                            let max_bm25 = rows.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
                            let range = min_bm25 - max_bm25;
                            rows.into_iter()
                                .map(|(id, bm25)| {
                                    let score = if range < 0.0 { ((bm25 - max_bm25) / range) as f32 } else { 1.0 };
                                    (id, score)
                                })
                                .collect()
                        }
                    }
                };

                let all_ids: HashSet<&str> = vec_scores
                    .keys()
                    .map(|s| s.as_str())
                    .chain(fts_scores.keys().map(|s| s.as_str()))
                    .collect();

                let mut results = Vec::new();
                for id in all_ids {
                    let vec_sim = vec_scores.get(id).copied().unwrap_or(0.0);
                    let fts = fts_scores.get(id).copied().unwrap_or(0.0);
                    let score = (vec_sim + fts * 0.3).min(1.0);

                    let (content, created_at, expires_at) =
                        if let Some(cached) = row_cache.remove(id) {
                            cached
                        } else if let Ok(r) = conn.query_row(
                            "SELECT content, created_at, expires_at FROM memories WHERE id = ?1",
                            [id],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        ) {
                            r
                        } else {
                            continue;
                        };

                    let meta = fetch_meta(conn, id);

                    // Apply meta filters
                    if !filters_owned.is_empty() {
                        let matches = filters_owned.iter().all(|(k, v)| {
                            meta.iter().any(|(mk, mv)| mk == k && mv == v)
                        });
                        if !matches {
                            continue;
                        }
                    }

                    results.push(MemorySearchResult {
                        id: Some(id.to_string()),
                        content,
                        created_at,
                        expires_at,
                        score,
                        meta,
                    });
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                results.truncate(limit);
                Ok(results)
            })
            .await
            .inspect(|results| {
                metrics::counter!("memory.hybrid_searches").increment(1);
                metrics::histogram!("memory.search.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                metrics::histogram!("memory.search.results_count")
                    .record(results.len() as f64);
            })
            .map_err(Into::into)
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
}

impl<'a> StoreBuilder<'a> {
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

            let embedding = self.store.embedder.embed(self.content).await?;

            let id = Uuid::new_v4().to_string();
            let created_at = Utc::now().timestamp();
            let content = self.content.to_string();
            let meta = self.meta;
            let expires_at = self.expires_at;
            let id_clone = id.clone();

            self.store
                .conn
                .call(move |conn| {
                    let tx = conn.transaction()?;

                    tx.execute(
                        "INSERT INTO memories (id, content, created_at, expires_at)
                         VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![id_clone, content, created_at, expires_at],
                    )?;

                    let bytes = embedding_to_bytes(&embedding);
                    tx.execute(
                        "INSERT INTO memories_vec (id, embedding) VALUES (?1, ?2)",
                        rusqlite::params![id_clone, bytes],
                    )?;

                    for (key, value) in &meta {
                        tx.execute(
                            "INSERT INTO memory_meta (memory_id, key, value) VALUES (?1, ?2, ?3)",
                            rusqlite::params![id_clone, key, value],
                        )?;
                    }

                    tx.commit()?;
                    Ok(id_clone)
                })
                .await
                .inspect(|_| {
                    metrics::counter!("memory.stores").increment(1);
                    metrics::histogram!("memory.store.duration_seconds")
                        .record(start.elapsed().as_secs_f64());
                })
                .map_err(Into::into)
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingProvider;
    use async_trait::async_trait;
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

    async fn test_store() -> MemoryStore {
        crate::test_util::register_sqlite_vec();
        let dir = tempdir().unwrap();
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(ConstantEmbedder);
        let store = MemoryStore::connect(&dir.path().join("test.db"), embedder)
            .await
            .unwrap();
        std::mem::forget(dir);
        store
    }

    #[tokio::test]
    async fn test_store_and_search() {
        let store = test_store().await;
        store.store("User likes Rust").await.unwrap();

        let results = store.search("Rust").await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_store_with_meta() {
        let store = test_store().await;
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

    #[tokio::test]
    async fn test_search_with_filter() {
        let store = test_store().await;
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

    #[tokio::test]
    async fn test_delete() {
        let store = test_store().await;
        let id = store.store("temp").await.unwrap();
        store.delete(&id).await.unwrap();
        let results = store.search("temp").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_expired() {
        let store = test_store().await;
        store.store("expired").expires_at(1).await.unwrap();
        store.store("valid").await.unwrap();

        let count = store.delete_expired().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_store_with_expiry() {
        let store = test_store().await;
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
}
