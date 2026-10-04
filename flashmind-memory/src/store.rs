//! Vector memory store using SQLite + sqlite-vec + FTS5, or Postgres +
//! pgvector with the `postgres` feature.
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

// Without the `postgres` feature, `Backend` has one variant, so each
// `let conn = match &self.backend` is infallible.
#![cfg_attr(
    not(feature = "postgres"),
    allow(clippy::infallible_destructuring_match)
)]

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

#[cfg(feature = "postgres")]
use sqlx::PgPool;

use crate::embeddings::EmbeddingProvider;
use crate::error::{FlashmemError, Result};
#[cfg(feature = "postgres")]
use crate::pg;
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

/// SQL conditions requiring memory `m` to carry every metadata pair, with
/// their values pushed onto `params`.
fn meta_conditions(
    filters: &[(String, String)],
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
) -> Vec<String> {
    filters
        .iter()
        .map(|(key, value)| {
            let key_at = params.len() + 1;
            params.push(Box::new(key.clone()));
            params.push(Box::new(value.clone()));
            format!(
                "m.id IN (SELECT memory_id FROM memory_meta WHERE key = ?{key_at} AND value = ?{})",
                key_at + 1
            )
        })
        .collect()
}

/// Turns free text into an FTS5 query that matches any of its words.
fn fts_query(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| format!("\"{word}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// A vector match before metadata is attached.
struct VecRow {
    id: String,
    score: f32,
    content: String,
    created_at: i64,
    expires_at: Option<i64>,
}

/// The `k` unexpired memories closest to `bytes`. With metadata filters the
/// matching memories are scanned exactly, so other memories cannot crowd
/// them out of the nearest `k`.
fn nearest(
    conn: &rusqlite::Connection,
    bytes: &[u8],
    k: usize,
    now: i64,
    filters: &[(String, String)],
) -> rusqlite::Result<Vec<VecRow>> {
    let map = |row: &rusqlite::Row| {
        let distance: f32 = row.get(1)?;
        Ok(VecRow {
            id: row.get(0)?,
            score: 1.0 - (distance / 2.0),
            content: row.get(2)?,
            created_at: row.get(3)?,
            expires_at: row.get(4)?,
        })
    };
    if filters.is_empty() {
        let mut stmt = conn.prepare(
            "SELECT v.id, v.distance, m.content, m.created_at, m.expires_at
             FROM memories_vec v
             JOIN memories m ON m.id = v.id
             WHERE v.embedding MATCH ?1 AND k = ?2
               AND (m.expires_at IS NULL OR m.expires_at > ?3)
             ORDER BY v.distance",
        )?;
        return stmt
            .query_map(rusqlite::params![bytes, k as i64, now], map)?
            .collect();
    }
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
        vec![Box::new(bytes.to_vec()), Box::new(now), Box::new(k as i64)];
    let conditions = meta_conditions(filters, &mut params).join(" AND ");
    let sql = format!(
        "SELECT m.id, vec_distance_l2(v.embedding, ?1) AS distance, m.content, m.created_at,
                m.expires_at
         FROM memories m
         JOIN memories_vec v ON v.id = m.id
         WHERE (m.expires_at IS NULL OR m.expires_at > ?2) AND {conditions}
         ORDER BY distance
         LIMIT ?3"
    );
    let refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    stmt.query_map(refs.as_slice(), map)?.collect()
}

fn validate_memory_prefix(prefix: &str) -> std::result::Result<(), rusqlite::Error> {
    if prefix.len() < 8 {
        return Err(rusqlite::Error::InvalidParameterName(
            "memory ID prefix must be at least 8 characters".into(),
        ));
    }
    Ok(())
}

fn resolve_memory_id(conn: &rusqlite::Connection, prefix: &str) -> rusqlite::Result<String> {
    validate_memory_prefix(prefix)?;
    let mut stmt = conn.prepare("SELECT id FROM memories WHERE id LIKE ?1 || '%' LIMIT 2")?;
    let ids = stmt
        .query_map(rusqlite::params![prefix], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    match ids.len() {
        0 => Err(rusqlite::Error::QueryReturnedNoRows),
        1 => Ok(ids.into_iter().next().expect("single id")),
        _ => Err(rusqlite::Error::InvalidParameterName(
            "memory ID prefix is ambiguous".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Where a [`MemoryStore`] keeps its rows.
#[derive(Clone)]
enum Backend {
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
    pub async fn connect(db_path: &Path, embedder: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        crate::register_sqlite_vec();

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let embedding_dim = embedding_dim(embedder.as_ref()).await?;
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
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                return pg::delete(pool, id).await.map(|()| {
                    metrics::counter!("memory.deletes").increment(1);
                });
            }
        };
        let id = id.to_string();
        let id_for_err = id.clone();

        conn.call(move |conn| {
                let full_id = resolve_memory_id(conn, &id)?;

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
                } else if let tokio_rusqlite::Error::Error(rusqlite::Error::InvalidParameterName(msg)) = &e {
                    FlashmemError::Memory(format!("{msg}: '{id_for_err}'"))
                } else {
                    e.into()
                }
            })
    }

    /// Delete all expired memories. Returns count of deleted rows.
    pub async fn delete_expired(&self) -> Result<usize> {
        let now = Utc::now().timestamp();
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                let count = pg::delete_expired(pool, now).await?;
                tracing::debug!(count, "deleted expired memories");
                return Ok(count);
            }
        };
        conn.call(move |conn| {
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
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                return pg::delete_matching(pool, &filters).await.inspect(|count| {
                    metrics::counter!("memory.deletes").increment(*count as u64);
                });
            }
        };
        conn.call(move |conn| {
                let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                let conditions = meta_conditions(&filters, &mut params).join(" AND ");
                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();
                let tx = conn.transaction()?;
                tx.execute(
                    &format!(
                        "DELETE FROM memories_vec WHERE id IN \
                         (SELECT m.id FROM memories m WHERE {conditions})"
                    ),
                    refs.as_slice(),
                )?;
                let count = tx.execute(
                    &format!("DELETE FROM memories WHERE id IN (SELECT m.id FROM memories m WHERE {conditions})"),
                    refs.as_slice(),
                )?;
                tx.commit()?;
                Ok(count)
            })
            .await
            .inspect(|count| {
                metrics::counter!("memory.deletes").increment(*count as u64);
            })
            .map_err(Into::into)
    }

    /// The memory with exactly this ID, if there is one. Unlike the
    /// prefix lookups, a short ID never matches.
    pub async fn get(&self, id: &str) -> Result<Option<MemoryRecord>> {
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => return pg::get(pool, id).await,
        };
        let id = id.to_string();
        conn.call(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, created_at, expires_at FROM memories WHERE id = ?1",
            )?;
            let mut rows = stmt.query_map([&id], |row| row_to_record(conn, row))?;
            rows.next().transpose()
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

        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => {
                let changes = pg::MemoryUpdate {
                    content,
                    embedding: new_embedding,
                    meta,
                    expires_at,
                };
                return pg::update(pool, &id, changes).await;
            }
        };
        conn.call(move |conn| {
                let full_id = resolve_memory_id(conn, &id)?;

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
                } else if let tokio_rusqlite::Error::Error(rusqlite::Error::InvalidParameterName(msg)) = &e {
                    FlashmemError::Memory(format!("{msg}: '{id_for_err}'"))
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
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(_) => return Ok(0),
        };
        let embedding_dim = self.embedder.dimensions();

        conn.call(move |conn| {
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
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => return pg::reindex(pool, id, embedding).await,
        };
        let id = id.to_string();

        conn.call(move |conn| {
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
        let conn = match &self.backend {
            Backend::Sqlite(conn) => conn,
            #[cfg(feature = "postgres")]
            Backend::Postgres(pool) => return pg::list_content_for_reindex(pool).await,
        };
        conn.call(|conn| {
            let mut stmt = conn.prepare("SELECT id, content FROM memories ORDER BY created_at")?;
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
        filters: Vec<(String, String)>,
    ) -> Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let now = Utc::now().timestamp();

        let results = match &self.backend {
            Backend::Sqlite(conn) => conn
                .call(move |conn| {
                    let bytes = embedding_to_bytes(&query_embedding);
                    let results = nearest(conn, &bytes, limit, now, &filters)?
                        .into_iter()
                        .map(|row| MemorySearchResult {
                            meta: fetch_meta(conn, &row.id),
                            id: Some(row.id),
                            content: row.content,
                            created_at: row.created_at,
                            expires_at: row.expires_at,
                            score: row.score,
                        })
                        .collect();
                    Ok(results)
                })
                .await
                .map_err(Into::into),
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
        let over_limit = limit * 3;
        let now = Utc::now().timestamp();
        let query_text_owned = query_text.to_string();
        let filters_owned: Vec<(String, String)> = filters.to_vec();

        let results = match &self.backend {
            Backend::Sqlite(conn) => conn.call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                type MemRow = (String, i64, Option<i64>);

                let mut vec_scores: HashMap<String, f32> = HashMap::new();
                let mut row_cache: HashMap<String, MemRow> = HashMap::new();
                for row in nearest(conn, &bytes, over_limit, now, &filters_owned)? {
                    vec_scores.insert(row.id.clone(), row.score);
                    row_cache.insert(row.id, (row.content, row.created_at, row.expires_at));
                }

                let fts_scores: HashMap<String, f32> = {
                    let fts = fts_query(&query_text_owned);

                    if fts.is_empty() {
                        HashMap::new()
                    } else {
                        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
                            vec![Box::new(fts), Box::new(now), Box::new(over_limit as i64)];
                        let conditions: String = meta_conditions(&filters_owned, &mut params)
                            .into_iter()
                            .map(|c| format!(" AND {c}"))
                            .collect();
                        let refs: Vec<&dyn rusqlite::types::ToSql> =
                            params.iter().map(|p| p.as_ref()).collect();
                        let mut stmt = conn.prepare(&format!(
                            "SELECT m.id, bm25(memories_fts) as score FROM memories m
                             JOIN memories_fts f ON f.rowid = m.rowid
                             WHERE memories_fts MATCH ?1 AND (m.expires_at IS NULL OR m.expires_at > ?2){conditions}
                             ORDER BY score LIMIT ?3",
                        ))?;
                        let rows: Vec<(String, f64)> = stmt
                            .query_map(refs.as_slice(), |row| {
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
            .map_err(Into::into),
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

            let id = Uuid::new_v4().to_string();
            let created_at = Utc::now().timestamp();
            let content = self.content.to_string();
            let meta = self.meta;
            let expires_at = self.expires_at;
            let id_clone = id.clone();

            let conn = match &self.store.backend {
                Backend::Sqlite(conn) => conn,
                #[cfg(feature = "postgres")]
                Backend::Postgres(pool) => {
                    let memory = pg::NewMemory {
                        id,
                        content,
                        created_at,
                        expires_at,
                        embedding,
                        meta,
                    };
                    return pg::insert(pool, memory).await.inspect(|_| {
                        metrics::counter!("memory.stores").increment(1);
                        metrics::histogram!("memory.store.duration_seconds")
                            .record(start.elapsed().as_secs_f64());
                    });
                }
            };
            conn.call(move |conn| {
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
        let ListBuilder {
            store,
            filters,
            contains,
            cursor,
            after,
            before,
            limit,
        } = self;
        Box::pin(async move {
            let conn = match &store.backend {
                Backend::Sqlite(conn) => conn,
                #[cfg(feature = "postgres")]
                Backend::Postgres(pool) => {
                    let query = pg::ListQuery {
                        filters,
                        contains,
                        cursor,
                        after,
                        before,
                        limit,
                    };
                    return pg::list(pool, query).await;
                }
            };
            conn.call(move |conn| {
                let mut conditions = vec!["1=1".to_string()];
                let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

                for upper in [cursor, before].into_iter().flatten() {
                    conditions.push(format!("m.created_at < ?{}", params.len() + 1));
                    params.push(Box::new(upper));
                }

                if let Some(text) = contains {
                    conditions.push(format!("m.content LIKE ?{}", params.len() + 1));
                    params.push(Box::new(format!("%{text}%")));
                }

                if let Some(after) = after {
                    conditions.push(format!("m.created_at >= ?{}", params.len() + 1));
                    params.push(Box::new(after));
                }

                conditions.extend(meta_conditions(&filters, &mut params));

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

                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    params.iter().map(|p| p.as_ref()).collect();

                let mut stmt = conn.prepare(&sql)?;
                let records = stmt
                    .query_map(refs.as_slice(), |row| row_to_record(conn, row))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(records)
            })
            .await
            .map_err(Into::into)
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
