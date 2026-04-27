//! Vector memory store using SQLite + sqlite-vec + FTS5.
//!
//! Provides hybrid search (vector similarity + BM25 keyword + RRF reranking),
//! memory tagging, TTL-based expiry, and cosine-similarity deduplication.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use chrono::Utc;
use tokio::time::timeout;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{FlashmemError, Result};
use std::str::FromStr;

use crate::schema::{self, Scope, Source, Tag};
use crate::search;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A search result with relevance score.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub id: Option<String>,
    pub content: String,
    pub source: Source,
    pub chat_key: Option<String>,
    pub created_at: i64,
    pub score: f32,
    pub tags: Vec<Tag>,
    pub expires_at: Option<i64>,
}

/// A full memory record with metadata, used by curation agent.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub source: Source,
    pub chat_key: Option<String>,
    pub identity: Option<String>,
    pub created_at: i64,
    pub tags: Vec<Tag>,
    pub expires_at: Option<i64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Raw row from vector search: (id, distance, content, source, chat_key, created_at, expires_at).
type VecSearchRow = (
    String,
    f32,
    String,
    String,
    Option<String>,
    i64,
    Option<i64>,
);

/// Convert a `Vec<f32>` embedding to little-endian bytes for sqlite-vec.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Convert little-endian bytes back to a `Vec<f32>` embedding.
#[allow(dead_code)]
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Compute cosine similarity between two embeddings.
/// https://en.wikipedia.org/wiki/Cosine_similarity#Definition
#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // dot product
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let mag_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Fetch tags for a memory ID, parsing each tag name through `Tag::from_str`.
fn fetch_tags(conn: &rusqlite::Connection, memory_id: &str) -> Vec<Tag> {
    let Ok(mut stmt) = conn.prepare(
        "SELECT t.name FROM memory_tags mt
         JOIN tags t ON t.id = mt.tag_id
         WHERE mt.memory_id = ?1
         ORDER BY t.name",
    ) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map(rusqlite::params![memory_id], |row| row.get::<_, String>(0))
    else {
        return Vec::new();
    };

    rows.filter_map(|r| r.ok())
        .filter_map(|name| Tag::from_str(&name).ok())
        .collect()
}

/// Row → MemorySearchResult (expects columns: id, content, source, chat_key, created_at, expires_at).
fn row_to_search_result(
    conn: &rusqlite::Connection,
    row: &rusqlite::Row,
    score: f32,
) -> rusqlite::Result<MemorySearchResult> {
    let id: String = row.get("id")?;
    let created_at: i64 = row.get("created_at")?;
    let source_str: String = row.get("source")?;
    let tags = fetch_tags(conn, &id);

    Ok(MemorySearchResult {
        id: Some(id),
        content: row.get("content")?,
        source: Source::from_str(&source_str).unwrap_or(Source::Manual),
        chat_key: row.get("chat_key")?,
        created_at,
        score,
        tags,
        expires_at: row.get("expires_at")?,
    })
}

/// Check if a memory's chat_key matches the requested scope.
///
/// - `scope = Some(Scope::Global)` → only memories with NULL chat_key
/// - `scope = Some(Scope::Local)`  → only memories matching the provided chat_key
/// - `scope = None`                → memories matching chat_key OR NULL (local + global)
fn matches_scope(
    mem_chat_key: &Option<String>,
    chat_key: Option<&str>,
    scope: Option<Scope>,
) -> bool {
    match scope {
        Some(Scope::Global) => mem_chat_key.is_none(),
        Some(Scope::Local) => chat_key.is_some_and(|ck| mem_chat_key.as_deref() == Some(ck)),
        None => {
            chat_key.is_none_or(|ck| mem_chat_key.as_deref() == Some(ck) || mem_chat_key.is_none())
        }
    }
}

/// Row → MemoryRecord. Expects columns: id, content, source, chat_key, identity, created_at, expires_at.
fn row_to_record(
    conn: &rusqlite::Connection,
    row: &rusqlite::Row,
) -> rusqlite::Result<MemoryRecord> {
    let id: String = row.get(0)?;
    let source_str: String = row.get(2)?;
    let created_at: i64 = row.get(5)?;
    let tags = fetch_tags(conn, &id);

    Ok(MemoryRecord {
        id,
        content: row.get(1)?,
        source: Source::from_str(&source_str).unwrap_or(Source::Manual),
        chat_key: row.get(3)?,
        identity: row.get(4)?,
        created_at,
        tags,
        expires_at: row.get(6)?,
    })
}

// ---------------------------------------------------------------------------
// VectorMemory
// ---------------------------------------------------------------------------

/// Vector memory store backed by SQLite with sqlite-vec for vector search
/// and FTS5 for keyword search.
///
/// The underlying `tokio_rusqlite::Connection` is `Clone + Send + Sync`,
/// so this can be safely shared across threads.
#[derive(Clone)]
pub struct DbStore {
    conn: tokio_rusqlite::Connection,
}

impl DbStore {
    /// Connect to SQLite at the given path.
    /// Creates parent directories and initializes schema if needed.
    pub async fn connect(db_path: &Path, embedding_dim: usize) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!(path = %db_path.display(), "connecting to SQLite memory store");

        let conn = tokio_rusqlite::Connection::open(db_path).await?;

        conn.call(move |conn| {
            schema::init_schema(conn, embedding_dim)?;
            Ok(())
        })
        .await?;

        info!("SQLite memory store ready");

        Ok(Self { conn })
    }

    /// Get a reference to the underlying connection.
    pub fn connection(&self) -> &tokio_rusqlite::Connection {
        &self.conn
    }

    /// Repair corrupted virtual tables (memories_vec and memories_fts).
    ///
    /// Drops and recreates both virtual tables, rebuilds FTS from existing
    /// `memories` rows, and returns the number of memories that need
    /// re-embedding (caller must generate embeddings and call `reindex`).
    pub async fn repair(&self, embedding_dim: usize) -> Result<usize> {
        self.conn
            .call(move |conn| {
                // Drop corrupted virtual tables and their sync triggers.
                conn.execute_batch(
                    "DROP TRIGGER IF EXISTS memories_ai;
                     DROP TRIGGER IF EXISTS memories_ad;
                     DROP TRIGGER IF EXISTS memories_au;
                     DROP TABLE IF EXISTS memories_vec;
                     DROP TABLE IF EXISTS memories_fts;",
                )?;

                // Recreate schema (virtual tables + triggers).
                schema::init_schema(conn, embedding_dim)?;

                // Rebuild FTS index from existing memories rows.
                conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")?;

                let count: usize =
                    conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

                info!(count, "repair complete — FTS rebuilt, vec index empty");
                Ok(count)
            })
            .await
            .map_err(Into::into)
    }

    /// Re-insert an embedding for an existing memory ID.
    ///
    /// Used after `repair()` to backfill the vector index.
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

    /// List all memory IDs and their content (for bulk re-embedding after repair).
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

    /// Store a new memory entry. Returns the generated UUID.
    #[allow(clippy::too_many_arguments)]
    pub async fn store(
        &self,
        content: &str,
        embedding: Vec<f32>,
        source: Source,
        chat_key: Option<&str>,
        identity: Option<&str>,
        tags: &[Tag],
        expires_at: Option<i64>,
        tool_name: Option<&str>,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let created_at = Utc::now().timestamp();
        let content = content.to_string();
        let source_str = source.to_string();
        let chat_key = chat_key.map(|s| s.to_string());
        let identity = identity.map(|s| s.to_string());
        let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
        let tool_name = tool_name.map(|s| s.to_string());
        let id_clone = id.clone();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                tx.execute(
                    "INSERT INTO memories (id, content, source, chat_key, identity, tool_name, created_at, expires_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![id_clone, content, source_str, chat_key, identity, tool_name, created_at, expires_at],
                )?;

                let bytes = embedding_to_bytes(&embedding);
                tx.execute(
                    "INSERT INTO memories_vec (id, embedding) VALUES (?1, ?2)",
                    rusqlite::params![id_clone, bytes],
                )?;

                for tag_name in &tags {
                    tx.execute(
                        "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id)
                         SELECT ?1, id FROM tags WHERE name = ?2",
                        rusqlite::params![id_clone, tag_name],
                    )?;
                }

                tx.commit()?;
                Ok(id_clone)
            })
            .await
            .map_err(Into::into)
    }

    /// Vector-only search. Returns results sorted by score descending.
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        self.search_filtered(query_embedding, limit, None, None)
            .await
    }

    /// Vector search with optional chat_key and scope filtering.
    pub async fn search_filtered(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        chat_key: Option<&str>,
        scope: Option<Scope>,
    ) -> Result<Vec<MemorySearchResult>> {
        let now = Utc::now().timestamp();
        let chat_key = chat_key.map(|s| s.to_string());

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                // Over-retrieve then filter in Rust (sqlite-vec doesn't support
                // additional WHERE predicates alongside MATCH).
                let over_limit = limit * 5;

                let mut stmt = conn.prepare(
                    "SELECT v.id, v.distance, m.content, m.source, m.chat_key,
                            m.created_at, m.expires_at
                     FROM memories_vec v
                     JOIN memories m ON m.id = v.id
                     WHERE v.embedding MATCH ?1 AND k = ?2
                       AND (m.expires_at IS NULL OR m.expires_at > ?3)
                     ORDER BY v.distance",
                )?;

                let rows: Vec<VecSearchRow> = stmt
                    .query_map(
                        rusqlite::params![bytes, over_limit, now],
                        |row: &rusqlite::Row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, f32>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                                row.get::<_, Option<String>>(4)?,
                                row.get::<_, i64>(5)?,
                                row.get::<_, Option<i64>>(6)?,
                            ))
                        },
                    )?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                let results = rows
                    .into_iter()
                    .filter(|(_, _, _, _, ck, _, _)| matches_scope(ck, chat_key.as_deref(), scope))
                    .take(limit)
                    .map(
                        |(id, distance, content, source, chat_key, created_at, expires_at)| {
                            let score = 1.0 - (distance / 2.0);
                            let tags = fetch_tags(conn, &id);
                            MemorySearchResult {
                                id: Some(id),
                                content,
                                source: Source::from_str(&source).unwrap_or(Source::Manual),
                                chat_key,
                                created_at,
                                score,
                                tags,
                                expires_at,
                            }
                        },
                    )
                    .collect();

                Ok(results)
            })
            .await
            .map_err(Into::into)
    }

    /// Search across both chat-scoped and global memories, deduplicated.
    pub async fn search_multi_context(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        chat_key: Option<&str>,
    ) -> Result<Vec<MemorySearchResult>> {
        let chat_key = match chat_key {
            Some(ck) => ck.to_string(),
            None => return self.search(query_embedding, limit).await,
        };

        let chat_results = self
            .search_filtered(
                query_embedding.clone(),
                limit,
                Some(&chat_key),
                Some(Scope::Local),
            )
            .await?;

        let global_results = self
            .search_filtered(query_embedding, limit, None, Some(Scope::Global))
            .await?;

        let mut seen = HashSet::new();
        let mut merged = Vec::new();

        for r in chat_results.into_iter().chain(global_results) {
            let id = r.id.clone().unwrap_or_default();
            if seen.insert(id) {
                merged.push(r);
            }
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        merged.truncate(limit);

        Ok(merged)
    }

    /// Hybrid search: combines vector similarity with FTS5 keyword matching.
    ///
    /// **How it works:**
    /// 1. **Vector search** (sqlite-vec): Finds memories with similar embeddings using cosine similarity
    /// 2. **FTS search** (FTS5): Finds memories matching keywords using BM25 ranking
    /// 3. **Combine**: Union of both result sets, scored together
    ///
    /// **Scoring:**
    /// - Vector score: `1.0 - (distance / 2.0)` → ranges 0-1 (higher = more similar)
    /// - FTS score: BM25 (negative, normalized to 0-1 where higher = better keyword match)
    /// - Combined: `min(1.0, vector_score + fts_score * 0.3)`
    ///
    /// The FTS weight (0.3) boosts results that match both semantically and keyword-wise.
    pub async fn search_hybrid(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        limit: usize,
        chat_key: Option<&str>,
        scope: Option<Scope>,
    ) -> Result<Vec<MemorySearchResult>> {
        // Over-retrieve by 3x to ensure enough candidates after filtering by scope
        let over_limit = limit * 3;
        let now = Utc::now().timestamp();
        let query_text_owned = query_text.to_string();
        let chat_key_owned = chat_key.map(|s| s.to_string());

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                // Row data fetched from memories table, keyed by ID.
                type MemRow = (String, String, Option<String>, i64, Option<i64>); // content, source, chat_key, created_at, expires_at

                // Step 1: Vector search via sqlite-vec
                //
                // Uses cosine distance: 0 = identical, 2 = opposite.
                // Convert to similarity: 1.0 - (distance / 2.0)
                // This gives 0-1 range where higher = more similar.
                let mut vec_scores: HashMap<String, f32> = HashMap::new();
                let mut row_cache: HashMap<String, MemRow> = HashMap::new();
                {
                    let mut stmt = conn.prepare(
                        "SELECT v.id, v.distance, m.content, m.source, m.chat_key,
                                m.created_at, m.expires_at
                         FROM memories_vec v
                         JOIN memories m ON m.id = v.id
                         WHERE v.embedding MATCH ?1 AND k = ?2
                           AND (m.expires_at IS NULL OR m.expires_at > ?3)
                         ORDER BY v.distance",
                    )?;
                    #[allow(clippy::type_complexity)]
                    let rows: Vec<(String, f32, String, String, Option<String>, i64, Option<i64>)> =
                        stmt.query_map(rusqlite::params![bytes, over_limit, now], |row| {
                            Ok((
                                row.get(0)?, row.get(1)?, row.get(2)?,
                                row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?,
                            ))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;

                    for (id, dist, content, source, chat_key, created_at, expires_at) in rows {
                        if !matches_scope(&chat_key, chat_key_owned.as_deref(), scope) {
                            continue;
                        }
                        vec_scores.insert(id.clone(), 1.0 - (dist / 2.0));
                        row_cache.insert(id, (content, source, chat_key, created_at, expires_at));
                    }
                }

                // Step 2: FTS5 keyword search
                //
                // Uses BM25 which accounts for term frequency, document length, and IDF.
                // BM25 returns negative values (more negative = better match).
                // We normalize to 0-1 where higher = better.
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
                            "SELECT m.id, m.chat_key, bm25(memories_fts) as score FROM memories m
                             JOIN memories_fts f ON f.rowid = m.rowid
                             WHERE memories_fts MATCH ?1 AND (m.expires_at IS NULL OR m.expires_at > ?2)
                             ORDER BY score LIMIT ?3",
                        )?;
                        let rows: Vec<(String, Option<String>, f64)> = stmt
                            .query_map(rusqlite::params![clean_query, now, over_limit], |row| {
                                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                            })?
                            .collect::<rusqlite::Result<Vec<_>>>()?;

                        if rows.is_empty() {
                            HashMap::new()
                        } else {
                            let min_bm25 = rows.iter().map(|(_, _, s)| *s).fold(0.0f64, f64::min);
                            let max_bm25 = rows.iter().map(|(_, _, s)| *s).fold(0.0f64, f64::max);
                            let range = min_bm25 - max_bm25;
                            rows.into_iter()
                                .filter(|(_, ck, _)| matches_scope(ck, chat_key_owned.as_deref(), scope))
                                .map(|(id, _, bm25)| {
                                    let score = if range < 0.0 { ((bm25 - max_bm25) / range) as f32 } else { 1.0 };
                                    (id, score)
                                })
                                .collect()
                        }
                    }
                };

                // Step 3: Combine and score
                //
                // Union of both result sets. FTS weighted at 0.3.
                let all_ids: HashSet<&str> = vec_scores.keys().map(|s| s.as_str())
                    .chain(fts_scores.keys().map(|s| s.as_str()))
                    .collect();

                let mut results = Vec::new();
                for id in all_ids {
                    let vec_sim = vec_scores.get(id).copied().unwrap_or(0.0);
                    let fts = fts_scores.get(id).copied().unwrap_or(0.0);
                    let score = (vec_sim + fts * 0.3).min(1.0);

                    // Use cached row data from vec search, or fetch for FTS-only results
                    let row = if let Some(cached) = row_cache.remove(id) {
                        cached
                    } else if let Ok(r) = conn.query_row(
                        "SELECT content, source, chat_key, created_at, expires_at FROM memories WHERE id = ?1",
                        [id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
                    ) {
                        r
                    } else {
                        continue;
                    };

                    let (content, source, chat_key, created_at, expires_at) = row;
                    results.push(MemorySearchResult {
                        id: Some(id.to_string()),
                        content,
                        source: Source::from_str(&source).unwrap_or(Source::Manual),
                        chat_key,
                        created_at,
                        score,
                        tags: fetch_tags(conn, id),
                        expires_at,
                    });
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                results.truncate(limit);
                Ok(results)
            })
            .await
            .map_err(Into::into)
    }

    /// Multi-context hybrid search (chat-scoped + global).
    pub async fn search_multi_context_hybrid(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        limit: usize,
        chat_key: Option<&str>,
    ) -> Result<Vec<MemorySearchResult>> {
        let chat_key = match chat_key {
            Some(ck) => ck,
            None => {
                return self
                    .search_hybrid(query_embedding, query_text, limit, None, None)
                    .await;
            }
        };

        let chat_results = self
            .search_hybrid(
                query_embedding.clone(),
                query_text,
                limit,
                Some(chat_key),
                Some(Scope::Local),
            )
            .await?;

        let global_results = self
            .search_hybrid(
                query_embedding,
                query_text,
                limit,
                None,
                Some(Scope::Global),
            )
            .await?;

        let mut seen = HashSet::new();
        let mut merged = Vec::new();

        for r in chat_results.into_iter().chain(global_results) {
            let id = r.id.clone().unwrap_or_default();
            if seen.insert(id) {
                merged.push(r);
            }
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        merged.truncate(limit);

        Ok(merged)
    }

    /// Multi-query search: run multiple embeddings in parallel, deduplicate by ID.
    pub async fn search_multi_query(
        &self,
        query_embeddings: Vec<(Vec<f32>, String)>,
        limit: usize,
        chat_key: Option<&str>,
        scope: Option<Scope>,
    ) -> Result<Vec<MemorySearchResult>> {
        let chat_key_owned = chat_key.map(|s| s.to_string());

        let futs: Vec<_> = query_embeddings
            .into_iter()
            .map(|(emb, text)| {
                let ck = chat_key_owned.clone();
                let this = self.clone();

                async move {
                    let result = timeout(
                        Duration::from_secs(3),
                        this.search_hybrid(emb, &text, limit, ck.as_deref(), scope),
                    )
                    .await;
                    match result {
                        Ok(Ok(r)) => {
                            debug!(
                                query = %text,
                                result_count = r.len(),
                                scores = ?r.iter().map(|m| m.score).collect::<Vec<_>>(),
                                "search_hybrid completed"
                            );
                            Some(r)
                        }
                        Ok(Err(e)) => {
                            warn!("search_hybrid failed: {e}");
                            None
                        }
                        Err(_) => {
                            warn!("search_hybrid timed out");
                            None
                        }
                    }
                }
            })
            .collect();

        let all_results = futures::future::join_all(futs).await;

        // Collect all results, keeping max score for duplicates
        let mut id_to_result: HashMap<String, MemorySearchResult> = HashMap::new();

        for result in all_results.into_iter().flatten() {
            for r in result {
                let id = r.id.clone().unwrap_or_default();
                if let Some(existing) = id_to_result.get(&id) {
                    // Keep the one with higher score
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

    /// Update the content (and embedding) of an existing memory by ID.
    ///
    /// The `id` can be a prefix (≥8 chars) — it will be matched with `LIKE`.
    /// Returns the full ID of the updated memory, or an error if not found.
    /// All update fields are optional - only provided values will be updated.
    pub async fn update(
        &self,
        id: &str,
        new_content: Option<&str>,
        new_embedding: Option<Vec<f32>>,
        tags: Option<&[Tag]>,
        chat_key: Option<Option<String>>, // None = don't change, Some(None) = set to NULL/global, Some(Some(ck)) = set to specific chat_key
        expires_at: Option<Option<i64>>, // None = don't change, Some(None) = remove expiry, Some(Some(v)) = set expiry
    ) -> Result<String> {
        let id = id.to_string();
        let id_for_err = id.clone();
        let content = new_content.map(|s| s.to_string());
        let tags = tags.map(|t| t.to_vec());
        let chat_key_inner = chat_key.map(|inner| inner.map(|s| s.to_string()));

        self.conn
            .call(move |conn| {
                // Resolve prefix to full ID.
                let full_id: String = conn
                    .query_row(
                        "SELECT id FROM memories WHERE id LIKE ?1 || '%' LIMIT 1",
                        rusqlite::params![id],
                        |row| row.get(0),
                    )
                    .map_err(|_| rusqlite::Error::QueryReturnedNoRows)?;

                let tx = conn.transaction()?;

                // Build dynamic update query
                let mut updates: Vec<&str> = vec![];
                let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![];

                if let Some(ref c) = content {
                    updates.push("content = ?");
                    params.push(Box::new(c.clone()));
                }

                if let Some(ref ck) = chat_key_inner {
                    updates.push("chat_key = ?");
                    params.push(Box::new(ck.clone()));
                }

                if let Some(exp) = expires_at {
                    updates.push("expires_at = ?");
                    params.push(Box::new(exp));
                }

                if !updates.is_empty() {
                    params.push(Box::new(full_id.clone()));
                    let sql = format!("UPDATE memories SET {} WHERE id = ?", updates.join(", "));
                    let params_ref: Vec<&dyn rusqlite::ToSql> =
                        params.iter().map(|p| p.as_ref()).collect();
                    tx.execute(sql.as_str(), params_ref.as_slice())?;
                }

                // Update embedding if provided
                if let Some(emb) = new_embedding {
                    let bytes = embedding_to_bytes(&emb);
                    tx.execute(
                        "UPDATE memories_vec SET embedding = ?1 WHERE id = ?2",
                        rusqlite::params![bytes, full_id],
                    )?;
                }

                // Update tags if provided
                if let Some(ref t) = tags {
                    tx.execute("DELETE FROM memory_tags WHERE memory_id = ?1", [&full_id])?;
                    for tag in t {
                        tx.execute(
                            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id)
                             SELECT ?1, id FROM tags WHERE name = ?2",
                            rusqlite::params![full_id, tag.to_string()],
                        )?;
                    }
                }

                tx.commit()?;
                Ok(full_id)
            })
            .await
            .map_err(|e| {
                if matches!(&e, tokio_rusqlite::Error::Error(re) if *re == rusqlite::Error::QueryReturnedNoRows) {
                    FlashmemError::Memory(format!("No memory found with ID prefix '{}'", id_for_err))
                } else {
                    e.into()
                }
            })
    }

    /// Delete a memory by ID (or prefix) from both the vector index and memories table.
    ///
    /// The `id` can be a prefix (≥8 chars) — it will be matched with `LIKE`.
    /// Returns an error if no memory matches.
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
            .map_err(|e| {
                if matches!(&e, tokio_rusqlite::Error::Error(re) if *re == rusqlite::Error::QueryReturnedNoRows) {
                    FlashmemError::Memory(format!("No memory found with ID prefix '{}'", id_for_err))
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
                let expired_ids: Vec<String> = conn
                    .prepare(
                        "SELECT id FROM memories
                         WHERE expires_at IS NOT NULL AND expires_at <= ?1",
                    )?
                    .query_map(rusqlite::params![now], |row: &rusqlite::Row| row.get(0))?
                    .filter_map(|r| r.ok())
                    .collect();

                let count = expired_ids.len();
                if count > 0 {
                    let tx = conn.transaction()?;
                    for id in &expired_ids {
                        tx.execute(
                            "DELETE FROM memories_vec WHERE id = ?1",
                            rusqlite::params![id],
                        )?;
                        tx.execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])?;
                    }
                    tx.commit()?;
                }

                debug!(count, "deleted expired memories");
                Ok(count)
            })
            .await
            .map_err(Into::into)
    }

    /// Find memories similar to the given embedding above a threshold.
    ///
    /// Returns (id, score) pairs sorted by score descending.
    pub async fn find_similar(
        &self,
        query_embedding: Vec<f32>,
        chat_key: Option<&str>,
        scope: Option<Scope>,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let results = self
            .search_filtered(query_embedding, limit * 3, chat_key, scope)
            .await?;

        let filtered: Vec<(String, f32)> = results
            .into_iter()
            .filter(|r| r.score >= threshold)
            .take(limit)
            .map(|r| (r.id.unwrap_or_default(), r.score))
            .collect();

        Ok(filtered)
    }

    /// Search for tool-related memories using hybrid search filtered by Tag::Tool.
    ///
    /// Optionally filters by a specific `tool_name`. Uses hybrid search (vector + FTS5)
    /// for semantic matching within the tool-tagged memory subset.
    pub async fn search_for_tool(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        tool_name: Option<&str>,
        limit: usize,
        chat_key: Option<&str>,
    ) -> Result<Vec<MemorySearchResult>> {
        let over_limit = limit * 3;
        let now = Utc::now().timestamp();
        let query_text = query_text.to_string();
        let chat_key = chat_key.map(|s| s.to_string());
        let tool_name = tool_name.map(|s| s.to_string());

        self.conn
            .call(move |conn| {
                let bytes = embedding_to_bytes(&query_embedding);

                // 1. Vector search IDs filtered by tag=tool
                let vec_ids: Vec<String> = {
                    let mut stmt = conn.prepare(
                        "SELECT v.id FROM memories_vec v
                         JOIN memories m ON m.id = v.id
                         JOIN memory_tags mt ON mt.memory_id = m.id
                         JOIN tags t ON t.id = mt.tag_id
                         WHERE v.embedding MATCH ?1 AND k = ?2
                           AND t.name = 'tool'
                           AND (m.expires_at IS NULL OR m.expires_at > ?3)
                         ORDER BY v.distance",
                    )?;

                    stmt.query_map(rusqlite::params![bytes, over_limit, now], |row| {
                        row.get::<_, String>(0)
                    })?
                    .filter_map(|r| r.ok())
                    .collect()
                };

                // 2. FTS search IDs filtered by tag=tool
                let fts_ids: Vec<String> = {
                    let clean_query: String = query_text
                        .chars()
                        .map(|c| {
                            if c.is_alphanumeric() || c == ' ' {
                                c
                            } else {
                                ' '
                            }
                        })
                        .collect();
                    let clean_query = clean_query.trim().to_string();

                    if clean_query.is_empty() {
                        Vec::new()
                    } else {
                        let mut stmt = conn.prepare(
                            "SELECT m.id FROM memories m
                             JOIN memories_fts f ON f.rowid = m.rowid
                             JOIN memory_tags mt ON mt.memory_id = m.id
                             JOIN tags t ON t.id = mt.tag_id
                             WHERE memories_fts MATCH ?1
                               AND t.name = 'tool'
                               AND (m.expires_at IS NULL OR m.expires_at > ?2)
                             ORDER BY f.rank
                             LIMIT ?3",
                        )?;

                        stmt.query_map(rusqlite::params![clean_query, now, over_limit], |row| {
                            row.get::<_, String>(0)
                        })?
                        .filter_map(|r| r.ok())
                        .collect()
                    }
                };

                // 3. RRF fusion
                let vec_refs: Vec<&str> = vec_ids.iter().map(|s| s.as_str()).collect();
                let fts_refs: Vec<&str> = fts_ids.iter().map(|s| s.as_str()).collect();
                let mut fused = search::rrf_fuse(&[vec_refs, fts_refs], 10.0);
                search::normalize_scores(&mut fused);

                // 4. Filter by scope and tool_name
                fused.retain(|(id, _)| {
                    let Ok((mem_ck, mem_tn)): rusqlite::Result<(Option<String>, Option<String>)> =
                        conn.query_row(
                            "SELECT chat_key, tool_name FROM memories WHERE id = ?1",
                            [id.as_str()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                    else {
                        return false;
                    };

                    if !matches_scope(&mem_ck, chat_key.as_deref(), None) {
                        return false;
                    }
                    if let Some(ref tn) = tool_name
                        && mem_tn.as_deref() != Some(tn.as_str())
                    {
                        return false;
                    }
                    true
                });

                fused.truncate(limit);

                // 5. Fetch full records
                let mut results = Vec::with_capacity(fused.len());
                for (id, score) in &fused {
                    let mut stmt = conn.prepare(
                        "SELECT id, content, source, chat_key, created_at, expires_at
                         FROM memories WHERE id = ?1",
                    )?;

                    if let Ok(result) = stmt.query_row(rusqlite::params![id], |row| {
                        row_to_search_result(conn, row, *score)
                    }) {
                        results.push(result);
                    }
                }

                Ok(results)
            })
            .await
            .map_err(Into::into)
    }

    /// List all memories with a specific tag.
    pub async fn list_tagged(
        &self,
        tag: &str,
        chat_key: Option<&str>,
    ) -> Result<Vec<MemoryRecord>> {
        let tag = tag.to_string();
        let chat_key = chat_key.map(|s| s.to_string());

        self.conn
            .call(move |conn| {
                let now = Utc::now().timestamp();

                let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
                    if let Some(ref ck) = chat_key {
                        (
                            "SELECT m.id, m.content, m.source, m.chat_key, m.identity,
                                    m.created_at, m.expires_at
                             FROM memories m
                             JOIN memory_tags mt ON mt.memory_id = m.id
                             JOIN tags t ON t.id = mt.tag_id
                             WHERE t.name = ?1
                               AND (m.expires_at IS NULL OR m.expires_at > ?2)
                               AND m.chat_key = ?3
                             ORDER BY m.created_at DESC"
                                .to_string(),
                            vec![
                                Box::new(tag) as Box<dyn rusqlite::types::ToSql>,
                                Box::new(now),
                                Box::new(ck.clone()),
                            ],
                        )
                    } else {
                        (
                            "SELECT m.id, m.content, m.source, m.chat_key, m.identity,
                                    m.created_at, m.expires_at
                             FROM memories m
                             JOIN memory_tags mt ON mt.memory_id = m.id
                             JOIN tags t ON t.id = mt.tag_id
                             WHERE t.name = ?1
                               AND (m.expires_at IS NULL OR m.expires_at > ?2)
                             ORDER BY m.created_at DESC"
                                .to_string(),
                            vec![
                                Box::new(tag) as Box<dyn rusqlite::types::ToSql>,
                                Box::new(now),
                            ],
                        )
                    };

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

    /// List all memories with optional pagination and content filter.
    ///
    /// `cursor` is an epoch-seconds value — returns records created before the cursor.
    pub async fn list_all(
        &self,
        limit: usize,
        cursor: Option<i64>,
        filter: Option<&str>,
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

                let where_clause = conditions.join(" AND ");
                let sql = format!(
                    "SELECT m.id, m.content, m.source, m.chat_key, m.identity,
                            m.created_at, m.expires_at
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn test_db() -> DbStore {
        crate::test_util::register_sqlite_vec();
        let dir = tempdir().unwrap();
        DbStore::connect(&dir.path().join("test.db"), 4)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_connect() {
        let _db = test_db().await;
    }

    #[tokio::test]
    async fn test_store_and_search() {
        let db = test_db().await;
        db.store(
            "User likes Rust",
            vec![1.0, 0.0, 0.0, 0.0],
            Source::Conversation,
            None,
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();

        let results = db.search(vec![0.9, 0.1, 0.0, 0.0], 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_delete() {
        let db = test_db().await;
        let id = db
            .store(
                "temp",
                vec![1.0, 0.0, 0.0, 0.0],
                Source::Manual,
                None,
                None,
                &[],
                None,
                None,
            )
            .await
            .unwrap();
        db.delete(&id).await.unwrap();
        let results = db.search(vec![1.0, 0.0, 0.0, 0.0], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_scoped_search() {
        let db = test_db().await;
        db.store(
            "chat mem",
            vec![1.0, 0.0, 0.0, 0.0],
            Source::Conversation,
            Some("tg:123"),
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();
        db.store(
            "global mem",
            vec![0.9, 0.1, 0.0, 0.0],
            Source::Conversation,
            None,
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();

        let chat = db
            .search_filtered(
                vec![1.0, 0.0, 0.0, 0.0],
                5,
                Some("tg:123"),
                Some(Scope::Local),
            )
            .await
            .unwrap();
        assert_eq!(chat.len(), 1);

        let global = db
            .search_filtered(vec![1.0, 0.0, 0.0, 0.0], 5, None, Some(Scope::Global))
            .await
            .unwrap();
        assert_eq!(global.len(), 1);
    }

    #[tokio::test]
    async fn test_multi_context() {
        let db = test_db().await;
        db.store(
            "chat mem",
            vec![1.0, 0.0, 0.0, 0.0],
            Source::Conversation,
            Some("tg:123"),
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();
        db.store(
            "global mem",
            vec![0.9, 0.1, 0.0, 0.0],
            Source::Conversation,
            None,
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();

        let results = db
            .search_multi_context(vec![1.0, 0.0, 0.0, 0.0], 10, Some("tg:123"))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_expired() {
        let db = test_db().await;
        db.store(
            "expired",
            vec![1.0, 0.0, 0.0, 0.0],
            Source::Manual,
            None,
            None,
            &[],
            Some(1),
            None,
        )
        .await
        .unwrap();
        db.store(
            "valid",
            vec![0.0, 1.0, 0.0, 0.0],
            Source::Manual,
            None,
            None,
            &[],
            None,
            None,
        )
        .await
        .unwrap();

        let count = db.delete_expired().await.unwrap();
        assert_eq!(count, 1);

        let results = db.search(vec![1.0, 0.0, 0.0, 0.0], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("valid"));
    }
}
