//! Postgres backend for [`MemoryStore`](crate::MemoryStore): pgvector for
//! embeddings and a generated `tsvector` column for keyword search.
//!
//! ## Tables
//!
//! ### `memories`
//! Content, timestamps, the embedding and its `fts` keyword index.
//!
//! ### `memory_meta`
//! Arbitrary key-value metadata, as in the SQLite schema.

use std::cmp::Ordering;
use std::collections::HashMap;

use pgvector::Vector;
use sqlx::postgres::{PgConnection, PgPool, PgRow};
use sqlx::{Postgres, QueryBuilder, Row};

use crate::error::{FlashmemError, Result};
use crate::store::{MemoryRecord, MemorySearchResult};

/// Create the extension, tables and indexes if missing. Fails if `memories`
/// already holds embeddings of another dimension.
pub(crate) async fn init_schema(pool: &PgPool, embedding_dim: usize) -> Result<()> {
    let mut tx = pool.begin().await?;
    // Two processes creating the schema at once would race on CREATE ... IF NOT EXISTS.
    sqlx::query("SELECT pg_advisory_xact_lock(hashtext('flashmind_memory_schema'))")
        .execute(&mut *tx)
        .await?;

    let has_vector: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            .fetch_one(&mut *tx)
            .await?;
    if !has_vector {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&mut *tx)
            .await?;
    }

    let exists: bool = sqlx::query_scalar("SELECT to_regclass('memories') IS NOT NULL")
        .fetch_one(&mut *tx)
        .await?;
    if exists {
        let dim: Option<i32> = sqlx::query_scalar(
            "SELECT atttypmod FROM pg_attribute
             WHERE attrelid = to_regclass('memories') AND attname = 'embedding'
               AND NOT attisdropped",
        )
        .fetch_optional(&mut *tx)
        .await?;
        match dim {
            Some(dim) if usize::try_from(dim).ok() == Some(embedding_dim) => {}
            Some(dim) => {
                return Err(FlashmemError::Config(format!(
                    "memories table stores {dim}-dimension embeddings but the embedder \
                     produces {embedding_dim}"
                )));
            }
            None => {
                return Err(FlashmemError::Config(
                    "a memories table exists without an embedding column".into(),
                ));
            }
        }
    }

    // No HNSW or IVFFlat index: those post-filter an approximate top k, so
    // metadata-filtered lookups would lose rows. Every vector scan is exact.
    sqlx::raw_sql(&format!(
        "CREATE TABLE IF NOT EXISTS memories (
            id          TEXT    PRIMARY KEY,
            content     TEXT    NOT NULL,
            created_at  BIGINT  NOT NULL,
            expires_at  BIGINT,
            embedding   vector({embedding_dim}) NOT NULL,
            fts         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
        );
        CREATE INDEX IF NOT EXISTS memories_created_at_idx ON memories (created_at);
        CREATE INDEX IF NOT EXISTS memories_expires_at_idx ON memories (expires_at)
            WHERE expires_at IS NOT NULL;
        CREATE INDEX IF NOT EXISTS memories_fts_idx ON memories USING GIN (fts);

        CREATE TABLE IF NOT EXISTS memory_meta (
            memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            key       TEXT NOT NULL,
            value     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS memory_meta_memory_id_idx ON memory_meta (memory_id);
        CREATE INDEX IF NOT EXISTS memory_meta_kv_idx ON memory_meta (key, value);"
    ))
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}

/// A memory to insert.
pub(crate) struct NewMemory {
    pub id: String,
    pub content: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub embedding: Vec<f32>,
    pub meta: Vec<(String, String)>,
}

pub(crate) async fn insert(pool: &PgPool, memory: NewMemory) -> Result<String> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memories (id, content, created_at, expires_at, embedding)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(&memory.id)
    .bind(&memory.content)
    .bind(memory.created_at)
    .bind(memory.expires_at)
    .bind(Vector::from(memory.embedding))
    .execute(&mut *tx)
    .await?;
    insert_meta(&mut tx, &memory.id, &memory.meta).await?;
    tx.commit().await?;
    Ok(memory.id)
}

async fn insert_meta(conn: &mut PgConnection, id: &str, meta: &[(String, String)]) -> Result<()> {
    if meta.is_empty() {
        return Ok(());
    }
    let (keys, values): (Vec<String>, Vec<String>) = meta.iter().cloned().unzip();
    sqlx::query(
        "INSERT INTO memory_meta (memory_id, key, value)
         SELECT $1, k, v FROM UNNEST($2::text[], $3::text[]) WITH ORDINALITY AS t(k, v, n)
         ORDER BY n",
    )
    .bind(id)
    .bind(keys)
    .bind(values)
    .execute(conn)
    .await?;
    Ok(())
}

/// Metadata for each of `ids`, in one query.
async fn fetch_meta(
    pool: &PgPool,
    ids: &[String],
) -> Result<HashMap<String, Vec<(String, String)>>> {
    let mut meta: HashMap<String, Vec<(String, String)>> = HashMap::new();
    if ids.is_empty() {
        return Ok(meta);
    }
    let rows =
        sqlx::query("SELECT memory_id, key, value FROM memory_meta WHERE memory_id = ANY($1)")
            .bind(ids)
            .fetch_all(pool)
            .await?;
    for row in rows {
        meta.entry(row.try_get("memory_id")?)
            .or_default()
            .push((row.try_get("key")?, row.try_get("value")?));
    }
    Ok(meta)
}

/// `AND` conditions requiring memory `m` to carry every metadata pair.
fn push_meta_conditions(qb: &mut QueryBuilder<'_, Postgres>, filters: &[(String, String)]) {
    for (key, value) in filters {
        qb.push(" AND m.id IN (SELECT memory_id FROM memory_meta WHERE key = ")
            .push_bind(key.clone())
            .push(" AND value = ")
            .push_bind(value.clone())
            .push(")");
    }
}

/// The full ID for `prefix`, with the same errors as the SQLite backend.
async fn resolve_memory_id(conn: &mut PgConnection, prefix: &str) -> Result<String> {
    if prefix.len() < 8 {
        return Err(FlashmemError::Memory(format!(
            "memory ID prefix must be at least 8 characters: '{prefix}'"
        )));
    }
    // lower() on both sides mirrors SQLite's case-insensitive LIKE.
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM memories WHERE starts_with(lower(id), lower($1)) LIMIT 2",
    )
    .bind(prefix)
    .fetch_all(conn)
    .await?;
    let mut ids = ids.into_iter();
    match (ids.next(), ids.next()) {
        (Some(id), None) => Ok(id),
        (None, _) => Err(FlashmemError::Memory(format!(
            "No memory found with ID prefix '{prefix}'"
        ))),
        (Some(_), Some(_)) => Err(FlashmemError::Memory(format!(
            "memory ID prefix is ambiguous: '{prefix}'"
        ))),
    }
}

pub(crate) async fn delete(pool: &PgPool, prefix: &str) -> Result<()> {
    let mut tx = pool.begin().await?;
    let full_id = resolve_memory_id(&mut tx, prefix).await?;
    sqlx::query("DELETE FROM memories WHERE id = $1")
        .bind(&full_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

pub(crate) async fn delete_expired(pool: &PgPool, now: i64) -> Result<usize> {
    let done =
        sqlx::query("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= $1")
            .bind(now)
            .execute(pool)
            .await?;
    Ok(done.rows_affected() as usize)
}

pub(crate) async fn delete_matching(pool: &PgPool, filters: &[(String, String)]) -> Result<usize> {
    let mut qb = QueryBuilder::new("DELETE FROM memories m WHERE TRUE");
    push_meta_conditions(&mut qb, filters);
    let done = qb.build().execute(pool).await?;
    Ok(done.rows_affected() as usize)
}

fn row_to_record(
    row: &PgRow,
    meta: &mut HashMap<String, Vec<(String, String)>>,
) -> Result<MemoryRecord> {
    let id: String = row.try_get("id")?;
    Ok(MemoryRecord {
        meta: meta.remove(&id).unwrap_or_default(),
        id,
        content: row.try_get("content")?,
        created_at: row.try_get("created_at")?,
        expires_at: row.try_get("expires_at")?,
    })
}

/// Fetch metadata for `rows` in one query and turn them into records.
async fn rows_to_records(pool: &PgPool, rows: Vec<PgRow>) -> Result<Vec<MemoryRecord>> {
    let ids = rows
        .iter()
        .map(|row| row.try_get("id"))
        .collect::<sqlx::Result<Vec<String>>>()?;
    let mut meta = fetch_meta(pool, &ids).await?;
    rows.iter()
        .map(|row| row_to_record(row, &mut meta))
        .collect()
}

pub(crate) async fn get(pool: &PgPool, id: &str) -> Result<Option<MemoryRecord>> {
    let rows =
        sqlx::query("SELECT id, content, created_at, expires_at FROM memories WHERE id = $1")
            .bind(id)
            .fetch_all(pool)
            .await?;
    Ok(rows_to_records(pool, rows).await?.into_iter().next())
}

/// Changes for [`update`]. `None` leaves a field as it is.
pub(crate) struct MemoryUpdate {
    pub content: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub meta: Option<Vec<(String, String)>>,
    pub expires_at: Option<Option<i64>>,
}

pub(crate) async fn update(pool: &PgPool, prefix: &str, changes: MemoryUpdate) -> Result<String> {
    let mut tx = pool.begin().await?;
    let full_id = resolve_memory_id(&mut tx, prefix).await?;

    if let Some(content) = changes.content {
        sqlx::query("UPDATE memories SET content = $1 WHERE id = $2")
            .bind(content)
            .bind(&full_id)
            .execute(&mut *tx)
            .await?;
    }
    if let Some(expires_at) = changes.expires_at {
        sqlx::query("UPDATE memories SET expires_at = $1 WHERE id = $2")
            .bind(expires_at)
            .bind(&full_id)
            .execute(&mut *tx)
            .await?;
    }
    if let Some(embedding) = changes.embedding {
        sqlx::query("UPDATE memories SET embedding = $1 WHERE id = $2")
            .bind(Vector::from(embedding))
            .bind(&full_id)
            .execute(&mut *tx)
            .await?;
    }
    if let Some(meta) = changes.meta {
        sqlx::query("DELETE FROM memory_meta WHERE memory_id = $1")
            .bind(&full_id)
            .execute(&mut *tx)
            .await?;
        insert_meta(&mut tx, &full_id, &meta).await?;
    }

    tx.commit().await?;
    Ok(full_id)
}

pub(crate) async fn reindex(pool: &PgPool, id: &str, embedding: Vec<f32>) -> Result<()> {
    sqlx::query("UPDATE memories SET embedding = $1 WHERE id = $2")
        .bind(Vector::from(embedding))
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub(crate) async fn list_content_for_reindex(pool: &PgPool) -> Result<Vec<(String, String)>> {
    Ok(
        sqlx::query_as("SELECT id, content FROM memories ORDER BY created_at")
            .fetch_all(pool)
            .await?,
    )
}

/// Filters for [`list`].
pub(crate) struct ListQuery {
    pub filters: Vec<(String, String)>,
    pub contains: Option<String>,
    pub cursor: Option<i64>,
    pub after: Option<i64>,
    pub before: Option<i64>,
    pub limit: usize,
}

pub(crate) async fn list(pool: &PgPool, query: ListQuery) -> Result<Vec<MemoryRecord>> {
    let mut qb = QueryBuilder::new(
        "SELECT m.id, m.content, m.created_at, m.expires_at FROM memories m WHERE TRUE",
    );
    for upper in [query.cursor, query.before].into_iter().flatten() {
        qb.push(" AND m.created_at < ").push_bind(upper);
    }
    if let Some(text) = query.contains {
        // ILIKE stands in for SQLite's case-insensitive LIKE.
        qb.push(" AND m.content ILIKE ")
            .push_bind(format!("%{text}%"));
    }
    if let Some(after) = query.after {
        qb.push(" AND m.created_at >= ").push_bind(after);
    }
    push_meta_conditions(&mut qb, &query.filters);
    qb.push(" ORDER BY m.created_at DESC LIMIT ")
        .push_bind(query.limit as i64);

    let rows = qb.build().fetch_all(pool).await?;
    rows_to_records(pool, rows).await
}

/// A matching row before metadata is attached.
struct Candidate {
    id: String,
    content: String,
    created_at: i64,
    expires_at: Option<i64>,
    vec_score: f32,
    fts_score: f32,
}

/// The `k` unexpired memories closest to `embedding`, with the vector score
/// set. The scan is exact, so a filter cannot be
/// crowded out by other memories.
async fn nearest(
    pool: &PgPool,
    embedding: Vec<f32>,
    k: usize,
    now: i64,
    filters: &[(String, String)],
) -> Result<Vec<Candidate>> {
    let mut qb = QueryBuilder::new("SELECT m.id, (m.embedding <-> ");
    qb.push_bind(Vector::from(embedding))
        .push(")::float8 AS distance, m.content, m.created_at, m.expires_at FROM memories m")
        .push(" WHERE (m.expires_at IS NULL OR m.expires_at > ")
        .push_bind(now)
        .push(")");
    push_meta_conditions(&mut qb, filters);
    qb.push(" ORDER BY distance LIMIT ").push_bind(k as i64);

    qb.build()
        .fetch_all(pool)
        .await?
        .iter()
        .map(|row| {
            let distance: f64 = row.try_get("distance")?;
            Ok(Candidate {
                id: row.try_get("id")?,
                content: row.try_get("content")?,
                created_at: row.try_get("created_at")?,
                expires_at: row.try_get("expires_at")?,
                vec_score: 1.0 - (distance as f32 / 2.0),
                fts_score: 0.0,
            })
        })
        .collect()
}

/// Turns free text into a tsquery that matches any of its words. Words hold
/// only alphanumerics, so quoting them is enough to keep operators out.
fn ts_query(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| format!("'{word}'"))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Attach metadata to the best `limit` candidates, scoring each with `score`.
async fn finish(
    pool: &PgPool,
    candidates: Vec<Candidate>,
    limit: usize,
    score: impl Fn(&Candidate) -> f32,
) -> Result<Vec<MemorySearchResult>> {
    let mut scored: Vec<(Candidate, f32)> = candidates
        .into_iter()
        .map(|c| {
            let s = score(&c);
            (c, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(limit);

    let ids: Vec<String> = scored.iter().map(|(c, _)| c.id.clone()).collect();
    let mut meta = fetch_meta(pool, &ids).await?;
    Ok(scored
        .into_iter()
        .map(|(c, score)| MemorySearchResult {
            meta: meta.remove(&c.id).unwrap_or_default(),
            id: Some(c.id),
            content: c.content,
            created_at: c.created_at,
            expires_at: c.expires_at,
            score,
        })
        .collect())
}

/// Nearest memories by vector distance alone.
pub(crate) async fn search_vec(
    pool: &PgPool,
    embedding: Vec<f32>,
    limit: usize,
    now: i64,
    filters: &[(String, String)],
) -> Result<Vec<MemorySearchResult>> {
    let candidates = nearest(pool, embedding, limit, now, filters).await?;
    finish(pool, candidates, limit, |c| c.vec_score).await
}

/// Vector and keyword search combined as in the SQLite backend: keyword
/// rank is normalised so the best match scores 1, then adds 0.3 of it to the
/// vector score, capped at 1.
pub(crate) async fn search_hybrid(
    pool: &PgPool,
    embedding: Vec<f32>,
    text: &str,
    limit: usize,
    now: i64,
    filters: &[(String, String)],
) -> Result<Vec<MemorySearchResult>> {
    let over_limit = limit * 3;
    let mut candidates: HashMap<String, Candidate> =
        nearest(pool, embedding, over_limit, now, filters)
            .await?
            .into_iter()
            .map(|c| (c.id.clone(), c))
            .collect();

    let query = ts_query(text);
    if !query.is_empty() {
        let mut qb = QueryBuilder::new(
            "SELECT m.id, ts_rank(m.fts, q)::float8 AS rank, m.content, m.created_at, m.expires_at
             FROM memories m, to_tsquery('simple', ",
        );
        qb.push_bind(query)
            .push(") q WHERE m.fts @@ q AND (m.expires_at IS NULL OR m.expires_at > ")
            .push_bind(now)
            .push(")");
        push_meta_conditions(&mut qb, filters);
        qb.push(" ORDER BY rank DESC LIMIT ")
            .push_bind(over_limit as i64);
        let rows = qb.build().fetch_all(pool).await?;

        let mut ranked = Vec::with_capacity(rows.len());
        for row in &rows {
            let rank: f64 = row.try_get("rank")?;
            ranked.push((row, rank));
        }
        let max_rank = ranked.iter().map(|(_, r)| *r).fold(0.0f64, f64::max);
        for (row, rank) in ranked {
            let fts_score = if max_rank > 0.0 {
                (rank / max_rank) as f32
            } else {
                1.0
            };
            let id: String = row.try_get("id")?;
            if let Some(existing) = candidates.get_mut(&id) {
                existing.fts_score = fts_score;
            } else {
                candidates.insert(
                    id.clone(),
                    Candidate {
                        id,
                        content: row.try_get("content")?,
                        created_at: row.try_get("created_at")?,
                        expires_at: row.try_get("expires_at")?,
                        vec_score: 0.0,
                        fts_score,
                    },
                );
            }
        }
    }

    finish(pool, candidates.into_values().collect(), limit, |c| {
        (c.vec_score + c.fts_score * 0.3).min(1.0)
    })
    .await
}
