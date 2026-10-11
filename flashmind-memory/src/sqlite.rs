//! SQLite backend for [`MemoryStore`](crate::MemoryStore): sqlite-vec for
//! embeddings and FTS5 for keyword search. The schema is in
//! [`schema`](crate::schema).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use rusqlite::types::ToSql;
use rusqlite::{Row, params};
use tokio_rusqlite::Connection;

use crate::error::{FlashmemError, Result};
use crate::schema;
use crate::store::{ListQuery, MemoryRecord, MemorySearchResult, MemoryUpdate, NewMemory};

/// Open the database at `db_path`, creating parent directories and the
/// schema on first run.
pub(crate) async fn open(db_path: &Path, embedding_dim: usize) -> Result<Connection> {
    crate::register_sqlite_vec();

    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(db_path).await?;
    conn.call(move |conn| {
        schema::init_schema(conn, embedding_dim)?;
        #[cfg(feature = "session")]
        crate::session::schema::init_session_schema(conn)?;
        Ok(())
    })
    .await?;
    Ok(conn)
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn fetch_meta(conn: &rusqlite::Connection, memory_id: &str) -> Vec<(String, String)> {
    let Ok(mut stmt) = conn.prepare("SELECT key, value FROM memory_meta WHERE memory_id = ?1")
    else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map(params![memory_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return Vec::new();
    };

    rows.filter_map(|r| r.ok()).collect()
}

fn row_to_record(conn: &rusqlite::Connection, row: &Row) -> rusqlite::Result<MemoryRecord> {
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
fn meta_conditions(filters: &[(String, String)], params: &mut Vec<Box<dyn ToSql>>) -> Vec<String> {
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
    let map = |row: &Row| {
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
            .query_map(params![bytes, k as i64, now], map)?
            .collect();
    }
    let mut params: Vec<Box<dyn ToSql>> =
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
    let refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    stmt.query_map(refs.as_slice(), map)?.collect()
}

fn validate_memory_prefix(prefix: &str) -> rusqlite::Result<()> {
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
    let mut ids = stmt
        .query_map(params![prefix], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter();

    match (ids.next(), ids.next()) {
        (Some(id), None) => Ok(id),
        (None, _) => Err(rusqlite::Error::QueryReturnedNoRows),
        (Some(_), Some(_)) => Err(rusqlite::Error::InvalidParameterName(
            "memory ID prefix is ambiguous".into(),
        )),
    }
}

/// Turns the errors of [`resolve_memory_id`] into messages naming `prefix`.
fn prefix_error(e: tokio_rusqlite::Error, prefix: &str) -> FlashmemError {
    match e {
        tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows) => {
            FlashmemError::Memory(format!("No memory found with ID prefix '{prefix}'"))
        }
        tokio_rusqlite::Error::Error(rusqlite::Error::InvalidParameterName(msg)) => {
            FlashmemError::Memory(format!("{msg}: '{prefix}'"))
        }
        e => e.into(),
    }
}

pub(crate) async fn insert(conn: &Connection, memory: NewMemory) -> Result<String> {
    conn.call(move |conn| {
        let tx = conn.transaction()?;

        tx.execute(
            "INSERT INTO memories (id, content, created_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4)",
            params![
                memory.id,
                memory.content,
                memory.created_at,
                memory.expires_at
            ],
        )?;

        let bytes = embedding_to_bytes(&memory.embedding);
        tx.execute(
            "INSERT INTO memories_vec (id, embedding) VALUES (?1, ?2)",
            params![memory.id, bytes],
        )?;

        for (key, value) in &memory.meta {
            tx.execute(
                "INSERT INTO memory_meta (memory_id, key, value) VALUES (?1, ?2, ?3)",
                params![memory.id, key, value],
            )?;
        }

        tx.commit()?;
        Ok(memory.id)
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn delete(conn: &Connection, prefix: &str) -> Result<()> {
    let id = prefix.to_string();
    conn.call(move |conn| {
        let full_id = resolve_memory_id(conn, &id)?;

        let tx = conn.transaction()?;
        tx.execute("DELETE FROM memories_vec WHERE id = ?1", params![full_id])?;
        tx.execute("DELETE FROM memories WHERE id = ?1", params![full_id])?;
        tx.commit()?;
        Ok(())
    })
    .await
    .map_err(|e| prefix_error(e, prefix))
}

pub(crate) async fn delete_expired(conn: &Connection, now: i64) -> Result<usize> {
    conn.call(move |conn| {
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM memories_vec WHERE id IN \
                 (SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?1)",
            params![now],
        )?;
        let count = tx.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?1",
            params![now],
        )?;
        tx.commit()?;
        Ok(count)
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn delete_matching(
    conn: &Connection,
    filters: Vec<(String, String)>,
) -> Result<usize> {
    conn.call(move |conn| {
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();
        let conditions = meta_conditions(&filters, &mut params).join(" AND ");
        let refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let tx = conn.transaction()?;
        tx.execute(
            &format!(
                "DELETE FROM memories_vec WHERE id IN \
                 (SELECT m.id FROM memories m WHERE {conditions})"
            ),
            refs.as_slice(),
        )?;
        let count = tx.execute(
            &format!(
                "DELETE FROM memories WHERE id IN (SELECT m.id FROM memories m WHERE {conditions})"
            ),
            refs.as_slice(),
        )?;
        tx.commit()?;
        Ok(count)
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn get(conn: &Connection, id: &str) -> Result<Option<MemoryRecord>> {
    let id = id.to_string();
    conn.call(move |conn| {
        let mut stmt =
            conn.prepare("SELECT id, content, created_at, expires_at FROM memories WHERE id = ?1")?;
        let mut rows = stmt.query_map([&id], |row| row_to_record(conn, row))?;
        rows.next().transpose()
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn update(
    conn: &Connection,
    prefix: &str,
    changes: MemoryUpdate,
) -> Result<String> {
    let id = prefix.to_string();
    conn.call(move |conn| {
        let full_id = resolve_memory_id(conn, &id)?;

        let tx = conn.transaction()?;

        if let Some(content) = changes.content {
            tx.execute(
                "UPDATE memories SET content = ?1 WHERE id = ?2",
                params![content, full_id],
            )?;
        }

        if let Some(expires_at) = changes.expires_at {
            tx.execute(
                "UPDATE memories SET expires_at = ?1 WHERE id = ?2",
                params![expires_at, full_id],
            )?;
        }

        if let Some(embedding) = changes.embedding {
            let bytes = embedding_to_bytes(&embedding);
            tx.execute(
                "UPDATE memories_vec SET embedding = ?1 WHERE id = ?2",
                params![bytes, full_id],
            )?;
        }

        if let Some(meta) = changes.meta {
            tx.execute("DELETE FROM memory_meta WHERE memory_id = ?1", [&full_id])?;
            for (k, v) in meta {
                tx.execute(
                    "INSERT INTO memory_meta (memory_id, key, value) VALUES (?1, ?2, ?3)",
                    params![full_id, k, v],
                )?;
            }
        }

        tx.commit()?;
        Ok(full_id)
    })
    .await
    .map_err(|e| prefix_error(e, prefix))
}

/// Drop and rebuild the virtual tables. Returns how many memories need
/// re-embedding.
pub(crate) async fn repair(conn: &Connection, embedding_dim: usize) -> Result<usize> {
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

        let count: usize = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

        tracing::info!(count, "repair complete, FTS rebuilt, vec index empty");
        Ok(count)
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn reindex(conn: &Connection, id: &str, embedding: Vec<f32>) -> Result<()> {
    let id = id.to_string();
    conn.call(move |conn| {
        let bytes = embedding_to_bytes(&embedding);
        conn.execute(
            "INSERT OR REPLACE INTO memories_vec (id, embedding) VALUES (?1, ?2)",
            params![id, bytes],
        )?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

pub(crate) async fn list_content_for_reindex(conn: &Connection) -> Result<Vec<(String, String)>> {
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

pub(crate) async fn list(conn: &Connection, query: ListQuery) -> Result<Vec<MemoryRecord>> {
    conn.call(move |conn| {
        let mut conditions = vec!["1=1".to_string()];
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();

        for upper in [query.cursor, query.before].into_iter().flatten() {
            conditions.push(format!("m.created_at < ?{}", params.len() + 1));
            params.push(Box::new(upper));
        }

        if let Some(text) = query.contains {
            conditions.push(format!("m.content LIKE ?{}", params.len() + 1));
            params.push(Box::new(format!("%{text}%")));
        }

        if let Some(after) = query.after {
            conditions.push(format!("m.created_at >= ?{}", params.len() + 1));
            params.push(Box::new(after));
        }

        conditions.extend(meta_conditions(&query.filters, &mut params));

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT m.id, m.content, m.created_at, m.expires_at
                 FROM memories m
                 WHERE {where_clause}
                 ORDER BY m.created_at DESC
                 LIMIT ?{}",
            params.len() + 1
        );
        params.push(Box::new(query.limit as i64));

        let refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let records = stmt
            .query_map(refs.as_slice(), |row| row_to_record(conn, row))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(records)
    })
    .await
    .map_err(Into::into)
}

/// Nearest memories by vector distance alone.
pub(crate) async fn search_vec(
    conn: &Connection,
    embedding: Vec<f32>,
    limit: usize,
    now: i64,
    filters: Vec<(String, String)>,
) -> Result<Vec<MemorySearchResult>> {
    conn.call(move |conn| {
        let bytes = embedding_to_bytes(&embedding);
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
    .map_err(Into::into)
}

/// Vector and BM25 keyword search combined: the keyword score is normalised
/// so the best match scores 1, then 0.3 of it is added to the vector score,
/// capped at 1.
pub(crate) async fn search_hybrid(
    conn: &Connection,
    embedding: Vec<f32>,
    text: &str,
    limit: usize,
    now: i64,
    filters: &[(String, String)],
) -> Result<Vec<MemorySearchResult>> {
    let over_limit = limit * 3;
    let fts = fts_query(text);
    let filters = filters.to_vec();

    conn.call(move |conn| {
        let bytes = embedding_to_bytes(&embedding);

        type MemRow = (String, i64, Option<i64>);

        let mut vec_scores: HashMap<String, f32> = HashMap::new();
        let mut row_cache: HashMap<String, MemRow> = HashMap::new();
        for row in nearest(conn, &bytes, over_limit, now, &filters)? {
            vec_scores.insert(row.id.clone(), row.score);
            row_cache.insert(row.id, (row.content, row.created_at, row.expires_at));
        }

        let fts_scores: HashMap<String, f32> = if fts.is_empty() {
            HashMap::new()
        } else {
            let mut params: Vec<Box<dyn ToSql>> =
                vec![Box::new(fts), Box::new(now), Box::new(over_limit as i64)];
            let conditions: String = meta_conditions(&filters, &mut params)
                .into_iter()
                .map(|c| format!(" AND {c}"))
                .collect();
            let refs: Vec<&dyn ToSql> = params.iter().map(|p| p.as_ref()).collect();
            let mut stmt = conn.prepare(&format!(
                "SELECT m.id, bm25(memories_fts) as score FROM memories m
                 JOIN memories_fts f ON f.rowid = m.rowid
                 WHERE memories_fts MATCH ?1 AND (m.expires_at IS NULL OR m.expires_at > ?2){conditions}
                 ORDER BY score LIMIT ?3",
            ))?;
            let rows: Vec<(String, f64)> = stmt
                .query_map(refs.as_slice(), |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;

            let min_bm25 = rows.iter().map(|(_, s)| *s).fold(0.0f64, f64::min);
            let max_bm25 = rows.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
            let range = min_bm25 - max_bm25;
            rows.into_iter()
                .map(|(id, bm25)| {
                    let score = if range < 0.0 {
                        ((bm25 - max_bm25) / range) as f32
                    } else {
                        1.0
                    };
                    (id, score)
                })
                .collect()
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

            let (content, created_at, expires_at) = if let Some(cached) = row_cache.remove(id) {
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

            results.push(MemorySearchResult {
                id: Some(id.to_string()),
                content,
                created_at,
                expires_at,
                score,
                meta: fetch_meta(conn, id),
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    })
    .await
    .map_err(Into::into)
}
