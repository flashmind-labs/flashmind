//! Client-side session metadata storage for the session picker.
//!
//! Stores session metadata (key, prompt, cwd, model, updated_at) in SQLite.
//! This replaces the previous connect_sessions.jsonl file.

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_rusqlite::Connection;

use crate::error::Result;

/// The mode a session was started in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionMode {
    Normal,
    Code,
}

impl SessionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Code => "code",
        }
    }

    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s {
            Some("code") => Self::Code,
            _ => Self::Normal,
        }
    }
}

/// A local session entry for the session picker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSession {
    pub key: String,
    pub prompt: String,
    pub cwd: Option<String>,
    pub model: Option<String>,
    pub updated_at: i64,
    pub title: Option<String>,
    pub mode: SessionMode,
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64
}

/// List all local sessions ordered by updated_at descending.
pub async fn list(conn: &Connection, cwd: Option<&str>) -> Result<Vec<LocalSession>> {
    let cwd_owned = cwd.map(|s| s.to_string());
    conn.call(move |conn| -> rusqlite::Result<Vec<LocalSession>> {
        let base = "SELECT key, prompt, cwd, model, updated_at, title, mode FROM local_sessions";
        let query = match &cwd_owned {
            Some(_) => format!("{base} WHERE cwd = ?1 ORDER BY updated_at DESC"),
            None => format!("{base} ORDER BY updated_at DESC"),
        };
        let mut stmt = conn.prepare(&query)?;

        let map_row = |row: &rusqlite::Row| {
            Ok(LocalSession {
                key: row.get(0)?,
                prompt: row.get(1)?,
                cwd: row.get(2)?,
                model: row.get(3)?,
                updated_at: row.get(4)?,
                title: row.get(5)?,
                mode: SessionMode::from_str_opt(row.get::<_, Option<String>>(6)?.as_deref()),
            })
        };

        match &cwd_owned {
            Some(cwd) => stmt.query_map([cwd.as_str()], map_row)?.collect(),
            None => stmt.query_map([], map_row)?.collect(),
        }
    })
    .await
    .map_err(Into::into)
}

/// Save (upsert) a local session entry. Updates updated_at to current time.
pub async fn save(
    conn: &Connection,
    key: &str,
    prompt: &str,
    cwd: Option<&str>,
    model: Option<&str>,
    mode: Option<SessionMode>,
) -> Result<()> {
    let k = key.to_string();
    let p = prompt.to_string();
    let cwd = cwd.map(String::from);
    let model = model.map(String::from);
    let mode_str = mode.map(|m| m.as_str().to_string());
    let now = now_epoch();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO local_sessions (key, prompt, cwd, model, updated_at, mode)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(key) DO UPDATE SET
                 prompt      = excluded.prompt,
                 cwd         = excluded.cwd,
                 model       = excluded.model,
                 updated_at  = excluded.updated_at,
                 mode        = COALESCE(excluded.mode, local_sessions.mode)",
            params![k, p, cwd, model, now, mode_str],
        )?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

/// Upsert a local session with a specific timestamp (used for merging from server).
pub async fn merge_session(
    conn: &Connection,
    key: &str,
    prompt: &str,
    cwd: Option<&str>,
    model: Option<&str>,
    last_accessed: i64,
) -> Result<()> {
    let k = key.to_string();
    let p = prompt.to_string();
    let cwd = cwd.map(String::from);
    let model = model.map(String::from);

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO local_sessions (key, prompt, cwd, model, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(key) DO UPDATE SET
                 prompt     = excluded.prompt,
                 cwd        = excluded.cwd,
                 model      = excluded.model,
                 updated_at = MAX(local_sessions.updated_at, excluded.updated_at)",
            params![k, p, cwd, model, last_accessed],
        )?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

/// Update only the model for an existing session. Does NOT touch `updated_at`,
/// so switching models via `/model` doesn't spuriously bump last-activity time.
pub async fn set_model(conn: &Connection, key: &str, model: Option<&str>) -> Result<()> {
    let k = key.to_string();
    let m = model.map(String::from);
    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "UPDATE local_sessions SET model = ?2 WHERE key = ?1",
            params![k, m],
        )?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

/// Update only the title for an existing session. Does NOT touch `updated_at`.
pub async fn set_title(conn: &Connection, key: &str, title: Option<&str>) -> Result<()> {
    let k = key.to_string();
    let t = title.map(String::from);
    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "UPDATE local_sessions SET title = ?2 WHERE key = ?1",
            params![k, t],
        )?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

/// Delete a local session by key.
pub async fn delete(conn: &Connection, key: &str) -> Result<()> {
    let k = key.to_string();
    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute("DELETE FROM local_sessions WHERE key = ?1", params![k])?;
        Ok(())
    })
    .await
    .map_err(Into::into)
}

/// Get a single local session by key.
pub async fn get(conn: &Connection, key: &str) -> Result<Option<LocalSession>> {
    let k = key.to_string();
    conn.call(move |conn| -> rusqlite::Result<Option<LocalSession>> {
        let mut stmt = conn.prepare(
            "SELECT key, prompt, cwd, model, updated_at, title, mode
             FROM local_sessions WHERE key = ?1",
        )?;

        let result = stmt
            .query_row(params![k], |row| {
                Ok(LocalSession {
                    key: row.get(0)?,
                    prompt: row.get(1)?,
                    cwd: row.get(2)?,
                    model: row.get(3)?,
                    updated_at: row.get(4)?,
                    title: row.get(5)?,
                    mode: SessionMode::from_str_opt(row.get::<_, Option<String>>(6)?.as_deref()),
                })
            })
            .optional()?;

        Ok(result)
    })
    .await
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn open_db() -> Connection {
        let conn = Connection::open_in_memory().await.unwrap();
        conn.call(|c| {
            c.execute_batch(
                "CREATE TABLE IF NOT EXISTS local_sessions (
                    key        TEXT PRIMARY KEY,
                    prompt     TEXT,
                    cwd        TEXT,
                    model      TEXT,
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    title      TEXT,
                    mode       TEXT
                );",
            )
        })
        .await
        .unwrap();
        conn
    }

    #[tokio::test]
    async fn test_set_title_persists() {
        let conn = open_db().await;

        save(&conn, "sess1", "hello world", None, None, None)
            .await
            .unwrap();

        // Title is None initially
        let s = get(&conn, "sess1").await.unwrap().unwrap();
        assert!(s.title.is_none());

        // Capture updated_at before to verify set_title doesn't mutate it
        let updated_at_before = get(&conn, "sess1").await.unwrap().unwrap().updated_at;

        // Set a title
        set_title(&conn, "sess1", Some("My Project")).await.unwrap();
        let s = get(&conn, "sess1").await.unwrap().unwrap();
        assert_eq!(s.title.as_deref(), Some("My Project"));

        let updated_at_after = get(&conn, "sess1").await.unwrap().unwrap().updated_at;
        assert_eq!(
            updated_at_before, updated_at_after,
            "set_title must not change updated_at"
        );

        // Clear the title
        set_title(&conn, "sess1", None).await.unwrap();
        let s = get(&conn, "sess1").await.unwrap().unwrap();
        assert!(s.title.is_none());
    }

    #[tokio::test]
    async fn test_list_includes_title() {
        let conn = open_db().await;

        save(&conn, "sess1", "hello", None, None, None)
            .await
            .unwrap();
        set_title(&conn, "sess1", Some("Work Session"))
            .await
            .unwrap();

        let sessions = list(&conn, None).await.unwrap();
        let s = sessions.iter().find(|s| s.key == "sess1").unwrap();
        assert_eq!(s.title.as_deref(), Some("Work Session"));
    }
}
