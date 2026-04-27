//! Session and chat settings persistence via SQLite.
//!
//! Sessions are stored as one row per conversation entry with structured columns.
//! Chat settings are stored as JSON blobs per `chat_key`.

use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};

use crate::error::{FlashmemError, Result};

// ---------------------------------------------------------------------------
// Session entry — the row type that crosses the crate boundary
// ---------------------------------------------------------------------------

/// A single session entry as stored in the database.
/// The main crate maps between `ConversationEntry` and this type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub entry_kind: String,
    pub content: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub metadata: Option<String>,
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// Append entries to a session without replacing existing ones.
/// Assigns turn_index starting after the current max for this chat_key.
pub async fn append(
    conn: &tokio_rusqlite::Connection,
    chat_key: &str,
    entries: &[SessionEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    let ck = chat_key.to_string();
    let entries = entries.to_vec();

    conn.call(move |conn| -> rusqlite::Result<()> {
        let max_turn: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(turn_index), -1) FROM sessions WHERE chat_key = ?1",
                [&ck],
                |row| row.get(0),
            )
            .unwrap_or(-1);

        let mut stmt = conn.prepare_cached(
            "INSERT INTO sessions (chat_key, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;

        for (i, entry) in entries.iter().enumerate() {
            stmt.execute(rusqlite::params![
                ck,
                entry.entry_kind,
                entry.content,
                entry.tool_calls,
                entry.tool_call_id,
                entry.tool_name,
                entry.metadata,
                max_turn + 1 + i as i64,
                entry.created_at,
            ])?;
        }

        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Append session failed: {e}")))
}

/// Save a full session — deletes existing entries for this chat_key, then inserts all new ones.
/// Use this after compaction to replace the entire conversation.
pub async fn save(
    conn: &tokio_rusqlite::Connection,
    chat_key: &str,
    entries: &[SessionEntry],
) -> Result<()> {
    let ck = chat_key.to_string();
    let entries = entries.to_vec();

    conn.call(move |conn| -> rusqlite::Result<()> {
        let tx = conn.transaction()?;

        tx.execute("DELETE FROM sessions WHERE chat_key = ?1", [&ck])?;

        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO sessions (chat_key, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;

            for (i, entry) in entries.iter().enumerate() {
                stmt.execute(rusqlite::params![
                    ck,
                    entry.entry_kind,
                    entry.content,
                    entry.tool_calls,
                    entry.tool_call_id,
                    entry.tool_name,
                    entry.metadata,
                    i as i64,
                    entry.created_at,
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Save session failed: {e}")))?;

    tracing::debug!(chat_key, "session saved");
    Ok(())
}

/// Load all session entries for a chat_key, ordered by turn_index.
pub async fn load(conn: &tokio_rusqlite::Connection, chat_key: &str) -> Result<Vec<SessionEntry>> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<Vec<SessionEntry>> {
        let mut stmt = conn.prepare(
            "SELECT entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, created_at
             FROM sessions WHERE chat_key = ?1 ORDER BY turn_index",
        )?;

        let entries = stmt
            .query_map([&ck], |row| {
                Ok(SessionEntry {
                    entry_kind: row.get(0)?,
                    content: row.get(1)?,
                    tool_calls: row.get(2)?,
                    tool_call_id: row.get(3)?,
                    tool_name: row.get(4)?,
                    metadata: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Load session failed: {e}")))
}

/// List all chat_keys that have saved sessions.
pub async fn list(conn: &tokio_rusqlite::Connection) -> Result<Vec<String>> {
    conn.call(|conn| -> rusqlite::Result<Vec<String>> {
        let mut stmt = conn.prepare("SELECT DISTINCT chat_key FROM sessions ORDER BY chat_key")?;
        let keys: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(keys)
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("List sessions failed: {e}")))
}

/// Copy all entries from one session to another.
pub async fn copy(conn: &tokio_rusqlite::Connection, from_key: &str, to_key: &str) -> Result<()> {
    let from = from_key.to_string();
    let to = to_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at)
             SELECT ?1, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at
             FROM sessions WHERE chat_key = ?2 ORDER BY turn_index",
            rusqlite::params![to, from],
        )?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Copy session failed: {e}")))?;

    tracing::debug!(from_key, to_key, "session copied");
    Ok(())
}

/// Delete a session.
pub async fn delete(conn: &tokio_rusqlite::Connection, chat_key: &str) -> Result<()> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute("DELETE FROM sessions WHERE chat_key = ?1", [&ck])?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Delete session failed: {e}")))?;

    tracing::debug!(chat_key, "session deleted");
    Ok(())
}

/// Count entries for a chat_key.
pub async fn count(conn: &tokio_rusqlite::Connection, chat_key: &str) -> Result<usize> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<usize> {
        conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE chat_key = ?1",
            [&ck],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c as usize)
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Count session failed: {e}")))
}

/// Find sessions where the last entry is a user message (interrupted conversations).
/// Returns `(chat_key, content)` pairs.
pub async fn find_interrupted(conn: &tokio_rusqlite::Connection) -> Result<Vec<(String, String)>> {
    conn.call(|conn| -> rusqlite::Result<Vec<(String, String)>> {
        let mut stmt = conn.prepare(
            "SELECT s.chat_key, s.content
                 FROM sessions s
                 INNER JOIN (
                     SELECT chat_key, MAX(turn_index) as max_turn
                     FROM sessions
                     GROUP BY chat_key
                 ) latest ON s.chat_key = latest.chat_key AND s.turn_index = latest.max_turn
                 WHERE s.entry_kind = 'user' AND s.content IS NOT NULL",
        )?;

        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(rows)
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Find interrupted failed: {e}")))
}

/// Remove the last entry from a session.
pub async fn pop_last(conn: &tokio_rusqlite::Connection, chat_key: &str) -> Result<()> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "DELETE FROM sessions WHERE chat_key = ?1 AND turn_index = (
                 SELECT MAX(turn_index) FROM sessions WHERE chat_key = ?1
             )",
            [&ck],
        )?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Pop last failed: {e}")))
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Summary of sessions: chat_key + entry count.
pub async fn list_with_counts(conn: &tokio_rusqlite::Connection) -> Result<Vec<(String, usize)>> {
    conn.call(|conn| -> rusqlite::Result<Vec<(String, usize)>> {
        let mut stmt = conn.prepare(
            "SELECT chat_key, COUNT(*) FROM sessions GROUP BY chat_key ORDER BY chat_key",
        )?;
        Ok(stmt
            .query_map([], |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as usize)))?
            .filter_map(|r| r.ok())
            .collect())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("List with counts failed: {e}")))
}

/// Search session content with LIKE matching.
pub async fn search(
    conn: &tokio_rusqlite::Connection,
    query: &str,
    chat_key: Option<&str>,
    entry_kind: Option<&str>,
    limit: usize,
    offset: usize,
) -> Result<Vec<SessionSearchResult>> {
    let query = format!("%{}%", query);
    let chat_key = chat_key.map(|s| s.to_string());
    let entry_kind = entry_kind.map(|s| s.to_string());

    conn.call(move |conn| -> rusqlite::Result<Vec<SessionSearchResult>> {
        let mut sql = String::from(
            "SELECT chat_key, entry_kind, content, created_at FROM sessions WHERE content LIKE ?1",
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(query)];

        if let Some(ref ck) = chat_key {
            sql.push_str(&format!(" AND chat_key = ?{}", params.len() + 1));
            params.push(Box::new(ck.clone()));
        }
        if let Some(ref ek) = entry_kind {
            sql.push_str(&format!(" AND entry_kind = ?{}", params.len() + 1));
            params.push(Box::new(ek.clone()));
        }

        sql.push_str(&format!(
            " ORDER BY created_at DESC LIMIT ?{} OFFSET ?{}",
            params.len() + 1,
            params.len() + 2
        ));
        params.push(Box::new(limit as i64));
        params.push(Box::new(offset as i64));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        Ok(stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(SessionSearchResult {
                    chat_key: row.get(0)?,
                    entry_kind: row.get(1)?,
                    content: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Search sessions failed: {e}")))
}

/// Get entries for a specific session with optional kind filter.
pub async fn get_entries(
    conn: &tokio_rusqlite::Connection,
    chat_key: &str,
    entry_kind: Option<&str>,
    limit: usize,
    offset: usize,
) -> Result<Vec<SessionEntry>> {
    let ck = chat_key.to_string();
    let entry_kind = entry_kind.map(|s| s.to_string());

    conn.call(move |conn| -> rusqlite::Result<Vec<SessionEntry>> {
        let mut sql = String::from(
            "SELECT entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, created_at
             FROM sessions WHERE chat_key = ?1",
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(ck)];

        if let Some(ref ek) = entry_kind {
            sql.push_str(" AND entry_kind = ?2");
            params.push(Box::new(ek.clone()));
        }

        sql.push_str(&format!(
            " ORDER BY turn_index LIMIT ?{} OFFSET ?{}",
            params.len() + 1,
            params.len() + 2
        ));
        params.push(Box::new(limit as i64));
        params.push(Box::new(offset as i64));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        Ok(stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(SessionEntry {
                    entry_kind: row.get(0)?,
                    content: row.get(1)?,
                    tool_calls: row.get(2)?,
                    tool_call_id: row.get(3)?,
                    tool_name: row.get(4)?,
                    metadata: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Get entries failed: {e}")))
}

/// Entry count per entry_kind, optionally scoped to a chat_key.
pub async fn stats(
    conn: &tokio_rusqlite::Connection,
    chat_key: Option<&str>,
) -> Result<Vec<(String, usize)>> {
    let chat_key = chat_key.map(|s| s.to_string());

    conn.call(move |conn| -> rusqlite::Result<Vec<(String, usize)>> {
        let (sql, params): (_, Vec<Box<dyn rusqlite::types::ToSql>>) = match chat_key {
            Some(ref ck) => (
                "SELECT entry_kind, COUNT(*) FROM sessions WHERE chat_key = ?1 GROUP BY entry_kind ORDER BY COUNT(*) DESC",
                vec![Box::new(ck.clone()) as _],
            ),
            None => (
                "SELECT entry_kind, COUNT(*) FROM sessions GROUP BY entry_kind ORDER BY COUNT(*) DESC",
                vec![],
            ),
        };

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(sql)?;
        Ok(stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get(0)?, row.get::<_, i64>(1)? as usize))
            })?
            .filter_map(|r| r.ok())
            .collect())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Stats failed: {e}")))
}

/// A search result from session content.
#[derive(Debug, Clone)]
pub struct SessionSearchResult {
    pub chat_key: String,
    pub entry_kind: String,
    pub content: Option<String>,
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Chat Settings
// ---------------------------------------------------------------------------

/// Save chat settings as a JSON blob.
pub async fn save_settings(
    conn: &tokio_rusqlite::Connection,
    chat_key: &str,
    json: &str,
) -> Result<()> {
    let ck = chat_key.to_string();
    let json = json.to_string();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR REPLACE INTO chat_settings (chat_key, settings) VALUES (?1, ?2)",
            rusqlite::params![ck, json],
        )?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Save settings failed: {e}")))
}

/// Load chat settings JSON. Returns `None` if not found.
pub async fn load_settings(
    conn: &tokio_rusqlite::Connection,
    chat_key: &str,
) -> Result<Option<String>> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<Option<String>> {
        conn.query_row(
            "SELECT settings FROM chat_settings WHERE chat_key = ?1",
            [&ck],
            |row| row.get::<_, String>(0),
        )
        .optional()
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Load settings failed: {e}")))
}

/// Delete chat settings.
pub async fn delete_settings(conn: &tokio_rusqlite::Connection, chat_key: &str) -> Result<()> {
    let ck = chat_key.to_string();

    conn.call(move |conn| -> rusqlite::Result<()> {
        conn.execute("DELETE FROM chat_settings WHERE chat_key = ?1", [&ck])?;
        Ok(())
    })
    .await
    .map_err(|e| FlashmemError::Memory(format!("Delete settings failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_conn() -> tokio_rusqlite::Connection {
        crate::test_util::register_sqlite_vec();
        let conn = tokio_rusqlite::Connection::open_in_memory().await.unwrap();
        conn.call(|conn| -> rusqlite::Result<()> {
            crate::schema::init_schema(conn, 4)?;
            Ok(())
        })
        .await
        .unwrap();
        conn
    }

    fn user_entry(content: &str) -> SessionEntry {
        SessionEntry {
            entry_kind: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    fn assistant_entry(content: &str) -> SessionEntry {
        SessionEntry {
            entry_kind: "assistant".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    #[tokio::test]
    async fn test_save_and_load() {
        let conn = test_conn().await;
        let entries = vec![user_entry("hello"), assistant_entry("hi there")];

        save(&conn, "tg:123", &entries).await.unwrap();

        let loaded = load(&conn, "tg:123").await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].entry_kind, "user");
        assert_eq!(loaded[0].content.as_deref(), Some("hello"));
        assert_eq!(loaded[1].entry_kind, "assistant");
    }

    #[tokio::test]
    async fn test_load_missing() {
        let conn = test_conn().await;
        assert!(load(&conn, "nope").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_save_replaces() {
        let conn = test_conn().await;
        save(&conn, "tg:1", &[user_entry("v1")]).await.unwrap();
        save(&conn, "tg:1", &[user_entry("v2")]).await.unwrap();

        let loaded = load(&conn, "tg:1").await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content.as_deref(), Some("v2"));
    }

    #[tokio::test]
    async fn test_list() {
        let conn = test_conn().await;
        save(&conn, "slack:gen", &[user_entry("a")]).await.unwrap();
        save(&conn, "tg:1", &[user_entry("b")]).await.unwrap();

        let keys = list(&conn).await.unwrap();
        assert_eq!(keys, vec!["slack:gen", "tg:1"]);
    }

    #[tokio::test]
    async fn test_delete() {
        let conn = test_conn().await;
        save(&conn, "tg:1", &[user_entry("x")]).await.unwrap();
        delete(&conn, "tg:1").await.unwrap();
        assert!(load(&conn, "tg:1").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_find_interrupted() {
        let conn = test_conn().await;
        save(&conn, "tg:1", &[user_entry("hello")]).await.unwrap();
        save(&conn, "tg:2", &[user_entry("hi"), assistant_entry("hey")])
            .await
            .unwrap();

        let interrupted = find_interrupted(&conn).await.unwrap();
        assert_eq!(interrupted.len(), 1);
        assert_eq!(interrupted[0].0, "tg:1");
        assert_eq!(interrupted[0].1, "hello");
    }

    #[tokio::test]
    async fn test_pop_last() {
        let conn = test_conn().await;
        save(
            &conn,
            "tg:1",
            &[assistant_entry("ready"), user_entry("hello")],
        )
        .await
        .unwrap();

        pop_last(&conn, "tg:1").await.unwrap();

        let loaded = load(&conn, "tg:1").await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].entry_kind, "assistant");
    }

    #[tokio::test]
    async fn test_settings_roundtrip() {
        let conn = test_conn().await;
        save_settings(&conn, "tg:1", r#"{"model":"gpt-4"}"#)
            .await
            .unwrap();

        let data = load_settings(&conn, "tg:1").await.unwrap();
        assert_eq!(data.as_deref(), Some(r#"{"model":"gpt-4"}"#));

        delete_settings(&conn, "tg:1").await.unwrap();
        assert!(load_settings(&conn, "tg:1").await.unwrap().is_none());
    }
}
