//! Server-side session metadata for cross-device sync.
//!
//! Stores session metadata (prompt, model, cwd) in SQLite.
//! Display data (JSONL) is stored as files on the filesystem,
//! not in the database.

use rusqlite::{Connection, Result, params};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Session metadata returned by [`list_sessions`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSessionMeta {
    pub key: String,
    pub last_prompt: Option<String>,
    pub model: Option<String>,
    pub cwd: Option<String>,
    pub updated_at: i64,
}

/// Resolve a username to its user_id.
fn user_id(conn: &Connection, username: &str) -> Result<i64> {
    conn.query_row(
        "SELECT id FROM users WHERE username = ?1",
        params![username],
        |row| row.get(0),
    )
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64
}

/// Upsert session metadata.
pub fn push_session(
    conn: &Connection,
    username: &str,
    session_key: &str,
    last_prompt: Option<&str>,
    model: Option<&str>,
    cwd: Option<&str>,
    updated_at: Option<i64>,
) -> Result<()> {
    let uid = user_id(conn, username)?;
    let now = updated_at.unwrap_or_else(now_epoch);

    conn.execute(
        "INSERT INTO user_sessions (user_id, session_key, last_prompt, model, cwd, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(user_id, session_key) DO UPDATE SET
             last_prompt  = excluded.last_prompt,
             model        = excluded.model,
             cwd          = excluded.cwd,
             updated_at   = excluded.updated_at",
        params![uid, session_key, last_prompt, model, cwd, now],
    )?;

    Ok(())
}

/// List all sessions for a user, ordered by most recently updated first.
pub fn list_sessions(conn: &Connection, username: &str) -> Result<Vec<UserSessionMeta>> {
    let uid = match user_id(conn, username) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };

    let mut stmt = conn.prepare(
        "SELECT session_key, last_prompt, model, cwd, updated_at
         FROM user_sessions
         WHERE user_id = ?1
         ORDER BY updated_at DESC",
    )?;

    let rows = stmt
        .query_map(params![uid], |row| {
            Ok(UserSessionMeta {
                key: row.get(0)?,
                last_prompt: row.get(1)?,
                model: row.get(2)?,
                cwd: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;

    Ok(rows)
}

/// Look up a single session's `updated_at`. Returns `None` if either the
/// user or the (user, session_key) row is missing — that's "no entry in the
/// DB", not an error.
pub fn get_session_updated_at(
    conn: &Connection,
    username: &str,
    session_key: &str,
) -> Result<Option<i64>> {
    let uid = match user_id(conn, username) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
        Err(e) => return Err(e),
    };

    match conn.query_row(
        "SELECT updated_at FROM user_sessions WHERE user_id = ?1 AND session_key = ?2",
        params![uid, session_key],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(ts) => Ok(Some(ts)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Delete a session.
///
/// Returns `true` if the session existed and was deleted.
/// Caller is responsible for deleting the corresponding JSONL file.
pub fn delete_session(conn: &Connection, username: &str, session_key: &str) -> Result<bool> {
    let uid = match user_id(conn, username) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
        Err(e) => return Err(e),
    };

    let rows = conn.execute(
        "DELETE FROM user_sessions WHERE user_id = ?1 AND session_key = ?2",
        params![uid, session_key],
    )?;

    Ok(rows > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::init_schema;
    use crate::test_util::register_sqlite_vec;
    use crate::users::create_user;

    fn open_db() -> Connection {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 384).unwrap();
        conn
    }

    #[test]
    fn test_push_and_list_sessions() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        push_session(
            &conn,
            "alice",
            "sess-1",
            Some("hello"),
            Some("gpt-4"),
            Some("/home/alice"),
            None,
        )
        .unwrap();

        let list = list_sessions(&conn, "alice").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].key, "sess-1");
        assert_eq!(list[0].last_prompt.as_deref(), Some("hello"));
        assert_eq!(list[0].model.as_deref(), Some("gpt-4"));
        assert_eq!(list[0].cwd.as_deref(), Some("/home/alice"));
    }

    #[test]
    fn test_push_upserts_metadata() {
        let conn = open_db();
        create_user(&conn, "bob", false).unwrap();

        push_session(
            &conn,
            "bob",
            "sess-1",
            Some("first"),
            Some("model-a"),
            None,
            None,
        )
        .unwrap();
        push_session(
            &conn,
            "bob",
            "sess-1",
            Some("second"),
            Some("model-b"),
            Some("/tmp"),
            None,
        )
        .unwrap();

        let list = list_sessions(&conn, "bob").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].last_prompt.as_deref(), Some("second"));
        assert_eq!(list[0].model.as_deref(), Some("model-b"));
        assert_eq!(list[0].cwd.as_deref(), Some("/tmp"));
    }

    #[test]
    fn test_delete_session() {
        let conn = open_db();
        create_user(&conn, "dave", false).unwrap();

        push_session(&conn, "dave", "sess-1", Some("hi"), None, None, None).unwrap();

        let deleted = delete_session(&conn, "dave", "sess-1").unwrap();
        assert!(deleted);

        let list = list_sessions(&conn, "dave").unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_sessions_unknown_user() {
        let conn = open_db();

        let list = list_sessions(&conn, "nobody").unwrap();
        assert!(list.is_empty());
    }
}
