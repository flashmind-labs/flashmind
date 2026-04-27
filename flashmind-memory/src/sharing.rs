//! Session sharing — token-based snapshot sharing between users.
//!
//! When a user shares a session, the current entries are copied to a
//! `shared:{token}` chat_key in the `sessions` table, and metadata is
//! recorded in `shared_sessions`. Pulling reads the snapshot and
//! increments the pull counter.

use rusqlite::{Connection, Result, params};
use serde::{Deserialize, Serialize};

use std::time::{SystemTime, UNIX_EPOCH};

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64
}

/// Resolve a username to its user_id.
fn user_id(conn: &Connection, username: &str) -> Result<i64> {
    conn.query_row(
        "SELECT id FROM users WHERE username = ?1",
        params![username],
        |row| row.get(0),
    )
}

/// Metadata about a shared session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareInfo {
    pub token: String,
    pub source_key: String,
    pub last_prompt: Option<String>,
    pub model: Option<String>,
    pub created_at: i64,
    pub expires_at: i64,
    pub max_pulls: Option<i64>,
    pub pull_count: i64,
}

/// Data returned when pulling a shared session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareData {
    pub token: String,
    pub shared_by: String,
    pub last_prompt: Option<String>,
    pub model: Option<String>,
    pub entries: Vec<crate::SessionEntry>,
}

/// Default TTL: 7 days in seconds.
const DEFAULT_TTL_SECS: i64 = 7 * 24 * 3600;

/// Create a share — copies session entries and records metadata.
///
/// `ttl_secs`: if `None`, defaults to 7 days.
/// `max_pulls`: if `None`, unlimited.
#[allow(clippy::too_many_arguments)]
pub fn create_share(
    conn: &Connection,
    username: &str,
    source_key: &str,
    token: &str,
    last_prompt: Option<&str>,
    model: Option<&str>,
    ttl_secs: Option<i64>,
    max_pulls: Option<i64>,
) -> Result<()> {
    let uid = user_id(conn, username)?;
    let now = now_epoch();
    let expires_at = now + ttl_secs.unwrap_or(DEFAULT_TTL_SECS);
    let shared_key = format!("shared:{}", token);

    let tx = conn.unchecked_transaction()?;

    // Insert metadata
    tx.execute(
        "INSERT INTO shared_sessions (token, user_id, source_key, last_prompt, model, created_at, expires_at, max_pulls, pull_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)",
        params![token, uid, source_key, last_prompt, model, now, expires_at, max_pulls],
    )?;

    // Copy session entries to shared:{token}
    tx.execute(
        "INSERT INTO sessions (chat_key, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at)
         SELECT ?1, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at
         FROM sessions WHERE chat_key = ?2 ORDER BY turn_index",
        params![shared_key, source_key],
    )?;

    tx.commit()?;
    Ok(())
}

/// Pull a shared session — validates token, increments pull count, returns entries + metadata.
///
/// Returns `None` if the token is invalid, expired, or exhausted.
pub fn pull_share(conn: &Connection, token: &str) -> Result<Option<ShareData>> {
    let now = now_epoch();

    // Look up the share metadata + owner username in one query
    let row = conn.query_row(
        "SELECT s.user_id, s.last_prompt, s.model, s.expires_at, s.max_pulls, s.pull_count, u.username
         FROM shared_sessions s
         JOIN users u ON u.id = s.user_id
         WHERE s.token = ?1",
        params![token],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,       // user_id
                row.get::<_, Option<String>>(1)?, // last_prompt
                row.get::<_, Option<String>>(2)?, // model
                row.get::<_, i64>(3)?,       // expires_at
                row.get::<_, Option<i64>>(4)?, // max_pulls
                row.get::<_, i64>(5)?,       // pull_count
                row.get::<_, String>(6)?,    // username
            ))
        },
    );

    let (_uid, last_prompt, model, expires_at, max_pulls, pull_count, username) = match row {
        Ok(r) => r,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
        Err(e) => return Err(e),
    };

    // Check expiry
    if now >= expires_at {
        return Ok(None);
    }

    // Check pull limit
    if let Some(max) = max_pulls
        && pull_count >= max
    {
        return Ok(None);
    }

    // Increment pull count
    conn.execute(
        "UPDATE shared_sessions SET pull_count = pull_count + 1 WHERE token = ?1",
        params![token],
    )?;

    // Load entries
    let shared_key = format!("shared:{}", token);
    let mut stmt = conn.prepare(
        "SELECT entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, created_at
         FROM sessions WHERE chat_key = ?1 ORDER BY turn_index",
    )?;

    let entries: Vec<crate::SessionEntry> = stmt
        .query_map(params![shared_key], |row| {
            Ok(crate::SessionEntry {
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

    Ok(Some(ShareData {
        token: token.to_string(),
        shared_by: username,
        last_prompt,
        model,
        entries,
    }))
}

/// List all shares created by a user.
pub fn list_shares(conn: &Connection, username: &str) -> Result<Vec<ShareInfo>> {
    let uid = match user_id(conn, username) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };

    let mut stmt = conn.prepare(
        "SELECT token, source_key, last_prompt, model, created_at, expires_at, max_pulls, pull_count
         FROM shared_sessions
         WHERE user_id = ?1
         ORDER BY created_at DESC, rowid DESC",
    )?;

    let rows = stmt
        .query_map(params![uid], |row| {
            Ok(ShareInfo {
                token: row.get(0)?,
                source_key: row.get(1)?,
                last_prompt: row.get(2)?,
                model: row.get(3)?,
                created_at: row.get(4)?,
                expires_at: row.get(5)?,
                max_pulls: row.get(6)?,
                pull_count: row.get(7)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Revoke a share — deletes metadata and snapshot entries.
///
/// Returns `true` if the share existed and belonged to the user.
pub fn revoke_share(conn: &Connection, token: &str, username: &str) -> Result<bool> {
    let uid = match user_id(conn, username) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
        Err(e) => return Err(e),
    };

    let shared_key = format!("shared:{}", token);

    let rows = conn.execute(
        "DELETE FROM shared_sessions WHERE token = ?1 AND user_id = ?2",
        params![token, uid],
    )?;

    if rows > 0 {
        conn.execute(
            "DELETE FROM sessions WHERE chat_key = ?1",
            params![shared_key],
        )?;
    }

    Ok(rows > 0)
}

/// Delete all expired shares and their session entries.
/// Returns the number of shares cleaned up.
pub fn cleanup_expired_shares(conn: &Connection) -> Result<usize> {
    let now = now_epoch();

    // Collect expired tokens first so we can clean up session entries
    let mut stmt = conn.prepare("SELECT token FROM shared_sessions WHERE expires_at <= ?1")?;

    let tokens: Vec<String> = stmt
        .query_map(params![now], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    if tokens.is_empty() {
        return Ok(0);
    }

    for token in &tokens {
        let shared_key = format!("shared:{}", token);
        conn.execute(
            "DELETE FROM sessions WHERE chat_key = ?1",
            params![shared_key],
        )?;
    }

    conn.execute(
        "DELETE FROM shared_sessions WHERE expires_at <= ?1",
        params![now],
    )?;

    Ok(tokens.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::init_schema;
    use crate::session::SessionEntry;
    use crate::test_util::register_sqlite_vec;
    use crate::users::create_user;

    fn open_db() -> Connection {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 4).unwrap();
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
            created_at: 1000,
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
            created_at: 1001,
        }
    }

    #[test]
    fn test_create_and_pull_share() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        // Save a session for alice
        let entries = [user_entry("hello"), assistant_entry("hi there")];
        // Use sync save via raw SQL (session module is async)
        for (i, e) in entries.iter().enumerate() {
            conn.execute(
                "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    "connect:alice:sess1",
                    e.entry_kind,
                    e.content,
                    i as i64,
                    e.created_at
                ],
            )
            .unwrap();
        }

        // Share
        create_share(
            &conn,
            "alice",
            "connect:alice:sess1",
            "flash-abc12345",
            Some("hello"),
            Some("gpt-4"),
            None,
            None,
        )
        .unwrap();

        // Pull
        let data = pull_share(&conn, "flash-abc12345").unwrap().unwrap();
        assert_eq!(data.shared_by, "alice");
        assert_eq!(data.entries.len(), 2);
        assert_eq!(data.entries[0].content.as_deref(), Some("hello"));
        assert_eq!(data.last_prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn test_pull_increments_count() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(&conn, "alice", "sess", "flash-cnt", None, None, None, None).unwrap();

        pull_share(&conn, "flash-cnt").unwrap();
        pull_share(&conn, "flash-cnt").unwrap();

        let shares = list_shares(&conn, "alice").unwrap();
        assert_eq!(shares[0].pull_count, 2);
    }

    #[test]
    fn test_max_pulls_enforced() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(
            &conn,
            "alice",
            "sess",
            "flash-once",
            None,
            None,
            None,
            Some(1),
        )
        .unwrap();

        let first = pull_share(&conn, "flash-once").unwrap();
        assert!(first.is_some());

        let second = pull_share(&conn, "flash-once").unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn test_expired_share_returns_none() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        // TTL of 0 = already expired
        create_share(
            &conn,
            "alice",
            "sess",
            "flash-exp",
            None,
            None,
            Some(0),
            None,
        )
        .unwrap();

        let result = pull_share(&conn, "flash-exp").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_revoke_share() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(&conn, "alice", "sess", "flash-rev", None, None, None, None).unwrap();

        let revoked = revoke_share(&conn, "flash-rev", "alice").unwrap();
        assert!(revoked);

        // Should be gone
        let result = pull_share(&conn, "flash-rev").unwrap();
        assert!(result.is_none());

        // Session entries should be cleaned up
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE chat_key = 'shared:flash-rev'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_revoke_wrong_user() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();
        create_user(&conn, "bob", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(&conn, "alice", "sess", "flash-own", None, None, None, None).unwrap();

        let revoked = revoke_share(&conn, "flash-own", "bob").unwrap();
        assert!(!revoked);
    }

    #[test]
    fn test_cleanup_expired_shares() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(
            &conn,
            "alice",
            "sess",
            "flash-old",
            None,
            None,
            Some(0),
            None,
        )
        .unwrap();

        let cleaned = cleanup_expired_shares(&conn).unwrap();
        assert_eq!(cleaned, 1);

        let shares = list_shares(&conn, "alice").unwrap();
        assert!(shares.is_empty());
    }

    #[test]
    fn test_list_shares() {
        let conn = open_db();
        create_user(&conn, "alice", false).unwrap();

        conn.execute(
            "INSERT INTO sessions (chat_key, entry_kind, content, turn_index, created_at)
             VALUES ('sess', 'user', 'hi', 0, 1000)",
            [],
        )
        .unwrap();

        create_share(
            &conn,
            "alice",
            "sess",
            "flash-a",
            Some("first"),
            None,
            None,
            None,
        )
        .unwrap();
        create_share(
            &conn,
            "alice",
            "sess",
            "flash-b",
            Some("second"),
            None,
            None,
            None,
        )
        .unwrap();

        let shares = list_shares(&conn, "alice").unwrap();
        assert_eq!(shares.len(), 2);
        // Ordered by created_at DESC
        assert_eq!(shares[0].token, "flash-b");
        assert_eq!(shares[1].token, "flash-a");
    }
}
