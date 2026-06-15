//! CRUD operations for session persistence.

use super::types::{SessionEntry, SessionEntryKind};

/// Serialized entry row: (chat_key, entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, turn_index, created_at).
type EntryRow = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    i64,
    i64,
);

/// Partial row for branch copy: (entry_kind, content, tool_calls, tool_call_id, tool_name, metadata, created_at).
type BranchRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    i64,
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Summary of a stored session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Conversation key.
    pub chat_key: String,
    /// Number of entries in the session.
    pub entry_count: i64,
    /// Unix timestamp of the most recent entry.
    pub last_updated: i64,
    /// Optional human-readable title for the session.
    pub title: Option<String>,
    /// Model used for this session (e.g. `openrouter:anthropic/claude-sonnet-4`).
    pub model: Option<String>,
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

/// Persistent store for conversation session entries backed by SQLite.
#[derive(Clone)]
pub struct SessionStore {
    conn: tokio_rusqlite::Connection,
}

impl SessionStore {
    /// Create a new `SessionStore` wrapping an existing connection.
    pub fn new(conn: tokio_rusqlite::Connection) -> Self {
        Self { conn }
    }

    /// Insert a single entry and return the generated row ID.
    pub async fn save_entry(&self, entry: &SessionEntry) -> anyhow::Result<i64> {
        let chat_key = entry.chat_key.clone();
        let entry_kind = serde_json::to_string(&entry.entry_kind)?;
        let content = entry.content.clone();
        let tool_calls = entry.tool_calls.as_ref().map(|v| v.to_string());
        let tool_call_id = entry.tool_call_id.clone();
        let tool_name = entry.tool_name.clone();
        let metadata = entry.metadata.as_ref().map(|v| v.to_string());
        let turn_index = entry.turn_index;
        let created_at = entry.created_at;

        let id = self
            .conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO sessions
                        (chat_key, entry_kind, content, tool_calls, tool_call_id,
                         tool_name, metadata, turn_index, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    rusqlite::params![
                        chat_key,
                        entry_kind,
                        content,
                        tool_calls,
                        tool_call_id,
                        tool_name,
                        metadata,
                        turn_index,
                        created_at,
                    ],
                )?;
                Ok::<_, rusqlite::Error>(conn.last_insert_rowid())
            })
            .await?;

        Ok(id)
    }

    /// Insert multiple entries in a single transaction.
    pub async fn save_entries(&self, entries: &[SessionEntry]) -> anyhow::Result<()> {
        let owned: Vec<EntryRow> = entries
            .iter()
            .map(|e| {
                Ok((
                    e.chat_key.clone(),
                    serde_json::to_string(&e.entry_kind)?,
                    e.content.clone(),
                    e.tool_calls.as_ref().map(|v| v.to_string()),
                    e.tool_call_id.clone(),
                    e.tool_name.clone(),
                    e.metadata.as_ref().map(|v| v.to_string()),
                    e.turn_index,
                    e.created_at,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                for (
                    chat_key,
                    entry_kind,
                    content,
                    tool_calls,
                    tool_call_id,
                    tool_name,
                    metadata,
                    turn_index,
                    created_at,
                ) in &owned
                {
                    tx.execute(
                        "INSERT INTO sessions
                            (chat_key, entry_kind, content, tool_calls, tool_call_id,
                             tool_name, metadata, turn_index, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        rusqlite::params![
                            chat_key,
                            entry_kind,
                            content,
                            tool_calls,
                            tool_call_id,
                            tool_name,
                            metadata,
                            turn_index,
                            created_at,
                        ],
                    )?;
                }
                tx.commit()?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }

    /// Load all entries for a conversation, ordered by turn index.
    pub async fn load(&self, chat_key: &str) -> anyhow::Result<Vec<SessionEntry>> {
        let chat_key = chat_key.to_string();

        let entries = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, chat_key, entry_kind, content, tool_calls, tool_call_id,
                            tool_name, metadata, turn_index, created_at
                     FROM sessions
                     WHERE chat_key = ?1
                     ORDER BY turn_index ASC",
                )?;

                let rows = stmt
                    .query_map(rusqlite::params![chat_key], |row| {
                        Ok(RawRow {
                            id: row.get(0)?,
                            chat_key: row.get(1)?,
                            entry_kind: row.get(2)?,
                            content: row.get(3)?,
                            tool_calls: row.get(4)?,
                            tool_call_id: row.get(5)?,
                            tool_name: row.get(6)?,
                            metadata: row.get(7)?,
                            turn_index: row.get(8)?,
                            created_at: row.get(9)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                Ok::<_, rusqlite::Error>(rows)
            })
            .await?;

        entries.into_iter().map(raw_to_entry).collect()
    }

    /// Delete all entries for a conversation. Returns the number of deleted rows.
    pub async fn delete_session(&self, chat_key: &str) -> anyhow::Result<u64> {
        let chat_key = chat_key.to_string();

        let count = self
            .conn
            .call(move |conn| {
                let count = conn.execute(
                    "DELETE FROM sessions WHERE chat_key = ?1",
                    rusqlite::params![chat_key],
                )?;
                Ok::<_, rusqlite::Error>(count as u64)
            })
            .await?;

        Ok(count)
    }

    /// Delete all sessions whose chat_key starts with the given prefix.
    pub async fn delete_sessions_by_prefix(&self, prefix: &str) -> anyhow::Result<u64> {
        let prefix = format!("{prefix}%");
        let count = self
            .conn
            .call(move |conn| {
                let count = conn.execute(
                    "DELETE FROM sessions WHERE chat_key LIKE ?1",
                    rusqlite::params![prefix],
                )?;
                Ok::<_, rusqlite::Error>(count as u64)
            })
            .await?;
        Ok(count)
    }

    /// Delete all existing entries for a conversation and replace them with new ones
    /// in a single transaction.
    pub async fn rewrite(&self, chat_key: &str, entries: &[SessionEntry]) -> anyhow::Result<()> {
        let chat_key_owned = chat_key.to_string();
        let owned: Vec<EntryRow> = entries
            .iter()
            .map(|e| {
                Ok((
                    e.chat_key.clone(),
                    serde_json::to_string(&e.entry_kind)?,
                    e.content.clone(),
                    e.tool_calls.as_ref().map(|v| v.to_string()),
                    e.tool_call_id.clone(),
                    e.tool_name.clone(),
                    e.metadata.as_ref().map(|v| v.to_string()),
                    e.turn_index,
                    e.created_at,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM sessions WHERE chat_key = ?1",
                    rusqlite::params![chat_key_owned],
                )?;
                for (
                    chat_key,
                    entry_kind,
                    content,
                    tool_calls,
                    tool_call_id,
                    tool_name,
                    metadata,
                    turn_index,
                    created_at,
                ) in &owned
                {
                    tx.execute(
                        "INSERT INTO sessions
                            (chat_key, entry_kind, content, tool_calls, tool_call_id,
                             tool_name, metadata, turn_index, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        rusqlite::params![
                            chat_key,
                            entry_kind,
                            content,
                            tool_calls,
                            tool_call_id,
                            tool_name,
                            metadata,
                            turn_index,
                            created_at,
                        ],
                    )?;
                }
                tx.commit()?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }

    /// Copy all entries from one conversation to another with reset turn indices.
    /// Returns the number of copied entries.
    pub async fn branch(&self, from_key: &str, to_key: &str) -> anyhow::Result<u64> {
        let from_key = from_key.to_string();
        let to_key = to_key.to_string();

        let count = self
            .conn
            .call(move |conn| {
                let rows: Vec<BranchRow> = {
                    let mut stmt = conn.prepare(
                        "SELECT entry_kind, content, tool_calls, tool_call_id,
                                tool_name, metadata, created_at
                         FROM sessions
                         WHERE chat_key = ?1
                         ORDER BY turn_index ASC",
                    )?;

                    stmt.query_map(rusqlite::params![from_key], |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?
                };

                let count = rows.len() as u64;

                let tx = conn.transaction()?;
                for (
                    i,
                    (
                        entry_kind,
                        content,
                        tool_calls,
                        tool_call_id,
                        tool_name,
                        metadata,
                        created_at,
                    ),
                ) in rows.iter().enumerate()
                {
                    tx.execute(
                        "INSERT INTO sessions
                            (chat_key, entry_kind, content, tool_calls, tool_call_id,
                             tool_name, metadata, turn_index, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        rusqlite::params![
                            to_key,
                            entry_kind,
                            content,
                            tool_calls,
                            tool_call_id,
                            tool_name,
                            metadata,
                            i as i64,
                            created_at,
                        ],
                    )?;
                }
                tx.commit()?;

                Ok::<_, rusqlite::Error>(count)
            })
            .await?;

        Ok(count)
    }

    /// Delete sessions whose most recent entry is older than `max_age_days`.
    /// Returns the number of deleted rows.
    pub async fn prune(&self, max_age_days: u32) -> anyhow::Result<u64> {
        let cutoff = chrono::Utc::now().timestamp() - i64::from(max_age_days) * 86400;

        let count = self
            .conn
            .call(move |conn| {
                let count = conn.execute(
                    "DELETE FROM sessions
                     WHERE chat_key IN (
                         SELECT chat_key FROM sessions
                         GROUP BY chat_key
                         HAVING MAX(created_at) < ?1
                     )",
                    rusqlite::params![cutoff],
                )?;
                Ok::<_, rusqlite::Error>(count as u64)
            })
            .await?;

        Ok(count)
    }

    /// List all sessions with entry count, last update timestamp, and optional
    /// metadata (title, model) from the `session_meta` table.
    pub async fn list_sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let summaries = self
            .conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT s.chat_key,
                            COUNT(*) AS entry_count,
                            MAX(s.created_at) AS last_updated,
                            m.title,
                            m.model
                     FROM sessions s
                     LEFT JOIN session_meta m ON s.chat_key = m.chat_key
                     GROUP BY s.chat_key
                     ORDER BY last_updated DESC",
                )?;

                let rows = stmt
                    .query_map([], |row| {
                        Ok(SessionSummary {
                            chat_key: row.get(0)?,
                            entry_count: row.get(1)?,
                            last_updated: row.get(2)?,
                            title: row.get(3)?,
                            model: row.get(4)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                Ok::<_, rusqlite::Error>(rows)
            })
            .await?;

        Ok(summaries)
    }
    // -----------------------------------------------------------------------
    // Session metadata
    // -----------------------------------------------------------------------

    /// Insert or replace metadata (title, model) for a session.
    pub async fn save_meta(
        &self,
        chat_key: &str,
        title: Option<&str>,
        model: Option<&str>,
    ) -> anyhow::Result<()> {
        let chat_key = chat_key.to_string();
        let title = title.map(str::to_string);
        let model = model.map(str::to_string);
        let created_at = chrono::Utc::now().timestamp();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT OR REPLACE INTO session_meta (chat_key, title, model, created_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![chat_key, title, model, created_at],
                )?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }

    /// Update only the title for an existing session metadata row.
    pub async fn update_title(&self, chat_key: &str, title: &str) -> anyhow::Result<()> {
        let chat_key = chat_key.to_string();
        let title = title.to_string();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "UPDATE session_meta SET title = ?1 WHERE chat_key = ?2",
                    rusqlite::params![title, chat_key],
                )?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }

    /// Delete metadata for a session. Should be called alongside
    /// [`delete_session`] for full cleanup.
    pub async fn delete_meta(&self, chat_key: &str) -> anyhow::Result<()> {
        let chat_key = chat_key.to_string();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "DELETE FROM session_meta WHERE chat_key = ?1",
                    rusqlite::params![chat_key],
                )?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct RawRow {
    id: i64,
    chat_key: String,
    entry_kind: String,
    content: String,
    tool_calls: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    metadata: Option<String>,
    turn_index: i64,
    created_at: i64,
}

fn raw_to_entry(raw: RawRow) -> anyhow::Result<SessionEntry> {
    let entry_kind: SessionEntryKind = serde_json::from_str(&raw.entry_kind)?;
    let tool_calls = raw
        .tool_calls
        .map(|s| serde_json::from_str(&s))
        .transpose()?;
    let metadata = raw.metadata.map(|s| serde_json::from_str(&s)).transpose()?;

    Ok(SessionEntry {
        id: raw.id,
        chat_key: raw.chat_key,
        entry_kind,
        content: raw.content,
        tool_calls,
        tool_call_id: raw.tool_call_id,
        tool_name: raw.tool_name,
        metadata,
        turn_index: raw.turn_index,
        created_at: raw.created_at,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    async fn test_store() -> SessionStore {
        let conn = tokio_rusqlite::Connection::open_in_memory().await.unwrap();
        conn.call(|conn| {
            super::super::schema::init_session_schema(conn)?;
            Ok::<_, rusqlite::Error>(())
        })
        .await
        .unwrap();
        SessionStore::new(conn)
    }

    fn make_entry(
        chat_key: &str,
        kind: SessionEntryKind,
        content: &str,
        turn: i64,
    ) -> SessionEntry {
        SessionEntry {
            id: 0,
            chat_key: chat_key.to_string(),
            entry_kind: kind,
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            turn_index: turn,
            created_at: Utc::now().timestamp(),
        }
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let store = test_store().await;

        let entries = vec![
            make_entry("chat-1", SessionEntryKind::User, "hello", 0),
            make_entry("chat-1", SessionEntryKind::Assistant, "hi there", 1),
        ];

        store.save_entries(&entries).await.unwrap();

        let loaded = store.load("chat-1").await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[0].entry_kind, SessionEntryKind::User);
        assert_eq!(loaded[1].content, "hi there");
        assert_eq!(loaded[1].entry_kind, SessionEntryKind::Assistant);
    }

    #[tokio::test]
    async fn save_entry_returns_id() {
        let store = test_store().await;

        let entry = make_entry("chat-1", SessionEntryKind::User, "test", 0);
        let id = store.save_entry(&entry).await.unwrap();
        assert!(id > 0);
    }

    #[tokio::test]
    async fn delete_session_removes_all_entries() {
        let store = test_store().await;

        let entries = vec![
            make_entry("chat-1", SessionEntryKind::User, "a", 0),
            make_entry("chat-1", SessionEntryKind::Assistant, "b", 1),
            make_entry("chat-2", SessionEntryKind::User, "c", 0),
        ];
        store.save_entries(&entries).await.unwrap();

        let count = store.delete_session("chat-1").await.unwrap();
        assert_eq!(count, 2);

        let loaded = store.load("chat-1").await.unwrap();
        assert!(loaded.is_empty());

        let other = store.load("chat-2").await.unwrap();
        assert_eq!(other.len(), 1);
    }

    #[tokio::test]
    async fn branch_copies_entries() {
        let store = test_store().await;

        let entries = vec![
            make_entry("src", SessionEntryKind::User, "msg-1", 0),
            make_entry("src", SessionEntryKind::Assistant, "msg-2", 1),
            make_entry("src", SessionEntryKind::User, "msg-3", 2),
        ];
        store.save_entries(&entries).await.unwrap();

        let count = store.branch("src", "dst").await.unwrap();
        assert_eq!(count, 3);

        let dst = store.load("dst").await.unwrap();
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[0].turn_index, 0);
        assert_eq!(dst[1].turn_index, 1);
        assert_eq!(dst[2].turn_index, 2);
        assert_eq!(dst[0].content, "msg-1");

        // Source is untouched.
        let src = store.load("src").await.unwrap();
        assert_eq!(src.len(), 3);
    }

    #[tokio::test]
    async fn prune_removes_old_sessions() {
        let store = test_store().await;

        let old_ts = Utc::now().timestamp() - 100 * 86400;
        let mut old_entry = make_entry("old-chat", SessionEntryKind::User, "old", 0);
        old_entry.created_at = old_ts;
        store.save_entry(&old_entry).await.unwrap();

        let fresh = make_entry("new-chat", SessionEntryKind::User, "new", 0);
        store.save_entry(&fresh).await.unwrap();

        let deleted = store.prune(30).await.unwrap();
        assert_eq!(deleted, 1);

        let old = store.load("old-chat").await.unwrap();
        assert!(old.is_empty());

        let new = store.load("new-chat").await.unwrap();
        assert_eq!(new.len(), 1);
    }

    #[tokio::test]
    async fn list_sessions_returns_summaries() {
        let store = test_store().await;

        let entries = vec![
            make_entry("chat-a", SessionEntryKind::User, "1", 0),
            make_entry("chat-a", SessionEntryKind::Assistant, "2", 1),
            make_entry("chat-b", SessionEntryKind::User, "3", 0),
        ];
        store.save_entries(&entries).await.unwrap();

        let sessions = store.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 2);

        let a = sessions.iter().find(|s| s.chat_key == "chat-a").unwrap();
        assert_eq!(a.entry_count, 2);

        let b = sessions.iter().find(|s| s.chat_key == "chat-b").unwrap();
        assert_eq!(b.entry_count, 1);
    }
}
