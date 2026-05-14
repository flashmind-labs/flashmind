//! Session persistence.
//!
//! Wraps `flashmind_memory::SessionStore` and maps between
//! `ConversationEntry` and `SessionEntry` for SQLite storage.

use std::path::Path;

use anyhow::Result;
use chrono::Utc;

use flashmind_core::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::rusqlite;
use flashmind_memory::session::schema::init_session_schema;
use flashmind_memory::session::{SessionEntry, SessionEntryKind, SessionStore};
use flashmind_memory::tokio_rusqlite;
use flashmind_types::{ContentPart, ToolCall};

// ---------------------------------------------------------------------------
// Sessions wrapper
// ---------------------------------------------------------------------------

/// High-level session manager backed by SQLite.
#[derive(Clone)]
pub struct Sessions {
    store: SessionStore,
    conn: tokio_rusqlite::Connection,
}

impl Sessions {
    pub async fn connect(db_path: &Path) -> Result<Self> {
        let conn = tokio_rusqlite::Connection::open(db_path).await?;
        conn.call(|c| {
            init_session_schema(c)?;
            init_local_sessions_schema(c)?;
            Ok::<_, rusqlite::Error>(())
        })
        .await?;
        let store = SessionStore::new(conn.clone());
        Ok(Self { store, conn })
    }

    pub async fn save(&self, chat_key: &str, conversation: &Conversation) -> Result<()> {
        let entries: Vec<SessionEntry> = conversation
            .entries()
            .iter()
            .map(to_session_entry)
            .collect();
        self.store.delete_session(chat_key).await?;
        self.store.save_entries(&entries).await?;
        tracing::debug!("saved {} entries for {chat_key}", entries.len());
        Ok(())
    }

    pub async fn load(&self, chat_key: &str) -> Result<Option<Conversation>> {
        let entries = self.store.load(chat_key).await?;
        if entries.is_empty() {
            return Ok(None);
        }
        let mut conv = Conversation::new();
        for entry in &entries {
            if let Some(ce) = from_session_entry(entry) {
                conv.add(ce);
            }
        }
        Ok(Some(conv))
    }

    pub async fn delete(&self, chat_key: &str) -> Result<()> {
        self.store.delete_session(chat_key).await?;
        delete_local_session(&self.conn, chat_key).await?;
        Ok(())
    }

    pub async fn save_metadata(
        &self,
        key: &str,
        prompt: &str,
        model: &str,
        cwd: &str,
    ) -> Result<()> {
        save_local_session(&self.conn, key, prompt, model, cwd).await
    }

    pub async fn list_local(&self) -> Result<Vec<LocalSession>> {
        list_local_sessions(&self.conn).await
    }

    pub fn connection(&self) -> &tokio_rusqlite::Connection {
        &self.conn
    }

    /// Update the title for a local session.
    pub async fn set_title(&self, key: &str, title: Option<&str>) -> Result<()> {
        set_title(&self.conn, key, title).await
    }
}

// ---------------------------------------------------------------------------
// ConversationEntry <-> SessionEntry mapping
// ---------------------------------------------------------------------------

fn to_session_entry(ce: &ConversationEntry) -> SessionEntry {
    let created_at = ce.timestamp.timestamp();
    let (entry_kind, content, tool_calls, tool_call_id, tool_name, metadata) = match &ce.kind {
        EntryKind::SystemPrompt(s) => (
            SessionEntryKind::SystemPrompt,
            s.clone(),
            None,
            None,
            None,
            None,
        ),
        EntryKind::Developer {
            content,
            tag,
            metadata,
        } => (
            SessionEntryKind::Developer { tag: tag.clone() },
            content.clone(),
            None,
            None,
            None,
            metadata.as_ref().and_then(|m| serde_json::to_value(m).ok()),
        ),
        EntryKind::User { content, parts } => (
            SessionEntryKind::User,
            content.clone(),
            None,
            None,
            None,
            parts.as_ref().and_then(|p| serde_json::to_value(p).ok()),
        ),
        EntryKind::Assistant {
            content,
            tool_calls,
        } => (
            SessionEntryKind::Assistant,
            content.clone(),
            tool_calls
                .as_ref()
                .and_then(|tc| serde_json::to_value(tc).ok()),
            None,
            None,
            None,
        ),
        EntryKind::Tool { call_id, output } => (
            SessionEntryKind::Tool,
            output.clone(),
            None,
            Some(call_id.clone()),
            None,
            None,
        ),
    };

    SessionEntry {
        id: 0,
        chat_key: String::new(),
        entry_kind,
        content,
        tool_calls,
        tool_call_id,
        tool_name,
        metadata,
        turn_index: 0,
        created_at,
    }
}

fn from_session_entry(se: &SessionEntry) -> Option<ConversationEntry> {
    let kind = match &se.entry_kind {
        SessionEntryKind::SystemPrompt => EntryKind::SystemPrompt(se.content.clone()),
        SessionEntryKind::Developer { tag } => EntryKind::Developer {
            content: se.content.clone(),
            tag: tag.clone(),
            metadata: se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_value(m.clone()).ok()),
        },
        SessionEntryKind::User => {
            let parts: Option<Vec<ContentPart>> = se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_value(m.clone()).ok());
            EntryKind::User {
                content: se.content.clone(),
                parts,
            }
        }
        SessionEntryKind::Assistant => {
            let tool_calls: Option<Vec<ToolCall>> = se
                .tool_calls
                .as_ref()
                .and_then(|tc| serde_json::from_value(tc.clone()).ok());
            EntryKind::Assistant {
                content: se.content.clone(),
                tool_calls,
            }
        }
        SessionEntryKind::Tool => EntryKind::Tool {
            call_id: se.tool_call_id.clone().unwrap_or_default(),
            output: se.content.clone(),
        },
    };

    Some(ConversationEntry {
        kind,
        timestamp: chrono::DateTime::from_timestamp(se.created_at, 0).unwrap_or_else(Utc::now),
    })
}

// ---------------------------------------------------------------------------
// Local session metadata table
// ---------------------------------------------------------------------------

/// Metadata about a local session (key, first prompt, model, etc.).
#[derive(Debug, Clone)]
pub struct LocalSession {
    pub key: String,
    pub prompt: String,
    pub model: String,
    pub cwd: Option<String>,
    pub updated_at: i64,
    pub title: Option<String>,
}

fn init_local_sessions_schema(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS local_sessions (
            key        TEXT PRIMARY KEY,
            prompt     TEXT NOT NULL DEFAULT '',
            model      TEXT NOT NULL DEFAULT '',
            cwd        TEXT NOT NULL DEFAULT '',
            updated_at INTEGER NOT NULL DEFAULT 0,
            title      TEXT
        );",
    )?;
    // Migration: add title column if it doesn't exist
    let _ = conn.execute_batch("ALTER TABLE local_sessions ADD COLUMN title TEXT;");
    Ok(())
}

async fn save_local_session(
    conn: &tokio_rusqlite::Connection,
    key: &str,
    prompt: &str,
    model: &str,
    cwd: &str,
) -> Result<()> {
    let key = key.to_string();
    let prompt = prompt.chars().take(200).collect::<String>();
    let model = model.to_string();
    let cwd = cwd.to_string();
    let now = Utc::now().timestamp();

    conn.call(move |c| {
        c.execute(
            "INSERT INTO local_sessions (key, prompt, model, cwd, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(key) DO UPDATE SET prompt = ?2, model = ?3, cwd = ?4, updated_at = ?5",
            rusqlite::params![key, prompt, model, cwd, now],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

async fn delete_local_session(conn: &tokio_rusqlite::Connection, key: &str) -> Result<()> {
    let key = key.to_string();
    conn.call(move |c| {
        c.execute(
            "DELETE FROM local_sessions WHERE key = ?1",
            rusqlite::params![key],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

async fn list_local_sessions(conn: &tokio_rusqlite::Connection) -> Result<Vec<LocalSession>> {
    let rows = conn
        .call(|c| {
            let mut stmt = c.prepare(
                "SELECT key, prompt, model, cwd, updated_at, title
                 FROM local_sessions
                 ORDER BY updated_at DESC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    let cwd: String = row.get(3)?;
                    Ok(LocalSession {
                        key: row.get(0)?,
                        prompt: row.get(1)?,
                        model: row.get(2)?,
                        cwd: if cwd.is_empty() { None } else { Some(cwd) },
                        updated_at: row.get(4)?,
                        title: row.get(5)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok::<_, rusqlite::Error>(rows)
        })
        .await?;
    Ok(rows)
}

pub async fn set_title(
    conn: &tokio_rusqlite::Connection,
    key: &str,
    title: Option<&str>,
) -> Result<()> {
    let key = key.to_string();
    let title = title.map(|s| s.to_string());
    conn.call(move |c| {
        c.execute(
            "UPDATE local_sessions SET title = ?2 WHERE key = ?1",
            rusqlite::params![key, title],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session pick
// ---------------------------------------------------------------------------

/// Result of choosing a session (used by both CLI picker and desktop UI).
pub enum SessionPick {
    /// Resume an existing session.
    Resume(String),
    /// Start a new session with the given key.
    New(String),
}

/// Generate a new session key from the current timestamp.
pub fn new_session_key() -> String {
    format!("{}", Utc::now().timestamp())
}
