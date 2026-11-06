//! Read access to legacy SQLite sessions for import.
use super::{
    store::SessionSummary,
    types::{SessionEntry, SessionEntryKind},
};
use flashmind_types::ReasoningLevel;
use rusqlite::{OptionalExtension, types::Type};
pub(super) struct SqliteSessionStore {
    conn: tokio_rusqlite::Connection,
}
impl SqliteSessionStore {
    pub fn new(conn: tokio_rusqlite::Connection) -> Self {
        Self { conn }
    }
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
    pub async fn list_sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let summaries = self
            .conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT s.chat_key,
                            COUNT(*) AS entry_count,
                            MAX(s.created_at) AS last_updated,
                            m.title,
                            m.model,
                            (SELECT content FROM sessions
                             WHERE chat_key = s.chat_key
                               AND entry_kind = '{\"type\":\"user\"}'
                             ORDER BY turn_index ASC LIMIT 1) AS first_message,
                            (SELECT content FROM sessions
                             WHERE chat_key = s.chat_key
                               AND entry_kind IN ('{\"type\":\"user\"}', '{\"type\":\"assistant\"}')
                               AND content IS NOT NULL AND content != ''
                             ORDER BY turn_index DESC, created_at DESC LIMIT 1) AS last_message,
                            m.working_dir,
                            m.total_cost,
                            m.last_usage,
                            m.total_usage
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
                            first_message: row.get(5)?,
                            last_message: row.get(6)?,
                            working_dir: row.get(7)?,
                            total_cost: row.get(8)?,
                            last_usage: row.get(9)?,
                            total_usage: row
                                .get::<_, Option<String>>(10)?
                                .map(|json| {
                                    serde_json::from_str(&json).map_err(|error| {
                                        rusqlite::Error::FromSqlConversionFailure(
                                            10,
                                            Type::Text,
                                            Box::new(error),
                                        )
                                    })
                                })
                                .transpose()?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                Ok::<_, rusqlite::Error>(rows)
            })
            .await?;

        Ok(summaries)
    }
    pub async fn reasoning(&self, chat_key: &str) -> anyhow::Result<Option<ReasoningLevel>> {
        let chat_key = chat_key.to_owned();
        let value: Option<String> = self
            .conn
            .call(move |conn| {
                conn.query_row(
                    "SELECT reasoning FROM session_meta WHERE chat_key = ?1",
                    [chat_key],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await?;
        value
            .map(|value| serde_json::from_str(&value))
            .transpose()
            .map_err(Into::into)
    }
}
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
