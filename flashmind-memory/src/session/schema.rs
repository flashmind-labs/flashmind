//! DDL for the sessions table.

use rusqlite::{Connection, Result};

/// Create the sessions table and indexes if they don't already exist.
pub fn init_session_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_key     TEXT    NOT NULL,
            entry_kind   TEXT    NOT NULL,
            content      TEXT    NOT NULL DEFAULT '',
            tool_calls   TEXT,
            tool_call_id TEXT,
            tool_name    TEXT,
            metadata     TEXT,
            turn_index   INTEGER NOT NULL,
            created_at   INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_chat_turn
            ON sessions (chat_key, turn_index);",
    )?;
    Ok(())
}
