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
            ON sessions (chat_key, turn_index);

        CREATE TABLE IF NOT EXISTS session_meta (
            chat_key    TEXT PRIMARY KEY,
            title       TEXT,
            model       TEXT,
            working_dir TEXT,
            created_at  INTEGER NOT NULL
        );",
    )?;

    // Add missing metadata columns when opening older databases.
    for column in ["working_dir", "total_cost", "last_usage", "total_usage"] {
        let exists: bool = conn
            .prepare("SELECT COUNT(*) FROM pragma_table_info('session_meta') WHERE name = ?1")?
            .query_row([column], |row| row.get::<_, i64>(0))
            .map(|c| c > 0)?;
        if !exists {
            conn.execute_batch(&format!(
                "ALTER TABLE session_meta ADD COLUMN {column} TEXT"
            ))?;
        }
    }

    Ok(())
}
