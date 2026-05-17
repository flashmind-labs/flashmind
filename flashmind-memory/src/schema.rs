//! SQLite schema definitions for the memory store.
//!
//! Schema is initialized on startup via [`init_schema`] — all statements use
//! `IF NOT EXISTS` for idempotency.
//!
//! ## Tables
//!
//! ### `memories`
//! Core storage for long-term memories. Minimal: content + timestamps.
//!
//! ### `memories_vec`
//! sqlite-vec virtual table storing embedding vectors alongside memory IDs.
//!
//! ### `memories_fts`
//! FTS5 virtual table for BM25 keyword search over memory content.
//!
//! ### `memory_meta`
//! Arbitrary key-value metadata. Consumers define their own keys.

use rusqlite::{Connection, Result};

/// Initialize the database schema. Idempotent — safe to call on every startup.
pub fn init_schema(conn: &Connection, embedding_dim: usize) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id          TEXT    PRIMARY KEY,
            content     TEXT    NOT NULL,
            created_at  INTEGER NOT NULL,
            expires_at  INTEGER
        );",
    )?;

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at);
         CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories (expires_at)
             WHERE expires_at IS NOT NULL;",
    )?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory_meta (
            memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            key       TEXT NOT NULL,
            value     TEXT NOT NULL
        );",
    )?;

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_meta_memory_id ON memory_meta (memory_id);
         CREATE INDEX IF NOT EXISTS idx_meta_kv ON memory_meta (key, value);",
    )?;

    // -- memories_vec: sqlite-vec virtual table for vector similarity search --
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
             USING vec0(id TEXT PRIMARY KEY, embedding float[{embedding_dim}]);",
    ))?;

    // -- memories_fts: FTS5 for BM25 keyword search --
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
             USING fts5(content, content=memories, content_rowid=rowid);",
    )?;

    // FTS sync triggers
    conn.execute_batch(
        "CREATE TRIGGER IF NOT EXISTS memories_ai
             AFTER INSERT ON memories BEGIN
                 INSERT INTO memories_fts(rowid, content)
                 VALUES (new.rowid, new.content);
             END;

         CREATE TRIGGER IF NOT EXISTS memories_ad
             AFTER DELETE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                 VALUES ('delete', old.rowid, old.content);
             END;

         CREATE TRIGGER IF NOT EXISTS memories_au
             AFTER UPDATE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                 VALUES ('delete', old.rowid, old.content);
                 INSERT INTO memories_fts(rowid, content)
                 VALUES (new.rowid, new.content);
             END;",
    )?;

    // Migrate legacy schema: move old columns into memory_meta
    migrate_legacy(conn)?;

    Ok(())
}

/// Migrate data from the legacy schema (source, chat_key, identity, tool_name columns
/// and tags/memory_tags tables) into the new memory_meta table.
fn migrate_legacy(conn: &Connection) -> Result<()> {
    let has_source: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('memories') WHERE name = 'source'",
        [],
        |row| row.get(0),
    )?;

    if !has_source {
        return Ok(());
    }

    let has_meta: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = 'memory_meta'",
        [],
        |row| row.get(0),
    )?;

    // memory_meta was just created above, but check if we already migrated
    let already_migrated: bool = conn
        .query_row("SELECT COUNT(*) > 0 FROM memory_meta", [], |row| row.get(0))
        .unwrap_or(false);

    if already_migrated {
        return Ok(());
    }

    if !has_meta {
        return Ok(());
    }

    // Migrate source column
    conn.execute_batch(
        "INSERT OR IGNORE INTO memory_meta (memory_id, key, value)
         SELECT id, 'source', source FROM memories WHERE source IS NOT NULL;",
    )?;

    // Migrate chat_key column
    let has_chat_key: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('memories') WHERE name = 'chat_key'",
        [],
        |row| row.get(0),
    )?;
    if has_chat_key {
        conn.execute_batch(
            "INSERT OR IGNORE INTO memory_meta (memory_id, key, value)
             SELECT id, 'chat_key', chat_key FROM memories WHERE chat_key IS NOT NULL;",
        )?;
    }

    // Migrate identity column
    let has_identity: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('memories') WHERE name = 'identity'",
        [],
        |row| row.get(0),
    )?;
    if has_identity {
        conn.execute_batch(
            "INSERT OR IGNORE INTO memory_meta (memory_id, key, value)
             SELECT id, 'identity', identity FROM memories WHERE identity IS NOT NULL;",
        )?;
    }

    // Migrate tool_name column
    let has_tool_name: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('memories') WHERE name = 'tool_name'",
        [],
        |row| row.get(0),
    )?;
    if has_tool_name {
        conn.execute_batch(
            "INSERT OR IGNORE INTO memory_meta (memory_id, key, value)
             SELECT id, 'tool_name', tool_name FROM memories WHERE tool_name IS NOT NULL;",
        )?;
    }

    // Migrate tags from memory_tags join table
    let has_tags_table: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = 'memory_tags'",
        [],
        |row| row.get(0),
    )?;
    if has_tags_table {
        conn.execute_batch(
            "INSERT OR IGNORE INTO memory_meta (memory_id, key, value)
             SELECT mt.memory_id, 'tag', t.name
             FROM memory_tags mt
             JOIN tags t ON t.id = mt.tag_id;",
        )?;
    }

    Ok(())
}
