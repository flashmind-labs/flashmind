//! SQLite schema definitions for the memory store.
//!
//! Schema is initialized on startup via [`init_schema`] — all statements use
//! `IF NOT EXISTS` for idempotency.
//!
//! ## Tables
//!
//! ### `memories`
//! Core storage for long-term memories. Each row is a piece of information the
//! agent has decided to remember (facts, preferences, tool configs, etc.).
//! Scoped by `chat_key` (NULL = global, `"telegram:123"` = chat-local).
//!
//! ### `memories_vec`
//! sqlite-vec virtual table storing embedding vectors alongside memory IDs.
//! Enables cosine-similarity vector search via `WHERE embedding MATCH ?`.
//!
//! ### `memories_fts`
//! FTS5 virtual table for BM25 keyword search over memory content.
//! Kept in sync with `memories` via triggers (insert/update/delete).
//!
//! ### `tags`
//! Fixed vocabulary of memory categories. Enforced by the [`Tag`] enum —
//! new tags require adding a variant. Seeded on startup.
//!
//! ### `memory_tags`
//! Junction table linking memories to tags (many-to-many). CASCADE delete.

use rusqlite::{Connection, Result};
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};

/// How a memory was created.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Display, EnumString, IntoStaticStr, Serialize, Deserialize,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Extracted from a conversation turn by the LLM capture agent.
    Conversation,
    /// Extracted by the background capture agent post-turn.
    Capture,
    /// Explicitly stored by the user via `memory_store` tool.
    Manual,
    /// Extracted by keyword-based semantic capture (no LLM).
    Semantic,
    /// Stored by the periodic curation agent (merge/split/remove).
    Curation,
}

/// Scope for memory storage and search.
///
/// Each memory is either *global* (shared across all chats, `chat_key` is NULL)
/// or *local* (scoped to a single `chat_key`). Passing `None` to search means
/// "match both scopes".
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Display, EnumString, IntoStaticStr, Serialize, Deserialize,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Global memories — `chat_key IS NULL`, accessible across all chats.
    Global,
    /// Local memories — `chat_key = <key>`, scoped to the current chat.
    Local,
}

/// Memory tag — fixed vocabulary for categorization.
/// New tags require adding a variant here; seeded into the `tags` table on startup.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Display,
    EnumString,
    EnumIter,
    IntoStaticStr,
    Serialize,
    Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum Tag {
    /// A concrete piece of information (timezone, name, account ID).
    #[strum(serialize = "fact")]
    Fact,
    /// A user preference (coding style, output format, tool behavior).
    #[strum(serialize = "preference")]
    Preference,
    /// A time-bound event or interaction worth remembering.
    #[strum(serialize = "episode")]
    Episode,
    /// Project-level context (deadlines, architecture decisions, team info).
    #[strum(serialize = "project")]
    Project,
    /// Synthesized user profile (consolidated from other memories).
    #[strum(serialize = "user-profile")]
    UserProfile,
    /// Tool-specific preference or configuration. Use with `tool_name` column
    /// for precise pre-tool RAG filtering.
    #[strum(serialize = "tool")]
    Tool,
}

/// Initialize the database schema. Idempotent — safe to call on every startup.
pub fn init_schema(conn: &Connection, embedding_dim: usize) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;

    // -- memories: core storage for long-term memories --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id          TEXT    PRIMARY KEY,  -- UUID
            content     TEXT    NOT NULL,     -- the memory text
            source      TEXT    NOT NULL,     -- Source enum: conversation, capture, manual, semantic
            chat_key    TEXT,                 -- NULL = global, 'telegram:123' = chat-scoped
            identity    TEXT,                 -- agent-curated canonical user identity (cross-channel)
            tool_name   TEXT,                 -- set when tag = 'tool', e.g. 'bash', 'file_write'
            created_at  INTEGER NOT NULL,     -- unix epoch seconds
            expires_at  INTEGER              -- unix epoch seconds, NULL = never expires
        );",
    )?;

    // Migration: add tool_name column if it doesn't exist (for existing databases)
    let has_tool_name: Result<i64> = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'tool_name'",
        [],
        |row| row.get(0),
    );
    if Ok(0) == has_tool_name {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN tool_name TEXT;")?;
    }

    // Migration: add identity column if it doesn't exist
    let has_identity: Result<i64> = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'identity'",
        [],
        |row| row.get(0),
    );
    if Ok(0) == has_identity {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN identity TEXT;")?;
    }

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_chat_key ON memories (chat_key);
         CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at);
         CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories (expires_at)
             WHERE expires_at IS NOT NULL;",
    )?;

    // -- memories_vec: sqlite-vec virtual table for vector similarity search --
    // Joined with memories by id. Uses cosine distance internally.
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
             USING vec0(id TEXT PRIMARY KEY, embedding float[{embedding_dim}]);",
    ))?;

    // -- memories_fts: FTS5 for BM25 keyword search --
    // content= makes it a content-sync table (reads from memories).
    // Triggers below keep it in sync on insert/update/delete.
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
             USING fts5(content, content=memories, content_rowid=rowid);",
    )?;

    // FTS sync triggers — maintain memories_fts when memories changes
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

    // -- tags: fixed vocabulary, seeded from Tag enum --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tags (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    NOT NULL UNIQUE
        );",
    )?;

    // -- memory_tags: many-to-many junction, CASCADE on memory delete --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id TEXT    NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            tag_id    INTEGER NOT NULL REFERENCES tags(id),
            PRIMARY KEY (memory_id, tag_id)
        );",
    )?;

    // Seed tag variants from the Tag enum
    for tag in Tag::iter() {
        conn.execute(
            "INSERT OR IGNORE INTO tags (name) VALUES (?1);",
            rusqlite::params![tag.to_string()],
        )?;
    }

    Ok(())
}
