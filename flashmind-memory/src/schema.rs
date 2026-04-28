//! SQLite schema definitions for Flash's persistence layer.
//!
//! All tables live in a single `flash.db` file. Schema is initialized on startup
//! via [`init_schema`] — all statements use `IF NOT EXISTS` for idempotency.
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
//! Separate from `memories` because sqlite-vec requires a virtual table —
//! vectors can't be a column on a regular table.
//!
//! ### `memories_fts`
//! FTS5 virtual table for BM25 keyword search over memory content.
//! Used in hybrid search: vector search finds semantic matches, FTS5 finds
//! exact keyword matches (URLs, error codes, proper nouns). Results are fused
//! via Reciprocal Rank Fusion (see [`crate::search`]).
//! Kept in sync with `memories` via triggers (insert/update/delete).
//!
//! ### `tags`
//! Fixed vocabulary of memory categories. Enforced by the [`Tag`] enum —
//! new tags require adding a variant. Seeded on startup.
//!
//! ### `memory_tags`
//! Junction table linking memories to tags (many-to-many). Enables filtering
//! like "find all tool-related memories" without string parsing.
//! CASCADE delete: removing a memory removes its tag associations.
//!
//! ### `sessions`
//! Conversation history — one row per conversation entry. Columns map to
//! the main crate's `EntryKind` variants (user, assistant, tool, system_prompt,
//! etc.). Enables granular queries like "find all user messages in this chat"
//! or "count tool calls per session".
//!
//! ### `chat_settings`
//! Per-chat overrides (model, temperature, provider) stored as JSON blobs.
//! Survives agent restarts. Empty settings = row deleted.

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

    // -- sessions: one row per conversation entry --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_key     TEXT    NOT NULL,     -- 'telegram:123', 'slack:general'
            entry_kind   TEXT    NOT NULL,     -- 'user', 'assistant', 'tool', 'system_prompt', etc.
            content      TEXT,                 -- message text / tool output
            tool_calls   TEXT,                 -- JSON array of tool calls (assistant entries)
            tool_call_id TEXT,                 -- tool call ID (tool result entries)
            tool_name    TEXT,                 -- tool name (tool result entries)
            metadata     TEXT,                 -- JSON for overflow fields (parts, memory id/score, etc.)
            turn_index   INTEGER NOT NULL DEFAULT 0,  -- ordering within a session
            created_at   INTEGER NOT NULL      -- unix epoch seconds
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_chat_key
            ON sessions (chat_key, turn_index);",
    )?;

    // -- chat_settings: per-chat JSON overrides (model, temperature, provider) --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chat_settings (
            chat_key TEXT PRIMARY KEY,
            settings TEXT NOT NULL              -- JSON blob of ChatSettings
        );",
    )?;

    // -- users: registered connect users --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            created_at INTEGER NOT NULL
        );",
    )?;

    // -- user_api_keys: multiple API keys per user --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS user_api_keys (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key_hash   TEXT    NOT NULL UNIQUE,
            prefix     TEXT    NOT NULL,
            created_at INTEGER NOT NULL,
            private    INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_hash
            ON user_api_keys(key_hash);",
    )?;

    // Migration: add `private` column if missing (existing installs)
    let has_private: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('user_api_keys') WHERE name = 'private'")?
        .query_row([], |row| row.get::<_, i64>(0))
        .unwrap_or(0)
        > 0;
    if !has_private {
        conn.execute_batch(
            "ALTER TABLE user_api_keys ADD COLUMN private INTEGER NOT NULL DEFAULT 0;",
        )?;
    }

    // -- user_sessions: per-user session metadata for cross-device sync --
    // Display data stored as files in ~/.flashagent/sessions/{username}/{key}.jsonl
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS user_sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            session_key  TEXT    NOT NULL,
            last_prompt  TEXT,
            model        TEXT,
            cwd          TEXT,
            updated_at   INTEGER NOT NULL,
            UNIQUE(user_id, session_key)
        );",
    )?;

    // -- local_sessions: client-side session metadata for the session picker --
    // Replaces connect_sessions.jsonl
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS local_sessions (
            key           TEXT PRIMARY KEY,
            prompt        TEXT,
            cwd           TEXT,
            model         TEXT,
            updated_at    INTEGER NOT NULL DEFAULT 0
        );",
    )?;

    // Migration: rename last_accessed → updated_at for existing installs
    let has_last_accessed: bool = conn
        .prepare(
            "SELECT COUNT(*) FROM pragma_table_info('local_sessions') WHERE name = 'last_accessed'",
        )?
        .query_row([], |row| row.get::<_, i64>(0))
        .unwrap_or(0)
        > 0;
    if has_last_accessed {
        conn.execute_batch(
            "ALTER TABLE local_sessions RENAME COLUMN last_accessed TO updated_at;",
        )?;
    }

    // Migration: add title column if it doesn't exist
    let has_title: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('local_sessions') WHERE name = 'title'")?
        .query_row([], |row| row.get::<_, i64>(0))
        .unwrap_or(0)
        > 0;
    if !has_title {
        conn.execute_batch("ALTER TABLE local_sessions ADD COLUMN title TEXT;")?;
    }

    // Migration: add mode column if it doesn't exist
    let has_mode: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('local_sessions') WHERE name = 'mode'")?
        .query_row([], |row| row.get::<_, i64>(0))
        .unwrap_or(0)
        > 0;
    if !has_mode {
        conn.execute_batch("ALTER TABLE local_sessions ADD COLUMN mode TEXT;")?;
    }

    // -- shared_sessions: token-based session sharing between users --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS shared_sessions (
            token         TEXT    PRIMARY KEY,
            user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            source_key    TEXT    NOT NULL,
            last_prompt   TEXT,
            model         TEXT,
            created_at    INTEGER NOT NULL,
            expires_at    INTEGER NOT NULL,
            max_pulls     INTEGER,
            pull_count    INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_shared_sessions_user
            ON shared_sessions (user_id);
        CREATE INDEX IF NOT EXISTS idx_shared_sessions_expires
            ON shared_sessions (expires_at);",
    )?;

    // -- user_identities: canonical cross-channel user identity --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS user_identities (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    NOT NULL UNIQUE,  -- canonical lowercase username
            display_name TEXT,                     -- human-readable name
            email        TEXT,                     -- for identity matching
            created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );",
    )?;

    // -- user_channels: links platform accounts to a user identity --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS user_channels (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL REFERENCES user_identities(id) ON DELETE CASCADE,
            channel         TEXT    NOT NULL,   -- 'slack', 'telegram', 'connect', etc.
            channel_user_id TEXT    NOT NULL,   -- platform-specific ID
            metadata        TEXT,               -- JSON: display_name, avatar, etc.
            linked_by       TEXT    NOT NULL,   -- 'admin', 'identity_agent', 'self'
            confidence      REAL    NOT NULL DEFAULT 1.0,
            linked_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(channel, channel_user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_user_channels_user
            ON user_channels (user_id);",
    )?;

    // -- oauth_tokens: per-user OAuth tokens for MCP servers --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS oauth_tokens (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL REFERENCES user_identities(id) ON DELETE CASCADE,
            mcp_server   TEXT    NOT NULL,
            provider     TEXT    NOT NULL,
            access_token TEXT    NOT NULL,
            refresh_token TEXT,
            expires_at   TEXT,
            scopes       TEXT    NOT NULL,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(user_id, mcp_server)
        );",
    )?;

    // -- user_oauth_providers: per-user OAuth app credentials --
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS user_oauth_providers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES user_identities(id) ON DELETE CASCADE,
            provider      TEXT    NOT NULL,
            client_id     TEXT    NOT NULL,
            client_secret TEXT    NOT NULL,
            auth_url      TEXT,
            token_url     TEXT,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(user_id, provider)
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
