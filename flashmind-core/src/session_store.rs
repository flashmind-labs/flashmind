//! Session persistence via SQLite (`flash.db`).
//!
//! Maps between the main crate's [`ConversationEntry`] / [`EntryKind`] and
//! flashmind_memory's [`SessionEntry`] row type. Each conversation entry becomes one
//! row in the `sessions` table with structured columns.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`SessionStore`] | High-level CRUD for per-chat conversations (append, load, list, clear, rewrite) |
//! | [`PruneConfig`] | Configuration for age-based session pruning |
//!
//! # Creating a store
//!
//! ```ignore
//! use flashmind_core::session_store::SessionStore;
//! use flashmind_memory::DbStore;
//!
//! let db = DbStore::open("path/to/flash.db").await?;
//! let store = SessionStore::new(db);
//! ```

use flashmind_memory::{DbStore, SessionEntry};
use flashmind_types::{ContentPart, ToolCall};

use crate::conversation::{Conversation, ConversationEntry, EntryKind};

/// Configuration for session pruning by age.
///
/// When applied via [`SessionStore::prune`], sessions whose last entry is older
/// than the configured threshold are permanently deleted from the database.
/// If `max_age_days` is `None`, pruning is a no-op and nothing is removed.
pub struct PruneConfig {
    /// Maximum age of a session in days before it is eligible for deletion.
    ///
    /// Set to `Some(n)` to prune sessions older than `n` days. Use `None` to
    /// disable age-based pruning entirely.
    pub max_age_days: Option<u64>,
}

/// Persists per-chat conversations in SQLite via flashmind_memory.
///
/// `SessionStore` is a thin, cloneable wrapper around a [`DbStore`] that provides
/// high-level CRUD operations (append, load, list, clear, rewrite) on individual
/// conversation sessions identified by a string *scope* (typically a chat ID).
///
/// Under the hood it maps the crate's domain types ([`ConversationEntry`],
/// [`EntryKind`]) to/from the storage-layer [`SessionEntry`] row type, storing
/// each entry as a single row in the `sessions` table.
///
/// # Creating a store
///
/// ```ignore
/// use flashmind_core::session_store::SessionStore;
/// use flashmind_memory::DbStore;
///
/// let db = DbStore::open("path/to/flash.db").await?;
/// let store = SessionStore::new(db);
/// ```
#[derive(Clone)]
pub struct SessionStore {
    vm: DbStore,
}

impl SessionStore {
    /// Creates a new [`SessionStore`] wrapping the given [`DbStore`].
    ///
    /// The `vm` parameter owns the SQLite connection; the returned store is
    /// cloneable and can be shared across tasks.
    pub fn new(vm: DbStore) -> Self {
        Self { vm }
    }

    /// Returns a reference to the underlying SQLite connection.
    ///
    /// Exposed for advanced operations (e.g., running custom SQL queries or
    /// performing transactional work) that go beyond the convenience methods on
    /// this store. Callers must respect any concurrency constraints imposed by
    /// `tokio_rusqlite`.
    pub fn connection(&self) -> &flashmind_memory::tokio_rusqlite::Connection {
        self.vm.connection()
    }

    /// Returns `true` if this store has a valid backend.
    ///
    /// Currently always returns `true` since `tokio_rusqlite::Connection` does not
    /// expose an `is_valid` method. May be used as a no-op health check placeholder
    /// for callers that need to verify the store is usable before performing I/O.
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        // Check by doing a simple query - tokio_rusqlite doesn't have is_valid
        true
    }

    /// Appends entries to an existing session (delta / incremental write).
    ///
    /// This is the primary write path: call it after each LLM turn to persist the
    /// new messages without overwriting previous data. Entries are converted from
    /// [`ConversationEntry`] to [`SessionEntry`] and inserted in order.
    ///
    /// # Parameters
    /// - `scope` — unique identifier for the chat/session (e.g., a chat ID).
    /// - `entries` — slice of conversation entries to append. If empty, this is a no-op.
    ///
    /// # Returns
    /// `Ok(())` on success; propagates the underlying database error otherwise.
    pub async fn append(&self, scope: &str, entries: &[ConversationEntry]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let session_entries: Vec<SessionEntry> = entries.iter().map(to_session_entry).collect();

        flashmind_memory::session::append(self.vm.connection(), scope, &session_entries).await?;

        tracing::debug!(
            "session_store: appended {} entries for {}",
            entries.len(),
            scope
        );
        Ok(())
    }

    /// Loads a full conversation from SQLite for the given scope.
    ///
    /// Reads all rows belonging to `scope`, converts them back from [`SessionEntry`]
    /// to [`ConversationEntry`], and returns them as a reconstructed [`Conversation`].
    ///
    /// # Parameters
    /// - `scope` — unique identifier of the chat/session to load.
    ///
    /// # Returns
    /// - `Ok(Some(conv))` if the session exists and contains entries.
    /// - `Ok(None)` if the session does not exist or has no entries.
    /// - `Err(e)` if the database read fails.
    pub async fn load(&self, scope: &str) -> anyhow::Result<Option<Conversation>> {
        let entries = flashmind_memory::session::load(self.vm.connection(), scope).await?;

        if entries.is_empty() {
            return Ok(None);
        }

        let mut conv = Conversation::new();
        for entry in &entries {
            if let Some(ce) = from_session_entry(entry) {
                conv.add(ce);
            }
        }

        tracing::debug!(
            "session_store: loaded {} entries for {}",
            conv.entries().len(),
            scope
        );
        Ok(Some(conv))
    }

    /// Lists all persisted session scopes (chat IDs) in the database.
    ///
    /// Returns a sorted list of unique scope strings so callers can enumerate
    /// every stored conversation, e.g., for building a chat-history sidebar.
    ///
    /// # Returns
    /// A lexicographically sorted `Vec<String>` of scope identifiers.
    pub async fn list(&self) -> anyhow::Result<Vec<String>> {
        let mut keys = flashmind_memory::session::list(self.vm.connection()).await?;
        keys.sort();
        Ok(keys)
    }

    /// Permanently deletes an entire session and all its entries.
    ///
    /// Use this to implement "delete chat" functionality. After clearing, a
    /// subsequent [`SessionStore::load`] for the same scope will return `None`.
    ///
    /// # Parameters
    /// - `scope` — unique identifier of the session to delete.
    ///
    /// # Returns
    /// `Ok(())` on success; propagates any database error otherwise.
    pub async fn clear(&self, scope: &str) -> anyhow::Result<()> {
        flashmind_memory::session::delete(self.vm.connection(), scope).await?;
        tracing::debug!("session_store: cleared {}", scope);
        Ok(())
    }

    /// Rewrites an entire session with the given entries (full replace).
    ///
    /// Unlike [`SessionStore::append`], this atomically replaces all existing rows
    /// for `scope` with the supplied slice. It is intended for compaction or
    /// rebuild scenarios where you want to materialise a trimmed conversation back
    /// into the database in a single operation.
    ///
    /// # Parameters
    /// - `scope` — unique identifier of the session to overwrite.
    /// - `entries` — complete list of entries that should constitute the session after rewrite.
    ///
    /// # Returns
    /// `Ok(())` on success; propagates any database error otherwise.
    pub async fn rewrite(&self, scope: &str, entries: &[ConversationEntry]) -> anyhow::Result<()> {
        let session_entries: Vec<SessionEntry> = entries.iter().map(to_session_entry).collect();
        flashmind_memory::session::save(self.vm.connection(), scope, &session_entries).await?;
        tracing::debug!(
            "session_store: rewrote {} entries for {}",
            entries.len(),
            scope
        );
        Ok(())
    }

    /// Prunes a session based on age-based configuration.
    ///
    /// Loads all entries for `scope` and checks whether the most recent entry is
    /// older than the configured threshold. If so, the entire session is deleted.
    /// This implements simple auto-cleanup of stale conversations.
    ///
    /// # Parameters
    /// - `scope` — unique identifier of the session to check.
    /// - `config` — pruning rules; if `max_age_days` is `None` this is a no-op.
    ///
    /// # Returns
    /// `Ok(())` regardless of whether anything was actually pruned.
    pub async fn prune(&self, scope: &str, config: &PruneConfig) -> anyhow::Result<()> {
        let Some(max_age_days) = config.max_age_days else {
            return Ok(());
        };

        let entries = flashmind_memory::session::load(self.vm.connection(), scope).await?;

        if entries.is_empty() {
            return Ok(());
        }

        let cutoff = chrono::Utc::now().timestamp() - (max_age_days as i64 * 86400);

        if let Some(last) = entries.last()
            && last.created_at < cutoff
        {
            flashmind_memory::session::delete(self.vm.connection(), scope).await?;
            tracing::debug!("session_store: pruned session for {} (too old)", scope);
        }

        Ok(())
    }

    /// Finds sessions where the last entry is from the user (interrupted conversations).
    ///
    /// Scans all sessions and returns those whose most recent message was sent by the
    /// user but never received an assistant reply. Useful for resuming or replaying
    /// unfinished chats.
    ///
    /// # Returns
    /// A `Vec` of `(scope, last_message_content)` pairs. Silently returns an empty vec
    /// on database errors (`unwrap_or_default`).
    pub async fn find_interrupted(&self) -> Vec<(String, String)> {
        flashmind_memory::session::find_interrupted(self.vm.connection())
            .await
            .unwrap_or_default()
    }

    /// Copies all entries from one session to another scope (for branching).
    ///
    /// Creates a new session under `to` that is an exact copy of the session at
    /// `from`. The original session is left untouched. Commonly used when
    /// forking a conversation into a new chat thread.
    ///
    /// # Parameters
    /// - `from` — source session scope identifier.
    /// - `to` — destination session scope identifier.
    ///
    /// # Returns
    /// `Ok(())` on success; propagates any database error otherwise.
    pub async fn copy(&self, from: &str, to: &str) -> anyhow::Result<()> {
        flashmind_memory::session::copy(self.vm.connection(), from, to).await?;
        tracing::debug!("session_store: copied {} -> {}", from, to);
        Ok(())
    }

    /// Removes the last entry from a session.
    ///
    /// Useful for implementing "undo last message" or reverting the most recent
    /// assistant reply before retrying with a different prompt.
    ///
    /// # Parameters
    /// - `scope` — unique identifier of the session to modify.
    ///
    /// # Returns
    /// `Ok(())` on success; propagates any database error otherwise.
    pub async fn pop_last(&self, scope: &str) -> anyhow::Result<()> {
        flashmind_memory::session::pop_last(self.vm.connection(), scope).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mapping: ConversationEntry ↔ SessionEntry
// ---------------------------------------------------------------------------

/// Converts a domain [`ConversationEntry`] into a storage-layer [`SessionEntry`].
///
/// Maps each [`EntryKind`] variant to its string tag and serialises rich fields
/// (tool calls, user content parts, memory metadata) as JSON strings in the
/// appropriate `SessionEntry` columns. The entry's timestamp is stored as a Unix
/// epoch integer.
fn to_session_entry(ce: &ConversationEntry) -> SessionEntry {
    let created_at = ce.timestamp.timestamp();

    match &ce.kind {
        EntryKind::SystemPrompt(s) => SessionEntry {
            entry_kind: "system_prompt".into(),
            content: Some(s.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at,
        },
        EntryKind::SystemMessage(s) => SessionEntry {
            entry_kind: "system_message".into(),
            content: Some(s.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at,
        },
        EntryKind::Reminder(s) => SessionEntry {
            entry_kind: "reminder".into(),
            content: Some(s.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at,
        },
        EntryKind::User { content, parts } => SessionEntry {
            entry_kind: "user".into(),
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: parts.as_ref().and_then(|p| serde_json::to_string(p).ok()),
            created_at,
        },
        EntryKind::Assistant {
            content,
            tool_calls,
        } => SessionEntry {
            entry_kind: "assistant".into(),
            content: Some(content.clone()),
            tool_calls: tool_calls
                .as_ref()
                .and_then(|tc| serde_json::to_string(tc).ok()),
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at,
        },
        EntryKind::Tool { call_id, output } => SessionEntry {
            entry_kind: "tool".into(),
            content: Some(output.clone()),
            tool_calls: None,
            tool_call_id: Some(call_id.clone()),
            tool_name: None,
            metadata: None,
            created_at,
        },
        EntryKind::Memory { content, id, score } => SessionEntry {
            entry_kind: "memory".into(),
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: serde_json::to_string(&serde_json::json!({"id": id, "score": score})).ok(),
            created_at,
        },
        EntryKind::SubagentProgress { id, content } => SessionEntry {
            entry_kind: "subagent_progress".into(),
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: serde_json::to_string(&serde_json::json!({"id": id})).ok(),
            created_at,
        },
        EntryKind::Summary(s) => SessionEntry {
            entry_kind: "summary".into(),
            content: Some(s.clone()),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: None,
            created_at,
        },
    }
}

/// Converts a storage-layer [`SessionEntry`] back into a domain [`ConversationEntry`].
///
/// Reverse of [`to_session_entry`]: reads the `entry_kind` string tag, deserialises
/// any JSON columns (tool calls, user content parts, memory metadata), and reconstructs
/// the original [`EntryKind`] variant. The Unix timestamp is converted back to a
/// [`chrono::DateTime`].
///
/// # Returns
/// - `Some(ConversationEntry)` when the entry kind is recognised.
/// - `None` if the `entry_kind` string is unrecognised (e.g., added by a newer
///   version of the crate). This silently drops unknown rows to prevent panics on
///   schema evolution.
fn from_session_entry(se: &SessionEntry) -> Option<ConversationEntry> {
    let kind = match se.entry_kind.as_str() {
        "system_prompt" => EntryKind::SystemPrompt(se.content.clone().unwrap_or_default()),
        "system_message" => EntryKind::SystemMessage(se.content.clone().unwrap_or_default()),
        "reminder" => EntryKind::Reminder(se.content.clone().unwrap_or_default()),
        "user" => {
            let parts: Option<Vec<ContentPart>> = se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_str(m).ok());
            EntryKind::User {
                content: se.content.clone().unwrap_or_default(),
                parts,
            }
        }
        "assistant" => {
            let tool_calls: Option<Vec<ToolCall>> = se
                .tool_calls
                .as_ref()
                .and_then(|tc| serde_json::from_str(tc).ok());
            EntryKind::Assistant {
                content: se.content.clone().unwrap_or_default(),
                tool_calls,
            }
        }
        "tool" => EntryKind::Tool {
            call_id: se.tool_call_id.clone().unwrap_or_default(),
            output: se.content.clone().unwrap_or_default(),
        },
        "memory" => {
            let meta: serde_json::Value = se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_str(m).ok())
                .unwrap_or_default();
            EntryKind::Memory {
                content: se.content.clone().unwrap_or_default(),
                id: meta["id"].as_str().unwrap_or("").to_string(),
                score: meta["score"].as_f64().unwrap_or(0.0),
            }
        }
        "subagent_progress" => {
            let meta: serde_json::Value = se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_str(m).ok())
                .unwrap_or_default();
            EntryKind::SubagentProgress {
                id: meta["id"].as_str().unwrap_or("").to_string(),
                content: se.content.clone().unwrap_or_default(),
            }
        }
        "summary" => EntryKind::Summary(se.content.clone().unwrap_or_default()),
        _ => return None,
    };

    Some(ConversationEntry {
        kind,
        timestamp: chrono::DateTime::from_timestamp(se.created_at, 0)
            .unwrap_or_else(chrono::Utc::now),
    })
}
