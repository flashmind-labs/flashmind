//! Session persistence via SQLite (flash.db).
//!
//! Maps between the main crate's `ConversationEntry` / `EntryKind` and
//! flashmind_memory's `SessionEntry` row type. Each conversation entry becomes one
//! row in the `sessions` table with structured columns.


use flashmind_memory::{DbStore, SessionEntry};
use flashmind_types::{ContentPart, ToolCall};

use crate::conversation::{Conversation, ConversationEntry, EntryKind};

/// Session pruning configuration.
pub struct PruneConfig {
    /// Drop messages older than N days.
    pub max_age_days: Option<u64>,
}

/// Persists per-chat conversations in SQLite via flashmind_memory.
#[derive(Clone)]
pub struct SessionStore {
    vm: DbStore,
}

impl SessionStore {
    pub fn new(vm: DbStore) -> Self {
        Self { vm }
    }

    /// Returns the underlying SQLite connection for local_sessions operations.
    pub fn connection(&self) -> &flashmind_memory::tokio_rusqlite::Connection {
        self.vm.connection()
    }

    /// Returns true if this store has a valid backend.
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        // Check by doing a simple query - tokio_rusqlite doesn't have is_valid
        true
    }

    /// Append entries to a session (delta write). Use after each turn.
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

    /// Load a conversation from SQLite.
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

    /// List all persisted session scopes.
    pub async fn list(&self) -> anyhow::Result<Vec<String>> {
        let mut keys = flashmind_memory::session::list(self.vm.connection()).await?;
        keys.sort();
        Ok(keys)
    }

    /// Delete a session.
    pub async fn clear(&self, scope: &str) -> anyhow::Result<()> {
        flashmind_memory::session::delete(self.vm.connection(), scope).await?;
        tracing::debug!("session_store: cleared {}", scope);
        Ok(())
    }

    /// Rewrite a session with the given entries (e.g. after compaction).
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

    /// Prune a session by age.
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

    /// Find sessions where the last entry is from the user (interrupted conversations).
    pub async fn find_interrupted(&self) -> Vec<(String, String)> {
        flashmind_memory::session::find_interrupted(self.vm.connection())
            .await
            .unwrap_or_default()
    }

    /// Copy a session to a new key (for branching).
    pub async fn copy(&self, from: &str, to: &str) -> anyhow::Result<()> {
        flashmind_memory::session::copy(self.vm.connection(), from, to).await?;
        tracing::debug!("session_store: copied {} -> {}", from, to);
        Ok(())
    }

    /// Remove the last entry from a session.
    pub async fn pop_last(&self, scope: &str) -> anyhow::Result<()> {
        flashmind_memory::session::pop_last(self.vm.connection(), scope).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mapping: ConversationEntry ↔ SessionEntry
// ---------------------------------------------------------------------------

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
