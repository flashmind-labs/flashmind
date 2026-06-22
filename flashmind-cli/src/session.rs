use std::path::PathBuf;

use anyhow::Result;

use flashmind_core::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::session::{self, SessionEntry, SessionEntryKind, SessionStore, SessionSummary};

use crate::config::config_dir;

async fn open_session_db(db_path: PathBuf) -> Result<SessionStore> {
    let conn = tokio_rusqlite::Connection::open(db_path).await?;
    conn.call(
        |c| -> std::result::Result<(), flashmind_memory::rusqlite::Error> {
            session::schema::init_session_schema(c)?;
            Ok(())
        },
    )
    .await?;
    Ok(SessionStore::new(conn))
}

pub async fn open_session_store() -> Result<SessionStore> {
    let cwd = std::env::current_dir()?;
    let dir_hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cwd.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    };
    let sessions_dir = config_dir()?.join("sessions").join(&dir_hash);
    std::fs::create_dir_all(&sessions_dir)?;
    open_session_db(sessions_dir.join("sessions.db")).await
}

/// Open every session store across all working directories.
/// Returns `(store, sessions)` pairs so the caller can load from the right store.
pub async fn list_all_sessions() -> Result<Vec<(SessionStore, Vec<SessionSummary>)>> {
    let sessions_root = config_dir()?.join("sessions");
    if !sessions_root.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    let entries = std::fs::read_dir(&sessions_root)?;
    for entry in entries {
        let entry = entry?;
        let db_path = entry.path().join("sessions.db");
        if !db_path.exists() {
            continue;
        }
        let store = match open_session_db(db_path).await {
            Ok(s) => s,
            Err(_) => continue,
        };
        let sessions = store.list_sessions().await.unwrap_or_default();
        if !sessions.is_empty() {
            results.push((store, sessions));
        }
    }

    Ok(results)
}

pub fn new_session_key() -> String {
    format!("session_{}", uuid::Uuid::new_v4().as_simple())
}

pub fn conv_to_session(entry: &ConversationEntry, chat_key: &str, turn_index: i64) -> SessionEntry {
    let (kind, tool_calls, tool_call_id, metadata) = match &entry.kind {
        EntryKind::SystemPrompt(_) => (SessionEntryKind::SystemPrompt, None, None, None),
        EntryKind::Developer { tag, metadata, .. } => (
            SessionEntryKind::Developer { tag: tag.clone() },
            None,
            None,
            metadata.clone(),
        ),
        EntryKind::User { .. } => (SessionEntryKind::User, None, None, None),
        EntryKind::Assistant {
            tool_calls,
            reasoning,
            ..
        } => {
            let tc = tool_calls
                .as_ref()
                .and_then(|v| serde_json::to_value(v).ok());
            let meta = reasoning
                .as_ref()
                .map(|r| serde_json::json!({ "reasoning": r }));
            (SessionEntryKind::Assistant, tc, None, meta)
        }
        EntryKind::Tool { call_id, .. } => {
            (SessionEntryKind::Tool, None, Some(call_id.clone()), None)
        }
    };

    SessionEntry {
        id: 0,
        chat_key: chat_key.to_string(),
        entry_kind: kind,
        content: entry.content().to_string(),
        tool_calls,
        tool_call_id,
        tool_name: None,
        metadata,
        turn_index,
        created_at: entry.timestamp.timestamp(),
    }
}

pub fn session_to_conv(entry: &SessionEntry) -> ConversationEntry {
    let ts = chrono::DateTime::from_timestamp(entry.created_at, 0).unwrap_or_else(chrono::Utc::now);

    let kind = match &entry.entry_kind {
        SessionEntryKind::SystemPrompt => EntryKind::SystemPrompt(entry.content.clone()),
        SessionEntryKind::Developer { tag } => EntryKind::Developer {
            content: entry.content.clone(),
            tag: tag.clone(),
            metadata: entry.metadata.clone(),
        },
        SessionEntryKind::User => EntryKind::User {
            content: entry.content.clone(),
            parts: None,
        },
        SessionEntryKind::Assistant => {
            let tool_calls = entry
                .tool_calls
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            let reasoning = entry
                .metadata
                .as_ref()
                .and_then(|m| m.get("reasoning"))
                .and_then(|v| v.as_str())
                .map(String::from);
            EntryKind::Assistant {
                content: entry.content.clone(),
                tool_calls,
                reasoning,
            }
        }
        SessionEntryKind::Tool => EntryKind::Tool {
            call_id: entry.tool_call_id.clone().unwrap_or_default(),
            output: entry.content.clone(),
        },
    };

    ConversationEntry {
        kind,
        timestamp: ts,
    }
}

pub async fn load_conversation_from(
    store: &SessionStore,
    chat_key: &str,
    system_prompt: &str,
) -> Result<Conversation> {
    let mut conversation = Conversation::new();
    conversation.set_system(system_prompt);

    let entries = store.load(chat_key).await?;
    for entry in &entries {
        if matches!(entry.entry_kind, SessionEntryKind::SystemPrompt) {
            continue;
        }
        conversation.add(session_to_conv(entry));
    }

    Ok(conversation)
}

pub async fn save_turn(
    store: &SessionStore,
    chat_key: &str,
    conversation: &Conversation,
) -> Result<()> {
    let entries: Vec<SessionEntry> = conversation
        .entries()
        .iter()
        .enumerate()
        .map(|(i, e)| conv_to_session(e, chat_key, i as i64))
        .collect();

    store.rewrite(chat_key, &entries).await?;
    Ok(())
}

pub fn format_session_age(unix_ts: i64) -> String {
    let now = chrono::Utc::now().timestamp();
    let secs = (now - unix_ts).max(0);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}
