use anyhow::Result;

use flashmind_core::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::session::{self, SessionEntry, SessionEntryKind, SessionStore};

use crate::config::config_dir;

pub const CHAT_KEY: &str = "default";

pub async fn open_session_store() -> Result<SessionStore> {
    let db_path = config_dir()?.join("sessions.db");
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

pub fn conv_to_session(entry: &ConversationEntry, turn_index: i64) -> SessionEntry {
    let (kind, tool_calls, tool_call_id) = match &entry.kind {
        EntryKind::SystemPrompt(_) => (SessionEntryKind::SystemPrompt, None, None),
        EntryKind::Developer { tag, .. } => {
            (SessionEntryKind::Developer { tag: tag.clone() }, None, None)
        }
        EntryKind::User { .. } => (SessionEntryKind::User, None, None),
        EntryKind::Assistant { tool_calls, .. } => {
            let tc = tool_calls
                .as_ref()
                .and_then(|v| serde_json::to_value(v).ok());
            (SessionEntryKind::Assistant, tc, None)
        }
        EntryKind::Tool { call_id, .. } => (SessionEntryKind::Tool, None, Some(call_id.clone())),
    };

    SessionEntry {
        id: 0,
        chat_key: CHAT_KEY.to_string(),
        entry_kind: kind,
        content: entry.content().to_string(),
        tool_calls,
        tool_call_id,
        tool_name: None,
        metadata: match &entry.kind {
            EntryKind::Developer { metadata, .. } => metadata.clone(),
            _ => None,
        },
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
            EntryKind::Assistant {
                content: entry.content.clone(),
                tool_calls,
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

pub async fn load_conversation(
    store: &SessionStore,
    system_prompt: &str,
    restore: bool,
) -> Result<Conversation> {
    let mut conversation = Conversation::new();
    conversation.set_system(system_prompt);

    if !restore {
        return Ok(conversation);
    }

    let entries = store.load(CHAT_KEY).await?;
    if entries.is_empty() {
        return Ok(conversation);
    }

    for entry in &entries {
        if matches!(entry.entry_kind, SessionEntryKind::SystemPrompt) {
            continue;
        }
        conversation.add(session_to_conv(entry));
    }

    Ok(conversation)
}

pub async fn save_turn(store: &SessionStore, conversation: &Conversation) -> Result<()> {
    let entries: Vec<SessionEntry> = conversation
        .entries()
        .iter()
        .enumerate()
        .map(|(i, e)| conv_to_session(e, i as i64))
        .collect();

    store.rewrite(CHAT_KEY, &entries).await?;
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
