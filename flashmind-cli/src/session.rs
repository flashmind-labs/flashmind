use std::path::{Path, PathBuf};

use anyhow::Result;

use flashmind_core::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::session::{SessionEntry, SessionEntryKind, SessionStore, SessionSummary};

use crate::paths::data_dir;

pub(crate) async fn open_session_directory(directory: PathBuf) -> Result<SessionStore> {
    SessionStore::open(directory)
}

pub(crate) async fn open_ephemeral_store() -> Result<SessionStore> {
    Ok(SessionStore::in_memory())
}

pub async fn open_session_store() -> Result<SessionStore> {
    let cwd = std::env::current_dir()?;
    open_session_store_for(&cwd).await
}

pub async fn open_session_store_for(cwd: &Path) -> Result<SessionStore> {
    let dir_hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cwd.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    };
    let sessions_dir = data_dir()?.join("sessions").join(&dir_hash);
    std::fs::create_dir_all(&sessions_dir)?;
    open_session_directory(sessions_dir).await
}

/// Open every session store across all working directories.
/// Returns `(store, sessions)` pairs so the caller can load from the right store.
pub async fn list_all_sessions() -> Result<Vec<(SessionStore, Vec<SessionSummary>)>> {
    let sessions_root = data_dir()?.join("sessions");
    if !sessions_root.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    let entries = std::fs::read_dir(&sessions_root)?;
    for entry in entries {
        let entry = entry?;
        let directory = entry.path();
        if !directory.is_dir() || entry.file_name() == "display" {
            continue;
        }
        let store = match open_session_directory(directory).await {
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
        EntryKind::User { parts, .. } => (
            SessionEntryKind::User,
            None,
            None,
            parts
                .as_ref()
                .map(|parts| serde_json::json!({ "parts": parts })),
        ),
        EntryKind::Assistant {
            tool_calls,
            reasoning,
            reasoning_details,
            reasoning_model,
            reasoning_prefix,
            ..
        } => {
            let tc = tool_calls
                .as_ref()
                .and_then(|v| serde_json::to_value(v).ok());
            let meta =
                (reasoning.is_some() || !reasoning_details.is_empty() || reasoning_model.is_some())
                    .then(|| {
                        serde_json::json!({
                            "reasoning": reasoning,
                            "reasoning_details": reasoning_details,
                            "reasoning_model": reasoning_model,
                            "reasoning_prefix": reasoning_prefix,
                        })
                    });
            (SessionEntryKind::Assistant, tc, None, meta)
        }
        EntryKind::Tool { call_id, parts, .. } => (
            SessionEntryKind::Tool,
            None,
            Some(call_id.clone()),
            (!parts.is_empty()).then(|| serde_json::json!({ "parts": parts })),
        ),
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
            parts: entry
                .metadata
                .as_ref()
                .and_then(|m| m.get("parts"))
                .and_then(|p| serde_json::from_value(p.clone()).ok()),
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
                reasoning_details: entry
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("reasoning_details"))
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default(),
                reasoning_model: entry
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("reasoning_model"))
                    .and_then(|v| serde_json::from_value(v.clone()).ok()),
                reasoning_prefix: entry
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("reasoning_prefix"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
            }
        }
        SessionEntryKind::Tool => EntryKind::Tool {
            call_id: entry.tool_call_id.clone().unwrap_or_default(),
            output: entry.content.clone(),
            parts: entry
                .metadata
                .as_ref()
                .and_then(|m| m.get("parts"))
                .and_then(|p| serde_json::from_value(p.clone()).ok())
                .unwrap_or_default(),
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
    format_session_age_at(unix_ts, chrono::Utc::now())
}

/// Format a session's age relative to `now`. Recent sessions use a relative
/// label (`just now`, `Nm ago`, `Nh ago`, `Nd ago` up to a week); older
/// sessions fall back to an absolute date (`Jun 17`, or `Jun 17 2025` when the
/// year differs).
fn format_session_age_at(unix_ts: i64, now: chrono::DateTime<chrono::Utc>) -> String {
    use chrono::{Datelike, TimeZone};

    let now_ts = now.timestamp();
    let secs = (now_ts - unix_ts).max(0);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else if secs < 7 * 86400 {
        format!("{}d ago", secs / 86400)
    } else {
        let dt = chrono::Local.timestamp_opt(unix_ts, 0).single();
        match dt {
            Some(dt) if dt.year() == now.with_timezone(&chrono::Local).year() => {
                dt.format("%b %-d").to_string()
            }
            Some(dt) => dt.format("%b %-d %Y").to_string(),
            None => format!("{}d ago", secs / 86400),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn ephemeral_sessions_are_not_available_to_a_new_store() {
        let store = open_ephemeral_store().await.unwrap();
        let mut conversation = Conversation::with_system("Instructions");
        conversation.add(ConversationEntry::user("Private prompt"));
        save_turn(&store, "example", &conversation).await.unwrap();
        assert_eq!(store.list_sessions().await.unwrap().len(), 1);
        drop(store);

        let fresh = open_ephemeral_store().await.unwrap();
        assert!(fresh.list_sessions().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn native_compaction_checkpoints_survive_database_reload_without_losing_the_transcript() {
        use async_trait::async_trait;
        use flashmind_types::{
            CompletionRequest, CompletionStream, LlmProvider, NativeCompaction,
            NativeReasoningFormat, Provider, ReasoningDetails, TokenUsage,
        };
        struct NativeProvider(NativeReasoningFormat);
        #[async_trait]
        impl LlmProvider for NativeProvider {
            fn name(&self) -> &str {
                "native"
            }
            fn provider(&self) -> Provider {
                Provider::OpenAi
            }
            fn complete(&self, _: CompletionRequest) -> CompletionStream {
                panic!("native compaction must not use local summarization")
            }
            async fn compact(
                &self,
                _: CompletionRequest,
                _: &str,
            ) -> Result<Option<NativeCompaction>> {
                let block = match self.0 {
                    NativeReasoningFormat::Responses => {
                        serde_json::json!({"type":"compaction","encrypted_content":"opaque","extra":"keep"})
                    }
                    NativeReasoningFormat::Anthropic => {
                        serde_json::json!({"type":"compaction","content":"summary","signature":"signed-summary"})
                    }
                    NativeReasoningFormat::Mistral => unreachable!(),
                };
                Ok(Some(NativeCompaction {
                    context: ReasoningDetails::native(self.0, vec![block]),
                    summary: "native checkpoint".into(),
                    usage: TokenUsage::default(),
                }))
            }
        }
        for (format, model) in [
            (NativeReasoningFormat::Responses, "openai:example-model"),
            (NativeReasoningFormat::Anthropic, "anthropic:example-model"),
        ] {
            let model = model.parse().unwrap();
            let directory = tempfile::tempdir().unwrap();
            let database = directory.path().join("sessions");
            let store = open_session_directory(database.clone()).await.unwrap();
            let mut conversation = Conversation::with_system("original instructions");
            conversation.add(ConversationEntry::user("original task"));
            conversation.add(ConversationEntry::assistant("original answer"));
            conversation
                .compact_with_llm_keeping_tokens(&NativeProvider(format), &model, 0)
                .await
                .unwrap()
                .unwrap();
            let expected = conversation.to_messages_for_model(&model)[1]
                .reasoning_details
                .clone();
            save_turn(&store, "example", &conversation).await.unwrap();
            drop(store);
            let store = open_session_directory(database).await.unwrap();
            let restored = load_conversation_from(&store, "example", "original instructions")
                .await
                .unwrap();
            let messages = restored.to_messages_for_model(&model);
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[1].reasoning_details, expected);
            let other =
                restored.to_messages_for_model(&"openrouter:example/other-model".parse().unwrap());
            assert_eq!(
                other
                    .iter()
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>(),
                ["original instructions", "original task", "original answer"]
            );
        }
    }

    #[tokio::test]
    async fn reasoning_state_survives_saving_and_reopening_a_session() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("sessions");
        let store = open_session_directory(database.clone()).await.unwrap();
        let model = "openrouter:example/model".parse().unwrap();
        let details = serde_json::json!([
            {"type":"reasoning.text","text":"Check facts","signature":"signature","index":0},
            {"type":"reasoning.encrypted","data":"opaque","index":1,"extra":{"keep":true}}
        ]);
        let mut original =
            ConversationEntry::assistant_with_reasoning("answer", Some("Check facts".into()))
                .with_reasoning_details(serde_json::from_value(details.clone()).unwrap(), model);
        if let EntryKind::Assistant {
            reasoning_prefix, ..
        } = &mut original.kind
        {
            *reasoning_prefix = Some("example-fingerprint".into());
        }
        let mut conversation = Conversation::with_system("Example instructions.");
        conversation.add(ConversationEntry::user("Example question."));
        conversation.add(original.clone());
        save_turn(&store, "example", &conversation).await.unwrap();
        drop(store);
        let store = open_session_directory(database).await.unwrap();
        let restored = load_conversation_from(&store, "example", "Example instructions.")
            .await
            .unwrap();
        let entry = restored
            .entries()
            .iter()
            .find(|entry| entry.is_assistant())
            .unwrap();
        assert_eq!(
            serde_json::to_value(&entry.kind).unwrap(),
            serde_json::to_value(&original.kind).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&entry.to_message().reasoning_details).unwrap(),
            details
        );
    }

    #[test]
    fn old_plaintext_reasoning_sessions_still_load() {
        let original =
            ConversationEntry::assistant_with_reasoning("answer", Some("thinking".into()));
        let mut stored = conv_to_session(&original, "example", 0);
        stored.metadata = Some(serde_json::json!({"reasoning":"thinking"}));
        let restored = session_to_conv(&stored).to_message();
        assert_eq!(restored.reasoning.as_deref(), Some("thinking"));
        assert!(restored.reasoning_details.is_empty());
    }

    #[tokio::test]
    async fn selected_mode_survives_saving_and_reopening_a_session() {
        use flashmind_types::InteractionMode;
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("sessions");
        let store = open_session_directory(database.clone()).await.unwrap();
        let mut conversation = Conversation::with_system("Example instructions.");
        conversation.set_interaction_mode(InteractionMode::Read);
        conversation.add(ConversationEntry::user("Example question."));
        save_turn(&store, "example", &conversation).await.unwrap();
        drop(store);
        let store = open_session_directory(database).await.unwrap();
        let restored = load_conversation_from(&store, "example", "Example instructions.")
            .await
            .unwrap();
        assert_eq!(restored.interaction_mode(), InteractionMode::Read);
    }

    #[tokio::test]
    async fn tool_images_survive_saving_and_reopening_a_session() {
        use flashmind_types::{ContentPart, ToolCall};

        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("sessions");
        let store = open_session_directory(database.clone()).await.unwrap();
        let mut conversation = Conversation::with_system("Example instructions.");
        conversation.add(ConversationEntry::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call".into(),
                name: "image_read".into(),
                arguments: serde_json::json!({}),
            }],
            None,
        ));
        conversation.add(ConversationEntry::tool_with_parts(
            "call",
            "Image loaded",
            vec![ContentPart::Image {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
            }],
        ));
        save_turn(&store, "example", &conversation).await.unwrap();
        drop(store);
        let store = open_session_directory(database).await.unwrap();
        let restored = load_conversation_from(&store, "example", "Example instructions.")
            .await
            .unwrap();
        let messages = restored.to_messages();
        assert!(messages.iter().any(|message| matches!(message.parts.as_deref(), Some([ContentPart::Image { media_type, data }]) if media_type == "image/png" && data == "aW1hZ2U=")));
    }

    #[test]
    fn old_tool_entries_load_without_parts() {
        let mut stored = conv_to_session(&ConversationEntry::tool("call", "output"), "example", 0);
        stored.metadata = None;
        let entry = session_to_conv(&stored);
        assert!(matches!(entry.kind, EntryKind::Tool { parts, .. } if parts.is_empty()));
    }

    #[test]
    fn age_relative_branches() {
        let now = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let now_ts = now.timestamp();
        assert_eq!(format_session_age_at(now_ts - 10, now), "just now");
        assert_eq!(format_session_age_at(now_ts - 120, now), "2m ago");
        assert_eq!(format_session_age_at(now_ts - 7200, now), "2h ago");
        assert_eq!(format_session_age_at(now_ts - 2 * 86400, now), "2d ago");
    }

    #[test]
    fn age_absolute_for_old_sessions() {
        let now = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let now_ts = now.timestamp();
        // Older than a week -> absolute date (format depends on local tz, but
        // must not be a relative "ago" label).
        let s = format_session_age_at(now_ts - 30 * 86400, now);
        assert!(!s.ends_with("ago"), "expected absolute date, got {s}");
        assert!(!s.is_empty());
    }
    #[tokio::test]
    async fn pinned_directory_survives_reopening_the_session_store() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("sessions");
        let worktree = directory.path().join(".worktrees/task");
        tokio::fs::create_dir_all(&worktree).await.unwrap();
        {
            let store = open_session_directory(database.clone()).await.unwrap();
            let mut conversation = Conversation::new();
            conversation.add(ConversationEntry::user("Example task"));
            save_turn(&store, "example-session", &conversation)
                .await
                .unwrap();
            store
                .save_meta(
                    "example-session",
                    Some("Example"),
                    Some("provider:model"),
                    directory.path().to_str(),
                )
                .await
                .unwrap();
            store
                .update_working_dir("example-session", &worktree)
                .await
                .unwrap();
        }
        let reopened = open_session_directory(database).await.unwrap();
        let sessions = reopened.list_sessions().await.unwrap();
        let summary = &sessions[0];
        assert_eq!(summary.working_dir.as_deref(), worktree.to_str());
        assert_eq!(summary.title.as_deref(), Some("Example"));
        let restored = load_conversation_from(&reopened, &summary.chat_key, "Example instructions")
            .await
            .unwrap();
        assert!(
            restored
                .entries()
                .iter()
                .any(|entry| entry.is_user() && entry.content() == "Example task")
        );
    }
}
