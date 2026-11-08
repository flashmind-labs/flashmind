#![cfg(feature = "session-sqlite")]

use flashmind_memory::{SessionEntry, SessionEntryKind, SqliteSessionStore};
use flashmind_types::{Model, ReasoningLevel};

#[tokio::test]
async fn sqlite_session_public_api_persists_and_branches_history() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("history/sessions.db");
    let store = SqliteSessionStore::open(&path).await.unwrap();
    let entry = SessionEntry {
        id: 0,
        chat_key: "example".into(),
        entry_kind: SessionEntryKind::User,
        content: "question".into(),
        tool_calls: None,
        tool_call_id: None,
        tool_name: None,
        metadata: None,
        turn_index: 0,
        created_at: 1,
    };
    store.save_entry(&entry).await.unwrap();
    store
        .save_meta("example", Some("Title"), None, None)
        .await
        .unwrap();
    let model: Model = "ollama:model".parse().unwrap();
    store.update_model("example", &model).await.unwrap();
    store
        .update_reasoning("example", ReasoningLevel::High)
        .await
        .unwrap();
    drop(store);

    let store = SqliteSessionStore::open(&path).await.unwrap();
    let mut entries = store.load("example").await.unwrap();
    assert_eq!(entries[0].content, "question");
    assert_eq!(
        store.reasoning("example").await.unwrap(),
        Some(ReasoningLevel::High)
    );
    let summaries = store.list_sessions().await.unwrap();
    assert_eq!(summaries[0].title.as_deref(), Some("Title"));
    assert_eq!(summaries[0].model.as_deref(), Some("ollama:model"));

    entries[0].content = "edited question".into();
    store.rewrite("example", &entries).await.unwrap();
    assert_eq!(store.branch("example", "copy").await.unwrap(), 1);
    assert_eq!(store.delete_session("example").await.unwrap(), 1);
    store.delete_meta("example").await.unwrap();
    assert!(store.load("example").await.unwrap().is_empty());
    assert_eq!(store.reasoning("example").await.unwrap(), None);
    assert_eq!(
        store.load("copy").await.unwrap()[0].content,
        "edited question"
    );
}
