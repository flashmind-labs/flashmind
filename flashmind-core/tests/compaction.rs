use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use flashmind_core::agent::DEFAULT_KEEP_RECENT_TOKENS;
use flashmind_core::compaction::try_compact;
use flashmind_core::{Conversation, ConversationEntry};
use flashmind_types::{
    AgentEvent, CompletionRequest, CompletionStream, LlmProvider, Model, Provider, StreamEvent,
    ToolCall,
};
use futures::StreamExt;

#[derive(Default)]
struct SummaryProvider {
    calls: AtomicUsize,
}

#[async_trait]
impl LlmProvider for SummaryProvider {
    fn name(&self) -> &str {
        "test"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(futures::stream::iter([Ok(StreamEvent::ContentDelta(
            "Conversation summary.".into(),
        ))]))
    }
}

fn conversation(turns: usize, with_tools: bool) -> Conversation {
    let mut conversation = Conversation::with_system("Test system prompt.");
    for turn in 0..turns {
        conversation.add(ConversationEntry::user(format!("Question {turn}")));
        if with_tools {
            let id = format!("call-{turn}");
            conversation.add(ConversationEntry::assistant_with_tool_calls(
                "Reading a file.",
                vec![ToolCall {
                    id: id.clone(),
                    name: "file_read".into(),
                    arguments: serde_json::json!({"path": "src/example.rs"}),
                }],
                None,
            ));
            conversation.add(ConversationEntry::tool(id, "File contents."));
        }
        conversation.add(ConversationEntry::assistant(format!("Answer {turn}")));
    }
    conversation
}

#[tokio::test]
async fn preserves_history_when_everything_is_within_recent_budget() {
    for turns in [2, 8] {
        for with_tools in [false, true] {
            let provider = SummaryProvider::default();
            let model: Model = "ollama:test".parse().expect("The test model is valid.");
            let mut conversation = conversation(turns, with_tools);
            let before = serde_json::to_value(conversation.entries())
                .expect("Test conversation entries are serializable.");

            let events: Vec<_> = try_compact(
                &mut conversation,
                128_000,
                128_000,
                DEFAULT_KEEP_RECENT_TOKENS,
                &provider,
                &model,
            )
            .collect()
            .await;

            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(
                serde_json::to_value(conversation.entries())
                    .expect("Test conversation entries are serializable."),
                before,
                "History must be preserved for {turns} turns with tools={with_tools}",
            );
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::Compacted(_)))
            );
        }
    }
}

#[tokio::test]
async fn summarizes_older_history_when_recent_budget_is_exceeded() {
    let provider = SummaryProvider::default();
    let model: Model = "ollama:test".parse().expect("The test model is valid.");
    let mut conversation = conversation(8, false);

    let events: Vec<_> = try_compact(&mut conversation, 128_000, 128_000, 1, &provider, &model)
        .collect()
        .await;

    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(conversation.entries().len(), 4);
    assert!(conversation.entries()[1].is_summary());
    assert_eq!(conversation.entries()[2].content(), "Question 7");
    assert_eq!(conversation.entries()[3].content(), "Answer 7");
    assert!(events.iter().any(
        |event| matches!(event, AgentEvent::Compacted(text) if text == "Conversation summary.")
    ));
}
