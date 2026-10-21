use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use flashmind_core::agent::DEFAULT_KEEP_RECENT_TOKENS;
use flashmind_core::compaction::try_compact;
use flashmind_core::{CancellationToken, Conversation, ConversationEntry, handle_llm_error};
use flashmind_types::{
    AgentEvent, CompletionRequest, CompletionStream, LlmProvider, Model, Provider, StreamEvent,
    ToolCall,
};
use futures::StreamExt;

#[derive(Default)]
struct SummaryProvider {
    calls: AtomicUsize,
    empty_calls: usize,
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
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call < self.empty_calls {
            return Box::pin(futures::stream::empty());
        }
        Box::pin(futures::stream::iter([Ok(StreamEvent::ContentDelta(
            "Conversation summary.".into(),
        ))]))
    }
}

#[tokio::test]
async fn context_overflow_recovery_reports_compaction() {
    struct Case {
        attempt: u8,
        empty_calls: usize,
        with_tools: bool,
        expected: &'static str,
    }
    for case in [
        Case {
            attempt: 1,
            empty_calls: 0,
            with_tools: false,
            expected: "Conversation summary.",
        },
        Case {
            attempt: 1,
            empty_calls: 1,
            with_tools: true,
            expected: "Conversation summary.",
        },
        Case {
            attempt: 1,
            empty_calls: usize::MAX,
            with_tools: true,
            expected: "Context reduced without a summary.",
        },
        Case {
            attempt: 1,
            empty_calls: usize::MAX,
            with_tools: false,
            expected: "Only the last exchange was kept; no summary was available.",
        },
        Case {
            attempt: 2,
            empty_calls: 0,
            with_tools: false,
            expected: "Only the last exchange was kept; no summary was available.",
        },
    ] {
        let provider = SummaryProvider {
            empty_calls: case.empty_calls,
            ..Default::default()
        };
        let model: Model = "ollama:test".parse().expect("The test model is valid.");
        let mut conversation = conversation(8, case.with_tools);
        conversation.add(ConversationEntry::assistant(
            "Recent context. ".repeat(6_000),
        ));
        let cancel = CancellationToken::new();
        let events: Vec<_> =
            handle_llm_error(&mut conversation, &provider, &model, &cancel, case.attempt)
                .collect()
                .await;
        let summaries: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::Compacted(summary) => Some(summary.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(summaries, [case.expected]);
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
