use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use flashmind_core::agent::DEFAULT_KEEP_RECENT_TOKENS;
use flashmind_core::compaction::try_compact;
use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry, handle_llm_error};
use flashmind_types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, InteractionMode,
    LlmProvider, Model, Provider, StreamEvent, TokenUsage, ToolCall,
};
use futures_util::StreamExt;

#[derive(Default)]
struct SummaryProvider {
    calls: AtomicUsize,
    empty_calls: usize,
    fail: bool,
    delay: Duration,
}

#[async_trait]
impl LlmProvider for SummaryProvider {
    fn name(&self) -> &str {
        "test"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    async fn context_window(&self, _model: &Model) -> Option<u32> {
        Some(128_000)
    }

    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Box::pin(futures_util::stream::iter([Err(anyhow::anyhow!(
                "Example summary failure"
            ))]));
        }
        if call < self.empty_calls {
            return Box::pin(futures_util::stream::empty());
        }
        let delay = self.delay;
        Box::pin(async_stream::stream! {
            tokio::time::sleep(delay).await;
            yield Ok(StreamEvent::ContentDelta("Conversation summary.".into()));
        })
    }
}

#[tokio::test(start_paused = true)]
async fn compaction_allows_five_minutes_and_preserves_history_on_timeout() {
    let model: Model = "ollama:test".parse().unwrap();
    for (delay_secs, succeeds) in [(180, true), (600, false)] {
        let provider = SummaryProvider {
            delay: Duration::from_secs(delay_secs),
            ..Default::default()
        };
        let mut conversation = conversation(8, true);
        let before = serde_json::to_value(&conversation).unwrap();
        let started = tokio::time::Instant::now();
        let events: Vec<_> = try_compact(&mut conversation, 128_000, 128_000, 1, &provider, &model)
            .collect()
            .await;
        assert_eq!(started.elapsed(), Duration::from_secs(delay_secs.min(300)));
        assert_eq!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Compacted(_))),
            succeeds,
            "{events:?}"
        );
        if !succeeds {
            assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::Error(message) if message.contains("compaction timed out after 300s")
            )));
        }
    }
}

#[tokio::test]
async fn failed_compaction_preserves_the_entire_conversation() {
    let model: Model = "ollama:test".parse().expect("The test model is valid.");
    for turns in [2, 8] {
        for with_tools in [false, true] {
            for fail in [false, true] {
                let provider = SummaryProvider {
                    empty_calls: usize::MAX,
                    fail,
                    ..Default::default()
                };
                let mut conversation = conversation(turns, with_tools);
                let before = serde_json::to_value(&conversation).unwrap();
                let events: Vec<_> =
                    try_compact(&mut conversation, 128_000, 128_000, 1, &provider, &model)
                        .collect()
                        .await;
                assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
                assert!(
                    events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::Error(_)))
                );
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::Compacted(_)))
                );
            }
        }
    }
}

#[tokio::test]
async fn context_overflow_recovery_only_reports_success_for_a_summary() {
    let model: Model = "ollama:test".parse().expect("The test model is valid.");
    for attempt in [1, 2] {
        for with_tools in [false, true] {
            for fail in [false, true] {
                for empty in [false, true] {
                    let provider = SummaryProvider {
                        empty_calls: if empty { usize::MAX } else { 0 },
                        fail,
                        ..Default::default()
                    };
                    let mut conversation = conversation(8, with_tools);
                    conversation.add(ConversationEntry::assistant(
                        "Recent context. ".repeat(6_000),
                    ));
                    let before = serde_json::to_value(&conversation).unwrap();
                    let cancel = CancellationToken::new();
                    let events: Vec<_> =
                        handle_llm_error(&mut conversation, &provider, &model, &cancel, attempt)
                            .collect()
                            .await;
                    if fail || empty {
                        assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
                        assert!(
                            !events
                                .iter()
                                .any(|event| matches!(event, AgentEvent::Compacted(_)))
                        );
                        assert!(
                            events
                                .iter()
                                .any(|event| matches!(event, AgentEvent::Error(_)))
                        );
                    } else {
                        assert!(events.iter().any(|event| matches!(event, AgentEvent::Compacted(summary) if summary.starts_with("Conversation summary."))));
                        assert!(conversation.entries()[1].is_summary());
                        if attempt == 1 {
                            assert_eq!(
                                conversation.entries().last().unwrap().content(),
                                "Recent context. ".repeat(6_000)
                            );
                        }
                    }
                }
            }
        }
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
            conversation.add(ConversationEntry::tool(id, "File contents.".repeat(60)));
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

#[tokio::test]
async fn mode_instruction_survives_compaction_without_becoming_summary_input() {
    let model: Model = "ollama:test".parse().unwrap();
    for budget in [0, 1] {
        let provider = SummaryProvider::default();
        let mut conversation = conversation(8, true);
        conversation.set_interaction_mode(InteractionMode::Read);
        try_compact(
            &mut conversation,
            128_000,
            128_000,
            budget,
            &provider,
            &model,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(conversation.interaction_mode(), InteractionMode::Read);
        assert_eq!(
            conversation
                .to_messages()
                .iter()
                .filter(|message| message.content == InteractionMode::Read.developer_instruction())
                .count(),
            1
        );
        assert_eq!(
            conversation
                .entries()
                .iter()
                .filter(|entry| entry.is_summary())
                .count(),
            1
        );
        conversation.set_interaction_mode(InteractionMode::Act);
        assert_eq!(conversation.interaction_mode(), InteractionMode::Act);
    }
}

#[tokio::test]
async fn automatic_compaction_and_the_next_turn_use_the_selected_model() {
    struct RecordingProvider {
        models: Mutex<Vec<Model>>,
        fail_summary: bool,
    }
    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "example"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            let mut models = self.models.lock().unwrap();
            let index = models.len();
            models.push(request.model);
            if index == 1 && self.fail_summary {
                return Box::pin(futures_util::stream::iter([Err(anyhow::anyhow!(
                    "Example summary failure"
                ))]));
            }
            Box::pin(futures_util::stream::iter([
                Ok(StreamEvent::ContentDelta("Example response.".into())),
                Ok(StreamEvent::Usage(TokenUsage {
                    prompt_tokens: if index == 0 { 128_000 } else { 100 },
                    ..Default::default()
                })),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }
    for fail_summary in [false, true] {
        let provider = Arc::new(RecordingProvider {
            models: Mutex::new(Vec::new()),
            fail_summary,
        });
        let mut agent = Agent::builder(provider.clone())
            .keep_recent_tokens(1)
            .build_sync();
        agent.llm_mut().model = "ollama:previous".parse().unwrap();
        let selected: Model = "ollama:selected".parse().unwrap();
        agent.llm_mut().model = selected.clone();
        let mut conversation = conversation(8, true);
        let original = serde_json::to_value(conversation.entries())
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        for turn in 0..2 {
            let events: Vec<_> = agent
                .start(
                    &mut conversation,
                    CancellationToken::new(),
                    AgentInput::User {
                        content: format!("New question {turn}"),
                        context: None,
                        parts: None,
                    },
                    None,
                )
                .collect()
                .await;
            assert_eq!(agent.llm().model, selected);
            if turn == 0 {
                assert_eq!(
                    events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::Compacted(_))),
                    !fail_summary
                );
                if fail_summary {
                    let entries = serde_json::to_value(conversation.entries()).unwrap();
                    assert_eq!(
                        &entries.as_array().unwrap()[..original.len()],
                        original.as_slice()
                    );
                }
            }
        }
        let models = provider.models.lock().unwrap();
        assert_eq!(models.len(), 3);
        assert!(models.iter().all(|model| model == &selected));
    }
}
