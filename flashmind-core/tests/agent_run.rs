use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use flashmind_core::{
    Agent, AgentRun, CancellationToken, Checkpoint, Conversation, ConversationEntry, RunBoundary,
};
use flashmind_types::tool::ToolResult;
use flashmind_types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Message, Provider, StreamEvent, TokenUsage, Tool, ToolCall, ToolContext, ToolRegistry,
};
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};

struct ScriptedProvider {
    responses: Mutex<VecDeque<Vec<StreamEvent>>>,
    requests: Mutex<Vec<CompletionRequest>>,
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn name(&self) -> &str {
        "example"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        self.requests.lock().unwrap().push(request);
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("The test supplies every expected model response.");
        Box::pin(stream::iter(response.into_iter().map(Ok)))
    }
}

struct ChangeTool(Arc<Mutex<Vec<Value>>>);

#[async_trait]
impl Tool for ChangeTool {
    fn name(&self) -> &str {
        "change"
    }

    fn description(&self) -> &str {
        "Change example state."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {"text": {"type": "string"}}})
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        self.0.lock().unwrap().push(ctx.args);
        Ok(ToolResult::success(
            ctx.tool_call_id,
            "Changed example state.",
        ))
    }
}

struct Setup {
    agent: Agent,
    provider: Arc<ScriptedProvider>,
    executed: Arc<Mutex<Vec<Value>>>,
}

fn setup(responses: Vec<Vec<StreamEvent>>) -> Setup {
    let provider = Arc::new(ScriptedProvider {
        responses: Mutex::new(responses.into()),
        requests: Mutex::new(Vec::new()),
    });
    let executed = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(ChangeTool(executed.clone())));
    Setup {
        agent: Agent::builder(provider.clone()).tools(tools).build_sync(),
        provider,
        executed,
    }
}

struct Step {
    events: Vec<AgentEvent>,
    boundary: RunBoundary,
}

async fn advance(run: &mut AgentRun, agent: &mut Agent, conversation: &mut Conversation) -> Step {
    let cancel = CancellationToken::new();
    let mut step = run.advance(agent, conversation, &cancel);
    let events = step.by_ref().collect().await;
    let boundary = step
        .take_result()
        .expect("A completed step provides a boundary.")
        .expect("The scripted step succeeds.");
    Step { events, boundary }
}

async fn execute(agent: &mut Agent, conversation: &mut Conversation, calls: &[ToolCall]) {
    let cancel = CancellationToken::new();
    for batch in agent.tool_batches(calls) {
        let completed = {
            let mut execution = agent.execute_tool_batch(&batch, &cancel);
            while execution.next().await.is_some() {}
            execution
                .take_result()
                .expect("A completed batch returns its tool results.")
        };
        for done in completed {
            done.finish(conversation);
        }
    }
}

fn text(content: &str, reason: FinishReason) -> Vec<StreamEvent> {
    vec![
        StreamEvent::ContentDelta(content.into()),
        usage(),
        StreamEvent::Finished(reason),
    ]
}

fn call(content: &str, reason: FinishReason) -> Vec<StreamEvent> {
    vec![
        StreamEvent::ToolCallStart {
            index: 0,
            id: "example-call".into(),
            name: "change".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            arguments: json!({"text": content}).to_string(),
        },
        usage(),
        StreamEvent::Finished(reason),
    ]
}

fn usage() -> StreamEvent {
    StreamEvent::Usage(TokenUsage {
        prompt_tokens: 10,
        completion_tokens: 10,
        total_tokens: 20,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    })
}

fn messages(messages: &[Message]) -> Value {
    // Generated call IDs and timestamps differ between otherwise identical runs.
    Value::Array(
        messages
            .iter()
            .map(|message| {
                let calls = message.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|call| json!({"name": call.name, "arguments": call.arguments}))
                        .collect::<Vec<_>>()
                });
                json!({
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": calls,
                    "is_tool_result": message.tool_call_id.is_some(),
                    "parts": message.parts,
                })
            })
            .collect(),
    )
}

#[tokio::test]
async fn stepping_and_start_produce_the_same_requests_results_and_conversation() {
    let scenarios = [
        vec![
            text("", FinishReason::Stop),
            text("Finished.", FinishReason::Stop),
        ],
        vec![
            call("partial", FinishReason::Length),
            call("complete", FinishReason::ToolCalls),
            text("Finished.", FinishReason::Stop),
        ],
        vec![
            text("First ", FinishReason::Length),
            text("second.", FinishReason::Stop),
        ],
    ];

    for responses in scenarios {
        let mut direct = setup(responses.clone());
        let mut direct_conversation = Conversation::with_system("Example instructions.");
        let events: Vec<_> = direct
            .agent
            .start(
                &mut direct_conversation,
                CancellationToken::new(),
                AgentInput::user("Make the example change."),
                Some(10),
            )
            .collect()
            .await;
        let direct_result = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::Done(content) => Some(content.clone()),
                _ => None,
            })
            .expect("The direct run finishes successfully.");

        let mut stepped = setup(responses);
        let mut conversation = Conversation::with_system("Example instructions.");
        let mut run = AgentRun::new(
            &mut conversation,
            AgentInput::user("Make the example change."),
            Some(10),
        );
        let stepped_result = loop {
            match advance(&mut run, &mut stepped.agent, &mut conversation)
                .await
                .boundary
            {
                RunBoundary::BeforeModel | RunBoundary::Checkpoint(_) => {}
                RunBoundary::Tools(calls) => {
                    execute(&mut stepped.agent, &mut conversation, &calls).await
                }
                RunBoundary::Done(content) => break content,
            }
        };

        assert_eq!(stepped_result, direct_result);
        assert_eq!(
            *stepped.executed.lock().unwrap(),
            *direct.executed.lock().unwrap()
        );
        assert_eq!(
            messages(&conversation.to_messages()),
            messages(&direct_conversation.to_messages())
        );
        let stepped_requests = stepped.provider.requests.lock().unwrap();
        let direct_requests = direct.provider.requests.lock().unwrap();
        assert_eq!(stepped_requests.len(), direct_requests.len());
        for (stepped_request, direct_request) in stepped_requests.iter().zip(direct_requests.iter())
        {
            assert_eq!(
                messages(&stepped_request.messages),
                messages(&direct_request.messages)
            );
        }
    }
}

#[tokio::test]
async fn checkpoints_release_the_conversation_before_tools_and_the_next_model_request() {
    let mut setup = setup(vec![
        call("complete", FinishReason::ToolCalls),
        text("Finished.", FinishReason::Stop),
    ]);
    let mut conversation = Conversation::new();
    let mut run = AgentRun::new(
        &mut conversation,
        AgentInput::user("Make the example change."),
        None,
    );

    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::BeforeModel
    ));
    assert!(setup.provider.requests.lock().unwrap().is_empty());
    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::Checkpoint(Checkpoint::Assistant)
    ));
    assert_eq!(setup.provider.requests.lock().unwrap().len(), 1);
    assert!(setup.executed.lock().unwrap().is_empty());
    let assistant_checkpoint = serde_json::to_string(&conversation).unwrap();
    assert!(!assistant_checkpoint.is_empty());

    let RunBoundary::Tools(calls) = advance(&mut run, &mut setup.agent, &mut conversation)
        .await
        .boundary
    else {
        panic!("The assistant checkpoint must precede tool execution.");
    };
    assert!(setup.executed.lock().unwrap().is_empty());
    execute(&mut setup.agent, &mut conversation, &calls).await;
    assert_eq!(setup.executed.lock().unwrap().len(), 1);
    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::Checkpoint(Checkpoint::Tools)
    ));
    assert_eq!(setup.provider.requests.lock().unwrap().len(), 1);
    let tool_checkpoint = serde_json::to_string(&conversation).unwrap();
    assert_ne!(assistant_checkpoint, tool_checkpoint);
    conversation.add(ConversationEntry::system_message("Checkpoint persisted."));
    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::BeforeModel
    ));
    assert_eq!(setup.provider.requests.lock().unwrap().len(), 1);
    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::Checkpoint(Checkpoint::Assistant)
    ));
    let requests = setup.provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1]
            .messages
            .iter()
            .any(|message| message.content == "Checkpoint persisted.")
    );
    assert!(
        requests[1]
            .messages
            .iter()
            .any(|message| message.tool_call_id.as_ref() == Some(&calls[0].id))
    );
}

#[tokio::test]
async fn compaction_is_checkpointed_before_the_next_model_request() {
    let mut setup = setup(vec![
        call("complete", FinishReason::ToolCalls),
        text("Example summary.", FinishReason::Stop),
        text("Finished.", FinishReason::Stop),
    ]);
    setup.agent.set_compact_at_tokens(Some(20));
    let mut conversation = Conversation::with_system("Example instructions.");
    for _ in 0..8 {
        conversation.add(ConversationEntry::user("Earlier question."));
        conversation.add(ConversationEntry::assistant("Earlier answer."));
    }
    // Exceed the normal recent-history budget so compaction has older input.
    conversation.add(ConversationEntry::user("Earlier context. ".repeat(6_000)));
    conversation.add(ConversationEntry::assistant("Earlier context recorded."));
    let mut run = AgentRun::new(
        &mut conversation,
        AgentInput::user("Make the example change."),
        None,
    );

    loop {
        let step = advance(&mut run, &mut setup.agent, &mut conversation).await;
        match step.boundary {
            RunBoundary::BeforeModel
            | RunBoundary::Checkpoint(Checkpoint::Assistant | Checkpoint::Tools) => {}
            RunBoundary::Tools(calls) => execute(&mut setup.agent, &mut conversation, &calls).await,
            RunBoundary::Checkpoint(Checkpoint::Compaction) => {
                assert!(step.events.iter().any(|event| matches!(event, AgentEvent::Compacted(summary) if summary.starts_with("Example summary."))));
                break;
            }
            _ => panic!("The tool round must compact before another model request."),
        }
    }

    assert_eq!(setup.provider.requests.lock().unwrap().len(), 2);
    assert!(
        conversation
            .entries()
            .iter()
            .any(ConversationEntry::is_summary)
    );
    let persisted = serde_json::to_string(&conversation).unwrap();
    let restored: Conversation = serde_json::from_str(&persisted).unwrap();
    assert_eq!(
        messages(&restored.to_messages()),
        messages(&conversation.to_messages())
    );
    assert!(matches!(
        advance(&mut run, &mut setup.agent, &mut conversation)
            .await
            .boundary,
        RunBoundary::BeforeModel
    ));
    assert_eq!(setup.provider.requests.lock().unwrap().len(), 2);
    advance(&mut run, &mut setup.agent, &mut conversation).await;
    let requests = setup.provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[2]
            .messages
            .iter()
            .any(|message| message.content.starts_with("Example summary."))
    );
}

struct PartialProvider;

#[async_trait]
impl LlmProvider for PartialProvider {
    fn name(&self) -> &str {
        "example"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        Box::pin(
            stream::iter([
                Ok(StreamEvent::ReasoningDelta("Partial reasoning.".into())),
                Ok(StreamEvent::ContentDelta("Partial reply.".into())),
            ])
            .chain(stream::pending()),
        )
    }
}

#[tokio::test]
async fn cancellation_preserves_visible_partial_text_at_the_assistant_checkpoint() {
    let mut agent = Agent::builder(Arc::new(PartialProvider)).build_sync();
    let mut conversation = Conversation::new();
    let mut run = AgentRun::new(
        &mut conversation,
        AgentInput::user("Example request."),
        None,
    );
    advance(&mut run, &mut agent, &mut conversation).await;

    let cancel = CancellationToken::new();
    let boundary = {
        let mut step = run.advance(&mut agent, &mut conversation, &cancel);
        while let Some(event) = step.next().await {
            if matches!(event, AgentEvent::TextDelta(ref text) if text == "Partial reply.") {
                cancel.cancel();
            }
        }
        step.take_result().unwrap().unwrap()
    };
    assert!(matches!(
        boundary,
        RunBoundary::Checkpoint(Checkpoint::Assistant)
    ));
    let partial = conversation
        .entries()
        .iter()
        .find(|entry| entry.is_assistant())
        .expect("The visible partial reply is retained.");
    assert_eq!(partial.content(), "Partial reply.");
    assert_eq!(partial.reasoning(), Some("Partial reasoning."));
    let mut step = run.advance(&mut agent, &mut conversation, &cancel);
    while step.next().await.is_some() {}
    assert!(step.take_result().unwrap().is_err());
}
