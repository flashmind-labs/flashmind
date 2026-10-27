use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use flashmind_core::{Agent, CancellationToken, Conversation, MAX_CONTINUATIONS};
use flashmind_types::tool::ToolResult;
use flashmind_types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Provider, Role, StreamEvent, Tool, ToolContext, ToolRegistry,
};
use futures_util::StreamExt;
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
        let events = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(answer);
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}

struct RecordingTool(Arc<Mutex<Vec<Value>>>);

#[async_trait]
impl Tool for RecordingTool {
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

struct Run {
    provider: Arc<ScriptedProvider>,
    executed: Arc<Mutex<Vec<Value>>>,
    events: Vec<AgentEvent>,
}

async fn run(responses: Vec<Vec<StreamEvent>>) -> Run {
    let provider = Arc::new(ScriptedProvider {
        responses: Mutex::new(responses.into()),
        requests: Mutex::new(Vec::new()),
    });
    let executed = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(RecordingTool(executed.clone())));
    let mut agent = Agent::builder(provider.clone()).tools(tools).build_sync();
    let mut conversation = Conversation::new();
    let events = agent
        .start(
            &mut conversation,
            CancellationToken::new(),
            AgentInput::user("Make the example change."),
            Some(10),
        )
        .collect()
        .await;
    Run {
        provider,
        executed,
        events,
    }
}

fn tool_call(arguments: &str, reason: FinishReason) -> Vec<StreamEvent> {
    vec![
        StreamEvent::ToolCallStart {
            index: 0,
            id: "example-call".into(),
            name: "change".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            arguments: arguments.into(),
        },
        StreamEvent::Finished(reason),
    ]
}

fn answer() -> Vec<StreamEvent> {
    vec![
        StreamEvent::ContentDelta("Finished.".into()),
        StreamEvent::Finished(FinishReason::Stop),
    ]
}

#[tokio::test]
async fn truncated_calls_with_valid_json_wait_for_complete_replacements() {
    let run = run(vec![
        tool_call(r#"{"text":"partial"}"#, FinishReason::Length),
        tool_call(
            r#"{"text":"complete replacement"}"#,
            FinishReason::ToolCalls,
        ),
        answer(),
    ])
    .await;

    assert_eq!(
        *run.executed.lock().unwrap(),
        [json!({"text": "complete replacement"})]
    );
    assert_eq!(
        run.events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolStart { .. }))
            .count(),
        1
    );
    let requests = run.provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let retry = &requests[1].messages;
    assert!(retry.iter().any(|message| {
        message.role == Role::Developer
            && message.content.to_lowercase().contains("complete")
            && message.content.to_lowercase().contains("tool")
    }));
    let rejected_call = retry
        .iter()
        .filter_map(|message| message.tool_calls.as_ref())
        .flatten()
        .find(|call| call.arguments == json!({"text": "partial"}))
        .expect("The rejected call remains in the retry transcript.");
    assert!(retry.iter().any(|message| {
        message.role == Role::Tool && message.tool_call_id.as_ref() == Some(&rejected_call.id)
    }));
    assert!(
        run.events
            .iter()
            .any(|event| matches!(event, AgentEvent::Done(text) if text == "Finished."))
    );
}

#[tokio::test]
async fn malformed_truncated_calls_never_reach_tools() {
    let run = run(vec![
        tool_call(r#"{"text":"unfinished"#, FinishReason::Length),
        answer(),
    ])
    .await;

    assert!(run.executed.lock().unwrap().is_empty());
    assert!(
        !run.events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolStart { .. }))
    );
    assert_eq!(run.provider.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn repeated_truncated_calls_stop_at_the_continuation_limit() {
    let responses = (0..usize::from(MAX_CONTINUATIONS) + 2)
        .map(|_| tool_call(r#"{"text":"partial"}"#, FinishReason::Length))
        .collect();
    let run = run(responses).await;

    assert!(run.executed.lock().unwrap().is_empty());
    assert_eq!(
        run.provider.requests.lock().unwrap().len(),
        usize::from(MAX_CONTINUATIONS) + 1
    );
    assert!(
        !run.events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolStart { .. }))
    );
}
