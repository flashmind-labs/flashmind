use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use flashmind_core::{Agent, CancellationToken, Conversation};
use flashmind_types::tool::{InterruptPayload, ToolResult};
use flashmind_types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Provider, StreamEvent, Tool, ToolContext, ToolRegistry,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

struct ToolProvider;

#[async_trait]
impl LlmProvider for ToolProvider {
    fn name(&self) -> &str {
        "example"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    fn complete(&self, _: CompletionRequest) -> CompletionStream {
        Box::pin(futures_util::stream::iter([
            Ok(StreamEvent::ToolCallStart {
                index: 0,
                id: "example-call".into(),
                name: "ask".into(),
            }),
            Ok(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            }),
            Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
        ]))
    }
}

#[derive(Debug)]
struct Question;

impl InterruptPayload for Question {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn display_output(&self) -> String {
        "Example question.".into()
    }
}

struct CancelAndAsk(CancellationToken);

#[async_trait]
impl Tool for CancelAndAsk {
    fn name(&self) -> &str {
        "ask"
    }

    fn description(&self) -> &str {
        "Ask an example question."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object"})
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        self.0.cancel();
        Ok(ToolResult::interrupt(ctx.tool_call_id, Arc::new(Question)))
    }
}

#[tokio::test]
async fn cancellation_takes_priority_over_a_simultaneous_tool_interrupt() {
    let cancel = CancellationToken::new();
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CancelAndAsk(cancel.clone())));
    let mut agent = Agent::builder(Arc::new(ToolProvider))
        .tools(tools)
        .build_sync();
    let mut conversation = Conversation::new();
    let events = agent
        .start(
            &mut conversation,
            cancel,
            AgentInput::user("Ask the example question."),
            Some(2),
        )
        .collect::<Vec<_>>()
        .await;

    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::Interrupted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::Error(error) if error == "cancelled"))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::Done(_)))
    );
}
