use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry};
use flashmind_types::tool::ToolResult;
use flashmind_types::{
    AgentInput, CompletionRequest, CompletionStream, FinishReason, InteractionMode, LlmProvider,
    Provider, Role, StreamEvent, Tool, ToolContext, ToolRegistry,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[derive(Default)]
struct RecordingProvider {
    requests: Mutex<Vec<CompletionRequest>>,
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
        let mut requests = self.requests.lock().unwrap();
        let first = requests.is_empty();
        requests.push(request);
        let events = if first {
            vec![
                StreamEvent::ToolCallStart {
                    index: 0,
                    id: "example-call".into(),
                    name: "change".into(),
                },
                StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: "{}".into(),
                },
                StreamEvent::Finished(FinishReason::ToolCalls),
            ]
        } else {
            vec![
                StreamEvent::ContentDelta("Example answer.".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ]
        };
        Box::pin(futures_util::stream::iter(events.into_iter().map(Ok)))
    }
}

struct ChangeTool(Arc<AtomicUsize>);

#[async_trait]
impl Tool for ChangeTool {
    fn name(&self) -> &str {
        "change"
    }
    fn description(&self) -> &str {
        "Change example state."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(
            ctx.tool_call_id,
            "Changed example state.",
        ))
    }
}

#[tokio::test]
async fn mode_is_a_developer_instruction_and_keeps_tools_available() {
    let provider = Arc::new(RecordingProvider::default());
    let changes = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(ChangeTool(changes.clone())));
    let mut agent = Agent::builder(provider.clone()).tools(tools).build_sync();
    let mut conversation = Conversation::with_system("Example system instructions.");
    for mode in [InteractionMode::Read, InteractionMode::Act] {
        conversation.set_interaction_mode(mode);
        let stream = agent.start(
            &mut conversation,
            CancellationToken::new(),
            AgentInput::User {
                content: "Example request.".into(),
                context: None,
                parts: None,
            },
            None,
        );
        stream.collect::<Vec<_>>().await;
    }
    assert_eq!(changes.load(Ordering::SeqCst), 1);
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    for (request, mode) in requests.iter().zip([
        InteractionMode::Read,
        InteractionMode::Read,
        InteractionMode::Act,
    ]) {
        let instructions: Vec<_> = request
            .messages
            .iter()
            .filter(|message| message.content.contains("You are in "))
            .collect();
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].role, Role::Developer);
        assert_eq!(instructions[0].content, mode.developer_instruction());
        assert_eq!(request.tools.len(), 1);
    }
}

#[test]
fn switching_modes_replaces_the_instruction_and_survives_serialization() {
    let mut conversation = Conversation::with_system("Example instructions.");
    assert_eq!(conversation.interaction_mode(), InteractionMode::Act);
    conversation.add(ConversationEntry::user("Example question."));
    for mode in [
        InteractionMode::Read,
        InteractionMode::Act,
        InteractionMode::Read,
    ] {
        conversation.set_interaction_mode(mode);
        assert_eq!(conversation.entries().len(), 3);
        assert_eq!(conversation.interaction_mode(), mode);
        assert_eq!(
            conversation.to_messages().last().unwrap().role,
            Role::Developer
        );
    }
    let saved = serde_json::to_string(&conversation).unwrap();
    let restored: Conversation = serde_json::from_str(&saved).unwrap();
    assert_eq!(restored.interaction_mode(), InteractionMode::Read);
}
