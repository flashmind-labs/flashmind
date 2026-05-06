//! Agent with a custom tool.
//!
//! Demonstrates implementing the `Tool` trait and registering it so the LLM
//! can invoke it during a conversation. Uses a mock provider that always
//! requests the tool, so you can run this without any LLM backend.
//!
//! ```sh
//! cargo run -p flashmind --example custom_tool
//! ```

use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use flashmind::core::{Agent, Conversation, ConversationEntry};
use flashmind::types::tool::{Tool, ToolContext, ToolResult};
use flashmind::types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Provider, StreamEvent, ToolRegistry,
};

// ---------------------------------------------------------------------------
// Custom tool
// ---------------------------------------------------------------------------

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a city."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let city = ctx.args["city"].as_str().unwrap_or("unknown");
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("{city}: Sunny, 22°C"),
        ))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        format!(
            "Checking weather for {}",
            args["city"].as_str().unwrap_or("?")
        )
    }
}

// ---------------------------------------------------------------------------
// Mock provider that requests the tool then responds
// ---------------------------------------------------------------------------

struct ToolCallingProvider;

#[async_trait]
impl LlmProvider for ToolCallingProvider {
    fn name(&self) -> &str {
        "mock-tool-caller"
    }
    fn provider(&self) -> Provider {
        Provider::Ollama
    }
    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        let has_tool_result = request
            .messages
            .iter()
            .any(|m| m.role == flashmind::types::Role::Tool);

        Box::pin(stream! {
            if has_tool_result {
                yield Ok(StreamEvent::ContentDelta("Based on the weather data: it's a beautiful day!".into()));
                yield Ok(StreamEvent::Finished(FinishReason::Stop));
            } else {
                yield Ok(StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "get_weather".into(),
                });
                yield Ok(StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: r#"{"city":"Tokyo"}"#.into(),
                });
                yield Ok(StreamEvent::Finished(FinishReason::ToolCalls));
            }
        })
    }
}

#[tokio::main]
async fn main() {
    let provider: Arc<dyn LlmProvider> = Arc::new(ToolCallingProvider);

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(WeatherTool));
    println!("Registered tools: {:?}", tools.list());

    let mut agent = Agent::builder(provider).tools(tools).build();

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system("You have access to tools."));

    let s = agent.start(
        &mut conversation,
        AgentInput::user("What's the weather in Tokyo?"),
        None,
    );
    tokio::pin!(s);

    while let Some(event) = s.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),
            AgentEvent::ToolStart {
                name, humanized, ..
            } => {
                println!("[tool:start] {name} — {humanized}");
            }
            AgentEvent::ToolResult {
                name,
                output,
                success,
                ..
            } => {
                println!("[tool:done]  {name} → {output} (ok={success})");
            }
            AgentEvent::Done(resp) => {
                println!("\n\nFinal: {resp}");
            }
            _ => {}
        }
    }
}
