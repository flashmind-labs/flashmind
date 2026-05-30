//! Subagent debate — two agents argue opposing sides, moderator synthesizes.
//!
//! The moderator spawns two agents with opposing system prompts, waits for
//! both to make their case, then delivers a balanced verdict.
//!
//! Requires Ollama running locally (defaults to gemma4:e4b, override with MODEL).
//!
//! ```sh
//! cargo run -p flashmind --example subagents
//! MODEL=qwen3.5:2b cargo run -p flashmind --example subagents
//! ```

use std::io::{Write, stdout};
use std::sync::Arc;

use futures::StreamExt;

use flashmind::core::{Agent, AgentManager, CancellationToken, Conversation};
use flashmind::llm::OllamaProvider;
use flashmind::tools::ToolBuilder;
use flashmind::types::{AgentEvent, AgentInput, AgentLlmConfig, LlmProvider};

const SYSTEM_PROMPT: &str = "\
You are a debate moderator. When given a topic:

1. Use `delegate` to spawn two agents with opposing viewpoints. \
   Give each a system_prompt establishing their position and the name of their opponent. \
   Their task is to argue their side of the debate.
2. Use `agent_wait` with a timeout of 120 seconds for each agent.
3. Synthesize both perspectives into a balanced 3-4 sentence verdict.

Only use `delegate` and `agent_wait`.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_str = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e4b".into());
    let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(None, None)?);

    let model = format!("ollama:{model_str}").parse()?;
    let llm = AgentLlmConfig::new(model);

    let manager = Arc::new(AgentManager::new(4, 2));

    let tools = ToolBuilder::new()
        .subagents(manager, provider.clone(), Some(llm.clone()))
        .build();

    let mut agent = Agent::builder(provider).llm(llm).tools(tools).build().await;

    let mut conversation = Conversation::new();
    conversation.set_system(SYSTEM_PROMPT);

    println!("=== Subagent Debate ({model_str}) ===\n");

    {
        let cancel = CancellationToken::new();
        let stream = agent.start(
            &mut conversation,
            cancel,
            AgentInput::user("Will AI replace most human jobs within 20 years?"),
            None,
        );
        tokio::pin!(stream);

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(text) => {
                    print!("{text}");
                    stdout().flush().unwrap();
                }
                AgentEvent::ToolStart {
                    name, humanized, ..
                } => {
                    println!("\n  [{name}] {humanized}");
                }
                AgentEvent::ToolResult {
                    name,
                    output,
                    success,
                    ..
                } => {
                    let preview: String = output.chars().take(120).collect();
                    let ellipsis = if output.len() > 120 { "..." } else { "" };
                    println!("  [{name}] → {preview}{ellipsis} (ok={success})");
                }
                AgentEvent::Done(_) => println!(),
                AgentEvent::Error(err) => {
                    eprintln!("\n[error] {err}");
                    break;
                }
                _ => {}
            }
        }
    }

    println!("Conversation: {} entries", conversation.entries().len());

    Ok(())
}
