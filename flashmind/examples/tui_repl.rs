//! Interactive TUI REPL powered by flashmind-tui with a local Ollama model.
//!
//! ```sh
//! cargo run -p flashmind --example tui_repl
//! # or with a specific model:
//! MODEL=qwen3 cargo run -p flashmind --example tui_repl
//! ```

use std::sync::Arc;

use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::llm::OllamaProvider;
use flashmind::types::{AgentInput, AgentLlmConfig, LlmProvider};
use flashmind_tui::{Repl, ReplConfig, ReplEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_str = std::env::var("MODEL").unwrap_or_else(|_| "llama3.2".into());
    let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(None, None)?);

    let model = format!("ollama:{model_str}").parse()?;
    let mut agent = Agent::builder(provider)
        .llm(AgentLlmConfig::new(model))
        .build()
        .await;

    let mut conversation = Conversation::new();
    conversation.set_system("You are a helpful assistant. Respond concisely.");

    let config = ReplConfig {
        prompt: "x".to_string(),
        greeting: Some(format!(
            "flashmind-tui REPL — model: {model_str} — Ctrl-D to quit"
        )),
        ..Default::default()
    };

    let mut repl = Repl::new(config);
    repl.print_greeting()?;

    while let ReplEvent::UserInput(text) = repl.read_input()? {
        let cancel = CancellationToken::new();
        let stream = agent.start(
            &mut conversation,
            cancel.clone(),
            AgentInput::user(text),
            None,
        );
        repl.stream_response(cancel, Box::pin(stream)).await?;
    }

    Ok(())
}
