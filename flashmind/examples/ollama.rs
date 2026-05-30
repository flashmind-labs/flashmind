//! Chat with a local Ollama model.
//!
//! Requires Ollama running at localhost:11434 with a model pulled
//! (defaults to llama3.2, override with `MODEL=qwen3 cargo run ...`).
//!
//! ```sh
//! ollama pull llama3.2
//! cargo run -p flashmind --example ollama
//! ```

use std::io::{Write, stdin, stdout};
use std::sync::Arc;

use flashmind_types::AgentLlmConfig;
use futures::StreamExt;

use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::llm::OllamaProvider;
use flashmind::types::{AgentEvent, AgentInput, LlmProvider};

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
    conversation.set_system(
        "You are a helpful assistant. Every response should include mentioning goblins.",
    );

    println!("Chatting with Ollama ({model_str}). Type 'quit' to exit.\n");

    loop {
        print!("> ");
        stdout().flush().unwrap();

        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" {
            break;
        }

        let cancel = CancellationToken::new();
        let stream = agent.start(&mut conversation, cancel, AgentInput::user(input), None);
        tokio::pin!(stream);

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(text) => {
                    print!("{text}");
                    stdout().flush().unwrap();
                }
                AgentEvent::Done(_) => println!("\n"),
                AgentEvent::Error(err) => {
                    eprintln!("\nError: {err}\n");
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(())
}
