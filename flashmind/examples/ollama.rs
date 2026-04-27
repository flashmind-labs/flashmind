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

use futures::StreamExt;

use flashmind::core::{Agent, Conversation, ConversationEntry};
use flashmind::llm::OllamaProvider;
use flashmind::types::{AgentEvent, AgentInput, LlmProvider};

#[tokio::main]
async fn main() {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "llama3.2".into());
    let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(None, None));

    let mut agent = Agent::builder(provider)
        .scope("ollama-example")
        .max_iterations(20)
        .build();

    agent.llm_mut().model = format!("ollama:{model}").parse().unwrap();

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system(
        "You are a helpful assistant. Keep answers concise.",
    ));

    println!("Chatting with Ollama ({model}). Type 'quit' to exit.\n");

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

        let stream = agent.start(&mut conversation, AgentInput::user(input));
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
}
