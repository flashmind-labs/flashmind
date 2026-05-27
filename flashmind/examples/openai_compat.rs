//! Create a custom OpenAI-compatible provider for vLLM or similar backends.
//!
//! Shows three ways to customise `OpenAiProvider` via the builder API:
//! 1. Simple — just a name and URL
//! 2. Extra body — static JSON merged into every request
//! 3. Transform — dynamic request mutation based on request context
//!
//! ```sh
//! # Start a vLLM server, then:
//! MODEL=Qwen/Qwen3-8B cargo run -p flashmind --example openai_compat
//! ```

use std::io::{Write, stdin, stdout};
use std::sync::Arc;

use flashmind_types::AgentLlmConfig;
use futures::StreamExt;

use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::llm::OpenAiProvider;
use flashmind::types::{AgentEvent, AgentInput, LlmProvider};

#[tokio::main]
async fn main() {
    let model_str = std::env::var("MODEL").unwrap_or_else(|_| "Qwen/Qwen3-8B".into());
    let base_url = std::env::var("BASE_URL").unwrap_or_else(|_| "http://localhost:8000/".into());
    let api_key = std::env::var("API_KEY").ok();

    // -- Pick one of these three styles: --

    // 1) Simple: just a name and URL
    // let provider = OpenAiProvider::builder(&base_url)
    //     .name("vllm")
    //     .build();

    // 2) Extra body: static JSON deep-merged into every request
    // let provider = OpenAiProvider::builder(&base_url)
    //     .name("vllm")
    //     .extra_body(serde_json::json!({
    //         "guided_json": {"type": "object"},
    //     }))
    //     .build();

    // 3) Transform: dynamic mutation based on request context
    let mut builder = OpenAiProvider::builder(&base_url).name("vllm");
    if let Some(key) = api_key {
        builder = builder.api_key(key);
    }
    let provider: Arc<dyn LlmProvider> = Arc::new(
        builder
            .transform_body(|body, req| {
                if req.reasoning.is_on() {
                    body["chat_template_kwargs"]["enable_thinking"] = serde_json::Value::Bool(true);
                }
            })
            .build(),
    );

    let model = format!("openai:{model_str}").parse().unwrap();
    let mut agent = Agent::builder(provider)
        .llm(AgentLlmConfig::new(model))
        .build()
        .await;

    let mut conversation = Conversation::new();
    conversation.set_system("You are a helpful assistant.");

    println!("Chatting with {model_str} via custom OpenAI-compatible provider.");
    println!("Type 'quit' to exit.\n");

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
}
