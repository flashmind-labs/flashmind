//! Inspect every AgentEvent as it streams from the agent loop.
//!
//! Uses a mock provider that emits tokens word-by-word with usage data.
//! Shows how to handle multi-turn conversations and track token usage.
//!
//! ```sh
//! cargo run -p flashmind --example streaming
//! ```

use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;

use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::types::{
    AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
    Provider, StreamEvent, TokenUsage,
};

struct WordByWordProvider;

#[async_trait]
impl LlmProvider for WordByWordProvider {
    fn name(&self) -> &str {
        "word-by-word"
    }
    fn provider(&self) -> Provider {
        Provider::Ollama
    }
    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        Box::pin(stream! {
            for word in ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog."] {
                yield Ok(StreamEvent::ContentDelta(word.into()));
            }
            yield Ok(StreamEvent::Usage(TokenUsage {
                prompt_tokens: 42,
                completion_tokens: 9,
                total_tokens: 51,
            }));
            yield Ok(StreamEvent::Finished(FinishReason::Stop));
        })
    }
}

#[tokio::main]
async fn main() {
    let provider: Arc<dyn LlmProvider> = Arc::new(WordByWordProvider);
    let mut agent = Agent::builder(provider).build().await;
    let mut conversation = Conversation::new();
    conversation.set_system("You are helpful.");

    for (i, prompt) in ["Tell me something", "Tell me more"].iter().enumerate() {
        println!("=== Turn {} ===", i + 1);

        let cancel = CancellationToken::new();
        let s = agent.start(&mut conversation, cancel, AgentInput::user(*prompt), None);
        tokio::pin!(s);

        while let Some(event) = s.next().await {
            match event {
                AgentEvent::TextDelta(text) => print!("{text}"),
                AgentEvent::Usage(u) => {
                    println!(
                        "\n[usage] prompt={} completion={} total={}",
                        u.prompt_tokens, u.completion_tokens, u.total_tokens
                    );
                }
                AgentEvent::Done(resp) => println!("[done] {resp}"),
                AgentEvent::Error(err) => eprintln!("[error] {err}"),
                _ => {}
            }
        }
        println!();
    }

    println!(
        "Conversation has {} entries after 2 turns",
        conversation.entries().len()
    );
}
