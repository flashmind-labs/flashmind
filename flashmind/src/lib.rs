//! Flashmind — AI agent framework for Rust.
//!
//! This is the unified facade crate that re-exports all Flashmind sub-crates
//! under a single namespace. Add `flashmind` as a dependency to get everything,
//! or depend on individual crates for finer control.
//!
//! # Sub-crates
//!
//! | Namespace | Crate | Description |
//! |-----------|-------|-------------|
//! | `flashmind::types` | `flashmind-types` | Traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events |
//! | `flashmind::core` | `flashmind-core` | `Agent`, `AgentBuilder`, `Conversation`, streaming, compaction |
//! | `flashmind::llm` | `flashmind-llm` | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama) |
//! | `flashmind::tools` | `flashmind-tools` | 30+ built-in tool implementations + agent delegation tools |
//! | `flashmind::prompts` | `flashmind-prompts` | Common prompt helpers (project instructions, etc.) |
//! | `flashmind::memory` | `flashmind-memory` | Vector memory (SQLite + sqlite-vec + FTS5); session persistence via `session` feature |
//! | `flashmind::cron` | `flashmind-cron` | Cron job scheduling with pluggable storage |
//! | `flashmind::skills` | `flashmind-skills` | Skill discovery, loading, and execution |
//! | `flashmind::tailscale` | `flashmind-tailscale` | Tailscale local API client and Funnel helpers |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use flashmind::core::{Agent, Conversation, ConversationEntry};
//! use flashmind::types::{AgentEvent, AgentInput, LlmProvider};
//! use futures::StreamExt;
//!
//! // 1. Create a provider (or use one from flashmind::llm)
//! let provider: Arc<dyn LlmProvider> = /* ... */;
//!
//! // 2. Build the agent
//! let mut agent = Agent::builder(provider).build();
//!
//! // 3. Run a turn
//! let mut conversation = Conversation::new();
//! conversation.set_system("You are helpful.");
//!
//! let stream = agent.start(&mut conversation, AgentInput::user("Hello!"), None);
//! tokio::pin!(stream);
//! while let Some(event) = stream.next().await {
//!     match event {
//!         AgentEvent::TextDelta(text) => print!("{text}"),
//!         AgentEvent::Done(response) => println!("\n{response}"),
//!         _ => {}
//!     }
//! }
//! ```

pub use flashmind_core as core;
pub use flashmind_cron as cron;
pub use flashmind_llm as llm;
pub use flashmind_memory as memory;
pub use flashmind_prompts as prompts;
pub use flashmind_skills as skills;
pub use flashmind_tailscale as tailscale;
pub use flashmind_tools as tools;
#[cfg(feature = "tui")]
pub use flashmind_tui as tui;
pub use flashmind_types as types;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_stream::stream;
    use async_trait::async_trait;
    use futures::StreamExt;

    use crate::core::{Agent, CancellationToken, Conversation};
    use crate::types::{
        AgentEvent, AgentInput, CompletionRequest, CompletionStream, FinishReason, LlmProvider,
        Provider, StreamEvent, ToolRegistry,
    };

    struct EchoProvider;

    #[async_trait]
    impl LlmProvider for EchoProvider {
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            let user_text = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == crate::types::Role::User)
                .map(|m| m.content.clone())
                .unwrap_or_default();

            Box::pin(stream! {
                yield Ok(StreamEvent::ContentDelta(format!("Echo: {user_text}")));
                yield Ok(StreamEvent::Finished(FinishReason::Stop));
            })
        }

        fn name(&self) -> &str {
            "echo"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }
    }

    #[tokio::test]
    async fn facade_builder_end_to_end() {
        let provider: Arc<dyn LlmProvider> = Arc::new(EchoProvider);

        let mut agent = Agent::builder(provider)
            .tools(ToolRegistry::new())
            .build_sync();

        let mut conversation = Conversation::new();
        conversation.set_system("You echo messages");

        let mut response = String::new();
        let cancel = CancellationToken::new();
        let s = agent.start(
            &mut conversation,
            cancel,
            AgentInput::user("hello world"),
            None,
        );
        tokio::pin!(s);
        while let Some(ev) = s.next().await {
            if let AgentEvent::Done(text) = ev {
                response = text;
            }
        }

        assert_eq!(response, "Echo: hello world");
    }

    #[tokio::test]
    async fn facade_builder_defaults_work() {
        let provider: Arc<dyn LlmProvider> = Arc::new(EchoProvider);

        let mut agent = Agent::builder(provider).build_sync();

        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let s = agent.start(&mut conversation, cancel, AgentInput::user("test"), None);
        tokio::pin!(s);

        let mut got_done = false;
        while let Some(ev) = s.next().await {
            match ev {
                AgentEvent::Done(_) => got_done = true,
                _ => {}
            }
        }

        assert!(got_done);
    }
}
