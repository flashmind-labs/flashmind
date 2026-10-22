//! Build streaming AI agents with tools, memory, and caller-owned conversations.
//!
//! This crate re-exports the Flashmind libraries. The default features provide
//! the agent runtime and LLM providers. Enable optional features for tools,
//! memory, scheduling, skills, networking, or a terminal UI.
//!
//! # Quick start
//!
//! Add these dependencies:
//!
//! ```toml
//! [dependencies]
//! flashmind = "0.1"
//! futures = "0.3"
//! tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
//! ```
//!
//! Run Ollama and pull the model you select before running this example.
//! It streams one turn, then keeps the conversation available for later turns.
//!
//! ```no_run
//! # #[cfg(all(feature = "core", feature = "llm"))]
//! # mod example {
//! use std::error::Error;
//!
//! use flashmind::core::{Agent, CancellationToken, Conversation};
//! use flashmind::llm::create_provider;
//! use flashmind::types::{AgentEvent, AgentInput, AgentLlmConfig, Provider};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn Error>> {
//!     let provider = create_provider(Provider::Ollama, None)?;
//!     let model = "ollama:llama3.2".parse()?;
//!     let mut agent = Agent::builder(provider)
//!         .llm(AgentLlmConfig::new(model))
//!         .build()
//!         .await;
//!     let mut conversation = Conversation::new();
//!     conversation.set_system("Answer briefly.");
//!
//!     let cancel = CancellationToken::new();
//!     let stream = agent.start(&mut conversation, cancel, AgentInput::user("Hello!"), None);
//!     tokio::pin!(stream);
//!     while let Some(event) = stream.next().await {
//!         match event {
//!             AgentEvent::TextDelta(text) => print!("{text}"),
//!             AgentEvent::Error(error) => eprintln!("{error}"),
//!             _ => {}
//!         }
//!     }
//!     Ok(())
//! }
//! # }
//! ```
//!
//! Keep consuming the stream until it ends; dropping it cancels the turn.
//! To cancel from another task, clone the cancellation token before passing it
//! to `Agent::start`, then call `cancel()` on the clone.
//!
//! # Features and modules
//!
//! | Feature | Module | API |
//! |---------|--------|-----|
//! | Always available | [`types`] | Provider and tool traits, messages, models, events |
//! | `core` (default) | `core` | Agent runtime, conversations, compaction |
//! | `llm` (default) | `llm` | Ollama, OpenRouter, Anthropic, and OpenAI-compatible providers |
//! | `tools` | `tools` | Built-in tools and subagent delegation |
//! | `memory` | `memory` | Vector storage and HTTP embedding providers |
//! | `sqlite` (default) | `memory`, `tools` | SQLite support when either module is enabled |
//! | `memory-postgres` | `memory` | PostgreSQL storage with pgvector; also enables `memory` |
//! | `session` | `memory::session` | SQLite conversation history; also enables `memory` |
//! | `prompts` | `prompts` | System prompt fragments and project instructions |
//! | `cron` | `cron` | Scheduling with pluggable storage |
//! | `skills` | `skills` | Skill discovery, installation, and execution |
//! | `tailscale` | `tailscale` | Local API client and Funnel helpers |
//! | `tui` | `tui` | Terminal widgets and streaming REPL with Markdown rendering |
//!
//! The `sqlite` feature does not enable `memory` or `tools` by itself. For
//! SQLite vector storage, use `features = ["memory"]` with the defaults enabled,
//! or `default-features = false, features = ["memory", "sqlite"]`.
//!
//! Integration features `mcp`, `gmail`, `google-calendar`, `google-contacts`,
//! `outlook`, `caldav`, `github`, `slack`, `composio`, and `lsp` also enable
//! `tools`. The `full` feature enables all modules and tool integrations,
//! SQLite memory, and sessions. Enable `memory-postgres` separately for the
//! PostgreSQL memory backend.
//!
//! ```toml
//! flashmind = { version = "0.1", features = ["memory", "mcp"] }
//! ```
//!
//! # Extending an agent
//!
//! Implement [`types::LlmProvider`] for a custom model backend or
//! [`types::Tool`] for a tool. Register tools in a [`types::ToolRegistry`] and
//! pass it to the agent builder. [`types::AgentLlmConfig`] selects the model,
//! sampling settings, reasoning level, and optional session identifier.
//!
//! Reuse the same conversation and session identifier across turns to retain
//! history and provider cache routing. Session persistence is provided by the
//! optional `memory::SessionStore`; the agent does not save conversations by
//! itself.
//!
//! Provider constructors are documented in
//! [flashmind-llm](https://docs.rs/flashmind-llm), tool configuration in
//! [flashmind-tools](https://docs.rs/flashmind-tools), and storage in
//! [flashmind-memory](https://docs.rs/flashmind-memory).

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "core")]
#[cfg_attr(docsrs, doc(cfg(feature = "core")))]
pub use flashmind_core as core;
#[cfg(feature = "cron")]
#[cfg_attr(docsrs, doc(cfg(feature = "cron")))]
pub use flashmind_cron as cron;
#[cfg(feature = "llm")]
#[cfg_attr(docsrs, doc(cfg(feature = "llm")))]
pub use flashmind_llm as llm;
#[cfg(feature = "memory")]
#[cfg_attr(docsrs, doc(cfg(feature = "memory")))]
pub use flashmind_memory as memory;
#[cfg(feature = "prompts")]
#[cfg_attr(docsrs, doc(cfg(feature = "prompts")))]
pub use flashmind_prompts as prompts;
#[cfg(feature = "skills")]
#[cfg_attr(docsrs, doc(cfg(feature = "skills")))]
pub use flashmind_skills as skills;
#[cfg(feature = "tailscale")]
#[cfg_attr(docsrs, doc(cfg(feature = "tailscale")))]
pub use flashmind_tailscale as tailscale;
#[cfg(feature = "tools")]
#[cfg_attr(docsrs, doc(cfg(feature = "tools")))]
pub use flashmind_tools as tools;
#[cfg(feature = "tui")]
#[cfg_attr(docsrs, doc(cfg(feature = "tui")))]
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
            .build()
            .await;

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

        let mut agent = Agent::builder(provider).build().await;

        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let s = agent.start(&mut conversation, cancel, AgentInput::user("test"), None);
        tokio::pin!(s);

        let mut got_done = false;
        while let Some(ev) = s.next().await {
            if let AgentEvent::Done(_) = ev {
                got_done = true;
            }
        }

        assert!(got_done);
    }
}
