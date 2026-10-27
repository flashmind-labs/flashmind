//! Agent runtime, conversations, streaming, and context compaction.
//!
//! [`Agent`] works with any [`LlmProvider`](flashmind_types::LlmProvider).
//! The caller owns the [`Conversation`] and passes it into each turn. Providers
//! live in `flashmind-llm`; persistent session storage lives in `flashmind-memory`.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use flashmind_core::{Agent, CancellationToken, Conversation};
//! use flashmind_types::{AgentEvent, AgentInput, LlmProvider};
//! use futures_util::StreamExt;
//!
//! async fn chat(provider: Arc<dyn LlmProvider>) {
//!     let mut agent = Agent::builder(provider).build().await;
//!     let mut conversation = Conversation::new();
//!     conversation.set_system("You are helpful.");
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
//! }
//! ```
//!
//! Consume the stream until it ends to finish the turn. Dropping it cancels
//! in-flight work. Clone the cancellation token before starting a turn if
//! another task needs to cancel it. The stream borrows both the agent and the
//! conversation, so drop it before starting another turn.
//!
//! The optional `subagent` feature enables child-agent management. `full`
//! enables it as well. No features are enabled by default.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Agent`] | Owns provider, tool registry, and LLM config. Stateless between turns. |
//! | [`AgentBuilder`] | Builder for constructing an [`Agent`] with sensible defaults. |
//! | [`Conversation`] | In-memory IR of the current session's entries. |
//! | [`ConversationEntry`] | User input, assistant output, tool results, and summaries |
//! | [`EntryKind`] | Discriminant for entry variants with metadata. |

pub mod agent;
pub mod compaction;
pub mod conversation;
mod recovery;
pub mod streaming;
#[cfg(feature = "subagent")]
pub mod subagent;
mod tool_execution;

pub use agent::{
    Agent, AgentBuilder, AgentRun, Checkpoint, MAX_CONTINUATIONS, RunBoundary,
    check_iteration_limits, handle_llm_error,
};
pub use conversation::{CompactionResult, Conversation, ConversationEntry, EntryKind};
pub use recovery::RecoveryAttempts;
#[cfg(feature = "subagent")]
pub use subagent::{AgentHandle, AgentManager, AgentStatus, SpawnBuilder};
pub use tokio_util::sync::CancellationToken;
pub use tool_execution::{CompletedToolCall, ToolBatch};
