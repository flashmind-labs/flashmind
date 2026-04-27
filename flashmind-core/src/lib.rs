//! Core agent runtime — conversation management, session persistence, and streaming.
//!
//! This crate provides the central agent loop that orchestrates LLM completions,
//! tool execution, conversation compaction, and memory integration. It is designed
//! to be provider-agnostic and works with any [`LlmProvider`](flashmind_types::LlmProvider)
//! implementation.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use flashmind_core::{Agent, Conversation, ConversationEntry};
//! use flashmind_types::{AgentEvent, AgentInput};
//! use futures::StreamExt;
//!
//! // Build an agent from any LlmProvider implementation
//! let mut agent = Agent::builder(provider)
//!     .scope("my-app")
//!     .max_iterations(50)
//!     .build();
//!
//! // Conversation is caller-owned — agent mutates it during the turn
//! let mut conversation = Conversation::new();
//! conversation.prepend(ConversationEntry::system("You are helpful."));
//!
//! // Stream events as the agent processes the turn
//! let stream = agent.start(&mut conversation, AgentInput::user("Hello!"));
//! tokio::pin!(stream);
//! while let Some(event) = stream.next().await {
//!     match event {
//!         AgentEvent::TextDelta(text) => print!("{text}"),
//!         AgentEvent::Done(response) => println!("\n{response}"),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! See the `flashmind` facade crate for runnable examples: `ollama`,
//! `custom_tool`, and `streaming`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │                   Agent                      │
//! │                                              │
//! │  ┌──────────┐    ┌──────────────┐           │
//! │  │Conversation│──►│stream_llm_response│      │
//! │  └──────────┘    └───────┬────────┘          │
//! │                          │                    │
//! │              ┌───────────▼──────────┐        │
//! │              │  execute_tool_calls   │        │
//! │              └───────────┬──────────┘        │
//! │                          │                    │
//! │              ┌───────────▼──────────┐        │
//! │              │    try_compact       │         │
//! │              └──────────────────────┘        │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Agent`] | Owns provider, tool registry, and LLM config. Stateless between turns. |
//! | [`AgentBuilder`] | Builder for constructing an [`Agent`] with sensible defaults. |
//! | [`Conversation`] | In-memory IR of the current session's entries. |
//! | [`ConversationEntry`] | Individual turn: user, assistant, tool result, summary… |
//! | [`EntryKind`] | Discriminant for entry variants with metadata. |
//! | [`SessionStore`] | SQLite-backed session persistence with JSON migration. |

pub mod agent;
pub mod compaction;
pub mod conversation;
pub mod host_env;
pub mod session_store;
pub mod streaming;

pub use agent::{Agent, AgentBuilder};
pub use conversation::{Conversation, ConversationEntry, EntryKind};
pub use session_store::SessionStore;
