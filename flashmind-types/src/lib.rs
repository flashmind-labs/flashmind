//! Shared types for the Flashmind AI agent framework.
//!
//! This crate defines the traits, wire types, and event structures shared by all
//! Flashmind crates. It is the foundation that both library consumers and provider
//! implementations depend on.
//!
//! # Core traits
//!
//! | Trait | Purpose |
//! |-------|---------|
//! | [`LlmProvider`] | Streaming completions from any LLM backend |
//! | [`Tool`](tool::Tool) | Function-calling tools the agent can invoke |
//! | [`MemoryProvider`] | Store/search/forget long-term memories |
//!
//! # Modules
//!
//! | Module | Types |
//! |--------|-------|
//! | [`event`] | [`AgentEvent`], [`AgentInput`], [`TurnStatus`], [`TurnUsage`] — telemetry |
//! | [`llm`] | [`LlmProvider`], [`CompletionRequest`], [`StreamEvent`], [`FinishReason`] |
//! | [`memory`] | [`MemoryEntry`], [`MemoryMetadata`], [`MemoryProvider`] |
//! | [`message`] | [`Message`], [`Role`], [`ToolCall`], [`ContentPart`] — LLM wire format |
//! | [`model`] | [`Model`], [`Provider`], [`SamplingParams`], [`ReasoningLevel`] |
//! | [`tool`] | [`Tool`], [`ToolRegistry`], [`ToolContext`], [`ToolResult`](tool::ToolResult) |
//! | [`stream`] | [`AgentStream`], [`Outcome`] — stream-with-final-result pattern |
//! | [`utils`] | UTF-8-safe string truncation helpers |
//!
//! # Example: implementing a provider
//!
//! ```rust,ignore
//! use async_trait::async_trait;
//! use flashmind_types::{
//!     CompletionRequest, CompletionStream, LlmProvider, Provider, StreamEvent, FinishReason,
//! };
//!
//! struct MyProvider;
//!
//! #[async_trait]
//! impl LlmProvider for MyProvider {
//!     fn name(&self) -> &str { "my-provider" }
//!     fn provider(&self) -> Provider { Provider::OpenAi }
//!
//!     fn complete(&self, request: CompletionRequest) -> CompletionStream {
//!         Box::pin(async_stream::stream! {
//!             yield Ok(StreamEvent::ContentDelta("Hello!".into()));
//!             yield Ok(StreamEvent::Finished(FinishReason::Stop));
//!         })
//!     }
//! }
//! ```

pub mod error;
pub mod event;
pub mod llm;
pub mod memory;
pub mod message;
pub mod model;
pub mod stream;
pub mod tool;
pub mod utils;
pub use error::ParseError;
pub use event::{
    AgentEvent, AgentInput, InjectEvent, InjectQueue, Source, TurnResult, TurnStatus, TurnUsage,
};
pub use llm::{
    AudioFormat, CompletionRequest, CompletionResponse, CompletionStream, FinishReason,
    LlmProvider, ModelCapabilities, ModelInfo, ProviderRegistry, StreamEvent, TokenUsage,
    ToolDefinition, TtsRequest, Voice,
};
pub use memory::{MemoryEntry, MemoryMetadata, MemoryProvider};
pub use message::{ContentPart, Message, Role, ToolCall, ToolResult};
pub use model::{AgentLlmConfig, AliasedModel, Model, Provider, ReasoningLevel, SamplingParams};
pub use stream::{AgentStream, Outcome};
pub use tool::{ForbiddenCmd, Tool, ToolContext, ToolRegistry};
