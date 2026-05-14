//! Shared application logic for Flashmind CLI and desktop.
//!
//! Owns config loading, provider instantiation, tool building,
//! session persistence, display log I/O, and system prompt resolution.

pub mod config;
pub mod display;
pub mod llm;
pub mod memory;
pub mod prompt;
pub mod provider;
pub mod session;
pub mod tools;

pub use config::{AgentConfig, AppConfig, LlmConfig, MemoryConfig, ProviderConfig, ToolsConfig};
pub use display::{DisplayEvent, DisplayLog, ServerMessage};
pub use session::{LocalSession, Sessions};
pub use tools::ToolSet;
