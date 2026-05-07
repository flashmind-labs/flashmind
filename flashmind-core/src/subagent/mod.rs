//! Agent management — spawn, communicate, and control child agents.
//!
//! Provides the infrastructure for delegating tasks to independent agent
//! instances that run in parallel as tokio tasks. Parent-child communication
//! uses the [`InjectQueue`](flashmind_types::InjectQueue) mechanism.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`AgentManager`] | Orchestrates active agents with concurrency limits |
//! | [`AgentHandle`] | Tracks a single running agent |
//! | [`AgentBuilder`] | Fluent builder for spawning agents |
//! | [`AgentStatus`] | Current state of a spawned agent |

pub mod builder;
pub mod handle;
pub mod manager;

pub use builder::SpawnBuilder;
pub use handle::{AgentHandle, AgentStatus};
pub use manager::AgentManager;
