//! Agent management — spawn and control child agents.
//!
//! Provides the infrastructure for delegating scoped tasks to independent agent
//! instances that run in parallel as tokio tasks.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`AgentManager`] | Orchestrates active agents with concurrency limits |
//! | [`AgentHandle`] | Tracks a single running agent |
//! | [`SpawnBuilder`] | Fluent builder for spawning agents |
//! | [`AgentStatus`] | Current state of a spawned agent |

pub mod builder;
pub mod handle;
pub mod manager;

pub use builder::SpawnBuilder;
pub use handle::{AgentHandle, AgentStatus};
pub use manager::AgentManager;
