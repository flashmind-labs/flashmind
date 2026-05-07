//! Session persistence for conversation history.
//!
//! Feature-gated behind `session`. Stores conversation entries in SQLite,
//! supporting load, save, branch, and prune operations.

pub mod schema;
pub mod store;
pub mod types;

pub use store::{SessionStore, SessionSummary};
pub use types::{SessionEntry, SessionEntryKind};
