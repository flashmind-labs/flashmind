//! Session persistence for conversation history.
//!
//! Feature-gated behind `session`. Stores conversation entries in JSONL,
//! supporting load, save, branch, and prune operations.

#[cfg(feature = "session-sqlite-import")]
mod legacy;
#[cfg(feature = "session-sqlite-import")]
pub mod schema;
pub mod store;
pub mod types;

pub use store::{SessionStore, SessionSummary};
pub use types::{SessionEntry, SessionEntryKind};
