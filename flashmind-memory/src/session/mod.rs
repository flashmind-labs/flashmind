//! Session persistence for conversation history.
//!
//! `session` provides JSONL storage; `session-sqlite` provides SQLite storage.
//! Both support load, save, branch, and prune operations independently.

#[cfg(any(feature = "session-sqlite", feature = "session-sqlite-import"))]
pub mod schema;
#[cfg(any(feature = "session-sqlite", feature = "session-sqlite-import"))]
mod sqlite;
#[cfg(feature = "session")]
pub mod store;
pub mod types;

#[cfg(feature = "session-sqlite")]
pub use sqlite::SqliteSessionStore;
#[cfg(feature = "session")]
pub use store::SessionStore;
pub use types::{SessionEntry, SessionEntryKind, SessionSummary};
