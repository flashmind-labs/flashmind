//! Shared background agents: title enrichment, memory capture.

mod capture;
mod enrichment;
mod post_turn;
mod prompt;
mod types;

pub use capture::{extract_exchange, spawn_capture_agent};
pub use enrichment::{sanitize_title, spawn_title_enrichment};
pub use post_turn::{spawn_post_turn, spawn_post_turn_with_entries};
pub use types::PostTurnEvent;
