//! Shared background agents: title enrichment, memory capture.

mod enrichment;
mod prompt;
mod types;

pub use enrichment::{sanitize_title, spawn_title_enrichment};
pub use types::PostTurnEvent;
