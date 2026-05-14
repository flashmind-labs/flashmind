//! Types for post-turn background agent feedback.

use serde::Serialize;

/// Events emitted by background agents after a turn completes.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PostTurnEvent {
    /// Session title was generated/updated.
    TitleSet { title: String },
    /// A memory was stored by the capture agent.
    MemoryStored { content: String },
    /// A memory was forgotten (replaced/outdated) by the capture agent.
    MemoryForgotten { id: String },
    /// Capture agent finished its run.
    CaptureComplete { stored: usize, forgotten: usize },
}
