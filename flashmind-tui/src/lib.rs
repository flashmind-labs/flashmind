//! Terminal UI primitives for building interactive agent CLIs.
//!
//! This crate provides a streaming REPL that renders [`flashmind_types::AgentEvent`] output in
//! real time, handles multi-line user input with proper terminal cursor
//! positioning, and supports cancellation of in-flight agent turns.
//!
//! [`Repl`] owns the fullscreen terminal interface. [`EventRenderer`] converts
//! agent events into styled lines; [`TextArea`] handles input, and [`Spinner`]
//! displays activity. Use [`styles`] for shared styles and [`term`] for terminal
//! setup. The optional `markdown` feature enables Markdown rendering; `full`
//! enables it as well.
//!
//! # Quick start
//!
//! ```no_run
//! use std::io;
//!
//! use flashmind_tui::{Repl, ReplConfig, ReplEvent};
//! use flashmind_types::AgentEvent;
//! use futures_util::stream;
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> io::Result<()> {
//! let mut repl = Repl::new(ReplConfig {
//!     greeting: Some("Echo chat. Press Ctrl+D to quit.".into()),
//!     ..ReplConfig::default()
//! });
//!
//! repl.print_greeting()?;
//!
//! loop {
//!     match repl.read_input()? {
//!         ReplEvent::UserInput(text, _images) => {
//!             let cancel = CancellationToken::new();
//!             let events = stream::iter([AgentEvent::TextDelta(format!("You said: {text}"))]);
//!             repl.stream_events(&cancel, events).await?;
//!         }
//!         ReplEvent::Quit => break,
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Run this inside a Tokio runtime attached to an interactive terminal. Replace
//! the echo stream with a pinned agent stream to display model and tool events.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Repl`] | Main REPL loop: input, streaming output, cancellation |
//! | [`ReplConfig`] | Configuration (prompt string, optional greeting) |
//! | [`ReplEvent`] | Result of `read_input()`: user text or quit signal |
//! | [`EventRenderer`] | Converts [`flashmind_types::AgentEvent`] to styled lines |
//! | [`TextArea`] | Multi-line, wrap-aware terminal text input widget |
//! | [`Spinner`] | Animated thinking indicator |
//! | [`Tui`] | Terminal setup and styled output |
//!
//! # Design notes
//!
//! All interactive views use a fullscreen Ratatui terminal. Chat keeps the
//! input fixed at the bottom and scrolls the transcript inside the app.
//!
pub mod event_render;
mod hyperlinks;
#[cfg(feature = "markdown")]
pub mod markdown;
#[cfg(feature = "markdown")]
pub mod markdown_render;
mod present;
mod selection;
pub mod styles;
pub mod term;
mod transcript;
pub mod widgets;

pub use event_render::{EventRenderer, RenderAction, ToolOutput};
pub use term::Tui;
pub use widgets::repl::{ArgumentProvider, MentionProvider};
pub use widgets::{
    CommandInfo, Pet, PetState, PetWidget, Repl, ReplConfig, ReplEvent, Spinner, TextArea,
    random_pet,
};
