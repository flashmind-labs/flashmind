//! Terminal UI primitives for building interactive agent CLIs.
//!
//! This crate provides a streaming REPL that renders [`flashmind_types::AgentEvent`] output in
//! real time, handles multi-line user input with proper terminal cursor
//! positioning, and supports cancellation of in-flight agent turns.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐     ┌─────────────┐     ┌──────────────┐
//! │  Repl    │────►│ EventRender │────►│   ratatui    │
//! │          │     │             │     │  (crossterm) │
//! │ TextArea │◄────│ Spinner     │     └──────────────┘
//! └──────────┘     └─────────────┘           │
//!                                            ▼
//!                                      ┌──────────┐
//!                                      │ Terminal │
//!                                      └──────────┘
//! ```
//!
//! - **[`Repl`]** — orchestrates the read-eval-print loop: reads user input via
//!   [`TextArea`], streams agent responses, and renders events character-by-character.
//! - **[`EventRenderer`]** — translates [`flashmind_types::AgentEvent`]
//!   variants into styled [`ratatui::text::Line`] values (text deltas, tool status,
//!   diffs, errors, etc.).
//! - **[`TextArea`]** — a multi-line text input widget with word-aware cursor
//!   navigation, line wrapping, and placeholder support.
//! - **[`Spinner`]** — animated Unicode spinner displayed while the agent is thinking.
//! - **[`styles`]** — shared [`ratatui::style::Style`] constants for consistent coloring.
//! - **[`term`]** — low-level helpers for printing styled lines and rendering widgets
//!   directly to stdout without a full ratatui terminal backend.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use flashmind_tui::{Repl, ReplConfig, ReplEvent};
//!
//! let mut repl = Repl::new(ReplConfig {
//!     prompt: "▸".to_string(),
//!     greeting: Some("Flashmind v0.1 — type /help for commands".into()),
//! });
//!
//! repl.print_greeting()?;
//!
//! loop {
//!     match repl.read_input()? {
//!         ReplEvent::UserInput(text) => {
//!             let stream = agent.start(&mut conversation, AgentInput::user(&text));
//!             repl.stream_response(stream).await?;
//!         }
//!         ReplEvent::Quit => break,
//!     }
//! }
//! ```
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Repl`] | Main REPL loop: input, streaming output, cancellation |
//! | [`ReplConfig`] | Configuration (prompt string, optional greeting) |
//! | [`ReplEvent`] | Result of `read_input()` — user text or quit signal |
//! | [`EventRenderer`] | Converts [`flashmind_types::AgentEvent`] to styled lines |
//! | [`TextArea`] | Multi-line, wrap-aware terminal text input widget |
//! | [`Spinner`] | Animated thinking indicator |
//! | [`Tui`] | Low-level terminal wrapper for styled output without full-screen mode |
//!
//! # Design notes
//!
//! Unlike traditional TUI frameworks that use a full-screen render loop, this crate
//! uses an *append-mode* approach: output is printed incrementally to stdout and the
//! cursor is positioned at the bottom for the input widget.  This means the terminal
//! scrollback is preserved naturally and the UI behaves like a chat client rather than
//! a dashboard.
//!
//! # Dependencies
//!
//! - **[ratatui](https://crates.io/crates/ratatui)** — widget toolkit and styling primitives
//! - **[crossterm](https://crates.io/crates/crossterm)** — terminal control (accessed through ratatui)
//! - **[tokio](https://crates.io/crates/tokio)** — async runtime for streaming agent responses
//! - **[unicode-width](https://crates.io/crates/unicode-width)** — correct width calculation for CJK and other wide characters

pub mod event_render;
#[cfg(feature = "markdown")]
pub mod markdown;
#[cfg(feature = "markdown")]
pub mod markdown_render;
pub mod styles;
pub mod term;
pub mod widgets;

pub use event_render::EventRenderer;
pub use term::Tui;
pub use widgets::{Repl, ReplConfig, ReplEvent, Spinner, TextArea};
