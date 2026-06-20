//! Shared terminal style constants.
//!
//! Defines [`ratatui::style::Style`] values used throughout the crate for
//! consistent coloring of agent output, tool results, user input, and errors.
//!
//! # Style reference
//!
//! | Constant | Appearance | Used for |
//! |----------|-----------|----------|
//! | [`S_DIM`] | Dim | Metadata, elapsed times, secondary info |
//! | [`S_TOOL_RUN`] | Yellow | Tool currently running (▶) |
//! | [`S_TOOL_OK`] | Green | Tool completed successfully (✓) |
//! | [`S_TOOL_FAIL`] | Red | Tool failed (✗) |
//! | [`S_ERROR`] | Bold red | Error messages |
//! | [`S_USER`] | Bold green | User input label / prompt prefix |
//! | [`S_USER_ECHO`] | Subtle bg | Echoed user input text |
//! | [`S_AGENT`] | Bold cyan | Agent output, thinking indicator |
//! | [`S_SPAWNED`] | Magenta | Spawned agent event prefix |
//! | [`S_TEXT`] | Default (no overrides) | Plain agent response text |
//! | [`S_DIFF_ADD`] | Green | Diff added lines (+) |
//! | [`S_DIFF_DEL`] | Red | Diff removed lines (-) |
//! | [`S_STATUS`] | Dim italic | Status messages |

use ratatui::style::{Color, Modifier, Style};

/// Dimmed text — used for metadata, elapsed times, and secondary information.
pub const S_DIM: Style = Style::new().add_modifier(Modifier::DIM);

/// Tool currently running — yellow.
pub const S_TOOL_RUN: Style = Style::new().fg(Color::Yellow);

/// Tool completed successfully — green.
pub const S_TOOL_OK: Style = Style::new().fg(Color::Green);

/// Tool failed — red.
pub const S_TOOL_FAIL: Style = Style::new().fg(Color::Red);

/// Error messages — bold red.
pub const S_ERROR: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);

/// User input label (e.g., the prompt prefix) — bold green.
pub const S_USER: Style = Style::new().fg(Color::Green).add_modifier(Modifier::BOLD);

/// User echoed input — subtle highlighted background.
pub const S_USER_ECHO: Style = Style::new().bg(Color::Rgb(40, 40, 46));

/// Agent output / thinking indicator — bold cyan.
pub const S_AGENT: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);

/// Spawned agent event prefix — magenta.
pub const S_SPAWNED: Style = Style::new().fg(Color::Magenta);

/// Plain agent response text — default style (no color or modifier overrides).
pub const S_TEXT: Style = Style::new();

/// Diff added lines — green.
pub const S_DIFF_ADD: Style = Style::new().fg(Color::Green);

/// Diff removed lines — red.
pub const S_DIFF_DEL: Style = Style::new().fg(Color::Red);

/// Successful file-edit diff block — green background (pi-style success block).
/// Applied to the whole `FileDiff` block (path header + added/removed lines).
pub const S_DIFF_BLOCK_OK: Style = Style::new().bg(Color::Green).fg(Color::Black);

/// Status messages — dim italic.
pub const S_STATUS: Style = Style::new()
    .add_modifier(Modifier::DIM)
    .add_modifier(Modifier::ITALIC);
