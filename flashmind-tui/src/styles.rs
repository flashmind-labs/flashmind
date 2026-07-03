//! Shared terminal style constants.
//!
//! Defines [`ratatui::style::Style`] values used throughout the crate for
//! consistent coloring of agent output, tool results, user input, and errors.
//!
//! # Style reference
//!
//! | Constant | Appearance | Used for |
//! |----------|-----------|----------|
//! | [`S_ACCENT`] | Teal | Primary accent for glyphs and interactive elements |
//! | [`S_DIMMER`] | Grey | Dividers, status-line separators, secondary chrome |
//! | [`S_DIM`] | Dim | Metadata, elapsed times, secondary info |
//! | [`S_TOOL_RUN`] | Teal | Tool currently running (◌) |
//! | [`S_TOOL_OK`] | Teal | Tool completed successfully (✓) |
//! | [`S_TOOL_FAIL`] | Red | Tool failed (✗) |
//! | [`S_ERROR`] | Bold red | Error messages |
//! | [`S_USER`] | Bold green | User input label / prompt prefix |
//! | [`S_USER_ECHO`] | Subtle bg | Echoed user input text |
//! | [`S_AGENT`] | Bold teal | Agent output, thinking indicator |
//! | [`S_SPAWNED`] | Magenta | Spawned agent event prefix |
//! | [`S_TEXT`] | Default (no overrides) | Plain agent response text |
//! | [`S_DIFF_ADD`] | Teal | Diff added lines (+) |
//! | [`S_DIFF_DEL`] | Red | Diff removed lines (-) |
//! | [`S_STATUS`] | Dim italic | Status messages |

use ratatui::style::{Color, Modifier, Style};

/// Primary accent (teal). Used sparingly for the tool glyph, agent name,
/// active spinner, and the input prompt marker.
pub const S_ACCENT: Style = Style::new().fg(Color::Rgb(45, 191, 179));

/// Quiet grey for dividers, status-line separators, and secondary chrome.
/// One step above `S_DIM` in presence; used where a hairline should read as
/// structure rather than default-bright.
pub const S_DIMMER: Style = Style::new().fg(Color::Rgb(110, 110, 120));

/// Dimmed text — used for metadata, elapsed times, and secondary information.
pub const S_DIM: Style = Style::new().add_modifier(Modifier::DIM);

/// Tool currently running: teal ◌ glyph.
pub const S_TOOL_RUN: Style = S_ACCENT;

/// Tool completed successfully: teal ✓ glyph.
pub const S_TOOL_OK: Style = S_ACCENT;

/// Tool failed: soft red ✗ glyph.
pub const S_TOOL_FAIL: Style = Style::new().fg(Color::Rgb(224, 108, 117));

/// Error messages — bold red.
pub const S_ERROR: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);

/// User input label (e.g., the prompt prefix) — bold green.
pub const S_USER: Style = Style::new().fg(Color::Green).add_modifier(Modifier::BOLD);

/// User echoed input — subtle highlighted background.
pub const S_USER_ECHO: Style = Style::new().bg(Color::Rgb(40, 40, 46));

/// Agent output / thinking indicator: teal accent, bold.
pub const S_AGENT: Style = Style::new()
    .fg(Color::Rgb(45, 191, 179))
    .add_modifier(Modifier::BOLD);

/// Spawned agent event prefix — magenta.
pub const S_SPAWNED: Style = Style::new().fg(Color::Magenta);

/// Plain agent response text — default style (no color or modifier overrides).
pub const S_TEXT: Style = Style::new();

/// Diff header / added-line accent: teal.
pub const S_DIFF_ADD: Style = S_ACCENT;

/// Diff removed lines — red.
pub const S_DIFF_DEL: Style = Style::new().fg(Color::Red);

/// Successful file-edit diff block — subtle dark green background.
/// Applied to the `FileDiff` path header and added lines.  Removed lines use
/// [`S_DIFF_BLOCK_DEL`] (dark red background) so additions/deletions are both
/// visually distinct within the success block.
pub const S_DIFF_BLOCK_OK: Style = Style::new().bg(Color::Rgb(30, 50, 35));

/// Removed lines within a successful diff block — subtle dark red background.
pub const S_DIFF_BLOCK_DEL: Style = Style::new().bg(Color::Rgb(55, 30, 30));

/// Unchanged context lines in a diff — dim default text (no background), so
/// surrounding code reads as quiet locality without competing with the
/// green/red change lines.
pub const S_DIFF_CONTEXT: Style = Style::new().add_modifier(Modifier::DIM);

/// Active subagent header: teal accent (same hue as agent, not bold).
pub const S_SUBAGENT: Style = S_ACCENT;

/// Status messages — dim italic.
pub const S_STATUS: Style = Style::new()
    .add_modifier(Modifier::DIM)
    .add_modifier(Modifier::ITALIC);

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    #[test]
    fn accent_is_teal() {
        assert_eq!(S_ACCENT.fg, Some(Color::Rgb(45, 191, 179)));
    }

    #[test]
    fn dimmer_is_grey() {
        assert_eq!(S_DIMMER.fg, Some(Color::Rgb(110, 110, 120)));
    }

    #[test]
    fn tool_run_uses_accent() {
        assert_eq!(S_TOOL_RUN.fg, Some(Color::Rgb(45, 191, 179)));
    }
}
