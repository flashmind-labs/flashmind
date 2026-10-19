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
//! | [`S_DIM`] | Warm grey | Metadata, secondary info |
//! | [`S_ELAPSED`] | Amber | Tool and turn durations |
//! | [`S_HINT`] | Soft blue | Collapsed-output hints |
//! | [`S_TOOL_RUN`] | Teal | Tool currently running (◌) |
//! | [`S_TOOL_OK`] | Teal | Tool completed successfully (✓) |
//! | [`S_TOOL_FAIL`] | Red | Tool failed (✗) |
//! | [`S_ERROR`] | Bold red | Error messages |
//! | [`S_USER`] | Bold green | User input label / prompt prefix |
//! | [`S_USER_ECHO`] | Subtle bg | Echoed user input text |
//! | [`S_SHELL`] | Yellow | Shell mode prompt, border and command echo |
//! | [`S_SELECTION`] | Blue bg | Mouse selection in the transcript |
//! | [`S_AGENT`] | Bold teal | Agent output, thinking indicator |
//! | [`S_SPAWNED`] | Magenta | Spawned agent event prefix |
//! | [`S_TEXT`] | Default (no overrides) | Plain agent response text |
//! | [`S_DIFF_ADD`] | Teal | Diff added lines (+) |
//! | [`S_DIFF_DEL`] | Red | Diff removed lines (-) |
//! | [`S_STATUS`] | Dim italic | Status messages |

use ratatui::style::{Color, Modifier, Style};

const TEAL: Color = Color::Rgb(56, 222, 205);

/// Primary accent (teal). Used sparingly for the tool glyph, agent name,
/// active spinner, and the input prompt marker.
pub const S_ACCENT: Style = Style::new().fg(TEAL);

/// Quiet grey for dividers, status-line separators, and secondary chrome.
/// One step above `S_DIM` in presence; used where a hairline should read as
/// structure rather than default-bright.
pub const S_DIMMER: Style = Style::new().fg(Color::Rgb(110, 110, 120));

/// Warm grey for metadata and secondary information.
pub const S_DIM: Style = Style::new().fg(Color::Rgb(160, 152, 142));

/// Amber for tool and turn durations.
pub const S_ELAPSED: Style = Style::new().fg(Color::Rgb(229, 192, 123));

/// Soft blue for hints about collapsed output.
pub const S_HINT: Style = Style::new().fg(Color::Rgb(130, 170, 255));

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

/// Shell mode: the input prompt, border and echoed command, in yellow.
pub const S_SHELL: Style = Style::new().fg(Color::Yellow);

/// Selected transcript text: a blue background that keeps the text color.
pub const S_SELECTION: Style = Style::new().bg(Color::Rgb(38, 79, 120));

/// User echoed input — subtle highlighted background.
pub const S_USER_ECHO: Style = Style::new().bg(Color::Rgb(40, 40, 46));

/// Agent output / thinking indicator: teal accent, bold.
pub const S_AGENT: Style = Style::new().fg(TEAL).add_modifier(Modifier::BOLD);

/// Spawned agent event prefix — magenta.
pub const S_SPAWNED: Style = Style::new().fg(Color::Magenta);

/// Plain agent response text — default style (no color or modifier overrides).
pub const S_TEXT: Style = Style::new();

/// Diff header / added-line accent: teal.
pub const S_DIFF_ADD: Style = S_ACCENT;

/// Diff removed lines — red.
pub const S_DIFF_DEL: Style = Style::new().fg(Color::Red);

/// Added diff lines and omission markers use a dark green background.
/// Removed lines use [`S_DIFF_BLOCK_DEL`] for a dark red background.
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
        assert_eq!(S_ACCENT.fg, Some(TEAL));
    }

    #[test]
    fn dimmer_is_grey() {
        assert_eq!(S_DIMMER.fg, Some(Color::Rgb(110, 110, 120)));
    }

    #[test]
    fn tool_run_uses_accent() {
        assert_eq!(S_TOOL_RUN.fg, Some(TEAL));
    }
}
