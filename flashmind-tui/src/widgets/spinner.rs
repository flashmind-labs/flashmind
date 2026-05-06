//! Animated spinner widget.
//!
//! Provides a Unicode-based animated spinner for indicating that the agent is
//! processing.  Each call to [`tick`][Spinner::tick] advances to the next frame.
//!
//! The spinner uses 10 Braille-pattern frames (`⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏`) that
//! create a smooth rotating animation when cycled at ~80ms intervals.
//!
//! # Usage
//!
//! The spinner is typically used in a tick-driven loop alongside a tokio interval:
//!
//! ```rust,ignore
//! let mut spinner = Spinner::new();
//! loop {
//!     let line = spinner.line("thinking...");
//!     // render `line` to terminal...
//! }
//! ```

use ratatui::text::{Line, Span};

use crate::styles;

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// An animated terminal spinner.
///
/// Cycles through a set of Unicode frames on each [`tick`][Spinner::tick] call.
/// Use [`line`][Spinner::line] to get a fully styled line with a label.
pub struct Spinner {
    frame: usize,
}

impl Spinner {
    /// Create a new spinner at the first frame.
    pub fn new() -> Self {
        Self { frame: 0 }
    }

    /// Advance to the next frame and return the current character.
    pub fn tick(&mut self) -> char {
        let ch = FRAMES[self.frame % FRAMES.len()];
        self.frame += 1;
        ch
    }

    /// Return a styled line containing the spinner character and a label.
    ///
    /// The spinner character is rendered in the agent style (`S_AGENT`) and the
    /// label in dim text (`S_DIM`).
    pub fn line(&mut self, label: &str) -> Line<'static> {
        let ch = self.tick();
        Line::from(vec![
            Span::styled(format!("{ch} "), styles::S_AGENT),
            Span::styled(label.to_string(), styles::S_DIM),
        ])
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}
