//! Press Ctrl-C twice to exit.
//!
//! The first Ctrl-C arms the exit and the widget shows [`EXIT_HINT`]. A second
//! Ctrl-C within [`EXIT_WINDOW`] confirms it. Otherwise the hint expires.

use std::time::{Duration, Instant};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// How long the first Ctrl-C waits for the second.
pub const EXIT_WINDOW: Duration = Duration::from_secs(2);

/// Shown while an exit is armed.
pub const EXIT_HINT: &str = "Press Ctrl-C again to exit";

#[derive(Debug, Default)]
pub struct ExitConfirm {
    armed_at: Option<Instant>,
}

impl ExitConfirm {
    pub fn is_ctrl_c(key: &KeyEvent) -> bool {
        key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
    }

    /// Record a Ctrl-C press. Returns `true` when it confirms an armed exit.
    pub fn press(&mut self) -> bool {
        self.press_at(Instant::now())
    }

    /// Whether the exit hint should be shown.
    pub fn pending(&self) -> bool {
        self.pending_at(Instant::now())
    }

    fn press_at(&mut self, now: Instant) -> bool {
        if self.pending_at(now) {
            self.armed_at = None;
            return true;
        }
        self.armed_at = Some(now);
        false
    }

    fn pending_at(&self, now: Instant) -> bool {
        self.armed_at
            .is_some_and(|at| now.duration_since(at) < EXIT_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_press_inside_window_confirms() {
        let mut exit = ExitConfirm::default();
        let start = Instant::now();
        assert!(!exit.press_at(start));
        assert!(exit.pending_at(start));
        assert!(exit.press_at(start + Duration::from_millis(500)));
        assert!(!exit.pending_at(start + Duration::from_millis(500)));
    }

    #[test]
    fn press_after_window_rearms() {
        let mut exit = ExitConfirm::default();
        let start = Instant::now();
        assert!(!exit.press_at(start));
        let late = start + EXIT_WINDOW;
        assert!(!exit.pending_at(late));
        assert!(!exit.press_at(late));
        assert!(exit.pending_at(late));
    }
}
