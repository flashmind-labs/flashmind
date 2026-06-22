//! Composable status bar widget.
//!
//! Displays a status line with optional spinner, elapsed time, mode label,
//! toast messages, and arbitrary extra text.  The widget is display-only —
//! call [`StatusBar::tick`] each frame to animate the spinner and expire toasts.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::styles::S_DIM;

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TOAST_LIFETIME: usize = 63;

// ---------------------------------------------------------------------------
// Public types

#[derive(Debug, Clone)]
pub struct StatusBarSection {
    pub text: String,
    pub style: Style,
}

impl StatusBarSection {
    pub fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    pub fn dim(text: impl Into<String>) -> Self {
        Self::new(text, S_DIM)
    }
}

// ---------------------------------------------------------------------------
// Widget

pub struct StatusBar {
    pub base: String,
    pub extra: String,
    pub spinner_active: bool,
    spinner_tick: usize,
    toast: Option<(String, usize)>,
    tick_count: usize,
    sections: Vec<StatusBarSection>,
}

impl StatusBar {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            extra: String::new(),
            spinner_active: false,
            spinner_tick: 0,
            toast: None,
            tick_count: 0,
            sections: Vec::new(),
        }
    }

    pub fn set_extra(&mut self, extra: impl Into<String>) {
        self.extra = extra.into();
    }

    pub fn add_section(&mut self, section: StatusBarSection) {
        self.sections.push(section);
    }

    pub fn clear_sections(&mut self) {
        self.sections.clear();
    }

    pub fn start_spinner(&mut self) {
        self.spinner_active = true;
        self.spinner_tick = 0;
    }

    pub fn stop_spinner(&mut self) {
        self.spinner_active = false;
    }

    pub fn toast(&mut self, message: impl Into<String>) {
        self.toast = Some((message.into(), self.tick_count));
    }

    pub fn tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        if self.spinner_active {
            self.spinner_tick = self.spinner_tick.wrapping_add(1);
        }
        if let Some((_, shown_at)) = &self.toast
            && self.tick_count.wrapping_sub(*shown_at) > TOAST_LIFETIME
        {
            self.toast = None;
        }
    }

    pub fn line(&self, width: u16) -> Line<'static> {
        let mut spans = Vec::new();

        if self.spinner_active {
            let ch = SPINNER[self.spinner_tick % SPINNER.len()];
            spans.push(Span::styled(
                format!("{ch} "),
                Style::default().fg(Color::Yellow),
            ));
        }

        let status_text = self.display_text();
        if !status_text.is_empty() {
            let style = if self.spinner_active {
                Style::default().fg(Color::Yellow)
            } else {
                S_DIM
            };
            spans.push(Span::styled(status_text, style));
        }

        for section in &self.sections {
            spans.push(Span::styled(
                format!(" \u{00b7} {}", section.text),
                section.style,
            ));
        }

        if let Some((msg, _)) = &self.toast {
            let total = width as usize;
            let used: usize = spans.iter().map(|s| s.content.width()).sum();
            // Reserve one leading space; truncate the message to whatever
            // display columns remain so it never overflows or wraps.
            let avail = total.saturating_sub(used + 1);
            let shown = truncate_to_width(msg, avail);
            let toast_text = format!(" {shown}");
            let toast_w = toast_text.width();
            let gap = total.saturating_sub(used + toast_w);
            if gap > 0 {
                spans.push(Span::styled(" ".repeat(gap), S_DIM));
            }
            spans.push(Span::styled(toast_text, Style::default().fg(Color::Green)));
        }

        Line::from(spans)
    }

    fn display_text(&self) -> String {
        match (self.base.is_empty(), self.extra.is_empty()) {
            (true, true) => String::new(),
            (false, true) => self.base.clone(),
            (true, false) => self.extra.clone(),
            (false, false) => format!("{} \u{2502} {}", self.base, self.extra),
        }
    }
}

impl Default for StatusBar {
    fn default() -> Self {
        Self::new("")
    }
}

/// Truncate `text` to at most `max_width` display columns, appending `…` when
/// truncation occurs.  The result's display width never exceeds `max_width`.
fn truncate_to_width(text: &str, max_width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if text.width() <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    let budget = max_width - 1; // reserve a column for the ellipsis
    let mut out = String::new();
    let mut w = 0;
    for ch in text.chars() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > budget {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('\u{2026}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_width(line: &Line) -> usize {
        line.spans.iter().map(|s| s.content.width()).sum()
    }

    #[test]
    fn long_toast_is_truncated_to_width() {
        let mut bar = StatusBar::new("status");
        bar.toast("a very long toast message that should not overflow the bar");
        let line = bar.line(20);
        assert!(
            line_width(&line) <= 20,
            "line width {} exceeds 20",
            line_width(&line)
        );
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            text.contains('\u{2026}'),
            "truncated toast should end with …"
        );
    }

    #[test]
    fn short_toast_renders_in_full() {
        let mut bar = StatusBar::new("");
        bar.toast("ok");
        let line = bar.line(40);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("ok"));
        assert!(!text.contains('\u{2026}'));
    }

    #[test]
    fn truncate_to_width_never_exceeds_budget() {
        assert_eq!(truncate_to_width("abc", 10), "abc");
        assert_eq!(truncate_to_width("abc", 0), "");
        let wide = truncate_to_width("日本語テスト", 5);
        assert!(wide.width() <= 5);
    }
}
