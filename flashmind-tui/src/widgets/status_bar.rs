//! Composable status bar widget.
//!
//! Displays a status line with optional spinner, elapsed time, mode label,
//! toast messages, and arbitrary extra text.  The widget is display-only —
//! call [`StatusBar::tick`] each frame to animate the spinner and expire toasts.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

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
            let used: usize = spans.iter().map(|s| s.content.len()).sum();
            let toast_text = format!(" {msg}");
            let gap = (width as usize).saturating_sub(used + toast_text.len());
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
