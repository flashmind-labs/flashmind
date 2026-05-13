//! Input handling — key events, history navigation, reverse search.

use super::{HistRecord, TuiAction, TuiApp};
use flashmind_tui::styles::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Style};
use ratatui::text::Span;

impl TuiApp<'_> {
    pub(super) fn handle_key(&mut self, key: KeyEvent) -> TuiAction {
        // Reverse search mode intercepts all keys
        if self.search.active {
            return self.handle_search_key(key);
        }

        match key {
            // Ctrl-R: enter reverse search
            KeyEvent {
                code: KeyCode::Char('r'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if !self.history.entries.is_empty() => {
                self.search_start();
                TuiAction::None
            }

            // Esc: cancel
            KeyEvent {
                code: KeyCode::Esc, ..
            } => TuiAction::Cancel,

            // Ctrl+L: force full redraw
            KeyEvent {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.invalidate();
                TuiAction::None
            }

            // Ctrl-D: quit
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => TuiAction::Quit,

            // Ctrl-C: clear textarea, or quit if empty
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                if !self.textarea.is_empty() {
                    self.textarea.clear();
                    TuiAction::None
                } else {
                    TuiAction::Quit
                }
            }

            // Enter: submit (unless Shift/Alt held for newline)
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
                let text = self.textarea.text();
                if text.is_empty() {
                    return TuiAction::None;
                }

                let record = HistRecord {
                    content: text.clone(),
                };
                self.append_history(&record);
                self.history.entries.push(record);
                self.history.pos = None;

                self.textarea.clear();
                TuiAction::Submit(text)
            }

            // Shift+Enter or Alt+Enter: insert newline
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
                self.textarea.insert_newline();
                TuiAction::None
            }

            // Ctrl+J: insert newline
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.textarea.insert_newline();
                TuiAction::None
            }

            // Up arrow: history (when input is empty or browsing history)
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } if (self.textarea.is_empty() || self.history.pos.is_some())
                && !self.history.entries.is_empty() =>
            {
                self.history_up();
                TuiAction::None
            }

            // Down arrow: history
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } if self.textarea.is_empty() || self.history.pos.is_some() => {
                self.history_down();
                TuiAction::None
            }

            // Tab: autocomplete /commands
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.handle_tab_completion();
                TuiAction::None
            }

            // All other keys: pass to textarea
            _ => {
                self.completion.candidates.clear();
                self.textarea.input(key);
                TuiAction::None
            }
        }
    }

    // ========================================================================
    // History
    // ========================================================================

    fn history_up(&mut self) {
        let pos = match self.history.pos {
            Some(p) => {
                if p == 0 {
                    return;
                }
                p - 1
            }
            None => {
                self.history.saved_input = self.textarea.text();
                self.history.entries.len() - 1
            }
        };

        self.history.pos = Some(pos);
        let content = self.history.entries[pos].content.clone();
        self.set_textarea_content(&content);
    }

    fn history_down(&mut self) {
        let Some(pos) = self.history.pos else { return };

        if pos >= self.history.entries.len() - 1 {
            self.history.pos = None;
            let saved = self.history.saved_input.clone();
            self.set_textarea_content(&saved);
        } else {
            self.history.pos = Some(pos + 1);
            let content = self.history.entries[pos + 1].content.clone();
            self.set_textarea_content(&content);
        }
    }

    fn set_textarea_content(&mut self, content: &str) {
        self.textarea.clear();
        self.textarea.insert_str(content);
    }

    // ========================================================================
    // Tab completion
    // ========================================================================

    fn handle_tab_completion(&mut self) {
        let input = self.textarea.text();
        let Some(partial) = input.strip_prefix('/') else {
            return;
        };

        if self.completion.candidates.is_empty() {
            self.completion.candidates = self
                .slash_commands
                .iter()
                .filter(|cmd| cmd.starts_with(partial))
                .map(|cmd| format!("/{}", cmd))
                .collect();
            self.completion.idx = 0;
        } else {
            self.completion.idx = (self.completion.idx + 1) % self.completion.candidates.len();
        }

        if let Some(completion) = self.completion.candidates.get(self.completion.idx).cloned() {
            self.set_textarea_content(&completion);

            self.textarea.set_block(
                ratatui::widgets::Block::default()
                    .borders(ratatui::widgets::Borders::TOP)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(Span::styled(
                        format!(" {} ", completion),
                        Style::default().fg(Color::Cyan),
                    ))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            );
        }
    }

    // ========================================================================
    // Reverse search (Ctrl+R)
    // ========================================================================

    fn search_start(&mut self) {
        self.search.active = true;
        self.search.query.clear();
        self.search.match_idx = None;
        self.search.saved_input = self.textarea.text();
        self.update_search_title();
    }

    fn search_accept(&mut self) {
        self.search.active = false;
        self.textarea.set_block(self.default_input_block());
    }

    fn search_cancel(&mut self) {
        self.search.active = false;
        self.set_textarea_content(&self.search.saved_input.clone());
        self.textarea.set_block(self.default_input_block());
    }

    fn search_find(&self, start: Option<usize>) -> Option<usize> {
        if self.search.query.is_empty() {
            return None;
        }
        let query_lower = self.search.query.to_lowercase();
        let from = start.unwrap_or(self.history.entries.len());
        (0..from).rev().find(|&i| {
            self.history.entries[i]
                .content
                .to_lowercase()
                .contains(&query_lower)
        })
    }

    fn search_update(&mut self) {
        let found = self.search_find(self.search.match_idx.map(|i| i + 1));
        self.search.match_idx = found;
        if let Some(idx) = found {
            let content = self.history.entries[idx].content.clone();
            self.set_textarea_content(&content);
        }
        self.update_search_title();
    }

    fn search_next(&mut self) {
        if let Some(current) = self.search.match_idx
            && let Some(next) = self.search_find(Some(current))
        {
            self.search.match_idx = Some(next);
            let content = self.history.entries[next].content.clone();
            self.set_textarea_content(&content);
        }
        self.update_search_title();
    }

    fn update_search_title(&mut self) {
        let status = if self.search.match_idx.is_some() || self.search.query.is_empty() {
            "reverse-i-search"
        } else {
            "failing reverse-i-search"
        };
        let title = format!(" ({})`{}' ", status, self.search.query);
        self.textarea.set_block(
            ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::TOP)
                .border_style(S_DIM)
                .title(Span::styled(title, Style::default().fg(Color::Yellow)))
                .padding(ratatui::widgets::Padding::horizontal(1)),
        );
    }

    fn handle_search_key(&mut self, key: KeyEvent) -> TuiAction {
        match key {
            KeyEvent {
                code: KeyCode::Char('r'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.search_next();
                TuiAction::None
            }

            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                self.search_accept();
                TuiAction::None
            }

            KeyEvent {
                code: KeyCode::Esc, ..
            }
            | KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }
            | KeyEvent {
                code: KeyCode::Char('g'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.search_cancel();
                TuiAction::None
            }

            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.search.query.pop();
                self.search.match_idx = None;
                self.search_update();
                TuiAction::None
            }

            KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                self.search.query.push(c);
                self.search_update();
                TuiAction::None
            }

            _ => {
                self.search_accept();
                self.handle_key(key)
            }
        }
    }

    pub(super) fn default_input_block(&self) -> ratatui::widgets::Block<'static> {
        ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::TOP)
            .border_style(S_TEXT)
            .padding(ratatui::widgets::Padding::horizontal(1))
    }

    pub(super) fn command_input_block(&self) -> ratatui::widgets::Block<'static> {
        ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::TOP)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                " command ",
                Style::default().fg(Color::Cyan),
            ))
            .padding(ratatui::widgets::Padding::horizontal(1))
    }
}
