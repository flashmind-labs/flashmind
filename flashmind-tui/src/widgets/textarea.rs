//! Multi-line text input widget with word-aware cursor navigation.
//!
//! This module provides [`TextArea`], a ratatui-compatible widget that renders
//! into a [`Buffer`] and handles keyboard input directly.  It supports:
//!
//! - **Line wrapping** with correct Unicode width (CJK characters, emojis).
//! - **Word-aware navigation** — Alt+Left/Right moves by word boundaries;
//!   Ctrl+W deletes the previous word; Option+Backspace deletes backward by word.
//! - **Visual vertical movement** — Up/Down arrows navigate visual rows rather
//!   than logical lines, so wrapped lines are handled correctly.
//! - **Placeholder text** displayed when the widget is empty.
//! - **Scroll tracking** — the cursor is kept visible within the widget bounds.
//!
//! The TextArea stores text as a `Vec<String>` (one entry per logical line) and
//! tracks cursor position in byte offsets within each line.  Visual row/column
//! coordinates are computed on demand from the wrapping algorithm.
//!
//! # Integration
//!
//! TextArea is used by [`Repl`][crate::repl::Repl] to collect user input.  It
//! can also be embedded in custom UIs via its [`Widget`] implementation and
//! [`cursor_screen_pos`] method for accurate cursor placement.
//!
//! # Example
//!
//! ```rust,ignore
//! use flashmind_tui::TextArea;
//! use ratatui::widgets::{Block, Borders};
//!
//! let mut ta = TextArea::default();
//! ta.set_block(Block::default().borders(Borders::ALL).title(" Input "));
//! ta.set_placeholder_text("Type here...");
//!
//! // Handle key events
//! ta.input(key_event);
//!
//! // Render via ratatui Widget trait or call render() directly
//! ta.render(area, &mut buffer);
//!
//! // Get cursor position for crossterm placement
//! if let Some((x, y)) = ta.cursor_screen_pos(area) {
//!     execute!(stdout, MoveTo(x, y))?;
//! }
//! ```
//!
//! [`cursor_screen_pos`]: TextArea::cursor_screen_pos

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Widget};

/// A multi-line text input widget with word-aware cursor navigation and line wrapping.
///
/// Stores text as a vector of strings (one per logical line) and tracks the cursor
/// as `(line_index, byte_offset)` within the current line.  Visual row/column
/// positions are computed lazily based on the wrap width.
///
/// See the [module documentation](self) for an overview and examples.
pub struct TextArea<'a> {
    /// Logical lines of text. Each string may wrap to multiple visual rows.
    lines: Vec<String>,
    /// Cursor position as `(line_index, byte_offset)`.
    cursor: (usize, usize),
    /// Scroll offset in visual rows (used when content exceeds widget height).
    scroll: usize,
    /// Optional border/title block rendered around the textarea.
    block: Option<Block<'a>>,
    /// Style applied to the cursor's current line.
    cursor_line_style: Style,
    /// Placeholder text shown when the widget is empty.
    placeholder: String,
    /// Base style for all text spans.
    style: Style,
    /// Cached terminal width from the last render pass (used for vertical navigation).
    last_known_width: std::cell::Cell<u16>,
}

impl<'a> Default for TextArea<'a> {
    fn default() -> Self {
        Self {
            lines: vec![String::new()],
            cursor: (0, 0),
            scroll: 0,
            block: None,
            cursor_line_style: Style::default(),
            placeholder: String::new(),
            style: Style::default(),
            last_known_width: std::cell::Cell::new(80),
        }
    }
}

impl<'a> TextArea<'a> {
    /// Set the border and title block rendered around the textarea.
    pub fn set_block(&mut self, block: Block<'a>) {
        self.block = Some(block);
    }

    /// Set placeholder text displayed when the widget is empty.
    pub fn set_placeholder_text(&mut self, text: &str) {
        self.placeholder = text.to_string();
    }

    /// Set the style applied to the cursor's current visual line.
    pub fn set_cursor_line_style(&mut self, style: Style) {
        self.cursor_line_style = style;
    }

    /// Set the base style for all text spans.
    pub fn set_style(&mut self, style: Style) {
        self.style = style;
    }

    /// Return a reference to the logical lines of text.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Return `true` if the textarea contains no user-entered text.
    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines[0].is_empty()
    }

    /// Return the full text as a single string with newlines between logical lines.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Calculate the total number of visual rows the current content occupies
    /// when wrapped at the given width.
    ///
    /// Accounts for Unicode character widths (e.g., CJK characters take 2 columns).
    pub fn visual_line_count(&self, width: u16) -> usize {
        use unicode_width::UnicodeWidthStr;

        let w = width.max(1) as usize;
        self.lines
            .iter()
            .map(|line| {
                let lw = UnicodeWidthStr::width(line.as_str());
                if lw == 0 { 1 } else { lw.div_ceil(w) }
            })
            .sum()
    }

    /// Return the cursor position as `(line_index, char_column)`.
    ///
    /// The column is a character count (not byte offset) from the start of the line.
    pub fn cursor(&self) -> (usize, usize) {
        let row = self.cursor.0;
        let col = self.lines[row][..self.cursor.1].chars().count();
        (row, col)
    }

    /// Insert a string at the current cursor position, splitting on newlines.
    ///
    /// Carriage returns are stripped, tabs are converted to spaces, and other
    /// control characters are filtered out.
    pub fn insert_str(&mut self, s: &str) {
        let clean: String = s.replace("\r\n", "\n").replace('\r', "\n");
        for (i, chunk) in clean.split('\n').enumerate() {
            if i > 0 {
                self.insert_newline();
            }
            let sanitized: String = chunk
                .chars()
                .map(|c| if c == '\t' { ' ' } else { c })
                .filter(|c| !c.is_control())
                .collect();
            let line = &mut self.lines[self.cursor.0];
            line.insert_str(self.cursor.1, &sanitized);
            self.cursor.1 += sanitized.len();
        }
    }

    /// Insert a newline at the cursor position, splitting the current line.
    pub fn insert_newline(&mut self) {
        let tail = self.lines[self.cursor.0].split_off(self.cursor.1);
        self.cursor.0 += 1;
        self.cursor.1 = 0;
        self.lines.insert(self.cursor.0, tail);
    }

    /// Clear all text and reset the cursor to (0, 0).
    pub fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.cursor = (0, 0);
        self.scroll = 0;
    }

    /// Move the cursor to the end of the last line.
    pub fn move_cursor_to_end(&mut self) {
        self.cursor.0 = self.lines.len() - 1;
        self.cursor.1 = self.lines[self.cursor.0].len();
    }

    /// Process a keyboard event and update the cursor/text accordingly.
    ///
    /// # Supported key bindings
    ///
    /// | Key | Action |
    /// |-----|--------|
    /// | Printable char | Insert character at cursor |
    /// | Backspace | Delete previous character (join lines at start) |
    /// | Option+Backspace | Delete word backward |
    /// | Delete | Delete next character (join lines at end) |
    /// | Left / Right | Move by character, wrapping across lines |
    /// | Alt+Left / Alt+b | Move word backward |
    /// | Alt+Right / Alt+f | Move word forward |
    /// | Up / Down | Move by visual row (respects line wrapping) |
    /// | Home / End | Jump to start/end of logical line |
    /// | Ctrl+A | Beginning of line |
    /// | Ctrl+E | End of line |
    /// | Ctrl+K | Kill to end of line |
    /// | Ctrl+U | Kill to beginning of line |
    /// | Ctrl+W | Delete word backward |
    pub fn input(&mut self, key: KeyEvent) {
        match key {
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                self.lines[self.cursor.0].insert(self.cursor.1, ch);
                self.cursor.1 += ch.len_utf8();
            }

            // Option+Backspace: delete word backward
            KeyEvent {
                code: KeyCode::Backspace,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                if self.cursor.1 > 0 {
                    let start = prev_word_boundary(&self.lines[self.cursor.0], self.cursor.1);
                    self.lines[self.cursor.0].drain(start..self.cursor.1);
                    self.cursor.1 = start;
                } else if self.cursor.0 > 0 {
                    let current = self.lines.remove(self.cursor.0);
                    self.cursor.0 -= 1;
                    let prev_len = self.lines[self.cursor.0].len();
                    self.lines[self.cursor.0].push_str(&current);
                    let start = prev_word_boundary(&self.lines[self.cursor.0], prev_len);
                    self.lines[self.cursor.0].drain(start..prev_len);
                    self.cursor.1 = start;
                }
            }

            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if self.cursor.1 > 0 {
                    let prev = prev_char_boundary(&self.lines[self.cursor.0], self.cursor.1);
                    self.lines[self.cursor.0].drain(prev..self.cursor.1);
                    self.cursor.1 = prev;
                } else if self.cursor.0 > 0 {
                    let current = self.lines.remove(self.cursor.0);
                    self.cursor.0 -= 1;
                    self.cursor.1 = self.lines[self.cursor.0].len();
                    self.lines[self.cursor.0].push_str(&current);
                }
            }

            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                let line_len = self.lines[self.cursor.0].len();
                if self.cursor.1 < line_len {
                    let next = next_char_boundary(&self.lines[self.cursor.0], self.cursor.1);
                    self.lines[self.cursor.0].drain(self.cursor.1..next);
                } else if self.cursor.0 < self.lines.len() - 1 {
                    let next_line = self.lines.remove(self.cursor.0 + 1);
                    self.lines[self.cursor.0].push_str(&next_line);
                }
            }

            // Alt+Left / Alt+b: word backward
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::ALT,
                ..
            }
            | KeyEvent {
                code: KeyCode::Char('b'),
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                if self.cursor.1 > 0 {
                    self.cursor.1 = prev_word_boundary(&self.lines[self.cursor.0], self.cursor.1);
                } else if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                    self.cursor.1 = self.lines[self.cursor.0].len();
                }
            }

            // Alt+Right / Alt+f: word forward
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::ALT,
                ..
            }
            | KeyEvent {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                let line = &self.lines[self.cursor.0];
                if self.cursor.1 < line.len() {
                    self.cursor.1 = next_word_boundary(line, self.cursor.1);
                } else if self.cursor.0 < self.lines.len() - 1 {
                    self.cursor.0 += 1;
                    self.cursor.1 = 0;
                }
            }

            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                if self.cursor.1 > 0 {
                    self.cursor.1 = prev_char_boundary(&self.lines[self.cursor.0], self.cursor.1);
                } else if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                    self.cursor.1 = self.lines[self.cursor.0].len();
                }
            }

            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                let line = &self.lines[self.cursor.0];
                if self.cursor.1 < line.len() {
                    self.cursor.1 = next_char_boundary(line, self.cursor.1);
                } else if self.cursor.0 < self.lines.len() - 1 {
                    self.cursor.0 += 1;
                    self.cursor.1 = 0;
                }
            }

            KeyEvent {
                code: KeyCode::Up, ..
            } => self.move_visual_vertical(-1),

            KeyEvent {
                code: KeyCode::Down,
                ..
            } => self.move_visual_vertical(1),

            KeyEvent {
                code: KeyCode::Home,
                ..
            } => self.cursor.1 = 0,

            KeyEvent {
                code: KeyCode::End, ..
            } => self.cursor.1 = self.lines[self.cursor.0].len(),

            // Ctrl-A: beginning of line
            KeyEvent {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.cursor.1 = 0,

            // Ctrl-E: end of line
            KeyEvent {
                code: KeyCode::Char('e'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.cursor.1 = self.lines[self.cursor.0].len(),

            // Ctrl-K: kill to end of line
            KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.lines[self.cursor.0].truncate(self.cursor.1);
            }

            // Ctrl-U: kill to beginning of line
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.lines[self.cursor.0].drain(..self.cursor.1);
                self.cursor.1 = 0;
            }

            // Ctrl-W: delete word backwards
            KeyEvent {
                code: KeyCode::Char('w'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let start = prev_word_boundary(&self.lines[self.cursor.0], self.cursor.1);
                self.lines[self.cursor.0].drain(start..self.cursor.1);
                self.cursor.1 = start;
            }

            _ => {}
        }
    }

    fn wrap_line(line: &str, width: usize) -> Vec<&str> {
        use unicode_width::UnicodeWidthChar;

        if line.is_empty() {
            return vec![""];
        }

        let mut rows = Vec::new();
        let mut row_start = 0;
        let mut row_width = 0;
        let mut last_space_byte = None;

        for (byte_idx, ch) in line.char_indices() {
            let ch_w = ch.width().unwrap_or(0);
            if row_width + ch_w > width && row_width > 0 {
                if let Some(sp) = last_space_byte {
                    rows.push(&line[row_start..sp]);
                    row_start = sp;
                    while row_start < line.len() && line.as_bytes().get(row_start) == Some(&b' ') {
                        row_start += 1;
                    }
                    row_width = if row_start <= byte_idx {
                        line[row_start..byte_idx]
                            .chars()
                            .map(|c| c.width().unwrap_or(0))
                            .sum()
                    } else {
                        0
                    };
                } else {
                    rows.push(&line[row_start..byte_idx]);
                    row_start = byte_idx;
                    row_width = 0;
                }
                last_space_byte = None;
            }
            if ch == ' ' {
                last_space_byte = Some(byte_idx);
            }
            row_width += ch_w;
        }

        rows.push(&line[row_start..]);
        rows
    }

    fn build_wrapped_lines(&self, width: usize) -> Vec<Line<'a>> {
        if self.lines.len() == 1 && self.lines[0].is_empty() && !self.placeholder.is_empty() {
            return vec![Line::from(Span::styled(
                self.placeholder.clone(),
                Style::default().add_modifier(Modifier::DIM),
            ))];
        }

        let mut visual = Vec::new();
        for line in &self.lines {
            for row in Self::wrap_line(line, width.max(1)) {
                visual.push(Line::from(Span::styled(row.to_string(), self.style)));
            }
        }
        visual
    }

    /// Render the textarea content into the given buffer at the specified area.
    ///
    /// Text is wrapped to fit within the inner area (accounting for any block borders
    /// and padding).  The scroll offset determines which visual rows are visible.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let inner = if let Some(ref block) = self.block {
            let b = block.clone();
            let inner = b.inner(area);
            b.render(area, buf);
            inner
        } else {
            area
        };

        self.last_known_width.set(inner.width);

        if inner.width == 0 || inner.height == 0 {
            return;
        }

        let w = inner.width as usize;
        let wrapped = self.build_wrapped_lines(w);

        let visible: Vec<Line> = wrapped
            .into_iter()
            .skip(self.scroll)
            .take(inner.height as usize)
            .collect();

        for (row, line) in visible.iter().enumerate() {
            if (inner.y + row as u16) >= area.bottom() {
                break;
            }
            line.render(
                Rect::new(inner.x, inner.y + row as u16, inner.width, 1),
                buf,
            );
        }
    }

    /// Calculate the screen position `(x, y)` where the cursor should be drawn.
    ///
    /// Returns `None` if the cursor would fall outside the visible area (e.g., the
    /// widget is too small or the content has scrolled out of view).  The returned
    /// coordinates are absolute within the terminal, not relative to the widget area.
    pub fn cursor_screen_pos(&self, area: Rect) -> Option<(u16, u16)> {
        let inner = if let Some(ref block) = self.block {
            block.clone().inner(area)
        } else {
            area
        };

        if inner.width == 0 || inner.height == 0 {
            return None;
        }

        let (cursor_row, cursor_col) = self.visual_cursor_pos(inner.width);
        let visual_row = cursor_row.saturating_sub(self.scroll);

        if visual_row < inner.height as usize && (cursor_col as u16) < inner.width {
            Some((inner.x + cursor_col as u16, inner.y + visual_row as u16))
        } else {
            None
        }
    }

    fn visual_cursor_pos(&self, width: u16) -> (usize, usize) {
        use unicode_width::UnicodeWidthChar;

        let w = width as usize;
        if w == 0 {
            return (self.cursor.0, 0);
        }

        let mut visual_row = 0;

        for (i, line) in self.lines.iter().enumerate() {
            if i == self.cursor.0 {
                let cursor_byte = self.cursor.1;
                let mut row_start = 0;
                let mut row_width: usize = 0;
                let mut last_space_byte: Option<usize> = None;

                for (byte_idx, ch) in line.char_indices() {
                    let ch_w = ch.width().unwrap_or(0);

                    if row_width + ch_w > w && row_width > 0 {
                        let new_row_start = if let Some(sp) = last_space_byte {
                            let mut rs = sp;
                            while rs < line.len() && line.as_bytes().get(rs) == Some(&b' ') {
                                rs += 1;
                            }
                            rs
                        } else {
                            byte_idx
                        };

                        if cursor_byte >= row_start && cursor_byte < new_row_start {
                            let col: usize = line[row_start..cursor_byte]
                                .chars()
                                .map(|c| c.width().unwrap_or(0))
                                .sum();
                            return (visual_row, col);
                        }

                        visual_row += 1;
                        row_start = new_row_start;
                        row_width = if row_start <= byte_idx {
                            line[row_start..byte_idx]
                                .chars()
                                .map(|c| c.width().unwrap_or(0))
                                .sum()
                        } else {
                            0
                        };
                        last_space_byte = None;
                    }

                    if byte_idx == cursor_byte {
                        let col: usize = line[row_start..byte_idx]
                            .chars()
                            .map(|c| c.width().unwrap_or(0))
                            .sum();
                        return (visual_row, col);
                    }

                    if ch == ' ' {
                        last_space_byte = Some(byte_idx);
                    }
                    row_width += ch_w;
                }

                let col: usize = line[row_start..]
                    .chars()
                    .map(|c| c.width().unwrap_or(0))
                    .sum();
                return (visual_row, col);
            }

            let rows = Self::wrap_line(line, w);
            visual_row += rows.len();
        }

        (visual_row, 0)
    }

    fn cursor_from_visual(
        &self,
        target_row: usize,
        target_col: usize,
        width: usize,
    ) -> (usize, usize) {
        use unicode_width::UnicodeWidthChar;

        if width == 0 {
            return (0, 0);
        }

        let mut visual_row = 0;

        for (li, line) in self.lines.iter().enumerate() {
            let mut row_start = 0;
            let mut row_width: usize = 0;
            let mut last_space_byte: Option<usize> = None;

            for (byte_idx, ch) in line.char_indices() {
                let ch_w = ch.width().unwrap_or(0);

                if row_width + ch_w > width && row_width > 0 {
                    if visual_row == target_row {
                        return (li, Self::byte_at_col(line, row_start, byte_idx, target_col));
                    }

                    let new_row_start = if let Some(sp) = last_space_byte {
                        let mut rs = sp;
                        while rs < line.len() && line.as_bytes().get(rs) == Some(&b' ') {
                            rs += 1;
                        }
                        rs
                    } else {
                        byte_idx
                    };

                    visual_row += 1;
                    row_start = new_row_start;
                    row_width = if row_start <= byte_idx {
                        line[row_start..byte_idx]
                            .chars()
                            .map(|c| c.width().unwrap_or(0))
                            .sum()
                    } else {
                        0
                    };
                    last_space_byte = None;
                }

                if ch == ' ' {
                    last_space_byte = Some(byte_idx);
                }
                row_width += ch_w;
            }

            if visual_row == target_row {
                return (
                    li,
                    Self::byte_at_col(line, row_start, line.len(), target_col),
                );
            }
            visual_row += 1;
        }

        let last = self.lines.len().saturating_sub(1);
        (last, self.lines[last].len())
    }

    fn byte_at_col(line: &str, row_start: usize, row_end: usize, target_col: usize) -> usize {
        use unicode_width::UnicodeWidthChar;

        let mut col = 0;
        for (byte_idx, ch) in line[row_start..row_end].char_indices() {
            if col >= target_col {
                return row_start + byte_idx;
            }
            col += ch.width().unwrap_or(0);
        }
        row_end
    }

    fn move_visual_vertical(&mut self, delta: isize) {
        let w = self.last_known_width.get();
        if w == 0 {
            return;
        }
        let (vis_row, vis_col) = self.visual_cursor_pos(w);
        let total_visual = self
            .lines
            .iter()
            .map(|l| Self::wrap_line(l, w as usize).len())
            .sum::<usize>();

        let target = if delta < 0 {
            if vis_row == 0 {
                return;
            }
            vis_row - 1
        } else {
            if vis_row + 1 >= total_visual {
                return;
            }
            vis_row + 1
        };
        let (line, byte) = self.cursor_from_visual(target, vis_col, w as usize);
        if line < self.lines.len() {
            self.cursor = (line, byte);
        }
    }

    /// Adjust the scroll offset so that the cursor remains visible within the widget.
    ///
    /// If the cursor is above the visible area, scrolls up.  If below, scrolls down
    /// to keep it one row from the bottom.  Does nothing if the cursor is already
    /// within the visible region.
    pub fn ensure_cursor_visible(&mut self, width: u16, height: u16) {
        let total = self.visual_line_count(width);
        let h = height as usize;

        let max_scroll = total.saturating_sub(h);
        if self.scroll > max_scroll {
            self.scroll = max_scroll;
        }

        let (vis_row, _) = self.visual_cursor_pos(width);
        if vis_row < self.scroll {
            self.scroll = vis_row;
        } else if vis_row >= self.scroll + h {
            self.scroll = vis_row - h + 1;
        }
    }
}

impl Widget for &TextArea<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.render(area, buf);
    }
}

fn prev_char_boundary(s: &str, byte_idx: usize) -> usize {
    let mut i = byte_idx.saturating_sub(1);
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn next_char_boundary(s: &str, byte_idx: usize) -> usize {
    let mut i = byte_idx + 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[derive(PartialEq)]
enum CharClass {
    Word,
    Punctuation,
    Whitespace,
}

fn char_class(b: u8) -> CharClass {
    if b.is_ascii_whitespace() {
        CharClass::Whitespace
    } else if b.is_ascii_alphanumeric() || b == b'_' {
        CharClass::Word
    } else {
        CharClass::Punctuation
    }
}

fn next_word_boundary(s: &str, byte_idx: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = byte_idx;
    if i >= bytes.len() {
        return i;
    }
    let cls = char_class(bytes[i]);
    while i < bytes.len() && char_class(bytes[i]) == cls {
        i += 1;
    }
    while i < bytes.len() && char_class(bytes[i]) == CharClass::Whitespace {
        i += 1;
    }
    i
}

fn prev_word_boundary(s: &str, byte_idx: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = byte_idx;
    while i > 0 && char_class(bytes[i - 1]) == CharClass::Whitespace {
        i -= 1;
    }
    if i == 0 {
        return 0;
    }
    let cls = char_class(bytes[i - 1]);
    while i > 0 && char_class(bytes[i - 1]) == cls {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ta(text: &str) -> TextArea<'static> {
        let mut ta = TextArea::default();
        ta.insert_str(text);
        ta
    }

    fn ta_at(text: &str, line: usize, byte_col: usize) -> TextArea<'static> {
        let mut ta = TextArea::default();
        ta.insert_str(text);
        ta.cursor = (line, byte_col);
        ta
    }

    #[test]
    fn cursor_end_of_plain_text() {
        let ta = ta("hello world");
        assert_eq!(ta.cursor, (0, 11));
        assert_eq!(ta.visual_cursor_pos(80), (0, 11));
    }

    #[test]
    fn cursor_middle_of_plain_text() {
        let ta = ta_at("hello world", 0, 5);
        assert_eq!(ta.visual_cursor_pos(80), (0, 5));
    }

    #[test]
    fn cursor_wraps_to_second_row() {
        let ta = ta("abcdefghijklmno");
        assert_eq!(ta.visual_cursor_pos(10), (1, 5));
    }

    #[test]
    fn multiline_cursor() {
        let mut ta = ta("first line");
        ta.insert_newline();
        ta.insert_str("second");
        assert_eq!(ta.cursor, (1, 6));
        assert_eq!(ta.visual_cursor_pos(80), (1, 6));
    }

    #[test]
    fn clear_resets() {
        let mut ta = ta("hello");
        ta.clear();
        assert!(ta.is_empty());
        assert_eq!(ta.cursor, (0, 0));
    }

    #[test]
    fn text_round_trip() {
        let mut ta = ta("first");
        ta.insert_newline();
        ta.insert_str("second");
        assert_eq!(ta.text(), "first\nsecond");
    }
}
