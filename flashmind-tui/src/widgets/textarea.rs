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
//! TextArea is used by [`Repl`](crate::widgets::repl::Repl) to collect user input.  It
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
    /// Atomic tokens — indivisible units (e.g. `[image #N]`, `[pasted text +N L]`)
    /// that are deleted as a whole when any edit touches them. Survive cursor
    /// traversal (the cursor skips over them) and never break apart.
    atomic_tokens: Vec<AtomicToken>,
    /// IDs of tokens removed since the last [`drain_removed_token_ids`] call,
    /// so callers can prune backing stores (image data, pasted-text map).
    ///
    /// [`drain_removed_token_ids`]: TextArea::drain_removed_token_ids
    removed_token_ids: Vec<usize>,
    /// Dim suffix rendered after the input's last visual row (e.g. an
    /// autocomplete hint). Pass `""` to [`set_ghost_suffix`] to clear.
    ///
    /// [`set_ghost_suffix`]: TextArea::set_ghost_suffix
    ghost_suffix: String,
    /// Cached terminal width from the last render pass (used for vertical navigation).
    last_known_width: std::cell::Cell<u16>,
}

/// An indivisible token in the textarea (e.g. `[image #N]`).
///
/// Any edit that touches any byte in `start..end` removes the entire token.
/// Token offsets are byte positions within `TextArea::lines()[line]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicToken {
    pub id: usize,
    pub line: usize,
    pub start: usize,
    pub end: usize,
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
            atomic_tokens: Vec::new(),
            removed_token_ids: Vec::new(),
            ghost_suffix: String::new(),
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

    /// Set (or clear) a dim ghost suffix rendered after the input's last visual
    /// row. Used for inline autocomplete hints. Pass `""` to clear.
    pub fn set_ghost_suffix(&mut self, text: &str) {
        self.ghost_suffix = text.to_string();
    }

    /// Return a reference to the current ghost suffix.
    pub fn ghost_suffix(&self) -> &str {
        &self.ghost_suffix
    }

    // ------------------------------------------------------------------
    // Atomic tokens
    // ------------------------------------------------------------------

    /// Insert an atomic `[image #N]` token at the cursor, returning the id.
    ///
    /// The token is indivisible: any edit touching it removes the whole token
    /// (recorded in [`drain_removed_token_ids`]) rather than splitting it.
    ///
    /// [`drain_removed_token_ids`]: TextArea::drain_removed_token_ids
    pub fn insert_image_token(&mut self) -> usize {
        let id = self.atomic_tokens.iter().map(|t| t.id).max().unwrap_or(0) + 1;
        let text = format!("[image #{id}]");
        self.insert_atomic_text(&text, id)
    }

    /// Insert an atomic `[pasted text +N L]` placeholder for a long paste.
    ///
    /// Only the placeholder goes into the textarea; the caller stores the full
    /// pasted text keyed by the returned id and resolves it at submit time.
    pub fn insert_pasted_text_token(&mut self, line_count: usize) -> usize {
        let id = self.atomic_tokens.iter().map(|t| t.id).max().unwrap_or(0) + 1;
        let text = format!("[pasted text +{line_count} L]");
        self.insert_atomic_text(&text, id)
    }

    /// Shared helper: insert `text` as an atomic span at the cursor with `id`.
    fn insert_atomic_text(&mut self, text: &str, id: usize) -> usize {
        let line = self.cursor.0;
        let start = self.cursor.1;
        let end = start + text.len();

        self.lines[line].insert_str(start, text);

        // Shift any tokens on the same line that start at or after `start`.
        for tok in &mut self.atomic_tokens {
            if tok.line == line && tok.start >= start {
                tok.start += text.len();
                tok.end += text.len();
            }
        }

        self.atomic_tokens.push(AtomicToken {
            id,
            line,
            start,
            end,
        });
        self.cursor.1 = end;
        id
    }

    /// Tokens that overlap the byte range `[start, end)` on `line` (indices
    /// into `atomic_tokens`).
    fn overlapping_tokens(&self, line: usize, start: usize, end: usize) -> Vec<usize> {
        self.atomic_tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.line == line && t.start < end && t.end > start)
            .map(|(i, _)| i)
            .collect()
    }

    /// Remove the given token indices (sorted descending), deleting their text
    /// from the line and adjusting the cursor + remaining token positions.
    /// Returns the IDs of removed tokens (empty if none).
    pub(crate) fn remove_tokens(&mut self, mut indices: Vec<usize>) -> Vec<usize> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Sort descending so we can remove from the end first without
        // invalidating earlier indices.
        indices.sort_unstable_by(|a, b| b.cmp(a));

        for idx in &indices {
            let tok = self.atomic_tokens[*idx].clone();

            // Remove the token text from the line.
            self.lines[tok.line].drain(tok.start..tok.end);

            let removed_len = tok.end - tok.start;

            // Adjust cursor if it was inside or after the token.
            if self.cursor.0 == tok.line && self.cursor.1 >= tok.start {
                if self.cursor.1 <= tok.end {
                    self.cursor.1 = tok.start;
                } else {
                    self.cursor.1 -= removed_len;
                }
            }

            // Adjust other tokens on the same line that come after this one.
            for other in &mut self.atomic_tokens {
                if other.line == tok.line && other.start >= tok.end {
                    other.start -= removed_len;
                    other.end -= removed_len;
                }
            }
        }

        // Remove tokens from the vec (descending order preserves indices).
        let mut removed_ids = Vec::with_capacity(indices.len());
        for idx in &indices {
            let id = self.atomic_tokens.remove(*idx).id;
            removed_ids.push(id);
            self.removed_token_ids.push(id);
        }

        removed_ids
    }

    /// Return the IDs of all active atomic tokens.
    pub fn atomic_token_ids(&self) -> Vec<usize> {
        self.atomic_tokens.iter().map(|t| t.id).collect()
    }

    /// Drain token IDs removed since the last call (for pruning backing stores).
    pub fn drain_removed_token_ids(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.removed_token_ids)
    }

    /// Return a reference to all active atomic tokens.
    pub fn atomic_tokens(&self) -> &[AtomicToken] {
        &self.atomic_tokens
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
        let split = self.cursor.1;
        let tail = self.lines[self.cursor.0].split_off(split);
        // Tokens on the split line at/after the split point move to the new
        // line with offsets shifted by `split`.
        let old_line = self.cursor.0;
        for tok in &mut self.atomic_tokens {
            if tok.line == old_line && tok.start >= split {
                tok.line = old_line + 1;
                tok.start -= split;
                tok.end -= split;
            }
        }
        self.cursor.0 += 1;
        self.cursor.1 = 0;
        self.lines.insert(self.cursor.0, tail);
    }

    /// Replace all text with the given string and move the cursor to the end.
    ///
    /// Splits on newlines to populate the logical lines.  If the input is empty,
    /// a single empty line is retained (same as [`clear`](Self::clear)).  Scroll
    /// is reset to 0 after replacement.  All atomic tokens are dropped (their
    /// ids are recorded as removed for the next [`drain_removed_token_ids`]).
    ///
    /// [`drain_removed_token_ids`]: TextArea::drain_removed_token_ids
    pub fn set_text(&mut self, text: &str) {
        self.record_all_removed();
        self.atomic_tokens.clear();
        self.ghost_suffix.clear();
        self.lines = text.split('\n').map(String::from).collect();
        if self.lines.is_empty() {
            self.lines = vec![String::new()];
        }
        self.cursor.0 = self.lines.len() - 1;
        self.cursor.1 = self.lines[self.cursor.0].len();
        self.scroll = 0;
    }

    /// Replace text while keeping the cursor at its current byte position,
    /// clamped to the new content bounds.  All atomic tokens are dropped.
    pub fn set_text_preserve_cursor(&mut self, text: &str) {
        self.record_all_removed();
        self.atomic_tokens.clear();
        self.ghost_suffix.clear();
        let (row, col) = self.cursor;
        self.lines = text.split('\n').map(String::from).collect();
        if self.lines.is_empty() {
            self.lines = vec![String::new()];
        }
        self.cursor.0 = row.min(self.lines.len() - 1);
        self.cursor.1 = col.min(self.lines[self.cursor.0].len());
        while self.cursor.1 > 0 && !self.lines[self.cursor.0].is_char_boundary(self.cursor.1) {
            self.cursor.1 -= 1;
        }
    }

    /// Clear all text and reset the cursor to (0, 0).  All atomic tokens are
    /// dropped (their ids are recorded as removed).
    pub fn clear(&mut self) {
        self.record_all_removed();
        self.atomic_tokens.clear();
        self.ghost_suffix.clear();
        self.lines = vec![String::new()];
        self.cursor = (0, 0);
        self.scroll = 0;
    }

    /// Record every active token id as removed (used by wholesale replacements).
    fn record_all_removed(&mut self) {
        for tok in &self.atomic_tokens {
            self.removed_token_ids.push(tok.id);
        }
    }

    /// Move the cursor to the end of the last line.
    pub fn move_cursor_to_end(&mut self) {
        self.cursor.0 = self.lines.len() - 1;
        self.cursor.1 = self.lines[self.cursor.0].len();
    }

    /// Byte offset of the cursor within the full (newline-joined) text.
    ///
    /// Useful for mention/autocomplete token detection that spans the whole
    /// input rather than a single line.
    pub fn cursor_byte_offset(&self) -> usize {
        let mut off = self.cursor.1;
        for line in &self.lines[..self.cursor.0] {
            off += line.len() + 1; // +1 for '\n'
        }
        off
    }

    /// Move the cursor to a specific byte offset within the full text.
    ///
    /// Out-of-range offsets are clamped to the end of the text.
    pub fn set_cursor_byte_offset(&mut self, mut byte: usize) {
        for (i, line) in self.lines.iter().enumerate() {
            if byte <= line.len() {
                self.cursor = (i, byte);
                return;
            }
            byte -= line.len() + 1; // +1 for '\n'
        }
        self.move_cursor_to_end();
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
                    let hits = self.overlapping_tokens(self.cursor.0, start, self.cursor.1);
                    if self.remove_tokens(hits).is_empty() {
                        self.lines[self.cursor.0].drain(start..self.cursor.1);
                        self.cursor.1 = start;
                    }
                } else if self.cursor.0 > 0 {
                    let current = self.lines.remove(self.cursor.0);
                    self.cursor.0 -= 1;
                    let prev_len = self.lines[self.cursor.0].len();
                    self.lines[self.cursor.0].push_str(&current);
                    let start = prev_word_boundary(&self.lines[self.cursor.0], prev_len);
                    let hits = self.overlapping_tokens(self.cursor.0, start, prev_len);
                    if self.remove_tokens(hits).is_empty() {
                        self.lines[self.cursor.0].drain(start..prev_len);
                        self.cursor.1 = start;
                    }
                }
            }

            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if self.cursor.1 > 0 {
                    let prev = prev_char_boundary(&self.lines[self.cursor.0], self.cursor.1);
                    let hits = self.overlapping_tokens(self.cursor.0, prev, self.cursor.1);
                    if self.remove_tokens(hits).is_empty() {
                        self.lines[self.cursor.0].drain(prev..self.cursor.1);
                        self.cursor.1 = prev;
                    }
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
                    let hits = self.overlapping_tokens(self.cursor.0, self.cursor.1, next);
                    if self.remove_tokens(hits).is_empty() {
                        self.lines[self.cursor.0].drain(self.cursor.1..next);
                    }
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
                // Remove any tokens that extend to/past the cursor on this line.
                let line_len = self.lines[self.cursor.0].len();
                let hits = self.overlapping_tokens(self.cursor.0, self.cursor.1, line_len);
                self.remove_tokens(hits);
                self.lines[self.cursor.0].truncate(self.cursor.1);
            }

            // Ctrl-U: kill to beginning of line
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let hits = self.overlapping_tokens(self.cursor.0, 0, self.cursor.1);
                self.remove_tokens(hits);
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
                let hits = self.overlapping_tokens(self.cursor.0, start, self.cursor.1);
                if self.remove_tokens(hits).is_empty() {
                    self.lines[self.cursor.0].drain(start..self.cursor.1);
                    self.cursor.1 = start;
                }
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

        // Append the ghost suffix (dim) after the last visual row if it fits.
        if !self.ghost_suffix.is_empty() && !visual.is_empty() {
            use ratatui::style::Modifier;
            use unicode_width::UnicodeWidthChar;
            use unicode_width::UnicodeWidthStr;

            let last_idx = visual.len() - 1;
            let used: usize = visual[last_idx]
                .spans
                .iter()
                .map(|s| s.content.as_ref().width())
                .sum();
            let avail = width.saturating_sub(used);
            if avail > 0 {
                let mut shown = String::new();
                let mut w = 0usize;
                for ch in self.ghost_suffix.chars() {
                    let cw = ch.width().unwrap_or(0);
                    if w + cw > avail {
                        break;
                    }
                    shown.push(ch);
                    w += cw;
                }
                if !shown.is_empty() {
                    let ghost_style = Style::default().add_modifier(Modifier::DIM);
                    visual[last_idx]
                        .spans
                        .push(Span::styled(shown, ghost_style));
                }
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

/// If the cursor is at the end of an `[image #N]` token, return the start byte offset.
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

    #[test]
    fn set_text_single_line() {
        let mut ta = TextArea::default();
        ta.set_text("hello world");
        assert_eq!(ta.text(), "hello world");
        assert_eq!(ta.cursor, (0, 11));
        assert_eq!(ta.scroll, 0);
    }

    #[test]
    fn set_text_multiline() {
        let mut ta = TextArea::default();
        ta.set_text("first\nsecond\nthird");
        assert_eq!(ta.lines(), &["first", "second", "third"]);
        assert_eq!(ta.cursor, (2, 5));
    }

    #[test]
    fn set_text_empty() {
        let mut ta = ta("existing content");
        ta.set_text("");
        assert!(ta.is_empty());
        assert_eq!(ta.cursor, (0, 0));
    }

    // -----------------------------------------------------------------
    // Atomic tokens
    // -----------------------------------------------------------------

    #[test]
    fn atomic_insert_image_token() {
        let mut ta = TextArea::default();
        ta.insert_str("hello ");
        ta.insert_image_token();
        assert_eq!(ta.lines[0], "hello [image #1]");
        assert_eq!(ta.cursor, (0, 16));
        assert_eq!(ta.atomic_token_ids(), vec![1]);
    }

    #[test]
    fn atomic_backspace_removes_whole_token() {
        let mut ta = TextArea::default();
        ta.insert_str("hello ");
        ta.insert_image_token();
        // Cursor is right after the token. Backspace touches the last char of the token.
        ta.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(ta.lines[0], "hello ");
        assert_eq!(ta.cursor, (0, 6));
        assert!(ta.atomic_token_ids().is_empty());
        assert_eq!(ta.drain_removed_token_ids(), vec![1]);
    }

    #[test]
    fn atomic_delete_removes_whole_token() {
        let mut ta = TextArea::default();
        ta.insert_str("hello ");
        ta.insert_image_token();
        ta.insert_str(" world");
        // Move cursor to the start of the token (byte 6).
        ta.cursor = (0, 6);
        ta.input(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(ta.lines[0], "hello  world");
        assert_eq!(ta.cursor, (0, 6));
        assert!(ta.atomic_token_ids().is_empty());
    }

    #[test]
    fn atomic_backspace_from_middle_removes_whole_token() {
        let mut ta = TextArea::default();
        ta.insert_image_token();
        // Place cursor inside the token text (e.g. after "[ima").
        ta.cursor = (0, 4);
        ta.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(ta.lines[0], "");
        assert!(ta.atomic_token_ids().is_empty());
    }

    #[test]
    fn atomic_multiple_tokens() {
        let mut ta = TextArea::default();
        ta.insert_image_token();
        ta.insert_str(" ");
        ta.insert_image_token();
        assert_eq!(ta.lines[0], "[image #1] [image #2]");
        assert_eq!(ta.atomic_token_ids(), vec![1, 2]);

        // Delete the second token via backspace at end.
        ta.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(ta.lines[0], "[image #1] ");
        assert_eq!(ta.atomic_token_ids(), vec![1]);
    }

    #[test]
    fn atomic_clear_drops_tokens() {
        let mut ta = TextArea::default();
        ta.insert_image_token();
        assert!(!ta.atomic_token_ids().is_empty());
        ta.clear();
        assert!(ta.atomic_token_ids().is_empty());
        assert_eq!(ta.lines[0], "");
        assert_eq!(ta.drain_removed_token_ids(), vec![1]);
    }

    #[test]
    fn atomic_ctrl_w_removes_token() {
        let mut ta = TextArea::default();
        ta.insert_str("hello ");
        ta.insert_image_token();
        ta.input(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(ta.lines[0], "hello ");
        assert!(ta.atomic_token_ids().is_empty());
    }

    #[test]
    fn atomic_alt_backspace_removes_token() {
        let mut ta = TextArea::default();
        ta.insert_str("hello ");
        ta.insert_image_token();
        ta.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::ALT));
        assert_eq!(ta.lines[0], "hello ");
        assert!(ta.atomic_token_ids().is_empty());
    }

    #[test]
    fn atomic_id_increments() {
        let mut ta = TextArea::default();
        ta.insert_image_token();
        ta.insert_str(" ");
        ta.insert_image_token();
        ta.insert_str(" ");
        ta.insert_image_token();
        assert_eq!(ta.atomic_token_ids(), vec![1, 2, 3]);
    }

    #[test]
    fn atomic_text_after_token_preserved() {
        let mut ta = TextArea::default();
        ta.insert_image_token();
        ta.insert_str(" tail");
        // Delete the token via backspace from inside it.
        ta.cursor = (0, 5); // inside the token
        ta.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(ta.lines[0], " tail");
        assert_eq!(ta.cursor, (0, 0));
    }

    #[test]
    fn atomic_pasted_text_token_inserts() {
        let mut ta = TextArea::default();
        ta.insert_str("note: ");
        let id = ta.insert_pasted_text_token(42);
        assert_eq!(id, 1);
        assert_eq!(ta.lines[0], "note: [pasted text +42 L]");
        assert_eq!(ta.atomic_token_ids(), vec![1]);
    }

    #[test]
    fn atomic_newline_shifts_tokens() {
        let mut ta = TextArea::default();
        ta.insert_str("ab");
        ta.insert_image_token(); // line 0: "ab[image #1]"
        ta.insert_str("cd");
        // cursor at end of line 0. Move to split point between image and "cd".
        // "ab" = 2 bytes, "[image #1]" = 10 bytes → token ends at 12.
        ta.cursor = (0, 12);
        ta.insert_newline();
        assert_eq!(ta.lines, &["ab[image #1]", "cd"]);
        // Token moved to line 0 (start < split stays), so still line 0.
        assert_eq!(ta.atomic_tokens().len(), 1);
        assert_eq!(ta.atomic_tokens()[0].line, 0);
        assert_eq!(ta.atomic_tokens()[0].start, 2);
        assert_eq!(ta.atomic_tokens()[0].end, 12);
    }

    #[test]
    fn atomic_newline_moves_token_to_new_line() {
        let mut ta = TextArea::default();
        ta.insert_str("ab"); // line 0
        // Place cursor after "ab" and insert a token that will end up after the split.
        ta.cursor = (0, 2);
        // Insert text then a token so the token starts at/after the split point.
        ta.insert_image_token(); // line 0: "ab[image #1]", cursor at 12
        // Now split before the token: move cursor back to 2 and split there.
        ta.cursor = (0, 2);
        ta.insert_newline();
        assert_eq!(ta.lines, &["ab", "[image #1]"]);
        assert_eq!(ta.atomic_tokens()[0].line, 1);
        assert_eq!(ta.atomic_tokens()[0].start, 0);
        assert_eq!(ta.atomic_tokens()[0].end, 10);
    }

    #[test]
    fn ghost_suffix_roundtrip() {
        let mut ta = TextArea::default();
        ta.set_ghost_suffix(" <args>");
        assert_eq!(ta.ghost_suffix(), " <args>");
        ta.set_ghost_suffix("");
        assert_eq!(ta.ghost_suffix(), "");
    }

    #[test]
    fn clear_resets_ghost_suffix() {
        // After submitting a slash command, clear() is called. A stale ghost
        // suffix must not linger — otherwise it renders on the empty textarea
        // (the empty-line shortcut only short-circuits when a placeholder is
        // set, so a stale suffix would be appended to the blank line).
        let mut ta = TextArea::default();
        ta.insert_str("/reasoning");
        ta.set_ghost_suffix(" <off|low|medium|high>");
        ta.clear();
        assert_eq!(ta.ghost_suffix(), "");
        assert!(ta.atomic_tokens().is_empty());
    }

    #[test]
    fn set_text_resets_ghost_suffix() {
        let mut ta = TextArea::default();
        ta.set_ghost_suffix(" stale");
        ta.set_text("new text");
        assert_eq!(ta.ghost_suffix(), "");
    }

    #[test]
    fn resolve_tokens_expands_pasted_text_inline() {
        // Reproduces the submit path: a long-paste placeholder must expand to
        // its full backing text in the submitted string.
        let mut ta = TextArea::default();
        ta.insert_str("before ");
        let id = ta.insert_pasted_text_token(3);
        ta.insert_str(" after");
        // The backing store is held by the caller (Repl). Simulate resolve:
        // gather tokens, splice full text in place of the placeholder, drop
        // the [pasted text +N L] label.
        let line = &ta.lines()[0];
        let tok = ta.atomic_tokens().iter().find(|t| t.id == id).unwrap();
        let full = "line1\nline2\nline3";
        let resolved = format!("{}{}{}", &line[..tok.start], full, &line[tok.end..]);
        assert_eq!(resolved, "before line1\nline2\nline3 after");
    }

    #[test]
    fn cursor_screen_pos_respects_textarea_width() {
        // Regression: draw_input used the full terminal width for cursor
        // computation instead of the pet-aware textarea width, so wrapped
        // lines placed the cursor on the wrong row / into the pet column.
        // The cursor's visual row must be computed against the width the
        // text actually wraps at.
        let mut ta = TextArea::default();
        ta.insert_str("abcdefghijklmnopqrstuvwxyz"); // 26 chars
        // At width 13 (pet takes 14 cols of a 27-col terminal), 26 chars wrap
        // to 2 rows; cursor at end sits on row 1.
        ta.move_cursor_to_end();
        let (row, _col) = ta.visual_cursor_pos(13);
        assert_eq!(row, 1, "cursor should be on the second wrapped row");
        // At width 26 the whole line fits on one row.
        let (row, _col) = ta.visual_cursor_pos(26);
        assert_eq!(row, 0);
    }
}
