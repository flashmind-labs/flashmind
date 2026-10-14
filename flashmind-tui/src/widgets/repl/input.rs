//! Key handling, pasting, history persistence and clipboard helpers.

use base64::Engine;
use std::io::{self, Write};
use std::path::Path;

use ratatui::crossterm::event::{self, KeyCode, KeyModifiers};

use super::{KeyAction, PastedImage, Repl};
use crate::widgets::textarea::AtomicToken;

impl<'a> Repl<'a> {
    /// Process a key event and return the resulting action. Handles all common
    /// key bindings (history, autocomplete, editing) so both `read_input` and
    /// `stream_response` share identical behaviour.
    pub(super) fn process_key(&mut self, key: event::KeyEvent) -> KeyAction {
        self.dragging_scrollbar = false;
        self.pending_reasoning_click = None;
        if self.selecting_screen
            && let Some(action) = self.screen_selection.handle_event(&event::Event::Key(key))
        {
            self.apply_screen_action(action);
            return KeyAction::Redraw;
        }
        self.screen_selection.clear();
        self.selecting_screen = false;
        if self.selection.is_some() && key.code == KeyCode::Esc {
            self.selection = None;
            return KeyAction::Redraw;
        }
        if key.code == KeyCode::Char('c')
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && self.selection.is_some()
        {
            self.copy_selection();
            return KeyAction::Redraw;
        }
        self.selection = None;
        if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
            let expand = !self.renderer.expand_reasoning();
            if let Err(error) = self.set_expand_reasoning(expand) {
                self.toast = Some((format!("Toggle failed: {error}"), self.tick));
            }
            return KeyAction::Redraw;
        }
        if key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::CONTROL) {
            let expand = !self.renderer.expand_tools();
            if let Err(error) = self.set_expand_tools(expand) {
                self.toast = Some((format!("Toggle failed: {error}"), self.tick));
            }
            return KeyAction::Redraw;
        }
        if key.code == KeyCode::Char('f') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.transcript_search = Some(String::new());
            self.search_match = None;
            return KeyAction::Redraw;
        }
        if let Some(query) = self.transcript_search.as_mut() {
            match key.code {
                KeyCode::Esc => self.transcript_search = None,
                KeyCode::Enter => {
                    self.search_backwards = key.modifiers.contains(KeyModifiers::SHIFT);
                    self.search_pending = true;
                }
                KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    query.push(character);
                    self.search_match = None;
                    self.search_pending = true;
                }
                KeyCode::Backspace => {
                    query.pop();
                    self.search_match = None;
                    self.search_pending = true;
                }
                KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End => {
                    self.transcript_scroll.key(key);
                }
                _ => {}
            }
            return KeyAction::Redraw;
        }
        if self.transcript_scroll.key(key) {
            return KeyAction::Redraw;
        }
        // Reverse-i-search mode
        if self.reverse_search.is_some() {
            match key {
                // Ctrl+R — cycle to next match
                event::KeyEvent {
                    code: KeyCode::Char('r'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    let can_advance = self
                        .reverse_search
                        .as_ref()
                        .is_some_and(|(q, s)| self.find_reverse_match(q, s + 1).is_some());
                    if can_advance {
                        self.reverse_search.as_mut().unwrap().1 += 1;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Enter — accept match
                event::KeyEvent {
                    code: KeyCode::Enter,
                    ..
                } => {
                    self.reverse_search = None;
                    return KeyAction::Redraw;
                }
                // Esc / Ctrl+C — cancel, restore draft
                event::KeyEvent {
                    code: KeyCode::Esc, ..
                }
                | event::KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                } => {
                    self.reverse_search = None;
                    self.set_input_text(&self.history_draft.clone());
                    return KeyAction::Redraw;
                }
                // Backspace — shrink query
                event::KeyEvent {
                    code: KeyCode::Backspace,
                    ..
                } => {
                    if let Some((ref mut query, ref mut skip)) = self.reverse_search {
                        query.pop();
                        *skip = 0;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Printable char — extend query
                event::KeyEvent {
                    code: KeyCode::Char(ch),
                    modifiers,
                    ..
                } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                    if let Some((ref mut query, ref mut skip)) = self.reverse_search {
                        query.push(ch);
                        *skip = 0;
                    }
                    self.apply_reverse_search();
                    return KeyAction::Redraw;
                }
                // Anything else exits search and processes normally
                _ => {
                    self.reverse_search = None;
                }
            }
        }

        if let Some(action) = self.shell_mode_key(key) {
            return action;
        }
        match key {
            // Ctrl+R — enter reverse search
            event::KeyEvent {
                code: KeyCode::Char('r'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if !self.history.is_empty() => {
                self.history_draft = self.input_text();
                self.reverse_search = Some((String::new(), 0));
                self.apply_reverse_search();
                KeyAction::Redraw
            }

            // Ctrl+D on empty — quit
            event::KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if self.textarea.is_empty() => KeyAction::Quit,

            // Ctrl+C — interrupt
            event::KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => KeyAction::Interrupt,

            // Enter
            event::KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } => {
                // Accept dropdown selection. An accepted argument completes
                // the command, so it submits straight away.
                if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
                    let completes_argument = self.argument_start.is_some();
                    if self.accept_dropdown() && !completes_argument {
                        return KeyAction::Redraw;
                    }
                }
                // Shift/Alt+Enter — insert newline
                if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
                    self.textarea.insert_newline();
                    self.dropdown = None;
                    return KeyAction::Redraw;
                }
                if self.textarea.is_empty() {
                    return KeyAction::Nothing;
                }
                // Submit
                let (mut text, images) = self.resolve_tokens();
                if std::mem::take(&mut self.shell_mode) {
                    text.insert(0, '!');
                }
                self.history.push(text.clone());
                self.history_index = None;
                self.history_draft.clear();
                if let Some(path) = &self.config.history_file {
                    append_history(path, &text);
                }
                self.textarea.clear();
                self.dropdown = None;
                self.mention = None;
                self.argument_start = None;
                KeyAction::Submit { text, images }
            }

            // Tab — accept dropdown (slash command or @mention)
            event::KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                if self.accept_dropdown() {
                    KeyAction::Redraw
                } else {
                    KeyAction::Nothing
                }
            }

            // Esc — dismiss dropdown or signal escape
            event::KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                if self.dropdown.is_some() {
                    self.dropdown = None;
                    self.mention = None;
                    self.argument_start = None;
                    KeyAction::Redraw
                } else {
                    KeyAction::Escape
                }
            }

            // Up — dropdown nav / history / textarea
            event::KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if let Some(ref mut dropdown) = self.dropdown
                    && !dropdown.is_empty()
                {
                    dropdown.handle_key(key);
                    return KeyAction::Redraw;
                }
                if self.textarea.cursor().0 == 0 && !self.history.is_empty() {
                    match self.history_index {
                        None => {
                            self.history_draft = self.input_text();
                            let idx = self.history.len() - 1;
                            self.history_index = Some(idx);
                            self.set_input_text(&self.history[idx].clone());
                        }
                        Some(idx) if idx > 0 => {
                            let new_idx = idx - 1;
                            self.history_index = Some(new_idx);
                            self.set_input_text(&self.history[new_idx].clone());
                        }
                        _ => {}
                    }
                } else {
                    self.textarea.input(key);
                }
                KeyAction::Redraw
            }

            // Down — dropdown nav / history / textarea
            event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if let Some(ref mut dropdown) = self.dropdown
                    && !dropdown.is_empty()
                {
                    dropdown.handle_key(key);
                    return KeyAction::Redraw;
                }
                if let Some(idx) = self.history_index {
                    if idx + 1 >= self.history.len() {
                        self.history_index = None;
                        self.set_input_text(&self.history_draft.clone());
                    } else {
                        let new_idx = idx + 1;
                        self.history_index = Some(new_idx);
                        self.set_input_text(&self.history[new_idx].clone());
                    }
                } else {
                    self.textarea.input(key);
                }
                KeyAction::Redraw
            }

            // Ctrl+L — clear screen
            event::KeyEvent {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => KeyAction::ClearScreen,

            // Ctrl+V — paste image (or fall through to textarea for text paste)
            event::KeyEvent {
                code: KeyCode::Char('v'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                if let Some(img) = grab_clipboard_image() {
                    let id = self.textarea.insert_image_token();
                    self.pasted_images.insert(id, img);
                } else {
                    self.textarea.input(key);
                    self.update_autocomplete();
                }
                KeyAction::Redraw
            }

            // Ctrl+S — stash current input
            event::KeyEvent {
                code: KeyCode::Char('s'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let text = self.textarea.text();
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    self.stash.push(text);
                    self.textarea.clear();
                    self.prune_removed_tokens();
                    let n = self.stash.len();
                    self.toast = Some((format!("Stashed {n}"), self.tick));
                }
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Ctrl+P — pop stashed input into the textarea
            event::KeyEvent {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                match self.stash.pop() {
                    Some(text) => {
                        let remaining = self.stash.len();
                        self.textarea.set_text(&text);
                        self.prune_removed_tokens();
                        self.toast = Some((format!("Popped ({remaining} left)"), self.tick));
                    }
                    None => {
                        self.toast = Some(("stash empty".to_string(), self.tick));
                    }
                }
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Ctrl+K — copy textarea contents to the system clipboard
            event::KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let text = self.textarea.text();
                if !text.is_empty() {
                    match copy_to_clipboard(&text) {
                        Ok(()) => {
                            self.toast =
                                Some((format!("Copied {} chars", text.chars().count()), self.tick));
                        }
                        Err(e) => {
                            self.toast = Some((format!("Copy failed: {e}"), self.tick));
                        }
                    }
                }
                KeyAction::Redraw
            }

            // Ctrl+J — insert newline
            event::KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.textarea.insert_newline();
                self.update_autocomplete();
                KeyAction::Redraw
            }

            // Everything else — textarea
            other => {
                self.textarea.input(other);
                self.update_autocomplete();
                KeyAction::Redraw
            }
        }
    }

    /// Find the nth history entry (searching backwards) containing `query`.
    /// Returns the index into `self.history`.
    fn find_reverse_match(&self, query: &str, skip: usize) -> Option<usize> {
        if query.is_empty() {
            return self.history.len().checked_sub(skip + 1);
        }
        let query_lower = query.to_lowercase();
        self.history
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.to_lowercase().contains(&query_lower))
            .nth(skip)
            .map(|(i, _)| i)
    }

    /// Enter shell mode on `!` at the start of an empty input, and leave it
    /// on Backspace in an empty input or on Esc.
    fn shell_mode_key(&mut self, key: event::KeyEvent) -> Option<KeyAction> {
        let plain = !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        let entering = !self.shell_mode && key.code == KeyCode::Char('!') && plain;
        let leaving = self.shell_mode
            && (key.code == KeyCode::Esc && self.dropdown.is_none()
                || key.code == KeyCode::Backspace && self.textarea.is_empty());
        if entering && self.textarea.is_empty() || leaving {
            self.shell_mode = !self.shell_mode;
            self.dropdown = None;
            self.mention = None;
            self.argument_start = None;
            return Some(KeyAction::Redraw);
        }
        None
    }

    /// The input as it is submitted: prefixed with `!` in shell mode.
    pub(super) fn input_text(&self) -> String {
        let text = self.textarea.text();
        if self.shell_mode {
            format!("!{text}")
        } else {
            text
        }
    }

    /// Replace the input, entering shell mode when `text` starts with `!`.
    pub(super) fn set_input_text(&mut self, text: &str) {
        match text.strip_prefix('!') {
            Some(command) => {
                self.shell_mode = true;
                self.textarea.set_text(command);
            }
            None => {
                self.shell_mode = false;
                self.textarea.set_text(text);
            }
        }
    }

    /// Apply the current reverse search state to the textarea.
    fn apply_reverse_search(&mut self) {
        if let Some((ref query, skip)) = self.reverse_search
            && let Some(idx) = self.find_reverse_match(query, skip)
        {
            self.set_input_text(&self.history[idx].clone());
        }
    }

    /// Prune image/paste backing stores for atomic tokens removed by edits.
    /// Called after every textarea edit via [`update_autocomplete`].
    pub(super) fn prune_removed_tokens(&mut self) {
        for id in self.textarea.drain_removed_token_ids() {
            self.pasted_images.remove(&id);
            self.pasted_texts.remove(&id);
        }
    }

    /// Resolve atomic tokens at submit time: expand long-paste placeholders to
    /// their full text, strip `[image #N]` labels, and collect the referenced
    /// images in textual order. Clears the backing maps.
    fn resolve_tokens(&mut self) -> (String, Vec<PastedImage>) {
        let lines = self.textarea.lines();
        let tokens: Vec<AtomicToken> = self.textarea.atomic_tokens().to_vec();

        let mut out = String::new();
        let mut image_ids_in_order: Vec<usize> = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            if li > 0 {
                out.push('\n');
            }
            let mut line_toks: Vec<&AtomicToken> = tokens.iter().filter(|t| t.line == li).collect();
            line_toks.sort_by_key(|t| t.start);
            let mut pos = 0;
            for tok in line_toks {
                out.push_str(&line[pos..tok.start]);
                if let Some(full) = self.pasted_texts.get(&tok.id) {
                    out.push_str(full);
                } else if self.pasted_images.contains_key(&tok.id) {
                    image_ids_in_order.push(tok.id);
                    // strip the `[image #N]` label from the text
                } else {
                    // Orphaned label with no backing data — keep verbatim.
                    out.push_str(&line[tok.start..tok.end]);
                }
                pos = tok.end;
            }
            out.push_str(&line[pos..]);
        }

        let images: Vec<PastedImage> = image_ids_in_order
            .iter()
            .filter_map(|id| self.pasted_images.get(id).cloned())
            .collect();

        let cleaned = out.trim().to_string();
        let labeled = if images.is_empty() {
            cleaned
        } else {
            let tags: Vec<String> = (1..=images.len())
                .map(|i| format!("[image #{i}]"))
                .collect();
            let suffix = tags.join(" ");
            if cleaned.is_empty() {
                suffix
            } else {
                format!("{cleaned}\n{suffix}")
            }
        };

        self.pasted_images.clear();
        self.pasted_texts.clear();
        (labeled, images)
    }

    /// Handle a bracketed-paste event: prefer a clipboard image, otherwise
    /// insert the pasted text — collapsing long pastes (>10 lines) into an
    /// atomic placeholder backed by `pasted_texts`.
    pub(super) fn handle_paste(&mut self, text: &str) {
        if let Some(query) = self.transcript_search.as_mut() {
            query.push_str(text);
            self.search_match = None;
            self.search_pending = true;
            return;
        }
        if let Some(img) = image_from_path(text).or_else(grab_clipboard_image) {
            let id = self.textarea.insert_image_token();
            self.pasted_images.insert(id, img);
        } else {
            let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
            let line_count = normalized.lines().count();
            if line_count > 10 {
                let id = self.textarea.insert_pasted_text_token(line_count);
                self.pasted_texts.insert(id, normalized);
            } else {
                self.textarea.insert_str(&normalized);
                self.update_autocomplete();
            }
        }
        self.prune_removed_tokens();
    }
}

// ---------------------------------------------------------------------------
// History persistence helpers

/// Maximum number of history entries kept in memory and on disk.
const MAX_HISTORY: usize = 1000;

/// Load history entries from a file.
///
/// Each line in the file represents one entry.  Literal newlines within an entry
/// are stored as the two-character sequence `\n`.  Returns at most the
/// [`MAX_HISTORY`] most recent entries.  Returns an empty `Vec` if the file does
/// not exist or cannot be read.
pub(super) fn load_history(path: &Path) -> Vec<String> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let entries: Vec<String> = contents
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.replace("\\n", "\n"))
        .collect();
    if entries.len() > MAX_HISTORY {
        entries[entries.len() - MAX_HISTORY..].to_vec()
    } else {
        entries
    }
}

/// Append a single history entry to the file.
///
/// Literal newlines in `entry` are escaped as `\\n` so each entry occupies
/// exactly one file line.  The file (and parent directories) are created if they
/// do not exist.  Errors are silently ignored — history persistence is
/// best-effort.
///
/// To keep the file from growing without bound, once it exceeds
/// `2 * MAX_HISTORY` lines it is compacted in place to the most recent
/// [`MAX_HISTORY`] entries.  Compaction is amortized — it runs roughly once
/// every `MAX_HISTORY` appends rather than on every call.
fn append_history(path: &Path, entry: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let escaped = entry.replace('\n', "\\n");
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = writeln!(file, "{escaped}");
    drop(file);

    compact_history(path);
}

/// Rewrite the history file to the most recent [`MAX_HISTORY`] entries when it
/// has grown past `2 * MAX_HISTORY` lines.  Best-effort: any IO error leaves the
/// existing file untouched.
fn compact_history(path: &Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
    if lines.len() <= 2 * MAX_HISTORY {
        return;
    }
    let kept = &lines[lines.len() - MAX_HISTORY..];
    // Write to a sibling temp file then rename, so a crash mid-write can't
    // truncate the live history.
    let tmp = path.with_extension("tmp");
    let mut body = kept.join("\n");
    body.push('\n');
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

// ---------------------------------------------------------------------------
// Clipboard helpers

/// Copy text to the system clipboard. Returns an error message on failure.
///
/// Falls back to OSC 52 for remote terminals without a local clipboard.
pub(crate) fn copy_to_clipboard(text: &str) -> Result<(), String> {
    #[cfg(not(target_os = "android"))]
    if arboard::Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text))
        .is_ok()
    {
        return Ok(());
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let mut stdout = io::stdout();
    write!(stdout, "\x1b]52;c;{encoded}\x07")
        .and_then(|_| stdout.flush())
        .map_err(|error| error.to_string())
}

/// Read an image file when the pasted text is just its path.
///
/// Terminals paste a dragged file as its path, and some multiplexers save a
/// pasted image to a file and paste that path. Accepts surrounding quotes, a
/// `file://` prefix and backslash-escaped spaces.
fn image_from_path(text: &str) -> Option<PastedImage> {
    let text = text.trim();
    if text.contains('\n') {
        return None;
    }
    let text = text
        .strip_prefix('\'')
        .and_then(|t| t.strip_suffix('\''))
        .or_else(|| text.strip_prefix('"').and_then(|t| t.strip_suffix('"')))
        .unwrap_or(text);
    let text = text
        .strip_prefix("file://")
        .unwrap_or(text)
        .replace("\\ ", " ");
    let path = Path::new(&text);
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let media_type = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    };
    let bytes = std::fs::read(path).ok()?;
    Some(PastedImage {
        media_type: media_type.to_owned(),
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

/// Attempt to grab image data from the system clipboard.
///
/// On macOS, uses `osascript` to check for PNG data, then `pbpaste` isn't
/// suitable for binary data so we write an AppleScript that outputs base64.
/// Returns `None` if no image is present or on non-macOS platforms.
fn grab_clipboard_image() -> Option<PastedImage> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let tmp = std::env::temp_dir().join("flashmind_clip.png");
        let tmp_path = tmp.to_string_lossy().to_string();

        // Write clipboard PNG to a temp file, then base64-encode it.
        // Accessing the clipboard and writing the file both happen in the
        // outer osascript process, avoiding nested-subprocess permission
        // issues and fragile Unicode piping through sed/xxd.
        let script = format!(
            r#"try
    set imgData to the clipboard as «class PNGf»
    set fref to open for access POSIX file "{tmp_path}" with write permission
    set eof of fref to 0
    write imgData to fref
    close access fref
    set b64 to do shell script "base64 -i " & quoted form of "{tmp_path}" & " && rm -f " & quoted form of "{tmp_path}"
    return b64
on error
    return ""
end try"#
        );

        let output = Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .output()
            .ok()?;

        // Clean up in case the script errored after creating the file.
        let _ = std::fs::remove_file(&tmp);

        if !output.status.success() {
            return None;
        }

        let b64 = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if b64.is_empty() {
            return None;
        }

        Some(PastedImage {
            media_type: "image/png".to_string(),
            data: b64,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_HISTORY, append_history, image_from_path, load_history};
    use std::path::PathBuf;

    /// A unique temp file path that is removed on drop.
    struct TempHistory(PathBuf);

    impl TempHistory {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("flashmind-hist-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempHistory {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let _ = std::fs::remove_file(self.0.with_extension("tmp"));
        }
    }

    #[test]
    fn append_then_load_roundtrips_with_escaped_newlines() {
        let h = TempHistory::new("roundtrip");
        append_history(h.path(), "first");
        append_history(h.path(), "multi\nline");
        let loaded = load_history(h.path());
        assert_eq!(loaded, vec!["first".to_string(), "multi\nline".to_string()]);
    }

    #[test]
    fn append_compacts_when_file_exceeds_threshold() {
        let h = TempHistory::new("compact");
        // Append more than 2 * MAX_HISTORY entries; the file should be
        // compacted down to the most recent MAX_HISTORY.
        let total = 2 * MAX_HISTORY + 5;
        for i in 0..total {
            append_history(h.path(), &format!("entry {i}"));
        }
        let contents = std::fs::read_to_string(h.path()).unwrap();
        let line_count = contents.lines().filter(|l| !l.is_empty()).count();
        assert!(
            line_count <= 2 * MAX_HISTORY,
            "file should be compacted, has {line_count} lines"
        );
        let loaded = load_history(h.path());
        assert_eq!(loaded.len(), MAX_HISTORY);
        // The most-recent entry survives.
        assert_eq!(loaded.last().unwrap(), &format!("entry {}", total - 1));
    }

    #[test]
    fn pasting_an_image_path_attaches_the_image() {
        let dir = std::env::temp_dir().join(format!("flashmind-paste-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("example shot.png");
        std::fs::write(&path, b"png bytes").unwrap();
        let escaped = path.display().to_string().replace(' ', "\\ ");
        for pasted in [
            path.display().to_string(),
            format!("'{}'", path.display()),
            format!("file://{}", path.display()),
            escaped,
        ] {
            let image = image_from_path(&pasted).expect("an image path attaches the image");
            assert_eq!(image.media_type, "image/png");
            assert_eq!(image.data, "cG5nIGJ5dGVz");
        }
        assert!(image_from_path("not a path").is_none());
        assert!(image_from_path(&dir.join("missing.png").display().to_string()).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
