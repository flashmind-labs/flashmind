//! Autocomplete for slash commands, their arguments and `@mention`s.

use super::{Dropdown, Repl};

impl<'a> Repl<'a> {
    /// Replace the token the dropdown is completing with its selected value.
    /// Returns `false` when no dropdown selection is available.
    pub(super) fn accept_dropdown(&mut self) -> bool {
        let Some(value) = self
            .dropdown
            .as_ref()
            .filter(|d| !d.is_empty())
            .and_then(|d| d.selected_value())
            .map(str::to_owned)
        else {
            return false;
        };
        let text = self.textarea.text();
        if let Some(state) = self.mention.take() {
            let cursor = self.textarea.cursor_byte_offset();
            let token_end = cursor.min(text.len());
            let trigger = text.as_bytes().get(state.token_start).copied();
            let replacement = if trigger == Some(b'#') {
                // #search result: "file:line: content" -> @file:line
                let file_line = value.split(": ").next().unwrap_or(&value);
                format!("@{file_line} ")
            } else {
                format!("@{value} ")
            };
            let new_text = format!(
                "{}{}{}",
                &text[..state.token_start],
                replacement,
                &text[token_end..]
            );
            self.textarea.set_text(&new_text);
            let new_cursor = state.token_start + replacement.len();
            self.textarea.set_cursor_byte_offset(new_cursor);
        } else if let Some(start) = self.argument_start.take() {
            let start = start.min(text.len());
            self.textarea
                .set_text(&format!("{}{value}", &text[..start]));
        } else {
            // Slash command: replace the whole input.
            self.textarea.set_text(&format!("{value} "));
        }
        self.dropdown = None;
        true
    }

    /// Update the autocomplete dropdown based on current textarea content.
    ///
    /// Shows matching slash commands when the input starts with `/` and is a
    /// single line.  Hides the dropdown otherwise.
    pub(super) fn update_autocomplete(&mut self) {
        self.prune_removed_tokens();
        let text = self.textarea.text();
        let cursor = self.textarea.cursor_byte_offset();

        // --- @mention file completion (takes priority), then #content search ---
        if let Some(provider) = self.config.mention_provider.clone()
            && let Some(state) =
                detect_mention(&text, cursor).or_else(|| detect_content_search(&text, cursor))
        {
            let query = &text[state.token_start + 1..state.cursor.min(text.len())];
            let candidates = match text.as_bytes()[state.token_start] {
                b'@' => provider.complete(query),
                _ if query.len() >= 3 => provider.search_content(query),
                _ => Vec::new(),
            };
            self.mention = (!candidates.is_empty()).then_some(state);
            self.show_candidates(candidates);
            return;
        }

        // Not in a mention or search token: clear any lingering state.
        self.mention = None;
        self.argument_start = None;

        // --- slash-command argument completion ---
        if let Some(provider) = self.config.argument_provider.clone()
            && !text.contains('\n')
            && let Some(rest) = text.strip_prefix('/')
            && let Some((name, query)) = rest.split_once(' ')
        {
            let command = format!("/{name}");
            let candidates = provider.complete(&command, query);
            self.argument_start = (!candidates.is_empty()).then_some(command.len() + 1);
            self.show_candidates(candidates);
            self.update_command_hint();
            return;
        }

        // --- slash-command completion ---
        if !text.starts_with('/')
            || text.contains('\n')
            || self.config.available_commands.is_empty()
        {
            self.dropdown = None;
            self.update_command_hint();
            return;
        }
        let prefix = text.trim_end();
        let candidates: Vec<String> = self
            .config
            .available_commands
            .iter()
            .filter(|cmd| cmd.name.starts_with(prefix) && cmd.name != prefix)
            .map(|cmd| cmd.name.clone())
            .collect();
        self.show_candidates(candidates);
        self.update_command_hint();
    }

    /// Show `candidates` in the dropdown, or hide it when there are none.
    fn show_candidates(&mut self, candidates: Vec<String>) {
        if candidates.is_empty() {
            self.dropdown = None;
            return;
        }
        match &mut self.dropdown {
            Some(dd) => dd.set_candidates(candidates),
            None => self.dropdown = Some(Dropdown::new("", candidates)),
        }
    }

    /// Refresh the dim ghost-text suffix and input block title based on the
    /// current input. When the first line starts with `/`, matches it against
    /// [`available_commands`](super::ReplConfig::available_commands) and shows the
    /// remaining name + args hint as a ghost suffix, with the command
    /// description in the block title. Clears the suffix otherwise.
    fn update_command_hint(&mut self) {
        let first = self.textarea.lines().first().cloned().unwrap_or_default();
        let multi_line = self.textarea.lines().len() > 1;

        let Some(rest) = first.strip_prefix('/') else {
            self.textarea.set_ghost_suffix("");
            return;
        };

        let (name, after_name) = match rest.split_once(' ') {
            Some((n, a)) => (n, Some(a)),
            None => (rest, None),
        };

        let matched = self
            .config
            .available_commands
            .iter()
            .find(|c| c.name == format!("/{name}"))
            .or_else(|| {
                if name.is_empty() {
                    None
                } else {
                    let prefix = format!("/{name}");
                    self.config
                        .available_commands
                        .iter()
                        .find(|c| c.name.starts_with(&prefix))
                }
            });

        let Some(entry) = matched else {
            self.textarea.set_ghost_suffix("");
            return;
        };

        let full_name = &entry.name;
        let args_hint = &entry.args_hint;

        // Ghost text: remaining name chars + args hint. Suppress on multi-line.
        let ghost = if multi_line {
            String::new()
        } else if after_name.is_none() {
            let remaining = &full_name[name.len() + 1..]; // +1 for '/'
            match (remaining.is_empty(), args_hint.is_empty()) {
                (true, true) => String::new(),
                (true, false) => format!(" {args_hint}"),
                (false, true) => remaining.to_string(),
                (false, false) => format!("{remaining} {args_hint}"),
            }
        } else if after_name == Some("") && !args_hint.is_empty() {
            args_hint.to_string()
        } else {
            String::new()
        };
        self.textarea.set_ghost_suffix(&ghost);
    }
}

/// `@mention` autocompletion state.
#[derive(Clone, Debug)]
pub(super) struct MentionState {
    /// Byte offset in the full text where the `@` token begins.
    token_start: usize,
    /// Byte offset of the cursor when the mention was last evaluated.
    cursor: usize,
}

// ---------------------------------------------------------------------------
// @mention token detection (free function)

/// Detect an `@mention` token ending at `cursor` in `text`.
///
/// Returns the byte offset of the `@` if there is one at the start of the text
/// or preceded by ASCII whitespace, with no whitespace between it and the
/// cursor.  This keeps the mention "live" while the user types a path but
/// dismisses it once they space away.
fn detect_mention(text: &str, cursor: usize) -> Option<MentionState> {
    detect_token(text, cursor, '@')
}

fn detect_content_search(text: &str, cursor: usize) -> Option<MentionState> {
    detect_token(text, cursor, '#')
}

fn detect_token(text: &str, cursor: usize, trigger: char) -> Option<MentionState> {
    if cursor == 0 || cursor > text.len() {
        return None;
    }
    let before = &text[..cursor];
    let pos = before.rfind(trigger)?;
    if pos > 0 && !before.as_bytes()[pos - 1].is_ascii_whitespace() {
        return None;
    }
    if text[pos + 1..cursor].contains(char::is_whitespace) {
        return None;
    }
    Some(MentionState {
        token_start: pos,
        cursor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::repl::{ArgumentProvider, CommandInfo, KeyAction, ReplConfig};
    use ratatui::crossterm::event::{self, KeyCode, KeyModifiers};
    use std::sync::Arc;

    #[derive(Debug)]
    struct Models;

    impl ArgumentProvider for Models {
        fn complete(&self, command: &str, query: &str) -> Vec<String> {
            [
                "openrouter:z-ai/glm-4.6",
                "openrouter:z-ai/glm-4.5",
                "ollama:qwen3",
            ]
            .into_iter()
            .filter(|m| command == "/model" && m.contains(query))
            .map(String::from)
            .collect()
        }
    }

    fn repl() -> Repl<'static> {
        Repl::new(ReplConfig {
            available_commands: vec![CommandInfo::new("/model", "Switch model")],
            argument_provider: Some(Arc::new(Models)),
            ..Default::default()
        })
    }

    fn press(repl: &mut Repl<'_>, code: KeyCode) -> KeyAction {
        repl.process_key(event::KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn type_text(repl: &mut Repl<'_>, text: &str) {
        for character in text.chars() {
            press(repl, KeyCode::Char(character));
        }
    }

    #[test]
    fn typing_an_argument_lists_matching_models() {
        let mut repl = repl();
        type_text(&mut repl, "/model glm");
        let dropdown = repl.dropdown.as_ref().expect("glm matches two models");
        assert_eq!(
            dropdown.candidates,
            ["openrouter:z-ai/glm-4.6", "openrouter:z-ai/glm-4.5"]
        );
    }

    #[test]
    fn enter_on_a_selected_model_submits_the_command() {
        let mut repl = repl();
        type_text(&mut repl, "/model glm");
        press(&mut repl, KeyCode::Down);
        match press(&mut repl, KeyCode::Enter) {
            KeyAction::Submit { text, .. } => assert_eq!(text, "/model openrouter:z-ai/glm-4.5"),
            _ => panic!("enter on an argument candidate submits"),
        }
    }

    #[test]
    fn tab_fills_the_model_without_submitting() {
        let mut repl = repl();
        type_text(&mut repl, "/model qwen");
        press(&mut repl, KeyCode::Tab);
        assert_eq!(repl.textarea.text(), "/model ollama:qwen3");
        assert!(repl.dropdown.is_none());
    }

    #[test]
    fn unknown_commands_get_no_argument_candidates() {
        let mut repl = repl();
        type_text(&mut repl, "/other glm");
        assert!(repl.dropdown.is_none());
    }
}
