//! Drawing the transcript, input area, activity line and footer.

use std::io;

use crate::transcript::{layout_line, text_width};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Padding, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use unicode_width::UnicodeWidthStr;

use super::Repl;
use super::transcript_view::{TextPoint, find_transcript_match};
use crate::event_render::{REPLY_INDENT, format_secs, reply_lines, truncate_display};
use crate::styles;

impl<'a> Repl<'a> {
    fn input_title_spans(&mut self) -> Vec<Span<'static>> {
        if let Some(query) = &self.transcript_search {
            return vec![Span::styled(
                format!(" Find: {query} | Esc: close "),
                styles::S_ACCENT,
            )];
        }
        if let Some((query, _)) = &self.reverse_search {
            return vec![
                Span::styled(" reverse-i-search: ", styles::S_DIM),
                Span::styled(format!("{query} "), styles::S_USER),
            ];
        }
        let mut spans = Vec::new();
        if let Some(title) = &self.conversation_title {
            spans.push(Span::styled(
                format!(" Esc: back | {title} "),
                styles::S_AGENT,
            ));
        }
        if self.shell_mode {
            spans.push(Span::styled(" shell mode, esc to exit ", styles::S_SHELL));
        }
        if !self.pasted_images.is_empty() {
            let attached: Vec<usize> = self
                .textarea
                .atomic_tokens()
                .iter()
                .filter(|t| self.pasted_images.contains_key(&t.id))
                .map(|t| t.id)
                .collect();
            if !attached.is_empty() {
                let label = attached
                    .iter()
                    .map(|i| format!("#{i}"))
                    .collect::<Vec<_>>()
                    .join(",");
                spans.push(Span::styled(
                    format!("[{label}] "),
                    ratatui::style::Style::default().fg(ratatui::style::Color::Magenta),
                ));
            }
        }
        if let Some((toast, _)) = &self.toast {
            spans.push(Span::styled(
                format!("{toast} "),
                ratatui::style::Style::default().fg(ratatui::style::Color::Green),
            ));
        }

        spans
    }

    fn input_block(&mut self) -> Block<'static> {
        let title = Line::from(self.input_title_spans());
        Block::default()
            .borders(Borders::TOP | Borders::BOTTOM)
            .border_style(if self.transcript_search.is_some() {
                styles::S_ACCENT
            } else if self.shell_mode {
                styles::S_SHELL
            } else {
                styles::S_DIMMER
            })
            .title(title)
            .padding(self.input_padding())
    }

    /// Room for the prompt glyph on the left of the input text.
    fn input_padding(&self) -> Padding {
        Padding::new(self.config.prompt.width() as u16 + 1, 1, 0, 0)
    }

    /// The line above the input while a turn runs: spinner, label, elapsed.
    fn activity_line(&self) -> Option<Line<'static>> {
        let label = self.activity.as_deref()?;
        let label = if label.is_empty() { "Working" } else { label };
        let mut spans = vec![
            Span::styled(format!("{} ", self.spinner.current()), styles::S_AGENT),
            Span::styled(format!("{label}\u{2026}"), styles::S_ACCENT),
        ];
        if self.foreground_command.is_some() {
            spans.push(Span::styled(" | Ctrl+B: background", styles::S_DIM));
        }
        let mut details: Vec<String> = self
            .renderer
            .turn_elapsed()
            .map(|elapsed| format_secs(elapsed.as_secs()))
            .into_iter()
            .collect();
        details.extend(self.renderer.turn_usage_parts());
        if !details.is_empty() {
            spans.push(Span::styled(
                format!(" ({})", details.join(" \u{b7} ")),
                styles::S_DIM,
            ));
        }
        Some(Line::from(spans))
    }

    /// Columns reserved for the pet beside the textarea (0 when hidden).
    /// Narrow terminals, such as a phone, hide the pet to keep room for text.
    fn pet_cols(&self, width: u16) -> u16 {
        if self.pet.is_some() && width >= MIN_PET_WIDTH {
            14
        } else {
            0
        }
    }

    fn input_height(&self, width: u16) -> u16 {
        // The textarea shares the terminal width with an optional pet column.
        // Compute the inner width the way `draw_input` does so wrapping agrees
        // with what is actually rendered (mismatch undercounts wrapped rows
        // when a pet is enabled, clipping the bottom of the input).
        let pet_cols = self.pet_cols(width);
        let pet_enabled = pet_cols > 0;
        let ta_width = width.saturating_sub(pet_cols).max(1);
        let padding = self.input_padding();
        let inner_w = ta_width.saturating_sub(padding.left + padding.right).max(1);
        // +2 for the rules above and below
        let content = self.textarea.visual_line_count(inner_w) as u16 + 2;
        // The pet companion is 3 rows tall; when enabled, reserve at least
        // 3 rows + the rules so it isn't clipped vertically.
        let min = if pet_enabled { 5 } else { 3 };
        let max = self.config.max_input_height.saturating_add(2).max(1);
        content.clamp(min.min(max), max)
    }

    pub(super) fn echo_input(&mut self, text: &str) -> io::Result<()> {
        let actions = self.renderer.finish_reply();
        self.apply_actions(&actions)?;
        let lines = match text.strip_prefix('!') {
            Some(command) => {
                user_message_lines(command, &shell_prompt(&self.config.prompt), styles::S_SHELL)
            }
            None => user_message_lines(text, &self.config.prompt, styles::S_ACCENT),
        };
        self.commit_lines(&lines)
    }

    fn dynamic_lines(&self, width: u16) -> Vec<Line<'static>> {
        let mut lines = self.live_output.clone();
        let reasoning = self.renderer.pending_reasoning();
        if !reasoning.trim().is_empty() {
            let expanded = self.renderer.expand_reasoning();
            let (marker, action) = if expanded {
                ("v", "collapse")
            } else {
                (">", "expand")
            };
            lines.push(Line::from(Span::styled(
                format!("{marker} thinking (live, ctrl+t to {action})"),
                styles::S_DIM,
            )));
            let preview = if expanded {
                #[cfg(feature = "markdown")]
                {
                    crate::markdown_render::render_preview_to_lines(reasoning, width)
                }
                #[cfg(not(feature = "markdown"))]
                {
                    reasoning
                        .lines()
                        .map(|line| Line::from(line.to_owned()))
                        .collect::<Vec<_>>()
                }
            } else {
                let mut last = reasoning
                    .lines()
                    .rev()
                    .filter(|line| !line.trim().is_empty())
                    .take(10)
                    .map(|line| Line::from(line.to_owned()))
                    .collect::<Vec<_>>();
                last.reverse();
                last
            };
            let mut preview: Vec<_> = preview
                .iter()
                .flat_map(|line| layout_line(line, width))
                .collect();
            if !expanded {
                preview.retain(|line| !line.to_string().trim().is_empty());
                preview.drain(..preview.len().saturating_sub(10));
            }
            lines.extend(preview.into_iter().map(|mut line| {
                for span in &mut line.spans {
                    span.style = span.style.patch(styles::S_DIM);
                }
                line
            }));
        }
        let partial = self.renderer.pending_text();
        if !partial.is_empty() {
            let inner = text_width(width).saturating_sub(REPLY_INDENT as u16).max(1);
            #[cfg(feature = "markdown")]
            let preview = crate::markdown_render::render_preview_to_lines(partial, inner);
            #[cfg(not(feature = "markdown"))]
            let preview: Vec<Line<'static>> = {
                let _ = inner;
                partial
                    .lines()
                    .map(|line| Line::from(line.to_owned()))
                    .collect()
            };
            lines.extend(reply_lines(preview, self.renderer.reply_open()));
        }
        for (text, images) in &self.pending_inputs {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_owned(),
                    styles::S_USER_ECHO.add_modifier(ratatui::style::Modifier::DIM),
                )));
            }
            if !images.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("[{} images]", images.len()),
                    styles::S_DIM,
                )));
            }
            lines.push(Line::default());
        }
        lines.extend(self.subagent_progress.lines_for_model(
            width,
            self.status.as_ref().map(|status| status.model.as_str()),
        ));
        lines
    }

    pub(super) fn live_reasoning_row(&self) -> Option<usize> {
        (!self.renderer.pending_reasoning().trim().is_empty()).then(|| {
            self.transcript_layout.rows()
                + self
                    .live_output
                    .iter()
                    .map(|line| layout_line(line, self.transcript_area.width.max(1)).len())
                    .sum::<usize>()
        })
    }

    pub(super) fn draw_input(&mut self) -> io::Result<()> {
        if self.subagent_view.is_some() {
            return self.draw_subagent_view();
        }
        let tui = self.tui.clone();
        tui.draw(|frame, history| self.render_input_frame(frame, history))
    }

    pub(super) fn render_input_frame(&mut self, frame: &mut Frame<'_>, history: &[Line<'static>]) {
        let area = chat_area(frame.area());
        if area.is_empty() {
            return;
        }
        self.transcript_area = Rect::default();
        self.renderer
            .set_width(area.width.saturating_sub(1) as usize);
        self.textarea.set_placeholder_text(&self.config.placeholder);
        let pet_cols = self.pet_cols(area.width);
        let block = self.input_block();
        self.textarea.set_block(block);
        let dropdown_lines = self
            .dropdown
            .as_ref()
            .map(|dropdown| {
                dropdown.lines_for_height(area.width, (area.height.saturating_sub(3) / 2) as usize)
            })
            .unwrap_or_default();
        let dropdown_height = dropdown_lines.len() as u16;
        let input_height = self
            .input_height(area.width)
            .min(area.height.saturating_sub(dropdown_height))
            .max(1);
        let activity = self.activity_line();
        let activity_height = if activity.is_some() && area.height > input_height + 6 {
            2
        } else {
            0
        };
        let extra_height = area
            .height
            .saturating_sub(input_height + dropdown_height + activity_height + 1);
        if extra_height > 0 {
            let body = Rect::new(area.x, area.y, area.width.saturating_sub(1), extra_height);
            let width = body.width.max(1);
            let anchor = (!self.transcript_scroll.following
                && self.transcript_layout.width() != width)
                .then(|| self.transcript_layout.window(self.transcript_scroll.top, 1));
            let reflowed = anchor.is_some();
            self.transcript_layout.measure(history, width);
            let dynamic: Vec<Line<'static>> = self
                .dynamic_lines(width)
                .iter()
                .flat_map(|line| layout_line(line, width))
                .collect();
            let dynamic_rows = if dynamic.is_empty() {
                0
            } else {
                Paragraph::new(dynamic.clone())
                    .wrap(Wrap { trim: false })
                    .line_count(width)
            };
            self.transcript_scroll
                .update(self.transcript_layout.rows() + dynamic_rows, body.height);
            if let Some(anchor) = anchor
                && anchor.entries.start < history.len()
            {
                self.transcript_scroll.jump_to(
                    self.transcript_layout.entry_row(anchor.entries.start)
                        + anchor.row_offset as usize,
                );
            }
            if let Some(top) = self.pending_scroll_top.take() {
                self.transcript_scroll.jump_to(top);
            }
            if self.search_pending {
                self.search_pending = false;
                if let Some(query) = self
                    .transcript_search
                    .as_deref()
                    .filter(|query| !query.is_empty())
                {
                    if let Some(index) = find_transcript_match(
                        history,
                        query,
                        self.search_match,
                        self.search_backwards,
                    ) {
                        self.search_match = Some(index);
                        self.transcript_scroll
                            .jump_to(self.transcript_layout.entry_row(index));
                    } else {
                        self.toast = Some(("No matches".into(), self.tick));
                    }
                }
            }
            if reflowed {
                // Rows reflow at a new width, so the selected text moved.
                self.selection = None;
                self.copy_pending = false;
            }
            let highlight = self
                .search_match
                .filter(|_| self.transcript_search.is_some());
            let top = self.transcript_scroll.top;
            frame.render_widget(
                self.transcript_paragraph(history, &dynamic, top, body, highlight),
                body,
            );
            self.transcript_area = body;
            if let Some(selection) = &self.selection {
                for y in body.y..body.bottom() {
                    let row = top + (y - body.y) as usize;
                    for x in body.x..body.right() {
                        if selection.contains(TextPoint { row, column: x }) {
                            frame.buffer_mut()[(x, y)].set_style(styles::S_SELECTION);
                        }
                    }
                }
            }
            if std::mem::take(&mut self.copy_pending) {
                self.copy_selected_text(history, &dynamic, body);
            }
            // Ratatui adds the viewport length to the number of scroll positions.
            let mut scrollbar = ScrollbarState::new(self.transcript_scroll.max + 1)
                .position(self.transcript_scroll.top)
                .viewport_content_length(body.height as usize);
            if self.transcript_scroll.max > 0 {
                frame.render_stateful_widget(
                    Scrollbar::new(ScrollbarOrientation::VerticalRight)
                        .begin_symbol(None)
                        .end_symbol(None),
                    Rect::new(area.x, area.y, area.width, extra_height),
                    &mut scrollbar,
                );
            }
        }
        if let Some(activity) = activity.filter(|_| activity_height > 0) {
            frame.render_widget(
                activity,
                Rect::new(area.x, area.y + extra_height + 1, area.width, 1),
            );
        }
        let input_area = Rect::new(
            area.x,
            area.y + extra_height + activity_height,
            area.width.saturating_sub(pet_cols),
            input_height,
        );
        let padding = self.input_padding();
        self.textarea.ensure_cursor_visible(
            input_area
                .width
                .saturating_sub(padding.left + padding.right)
                .max(1),
            input_area.height.saturating_sub(2).max(1),
        );
        frame.render_widget(&self.textarea, input_area);
        if input_area.height > 2 && input_area.width > padding.left {
            let prompt = if self.shell_mode {
                Span::styled(shell_prompt(&self.config.prompt), styles::S_SHELL)
            } else {
                Span::styled(self.config.prompt.clone(), styles::S_ACCENT)
            };
            frame.render_widget(
                prompt,
                Rect::new(input_area.x, input_area.y + 1, padding.left, 1),
            );
        }
        if let Some(pet) = &self.pet
            && pet_cols > 0
        {
            let pet_area = Rect::new(input_area.right(), input_area.y, pet_cols, input_height);
            pet.render(frame.buffer_mut(), pet_area);
        }
        if dropdown_height > 0 {
            frame.render_widget(
                Paragraph::new(dropdown_lines),
                Rect::new(area.x, input_area.bottom(), area.width, dropdown_height),
            );
        }
        if area.height > 2 {
            let hint = if let Some(query) = &self.transcript_search {
                format!("Find: {query} | Enter: next | Shift+Enter: previous | Esc: close")
            } else if self.transcript_scroll.following {
                "/help \u{b7} PgUp scroll \u{b7} Ctrl+F find \u{b7} Ctrl+O tools \u{b7} Ctrl+T thinking".into()
            } else {
                "Reading history | Ctrl+End: follow latest | Ctrl+F: find | Ctrl+T thinking".into()
            };
            let width = area.width as usize;
            let mode = format!("{} (Shift+Tab)", self.mode());
            let mut footer = if self.transcript_search.is_some() {
                Vec::new()
            } else {
                vec![Span::styled(
                    truncate_display(&mode, width),
                    styles::S_ACCENT,
                )]
            };
            let mode_width = Line::from(footer.clone()).width();
            if self.transcript_search.is_none() {
                footer.extend(
                    self.status
                        .as_ref()
                        .map(|status| status.to_spans(width.saturating_sub(mode_width)))
                        .unwrap_or_default(),
                );
            }
            let used = Line::from(footer.clone()).width();
            if footer.is_empty() {
                footer.push(Span::styled(truncate_display(&hint, width), styles::S_DIM));
            } else if used + hint.width() + 2 <= width {
                footer.push(Span::raw(" ".repeat(width - used - hint.width())));
                footer.push(Span::styled(hint, styles::S_DIM));
            }
            frame.render_widget(
                Line::from(footer),
                Rect::new(area.x, area.bottom() - 1, area.width, 1),
            );
        }
        if let Some(cursor) = self.textarea.cursor_screen_pos(input_area) {
            frame.set_cursor_position(cursor);
        }
        if let Some(text) = self.pending_screen_copy.take() {
            let toast = match super::input::copy_to_clipboard(&text) {
                Ok(()) => "Copied selection".to_string(),
                Err(error) => format!("Copy failed: {error}"),
            };
            self.toast = Some((toast, self.tick));
        }
        self.screen_selection.render(frame.buffer_mut());
    }
}

pub(super) fn chat_area(area: Rect) -> Rect {
    let margin = if area.width >= 60 { 2 } else { 0 };
    Rect::new(
        area.x + margin,
        area.y,
        area.width.saturating_sub(margin * 2),
        area.height,
    )
}

/// `!` padded to the width of `prompt`, so shell mode keeps the input aligned.
fn shell_prompt(prompt: &str) -> String {
    format!("{:<width$}", "!", width = prompt.width().max(1))
}

/// A user message as a shaded bar, with the prompt hanging to its left.
fn user_message_lines(text: &str, prompt: &str, prompt_style: Style) -> Vec<Line<'static>> {
    let mut lines = vec![Line::default()];
    for (index, text) in text.split('\n').enumerate() {
        let prefix = if index == 0 {
            format!("{prompt} ")
        } else {
            " ".repeat(prompt.width() + 1)
        };
        lines.push(
            Line::from(vec![
                Span::styled(prefix, prompt_style),
                Span::raw(text.to_owned()),
            ])
            .style(styles::S_USER_ECHO),
        );
    }
    lines
}

/// Narrowest terminal that shows the pet. A phone in portrait is about 100
/// columns wide.
const MIN_PET_WIDTH: u16 = 120;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_render::{EventRenderer, RenderAction, apply_to_lines};
    use crate::widgets::pet::Pet;
    use crate::widgets::repl::{ReplConfig, StatusInfo};
    use flashmind_types::AgentEvent;
    use ratatui::backend::TestBackend;
    use ratatui::{Terminal, TerminalOptions, Viewport};
    use rust_decimal::Decimal;
    use std::io::Write;

    #[test]
    fn transcript_search_is_visible_and_preserves_the_draft() {
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        for width in [40, 80] {
            let mut repl = Repl::new(ReplConfig::default());
            repl.textarea.insert_str("draft input");
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            repl.process_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let screen = terminal.backend().buffer();
            let search_row = (0..screen.area.height)
                .find(|&y| {
                    (0..width)
                        .map(|x| screen[(x, y)].symbol())
                        .collect::<String>()
                        .contains("Find:  | Esc: close")
                })
                .expect("Search mode has a visible input title.");
            let footer: String = (0..width)
                .map(|x| screen[(x, screen.area.height - 1)].symbol())
                .collect();
            assert!(footer.contains("Enter: next"));

            repl.process_key(KeyEvent::from(KeyCode::Char('x')));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let search_title: String = (0..width)
                .map(|x| terminal.backend().buffer()[(x, search_row)].symbol())
                .collect();
            assert!(search_title.contains("Find: x | Esc: close"));
            assert_eq!(repl.textarea.text(), "draft input");

            repl.process_key(KeyEvent::from(KeyCode::Esc));
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let screen: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(!screen.contains("Find:"));
            assert_eq!(repl.textarea.text(), "draft input");
        }
    }

    #[test]
    fn subagent_shortcut_is_visible_while_a_command_runs() {
        use flashmind_types::tool::BackgroundRequest;

        let mut repl = Repl::new(ReplConfig::default());
        repl.activity = Some("Working".into());
        repl.apply_event(&AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "Inspect the code".into(),
            model: None,
            event: Box::new(AgentEvent::TextDelta(String::new())),
        })
        .unwrap();
        repl.apply_event(&AgentEvent::ToolBackgroundReady {
            id: "command".into(),
            request: Some(BackgroundRequest::default()),
        })
        .unwrap();

        assert!(
            repl.dynamic_lines(80)
                .iter()
                .any(|line| line.to_string().contains("Ctrl+G: open"))
        );
        assert!(
            repl.activity_line()
                .unwrap()
                .to_string()
                .contains("Ctrl+B: background")
        );
    }

    #[test]
    fn activity_has_a_blank_row_above_and_the_input_separator_below() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.activity = Some("Working".into());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &[Line::from("Latest output")]))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let rows: Vec<String> = (0..24)
            .map(|y| (0..80).map(|x| buffer[(x, y)].symbol()).collect())
            .collect();
        let status = rows.iter().position(|row| row.contains("Working")).unwrap();
        assert!(rows[status - 1].trim().is_empty());
        assert!(rows[status + 1].contains('\u{2500}'));
    }

    #[test]
    fn completed_commands_wrap_to_three_rows_with_the_time_after_the_text() {
        let mut repl = Repl::new(ReplConfig::default());
        let label = "Run example checks with a description that needs wrapping in a small terminal";
        repl.renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: label.into(),
        });
        let actions = repl.renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            success: true,
            elapsed_ms: 1_300,
            output: String::new(),
            sources: vec![],
        });
        let history = match &actions[0] {
            RenderAction::FinishTool { output, .. } => output.collapsed.clone(),
            _ => panic!("a completed tool replaces its live row"),
        };
        assert_eq!(history.len(), 1);
        assert!(history[0].to_string().contains(label));
        let mut terminal = Terminal::new(TestBackend::new(200, 24)).unwrap();
        for width in [200, 40, 100, 20, 200] {
            terminal.backend_mut().resize(width, 24);
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            assert!((1..=4).contains(&repl.transcript_layout.rows()));
            if width <= 20 {
                assert_eq!(repl.transcript_layout.rows(), 4);
            }
            let body = repl.transcript_area;
            let buffer = terminal.backend().buffer();
            let first_row: String = (body.x..body.right())
                .map(|x| buffer[(x, body.y)].symbol())
                .collect();
            assert!(first_row.trim_end().ends_with("  1.3 s"), "{first_row:?}");
            let text: String = (body.y..body.bottom())
                .flat_map(|y| (body.x..body.right()).map(move |x| buffer[(x, y)].symbol()))
                .collect();
            if width == 200 {
                assert_eq!(repl.transcript_layout.rows(), 1);
                assert!(text.contains(label));
            }
        }
    }

    #[test]
    fn footer_cost_updates_during_a_turn_without_replacing_other_fields() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_status(StatusInfo {
            working_dir: Some("/srv/project".into()),
            model: "provider/example-model".into(),
            cost: Some(Decimal::new(50, 2)),
            context: Some((25, 100)),
            git_branch: Some("example-branch".into()),
            ..StatusInfo::default()
        });
        let mut terminal = Terminal::new(TestBackend::new(200, 24)).unwrap();
        for (cost, expected) in [
            (Decimal::new(50, 2), "$0.500"),
            (Decimal::new(125, 2), "$1.25"),
            (Decimal::new(125, 2), "$1.25"),
        ] {
            repl.set_cost(cost);
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let footer: String = (0..200).map(|x| buffer[(x, 23)].symbol()).collect();
            for field in [
                "srv/project",
                "provider/example-model",
                "ctx 25%",
                "example-branch",
                expected,
            ] {
                assert!(footer.contains(field), "missing footer field: {field}");
            }
            if cost > Decimal::ONE {
                assert!(!footer.contains("$0.500"));
            }
        }
    }

    #[test]
    fn status_shows_only_the_parent_and_current_directory() {
        for (directory, expected) in [
            ("/srv/team/project", "team/project"),
            ("/workspace", "workspace"),
            ("/", "/"),
            ("parent/current/", "parent/current"),
            ("current", "current"),
            (".", "."),
        ] {
            let status = StatusInfo {
                working_dir: Some(directory.into()),
                ..Default::default()
            };
            assert_eq!(
                Line::from(status.to_spans(200)).to_string(),
                format!("  {expected}")
            );
        }
    }

    #[test]
    fn footer_shows_the_current_directory_after_a_workspace_switch() {
        use std::path::Path;

        let mut repl = Repl::new(ReplConfig::default());
        repl.set_status(StatusInfo {
            model: "provider/example-model".into(),
            ..Default::default()
        });
        for width in [40, 80, 200] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            for (directory, displayed) in [
                ("/srv/team/project", "team/project"),
                ("/path/to/work", "to/work"),
            ] {
                repl.set_working_dir(Path::new(directory));
                terminal
                    .draw(|frame| repl.render_input_frame(frame, &[]))
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let footer: String = (0..width).map(|x| buffer[(x, 23)].symbol()).collect();
                assert!(
                    footer.contains(displayed),
                    "missing directory at width {width}"
                );
                assert!(!footer.contains(directory));
                if directory == "/path/to/work" {
                    assert!(!footer.contains("team/project"));
                }
                if width == 200 {
                    assert!(footer.contains("provider/example-model"));
                }
            }
        }
    }

    #[test]
    fn chat_uses_the_terminal_width_and_footer_fields_do_not_overlap() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_status(StatusInfo {
            working_dir: Some("/srv/project".into()),
            model: "provider/example-model".into(),
            thinking: Some(flashmind_types::ReasoningLevel::Medium),
            cost: Some(rust_decimal::Decimal::new(123, 4)),
            context: Some((25, 100)),
            git_branch: Some("example-branch".into()),
        });
        repl.set_usage(1234, 56);
        let history = vec![Line::from("example text ".repeat(35))];
        for width in [24, 40, 80, 200] {
            let mut terminal = Terminal::new(TestBackend::new(width, 20)).unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &history))
                .unwrap();
            let margin = if width >= 60 { 2 } else { 0 };
            assert_eq!(repl.transcript_area.width, width - margin * 2 - 1);
            let buffer = terminal.backend().buffer();
            let footer: String = (0..width).map(|x| buffer[(x, 19)].symbol()).collect();
            assert!(footer.contains("Act"));
            if width == 200 {
                assert!(footer.contains("example-model"));
                assert!(footer.contains("Ctrl+O tools"));
                for y in 0..20 {
                    for x in width - margin..width {
                        assert_eq!(buffer[(x, y)].symbol(), " ");
                    }
                }
            }
            let status = repl.status.as_ref().unwrap().to_spans(width as usize);
            assert!(Line::from(status).width() <= width as usize);
        }
    }

    #[test]
    fn footer_shows_read_mode_even_on_a_narrow_terminal() {
        use flashmind_types::InteractionMode;
        let mut repl = Repl::new(ReplConfig::default());
        repl.set_mode(InteractionMode::Read);
        repl.set_status(StatusInfo {
            model: "provider/long-example-model".into(),
            ..Default::default()
        });
        for width in [24, 80, 200] {
            let mut terminal = Terminal::new(TestBackend::new(width, 20)).unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let footer: String = (0..width).map(|x| buffer[(x, 19)].symbol()).collect();
            assert!(footer.contains("Read (Shift+Tab)"));
        }
    }

    #[test]
    fn user_messages_are_a_shaded_bar_with_a_hanging_prompt() {
        let lines = user_message_lines("First line\nSecond line", ">", styles::S_ACCENT);
        assert_eq!(
            lines.iter().map(Line::to_string).collect::<Vec<_>>(),
            ["", "> First line", "  Second line"]
        );
        assert!(lines[1..].iter().all(|line| line.style.bg.is_some()));
        let rows = layout_line(&lines[1], 20);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].width(), 20);
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn completed_reply_reflows_when_the_terminal_shrinks_and_grows() {
        let mut repl = Repl::new(ReplConfig::default());
        let text = "A completed response should reflow to the current terminal width. ".repeat(8);
        repl.renderer.set_width(30);
        let actions = repl.renderer.render(&AgentEvent::TextDelta(text.clone()));
        repl.renderer.record_actions(&actions);
        let actions = repl.renderer.flush();
        repl.renderer.record_actions(&actions);
        let history = repl.renderer.lines().to_vec();
        assert_eq!(
            history
                .iter()
                .filter(|line| !line.to_string().is_empty())
                .count(),
            1
        );
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        let narrow_rows = repl.transcript_layout.rows();
        terminal.backend_mut().resize(200, 12);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.transcript_area.width, 195);
        assert!(repl.transcript_layout.rows() < narrow_rows);
        terminal.backend_mut().resize(40, 12);
        terminal
            .draw(|frame| repl.render_input_frame(frame, &history))
            .unwrap();
        assert_eq!(repl.transcript_layout.rows(), narrow_rows);
        let cursor = terminal.get_cursor_position().unwrap();
        assert!(terminal.backend().buffer().area.contains(cursor));
    }

    #[test]
    fn live_preview_keeps_all_lines_of_an_open_paragraph() {
        let mut repl = Repl::new(ReplConfig::default());
        let actions = repl
            .renderer
            .render(&AgentEvent::TextDelta("First line\nsecond line".into()));
        assert!(actions.is_empty());
        let text = repl
            .dynamic_lines(40)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("First line"));
        assert!(text.contains("second line"));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn live_preview_shows_unclosed_code_and_expanded_reasoning() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.renderer.render(&AgentEvent::TextDelta(
            "```rust\nlet first = 1;\nlet second = 2;".into(),
        ));
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("let first = 1;"));
        assert!(text.contains("let second = 2;"));
        repl.renderer
            .render(&AgentEvent::ReasoningDelta("considering options".into()));
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("considering options"));
        repl.set_expand_reasoning(false).unwrap();
        let text = repl
            .dynamic_lines(80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("thinking (live, ctrl+t to expand)"));
        assert!(text.contains("considering options"));
    }

    #[test]
    fn live_thinking_shows_only_the_latest_ten_rows() {
        let mut repl = Repl::new(ReplConfig::default());
        let thoughts = (0..12)
            .map(|index| format!("thought {index}\n\n"))
            .collect::<String>();
        repl.renderer.render(&AgentEvent::ReasoningDelta(thoughts));
        let lines = repl.dynamic_lines(80);
        assert!(
            lines[0]
                .to_string()
                .contains("thinking (live, ctrl+t to expand)")
        );
        assert_eq!(
            lines[1..].iter().map(Line::to_string).collect::<Vec<_>>(),
            (2..12)
                .map(|index| format!("thought {index}"))
                .collect::<Vec<_>>()
        );

        repl.renderer.render(&AgentEvent::ReasoningDelta(format!(
            "zero {}fourteen",
            "one two three four ".repeat(20)
        )));
        let lines = repl.dynamic_lines(20);
        assert_eq!(lines.len(), 11);
        assert!(lines[1..].iter().all(|line| line.width() <= 20));
        assert!(
            !lines[1..]
                .iter()
                .any(|line| line.to_string().contains("zero"))
        );
        assert!(
            lines
                .last()
                .unwrap()
                .to_string()
                .trim_end()
                .ends_with("fourteen")
        );
    }

    #[test]
    fn pet_hides_on_narrow_terminals() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.enable_pet(Pet::Cat);
        assert_eq!(repl.pet_cols(MIN_PET_WIDTH - 1), 0);
        assert!(repl.pet_cols(MIN_PET_WIDTH) > 0);
    }

    #[test]
    fn input_cursor_stays_inside_frame_with_progress_and_wrapped_input() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.textarea.insert_str(&"example input ".repeat(80));
        repl.enable_pet(Pet::Cat);
        repl.live_output = (0..40)
            .map(|index| Line::from(format!("progress {index}")))
            .collect();
        for (width, height) in [(80, 12), (40, 8), (16, 5)] {
            let area = Rect::new(0, 4, width, height);
            let mut terminal = Terminal::with_options(
                TestBackend::new(width, height + 4),
                TerminalOptions {
                    viewport: Viewport::Fixed(area),
                },
            )
            .unwrap();
            terminal
                .draw(|frame| repl.render_input_frame(frame, &[]))
                .unwrap();
            let cursor = terminal.get_cursor_position().unwrap();
            assert!(area.contains(cursor), "cursor {cursor:?} outside {area:?}");
            let screen: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(screen.contains("example"));
        }
    }

    #[test]
    fn a_new_transcript_line_redraws_a_few_rows_not_the_screen() {
        #[derive(Clone, Default)]
        struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let mut repl = Repl::new(ReplConfig::default());
        let mut history = user_message_lines("example question", ">", styles::S_ACCENT);
        history.extend((0..60).map(|index| {
            Line::from(vec![
                Span::raw("  "),
                Span::raw(format!("line {index} {}", "word ".repeat(20))),
            ])
        }));
        let sink = Sink::default();
        let mut presenter = crate::present::Presenter::new(sink.clone()).unwrap();
        let mut written = Vec::new();
        for _ in 0..2 {
            sink.0.lock().unwrap().clear();
            presenter
                .draw((120, 40), |frame| repl.render_input_frame(frame, &history))
                .unwrap();
            written.push(sink.0.lock().unwrap().len());
            history.push(Line::from("    \u{2713} exec  example command"));
        }
        assert!(written[1] * 5 < written[0], "{written:?}");
    }

    #[test]
    fn tool_progress_replaces_live_lines_and_preserves_preceding_output() {
        let mut renderer = EventRenderer::new();
        let mut live = vec![Line::from("preceding answer")];
        let start = AgentEvent::ToolStart {
            name: "exec".into(),
            id: "call-1".into(),
            humanized: "Example command".into(),
        };
        apply_to_lines(&mut live, &renderer.render(&start), true);
        for index in 0..100 {
            let event = AgentEvent::ToolProgress {
                id: "call-1".into(),
                line: format!("progress {index}"),
            };
            apply_to_lines(&mut live, &renderer.render(&event), true);
            assert!(live.len() <= 15);
        }
        let result = AgentEvent::ToolResult {
            name: "exec".into(),
            id: "call-1".into(),
            output: "command completed".into(),
            success: true,
            elapsed_ms: 1,
            sources: Vec::new(),
        };
        apply_to_lines(&mut live, &renderer.render(&result), true);
        let text: String = live
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect();
        assert!(text.contains("preceding answer"));
        assert!(text.contains("command completed"));
        assert!(!text.contains("progress 99"));
        assert!(!renderer.tool_running());
    }
}
