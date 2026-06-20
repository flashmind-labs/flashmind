//! Event-to-line rendering.
//!
//! Translates [`flashmind_types::AgentEvent`] variants into styled
//! [`ratatui::text::Line`] values suitable for incremental terminal output.
//!
//! # Buffering strategy
//!
//! Text deltas (`TextDelta` events) are accumulated in an internal buffer rather
//! than emitted immediately.  Complete lines (delimited by `\n`) are flushed as
//! they arrive, but any trailing partial line is held until the next newline or
//! until [`flush`][EventRenderer::flush] is called.  This ensures that rapid
//! token streaming doesn't produce fragmented single-character output on screen.
//!
//! Non-text events (tool start, tool result, file diff, status, usage, error)
//! are rendered immediately without buffering.
//!
//! # Render actions
//!
//! The renderer produces [`RenderAction`] values rather than plain lines.
//! Most events produce `Append` actions, but `ToolResult` produces a
//! `ReplaceTool` action that instructs the caller to erase the running-tool
//! lines and replace them with the final result lines (in-place update).
//!
//! # Event mapping
//!
//! | Event | Output |
//! |-------|--------|
//! | `TextDelta` | Buffered; flushed at newline boundaries in plain text style |
//! | `ReasoningDelta` | Indented dim text |
//! | `ToolStart` | Yellow ◌ icon with humanized description |
//! | `ToolResult` | Green ✓ or red ✗ with right-aligned elapsed time (in-place update) |
//! | `FileDiff` | Path header + green/red added/removed lines (truncated at 100) |
//! | `Status` | Dim italic message |
//! | `Usage` | Token counts stored for footer |
//! | `Error` | Bold red "error: ..." prefix |
//! | `Done` | Flushes buffered text, adds usage footer + blank separator |
//! | `SpawnedEvent` | Prefixes inner event output with `[task_name]` in magenta |

use std::time::Instant;

use flashmind_types::AgentEvent;
use flashmind_types::llm::TokenUsage;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::styles::*;

const SILENT_TOOLS: &[&str] = &["str_replace", "file_write"];

// ---------------------------------------------------------------------------
// Public types

/// An output action produced by [`EventRenderer::render`].
pub enum RenderAction {
    /// Append a line to the output.
    Append(Line<'static>),
    /// Replace the most recent tool lines (erase `erase_count` lines, then
    /// print `lines` in their place).
    ReplaceTool {
        erase_count: usize,
        lines: Vec<Line<'static>>,
    },
}

/// Renders [`AgentEvent`] variants into styled terminal lines.
///
/// Buffers partial text deltas across events so that only complete lines are
/// flushed until the final event (`Done` or `Error`) calls [`flush`][EventRenderer::flush].
///
/// When the `markdown` feature is enabled, buffered text is rendered through the
/// markdown parser (bold, headings, code blocks, tables, etc.) instead of plain text.
#[derive(Debug)]
pub struct EventRenderer {
    text_buffer: String,
    reasoning_buffer: String,
    in_reasoning: bool,
    /// Number of lines emitted for the current running tool (ToolStart).
    tool_line_count: usize,
    /// Name/humanized of the current running tool.
    tool_info: Option<(String, String)>,
    /// When the current tool started (client-side elapsed tracking).
    tool_start: Option<Instant>,
    /// Last usage for footer rendering.
    last_usage: Option<TokenUsage>,
    /// When the current turn started.
    turn_start: Option<Instant>,
    /// When false, reasoning is collapsed to a one-line summary on completion
    /// (the streaming deltas are buffered silently).  Default: `true`.
    expand_reasoning: bool,
    /// Terminal width for right-aligned elapsed times.
    width: usize,
}

impl EventRenderer {
    /// Create a new, empty renderer.
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            in_reasoning: false,
            expand_reasoning: true,
            tool_line_count: 0,
            tool_info: None,
            tool_start: None,
            last_usage: None,
            turn_start: None,
            width: 80,
        }
    }

    /// Set the terminal width for right-aligned elapsed times.
    pub fn set_width(&mut self, width: usize) {
        self.width = width;
    }

    /// Mark the start of a new turn (for elapsed time display).
    pub fn mark_turn_start(&mut self) {
        self.turn_start = Some(Instant::now());
    }

    /// Whether a tool is currently running (for live elapsed tick updates).
    pub fn tool_running(&self) -> bool {
        self.tool_info.is_some()
    }

    /// Render the current running tool line with live elapsed for tick updates.
    /// Returns a `ReplaceTool` action if a tool is running, empty otherwise.
    pub fn tick_tool(&mut self) -> Vec<RenderAction> {
        let Some((ref name, ref humanized)) = self.tool_info else {
            return Vec::new();
        };
        let ms = self
            .tool_start
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let elapsed = format_elapsed(ms);

        let max_text = self.width.saturating_sub(12);
        let display = if humanized.is_empty() {
            format!("  \u{25cc} {name}")
        } else {
            truncate_display(&format!("  \u{25cc} {humanized}"), max_text)
        };
        let tool_cols = UnicodeWidthStr::width(display.as_str());
        let padded_elapsed = format!("{:>w$}", elapsed, w = self.width.saturating_sub(tool_cols));

        let erase_count = self.tool_line_count;
        if erase_count > 0 {
            let lines = vec![Line::from(vec![
                Span::styled(display, S_TOOL_RUN),
                Span::styled(padded_elapsed, S_DIM),
            ])];
            self.tool_line_count = lines.len();
            vec![RenderAction::ReplaceTool {
                erase_count,
                lines,
            }]
        } else {
            Vec::new()
        }
    }

    /// Render an agent event into render actions.
    pub fn render(&mut self, event: &AgentEvent) -> Vec<RenderAction> {
        match event {
            AgentEvent::TextDelta(text) => {
                let mut actions = Vec::new();
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
                self.text_buffer.push_str(text);
                actions.extend(self.flush_complete());
                actions
            }

            AgentEvent::ReasoningDelta(text) => {
                let mut actions = Vec::new();
                if !self.in_reasoning {
                    self.in_reasoning = true;
                    actions.push(RenderAction::Append(Line::from("")));
                }
                self.reasoning_buffer.push_str(text);
                actions.extend(
                    self.flush_reasoning_complete()
                        .into_iter()
                        .map(RenderAction::Append),
                );
                actions
            }

            AgentEvent::ToolStart {
                name, humanized, ..
            } => {
                let mut actions = Vec::new();
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
                actions.extend(self.flush());

                if SILENT_TOOLS.contains(&name.as_str()) {
                    self.tool_line_count = 0;
                    self.tool_info = None;
                    self.tool_start = None;
                    return actions;
                }

                let (primary, body) = split_humanized(humanized);

                let max_text = self.width.saturating_sub(12);
                let display = if primary.is_empty() {
                    format!("  \u{25cc} {name}")
                } else {
                    let full = format!("  \u{25cc} {primary}");
                    truncate_display(&full, max_text)
                };

                actions.push(RenderAction::Append(Line::from(Span::styled(
                    display, S_TOOL_RUN,
                ))));
                let mut line_count = 1usize;

                for cont in body {
                    let indent = "      ";
                    let avail = self.width.saturating_sub(indent.len());
                    let rendered = truncate_display(&format!("{indent}{cont}"), avail);
                    actions.push(RenderAction::Append(Line::from(Span::styled(
                        rendered, S_TOOL_RUN,
                    ))));
                    line_count += 1;
                }

                self.tool_line_count = line_count;
                self.tool_info = Some((name.clone(), primary));
                self.tool_start = Some(Instant::now());
                actions
            }

            AgentEvent::ToolResult {
                name,
                success,
                elapsed_ms,
                output,
                ..
            } => {
                let (tool_name, humanized) = self
                    .tool_info
                    .take()
                    .unwrap_or_else(|| (name.clone(), String::new()));

                let erase_count = self.tool_line_count;
                self.tool_line_count = 0;
                self.tool_start = None;

                if *success && SILENT_TOOLS.contains(&name.as_str()) && erase_count == 0 {
                    return Vec::new();
                }

                let ms = if *elapsed_ms > 0 {
                    *elapsed_ms
                } else {
                    self.tool_start
                        .take()
                        .map(|t| t.elapsed().as_millis() as u64)
                        .unwrap_or(0)
                };
                let elapsed = format_elapsed(ms);

                let (icon, style) = if *success {
                    ("\u{2713}", S_TOOL_OK)
                } else {
                    ("\u{2717}", S_TOOL_FAIL)
                };

                let elapsed_col = 10;
                let max_text = self.width.saturating_sub(elapsed_col + 2);
                let tool_text = if humanized.is_empty() {
                    format!("  {icon} {tool_name}")
                } else {
                    truncate_display(&format!("  {icon} {humanized}"), max_text)
                };

                let tool_cols = UnicodeWidthStr::width(tool_text.as_str());
                let padded_elapsed =
                    format!("{:>w$}", elapsed, w = self.width.saturating_sub(tool_cols));

                let mut result_lines = vec![Line::from(vec![
                    Span::styled(tool_text, style),
                    Span::styled(padded_elapsed, S_DIM),
                ])];

                if !success && !output.is_empty() {
                    for l in output.lines().take(5) {
                        result_lines
                            .push(Line::from(Span::styled(format!("    {l}"), S_TOOL_FAIL)));
                    }
                }

                if erase_count > 0 {
                    vec![RenderAction::ReplaceTool {
                        erase_count,
                        lines: result_lines,
                    }]
                } else {
                    result_lines.into_iter().map(RenderAction::Append).collect()
                }
            }

            AgentEvent::FileDiff { path, diff } => {
                // Successful file edit (str_replace / file_write): render each
                // line on a subtle tinted background that fills the full
                // terminal width.  print_line emits Clear(UntilNewLine) when a
                // span has a background, so the bg extends to the right edge.
                let mut lines = vec![Line::from(Span::styled(
                    format!("  {path}"),
                    S_DIFF_ADD,
                ))];
                let total = diff.len();
                let render_dl = |dl: &flashmind_types::tool::DiffLine| -> Line<'static> {
                    match dl {
                        flashmind_types::tool::DiffLine::Added { content, .. } => {
                            Line::from(Span::styled(format!("    +{content}"), S_DIFF_BLOCK_OK))
                        }
                        flashmind_types::tool::DiffLine::Removed { content, .. } => {
                            Line::from(Span::styled(format!("    -{content}"), S_DIFF_BLOCK_DEL))
                        }
                    }
                };

                if total > 100 {
                    for dl in &diff[..50] {
                        lines.push(render_dl(dl));
                    }
                    lines.push(Line::from(Span::styled(
                        format!("    ... {} lines omitted ...", total - 100),
                        S_DIFF_BLOCK_OK,
                    )));
                    for dl in &diff[total - 50..] {
                        lines.push(render_dl(dl));
                    }
                } else {
                    for dl in diff {
                        lines.push(render_dl(dl));
                    }
                }
                lines.into_iter().map(RenderAction::Append).collect()
            }

            AgentEvent::Status(msg) => {
                vec![RenderAction::Append(Line::from(Span::styled(
                    format!("[{msg}]"),
                    S_DIM,
                )))]
            }

            AgentEvent::Usage(usage) => {
                if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                    self.last_usage = Some(usage.clone());
                }
                Vec::new()
            }

            AgentEvent::Error(msg) => {
                let mut actions = self.flush();
                actions.push(RenderAction::Append(Line::from(Span::styled(
                    format!("[error] {msg}"),
                    S_ERROR,
                ))));
                actions
            }

            AgentEvent::Compacted(summary) => {
                let mut actions = vec![
                    RenderAction::Append(Line::from("")),
                    RenderAction::Append(Line::from(Span::styled("[compacted]", S_DIM))),
                ];
                let lines = render_text_lines(summary);
                actions.extend(lines.into_iter().map(RenderAction::Append));
                actions
            }

            AgentEvent::Done(_) => {
                let mut actions = Vec::new();
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
                actions.extend(self.flush());

                self.last_usage.take();
                if let Some(elapsed) = self
                    .turn_start
                    .map(|t| format_elapsed(t.elapsed().as_millis() as u64))
                    && !elapsed.is_empty()
                {
                    actions.push(RenderAction::Append(Line::from("")));
                    actions.push(RenderAction::Append(Line::from(Span::styled(
                        elapsed, S_DIM,
                    ))));
                }

                actions.push(RenderAction::Append(Line::from("")));
                self.turn_start = None;
                actions
            }

            AgentEvent::SpawnedEvent { task, event, .. } => {
                let inner = self.render(event);
                inner
                    .into_iter()
                    .map(|action| match action {
                        RenderAction::Append(mut line) => {
                            line.spans
                                .insert(0, Span::styled(format!("  [{task}] "), S_SPAWNED));
                            RenderAction::Append(line)
                        }
                        other => other,
                    })
                    .collect()
            }

            _ => Vec::new(),
        }
    }

    /// Flush complete blocks from the reasoning buffer.
    fn flush_reasoning_complete(&mut self) -> Vec<Line<'static>> {
        if !self.expand_reasoning || self.reasoning_buffer.is_empty() {
            return Vec::new();
        }
        dim_lines(render_text_incremental(&mut self.reasoning_buffer))
    }

    /// Drain all remaining reasoning buffer content.
    fn flush_reasoning(&mut self) -> Vec<Line<'static>> {
        if self.reasoning_buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.reasoning_buffer);
        if self.expand_reasoning {
            dim_lines(render_text_lines(&text))
        } else {
            // Collapsed: emit a single summary line instead of the full text.
            let lines = text.lines().count();
            if lines > 0 {
                vec![Line::from(Span::styled(
                    format!("\u{25be} thinking ({lines} lines)"),
                    S_DIM,
                ))]
            } else {
                Vec::new()
            }
        }
    }

    /// End the current reasoning block: flush (or summarize) the buffer and
    /// append a separator.  Returns the render actions for the transition.
    fn end_reasoning(&mut self) -> Vec<RenderAction> {
        self.in_reasoning = false;
        let mut actions: Vec<RenderAction> = self
            .flush_reasoning()
            .into_iter()
            .map(RenderAction::Append)
            .collect();
        actions.push(RenderAction::Append(make_separator(self.width)));
        actions
    }

    /// Set whether reasoning blocks are expanded (full text) or collapsed
    /// (one-line summary).  Affects subsequent reasoning blocks.
    pub fn set_expand_reasoning(&mut self, expand: bool) {
        self.expand_reasoning = expand;
    }

    /// Whether reasoning blocks are currently expanded.
    pub fn expand_reasoning(&self) -> bool {
        self.expand_reasoning
    }

    /// Flush committed blocks/lines from the buffer.
    ///
    /// Any trailing partial text (no newline yet) stays in the buffer and is
    /// accessible via [`partial_text`] for the caller to display ephemerally.
    fn flush_complete(&mut self) -> Vec<RenderAction> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        render_text_incremental(&mut self.text_buffer)
            .into_iter()
            .map(RenderAction::Append)
            .collect()
    }

    /// Drain any remaining buffered text into styled lines.
    pub fn flush(&mut self) -> Vec<RenderAction> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.text_buffer);
        render_text_lines(&text)
            .into_iter()
            .map(RenderAction::Append)
            .collect()
    }

    /// The trailing incomplete line still being streamed.
    ///
    /// Returns only the text after the last newline - complete lines that are
    /// held in the buffer for structural reasons (e.g. table rows waiting for
    /// the block parser) are not included.  This keeps the ephemeral partial
    /// display to a single line that the caller can safely erase and redraw.
    pub fn partial_text(&self) -> &str {
        match self.text_buffer.rfind('\n') {
            Some(pos) => &self.text_buffer[pos + 1..],
            None => &self.text_buffer,
        }
    }
}

impl Default for EventRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers

fn make_separator(width: usize) -> Line<'static> {
    Line::from(Span::styled(
        "\u{2500}".repeat(width.saturating_sub(1)),
        S_DIM,
    ))
}

fn dim_lines(lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .map(|mut line| {
            for span in &mut line.spans {
                span.style = span.style.add_modifier(ratatui::style::Modifier::DIM);
            }
            line
        })
        .collect()
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

/// Split multi-line humanized text into a primary line and continuation lines.
fn split_humanized(humanized: &str) -> (String, Vec<String>) {
    let mut lines = humanized.lines();
    let primary = lines.next().unwrap_or("").to_string();
    let body: Vec<String> = lines.map(|l| l.to_string()).collect();
    (primary, body)
}

/// Truncate a display string to fit within `max_width`, appending `...` if needed.
fn truncate_display(text: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_width {
        text.to_string()
    } else {
        let mut result = String::new();
        let mut w = 0;
        for ch in text.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if w + cw >= max_width {
                break;
            }
            result.push(ch);
            w += cw;
        }
        result.push('\u{2026}');
        result
    }
}

/// Render all text through the markdown pipeline (final flush).
#[cfg(feature = "markdown")]
fn render_text_lines(text: &str) -> Vec<Line<'static>> {
    crate::markdown_render::render_to_lines(text)
}

/// Render text as plain styled spans (final flush).
#[cfg(not(feature = "markdown"))]
fn render_text_lines(text: &str) -> Vec<Line<'static>> {
    text_to_paragraph_lines(text)
}

/// Convert text to lines, joining single newlines into spaces.
#[cfg(not(feature = "markdown"))]
fn text_to_paragraph_lines(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for paragraph in text.split("\n\n") {
        let joined = paragraph
            .split('\n')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !joined.is_empty() {
            lines.push(Line::from(Span::styled(joined, S_TEXT)));
        }
        lines.push(Line::from(""));
    }
    if lines.last().is_some_and(|l| l.spans.is_empty()) {
        lines.pop();
    }
    lines
}

/// Incrementally flush complete lines through the markdown pipeline.
///
/// Renders every complete line (ending with `\n`) immediately - inline
/// formatting (bold, italic, code, headings, lists) appears as soon as the
/// line arrives, matching the pi.dev streaming style.
///
/// Structural blocks (code fences, tables) are held until the block parser
/// recognises them as complete - flushing individual rows would break their
/// rendering.
#[cfg(feature = "markdown")]
fn render_text_incremental(buffer: &mut String) -> Vec<Line<'static>> {
    let result = crate::markdown::parse_document(buffer);

    if result.incomplete {
        if result.blocks.is_empty() {
            return Vec::new();
        }
        let consumed = buffer.len() - result.rest.len();
        if consumed == 0 || !buffer.is_char_boundary(consumed) {
            return Vec::new();
        }
        let lines = crate::markdown_render::render_to_lines(&buffer[..consumed]);
        *buffer = buffer[consumed..].to_string();
        return lines;
    }

    if result.blocks.len() >= 2 {
        let before_last_offset = buffer.len() - result.before_last.len();
        if before_last_offset > 0 && buffer.is_char_boundary(before_last_offset) {
            let lines = crate::markdown_render::render_to_lines(&buffer[..before_last_offset]);
            *buffer = buffer[before_last_offset..].to_string();
            return lines;
        }
    }

    if buffer.trim_start().starts_with('|') {
        return Vec::new();
    }

    let Some(pos) = buffer.rfind('\n') else {
        return Vec::new();
    };

    let lines = crate::markdown_render::render_to_lines(&buffer[..=pos]);
    *buffer = buffer[pos + 1..].to_string();
    lines
}

/// Incrementally flush complete paragraphs (plain text mode).
#[cfg(not(feature = "markdown"))]
fn render_text_incremental(buffer: &mut String) -> Vec<Line<'static>> {
    if let Some(pos) = buffer.rfind("\n\n") {
        let complete = buffer[..pos].to_string();
        *buffer = buffer[pos + 2..].to_string();
        text_to_paragraph_lines(&complete)
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::AgentEvent;

    fn new_renderer() -> EventRenderer {
        let mut r = EventRenderer::new();
        r.set_width(80);
        r
    }

    #[test]
    fn collapsed_reasoning_emits_single_summary_line() {
        let mut r = new_renderer();
        r.set_expand_reasoning(false);

        // Stream some reasoning text.
        let _ = r.render(&AgentEvent::ReasoningDelta("considering options\n".into()));
        let _ = r.render(&AgentEvent::ReasoningDelta("second line\n".into()));
        let _ = r.render(&AgentEvent::ReasoningDelta("third".into()));

        // A TextDelta ends the reasoning block.
        let actions = r.render(&AgentEvent::TextDelta("answer".into()));

        // Find the summary line (not the separator, not the answer text).
        let summary = actions
            .iter()
            .filter_map(|a| match a {
                RenderAction::Append(Line { spans, .. }) => spans.first(),
                _ => None,
            })
            .find(|s| s.content.contains("thinking"))
            .expect("a thinking summary line");
        assert!(summary.content.contains("3 lines"), "got: {}", summary.content);
    }

    #[test]
    fn expanded_reasoning_emits_full_text() {
        let mut r = new_renderer();
        r.set_expand_reasoning(true);

        let a1 = r.render(&AgentEvent::ReasoningDelta("a thought\n".into()));
        let a2 = r.render(&AgentEvent::TextDelta("answer".into()));
        let actions: Vec<&RenderAction> = a1.iter().chain(a2.iter()).collect();

        let has_thought = actions.iter().any(|a| match a {
            RenderAction::Append(Line { spans, .. }) => {
                spans.iter().any(|s| s.content.contains("a thought"))
            }
            _ => false,
        });
        assert!(has_thought, "expanded reasoning should include full text");
    }
}
