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
//! # Event mapping
//!
//! | Event | Output |
//! |-------|--------|
//! | `TextDelta` | Buffered; flushed at newline boundaries in plain text style |
//! | `ReasoningDelta` | Indented dim text |
//! | `ToolStart` | Yellow ▶ icon with tool name and humanized args |
//! | `ToolResult` | Green ✓ or red ✗ with elapsed time; errors show up to 5 lines of output |
//! | `FileDiff` | Path header + green/red added/removed lines |
//! | `Status` | Dim italic message |
//! | `Usage` | Token counts in dim text |
//! | `Error` | Bold red "error: ..." prefix |
//! | `Done` | Flushes buffered text, adds a blank separator line |
//! | `SubagentEvent` | Prefixes inner event output with `[task_name]` in magenta |

use flashmind_types::AgentEvent;
use ratatui::text::{Line, Span};

use crate::styles::*;

/// Renders [`AgentEvent`] variants into styled terminal lines.
///
/// Buffers partial text deltas across events so that only complete lines are
/// flushed until the final event (`Done` or `Error`) calls [`flush`][EventRenderer::flush].
///
/// When the `markdown` feature is enabled, buffered text is rendered through the
/// markdown parser (bold, headings, code blocks, tables, etc.) instead of plain text.
pub struct EventRenderer {
    text_buffer: String,
}

impl EventRenderer {
    /// Create a new, empty renderer.
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
        }
    }

    /// Render an agent event into one or more styled lines.
    ///
    /// Text deltas are buffered internally and only emitted once a newline boundary
    /// is found.  Call [`flush`][EventRenderer::flush] to drain any remaining
    /// partial text (e.g., at the end of a turn).
    pub fn render(&mut self, event: &AgentEvent) -> Vec<Line<'static>> {
        match event {
            AgentEvent::TextDelta(text) => {
                self.text_buffer.push_str(text);
                self.flush_complete()
            }

            AgentEvent::ReasoningDelta(text) => {
                vec![Line::from(Span::styled(
                    format!("  {text}"),
                    S_DIM,
                ))]
            }

            AgentEvent::ToolStart { name, humanized, .. } => {
                let mut lines = self.flush();
                lines.push(Line::from(vec![
                    Span::styled("▶ ", S_TOOL_RUN),
                    Span::styled(name.clone(), S_TOOL_RUN),
                    Span::styled(format!(": {humanized}"), S_DIM),
                ]));
                lines
            }

            AgentEvent::ToolResult {
                name,
                success,
                elapsed_ms,
                output,
                ..
            } => {
                let (icon, style) = if *success {
                    ("✓", S_TOOL_OK)
                } else {
                    ("✗", S_TOOL_FAIL)
                };
                let elapsed = format_elapsed(*elapsed_ms);
                let mut lines = vec![Line::from(vec![
                    Span::styled(format!("{icon} "), style),
                    Span::styled(name.clone(), style),
                    Span::styled(format!(" ({elapsed})"), S_DIM),
                ])];
                if !success && !output.is_empty() {
                    for l in output.lines().take(5) {
                        lines.push(Line::from(Span::styled(
                            format!("  {l}"),
                            S_TOOL_FAIL,
                        )));
                    }
                }
                lines
            }

            AgentEvent::FileDiff { path, diff } => {
                let mut lines = vec![Line::from(Span::styled(
                    format!("  {path}"),
                    S_DIM,
                ))];
                for d in diff {
                    let line = match d {
                        flashmind_types::tool::DiffLine::Added { content, .. } => {
                            Line::from(Span::styled(format!("  + {content}"), S_DIFF_ADD))
                        }
                        flashmind_types::tool::DiffLine::Removed { content, .. } => {
                            Line::from(Span::styled(format!("  - {content}"), S_DIFF_DEL))
                        }
                    };
                    lines.push(line);
                }
                lines
            }

            AgentEvent::Status(msg) => {
                vec![Line::from(Span::styled(msg.clone(), S_STATUS))]
            }

            AgentEvent::Usage(_) => Vec::new(),

            AgentEvent::Error(msg) => {
                vec![Line::from(Span::styled(format!("error: {msg}"), S_ERROR))]
            }

            AgentEvent::Done(_) => {
                let mut lines = self.flush();
                lines.push(Line::from(""));
                lines
            }

            AgentEvent::SubagentEvent {
                task, event, ..
            } => {
                let inner = self.render(event);
                inner
                    .into_iter()
                    .map(|mut line| {
                        line.spans.insert(
                            0,
                            Span::styled(format!("  [{task}] "), S_SUBAGENT),
                        );
                        line
                    })
                    .collect()
            }

            _ => Vec::new(),
        }
    }

    /// Flush complete blocks/lines from the buffer, keeping any trailing
    /// incomplete content for the next delta.
    fn flush_complete(&mut self) -> Vec<Line<'static>> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        render_text_incremental(&mut self.text_buffer)
    }

    /// Drain any remaining buffered text into styled lines.
    ///
    /// Called automatically when a `Done` or `Error` event is rendered, but can
    /// also be called manually to ensure no text is lost.
    pub fn flush(&mut self) -> Vec<Line<'static>> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.text_buffer);
        render_text_lines(&text)
    }
}

impl Default for EventRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

/// Render all text through the markdown pipeline (final flush).
#[cfg(feature = "markdown")]
fn render_text_lines(text: &str) -> Vec<Line<'static>> {
    crate::markdown_render::render_to_lines(text)
}

/// Render text as plain styled spans (final flush).
/// Joins single newlines into spaces (paragraph wrapping) and splits only on
/// double newlines (paragraph breaks).
#[cfg(not(feature = "markdown"))]
fn render_text_lines(text: &str) -> Vec<Line<'static>> {
    text_to_paragraph_lines(text)
}

/// Convert text to lines, joining single newlines into spaces.
/// Double newlines create paragraph breaks (empty line between them).
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
    // Remove trailing empty line
    if lines.last().is_some_and(|l| l.spans.is_empty()) {
        lines.pop();
    }
    lines
}

/// Incrementally flush complete markdown blocks.
/// Keeps the last block in the buffer (it may still be accumulating).
/// Only flushes when there are 2+ parsed blocks — with a single block we
/// can't tell if it's complete yet.
#[cfg(feature = "markdown")]
fn render_text_incremental(buffer: &mut String) -> Vec<Line<'static>> {
    let result = crate::markdown::parse_document(buffer);

    if result.blocks.len() < 2 {
        return Vec::new();
    }

    let before_last_offset = buffer.len() - result.before_last.len();
    if before_last_offset == 0 || !buffer.is_char_boundary(before_last_offset) {
        return Vec::new();
    }

    let lines = crate::markdown_render::render_to_lines(&buffer[..before_last_offset]);
    let remainder = buffer[before_last_offset..].to_string();
    *buffer = remainder;
    lines
}

/// Incrementally flush complete paragraphs (plain text mode).
/// Only flushes when a double-newline (paragraph break) is found.
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
