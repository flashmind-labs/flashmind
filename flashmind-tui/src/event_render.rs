//! Event-to-line rendering.
//!
//! Translates [`flashmind_types::AgentEvent`] variants into styled
//! [`ratatui::text::Line`] values suitable for incremental terminal output.
//!
//! # Buffering strategy
//!
//! Text deltas (`TextDelta` events) are accumulated in an internal buffer rather
//! than emitted immediately. Complete blocks are committed to the transcript;
//! the open block stays available through [`EventRenderer::pending_text`] for
//! live preview until it closes or [`flush`][EventRenderer::flush] is called.
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
//! | `ToolStart` | Teal ◌ glyph with humanized description |
//! | `ToolResult` | Teal ✓ on success or soft-red ✗ on failure, with right-aligned elapsed time (in-place update) |
//! | `FileDiff` | Path header + green/red added/removed lines (truncated at 100) |
//! | `Status` | Dim italic message |
//! | `Usage` | Token counts stored for footer |
//! | `Error` | Bold red "error: ..." prefix |
//! | `Done` | Flushes buffered text, adds usage footer + blank separator |
//! | `SpawnedEvent` | Prefixes inner event output with `[task_name]` in magenta |

use std::time::Instant;

use flashmind_types::AgentEvent;
use flashmind_types::llm::TokenUsage;
use ratatui::style::Style;
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

#[derive(Debug)]
struct RunningTool {
    name: String,
    description: String,
    diffs: Vec<Line<'static>>,
}

/// Renders [`AgentEvent`] variants into styled terminal lines.
///
/// Buffers text deltas across events and commits complete blocks. The final
/// event (`Done` or `Error`) calls [`flush`][EventRenderer::flush].
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
    /// Name and full description of the current running tool.
    tool_info: Option<RunningTool>,
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
    /// Accumulated output lines for resize redraw.
    lines: Vec<Line<'static>>,
    /// Streaming progress lines from the current running tool.
    tool_progress_lines: Vec<String>,
    /// Whether tool output previews are shown (true) or collapsed (false).
    expand_tools: bool,
    /// When true, `SpawnedEvent` rendering is suppressed (handled by the
    /// `SubagentProgress` widget instead).
    suppress_spawned: bool,
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
            lines: Vec::new(),
            tool_progress_lines: Vec::new(),
            expand_tools: true,
            suppress_spawned: false,
        }
    }

    /// Set the terminal width for right-aligned elapsed times.
    ///
    /// Also regenerates any separator lines in the buffer to match the new
    /// width, so resize redraws show full-width horizontal rules.
    pub fn set_width(&mut self, width: usize) {
        if width != self.width {
            for line in &mut self.lines {
                if is_separator_line(line) {
                    *line = make_separator(width);
                }
            }
        }
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

    /// Discard any running-tool tracking without rendering a result.
    ///
    /// The spinner tick (`tick_tool`) and [`ToolResult`][AgentEvent::ToolResult]
    /// replacement both assume the running-tool line is still the last thing on
    /// screen.  That invariant holds only while a tool is genuinely executing.
    /// If a caller emits a [`ToolStart`][AgentEvent::ToolStart] but resolves the
    /// tool through some side channel (e.g. an interactive interrupt handled
    /// out-of-band) without emitting a matching `ToolResult`, the tracking is
    /// left stale — and the next tick would erase and re-print the phantom line
    /// in the wrong place, corrupting the transcript.
    ///
    /// Callers that begin appending unrelated output (e.g. the start of a new
    /// LLM turn) call this first as a safety net so a forgotten `ToolResult`
    /// can never drive the tick against a line that has scrolled away.  The
    /// already-printed running line is left untouched; only the in-memory
    /// tracking is reset.
    pub fn clear_running_tool(&mut self) {
        self.tool_info = None;
        self.tool_line_count = 0;
        self.tool_start = None;
        self.tool_progress_lines.clear();
    }

    /// Render the current running tool line with live elapsed for tick updates.
    /// Returns a `ReplaceTool` action if a tool is running, empty otherwise.
    pub fn tick_tool(&mut self) -> Vec<RenderAction> {
        if self.tool_info.is_none() {
            return Vec::new();
        }
        let erase_count = self.tool_line_count;
        if erase_count == 0 {
            return Vec::new();
        }
        self.render_tool_with_progress()
    }

    fn render_tool_with_progress(&mut self) -> Vec<RenderAction> {
        let Some(tool) = &self.tool_info else {
            return Vec::new();
        };
        let ms = self
            .tool_start
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let elapsed = format_elapsed(ms);

        let erase_count = self.tool_line_count;
        let mut lines = tool_block(tool, "\u{25cc}", &elapsed, S_TOOL_RUN);

        const MAX_PROGRESS: usize = 10;
        let limit = if self.expand_tools { MAX_PROGRESS } else { 0 };
        let skip = self.tool_progress_lines.len().saturating_sub(limit);
        if skip > 0 && self.expand_tools {
            lines.push(tool_body(&format!("... {skip} more lines"), S_DIM));
        }
        for line in self.tool_progress_lines.iter().skip(skip) {
            lines.push(tool_body(line, S_DIM));
        }
        lines.extend(tool.diffs.iter().cloned());

        self.tool_line_count = lines.len();
        if erase_count > 0 {
            vec![RenderAction::ReplaceTool { erase_count, lines }]
        } else {
            lines.into_iter().map(RenderAction::Append).collect()
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

                let tool = RunningTool {
                    name: name.clone(),
                    description: humanized.clone(),
                    diffs: Vec::new(),
                };
                let lines = tool_block(&tool, "\u{25cc}", "", S_TOOL_RUN);
                self.tool_line_count = lines.len();
                actions.extend(lines.into_iter().map(RenderAction::Append));
                self.tool_info = Some(tool);
                self.tool_progress_lines.clear();
                self.tool_start = Some(Instant::now());
                actions
            }

            AgentEvent::ToolProgress { line, .. } => {
                if self.tool_info.is_none() {
                    return Vec::new();
                }
                self.tool_progress_lines.push(line.clone());
                self.render_tool_with_progress()
            }

            AgentEvent::ToolResult {
                name,
                success,
                elapsed_ms,
                output,
                ..
            } => {
                self.tool_progress_lines.clear();
                let tool = self.tool_info.take().unwrap_or_else(|| RunningTool {
                    name: name.clone(),
                    description: String::new(),
                    diffs: Vec::new(),
                });

                let erase_count = self.tool_line_count;
                self.tool_line_count = 0;
                let tool_start = self.tool_start.take();

                if *success && SILENT_TOOLS.contains(&name.as_str()) && erase_count == 0 {
                    return Vec::new();
                }

                let ms = if *elapsed_ms > 0 {
                    *elapsed_ms
                } else {
                    tool_start
                        .map(|t| t.elapsed().as_millis() as u64)
                        .unwrap_or(0)
                };
                let elapsed = format_elapsed(ms);

                let (icon, style) = if *success {
                    ("\u{2713}", S_TOOL_OK)
                } else {
                    ("\u{2717}", S_TOOL_FAIL)
                };

                let mut result_lines = tool_block(&tool, icon, &elapsed, style);

                if !output.is_empty() {
                    let is_silent = SILENT_TOOLS.contains(&name.as_str()) || !tool.diffs.is_empty();
                    // For silent edit tools the FileDiff event already shows
                    // the change, so we skip the textual output preview on
                    // success.  Failures always show a few lines (red).
                    if !success {
                        let limit = if self.expand_tools { usize::MAX } else { 5 };
                        for line in output.lines().take(limit) {
                            result_lines.push(tool_body(line, S_TOOL_FAIL));
                        }
                        let hidden = output.lines().count().saturating_sub(limit);
                        if hidden > 0 {
                            result_lines.push(tool_body(
                                &format!("... {hidden} more lines (/tools expand)"),
                                S_TOOL_FAIL,
                            ));
                        }
                    } else if !is_silent && self.expand_tools {
                        result_lines.extend(output.lines().map(|line| tool_body(line, S_DIM)));
                    }
                }
                result_lines.extend(tool.diffs);

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
                let mut lines = vec![
                    Line::default(),
                    Line::from(Span::styled(format!("  {path}"), S_DIFF_ADD)),
                ];
                let total = diff.len();
                let render_dl = |dl: &flashmind_types::tool::DiffLine| -> Line<'static> {
                    match dl {
                        flashmind_types::tool::DiffLine::Added { content, .. } => {
                            Line::from(Span::styled(format!("    +{content}"), S_DIFF_BLOCK_OK))
                        }
                        flashmind_types::tool::DiffLine::Removed { content, .. } => {
                            Line::from(Span::styled(format!("    -{content}"), S_DIFF_BLOCK_DEL))
                        }
                        flashmind_types::tool::DiffLine::Context { content, line } => {
                            if content.is_empty() && *line == 0 {
                                // Elision marker: a gap in the displayed diff.
                                Line::from(Span::styled("    …", S_DIM))
                            } else {
                                Line::from(Span::styled(format!("     {content}"), S_DIFF_CONTEXT))
                            }
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
                if let Some(tool) = self.tool_info.as_mut() {
                    tool.diffs.extend(lines);
                    self.render_tool_with_progress()
                } else {
                    lines.into_iter().map(RenderAction::Append).collect()
                }
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
                let lines = render_text_lines(summary, self.width);
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
                if self.suppress_spawned {
                    return Vec::new();
                }
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
        dim_lines(render_text_incremental(
            &mut self.reasoning_buffer,
            self.width,
        ))
    }

    /// Drain all remaining reasoning buffer content.
    fn flush_reasoning(&mut self) -> Vec<Line<'static>> {
        if self.reasoning_buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.reasoning_buffer);
        if self.expand_reasoning {
            dim_lines(render_text_lines(&text, self.width))
        } else {
            // Collapsed: emit a single summary line instead of the full text.
            let lines = text.lines().filter(|l| !l.trim().is_empty()).count();
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
    /// leave a blank line before the answer.
    fn end_reasoning(&mut self) -> Vec<RenderAction> {
        self.in_reasoning = false;
        let mut actions: Vec<RenderAction> = self
            .flush_reasoning()
            .into_iter()
            .map(RenderAction::Append)
            .collect();
        actions.push(RenderAction::Append(Line::default()));
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

    /// Set whether full tool output is shown or collapsed.
    pub fn set_expand_tools(&mut self, expand: bool) {
        self.expand_tools = expand;
    }

    /// Whether full tool output is currently shown.
    pub fn expand_tools(&self) -> bool {
        self.expand_tools
    }

    /// Suppress `SpawnedEvent` rendering (handled by the progress widget).
    pub fn set_suppress_spawned(&mut self, suppress: bool) {
        self.suppress_spawned = suppress;
    }

    /// Flush committed blocks/lines from the buffer.
    ///
    /// Any trailing partial text (no newline yet) stays in the buffer and is
    /// accessible via [`Self::pending_text`] for the live preview.
    fn flush_complete(&mut self) -> Vec<RenderAction> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        render_text_incremental(&mut self.text_buffer, self.width)
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
        render_text_lines(&text, self.width)
            .into_iter()
            .map(RenderAction::Append)
            .collect()
    }

    /// Markdown still being streamed, including complete lines in an open block.
    pub fn pending_text(&self) -> &str {
        &self.text_buffer
    }

    /// Uncommitted reasoning for the live preview when reasoning is expanded.
    pub fn pending_reasoning(&self) -> &str {
        if self.expand_reasoning {
            &self.reasoning_buffer
        } else {
            ""
        }
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

    /// Record emitted actions into the internal line buffer for resize redraw.
    pub fn record_actions(&mut self, actions: &[RenderAction]) {
        for action in actions {
            match action {
                RenderAction::Append(line) => {
                    self.lines.push(line.clone());
                }
                RenderAction::ReplaceTool { erase_count, lines } => {
                    let remove_from = self.lines.len().saturating_sub(*erase_count);
                    self.lines.truncate(remove_from);
                    self.lines.extend(lines.iter().cloned());
                }
            }
        }
    }

    /// All accumulated output lines (for resize redraw).
    pub fn lines(&self) -> &[Line<'static>] {
        &self.lines
    }

    /// Push an arbitrary line into the buffer (e.g. user echo, greeting).
    pub fn push_line(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }

    /// Clear accumulated lines (e.g. on new turn).
    pub fn clear_lines(&mut self) {
        self.lines.clear();
    }
}

impl Default for EventRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers

fn is_separator_line(line: &Line<'_>) -> bool {
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    !text.is_empty() && text.chars().all(|c| c == '\u{2500}')
}

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

// Tool headers keep three spans: marker, full label, and duration.
// Padding and truncation belong to the viewport so completed rows can resize.
fn tool_block(tool: &RunningTool, icon: &str, elapsed: &str, style: Style) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::default(),
        tool_header(icon, &tool.name, elapsed, style),
    ];
    lines.extend(tool.description.lines().map(|line| tool_body(line, S_TEXT)));
    lines
}

fn tool_body(text: &str, style: Style) -> Line<'static> {
    Line::from(Span::styled(format!("    {text}"), style))
}

fn tool_header(icon: &str, label: &str, elapsed: &str, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {icon} "), style),
        Span::styled(label.to_owned(), S_DIM),
        Span::styled(
            if elapsed.is_empty() {
                String::new()
            } else {
                format!("  {elapsed}")
            },
            S_DIM,
        ),
    ])
}

/// Fit a tool header to the current viewport, including headers from subagents.
pub(crate) fn fit_tool_line(line: &Line<'static>, width: usize) -> Line<'static> {
    let Some(marker_index) = line.spans.len().checked_sub(3) else {
        return line.clone();
    };
    if !matches!(
        line.spans[marker_index].content.as_ref(),
        "  \u{25cc} " | "  \u{2713} " | "  \u{2717} "
    ) {
        return line.clone();
    }
    let label = &line.spans[marker_index + 1];
    let elapsed = &line.spans[marker_index + 2];
    let duration = truncate_display(elapsed.content.trim(), width);
    let column = duration.width().max(8).min(width);
    let mut available = width.saturating_sub(column + 2);
    let mut spans = Vec::new();
    for span in &line.spans[..=marker_index] {
        let text = truncate_display(&span.content, available);
        available = available.saturating_sub(text.width());
        spans.push(Span::styled(text, span.style));
    }
    spans.push(Span::styled(
        truncate_display(&label.content, available),
        label.style,
    ));
    let used = spans.iter().map(Span::width).sum::<usize>() + duration.width();
    spans.push(Span::raw(" ".repeat(width.saturating_sub(used))));
    spans.push(Span::styled(duration, elapsed.style));
    Line {
        spans,
        ..line.clone()
    }
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else {
        format!("{:.1} s", ms as f64 / 1000.0)
    }
}

/// Truncate a display string to fit within `max_width`, appending `…` if needed.
///
/// The result's display width never exceeds `max_width`: the trailing ellipsis
/// occupies one column, so characters are kept only while they (plus the
/// ellipsis) still fit.  When `max_width` is `0` an empty string is returned.
pub(crate) fn truncate_display(text: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    // Reserve one column for the ellipsis we are about to append.
    let budget = max_width - 1;
    let mut result = String::new();
    let mut w = 0;
    for ch in text.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > budget {
            break;
        }
        result.push(ch);
        w += cw;
    }
    result.push('\u{2026}');
    result
}

/// Render all text through the markdown pipeline (final flush).
#[cfg(feature = "markdown")]
fn render_text_lines(text: &str, width: usize) -> Vec<Line<'static>> {
    crate::markdown_render::render_reflowable_lines(text, width.min(u16::MAX as usize) as u16)
}

/// Render text as plain styled spans (final flush).
#[cfg(not(feature = "markdown"))]
fn render_text_lines(text: &str, _width: usize) -> Vec<Line<'static>> {
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

/// Commit complete markdown blocks, leaving the last open block for preview.
/// Code fences and tables stay together until the parser can render them.
#[cfg(feature = "markdown")]
fn render_text_incremental(buffer: &mut String, width: usize) -> Vec<Line<'static>> {
    use crate::markdown::DocBlock;

    let Some(end) = buffer.rfind('\n').map(|pos| pos + 1) else {
        return Vec::new();
    };
    let complete = &buffer[..end];
    let result = crate::markdown::parse_document(complete);

    if result.incomplete {
        if result.blocks.is_empty() {
            return Vec::new();
        }
        let consumed = complete.len() - result.rest.len();
        if consumed == 0 || !buffer.is_char_boundary(consumed) {
            return Vec::new();
        }
        let lines = render_text_lines(&buffer[..consumed], width);
        *buffer = buffer[consumed..].to_string();
        return lines;
    }

    if result.blocks.len() >= 2 {
        let before_last_offset = complete.len() - result.before_last.len();
        if before_last_offset > 0 && buffer.is_char_boundary(before_last_offset) {
            let lines = render_text_lines(&buffer[..before_last_offset], width);
            *buffer = buffer[before_last_offset..].to_string();
            return lines;
        }
    }

    // Keep the last block live until a blank line closes it. A soft newline
    // alone can still be followed by another line in the same paragraph.
    let closed = match result.blocks.last() {
        Some(DocBlock::Paragraph(text)) => {
            let numbered = text
                .trim_start()
                .split_once(". ")
                .is_some_and(|(prefix, _)| {
                    !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit())
                });
            complete.ends_with("\n\n") && !numbered
        }
        Some(DocBlock::Heading { .. } | DocBlock::CodeBlock { .. } | DocBlock::HorizontalRule) => {
            true
        }
        _ => false,
    };
    if !closed {
        return Vec::new();
    }
    let lines = render_text_lines(complete, width);
    buffer.drain(..end);
    lines
}

/// Incrementally flush complete paragraphs (plain text mode).
#[cfg(not(feature = "markdown"))]
fn render_text_incremental(buffer: &mut String, _width: usize) -> Vec<Line<'static>> {
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
    #[cfg(feature = "markdown")]
    fn streaming_keeps_markdown_layout_independent_of_chunk_size() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        use ratatui::widgets::{Paragraph, Widget};

        let samples = [
            "First line\ncontinues here.\n\nSecond paragraph.\n",
            "Intro.\n1. **First finding**\nDetails.\n2. **Second finding**\nMore details.",
            "## Review\n\n1. **First finding**\n\nDetails here.\n\n2. **Second finding**\n\nMore details.",
            "- First item\n- Second item\n\nAfter the list.",
            "1. First item\n\n2. Second item\n\n3. Third item\n\nAfter the list.",
            "Before code.\n\n```rust\nfn example() {}\n```\n\nAfter code.",
            "Before table.\n\n| Name | Value |\n| --- | --- |\n| Example | One |\n\nAfter table.",
        ];
        for sample in samples {
            let chars: Vec<char> = sample.chars().collect();
            for chunk_size in [1, 7, chars.len()] {
                let mut renderer = new_renderer();
                for chunk in chars.chunks(chunk_size) {
                    let actions = renderer.render(&AgentEvent::TextDelta(chunk.iter().collect()));
                    renderer.record_actions(&actions);
                }
                let actions = renderer.flush();
                renderer.record_actions(&actions);
                let area = Rect::new(0, 0, 80, 100);
                let mut streamed = Buffer::empty(area);
                let mut complete = Buffer::empty(area);
                Paragraph::new(renderer.lines().to_vec()).render(area, &mut streamed);
                Paragraph::new(render_text_lines(sample, 80)).render(area, &mut complete);
                assert_eq!(streamed, complete, "chunk size {chunk_size}: {sample}");
            }
        }
    }

    #[test]
    fn collapsed_tools_keep_progress_quiet_and_errors_visible() {
        let mut renderer = new_renderer();
        renderer.set_expand_tools(false);
        renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "Run checks".into(),
        });
        let progress = renderer.render(&AgentEvent::ToolProgress {
            id: "1".into(),
            line: "verbose progress".into(),
        });
        assert!(!line_text(&progress).contains("verbose progress"));
        let result = renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            output: "check failed".into(),
            success: false,
            elapsed_ms: 12,
            sources: vec![],
        });
        assert!(line_text(&result).contains("check failed"));
        for width in [0, 1, 3, 8, 40, 120] {
            assert!(
                fit_tool_line(
                    &tool_header("\u{2713}", "A long example command", "12 ms", S_TOOL_OK),
                    width
                )
                .width()
                    <= width
            );
        }
    }

    #[test]
    fn tool_time_uses_the_start_instant_when_the_result_has_no_duration() {
        use std::time::Duration;

        let mut renderer = new_renderer();
        renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "Run checks".into(),
        });
        renderer.tool_start = Some(Instant::now() - Duration::from_millis(500));
        let actions = renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            success: true,
            elapsed_ms: 0,
            output: String::new(),
            sources: vec![],
        });
        let lines = match &actions[0] {
            RenderAction::ReplaceTool { lines, .. } => lines,
            _ => panic!("a completed tool replaces its live row"),
        };
        assert_ne!(lines[1].spans.last().unwrap().content.trim(), "0 ms");
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
        assert!(
            summary.content.contains("3 lines"),
            "got: {}",
            summary.content
        );
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

    #[test]
    fn successful_tool_shows_dimmed_output_preview() {
        let mut r = new_renderer();
        // A running tool must precede the result so the renderer captures
        // its name/humanized form.
        let _ = r.render(&AgentEvent::ToolStart {
            name: "grep".into(),
            id: "1".into(),
            humanized: "grep foo".into(),
        });
        let actions = r.render(&AgentEvent::ToolResult {
            name: "grep".into(),
            id: "1".into(),
            output: "line one\nline two\nline three".into(),
            success: true,
            elapsed_ms: 12,
            sources: vec![],
        });

        let joined = line_text(&actions);
        assert!(joined.contains("line one"), "preview should include output");
        assert!(joined.contains("line three"));
        assert!(
            !joined.contains("more lines"),
            "only 3 lines, no truncation footer"
        );
    }

    #[test]
    fn expanded_tool_output_keeps_every_line() {
        let mut r = new_renderer();
        let _ = r.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "exec ls".into(),
        });
        let many: String = (0..20).map(|i| format!("row {i}\n")).collect();
        let actions = r.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            output: many.trim_end().to_string(),
            success: true,
            elapsed_ms: 5,
            sources: vec![],
        });
        let joined = line_text(&actions);
        for index in 0..20 {
            assert!(
                joined.contains(&format!("row {index}\n"))
                    || joined.ends_with(&format!("row {index}"))
            );
        }
        assert!(!joined.contains("more lines"));
    }

    #[test]
    fn multiline_descriptions_survive_ticks_progress_and_completion() {
        for expanded in [false, true] {
            let mut renderer = new_renderer();
            renderer.set_expand_tools(expanded);
            renderer.set_width(12);
            let description = "Run checks\nfirst command with long arguments\n\n  second command";
            let start = renderer.render(&AgentEvent::ToolStart {
                name: "exec".into(),
                id: "1".into(),
                humanized: description.into(),
            });
            renderer.record_actions(&start);
            for event in [
                None,
                Some(AgentEvent::ToolProgress {
                    id: "1".into(),
                    line: "progress".into(),
                }),
                Some(AgentEvent::ToolResult {
                    name: "exec".into(),
                    id: "1".into(),
                    success: true,
                    elapsed_ms: 12,
                    output: String::new(),
                    sources: vec![],
                }),
            ] {
                let actions = match event {
                    Some(event) => renderer.render(&event),
                    None => renderer.tick_tool(),
                };
                renderer.record_actions(&actions);
                let lines = renderer.lines();
                assert_eq!(lines[0].to_string(), "");
                assert_eq!(lines[2].to_string(), "    Run checks");
                assert_eq!(
                    lines[3].to_string(),
                    "    first command with long arguments"
                );
                assert_eq!(lines[4].to_string(), "    ");
                assert_eq!(lines[5].to_string(), "      second command");
                let has_progress = expanded && !renderer.tool_progress_lines.is_empty();
                assert_eq!(lines.len(), 6 + usize::from(has_progress));
            }
        }
    }

    #[test]
    fn adjacent_tools_keep_a_blank_line_between_blocks() {
        let mut renderer = new_renderer();
        for name in ["read", "exec"] {
            let start = renderer.render(&AgentEvent::ToolStart {
                name: name.into(),
                id: name.into(),
                humanized: "Example action".into(),
            });
            renderer.record_actions(&start);
            let result = renderer.render(&AgentEvent::ToolResult {
                name: name.into(),
                id: name.into(),
                success: true,
                elapsed_ms: 1,
                output: String::new(),
                sources: vec![],
            });
            renderer.record_actions(&result);
        }
        assert_eq!(renderer.lines().len(), 6);
        assert!(renderer.lines()[1].to_string().contains("read"));
        assert_eq!(renderer.lines()[3].to_string(), "");
        assert!(renderer.lines()[4].to_string().contains("exec"));
    }

    #[test]
    fn edit_diffs_survive_the_completed_tool_replacement() {
        use flashmind_types::tool::DiffLine;

        let mut renderer = new_renderer();
        renderer.set_expand_tools(false);
        let start = renderer.render(&AgentEvent::ToolStart {
            name: "edit".into(),
            id: "1".into(),
            humanized: "Editing example.rs".into(),
        });
        renderer.record_actions(&start);
        let diff = renderer.render(&AgentEvent::FileDiff {
            path: "example.rs".into(),
            diff: vec![DiffLine::Added {
                content: "new content".into(),
                line: 1,
            }],
        });
        assert!(line_text(&diff).contains("+new content"));
        renderer.record_actions(&diff);
        let tick = renderer.tick_tool();
        renderer.record_actions(&tick);
        let result = renderer.render(&AgentEvent::ToolResult {
            name: "edit".into(),
            id: "1".into(),
            success: true,
            elapsed_ms: 1,
            output: "OK".into(),
            sources: vec![],
        });
        renderer.record_actions(&result);
        let lines: Vec<String> = renderer.lines().iter().map(Line::to_string).collect();
        assert_eq!(lines.len(), 6);
        assert!(lines[1].contains("edit"));
        assert_eq!(lines[2], "    Editing example.rs");
        assert_eq!(lines[4], "  example.rs");
        assert_eq!(lines[5], "    +new content");
    }

    #[test]
    fn interrupted_edit_keeps_its_diff_when_tracking_is_cleared() {
        use flashmind_types::tool::DiffLine;

        let mut renderer = new_renderer();
        let start = renderer.render(&AgentEvent::ToolStart {
            name: "edit".into(),
            id: "1".into(),
            humanized: "Editing example.rs".into(),
        });
        renderer.record_actions(&start);
        let diff = renderer.render(&AgentEvent::FileDiff {
            path: "example.rs".into(),
            diff: vec![DiffLine::Added {
                content: "new content".into(),
                line: 1,
            }],
        });
        renderer.record_actions(&diff);
        renderer.clear_running_tool();
        let done = renderer.render(&AgentEvent::Done(String::new()));
        renderer.record_actions(&done);
        assert!(
            renderer
                .lines()
                .iter()
                .any(|line| line.to_string() == "    +new content")
        );
    }

    #[test]
    fn silent_tool_success_omits_output_preview() {
        let mut r = new_renderer();
        let _ = r.render(&AgentEvent::ToolStart {
            name: "str_replace".into(),
            id: "1".into(),
            humanized: "str_replace foo.rs".into(),
        });
        let actions = r.render(&AgentEvent::ToolResult {
            name: "str_replace".into(),
            id: "1".into(),
            output: "edited".into(),
            success: true,
            elapsed_ms: 3,
            sources: vec![],
        });
        let joined = line_text(&actions);
        assert!(
            !joined.contains("edited"),
            "silent tools show diffs, not output"
        );
    }

    #[test]
    fn clear_running_tool_disarms_phantom_spinner_tick() {
        let mut r = new_renderer();
        // A tool starts (e.g. propose_choice) but its result is resolved
        // out-of-band (an interactive interrupt) and no ToolResult is emitted.
        let _ = r.render(&AgentEvent::ToolStart {
            name: "propose_choice".into(),
            id: "1".into(),
            humanized: "Propose \"pick\" (2 options)".into(),
        });

        // Without clearing, the tick stays armed and would re-print the running
        // line on every tick — the corruption seen in the bug report.
        assert!(r.tool_running(), "tool start arms the spinner tick");
        assert!(
            !r.tick_tool().is_empty(),
            "armed tick re-renders the running-tool line"
        );

        // Clearing the stale state (done at the start of the next LLM stream)
        // disarms the tick so it can never re-print the phantom line.
        r.clear_running_tool();
        assert!(
            !r.tool_running(),
            "clear_running_tool resets the running flag"
        );
        assert!(
            r.tick_tool().is_empty(),
            "a cleared tick produces no output"
        );
    }

    #[test]
    fn tool_progress_shows_truncation_indicator() {
        let mut r = new_renderer();
        let _ = r.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "exec build".into(),
        });
        // Emit more progress lines than MAX_PROGRESS (10) so older ones drop.
        let mut actions = Vec::new();
        for i in 0..13 {
            actions = r.render(&AgentEvent::ToolProgress {
                id: "1".into(),
                line: format!("step {i}"),
            });
        }
        let joined = line_text(&actions);
        assert!(
            joined.contains("3 more lines"),
            "should indicate dropped progress lines, got: {joined}"
        );
        // The most recent lines survive; the oldest are dropped.
        assert!(joined.contains("step 12"), "newest line shown");
        assert!(!joined.contains("step 0"), "oldest line dropped");
    }

    #[test]
    fn whitespace_only_reasoning_emits_nothing() {
        let mut r = new_renderer();
        r.set_expand_reasoning(false);
        let _ = r.render(&AgentEvent::ReasoningDelta("   \n\n  \n".into()));
        let actions = r.render(&AgentEvent::TextDelta("answer".into()));
        let joined = line_text(&actions);
        assert!(
            !joined.contains("thinking"),
            "whitespace-only reasoning should not emit a summary, got: {joined}"
        );
    }

    #[test]
    fn truncate_display_respects_max_width() {
        // Fits: returned unchanged.
        assert_eq!(truncate_display("abc", 5), "abc");
        // max_width 0: empty.
        assert_eq!(truncate_display("abc", 0), "");
        // Ellipsis never overflows the budget.
        let out = truncate_display("abcdefgh", 4);
        assert_eq!(UnicodeWidthStr::width(out.as_str()), 4);
        assert!(out.ends_with('\u{2026}'));
        // Wide (CJK) chars: each is 2 cols, so width stays within bounds.
        let wide = truncate_display("日本語テスト", 5);
        assert!(UnicodeWidthStr::width(wide.as_str()) <= 5, "got {wide}");
        assert!(wide.ends_with('\u{2026}'));
    }

    #[test]
    fn tool_lines_use_glyph_vocabulary() {
        let mut r = new_renderer();
        let start = r.render(&AgentEvent::ToolStart {
            name: "grep".into(),
            id: "1".into(),
            humanized: "grep foo".into(),
        });
        assert!(
            line_text(&start).contains('\u{25cc}'),
            "running tool line should use the ◌ glyph"
        );

        let done = r.render(&AgentEvent::ToolResult {
            name: "grep".into(),
            id: "1".into(),
            output: String::new(),
            success: true,
            elapsed_ms: 12,
            sources: vec![],
        });
        let joined = line_text(&done);
        assert!(joined.contains('\u{2713}'), "success result should use ✓");

        let fail = r.render(&AgentEvent::ToolResult {
            name: "grep".into(),
            id: "1".into(),
            output: "boom".into(),
            success: false,
            elapsed_ms: 3,
            sources: vec![],
        });
        assert!(
            line_text(&fail).contains('\u{2717}'),
            "failure result should use ✗"
        );
    }

    #[test]
    fn file_diff_header_uses_accent_style() {
        use flashmind_types::tool::DiffLine;
        let mut r = new_renderer();
        let actions = r.render(&AgentEvent::FileDiff {
            path: "src/lib.rs".into(),
            diff: vec![DiffLine::Added {
                content: "let x = 1;".into(),
                line: 1,
            }],
        });
        let header = actions.iter().find_map(|a| match a {
            RenderAction::Append(line) => line
                .spans
                .iter()
                .find(|s| s.content.contains("src/lib.rs"))
                .map(|s| s.style),
            _ => None,
        });
        assert_eq!(
            header.and_then(|s| s.fg),
            Some(ratatui::style::Color::Rgb(45, 191, 179)),
            "diff header should use the teal accent"
        );
    }

    /// Concatenate the text content of all lines produced by a render pass.
    fn line_text(actions: &[RenderAction]) -> String {
        let mut out = Vec::new();
        for a in actions {
            match a {
                RenderAction::Append(line) => out.push(line_text_of(line)),
                RenderAction::ReplaceTool { lines, .. } => {
                    for l in lines {
                        out.push(line_text_of(l));
                    }
                }
            }
        }
        out.join("\n")
    }

    fn line_text_of(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }
}
