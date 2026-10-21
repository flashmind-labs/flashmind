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
//! Most events produce `Append` actions. Tool progress produces `ReplaceTool`,
//! which redraws the running-tool lines in place, and `ToolResult` produces
//! `FinishTool`, which replaces them with a block carrying both a collapsed
//! and an expanded form so the transcript can switch between them.
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
//! | `Usage` | Nothing (the REPL keeps usage for its footer) |
//! | `Error` | Bold red "error: ..." prefix |
//! | `Done` | Flushes buffered text, adds usage footer + blank separator |
//! | `Cancelled` | Flushes buffered text and marks the turn interrupted |
//! | `SpawnedEvent` | Prefixes inner event output with `[task_name]` in magenta |

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use chrono::Local;
use flashmind_types::AgentEvent;
use flashmind_types::llm::{ModelPricing, TokenUsage};
use flashmind_types::tool::DiffLine;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use rust_decimal::Decimal;
use unicode_width::UnicodeWidthStr;

use crate::styles::*;
use crate::transcript::text_width;

const SILENT_TOOLS: &[&str] = &["str_replace", "file_write"];

/// Lines of a failed tool's output shown while tool output is collapsed.
const FAILURE_PREVIEW: usize = 5;
/// Description lines a collapsed tool shows, header included.
const COLLAPSED_DESCRIPTION: usize = 3;
/// Trailing output lines an expanded finished tool shows.
const EXPANDED_OUTPUT_TAIL: usize = 20;
const COMMAND_PREFIX: &str = "    ";
const COMMAND_OUTPUT_PREFIX: &str = "       ";

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
    /// Replace the running tool lines with a finished tool block that has a
    /// collapsed and an expanded form, so the transcript can switch between
    /// them later.
    FinishTool {
        erase_count: usize,
        output: ToolOutput,
    },
    /// Append a retained reasoning block that can be expanded later.
    Reasoning {
        output: ReasoningOutput,
        expanded: bool,
    },
}

/// A reasoning trace in both display forms.
#[derive(Debug, Clone)]
pub struct ReasoningOutput {
    /// A header summarizing the trace.
    pub collapsed: Vec<Line<'static>>,
    /// The header followed by the full trace.
    pub expanded: Vec<Line<'static>>,
}

impl ReasoningOutput {
    /// The lines for the given display mode.
    pub fn lines(&self, expanded: bool) -> &[Line<'static>] {
        if expanded {
            &self.expanded
        } else {
            &self.collapsed
        }
    }
}

/// A finished tool block in both display forms.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// Header, description, diffs and a hint in place of the output.
    pub collapsed: Vec<Line<'static>>,
    /// Header, description, diffs and every output line.
    pub expanded: Vec<Line<'static>>,
}

impl ToolOutput {
    /// The lines for the given display mode.
    pub fn lines(&self, expanded: bool) -> &[Line<'static>] {
        if expanded {
            &self.expanded
        } else {
            &self.collapsed
        }
    }
}

#[derive(Debug)]
struct RunningTool {
    id: String,
    name: String,
    description: String,
    diffs: Vec<Line<'static>>,
    /// When the tool started (client-side elapsed tracking).
    started: Option<Instant>,
    /// Number of lines last emitted for this tool, erased on the next update.
    line_count: usize,
    /// Streaming progress lines from the tool.
    progress: Vec<String>,
    completion: Option<ToolCompletion>,
}

impl RunningTool {
    fn new(id: &str, name: &str, description: &str) -> Self {
        Self {
            id: id.to_owned(),
            name: name.to_owned(),
            description: description.to_owned(),
            diffs: Vec::new(),
            started: Some(Instant::now()),
            line_count: 0,
            progress: Vec::new(),
            completion: None,
        }
    }
}

#[derive(Debug)]
struct ToolCompletion {
    name: String,
    output: String,
    success: bool,
    elapsed_ms: u64,
}

impl ToolCompletion {
    fn new(name: &str, output: &str, success: bool, elapsed_ms: u64) -> Self {
        Self {
            name: name.to_owned(),
            output: output.to_owned(),
            success,
            elapsed_ms,
        }
    }
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
    /// Tools in display order; the front owns the live preview.
    tools: VecDeque<RunningTool>,
    /// When the current turn started.
    turn_start: Option<Instant>,
    /// Whether the live reasoning preview shows the full trace.
    expand_reasoning: bool,
    /// Terminal width for right-aligned elapsed times.
    width: usize,
    /// Accumulated output lines for resize redraw.
    lines: Vec<Line<'static>>,
    /// Whether tool output is shown (true) or collapsed to a hint (false).
    expand_tools: bool,
    /// When true, `SpawnedEvent` rendering is suppressed (handled by the
    /// `SubagentProgress` widget instead).
    suppress_spawned: bool,
    /// Whether the current reply already has its `●` marker.
    reply_open: bool,
    /// Whether the last block in the transcript is a tool line.
    in_tool_run: bool,
    /// Whether the last reply line was blank.
    trailing_blank: bool,
    /// Tokens and cost of every request in the current turn.
    turn_usage: TurnTotals,
    /// Whether the turn footer shows tokens and cost.
    show_usage: bool,
}

/// Token counts and cost summed over the requests of one turn.
#[derive(Debug, Default)]
struct TurnTotals {
    usage: TokenUsage,
    /// Cost as billed, with cached tokens at the cache rates.
    cost: Option<Decimal>,
    /// Cost if every prompt token were billed at the full input rate.
    uncached_cost: Option<Decimal>,
}

impl TurnTotals {
    fn add(&mut self, usage: &TokenUsage, pricing: Option<&ModelPricing>) {
        self.usage.prompt_tokens += usage.prompt_tokens;
        self.usage.completion_tokens += usage.completion_tokens;
        self.usage.total_tokens += usage.total_tokens;
        self.usage.cache_read_tokens += usage.cache_read_tokens;
        self.usage.cache_creation_tokens += usage.cache_creation_tokens;
        let Some(pricing) = pricing else {
            return;
        };
        let uncached = TokenUsage {
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            ..usage.clone()
        };
        if let (Some(cost), Some(uncached)) = (usage.cost(pricing), uncached.cost(pricing)) {
            *self.cost.get_or_insert_default() += cost;
            *self.uncached_cost.get_or_insert_default() += uncached;
        }
    }
}

impl EventRenderer {
    /// Create a new, empty renderer.
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            in_reasoning: false,
            expand_reasoning: false,
            tools: VecDeque::new(),
            turn_start: None,
            width: 80,
            lines: Vec::new(),
            expand_tools: false,
            suppress_spawned: false,
            reply_open: false,
            in_tool_run: false,
            trailing_blank: false,
            turn_usage: TurnTotals::default(),
            show_usage: true,
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
        self.turn_usage = TurnTotals::default();
        self.reply_open = false;
        self.in_tool_run = false;
    }

    /// Add one request's usage to the turn summary shown when the turn ends.
    pub fn add_usage(&mut self, usage: &TokenUsage, pricing: Option<&ModelPricing>) {
        self.turn_usage.add(usage, pricing);
    }

    /// Tokens and cost of the turn so far, for the live activity line.
    pub fn turn_usage_parts(&self) -> Vec<String> {
        if self.show_usage {
            usage_parts(&self.turn_usage)
        } else {
            Vec::new()
        }
    }

    /// Set whether the turn footer shows tokens and cost.
    pub fn set_show_usage(&mut self, show: bool) {
        self.show_usage = show;
    }

    /// Time since the current turn started.
    pub fn turn_elapsed(&self) -> Option<Duration> {
        self.turn_start.map(|start| start.elapsed())
    }

    /// Whether the reply being streamed already has its `●` marker.
    pub fn reply_open(&self) -> bool {
        self.reply_open
    }

    /// Whether a tool is currently running (for live elapsed tick updates).
    pub fn tool_running(&self) -> bool {
        !self.tools.is_empty()
    }

    /// Discard tool tracking without changing the transcript. Call before
    /// unrelated output if a tool was resolved without a matching result,
    /// so its next tick cannot replace lines belonging to another block.
    pub fn clear_running_tool(&mut self) {
        self.tools.clear();
    }

    fn show_running_tool(&mut self) -> Vec<RenderAction> {
        let Some(tool) = self.tools.front_mut() else {
            return Vec::new();
        };
        if SILENT_TOOLS.contains(&tool.name.as_str()) {
            return Vec::new();
        }
        let lines = tool_block(tool, "\u{25cc}", "", S_TOOL_RUN, self.expand_tools);
        tool.line_count = lines.len();
        let mut actions = self.start_tool_run();
        actions.extend(lines.into_iter().map(RenderAction::Append));
        actions
    }

    fn start_tool(&mut self, id: &str, name: &str, description: &str) -> Vec<RenderAction> {
        self.tools
            .push_back(RunningTool::new(id, name, description));
        if self.tools.len() > 1 {
            return Vec::new();
        }
        let mut actions = self.end_reasoning();
        actions.extend(self.flush());
        actions.extend(self.show_running_tool());
        actions
    }

    fn tool_progress(&mut self, id: &str, line: &str) -> Vec<RenderAction> {
        if let Some(tool) = self.tools.iter_mut().find(|tool| tool.id == id) {
            tool.progress.push(line.to_owned());
        }
        if self.tools.front().is_some_and(|tool| tool.id == id) {
            self.render_tool_with_progress()
        } else {
            Vec::new()
        }
    }

    fn finish_tool(&mut self, id: &str, completion: ToolCompletion) -> Vec<RenderAction> {
        if let Some(tool) = self.tools.iter_mut().find(|tool| tool.id == id) {
            tool.completion = Some(completion);
        } else {
            let mut tool = RunningTool::new(id, &completion.name, "");
            if self.tools.is_empty() {
                tool.started = None;
            }
            tool.completion = Some(completion);
            self.tools.push_back(tool);
        }

        let mut actions = Vec::new();
        while let Some(mut tool) = self.tools.pop_front() {
            let Some(completion) = tool.completion.take() else {
                self.tools.push_front(tool);
                break;
            };
            if !(completion.success
                && SILENT_TOOLS.contains(&completion.name.as_str())
                && tool.line_count == 0)
            {
                if tool.line_count == 0 {
                    actions.extend(self.start_tool_run());
                }
                actions.push(RenderAction::FinishTool {
                    erase_count: tool.line_count,
                    output: finished_tool_output(tool, completion),
                });
            }
            actions.extend(self.show_running_tool());
            if self
                .tools
                .front()
                .is_some_and(|tool| tool.completion.is_none())
            {
                actions.extend(self.render_tool_with_progress());
            }
        }
        actions
    }

    /// Render the current running tool line with live elapsed for tick updates.
    /// Returns a `ReplaceTool` action if a tool is running, empty otherwise.
    pub fn tick_tool(&mut self) -> Vec<RenderAction> {
        self.render_tool_with_progress()
    }

    /// Re-render the running tool in place of the lines it last emitted.
    fn render_tool_with_progress(&mut self) -> Vec<RenderAction> {
        let expand_tools = self.expand_tools;
        let Some(tool) = self.tools.front_mut() else {
            return Vec::new();
        };
        if SILENT_TOOLS.contains(&tool.name.as_str()) {
            return Vec::new();
        }
        let elapsed = format_elapsed(
            tool.started
                .map_or(0, |start| start.elapsed().as_millis() as u64),
        );
        let mut lines = tool_block(tool, "\u{25cc}", &elapsed, S_TOOL_RUN, expand_tools);

        if expand_tools {
            let skip = tool.progress.len().saturating_sub(10);
            if skip > 0 {
                lines.push(tool_body(&format!("... {skip} more lines"), S_DIM));
            }
            lines.extend(
                tool.progress
                    .iter()
                    .skip(skip)
                    .map(|line| tool_body(line, S_DIM)),
            );
        } else if tool.name == "exec" {
            lines.extend(command_tail(
                tool.progress.iter().flat_map(|line| line.lines()),
                S_DIM,
            ));
        } else if !tool.progress.is_empty() {
            lines.push(tool_body(&expand_hint(tool.progress.len(), false), S_HINT));
        }
        lines.extend(tool.diffs.iter().cloned());

        let erase_count = std::mem::replace(&mut tool.line_count, lines.len());
        vec![RenderAction::ReplaceTool { erase_count, lines }]
    }

    /// Render an agent event into render actions.
    pub fn render(&mut self, event: &AgentEvent) -> Vec<RenderAction> {
        match event {
            AgentEvent::TextDelta(text) => {
                let mut actions = self.end_reasoning();
                self.text_buffer.push_str(text);
                actions.extend(self.flush_complete());
                actions
            }
            AgentEvent::ReasoningDelta(text) => {
                self.reasoning_buffer.push_str(text);
                if self.in_reasoning {
                    return Vec::new();
                }
                self.in_reasoning = true;
                self.reply_open = false;
                self.in_tool_run = false;
                vec![RenderAction::Append(Line::default())]
            }
            AgentEvent::ToolStart {
                id,
                name,
                humanized,
            } => self.start_tool(id, name, humanized),
            AgentEvent::ToolProgress { id, line } => self.tool_progress(id, line),
            AgentEvent::ToolResult {
                id,
                name,
                success,
                elapsed_ms,
                output,
                ..
            } => self.finish_tool(id, ToolCompletion::new(name, output, *success, *elapsed_ms)),
            AgentEvent::Interrupted {
                tool_call_id,
                tool_name,
                output,
                ..
            } => self.finish_tool(
                tool_call_id,
                ToolCompletion::new(tool_name, output, false, 0),
            ),
            AgentEvent::FileDiff { path, diff } => self.file_diff(path, diff),
            AgentEvent::Status(msg) => vec![RenderAction::Append(Line::from(Span::styled(
                format!("[{msg}]"),
                S_DIM,
            )))],
            AgentEvent::Error(msg) => {
                let mut actions = self.flush();
                actions.extend(self.end_reasoning());
                actions.push(RenderAction::Append(Line::from(Span::styled(
                    format!("[error] {msg}"),
                    S_ERROR,
                ))));
                actions
            }
            AgentEvent::Compacted(summary) => render_text_lines(summary, self.width)
                .into_iter()
                .map(RenderAction::Append)
                .collect(),
            AgentEvent::Done(_) => self.end_turn(false),
            AgentEvent::Cancelled => self.end_turn(true),
            AgentEvent::SpawnedEvent { task, event, .. } if !self.suppress_spawned => {
                let mut actions = self.render(event);
                for action in &mut actions {
                    let prefix = |line: &mut Line<'static>| {
                        line.spans
                            .insert(0, Span::styled(format!("  [{task}] "), S_SPAWNED));
                    };
                    match action {
                        RenderAction::Append(line) => prefix(line),
                        RenderAction::Reasoning { output, .. } => {
                            output
                                .collapsed
                                .iter_mut()
                                .chain(&mut output.expanded)
                                .for_each(prefix);
                        }
                        _ => {}
                    }
                }
                actions
            }
            _ => Vec::new(),
        }
    }

    fn file_diff(&mut self, path: &str, diff: &[DiffLine]) -> Vec<RenderAction> {
        let lines = diff_lines(path, diff);
        if let Some(tool) = self.tools.front_mut()
            && !SILENT_TOOLS.contains(&tool.name.as_str())
        {
            tool.diffs.extend(lines);
            self.render_tool_with_progress()
        } else {
            let mut actions = self.start_tool_run();
            actions.extend(lines.into_iter().map(RenderAction::Append));
            actions
        }
    }

    fn end_turn(&mut self, cancelled: bool) -> Vec<RenderAction> {
        let mut actions = Vec::new();
        if cancelled
            && let Some(tool) = self.tools.pop_front()
            && !SILENT_TOOLS.contains(&tool.name.as_str())
        {
            let mut collapsed = tool_block(&tool, "-", "Interrupted", S_DIM, false);
            let mut expanded = tool_block(&tool, "-", "Interrupted", S_DIM, true);
            expanded.extend(tool.progress.iter().map(|line| tool_body(line, S_DIM)));
            expanded.extend(tool.diffs.iter().cloned());
            if tool.name == "exec" {
                collapsed.extend(command_tail(
                    tool.progress.iter().flat_map(|line| line.lines()),
                    S_DIM,
                ));
            }
            collapsed.extend(tool.diffs);
            actions.push(RenderAction::FinishTool {
                erase_count: tool.line_count,
                output: ToolOutput {
                    collapsed,
                    expanded,
                },
            });
        }
        self.clear_running_tool();
        actions.extend(self.end_reasoning());
        actions.extend(self.flush());
        if let Some(start) = self.turn_start.take() {
            if !(self.reply_open && self.trailing_blank) {
                actions.push(RenderAction::Append(Line::default()));
            }
            let empty = TurnTotals::default();
            let totals = if self.show_usage {
                &self.turn_usage
            } else {
                &empty
            };
            actions.push(RenderAction::Append(turn_footer(
                start.elapsed().as_millis() as u64,
                totals,
                cancelled,
            )));
        } else if cancelled {
            actions.push(RenderAction::Append(Line::from(Span::styled(
                "Interrupted",
                S_DIM,
            ))));
        }
        self.turn_usage = TurnTotals::default();
        self.reply_open = false;
        self.in_tool_run = false;
        actions
    }

    /// Retain the full trace when its streaming preview ends.
    fn end_reasoning(&mut self) -> Vec<RenderAction> {
        if !std::mem::take(&mut self.in_reasoning) {
            return Vec::new();
        }
        self.expand_reasoning = false;
        let text = std::mem::take(&mut self.reasoning_buffer);
        let lines = text.lines().filter(|line| !line.trim().is_empty()).count();
        if lines == 0 {
            return Vec::new();
        }
        let header = |marker| {
            Line::from(Span::styled(
                format!("{marker} thinking ({lines} lines, click to toggle)"),
                S_DIM,
            ))
        };
        let mut expanded = vec![header("v")];
        expanded.extend(dim_lines(render_text_lines(&text, self.width)));
        vec![
            RenderAction::Reasoning {
                output: ReasoningOutput {
                    collapsed: vec![header(">")],
                    expanded,
                },
                expanded: false,
            },
            RenderAction::Append(Line::default()),
        ]
    }

    /// Set whether the live preview shows the full trace or its latest lines.
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
        let width = self.reply_width();
        let lines = render_text_incremental(&mut self.text_buffer, width);
        self.reply_actions(lines)
    }

    /// Text width inside the reply indent.
    fn reply_width(&self) -> usize {
        let width = text_width(self.width.min(u16::MAX as usize) as u16);
        (width as usize).saturating_sub(REPLY_INDENT).max(1)
    }

    /// Indent reply lines, opening the reply with a blank line and `●`.
    fn reply_actions(&mut self, lines: Vec<Line<'static>>) -> Vec<RenderAction> {
        let lines = reply_lines(lines, self.reply_open);
        if lines.is_empty() {
            return Vec::new();
        }
        self.reply_open = true;
        self.in_tool_run = false;
        self.trailing_blank = lines.last().is_some_and(|line| line.width() == 0);
        lines.into_iter().map(RenderAction::Append).collect()
    }

    /// A blank line before the first tool after other output.
    fn start_tool_run(&mut self) -> Vec<RenderAction> {
        self.reply_open = false;
        if std::mem::replace(&mut self.in_tool_run, true) {
            Vec::new()
        } else {
            vec![RenderAction::Append(Line::default())]
        }
    }

    /// Drain any remaining buffered text into styled lines.
    pub fn flush(&mut self) -> Vec<RenderAction> {
        if self.text_buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.text_buffer);
        let lines = render_text_lines(&text, self.reply_width());
        self.reply_actions(lines)
    }

    /// Markdown still being streamed, including complete lines in an open block.
    pub fn pending_text(&self) -> &str {
        &self.text_buffer
    }

    /// Uncommitted reasoning for the live preview.
    pub fn pending_reasoning(&self) -> &str {
        &self.reasoning_buffer
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
        apply_to_lines(&mut self.lines, actions, self.expand_tools);
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

    pub(crate) fn clear_transcript(&mut self) {
        self.lines.clear();
        self.text_buffer.clear();
        self.reasoning_buffer.clear();
        self.tools.clear();
        self.in_reasoning = false;
        self.reply_open = false;
        self.in_tool_run = false;
        self.trailing_blank = false;
    }
}

impl Default for EventRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers

/// Apply render actions to a line buffer: append lines, or erase the running
/// tool's lines and put the replacement in their place.
pub(crate) fn apply_to_lines(
    lines: &mut Vec<Line<'static>>,
    actions: &[RenderAction],
    expand_tools: bool,
) {
    for action in actions {
        match action {
            RenderAction::Append(line) => lines.push(line.clone()),
            RenderAction::ReplaceTool {
                erase_count,
                lines: replacement,
            } => {
                lines.truncate(lines.len().saturating_sub(*erase_count));
                lines.extend(replacement.iter().cloned());
            }
            RenderAction::FinishTool {
                erase_count,
                output,
            } => {
                lines.truncate(lines.len().saturating_sub(*erase_count));
                lines.extend(output.lines(expand_tools).iter().cloned());
            }
            RenderAction::Reasoning { output, expanded } => {
                lines.extend(output.lines(*expanded).iter().cloned());
            }
        }
    }
}

fn finished_tool_output(tool: RunningTool, result: ToolCompletion) -> ToolOutput {
    let ms = if result.elapsed_ms > 0 {
        result.elapsed_ms
    } else {
        tool.started
            .map_or(0, |start| start.elapsed().as_millis() as u64)
    };
    let elapsed = format_elapsed(ms);
    let (icon, style) = if result.success {
        ("\u{2713}", S_TOOL_OK)
    } else {
        ("\u{2717}", S_TOOL_FAIL)
    };
    let mut collapsed = tool_block(&tool, icon, &elapsed, style, false);
    let mut expanded = tool_block(&tool, icon, &elapsed, style, true);
    let silent = SILENT_TOOLS.contains(&result.name.as_str()) || !tool.diffs.is_empty();
    if !result.output.is_empty() && (!result.success || !silent) {
        let style = if result.success { S_DIM } else { S_TOOL_FAIL };
        let all: Vec<_> = result.output.lines().collect();
        let skip = all.len().saturating_sub(EXPANDED_OUTPUT_TAIL);
        if skip > 0 {
            let noun = if skip == 1 { "line" } else { "lines" };
            expanded.push(tool_body(&format!("... {skip} earlier {noun}"), S_DIM));
        }
        expanded.extend(all[skip..].iter().map(|line| tool_body(line, style)));
        if tool.name == "exec" {
            collapsed.extend(command_tail(result.output.lines(), style));
        } else if !result.success {
            collapsed.extend(
                all.iter()
                    .take(FAILURE_PREVIEW)
                    .map(|line| tool_body(line, style)),
            );
            let hidden = all.len().saturating_sub(FAILURE_PREVIEW);
            if hidden > 0 {
                collapsed.push(tool_body(&expand_hint(hidden, true), S_TOOL_FAIL));
            }
        }
    }
    expanded.extend(tool.diffs.iter().cloned());
    collapsed.extend(tool.diffs);
    ToolOutput {
        collapsed,
        expanded,
    }
}

fn diff_lines(path: &str, diff: &[DiffLine]) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(format!("    {path}"), S_DIFF_ADD))];
    let render_line = |line: &DiffLine| match line {
        DiffLine::Added { content, .. } => {
            Line::from(format!("    +{content}")).style(S_DIFF_BLOCK_OK)
        }
        DiffLine::Removed { content, .. } => {
            Line::from(format!("    -{content}")).style(S_DIFF_BLOCK_DEL)
        }
        DiffLine::Context { content, line } if content.is_empty() && *line == 0 => {
            Line::from(Span::styled("    \u{2026}", S_DIM))
        }
        DiffLine::Context { content, .. } => {
            Line::from(Span::styled(format!("     {content}"), S_DIFF_CONTEXT))
        }
    };
    if diff.len() > 100 {
        lines.extend(diff[..50].iter().map(render_line));
        lines.push(
            Line::from(format!("    ... {} lines omitted ...", diff.len() - 100))
                .style(S_DIFF_BLOCK_OK),
        );
        lines.extend(diff[diff.len() - 50..].iter().map(render_line));
    } else {
        lines.extend(diff.iter().map(render_line));
    }
    lines
}

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
// The header is the first description line, or the tool name when there is
// no description. Collapsed descriptions and commands show their first
// COLLAPSED_DESCRIPTION lines with a hint. Expanded commands retain their full text.
fn tool_block(
    tool: &RunningTool,
    icon: &str,
    elapsed: &str,
    style: Style,
    expanded: bool,
) -> Vec<Line<'static>> {
    if tool.name == "exec" {
        let marker = if icon == "\u{2717}" { icon } else { "\u{2022}" };
        let command = tool
            .description
            .strip_prefix("Running `")
            .and_then(|text| text.strip_suffix('`'))
            .unwrap_or(&tool.description);
        let mut description = command.lines();
        let first = description
            .next()
            .filter(|line| !line.is_empty())
            .unwrap_or(&tool.name);
        if !expanded {
            let mut lines = vec![tool_header(marker, first, elapsed, style)];
            lines.extend(
                description
                    .by_ref()
                    .take(COLLAPSED_DESCRIPTION - 1)
                    .map(|line| {
                        Line::from(vec![
                            Span::raw(COMMAND_PREFIX),
                            Span::styled(line.to_owned(), S_COMMAND),
                        ])
                    }),
            );
            let hidden = description.count();
            if hidden > 0 {
                lines.push(tool_body(&expand_hint(hidden, true), S_HINT));
            }
            return lines;
        }
        let mut lines = vec![Line::from(vec![
            Span::styled(format!("  {marker} "), style),
            Span::styled(first.to_owned(), S_COMMAND),
        ])];
        lines.extend(description.map(|line| tool_body(line, S_COMMAND)));
        if !elapsed.is_empty() {
            lines.push(tool_body(elapsed, S_ELAPSED));
        }
        return lines;
    }
    let mut description = tool.description.lines();
    let label = description
        .next()
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .unwrap_or(&tool.name);
    let mut lines = vec![tool_header(icon, label, elapsed, style)];
    let rest: Vec<&str> = description.collect();
    let shown = if expanded {
        rest.len()
    } else {
        COLLAPSED_DESCRIPTION - 1
    };
    lines.extend(
        rest.iter()
            .take(shown)
            .map(|line| tool_body(line, S_COMMAND)),
    );
    let hidden = rest.len().saturating_sub(shown);
    if hidden > 0 {
        lines.push(tool_body(&expand_hint(hidden, true), S_HINT));
    }
    lines
}

/// Columns taken by the `●` marker and the indent under it.
pub(crate) const REPLY_INDENT: usize = 2;

/// Prefix reply lines with the reply indent. Unless the reply is already
/// open, leading blank lines are dropped and the first line gets a blank line
/// above it and a `●` marker.
pub(crate) fn reply_lines(lines: Vec<Line<'static>>, open: bool) -> Vec<Line<'static>> {
    let mut lines = lines.into_iter();
    let mut out = Vec::new();
    if !open {
        let Some(mut first) = lines.by_ref().find(|line| line.width() > 0) else {
            return Vec::new();
        };
        first.spans.insert(0, Span::styled("\u{25cf} ", S_ACCENT));
        out.push(Line::default());
        out.push(first);
    }
    out.extend(lines.map(|mut line| {
        if line.width() > 0 {
            line.spans.insert(0, Span::raw(" ".repeat(REPLY_INDENT)));
        }
        line
    }));
    out
}

/// The dim line closing a turn: duration, clock time, tokens and cost.
fn turn_footer(elapsed_ms: u64, totals: &TurnTotals, cancelled: bool) -> Line<'static> {
    let outcome = if cancelled {
        "Interrupted after"
    } else {
        "Done in"
    };
    let mut parts = vec![
        format!("{outcome} {}", format_elapsed(elapsed_ms)),
        Local::now().format("%-I:%M %p").to_string(),
    ];
    parts.extend(usage_parts(totals));
    Line::from(vec![
        Span::styled("\u{273b} ", S_ACCENT),
        Span::styled(parts.join(" \u{b7} "), S_DIM),
    ])
}

/// Tokens in (and cached), tokens out, and cost with and without the cache.
fn usage_parts(totals: &TurnTotals) -> Vec<String> {
    let mut parts = Vec::new();
    let usage = &totals.usage;
    if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
        let mut sent = format!("{} in", format_tokens(usage.prompt_tokens));
        if usage.cache_read_tokens > 0 {
            sent.push_str(&format!(
                " ({} cached)",
                format_tokens(usage.cache_read_tokens)
            ));
        }
        parts.push(sent);
        parts.push(format!("{} out", format_tokens(usage.completion_tokens)));
    }
    if let (Some(cost), Some(uncached)) = (totals.cost, totals.uncached_cost) {
        let mut text = format_cost(cost);
        if uncached != cost {
            text.push_str(&format!(" ({} uncached)", format_cost(uncached)));
        }
        parts.push(text);
    }
    parts
}

fn format_tokens(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        format!("{:.2}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1000 {
        format!("{:.1}k", tokens as f64 / 1000.0)
    } else {
        tokens.to_string()
    }
}

fn format_cost(cost: Decimal) -> String {
    format!("${}", cost.round_dp(4).normalize())
}

/// Placeholder for output hidden while tool output is collapsed.
fn expand_hint(hidden: usize, more: bool) -> String {
    let noun = if hidden == 1 { "line" } else { "lines" };
    if more {
        format!("... {hidden} more {noun} (ctrl+o to expand)")
    } else {
        format!("{hidden} {noun} (ctrl+o to expand)")
    }
}

fn command_tail<'a>(
    lines: impl DoubleEndedIterator<Item = &'a str>,
    style: Style,
) -> Vec<Line<'static>> {
    let mut tail: Vec<_> = lines
        .rev()
        .take(5)
        .map(|line| {
            Line::from(vec![
                Span::raw(COMMAND_OUTPUT_PREFIX),
                Span::styled(line.to_owned(), style),
            ])
        })
        .collect();
    tail.reverse();
    tail
}

fn tool_body(text: &str, style: Style) -> Line<'static> {
    Line::from(Span::styled(format!("    {text}"), style))
}

fn tool_header(icon: &str, label: &str, elapsed: &str, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {icon} "), style),
        Span::styled(label.to_owned(), S_COMMAND),
        Span::styled(
            if elapsed.is_empty() {
                String::new()
            } else {
                format!("  {elapsed}")
            },
            S_ELAPSED,
        ),
    ])
}

/// Fit tool headers and compact command output to the current viewport.
pub(crate) fn fit_tool_line(line: &Line<'static>, width: usize) -> Line<'static> {
    if let [indent, output] = line.spans.as_slice()
        && matches!(
            indent.content.as_ref(),
            COMMAND_PREFIX | COMMAND_OUTPUT_PREFIX
        )
    {
        let prefix = truncate_display(&indent.content, width);
        let text = truncate_display(&output.content, width.saturating_sub(prefix.width()));
        return Line {
            spans: vec![
                Span::styled(prefix, indent.style),
                Span::styled(text, output.style),
            ],
            ..line.clone()
        };
    }
    let Some(marker_index) = line.spans.len().checked_sub(3) else {
        return line.clone();
    };
    if !matches!(
        line.spans[marker_index].content.as_ref(),
        "  \u{25cc} " | "  \u{2713} " | "  \u{2717} " | "  \u{2022} "
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
    } else if ms < 60_000 {
        format!("{:.1} s", ms as f64 / 1000.0)
    } else {
        format_secs(ms / 1000)
    }
}

/// Whole seconds as `12s`, `3m 15s` or `11h 59m 33s`.
pub(crate) fn format_secs(secs: u64) -> String {
    humantime::format_duration(Duration::from_secs(secs)).to_string()
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
mod tests;
