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
    started: Instant,
    /// Number of lines last emitted for this tool, erased on the next update.
    line_count: usize,
    /// Streaming progress lines from the tool.
    progress: Vec<String>,
    completion: Option<AgentEvent>,
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
    /// The current running tool, if any.
    tool_info: Option<RunningTool>,
    pending_tools: VecDeque<RunningTool>,
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
            tool_info: None,
            pending_tools: VecDeque::new(),
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
    pub fn turn_elapsed(&self) -> Option<std::time::Duration> {
        self.turn_start.map(|start| start.elapsed())
    }

    /// Whether the reply being streamed already has its `●` marker.
    pub fn reply_open(&self) -> bool {
        self.reply_open
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
        self.pending_tools.clear();
    }

    fn show_running_tool(&mut self, mut tool: RunningTool) -> Vec<RenderAction> {
        let mut actions = Vec::new();
        if !SILENT_TOOLS.contains(&tool.name.as_str()) {
            actions.extend(self.start_tool_run());
            let lines = tool_block(&tool, "\u{25cc}", "", S_TOOL_RUN, self.expand_tools);
            tool.line_count = lines.len();
            actions.extend(lines.into_iter().map(RenderAction::Append));
        }
        self.tool_info = Some(tool);
        actions
    }

    fn show_next_tool(&mut self) -> Vec<RenderAction> {
        let Some(mut tool) = self.pending_tools.pop_front() else {
            return Vec::new();
        };
        let completion = tool.completion.take();
        let mut actions = self.show_running_tool(tool);
        if let Some(event) = completion {
            actions.extend(self.render(&event));
        } else {
            actions.extend(self.render_tool_with_progress());
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
        let Some(tool) = self.tool_info.as_mut() else {
            return Vec::new();
        };
        if SILENT_TOOLS.contains(&tool.name.as_str()) {
            return Vec::new();
        }
        let elapsed = format_elapsed(tool.started.elapsed().as_millis() as u64);
        let mut lines = tool_block(tool, "\u{25cc}", &elapsed, S_TOOL_RUN, expand_tools);

        const MAX_PROGRESS: usize = 10;
        let limit = if expand_tools { MAX_PROGRESS } else { 0 };
        let skip = tool.progress.len().saturating_sub(limit);
        if skip > 0 && expand_tools {
            lines.push(tool_body(&format!("... {skip} more lines"), S_DIM));
        }
        for line in tool.progress.iter().skip(skip) {
            lines.push(tool_body(line, S_DIM));
        }
        if !expand_tools && !tool.progress.is_empty() {
            lines.push(tool_body(&expand_hint(tool.progress.len(), false), S_HINT));
        }
        lines.extend(tool.diffs.iter().cloned());

        // A running tool always emitted its block (at least two lines) when
        // it started, so there is always something to replace.
        let erase_count = std::mem::replace(&mut tool.line_count, lines.len());
        vec![RenderAction::ReplaceTool { erase_count, lines }]
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
                    self.reply_open = false;
                    self.in_tool_run = false;
                    actions.push(RenderAction::Append(Line::from("")));
                }
                self.reasoning_buffer.push_str(text);
                actions
            }

            AgentEvent::ToolStart {
                id,
                name,
                humanized,
            } => {
                let tool = RunningTool {
                    id: id.clone(),
                    name: name.clone(),
                    description: humanized.clone(),
                    diffs: Vec::new(),
                    started: Instant::now(),
                    line_count: 0,
                    progress: Vec::new(),
                    completion: None,
                };
                if self.tool_info.is_some() {
                    self.pending_tools.push_back(tool);
                    return Vec::new();
                }
                let mut actions = Vec::new();
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
                actions.extend(self.flush());

                actions.extend(self.show_running_tool(tool));
                actions
            }

            AgentEvent::ToolProgress { id, line } => {
                if let Some(tool) = self.pending_tools.iter_mut().find(|tool| tool.id == *id) {
                    tool.progress.push(line.clone());
                    return Vec::new();
                }
                let Some(tool) = self.tool_info.as_mut() else {
                    return Vec::new();
                };
                if tool.id != *id {
                    return Vec::new();
                }
                tool.progress.push(line.clone());
                self.render_tool_with_progress()
            }

            AgentEvent::ToolResult {
                id,
                name,
                success,
                elapsed_ms,
                output,
                ..
            } => {
                if let Some(tool) = self.pending_tools.iter_mut().find(|tool| tool.id == *id) {
                    tool.completion = Some(event.clone());
                    return Vec::new();
                }
                if self.tool_info.as_ref().is_some_and(|tool| tool.id != *id) {
                    self.pending_tools.push_back(RunningTool {
                        id: id.clone(),
                        name: name.clone(),
                        description: String::new(),
                        diffs: Vec::new(),
                        started: Instant::now(),
                        line_count: 0,
                        progress: Vec::new(),
                        completion: Some(event.clone()),
                    });
                    return Vec::new();
                }
                let running = self.tool_info.take();
                let erase_count = running.as_ref().map_or(0, |tool| tool.line_count);
                let tool_start = running.as_ref().map(|tool| tool.started);
                let tool = running.unwrap_or_else(|| RunningTool {
                    id: id.clone(),
                    name: name.clone(),
                    description: String::new(),
                    diffs: Vec::new(),
                    started: Instant::now(),
                    line_count: 0,
                    progress: Vec::new(),
                    completion: None,
                });

                if *success && SILENT_TOOLS.contains(&name.as_str()) && erase_count == 0 {
                    return self.show_next_tool();
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

                let mut collapsed = tool_block(&tool, icon, &elapsed, style, false);
                let mut expanded = tool_block(&tool, icon, &elapsed, style, true);

                // For silent edit tools the FileDiff event already shows the
                // change, so successful output is dropped.  Failures keep a
                // short preview while collapsed.
                let is_silent = SILENT_TOOLS.contains(&name.as_str()) || !tool.diffs.is_empty();
                if !output.is_empty() && (!success || !is_silent) {
                    let style = if *success { S_DIM } else { S_TOOL_FAIL };
                    let preview = if *success { 0 } else { FAILURE_PREVIEW };
                    let all: Vec<&str> = output.lines().collect();
                    let skip = all.len().saturating_sub(EXPANDED_OUTPUT_TAIL);
                    if skip > 0 {
                        let noun = if skip == 1 { "line" } else { "lines" };
                        expanded.push(tool_body(&format!("... {skip} earlier {noun}"), S_DIM));
                    }
                    expanded.extend(all[skip..].iter().map(|line| tool_body(line, style)));
                    collapsed.extend(
                        output
                            .lines()
                            .take(preview)
                            .map(|line| tool_body(line, style)),
                    );
                    let hidden = output.lines().count().saturating_sub(preview);
                    if hidden > 0 && !success {
                        collapsed.push(tool_body(&expand_hint(hidden, true), S_TOOL_FAIL));
                    }
                }
                expanded.extend(tool.diffs.iter().cloned());
                collapsed.extend(tool.diffs);

                let mut actions = if erase_count == 0 {
                    self.start_tool_run()
                } else {
                    Vec::new()
                };
                actions.push(RenderAction::FinishTool {
                    erase_count,
                    output: ToolOutput {
                        collapsed,
                        expanded,
                    },
                });
                actions.extend(self.show_next_tool());
                actions
            }

            AgentEvent::Interrupted {
                tool_call_id,
                tool_name,
                output,
                ..
            } => self.render(&AgentEvent::ToolResult {
                id: tool_call_id.clone(),
                name: tool_name.clone(),
                output: output.clone(),
                success: false,
                elapsed_ms: 0,
                sources: Vec::new(),
            }),

            AgentEvent::FileDiff { path, diff } => {
                // Line backgrounds fill the viewport, including wrapped rows.
                let mut lines = vec![Line::from(Span::styled(format!("    {path}"), S_DIFF_ADD))];
                let total = diff.len();
                let render_dl = |dl: &flashmind_types::tool::DiffLine| -> Line<'static> {
                    match dl {
                        flashmind_types::tool::DiffLine::Added { content, .. } => {
                            Line::from(format!("    +{content}")).style(S_DIFF_BLOCK_OK)
                        }
                        flashmind_types::tool::DiffLine::Removed { content, .. } => {
                            Line::from(format!("    -{content}")).style(S_DIFF_BLOCK_DEL)
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
                    lines.push(
                        Line::from(format!("    ... {} lines omitted ...", total - 100))
                            .style(S_DIFF_BLOCK_OK),
                    );
                    for dl in &diff[total - 50..] {
                        lines.push(render_dl(dl));
                    }
                } else {
                    for dl in diff {
                        lines.push(render_dl(dl));
                    }
                }
                if let Some(tool) = self.tool_info.as_mut()
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

            AgentEvent::Status(msg) => {
                vec![RenderAction::Append(Line::from(Span::styled(
                    format!("[{msg}]"),
                    S_DIM,
                )))]
            }

            AgentEvent::Error(msg) => {
                let mut actions = self.flush();
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
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

            AgentEvent::Done(_) | AgentEvent::Cancelled => {
                let cancelled = matches!(event, AgentEvent::Cancelled);
                let mut actions = Vec::new();
                if cancelled
                    && let Some(tool) = self.tool_info.take()
                    && !SILENT_TOOLS.contains(&tool.name.as_str())
                {
                    let mut collapsed = tool_block(&tool, "-", "Interrupted", S_DIM, false);
                    let mut expanded = tool_block(&tool, "-", "Interrupted", S_DIM, true);
                    expanded.extend(tool.progress.iter().map(|line| tool_body(line, S_DIM)));
                    expanded.extend(tool.diffs.iter().cloned());
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
                if self.in_reasoning {
                    actions.extend(self.end_reasoning());
                }
                actions.extend(self.flush());

                if let Some(start) = self.turn_start {
                    if !(self.reply_open && self.trailing_blank) {
                        actions.push(RenderAction::Append(Line::from("")));
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

                self.turn_start = None;
                self.turn_usage = TurnTotals::default();
                self.reply_open = false;
                self.in_tool_run = false;
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
                        RenderAction::Reasoning {
                            mut output,
                            expanded,
                        } => {
                            for line in output.collapsed.iter_mut().chain(&mut output.expanded) {
                                line.spans
                                    .insert(0, Span::styled(format!("  [{task}] "), S_SPAWNED));
                            }
                            RenderAction::Reasoning { output, expanded }
                        }
                        other => other,
                    })
                    .collect()
            }

            _ => Vec::new(),
        }
    }

    /// Retain the full trace when its streaming preview ends.
    fn end_reasoning(&mut self) -> Vec<RenderAction> {
        self.in_reasoning = false;
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
// no description. Descriptions already start with a verb ("Running `ls`",
// "Reading main.rs"), so the name would only repeat it. Collapsed, a
// multiline description (such as a long command) is cut after
// COLLAPSED_DESCRIPTION lines with a hint.
fn tool_block(
    tool: &RunningTool,
    icon: &str,
    elapsed: &str,
    style: Style,
    expanded: bool,
) -> Vec<Line<'static>> {
    let mut description = tool.description.lines();
    let label = match description.next() {
        Some(first) if !first.trim().is_empty() => first.trim().to_string(),
        _ => tool.name.clone(),
    };
    let mut lines = vec![tool_header(icon, &label, elapsed, style)];
    let rest: Vec<&str> = description.collect();
    let shown = if expanded {
        rest.len()
    } else {
        COLLAPSED_DESCRIPTION - 1
    };
    lines.extend(rest.iter().take(shown).map(|line| tool_body(line, S_TEXT)));
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
            S_ELAPSED,
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
mod tests {
    use super::*;
    use flashmind_types::AgentEvent;

    fn new_renderer() -> EventRenderer {
        let mut r = EventRenderer::new();
        r.set_width(80);
        r
    }

    fn tool_start(id: &str) -> AgentEvent {
        AgentEvent::ToolStart {
            id: id.into(),
            name: "exec".into(),
            humanized: format!("Command {id}"),
        }
    }

    fn tool_result(id: &str) -> AgentEvent {
        AgentEvent::ToolResult {
            id: id.into(),
            name: "exec".into(),
            output: format!("Output {id}"),
            success: true,
            elapsed_ms: 10,
            sources: Vec::new(),
        }
    }

    #[test]
    fn parallel_tools_keep_progress_and_results_with_their_descriptions() {
        let mut renderer = new_renderer();
        renderer.set_expand_tools(true);
        let start = renderer.render(&tool_start("a"));
        renderer.record_actions(&start);
        assert!(renderer.render(&tool_start("b")).is_empty());
        assert!(
            renderer
                .render(&AgentEvent::ToolProgress {
                    id: "b".into(),
                    line: "Progress b".into(),
                })
                .is_empty()
        );
        let progress = renderer.render(&AgentEvent::ToolProgress {
            id: "a".into(),
            line: "Progress a".into(),
        });
        renderer.record_actions(&progress);
        let text = renderer
            .lines()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Command a"));
        assert!(text.contains("Progress a"));
        assert!(!text.contains("Command b"));
        assert!(!text.contains("Progress b"));

        let first = renderer.render(&tool_result("a"));
        renderer.record_actions(&first);
        let RenderAction::FinishTool { output, .. } = &first[0] else {
            panic!("The first result should finish the first tool");
        };
        let text = output
            .expanded
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Command a"));
        assert!(text.contains("Output a"));
        assert!(!text.contains("Command b"));
        let text = renderer
            .lines()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Command b"));
        assert!(text.contains("Progress b"));
        assert!(!text.contains("Progress a"));

        let last = renderer.render(&tool_result("b"));
        renderer.record_actions(&last);
        assert!(!renderer.tool_running());
        assert!(renderer.tick_tool().is_empty());
        let text = renderer
            .lines()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Command a"));
        assert!(text.contains("Command b"));
        assert!(text.contains("Output a"));
        assert!(text.contains("Output b"));
    }

    #[test]
    fn parallel_results_can_finish_in_reverse_order() {
        let mut renderer = new_renderer();
        renderer.render(&tool_start("a"));
        renderer.render(&tool_start("b"));
        assert!(renderer.render(&tool_result("b")).is_empty());
        let actions = renderer.render(&tool_result("a"));
        let descriptions = actions
            .iter()
            .filter_map(|action| {
                if let RenderAction::FinishTool { output, .. } = action {
                    Some(output.collapsed[0].to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(descriptions.len(), 2);
        assert!(descriptions[0].contains("Command a"));
        assert!(descriptions[1].contains("Command b"));
        assert!(!renderer.tool_running());
        assert!(renderer.pending_tools.is_empty());
    }

    #[test]
    fn parallel_interrupts_and_cancellation_clear_running_tools() {
        let mut renderer = new_renderer();
        renderer.render(&tool_start("a"));
        renderer.render(&tool_start("b"));
        renderer.render(&AgentEvent::Interrupted {
            tool_call_id: "a".into(),
            tool_name: "exec".into(),
            output: "Needs input".into(),
            payload: None,
        });
        assert_eq!(renderer.tool_info.as_ref().unwrap().id, "b");
        renderer.render(&tool_start("c"));
        renderer.render(&AgentEvent::Cancelled);
        assert!(!renderer.tool_running());
        assert!(renderer.pending_tools.is_empty());
        assert!(renderer.tick_tool().is_empty());
    }

    #[test]
    fn silent_parallel_tool_does_not_erase_another_tools_live_lines() {
        let mut renderer = new_renderer();
        let start = renderer.render(&tool_start("a"));
        renderer.record_actions(&start);
        assert!(
            renderer
                .render(&AgentEvent::ToolStart {
                    id: "b".into(),
                    name: "file_write".into(),
                    humanized: "Write example.rs".into(),
                })
                .is_empty()
        );
        assert!(
            renderer
                .render(&AgentEvent::ToolResult {
                    id: "b".into(),
                    name: "file_write".into(),
                    output: "Written".into(),
                    success: true,
                    elapsed_ms: 10,
                    sources: Vec::new(),
                })
                .is_empty()
        );
        let result = renderer.render(&tool_result("a"));
        renderer.record_actions(&result);
        assert!(
            renderer
                .lines()
                .iter()
                .any(|line| line.to_string().contains("Command a"))
        );
        assert!(!renderer.tool_running());
    }

    #[test]
    fn cancellation_flushes_partial_output_and_closes_the_turn() {
        let mut renderer = new_renderer();
        renderer.mark_turn_start();
        renderer.render(&AgentEvent::ReasoningDelta("Thinking".into()));
        renderer.render(&AgentEvent::TextDelta("Partial reply".into()));
        let actions = renderer.render(&AgentEvent::Cancelled);
        let text = line_text(&actions);
        assert!(text.contains("Partial reply"));
        assert!(text.contains("Interrupted after"));
        assert!(!text.contains("Done in"));
        assert!(renderer.turn_start.is_none());
        assert!(renderer.text_buffer.is_empty());
        assert!(!renderer.in_reasoning);
        assert!(renderer.render(&AgentEvent::Done(String::new())).is_empty());

        renderer.mark_turn_start();
        renderer.render(&AgentEvent::TextDelta("Next reply".into()));
        let text = line_text(&renderer.render(&AgentEvent::Done(String::new())));
        assert!(text.contains("Next reply"));
        assert!(text.contains("Done in"));
        assert!(!text.contains("Interrupted"));
    }

    #[test]
    fn cancellation_replay_shows_the_interruption_without_a_timer() {
        let mut renderer = new_renderer();
        renderer.render(&AgentEvent::TextDelta("Partial reply".into()));
        let text = line_text(&renderer.render(&AgentEvent::Cancelled));
        assert!(text.contains("Partial reply"));
        assert!(text.contains("Interrupted"));
        assert!(!text.contains("Done in"));
    }

    #[test]
    fn cancellation_finishes_the_running_tool_and_keeps_its_progress() {
        let mut renderer = new_renderer();
        renderer.mark_turn_start();
        renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "Run checks".into(),
        });
        renderer.render(&AgentEvent::ToolProgress {
            id: "1".into(),
            line: "Checking".into(),
        });
        let actions = renderer.render(&AgentEvent::Cancelled);
        let RenderAction::FinishTool { output, .. } = &actions[0] else {
            panic!("Cancellation should finish the running tool block");
        };
        assert!(
            output
                .collapsed
                .iter()
                .any(|line| line.to_string().contains("Interrupted"))
        );
        assert!(
            output
                .expanded
                .iter()
                .any(|line| line.to_string().contains("Checking"))
        );
        assert!(renderer.tick_tool().is_empty());
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
                Paragraph::new(reply_lines(render_text_lines(sample, 78), false))
                    .render(area, &mut complete);
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
        assert!(line_text(&progress).contains("1 line (ctrl+o to expand)"));
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
    fn finished_tools_collapse_to_one_line_and_expand_to_full_output() {
        let mut renderer = new_renderer();
        renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "List files".into(),
        });
        let actions = renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            output: "a\nb\nc".into(),
            success: true,
            elapsed_ms: 5,
            sources: vec![],
        });
        let [RenderAction::FinishTool { output, .. }] = actions.as_slice() else {
            panic!("a finished tool carries both forms");
        };
        let text = |lines: &[Line<'static>]| -> Vec<String> {
            lines
                .iter()
                .map(|l| l.to_string().trim().to_owned())
                .collect()
        };
        let collapsed = text(&output.collapsed);
        assert_eq!(collapsed, ["\u{2713} List files  5 ms"]);
        let expanded = text(&output.expanded);
        assert_eq!(&expanded[expanded.len() - 3..], ["a", "b", "c"]);
    }

    #[test]
    fn elapsed_uses_minutes_and_hours_for_long_runs() {
        assert_eq!(format_elapsed(999), "999 ms");
        assert_eq!(format_elapsed(59_940), "59.9 s");
        assert_eq!(format_elapsed(65_000), "1m 5s");
        assert_eq!(format_elapsed(43_173_000), "11h 59m 33s");
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
        renderer.tool_info.as_mut().unwrap().started = Instant::now() - Duration::from_millis(500);
        let actions = renderer.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            success: true,
            elapsed_ms: 0,
            output: String::new(),
            sources: vec![],
        });
        let lines = match &actions[0] {
            RenderAction::FinishTool { output, .. } => &output.collapsed,
            _ => panic!("a completed tool replaces its live row"),
        };
        assert_ne!(lines[0].spans.last().unwrap().content.trim(), "0 ms");
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
                RenderAction::Reasoning { output, .. } => output.collapsed[0].spans.first(),
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
    fn finished_reasoning_collapses_and_retains_full_text() {
        let mut r = new_renderer();
        r.set_expand_reasoning(true);

        let a1 = r.render(&AgentEvent::ReasoningDelta("a thought\n".into()));
        let a2 = r.render(&AgentEvent::TextDelta("answer".into()));
        let actions: Vec<&RenderAction> = a1.iter().chain(a2.iter()).collect();

        let has_thought = actions.iter().any(|a| match a {
            RenderAction::Reasoning { output, expanded } => {
                assert!(!expanded);
                output
                    .expanded
                    .iter()
                    .any(|line| line.to_string().contains("a thought"))
            }
            _ => false,
        });
        assert!(
            has_thought,
            "the full trace remains available after collapse"
        );
        assert!(!r.expand_reasoning());
        assert!(r.pending_reasoning().is_empty());
    }

    #[test]
    fn reasoning_keeps_every_delta_when_display_changes_mid_stream() {
        let mut renderer = new_renderer();
        renderer.render(&AgentEvent::ReasoningDelta("first thought\n\n".into()));
        renderer.set_expand_reasoning(false);
        renderer.render(&AgentEvent::ReasoningDelta("second thought".into()));
        let actions = renderer.render(&AgentEvent::TextDelta("answer".into()));
        let RenderAction::Reasoning { output, expanded } = &actions[0] else {
            panic!("a completed reasoning block retains both display forms");
        };
        assert!(!expanded);
        assert_eq!(output.collapsed.len(), 1);
        let text = output
            .expanded
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("first thought"));
        assert!(text.contains("second thought"));
        assert!(renderer.pending_reasoning().is_empty());
    }

    #[test]
    fn terminal_events_retain_unfinished_reasoning() {
        for event in [
            AgentEvent::Done(String::new()),
            AgentEvent::Cancelled,
            AgentEvent::Error("example error".into()),
        ] {
            let mut renderer = new_renderer();
            renderer.set_expand_reasoning(false);
            renderer.render(&AgentEvent::ReasoningDelta("unfinished thought".into()));
            let actions = renderer.render(&event);
            assert!(actions.iter().any(|action| matches!(action, RenderAction::Reasoning { output, .. } if output.expanded.iter().any(|line| line.to_string().contains("unfinished thought")))));
            assert!(!renderer.in_reasoning);
            assert!(renderer.reasoning_buffer.is_empty());
        }
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
    fn expanded_tool_output_shows_the_tail() {
        let mut r = new_renderer();
        let _ = r.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "exec ls".into(),
        });
        let many: String = (0..50).map(|i| format!("row {i}\n")).collect();
        let actions = r.render(&AgentEvent::ToolResult {
            name: "exec".into(),
            id: "1".into(),
            output: many.trim_end().to_string(),
            success: true,
            elapsed_ms: 5,
            sources: vec![],
        });
        let RenderAction::FinishTool { output, .. } = actions.last().unwrap() else {
            panic!("expected a finished tool");
        };
        let text: Vec<String> = output.expanded.iter().map(|l| l.to_string()).collect();
        assert!(text.contains(&"    ... 30 earlier lines".to_string()));
        assert!(text.contains(&"    row 30".to_string()));
        assert!(text.contains(&"    row 49".to_string()));
        assert!(!text.contains(&"    row 29".to_string()));
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
                assert!(lines[1].to_string().contains("Run checks"));
                assert_eq!(
                    lines[2].to_string(),
                    "    first command with long arguments"
                );
                assert_eq!(lines[3].to_string(), "    ");
                // Collapsed cuts the description after three lines.
                if expanded {
                    assert_eq!(lines[4].to_string(), "      second command");
                } else {
                    assert_eq!(
                        lines[4].to_string(),
                        "    ... 1 more line (ctrl+o to expand)"
                    );
                }
                let rows = 5;
                // Expanded shows the progress line; collapsed shows a hint.
                let has_progress = renderer
                    .tool_info
                    .as_ref()
                    .is_some_and(|tool| !tool.progress.is_empty());
                assert_eq!(lines.len(), rows + usize::from(has_progress));
            }
        }
    }

    #[test]
    fn token_counts_scale_to_thousands_and_millions() {
        assert_eq!(format_tokens(812), "812");
        assert_eq!(format_tokens(12_300), "12.3k");
        assert_eq!(format_tokens(2_041_300), "2.04M");
    }

    #[test]
    fn turn_usage_parts_sum_the_requests_so_far() {
        let mut renderer = new_renderer();
        assert!(renderer.turn_usage_parts().is_empty());
        let usage = TokenUsage {
            prompt_tokens: 12_300,
            completion_tokens: 812,
            cache_read_tokens: 10_100,
            ..TokenUsage::default()
        };
        renderer.add_usage(&usage, None);
        renderer.add_usage(&usage, None);
        assert_eq!(
            renderer.turn_usage_parts().join(" \u{b7} "),
            "24.6k in (20.2k cached) \u{b7} 1.6k out"
        );
        renderer.set_show_usage(false);
        assert!(renderer.turn_usage_parts().is_empty());
    }

    #[test]
    fn adjacent_tools_share_one_block() {
        let mut renderer = new_renderer();
        for name in ["read", "exec"] {
            let start = renderer.render(&AgentEvent::ToolStart {
                name: name.into(),
                id: name.into(),
                humanized: format!("Example {name}"),
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
        assert_eq!(renderer.lines().len(), 3);
        assert_eq!(renderer.lines()[0].to_string(), "");
        assert!(renderer.lines()[1].to_string().contains("read"));
        assert!(renderer.lines()[2].to_string().contains("exec"));
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
        assert_eq!(lines.len(), 4);
        assert!(lines[1].contains("Editing example.rs"));
        assert_eq!(lines[2], "    example.rs");
        assert_eq!(lines[3], "    +new content");
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
        r.set_expand_tools(true);
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
    fn file_diff_background_fills_short_blank_and_wrapped_rows() {
        use crate::transcript::layout_line;
        use flashmind_types::tool::DiffLine;
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        use ratatui::widgets::{Paragraph, Widget, Wrap};

        for width in [20, 80] {
            for content in ["short".to_owned(), String::new(), "example ".repeat(30)] {
                let mut renderer = new_renderer();
                let actions = renderer.render(&AgentEvent::FileDiff {
                    path: "src/lib.rs".into(),
                    diff: vec![
                        DiffLine::Added {
                            content: content.clone(),
                            line: 1,
                        },
                        DiffLine::Removed { content, line: 1 },
                        DiffLine::Context {
                            content: "context".into(),
                            line: 2,
                        },
                    ],
                });
                renderer.record_actions(&actions);
                let lines = renderer.lines();
                for (line, style) in lines[lines.len() - 3..].iter().zip([
                    S_DIFF_BLOCK_OK,
                    S_DIFF_BLOCK_DEL,
                    S_DIFF_CONTEXT,
                ]) {
                    let rows = layout_line(line, width);
                    let area = Rect::new(0, 0, width, rows.len() as u16);
                    let mut buffer = Buffer::empty(area);
                    Paragraph::new(rows)
                        .wrap(Wrap { trim: false })
                        .render(area, &mut buffer);
                    for y in 0..area.height {
                        for x in 0..width {
                            assert_eq!(
                                buffer[(x, y)].bg,
                                style.bg.unwrap_or_default(),
                                "background at ({x}, {y}), width {width}"
                            );
                        }
                    }
                }
            }
        }
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
            S_ACCENT.fg,
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
                RenderAction::FinishTool { output, .. } => {
                    for l in output.lines(true) {
                        out.push(line_text_of(l));
                    }
                }
                RenderAction::Reasoning { output, expanded } => {
                    out.extend(output.lines(*expanded).iter().map(line_text_of));
                }
            }
        }
        out.join("\n")
    }

    fn line_text_of(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }
}
