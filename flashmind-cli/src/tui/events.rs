//! Agent event handling — converts AgentEvents into TUI output.

use std::time::Instant;

use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use flashmind_tui::styles::*;
use flashmind_types::AgentEvent;

use super::TuiApp;
use super::{format_duration, truncate_display};

// ---------------------------------------------------------------------------
// Subagent progress tracking
// ---------------------------------------------------------------------------

struct CompletedTool {
    name: String,
    humanized: String,
    success: bool,
    #[allow(dead_code)]
    elapsed_ms: u64,
}

struct SubagentProgress {
    id: String,
    task: String,
    completed_tools: Vec<CompletedTool>,
    current_tool: Option<(String, String)>,
    current_tool_start: Option<Instant>,
    last_completed: Option<CompletedTool>,
    started_at: Instant,
    finished_at: Option<Instant>,
    finished: bool,
}

// ---------------------------------------------------------------------------
// TuiState
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct TuiState {
    pub in_reasoning: bool,
    pub last_usage: Option<(u32, u32, u32)>,
    pub(super) tool_info: Option<(String, String)>,
    pub(super) tool_start_time: Option<Instant>,
    pub(super) tool_line: Option<usize>,
    pub(super) tool_body_lines: Vec<(usize, String)>,
    pub start_time: Option<Instant>,
    text_buffer: String,
    reasoning_buffer: String,
    subagents: Vec<SubagentProgress>,
}

fn format_token_count(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

impl TuiState {
    pub fn new() -> Self {
        Self::default()
    }

    fn find_subagent(&mut self, id: &str) -> Option<&mut SubagentProgress> {
        self.subagents.iter_mut().find(|s| s.id == id)
    }

    fn has_active_subagents(&self) -> bool {
        self.subagents.iter().any(|s| !s.finished)
    }
}

// ---------------------------------------------------------------------------
// ANSI → ratatui
// ---------------------------------------------------------------------------

fn split_humanized(humanized: &str) -> (String, Vec<String>) {
    let mut parts = humanized.split('\n');
    let primary = parts.next().unwrap_or("").trim_end().to_string();
    let mut body: Vec<String> = Vec::new();
    let mut prev_blank = false;
    for line in parts {
        let trimmed = line.trim_end().to_string();
        let is_blank = trimmed.is_empty();
        if is_blank && prev_blank {
            continue;
        }
        prev_blank = is_blank;
        body.push(trimmed);
    }
    while body.last().is_some_and(|s| s.is_empty()) {
        body.pop();
    }
    (primary, body)
}

fn ansi_to_lines(ansi_text: &str) -> Vec<Line<'static>> {
    use ::ansi_to_tui::IntoText;

    match ansi_text.into_text() {
        Ok(text) => text.lines.into_iter().collect(),
        Err(_) => vec![Line::raw(ansi_text.to_string())],
    }
}

fn flush_markdown(app: &mut TuiApp, text: &str) {
    let rendered = flashmind_tui::markdown_render::render_terminal(text);
    for line in ansi_to_lines(&rendered) {
        app.push_line(line);
    }
}

fn flush_reasoning(app: &mut TuiApp, text: &str) {
    let rendered = flashmind_tui::markdown_render::render_terminal(text);
    for mut line in ansi_to_lines(&rendered) {
        for span in &mut line.spans {
            span.style.fg = None;
            span.style = span.style.add_modifier(ratatui::style::Modifier::DIM);
        }
        app.push_line(line);
    }
}

// ---------------------------------------------------------------------------
// Subagent progress rendering
// ---------------------------------------------------------------------------

pub fn render_subagent_progress(state: &TuiState, width: usize) -> Vec<Line<'static>> {
    if state.subagents.is_empty() || !state.has_active_subagents() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let sep_width = width.saturating_sub(2);
    lines.push(Line::from(Span::styled("─".repeat(sep_width), S_DIM)));

    let start = state.subagents.len().saturating_sub(10);
    for sub in &state.subagents[start..] {
        let display_id = sub.id.rsplit(':').next().unwrap_or(&sub.id);
        let id_short = truncate_display(display_id, 8);
        let end = sub.finished_at.unwrap_or_else(Instant::now);
        let elapsed = format_duration(end.duration_since(sub.started_at).as_millis() as u64);

        let icon = if sub.finished { "\u{2714}" } else { "\u{25b8}" };
        let icon_style = if sub.finished { S_TOOL_OK } else { S_AGENT };

        let task_truncated = truncate_display(&sub.task, 50);

        let mut header_spans = vec![
            Span::styled(format!("  {} ", icon), icon_style),
            Span::styled(format!("{}  ", id_short), S_AGENT),
            Span::styled(task_truncated.to_string(), S_DIM),
        ];

        let header_text_len: usize = header_spans.iter().map(|s| s.content.len()).sum();
        let gap = width.saturating_sub(header_text_len + elapsed.len() + 2);
        header_spans.push(Span::styled(
            format!("{:>w$}", elapsed, w = gap + elapsed.len()),
            S_DIM,
        ));
        lines.push(Line::from(header_spans));

        if sub.finished {
            continue;
        }

        // Completed tools summary
        if sub.completed_tools.is_empty() {
            lines.push(Line::from(Span::raw("")));
        } else {
            let total = sub.completed_tools.len();
            let show_count = 3.min(total);
            let names: Vec<&str> = sub.completed_tools[total - show_count..]
                .iter()
                .map(|t| t.name.as_str())
                .collect();
            let summary = if total > show_count {
                format!(
                    "      \u{25cf} {}, +{} others",
                    names.join(", "),
                    total - show_count
                )
            } else {
                format!("      \u{25cf} {}", names.join(", "))
            };
            lines.push(Line::from(Span::styled(summary, S_DIM)));
        }

        // Current running tool
        if let Some((ref name, ref humanized)) = sub.current_tool {
            let display_text = if humanized.is_empty() {
                name
            } else {
                humanized
            };
            let display = format!("      \u{25cc} {}", display_text);
            lines.push(Line::from(Span::styled(display, S_TOOL_RUN)));
        } else if let Some(ref last) = sub.last_completed {
            let (icon, style) = if last.success {
                ("\u{2713}", S_TOOL_OK)
            } else {
                ("\u{2717}", S_TOOL_FAIL)
            };
            let display = if last.humanized.is_empty() {
                format!("      {} {}", icon, last.name)
            } else {
                format!("      {} {}", icon, last.humanized)
            };
            lines.push(Line::from(Span::styled(display, style)));
        } else {
            lines.push(Line::from(Span::raw("")));
        }
    }

    lines
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

impl TuiApp<'_> {
    pub fn handle_agent_event(&mut self, event: &AgentEvent, state: &mut TuiState) {
        match event {
            AgentEvent::ReasoningDelta(delta) => {
                if !state.in_reasoning {
                    state.in_reasoning = true;
                    self.newline();
                }
                state.reasoning_buffer.push_str(delta);
                let result = flashmind_tui::markdown::parse_document(&state.reasoning_buffer);
                if result.blocks.len() >= 2 {
                    let before_last_offset =
                        state.reasoning_buffer.len() - result.before_last.len();
                    if before_last_offset > 0
                        && state.reasoning_buffer.is_char_boundary(before_last_offset)
                    {
                        flush_reasoning(self, &state.reasoning_buffer[..before_last_offset]);
                        let remainder = state.reasoning_buffer[before_last_offset..].to_string();
                        state.reasoning_buffer = remainder;
                    }
                }
            }

            AgentEvent::TextDelta(delta) => {
                if state.in_reasoning {
                    state.in_reasoning = false;
                    if !state.reasoning_buffer.is_empty() {
                        flush_reasoning(self, &state.reasoning_buffer);
                        state.reasoning_buffer.clear();
                    }
                    self.separator();
                }

                state.text_buffer.push_str(delta);
                let result = flashmind_tui::markdown::parse_document(&state.text_buffer);
                if result.blocks.len() >= 2 {
                    let before_last_offset = state.text_buffer.len() - result.before_last.len();
                    if before_last_offset > 0
                        && state.text_buffer.is_char_boundary(before_last_offset)
                    {
                        flush_markdown(self, &state.text_buffer[..before_last_offset]);
                        let remainder = state.text_buffer[before_last_offset..].to_string();
                        state.text_buffer = remainder;
                    }
                }
            }

            AgentEvent::ToolStart {
                name, humanized, ..
            } => {
                if state.in_reasoning {
                    state.in_reasoning = false;
                    if !state.reasoning_buffer.is_empty() {
                        flush_reasoning(self, &state.reasoning_buffer);
                        state.reasoning_buffer.clear();
                    }
                    self.separator();
                }

                if !state.text_buffer.is_empty() {
                    flush_markdown(self, &state.text_buffer);
                    state.text_buffer.clear();
                }

                let (primary, body) = split_humanized(humanized);
                let w = self.width();
                let max_text = w.saturating_sub(12);
                let display = if primary.is_empty() {
                    format!("  \u{25cc} {}", name)
                } else {
                    let full = format!("  \u{25cc} {}", primary);
                    if UnicodeWidthStr::width(full.as_str()) > max_text {
                        format!(
                            "{}\u{2026}",
                            truncate_display(&full, max_text.saturating_sub(1))
                        )
                    } else {
                        full
                    }
                };

                state.tool_line = Some(self.lines.len());
                state.tool_info = Some((name.clone(), primary.clone()));
                state.tool_start_time = Some(Instant::now());
                self.push_line(Line::from(Span::styled(display, S_TOOL_RUN)));

                state.tool_body_lines.clear();
                for cont in body {
                    let indent = "      ";
                    let avail = w.saturating_sub(indent.len());
                    let rendered = if UnicodeWidthStr::width(cont.as_str()) > avail {
                        format!(
                            "{}{}\u{2026}",
                            indent,
                            truncate_display(&cont, avail.saturating_sub(1))
                        )
                    } else {
                        format!("{}{}", indent, cont)
                    };
                    let idx = self.lines.len();
                    self.push_line(Line::from(Span::styled(rendered.clone(), S_TOOL_RUN)));
                    state.tool_body_lines.push((idx, rendered));
                }
            }

            AgentEvent::ToolResult {
                success,
                elapsed_ms,
                ..
            } => {
                let (name, humanized) = state
                    .tool_info
                    .take()
                    .unwrap_or_else(|| (String::new(), String::new()));

                let ms = if *elapsed_ms > 0 {
                    *elapsed_ms
                } else {
                    state
                        .tool_start_time
                        .take()
                        .map(|t| t.elapsed().as_millis() as u64)
                        .unwrap_or(0)
                };
                let elapsed = format_duration(ms);

                let (icon, style) = if *success {
                    ("\u{2713}", S_TOOL_OK)
                } else {
                    ("\u{2717}", S_TOOL_FAIL)
                };

                let w = self.width();
                let elapsed_col = 10;
                let max_text = w.saturating_sub(elapsed_col + 2);
                let tool_text = if humanized.is_empty() {
                    format!("  {} {}", icon, name)
                } else {
                    let full = format!("  {} {}", icon, humanized);
                    if UnicodeWidthStr::width(full.as_str()) > max_text {
                        format!(
                            "{}\u{2026}",
                            truncate_display(&full, max_text.saturating_sub(1))
                        )
                    } else {
                        full
                    }
                };

                let tool_cols = UnicodeWidthStr::width(tool_text.as_str());
                let padded_elapsed = format!("{:>w$}", elapsed, w = w.saturating_sub(tool_cols));

                if let Some(idx) = state.tool_line.take() {
                    if idx < self.lines.len() {
                        self.update_line(
                            idx,
                            Line::from(vec![
                                Span::styled(tool_text, style),
                                Span::styled(padded_elapsed, S_DIM),
                            ]),
                        );
                    }
                } else {
                    self.push_line(Line::from(vec![
                        Span::styled(tool_text, style),
                        Span::styled(padded_elapsed, S_DIM),
                    ]));
                }

                for (idx, text) in std::mem::take(&mut state.tool_body_lines) {
                    if idx < self.lines.len() {
                        self.update_line(idx, Line::from(Span::styled(text, style)));
                    }
                }
            }

            AgentEvent::SpawnedEvent {
                id,
                task,
                event: inner,
                ..
            } => {
                self.handle_subagent_event(id, task, inner, state);
            }

            AgentEvent::Done(_) => {
                self.stop_spinner();

                if state.in_reasoning {
                    state.in_reasoning = false;
                    if !state.reasoning_buffer.is_empty() {
                        flush_reasoning(self, &state.reasoning_buffer);
                        state.reasoning_buffer.clear();
                    }
                    self.separator();
                }

                if !state.text_buffer.is_empty() {
                    flush_markdown(self, &state.text_buffer);
                    state.text_buffer.clear();
                }

                if let Some((total, prompt, completion)) = state.last_usage {
                    let elapsed_str = state
                        .start_time
                        .map(|t| format_duration(t.elapsed().as_millis() as u64))
                        .unwrap_or_default();

                    self.newline();
                    let footer = if elapsed_str.is_empty() {
                        format!("{}p + {}c ({})", prompt, completion, total)
                    } else {
                        format!(
                            "{}p + {}c ({}) \u{00b7} {}",
                            prompt, completion, total, elapsed_str
                        )
                    };
                    self.push_line(Line::from(Span::styled(footer, S_DIM)));
                }
            }

            AgentEvent::Error(msg) => {
                self.push_line(Line::from(Span::styled(
                    format!("[error] {}", msg),
                    S_ERROR,
                )));
            }

            AgentEvent::Usage(usage) => {
                if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
                    return;
                }
                let usage_tuple = (
                    usage.total_tokens,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                );
                state.last_usage = Some(usage_tuple);
                self.last_usage = Some(usage_tuple);
                self.cumulative_prompt += usage.prompt_tokens as u64;
                self.cumulative_completion += usage.completion_tokens as u64;
                let ctx_part = if self.context_window > 0 {
                    let pct =
                        (usage.prompt_tokens as f64 / self.context_window as f64 * 100.0) as u32;
                    format!(" · {}%", pct)
                } else {
                    String::new()
                };
                let cumulative =
                    format_token_count(self.cumulative_prompt + self.cumulative_completion);
                self.set_status_extra(format!(
                    "{}p + {}c{} · Σ{}",
                    usage.prompt_tokens, usage.completion_tokens, ctx_part, cumulative
                ));
            }

            AgentEvent::Status(status) => {
                self.push_line(Line::from(Span::styled(format!("[{}]", status), S_DIM)));
            }

            AgentEvent::Started { .. } => {
                state.start_time = Some(Instant::now());
            }

            AgentEvent::Compacted(summary) => {
                self.lines.clear();
                self.invalidate();
                self.newline();
                self.push_line(Line::from(Span::styled("[compacted]", S_DIM)));
                flush_markdown(self, summary);
            }

            AgentEvent::FileDiff { diff, .. } => {
                use flashmind_types::tool::DiffLine;
                let total = diff.len();

                let render_diff_line = |dl: &DiffLine| -> (String, ratatui::style::Style) {
                    match dl {
                        DiffLine::Added { content, .. } => {
                            (format!("    +{}", content), S_DIFF_ADD)
                        }
                        DiffLine::Removed { content, .. } => {
                            (format!("    -{}", content), S_DIFF_DEL)
                        }
                    }
                };

                if total > 100 {
                    for dl in &diff[..50] {
                        let (text, style) = render_diff_line(dl);
                        self.push_line(Line::from(Span::styled(text, style)));
                    }
                    self.push_line(Line::from(Span::styled(
                        format!("    ... {} lines omitted ...", total - 100),
                        S_DIM,
                    )));
                    for dl in &diff[total - 50..] {
                        let (text, style) = render_diff_line(dl);
                        self.push_line(Line::from(Span::styled(text, style)));
                    }
                } else {
                    for dl in diff {
                        let (text, style) = render_diff_line(dl);
                        self.push_line(Line::from(Span::styled(text, style)));
                    }
                }
            }

            AgentEvent::Interrupted { .. } => {
                self.stop_spinner();

                // Clear the running tool indicator (no ToolResult is emitted for interrupts).
                let (name, humanized) = state
                    .tool_info
                    .take()
                    .unwrap_or_else(|| (String::new(), String::new()));

                let ms = state
                    .tool_start_time
                    .take()
                    .map(|t| t.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                let elapsed = format_duration(ms);

                let w = self.width();
                let max_text = w.saturating_sub(12);
                let tool_text = if humanized.is_empty() {
                    format!("  \u{25a0} {}", name)
                } else {
                    let full = format!("  \u{25a0} {}", humanized);
                    if UnicodeWidthStr::width(full.as_str()) > max_text {
                        format!(
                            "{}\u{2026}",
                            truncate_display(&full, max_text.saturating_sub(1))
                        )
                    } else {
                        full
                    }
                };

                let tool_cols = UnicodeWidthStr::width(tool_text.as_str());
                let padded_elapsed = format!("{:>w$}", elapsed, w = w.saturating_sub(tool_cols));

                if let Some(idx) = state.tool_line.take()
                    && idx < self.lines.len()
                {
                    self.update_line(
                        idx,
                        Line::from(vec![
                            Span::styled(tool_text, S_DIM),
                            Span::styled(padded_elapsed, S_DIM),
                        ]),
                    );
                }

                for (idx, text) in std::mem::take(&mut state.tool_body_lines) {
                    if idx < self.lines.len() {
                        self.update_line(idx, Line::from(Span::styled(text, S_DIM)));
                    }
                }

                if !state.text_buffer.is_empty() {
                    flush_markdown(self, &state.text_buffer);
                    state.text_buffer.clear();
                }
            }

            AgentEvent::AudioChunk { .. } => {}
        }
    }

    fn handle_subagent_event(
        &mut self,
        id: &str,
        task: &str,
        event: &AgentEvent,
        state: &mut TuiState,
    ) {
        match event {
            AgentEvent::Status(_) | AgentEvent::Started { .. }
                if state.find_subagent(id).is_none() =>
            {
                state.subagents.push(SubagentProgress {
                    id: id.to_string(),
                    task: task.to_string(),
                    completed_tools: Vec::new(),
                    current_tool: None,
                    current_tool_start: None,
                    last_completed: None,
                    started_at: Instant::now(),
                    finished_at: None,
                    finished: false,
                });
            }

            AgentEvent::ToolStart {
                name, humanized, ..
            } => {
                if state.find_subagent(id).is_none() {
                    state.subagents.push(SubagentProgress {
                        id: id.to_string(),
                        task: task.to_string(),
                        completed_tools: Vec::new(),
                        current_tool: None,
                        current_tool_start: None,
                        last_completed: None,
                        started_at: Instant::now(),
                        finished_at: None,
                        finished: false,
                    });
                }
                if let Some(sub) = state.find_subagent(id) {
                    sub.current_tool = Some((name.clone(), humanized.clone()));
                    sub.current_tool_start = Some(Instant::now());
                }
            }

            AgentEvent::ToolResult {
                name,
                success,
                elapsed_ms,
                ..
            } => {
                if let Some(sub) = state.find_subagent(id) {
                    let ms = if *elapsed_ms > 0 {
                        *elapsed_ms
                    } else {
                        sub.current_tool_start
                            .take()
                            .map(|t| t.elapsed().as_millis() as u64)
                            .unwrap_or(0)
                    };

                    let (tool_name, humanized) = sub
                        .current_tool
                        .take()
                        .unwrap_or_else(|| (name.clone(), String::new()));

                    let completed = CompletedTool {
                        name: tool_name.clone(),
                        humanized: humanized.clone(),
                        success: *success,
                        elapsed_ms: ms,
                    };

                    sub.last_completed = Some(CompletedTool {
                        name: tool_name,
                        humanized,
                        success: *success,
                        elapsed_ms: ms,
                    });
                    sub.completed_tools.push(completed);
                }
            }

            AgentEvent::Done(_) => {
                if let Some(sub) = state.find_subagent(id) {
                    sub.finished = true;
                    sub.finished_at = Some(Instant::now());
                    sub.current_tool = None;
                }
            }

            AgentEvent::Error(_) => {
                if let Some(sub) = state.find_subagent(id) {
                    sub.finished = true;
                    sub.finished_at = Some(Instant::now());
                    sub.current_tool = None;
                }
            }

            _ => {}
        }
    }
}
