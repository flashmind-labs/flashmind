use super::*;
use flashmind_types::AgentEvent;

fn new_renderer() -> EventRenderer {
    let mut r = EventRenderer::new();
    r.set_width(80);
    r
}

fn displayed_lines(lines: &[Line<'static>], width: u16) -> Vec<Line<'static>> {
    lines
        .iter()
        .flat_map(|line| crate::transcript::layout_line(line, width))
        .collect()
}

#[test]
fn command_text_is_white_and_output_stays_grey() {
    use ratatui::style::Color;

    for expanded in [false, true] {
        let mut renderer = new_renderer();
        renderer.set_expand_tools(expanded);
        let start = renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "command".into(),
            humanized: "Running `echo first\necho second`".into(),
        });
        renderer.record_actions(&start);
        for event in [
            AgentEvent::ToolProgress {
                id: "command".into(),
                line: "first\nsecond".into(),
            },
            AgentEvent::ToolResult {
                name: "exec".into(),
                id: "command".into(),
                success: true,
                elapsed_ms: 12,
                output: "first\nsecond".into(),
                sources: Vec::new(),
            },
        ] {
            let actions = renderer.render(&event);
            renderer.record_actions(&actions);
            for line in renderer.lines() {
                for span in &line.spans {
                    if span.content.contains("echo ") {
                        assert_eq!(span.style.fg, Some(Color::White));
                    } else if matches!(span.content.trim(), "first" | "second") {
                        assert_eq!(span.style.fg, S_DIM.fg);
                    }
                }
            }
        }
    }
}

#[test]
fn compaction_displays_the_summary() {
    let mut renderer = new_renderer();
    let actions = renderer.render(&AgentEvent::Compacted("Conversation summary.".into()));
    assert_eq!(
        line_text(&actions).trim(),
        "[Context compacted.]\nConversation summary."
    );
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
    assert!(renderer.tools.is_empty());
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
    assert_eq!(renderer.tools.front().unwrap().id, "b");
    renderer.render(&tool_start("c"));
    renderer.render(&AgentEvent::Cancelled);
    assert!(!renderer.tool_running());
    assert!(renderer.tools.is_empty());
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
fn collapsed_commands_show_live_progress_and_errors() {
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
    assert!(line_text(&progress).contains("verbose progress"));
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
fn running_commands_show_three_command_lines_and_five_output_lines() {
    let mut renderer = new_renderer();
    let start = renderer.render(&AgentEvent::ToolStart {
        name: "exec".into(),
        id: "1".into(),
        humanized: "first command\nsecond command\nthird command\nfourth command".into(),
    });
    let text = line_text(&start);
    assert!(text.contains("first command"));
    assert!(text.contains("second command"));
    assert!(text.contains("third command"));
    assert!(text.contains("\u{2022} first command"));
    assert!(text.contains("... 1 more line (ctrl+o to expand)"));
    assert!(!text.contains("fourth command"));
    let mut progress = Vec::new();
    for index in 0..12 {
        progress = renderer.render(&AgentEvent::ToolProgress {
            id: "1".into(),
            line: format!("output-{index:02}"),
        });
    }
    let text = line_text(&progress);
    assert!(text.contains("output-07"));
    assert!(text.contains("output-11"));
    assert!(!text.contains("output-06"));
    assert!(!text.contains("fourth command"));
    renderer.set_expand_tools(true);
    assert!(line_text(&renderer.tick_tool()).contains("fourth command"));
}

#[test]
fn long_commands_wrap_before_the_three_row_limit_and_reflow_on_resize() {
    use crate::transcript::layout_line;
    let mut renderer = new_renderer();
    let command = format!("first {} hidden-command-tail", "argument ".repeat(20));
    let start = renderer.render(&AgentEvent::ToolStart {
        name: "exec".into(),
        id: "1".into(),
        humanized: command.clone(),
    });
    let header = start
        .iter()
        .find_map(|action| match action {
            RenderAction::Append(line) if line.width() > 0 => Some(line),
            _ => None,
        })
        .unwrap();
    let rows = layout_line(header, 24);
    assert_eq!(rows.len(), 4); // Three command rows and the expansion hint.
    let text = rows
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("\u{2022} first"));
    assert!(rows[1].to_string().contains("argument"));
    assert!(!text.contains("hidden-command-tail"));
    assert!(header.to_string().contains(&command));
    let wide = layout_line(header, 300);
    assert_eq!(wide.len(), 1);
    assert!(wide[0].to_string().contains("hidden-command-tail"));
    renderer.set_expand_tools(true);
    assert!(line_text(&renderer.tick_tool()).contains("hidden-command-tail"));
}

#[test]
fn command_row_limit_counts_wrapping_before_explicit_newlines() {
    use crate::transcript::layout_line;

    for body in ["argument ".repeat(40), "界".repeat(200)] {
        let command = format!("{body}\nhidden-second-command");
        let mut renderer = new_renderer();
        renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "command".into(),
            humanized: command,
        });
        let actions = renderer.tick_tool();
        let RenderAction::ReplaceTool { lines, .. } = &actions[0] else {
            panic!("The running command has a live preview.");
        };
        for width in 0..80 {
            let rows = layout_line(&lines[0], width);
            assert!(rows.len() <= 4);
            assert!(rows.iter().all(|row| row.width() <= usize::from(width)));
            assert!(
                !rows
                    .iter()
                    .any(|row| row.to_string().contains("hidden-second-command"))
            );
        }
        assert_eq!(layout_line(&lines[0], 40).len(), 4);
    }
}

#[test]
fn finished_commands_retain_the_output_tail_and_expand_to_full_output() {
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
    assert_eq!(collapsed, ["\u{2022} List files  5 ms", "a", "b", "c"]);
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
    renderer.tools.front_mut().unwrap().started = Some(Instant::now() - Duration::from_millis(500));
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
            joined.contains(&format!("row {index}\n")) || joined.ends_with(&format!("row {index}"))
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
            name: "read".into(),
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
                name: "read".into(),
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
                .tools
                .front()
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

#[test]
fn orphan_results_finish_immediately_or_after_the_current_tool() {
    let mut renderer = new_renderer();
    let immediate = renderer.render(&tool_result("orphan"));
    let finished: Vec<_> = immediate
        .iter()
        .filter_map(|action| match action {
            RenderAction::FinishTool {
                erase_count,
                output,
            } => Some((erase_count, output)),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 1);
    assert_eq!(*finished[0].0, 0);
    assert!(finished[0].1.collapsed[0].to_string().contains("exec"));
    assert!(line_text(&immediate).contains("Output orphan"));
    assert!(!renderer.tool_running());

    renderer.render(&tool_start("current"));
    assert!(renderer.render(&tool_result("orphan")).is_empty());
    let actions = renderer.render(&tool_result("current"));
    let finished: Vec<_> = actions
        .iter()
        .filter_map(|action| match action {
            RenderAction::FinishTool { output, .. } => Some(output),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 2);
    assert!(
        finished[0].collapsed[0]
            .to_string()
            .contains("Command current")
    );
    assert!(finished[1].collapsed[0].to_string().contains("exec"));
    let outputs: Vec<_> = finished
        .iter()
        .map(|output| {
            output
                .expanded
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect();
    assert!(outputs[0].contains("Output current"));
    assert!(!outputs[0].contains("Output orphan"));
    assert!(outputs[1].contains("Output orphan"));
    assert!(!renderer.tool_running());
    assert!(renderer.tick_tool().is_empty());
}

#[test]
fn queued_silent_failure_keeps_its_description_and_error() {
    let mut renderer = new_renderer();
    renderer.render(&tool_start("current"));
    assert!(
        renderer
            .render(&AgentEvent::ToolStart {
                id: "write".into(),
                name: "file_write".into(),
                humanized: "Write example.rs".into(),
            })
            .is_empty()
    );
    assert!(
        renderer
            .render(&AgentEvent::ToolResult {
                id: "write".into(),
                name: "file_write".into(),
                success: false,
                elapsed_ms: 4,
                output: "Write failed".into(),
                sources: Vec::new(),
            })
            .is_empty()
    );
    let actions = renderer.render(&tool_result("current"));
    let finished: Vec<_> = actions
        .iter()
        .filter_map(|action| match action {
            RenderAction::FinishTool {
                erase_count,
                output,
            } => Some((erase_count, output)),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 2);
    assert!(*finished[0].0 > 0);
    assert_eq!(*finished[1].0, 0);
    let failed = finished[1].1;
    assert!(failed.collapsed[0].to_string().contains("Write example.rs"));
    assert!(
        failed
            .collapsed
            .iter()
            .any(|line| line.to_string().contains("Write failed"))
    );
    assert!(
        failed
            .expanded
            .iter()
            .any(|line| line.to_string().contains("Write failed"))
    );
    assert!(!renderer.tool_running());
}

#[test]
fn done_keeps_the_live_preview_and_cancelled_finishes_only_the_current_tool() {
    for cancelled in [false, true] {
        let mut renderer = new_renderer();
        renderer.set_expand_tools(true);
        let actions = renderer.render(&tool_start("current"));
        renderer.record_actions(&actions);
        let actions = renderer.render(&AgentEvent::ToolProgress {
            id: "current".into(),
            line: "Current progress".into(),
        });
        renderer.record_actions(&actions);
        renderer.render(&tool_start("queued"));
        renderer.render(&tool_result("queued"));
        let event = if cancelled {
            AgentEvent::Cancelled
        } else {
            AgentEvent::Done(String::new())
        };
        let actions = renderer.render(&event);
        assert_eq!(
            actions
                .iter()
                .filter(|action| matches!(action, RenderAction::FinishTool { .. }))
                .count(),
            usize::from(cancelled)
        );
        renderer.record_actions(&actions);
        let text = renderer
            .lines()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Command current"));
        assert!(text.contains("Current progress"));
        assert!(!text.contains("Command queued"));
        assert!(!text.contains("Output queued"));
        assert_eq!(text.contains("Interrupted"), cancelled);
        assert!(!renderer.tool_running());
        assert!(renderer.tick_tool().is_empty());
    }
}

#[test]
fn large_diffs_keep_the_first_and_last_fifty_rows_and_elision_markers() {
    use flashmind_types::tool::DiffLine;

    let mut renderer = new_renderer();
    let mut diff: Vec<_> = (0..105)
        .map(|index| DiffLine::Context {
            line: index + 1,
            content: format!("context-{index:03}"),
        })
        .collect();
    diff[0] = DiffLine::Context {
        line: 0,
        content: String::new(),
    };
    let actions = renderer.render(&AgentEvent::FileDiff {
        path: "example.rs".into(),
        diff,
    });
    let lines: Vec<_> = actions
        .iter()
        .filter_map(|action| match action {
            RenderAction::Append(line) if line.width() > 0 => Some(line.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(lines.len(), 102);
    assert_eq!(lines[0], "    example.rs");
    assert_eq!(lines[1], "    \u{2026}");
    assert_eq!(lines[50], "     context-049");
    assert_eq!(lines[51], "    ... 5 lines omitted ...");
    assert_eq!(lines[52], "     context-055");
    assert_eq!(lines[101], "     context-104");
    assert!(!lines.iter().any(|line| line.contains("context-050")));
    assert!(!lines.iter().any(|line| line.contains("context-054")));
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

fn line_text_of(line: &Line<'static>) -> String {
    crate::transcript::layout_line(line, 80)
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn command_output_truncates_each_row_and_restores_it_on_resize() {
    use crate::transcript::layout_line;
    let mut renderer = new_renderer();
    renderer.render(&AgentEvent::ToolStart {
        name: "exec".into(),
        id: "1".into(),
        humanized: "python3 - <<'PY'\nprint('example')\nPY".into(),
    });
    let long = format!("example {} final-output", "argument ".repeat(20));
    for index in 0..7 {
        renderer.render(&AgentEvent::ToolProgress {
            id: "1".into(),
            line: format!("row-{index}"),
        });
    }
    let progress = renderer.render(&AgentEvent::ToolProgress {
        id: "1".into(),
        line: long.clone(),
    });
    let [RenderAction::ReplaceTool { lines, .. }] = progress.as_slice() else {
        panic!("expected a live preview");
    };
    let displayed = displayed_lines(lines, 80);
    assert_eq!(displayed.len(), 9);
    assert_eq!(
        displayed[3].to_string(),
        "       ... 3 earlier lines (ctrl+o to expand)"
    );
    assert_eq!(displayed[4].to_string(), "       row-3");
    assert!(
        displayed[0]
            .to_string()
            .contains("\u{2022} python3 - <<'PY'")
    );
    assert_eq!(displayed[1].to_string(), "    print('example')");
    assert_eq!(displayed[2].to_string(), "    PY");
    for width in [0, 1, 8, 24, 80, 300, 24] {
        for line in lines {
            for row in layout_line(line, width as u16) {
                assert!(row.width() <= width);
            }
        }
        let rows = layout_line(lines.last().unwrap(), width as u16);
        assert_eq!(rows.len(), 1);
        if width >= 300 {
            assert!(rows[0].to_string().contains("final-output"));
        } else if width >= 24 {
            assert!(rows[0].to_string().ends_with('\u{2026}'));
            assert!(!rows[0].to_string().contains("final-output"));
        }
    }
    let output = (0..8)
        .map(|index| format!("row-{index}\n"))
        .collect::<String>();
    let completed = renderer.render(&AgentEvent::ToolResult {
        name: "exec".into(),
        id: "1".into(),
        success: false,
        elapsed_ms: 1,
        output,
        sources: Vec::new(),
    });
    let RenderAction::FinishTool { output, .. } = &completed[0] else {
        panic!("expected completion");
    };
    let displayed = displayed_lines(&output.collapsed, 80);
    assert_eq!(displayed.len(), 9);
    assert!(displayed[0].to_string().contains('\u{2717}'));
    assert_eq!(
        displayed[3].to_string(),
        "       ... 3 earlier lines (ctrl+o to expand)"
    );
    assert_eq!(displayed[4].to_string(), "       row-3");
    assert_eq!(displayed[8].to_string(), "       row-7");
}

#[test]
fn collapsed_commands_keep_three_lines_through_updates_and_completion() {
    for cancelled in [false, true] {
        let mut renderer = new_renderer();
        let start = renderer.render(&AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "Running `python3 - <<'PY'\n  print('first')\n  print('second')\nPY`".into(),
        });
        renderer.record_actions(&start);
        for event in [
            None,
            Some(AgentEvent::ToolProgress {
                id: "1".into(),
                line: "first\nsecond".into(),
            }),
            Some(if cancelled {
                AgentEvent::Cancelled
            } else {
                AgentEvent::ToolResult {
                    name: "exec".into(),
                    id: "1".into(),
                    success: true,
                    elapsed_ms: 12,
                    output: "first\nsecond".into(),
                    sources: Vec::new(),
                }
            }),
        ] {
            let lines = displayed_lines(renderer.lines(), 80);
            assert!(lines[1].to_string().contains("python3 - <<'PY'"));
            assert_eq!(lines[2].to_string(), "      print('first')");
            assert_eq!(lines[3].to_string(), "      print('second')");
            assert_eq!(
                lines[4].to_string(),
                "    ... 1 more line (ctrl+o to expand)"
            );
            let actions = match event {
                Some(event) => renderer.render(&event),
                None => renderer.tick_tool(),
            };
            if let Some(RenderAction::FinishTool { output, .. }) = actions.first() {
                assert_eq!(output.expanded[3].to_string(), "    PY");
            }
            renderer.record_actions(&actions);
        }
        let lines = displayed_lines(renderer.lines(), 80);
        assert_eq!(lines[2].to_string(), "      print('first')");
        assert_eq!(lines[3].to_string(), "      print('second')");
        assert_eq!(
            lines[4].to_string(),
            "    ... 1 more line (ctrl+o to expand)"
        );
    }
}

#[test]
fn discarded_tool_reruns_as_one_block() {
    let mut renderer = new_renderer();
    let start = AgentEvent::ToolStart {
        name: "exec".into(),
        id: "command".into(),
        humanized: "Running `ls`".into(),
    };
    let actions = renderer.render(&start);
    renderer.record_actions(&actions);
    let actions = renderer.discard_tool("command");
    renderer.record_actions(&actions);
    for event in [
        start,
        AgentEvent::ToolResult {
            name: "exec".into(),
            id: "command".into(),
            success: true,
            elapsed_ms: 3,
            output: "listed".into(),
            sources: Vec::new(),
        },
    ] {
        let actions = renderer.render(&event);
        renderer.record_actions(&actions);
    }

    let text: Vec<String> = renderer
        .lines()
        .iter()
        .map(|line| line.to_string())
        .collect();
    assert_eq!(
        text.iter().filter(|line| line.contains("ls")).count(),
        1,
        "{text:#?}"
    );
    assert!(!renderer.tool_running());
}

#[test]
fn collapsed_command_output_says_how_many_lines_are_hidden() {
    let mut renderer = new_renderer();
    renderer.render(&AgentEvent::ToolStart {
        name: "exec".into(),
        id: "1".into(),
        humanized: "List files".into(),
    });
    let actions = renderer.render(&AgentEvent::ToolResult {
        name: "exec".into(),
        id: "1".into(),
        output: "a\nb\nc\nd\ne\nf\ng\nh".into(),
        success: true,
        elapsed_ms: 5,
        sources: vec![],
    });
    let [RenderAction::FinishTool { output, .. }] = actions.as_slice() else {
        panic!("a finished tool carries both forms");
    };
    let collapsed: Vec<String> = output
        .collapsed
        .iter()
        .map(|line| line.to_string().trim().to_owned())
        .collect();
    assert_eq!(
        collapsed,
        [
            "\u{2022} List files  5 ms",
            "... 3 earlier lines (ctrl+o to expand)",
            "d",
            "e",
            "f",
            "g",
            "h"
        ]
    );
}

#[test]
fn cost_rounds_instead_of_truncating() {
    use rust_decimal::Decimal;

    let cost = Decimal::new(36_6, 5); // $0.00366
    assert_eq!(format_cost(cost), "$0.0037");
    assert_eq!(format_cost(Decimal::new(4, 3)), "$0.0040");
    assert_eq!(format_cost(Decimal::new(1_234_5, 5)), "$0.123");
    assert_eq!(format_cost(Decimal::new(12_345, 3)), "$12.35");
}

#[test]
fn finished_thinking_names_its_key_and_leaves_one_blank_line() {
    let mut renderer = new_renderer();
    for event in [
        AgentEvent::ReasoningDelta("one thought".into()),
        AgentEvent::ToolStart {
            name: "exec".into(),
            id: "1".into(),
            humanized: "List files".into(),
        },
    ] {
        let actions = renderer.render(&event);
        renderer.record_actions(&actions);
    }
    let text: Vec<String> = renderer
        .lines()
        .iter()
        .map(|line| line.to_string())
        .collect();
    assert!(
        text.contains(&"> thinking, 1 line (ctrl+t to expand)".to_owned()),
        "{text:#?}"
    );
    assert!(
        !text
            .windows(2)
            .any(|pair| pair.iter().all(|line| line.is_empty())),
        "{text:#?}"
    );
}

#[test]
fn reply_after_thinking_starts_after_one_blank_line() {
    let mut renderer = new_renderer();
    for event in [
        AgentEvent::ReasoningDelta("one thought".into()),
        AgentEvent::TextDelta("The answer.".into()),
        AgentEvent::Done(String::new()),
    ] {
        let actions = renderer.render(&event);
        renderer.record_actions(&actions);
    }
    let text: Vec<String> = renderer
        .lines()
        .iter()
        .map(|line| line.to_string())
        .collect();
    let label = text
        .iter()
        .position(|line| line.starts_with("> thinking"))
        .expect("thinking block is shown");
    assert_eq!(text[label + 1], "", "{text:#?}");
    assert!(text[label + 2].contains("The answer."), "{text:#?}");
}
