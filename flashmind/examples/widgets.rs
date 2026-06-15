//! Interactive demo of all flashmind-tui widgets.
//!
//! Simulates a full agent session: input → thinking → streaming response →
//! plan approval → choice selection → animated agent tree.
//!
//! ```sh
//! cargo run -p flashmind --example widgets
//! ```

use std::io;
use std::time::Duration;

use async_stream::stream;
use ratatui::crossterm::event::{self, Event};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use flashmind_tui::widgets::*;
use flashmind_tui::{Tui, styles};
use flashmind_types::AgentEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut tui = Tui::new();

    // Greeting
    tui.println(&Line::from(Span::styled(
        "flashmind widget demo — type anything to begin",
        styles::S_AGENT,
    )))?;

    // Phase 1: Read input via Repl
    let config = ReplConfig {
        prompt: "▸".into(),
        ..Default::default()
    };
    let mut repl = Repl::new(config);
    let ReplEvent::UserInput(_text, _images) = repl.read_input()? else {
        return Ok(());
    };

    // Phase 2: Thinking spinner → streaming text response
    phase_header(&mut tui, "Streaming Response")?;

    let stream = fake_agent_stream();
    let cancel = tokio_util::sync::CancellationToken::new();
    repl.stream_response(cancel, Box::pin(stream)).await?;

    // Phase 3: Interactive plan picker
    phase_header(&mut tui, "Plan Approval")?;
    let plan_result = run_plan_picker(&mut tui)?;
    tui.println(&Line::from(Span::styled(
        format!(
            "  → {}",
            if plan_result.approved {
                "Plan approved"
            } else {
                "Plan rejected"
            }
        ),
        styles::S_AGENT,
    )))?;

    // Phase 4: Interactive choice picker
    phase_header(&mut tui, "Choice Selection")?;
    match run_choice_picker(&mut tui)? {
        Some(resp) => {
            tui.println(&Line::from(Span::styled(
                format!("  → Selected: {}", resp.label),
                styles::S_AGENT,
            )))?;
        }
        None => {
            tui.println(&Line::from(Span::styled("  → Cancelled", styles::S_DIM)))?;
        }
    }

    // Phase 5: Animated agent tree
    phase_header(&mut tui, "Subagent Progress")?;
    run_animated_tree(&mut tui).await?;

    // Phase 6: Interactive dropdown
    phase_header(&mut tui, "File Dropdown")?;
    match run_dropdown(&mut tui)? {
        Some(file) => {
            tui.println(&Line::from(Span::styled(
                format!("  → Selected: {file}"),
                styles::S_AGENT,
            )))?;
        }
        None => {
            tui.println(&Line::from(Span::styled("  → Cancelled", styles::S_DIM)))?;
        }
    }

    tui.println(&Line::default())?;
    tui.println(&Line::from(Span::styled("Demo complete.", styles::S_AGENT)))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers

fn phase_header(tui: &mut Tui, title: &str) -> io::Result<()> {
    tui.println(&Line::default())?;
    tui.println(&Line::from(Span::styled(
        format!("── {title} ──"),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )))
}

// ---------------------------------------------------------------------------
// Phase 2: Fake agent stream

fn fake_agent_stream() -> impl futures::Stream<Item = AgentEvent> {
    stream! {
        tokio::time::sleep(Duration::from_secs(1)).await;

        let words = [
            "I'll ", "analyze ", "your ", "request ", "and ", "break ", "it ",
            "down ", "into ", "steps.\n\n",
            "First, ", "let ", "me ", "check ", "the ", "existing ",
            "implementation. ", "Then ", "I'll ", "propose ", "a ", "plan.\n\n",
        ];
        for word in &words {
            yield AgentEvent::TextDelta(word.to_string());
            tokio::time::sleep(Duration::from_millis(60)).await;
        }

        yield AgentEvent::ToolStart {
            name: "read_file".into(),
            id: "t1".into(),
            humanized: "src/auth/mod.rs".into(),
        };
        tokio::time::sleep(Duration::from_millis(800)).await;
        yield AgentEvent::ToolResult {
            name: "read_file".into(),
            id: "t1".into(),
            output: "142 lines".into(),
            success: true,
            elapsed_ms: 800,
            sources: vec![],
        };

        let words2 = [
            "I've ", "reviewed ", "the ", "auth ", "module. ",
            "Here's ", "my ", "plan:\n\n",
        ];
        for word in &words2 {
            yield AgentEvent::TextDelta(word.to_string());
            tokio::time::sleep(Duration::from_millis(60)).await;
        }

        yield AgentEvent::Done("I've reviewed the auth module. Here's my plan:".into());
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Interactive plan picker

fn run_plan_picker(tui: &mut Tui) -> io::Result<PlanResponse> {
    let mut picker = PlanPicker::new(
        "Refactor auth module".into(),
        vec![
            PlanStep {
                id: "1".into(),
                description: "Extract JWT validation into its own crate".into(),
            },
            PlanStep {
                id: "2".into(),
                description: "Add refresh token rotation".into(),
            },
            PlanStep {
                id: "3".into(),
                description: "Write integration tests for OAuth flow".into(),
            },
        ],
    );

    let width = tui.width()?;
    let _raw = tui.raw_mode()?;
    let mut drawn = tui.draw_lines(&picker.lines(width))?;

    loop {
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };

        if let Some(action) = picker.handle_key(key) {
            tui.erase(drawn)?;
            drop(_raw);
            return Ok(match action {
                PlanPickerAction::Approve(r) | PlanPickerAction::Reject(r) => r,
            });
        }

        tui.erase(drawn)?;
        drawn = tui.draw_lines(&picker.lines(width))?;
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Interactive choice picker

fn run_choice_picker(tui: &mut Tui) -> io::Result<Option<ChoiceResponse>> {
    let mut picker = ChoicePicker::new(
        "How should we handle the migration?".into(),
        vec![
            ChoiceOption {
                label: "Run migration now".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "Schedule for off-peak hours".into(),
                accepts_input: false,
            },
            ChoiceOption {
                label: "Custom time".into(),
                accepts_input: true,
            },
        ],
    );

    let _raw = tui.raw_mode()?;
    let mut drawn = tui.draw_lines(&picker.lines())?;

    loop {
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };

        if let Some(action) = picker.handle_key(key) {
            tui.erase(drawn)?;
            drop(_raw);
            return Ok(match action {
                ChoicePickerAction::Select(r) => Some(r),
                ChoicePickerAction::Cancel => None,
            });
        }

        tui.erase(drawn)?;
        drawn = tui.draw_lines(&picker.lines())?;
    }
}

// ---------------------------------------------------------------------------
// Phase 5: Animated tree

async fn run_animated_tree(tui: &mut Tui) -> io::Result<()> {
    const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

    struct Node {
        label: String,
        status: Status,
        children: Vec<Node>,
    }

    enum Status {
        Pending,
        Running(usize),
        Done(u64),
        Failed(String),
    }

    impl Node {
        fn new(label: &str) -> Self {
            Self {
                label: label.into(),
                status: Status::Pending,
                children: vec![],
            }
        }

        fn spans(&self) -> Vec<Span<'static>> {
            match &self.status {
                Status::Pending => vec![
                    Span::styled("○ ", styles::S_DIM),
                    Span::styled(self.label.clone(), styles::S_DIM),
                ],
                Status::Running(tick) => vec![
                    Span::styled(
                        format!("{} ", SPINNER[tick % SPINNER.len()]),
                        styles::S_TOOL_RUN,
                    ),
                    Span::raw(self.label.clone()),
                ],
                Status::Done(ms) => vec![
                    Span::styled("✓ ", styles::S_TOOL_OK),
                    Span::raw(self.label.clone()),
                    Span::styled(format!(" ({ms}ms)"), styles::S_DIM),
                ],
                Status::Failed(msg) => vec![
                    Span::styled("✗ ", styles::S_TOOL_FAIL),
                    Span::raw(self.label.clone()),
                    Span::styled(format!(" — {msg}"), styles::S_TOOL_FAIL),
                ],
            }
        }

        fn to_tree_item(&self) -> TreeItem {
            TreeItem::branch(
                self.spans(),
                self.children.iter().map(|c| c.to_tree_item()).collect(),
            )
        }

        fn tick_spinners(&mut self, tick: usize) {
            if let Status::Running(_) = self.status {
                self.status = Status::Running(tick);
            }
            for child in &mut self.children {
                child.tick_spinners(tick);
            }
        }
    }

    fn at<'a>(nodes: &'a mut [Node], path: &[usize]) -> &'a mut Node {
        let mut node = &mut nodes[path[0]];
        for &i in &path[1..] {
            node = &mut node.children[i];
        }
        node
    }

    enum Event {
        Start(&'static [usize]),
        Done(&'static [usize], u64),
        Fail(&'static [usize], &'static str),
        AddChild(&'static [usize], &'static str),
    }

    // Nodes — initial structure (agents only, tools added dynamically)
    let mut nodes = vec![
        Node::new("analyze codebase"),
        Node::new("refactor auth module"),
        Node::new("run test suite"),
    ];

    // Timeline: (absolute_ms, event)
    // Concurrent execution: agent 0 runs first, then agents 1 & 2 run in parallel
    let timeline: Vec<(u64, Event)> = vec![
        // --- Agent 0: analyze codebase ---
        (0, Event::Start(&[0])),
        (0, Event::AddChild(&[0], "read src/auth/mod.rs")),
        (0, Event::Start(&[0, 0])),
        (400, Event::Done(&[0, 0], 400)),
        (400, Event::AddChild(&[0], "read src/auth/jwt.rs")),
        (400, Event::Start(&[0, 1])),
        (700, Event::Done(&[0, 1], 300)),
        (700, Event::AddChild(&[0], "grep -r \"pub fn\" src/auth/")),
        (700, Event::Start(&[0, 2])),
        (1000, Event::Done(&[0, 2], 300)),
        (1000, Event::Done(&[0], 1000)),
        // --- Agent 1 & 2 start concurrently ---
        (1100, Event::Start(&[1])),
        (1100, Event::Start(&[2])),
        // Agent 1: refactor auth module (tool calls)
        (1100, Event::AddChild(&[1], "edit src/auth/mod.rs")),
        (1100, Event::Start(&[1, 0])),
        (1700, Event::Done(&[1, 0], 600)),
        (1700, Event::AddChild(&[1], "edit src/auth/jwt.rs")),
        (1700, Event::Start(&[1, 1])),
        (2200, Event::Done(&[1, 1], 500)),
        (2200, Event::AddChild(&[1], "edit src/auth/middleware.rs")),
        (2200, Event::Start(&[1, 2])),
        (2700, Event::Done(&[1, 2], 500)),
        (2700, Event::Done(&[1], 1600)),
        // Agent 2: run tests (concurrent sub-commands)
        (1100, Event::AddChild(&[2], "cargo test --lib")),
        (1100, Event::Start(&[2, 0])),
        (1800, Event::Done(&[2, 0], 700)),
        (1800, Event::AddChild(&[2], "cargo test --integration")),
        (1800, Event::Start(&[2, 1])),
        (2900, Event::Fail(&[2, 1], "1 failed")),
        // Retry after refactor finishes
        (
            3000,
            Event::AddChild(&[2], "cargo test --integration (retry)"),
        ),
        (3000, Event::Start(&[2, 2])),
        (3700, Event::Done(&[2, 2], 700)),
        (3700, Event::Done(&[2], 2600)),
    ];

    let mut tick: usize = 0;
    let mut elapsed_ms: u64 = 0;
    let mut event_idx = 0;

    let tree = Tree::new(nodes.iter().map(|n| n.to_tree_item()).collect());
    let mut drawn = tui.draw_lines(&tree.lines())?;

    while event_idx < timeline.len() {
        tokio::time::sleep(Duration::from_millis(80)).await;
        elapsed_ms += 80;
        tick += 1;

        // Apply all events whose time has come
        while event_idx < timeline.len() && timeline[event_idx].0 <= elapsed_ms {
            match &timeline[event_idx].1 {
                Event::Start(path) => {
                    at(&mut nodes, path).status = Status::Running(tick);
                }
                Event::Done(path, ms) => {
                    at(&mut nodes, path).status = Status::Done(*ms);
                }
                Event::Fail(path, msg) => {
                    at(&mut nodes, path).status = Status::Failed(msg.to_string());
                }
                Event::AddChild(path, label) => {
                    at(&mut nodes, path).children.push(Node::new(label));
                }
            }
            event_idx += 1;
        }

        for node in &mut nodes {
            node.tick_spinners(tick);
        }

        tui.erase(drawn)?;
        let tree = Tree::new(nodes.iter().map(|n| n.to_tree_item()).collect());
        drawn = tui.draw_lines(&tree.lines())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 6: Interactive dropdown

fn run_dropdown(tui: &mut Tui) -> io::Result<Option<String>> {
    let width = tui.width()?;

    let mut dropdown = Dropdown::new(
        "files",
        vec![
            "src/main.rs".into(),
            "src/lib.rs".into(),
            "src/widgets/mod.rs".into(),
            "src/widgets/tree.rs".into(),
            "src/widgets/dropdown.rs".into(),
            "src/widgets/status_bar.rs".into(),
            "src/widgets/plan_picker.rs".into(),
            "src/widgets/choice_picker.rs".into(),
            "Cargo.toml".into(),
            "README.md".into(),
        ],
    );

    let _raw = tui.raw_mode()?;
    let max_w = width.min(45);
    let mut drawn = tui.draw_lines(&dropdown.lines(6, max_w))?;

    loop {
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };

        if let Some(action) = dropdown.handle_key(key) {
            tui.erase(drawn)?;
            drop(_raw);
            return Ok(match action {
                DropdownAction::Select(s) => Some(s),
                DropdownAction::Cancel => None,
            });
        }

        tui.erase(drawn)?;
        drawn = tui.draw_lines(&dropdown.lines(6, max_w))?;
    }
}
