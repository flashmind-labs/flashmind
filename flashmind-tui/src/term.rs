//! Ratatui terminal ownership and fullscreen rendering.

use std::collections::BTreeSet;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use ratatui::Frame;
use ratatui::crossterm::{
    cursor::Show,
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::{Constraint, Layout};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::event_render::{ReasoningOutput, ToolActivity, ToolOutput};
use crate::present::Presenter;
use crate::selection::{ScreenSelection, SelectionAction};
use crate::styles::S_DIM;
use crate::widgets::repl::copy_to_clipboard;

/// Shared fullscreen terminal and transcript. Clones use the same screen.
#[derive(Clone, Default)]
pub struct Tui {
    state: Arc<Mutex<ScreenState>>,
}

#[derive(Default)]
struct ScreenState {
    terminal: Option<Fullscreen>,
    /// The transcript as currently displayed.
    lines: Vec<Line<'static>>,
    /// Blocks in `lines` that can switch between collapsed and expanded.
    folds: Vec<Fold>,
    tools_expanded: bool,
    tool_batch: Option<usize>,
    tool_preview: Option<usize>,
}

struct Fold {
    /// Index of the block's first line in `lines`.
    start: usize,
    output: FoldOutput,
    expanded: bool,
}

enum FoldOutput {
    Tool(ToolBatch),
    Reasoning(ReasoningOutput),
}

#[derive(Default)]
struct ToolBatch {
    read: BTreeSet<PathBuf>,
    changed: BTreeSet<PathBuf>,
    commands: usize,
    searches: usize,
    other: usize,
    failed: usize,
    output: ToolOutput,
}

impl ToolBatch {
    fn push(&mut self, output: &ToolOutput) {
        match &output.activity {
            ToolActivity::Read(path) => {
                self.read.insert(path.clone());
            }
            ToolActivity::Changed(paths) => self.changed.extend(paths.iter().cloned()),
            ToolActivity::Command => self.commands += 1,
            ToolActivity::Search => self.searches += 1,
            ToolActivity::Other => self.other += 1,
        }
        self.failed += usize::from(output.failed);
        let summary = self.summary();
        self.output.collapsed = vec![Line::from(Span::styled(format!("> {summary}"), S_DIM))];
        if matches!(output.activity, ToolActivity::Command) {
            self.output.collapsed.extend_from_slice(&output.collapsed);
        }
        let header = Line::from(Span::styled(format!("v {summary}"), S_DIM));
        if self.output.expanded.is_empty() {
            self.output.expanded.push(header);
        } else {
            self.output.expanded[0] = header;
        }
        self.output.expanded.extend_from_slice(&output.expanded);
    }

    fn summary(&self) -> String {
        let mut parts = Vec::new();
        let mut add = |verb: &str, count: usize, noun: &str| {
            if count > 0 {
                parts.push(format!(
                    "{verb} {count} {noun}{}",
                    if count == 1 { "" } else { "s" }
                ));
            }
        };
        add("read", self.read.len(), "file");
        add("changed", self.changed.len(), "file");
        add("ran", self.commands, "command");
        add("searched", self.searches, "time");
        if self.other > 0 {
            let other = if parts.is_empty() { "" } else { "other " };
            parts.push(format!(
                "ran {} {other}tool{}",
                self.other,
                if self.other == 1 { "" } else { "s" }
            ));
        }
        let mut summary = parts.join(", ");
        if let Some(first) = summary.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        if self.failed > 0 {
            summary.push_str(&format!(" ({} failed)", self.failed));
        }
        summary
    }
}

impl FoldOutput {
    fn lines(&self, expanded: bool) -> &[Line<'static>] {
        match self {
            Self::Tool(batch) => batch.output.lines(expanded),
            Self::Reasoning(output) => output.lines(expanded),
        }
    }
}

impl ScreenState {
    fn hide_tool_preview(&mut self) -> bool {
        let Some(index) = self.tool_preview.take() else {
            return false;
        };
        let mut current = 0;
        self.rewrite_folds(|fold| {
            if current == index
                && let FoldOutput::Tool(batch) = &mut fold.output
            {
                batch.output.collapsed.truncate(1);
            }
            current += 1;
        });
        true
    }

    fn set_tools_expanded(&mut self, expanded: bool) {
        self.rewrite_folds(|fold| {
            if matches!(fold.output, FoldOutput::Tool(_)) {
                fold.expanded = expanded;
            }
        });
        self.tools_expanded = expanded;
    }

    fn rewrite_folds(&mut self, mut update: impl FnMut(&mut Fold)) {
        let mut old = std::mem::take(&mut self.lines).into_iter();
        let mut cursor = 0;
        for fold in &mut self.folds {
            self.lines.extend(old.by_ref().take(fold.start - cursor));
            let skipped = fold.output.lines(fold.expanded).len();
            old.by_ref().take(skipped).for_each(drop);
            cursor = fold.start + skipped;
            fold.start = self.lines.len();
            update(fold);
            self.lines
                .extend(fold.output.lines(fold.expanded).iter().cloned());
        }
        self.lines.extend(old);
    }
}

impl Tui {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_state(&self) -> io::Result<MutexGuard<'_, ScreenState>> {
        self.state
            .lock()
            .map_err(|_| io::Error::other("terminal state lock poisoned"))
    }

    #[cfg(test)]
    pub(crate) fn transcript_lines(&self) -> Vec<Line<'static>> {
        self.lock_state().unwrap().lines.clone()
    }

    pub fn size(&self) -> io::Result<(u16, u16)> {
        ratatui::crossterm::terminal::size()
    }

    pub fn width(&self) -> io::Result<u16> {
        self.size().map(|(width, _)| width)
    }

    /// Suspend the chat screen while a menu owns the terminal.
    pub fn fullscreen(&self) -> io::Result<Fullscreen> {
        self.suspend()?;
        Fullscreen::new(true)
    }

    pub fn suspend(&self) -> io::Result<()> {
        self.lock_state()?.terminal.take();
        Ok(())
    }

    pub fn append_lines(&self, lines: &[Line<'static>]) -> io::Result<()> {
        self.lock_state()?.lines.extend_from_slice(lines);
        Ok(())
    }

    /// Append a finished tool block, shown in the current display mode.
    pub fn append_tool_output(&self, output: ToolOutput) -> io::Result<()> {
        let mut state = self.lock_state()?;
        state.hide_tool_preview();
        let preview = matches!(output.activity, ToolActivity::Command);
        if let Some(index) = state.tool_batch {
            let mut current = 0;
            state.rewrite_folds(|fold| {
                if current == index
                    && let FoldOutput::Tool(batch) = &mut fold.output
                {
                    batch.push(&output);
                }
                current += 1;
            });
            state.tool_preview = preview.then_some(index);
            return Ok(());
        }
        let start = state.lines.len();
        let mut batch = ToolBatch::default();
        batch.push(&output);
        let expanded = state.tools_expanded;
        state.lines.extend_from_slice(batch.output.lines(expanded));
        state.tool_batch = Some(state.folds.len());
        state.tool_preview = preview.then_some(state.folds.len());
        state.folds.push(Fold {
            start,
            output: FoldOutput::Tool(batch),
            expanded,
        });
        Ok(())
    }

    /// Collapse the previous command preview when new activity resumes.
    pub(crate) fn hide_tool_preview(&self) -> io::Result<bool> {
        Ok(self.lock_state()?.hide_tool_preview())
    }

    /// Assistant text or a user message separates consecutive tool batches.
    pub fn end_tool_batch(&self) -> io::Result<()> {
        self.lock_state()?.tool_batch = None;
        Ok(())
    }

    /// Reveal retained tool output matching a transcript search.
    pub(crate) fn reveal_tool_matches(&self, query: &str) -> io::Result<bool> {
        if query.is_empty() {
            return Ok(false);
        }
        let query = query.to_lowercase();
        let mut changed = false;
        self.lock_state()?.rewrite_folds(|fold| {
            if !fold.expanded
                && let FoldOutput::Tool(batch) = &fold.output
                && batch
                    .output
                    .expanded
                    .iter()
                    .any(|line| line.to_string().to_lowercase().contains(&query))
            {
                fold.expanded = true;
                changed = true;
            }
        });
        Ok(changed)
    }

    /// Append a reasoning trace with its full text retained for toggling.
    pub fn append_reasoning_output(
        &self,
        output: ReasoningOutput,
        expanded: bool,
    ) -> io::Result<()> {
        let mut state = self.lock_state()?;
        let start = state.lines.len();
        state.lines.extend_from_slice(output.lines(expanded));
        state.folds.push(Fold {
            start,
            output: FoldOutput::Reasoning(output),
            expanded,
        });
        Ok(())
    }

    /// Expand or collapse every reasoning block already in the transcript.
    pub fn set_reasoning_expanded(&self, expanded: bool) -> io::Result<()> {
        self.lock_state()?.rewrite_folds(|fold| {
            if matches!(fold.output, FoldOutput::Reasoning(_)) {
                fold.expanded = expanded;
            }
        });
        Ok(())
    }

    pub(crate) fn toggle_fold_at(&self, entry: usize) -> io::Result<bool> {
        let mut state = self.lock_state()?;
        let Some(index) = state.folds.iter().position(|fold| fold.start == entry) else {
            return Ok(false);
        };
        let mut current = 0;
        state.rewrite_folds(|fold| {
            if current == index {
                fold.expanded = !fold.expanded;
            }
            current += 1;
        });
        Ok(true)
    }

    /// Switch every tool block in the transcript between collapsed and
    /// expanded output.
    pub fn set_tools_expanded(&self, expanded: bool) -> io::Result<()> {
        self.lock_state()?.set_tools_expanded(expanded);
        Ok(())
    }

    pub fn clear_transcript(&self) -> io::Result<()> {
        let mut state = self.lock_state()?;
        state.lines.clear();
        state.folds.clear();
        state.tool_batch = None;
        state.tool_preview = None;
        Ok(())
    }

    pub fn draw(&self, render: impl FnOnce(&mut Frame<'_>, &[Line<'static>])) -> io::Result<()> {
        let mut state = self.lock_state()?;
        if state.terminal.is_none() {
            state.terminal = Some(Fullscreen::new(false)?);
        }
        let ScreenState {
            terminal, lines, ..
        } = &mut *state;
        if let Some(terminal) = terminal {
            terminal.draw(|frame| render(frame, lines))?;
        }
        Ok(())
    }

    /// Add a styled line to the transcript and redraw its visible tail.
    pub fn println(&mut self, line: &Line<'_>) -> io::Result<()> {
        let owned = Line {
            spans: line
                .spans
                .iter()
                .map(|span| Span::styled(span.content.to_string(), span.style))
                .collect(),
            style: line.style,
            alignment: line.alignment,
        };
        self.append_lines(&[owned])?;
        self.draw(|frame, lines| {
            let area = frame.area();
            let paragraph = Paragraph::new(lines.to_vec()).wrap(Wrap { trim: false });
            let scroll = paragraph
                .line_count(area.width.max(1))
                .saturating_sub(area.height as usize)
                .min(u16::MAX as usize) as u16;
            frame.render_widget(paragraph.scroll((scroll, 0)), area);
        })
    }
}

/// Fullscreen renderer. Restores mouse, cursor, screen, and input modes on drop.
pub struct Fullscreen {
    presenter: Presenter<io::Stdout>,
    selection: Option<ScreenSelection>,
    _screen: AlternateScreen,
    _raw: crate::widgets::RawModeGuard,
}

struct AlternateScreen;

impl Drop for AlternateScreen {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            LeaveAlternateScreen,
            Show
        );
    }
}

impl Fullscreen {
    pub fn read_event(&mut self) -> io::Result<ratatui::crossterm::event::Event> {
        use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, MouseEventKind};
        let event = event::read()?;
        if let Some(action) = self
            .selection
            .as_mut()
            .and_then(|selection| selection.handle_event(&event))
        {
            if let SelectionAction::Copy(text) = action {
                copy_to_clipboard(&text).map_err(io::Error::other)?;
            }
            // Menu loops ignore focus events and redraw before the next read.
            return Ok(Event::FocusGained);
        }
        Ok(match event {
            Event::Mouse(mouse) if mouse.kind == MouseEventKind::ScrollUp => {
                Event::Key(KeyEvent::from(KeyCode::Up))
            }
            Event::Mouse(mouse) if mouse.kind == MouseEventKind::ScrollDown => {
                Event::Key(KeyEvent::from(KeyCode::Down))
            }
            event => event,
        })
    }
    fn new(selectable: bool) -> io::Result<Self> {
        let raw = crate::widgets::RawModeGuard::enable()?;
        let screen = AlternateScreen;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        Ok(Self {
            presenter: Presenter::new(io::stdout())?,
            selection: selectable.then(ScreenSelection::default),
            _screen: screen,
            _raw: raw,
        })
    }

    pub fn draw(&mut self, render: impl FnOnce(&mut Frame<'_>)) -> io::Result<()> {
        let size = ratatui::crossterm::terminal::size()?;
        let selection = &mut self.selection;
        self.presenter.draw(size, |frame| {
            render(frame);
            if let Some(selection) = selection {
                selection.render(frame.buffer_mut());
            }
        })
    }

    pub fn draw_lines(&mut self, lines: Vec<Line<'static>>) -> io::Result<()> {
        self.draw(|frame| frame.render_widget(Paragraph::new(lines), frame.area()))
    }

    pub fn draw_list(
        &mut self,
        lines: Vec<Line<'static>>,
        selected: usize,
        header: u16,
        footer: u16,
    ) -> io::Result<()> {
        self.draw(|frame| render_menu_list(frame, lines, selected, header, footer))
    }
}

fn render_menu_list(
    frame: &mut Frame<'_>,
    lines: Vec<Line<'static>>,
    selected: usize,
    header: u16,
    footer: u16,
) {
    let [top, body, bottom] = Layout::vertical([
        Constraint::Length(header),
        Constraint::Min(1),
        Constraint::Length(footer),
    ])
    .areas(frame.area());
    let start = (header as usize).min(lines.len());
    let end = lines.len().saturating_sub(footer as usize).max(start);
    let visible = body.height as usize;
    let scroll = selected
        .saturating_sub(start)
        .saturating_sub(visible / 2)
        .min((end - start).saturating_sub(visible));
    frame.render_widget(Paragraph::new(lines[..start].to_vec()), top);
    frame.render_widget(
        Paragraph::new(lines[start..end].to_vec())
            .scroll((scroll.min(u16::MAX as usize) as u16, 0)),
        body,
    );
    frame.render_widget(Paragraph::new(lines[end..].to_vec()), bottom);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn fold(name: &str) -> ToolOutput {
        ToolOutput {
            collapsed: vec![Line::from(name.to_owned()), Line::from("hint")],
            expanded: vec![
                Line::from(name.to_owned()),
                Line::from("a"),
                Line::from("b"),
                Line::from("c"),
            ],
            ..ToolOutput::default()
        }
    }

    fn text(state: &ScreenState) -> Vec<String> {
        state.lines.iter().map(Line::to_string).collect()
    }

    #[test]
    fn consecutive_tools_form_one_batch_until_assistant_text() {
        let tui = Tui::new();
        tui.append_lines(&[Line::from("before")]).unwrap();
        tui.append_tool_output(fold("one")).unwrap();
        tui.append_tool_output(fold("two")).unwrap();
        tui.end_tool_batch().unwrap();
        tui.append_lines(&[Line::from("reply")]).unwrap();
        tui.append_tool_output(fold("three")).unwrap();
        let collapsed = text(&tui.lock_state().unwrap());
        assert_eq!(
            collapsed,
            ["before", "> Ran 2 tools", "reply", "> Ran 1 tool"]
        );
        assert!(tui.toggle_fold_at(1).unwrap());
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            [
                "before",
                "v Ran 2 tools",
                "one",
                "a",
                "b",
                "c",
                "two",
                "a",
                "b",
                "c",
                "reply",
                "> Ran 1 tool"
            ]
        );
        // Global collapse also overrides batches opened by clicking.
        tui.set_tools_expanded(false).unwrap();
        assert_eq!(text(&tui.lock_state().unwrap()), collapsed);
        tui.set_tools_expanded(true).unwrap();
        assert!(text(&tui.lock_state().unwrap()).contains(&"three".into()));
        tui.set_tools_expanded(false).unwrap();
        assert_eq!(text(&tui.lock_state().unwrap()), collapsed);
    }

    #[test]
    fn batch_growth_preserves_later_lines_and_other_fold_positions() {
        let tui = Tui::new();
        tui.append_tool_output(fold("one")).unwrap();
        tui.append_lines(&[Line::from("status")]).unwrap();
        tui.append_reasoning_output(
            ReasoningOutput {
                collapsed: vec![Line::from("> thinking")],
                expanded: vec![Line::from("v thinking"), Line::from("trace")],
            },
            false,
        )
        .unwrap();
        tui.append_tool_output(fold("two")).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            ["> Ran 2 tools", "status", "> thinking"]
        );
        assert!(tui.toggle_fold_at(0).unwrap());
        assert!(tui.toggle_fold_at(10).unwrap());
        assert_eq!(text(&tui.lock_state().unwrap()).last().unwrap(), "trace");
        tui.set_tools_expanded(false).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            ["> Ran 2 tools", "status", "v thinking", "trace"]
        );
        tui.set_reasoning_expanded(false).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            ["> Ran 2 tools", "status", "> thinking"]
        );
        tui.clear_transcript().unwrap();
        assert!(!tui.toggle_fold_at(0).unwrap());
    }

    #[test]
    fn search_reveals_hidden_tool_output() {
        let tui = Tui::new();
        tui.append_tool_output(fold("needle")).unwrap();
        assert!(tui.reveal_tool_matches("NEEDLE").unwrap());
        assert!(text(&tui.lock_state().unwrap()).contains(&"needle".into()));
        assert!(!tui.reveal_tool_matches("NEEDLE").unwrap());
        tui.set_tools_expanded(false).unwrap();
        assert!(!tui.reveal_tool_matches("missing").unwrap());
    }

    #[test]
    fn activity_summary_deduplicates_files_and_includes_failures_and_other_tools() {
        let mut batch = ToolBatch::default();
        for activity in [
            ToolActivity::Read(PathBuf::from("example.rs")),
            ToolActivity::Read(PathBuf::from("example.rs")),
            ToolActivity::Read(PathBuf::from("other.rs")),
            ToolActivity::Changed(BTreeSet::from([PathBuf::from("example.rs")])),
            ToolActivity::Command,
            ToolActivity::Search,
            ToolActivity::Other,
        ] {
            batch.push(&ToolOutput {
                activity,
                ..ToolOutput::default()
            });
        }
        batch.push(&ToolOutput {
            activity: ToolActivity::Command,
            failed: true,
            ..ToolOutput::default()
        });
        assert_eq!(
            batch.summary(),
            "Read 2 files, changed 1 file, ran 2 commands, searched 1 time, ran 1 other tool (1 failed)"
        );
    }

    #[test]
    fn menu_scroll_keeps_selection_and_controls_visible() {
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        let mut lines = vec![Line::from("Menu"), Line::from("Search"), Line::default()];
        lines.extend((0..100).map(|index| Line::from(format!("item {index}"))));
        lines.extend([Line::default(), Line::from("Enter: select")]);
        terminal
            .draw(|frame| render_menu_list(frame, lines, 93, 3, 2))
            .unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(screen.contains("Menu"));
        assert!(screen.contains("item 90"));
        assert!(screen.contains("Enter: select"));
        assert!(!screen.contains("item 0"));
    }
}
