//! Ratatui terminal ownership and fullscreen rendering.

use std::io;
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

use crate::event_render::{ReasoningOutput, ToolOutput};
use crate::present::Presenter;
use crate::selection::{ScreenSelection, SelectionAction};
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
}

struct Fold {
    /// Index of the block's first line in `lines`.
    start: usize,
    output: FoldOutput,
    expanded: bool,
}

enum FoldOutput {
    Tool(ToolOutput),
    Reasoning(ReasoningOutput),
}

impl FoldOutput {
    fn lines(&self, expanded: bool) -> &[Line<'static>] {
        match self {
            Self::Tool(output) => output.lines(expanded),
            Self::Reasoning(output) => output.lines(expanded),
        }
    }
}

impl ScreenState {
    fn set_tools_expanded(&mut self, expanded: bool) {
        if expanded == self.tools_expanded {
            return;
        }
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
        let start = state.lines.len();
        let lines = output.lines(state.tools_expanded).to_vec();
        state.lines.extend(lines);
        let expanded = state.tools_expanded;
        state.folds.push(Fold {
            start,
            output: FoldOutput::Tool(output),
            expanded,
        });
        Ok(())
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

    pub(crate) fn toggle_reasoning_at(&self, entry: usize) -> io::Result<bool> {
        let mut state = self.lock_state()?;
        let Some(index) = state.folds.iter().position(|fold| {
            fold.start == entry && matches!(fold.output, FoldOutput::Reasoning(_))
        }) else {
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
        }
    }

    fn text(state: &ScreenState) -> Vec<String> {
        state.lines.iter().map(Line::to_string).collect()
    }

    #[test]
    fn toggling_tool_output_rewrites_every_block_in_place() {
        let tui = Tui::new();
        tui.append_lines(&[Line::from("before")]).unwrap();
        tui.append_tool_output(fold("one")).unwrap();
        tui.append_lines(&[Line::from("middle")]).unwrap();
        tui.append_tool_output(fold("two")).unwrap();
        tui.append_lines(&[Line::from("after")]).unwrap();
        let collapsed = text(&tui.lock_state().unwrap());
        assert_eq!(
            collapsed,
            ["before", "one", "hint", "middle", "two", "hint", "after"]
        );

        tui.set_tools_expanded(true).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            [
                "before", "one", "a", "b", "c", "middle", "two", "a", "b", "c", "after"
            ]
        );

        tui.append_tool_output(fold("three")).unwrap();
        tui.set_tools_expanded(false).unwrap();
        let mut expected = collapsed;
        expected.extend(["three".into(), "hint".into()]);
        assert_eq!(text(&tui.lock_state().unwrap()), expected);
    }

    #[test]
    fn reasoning_and_tool_folds_keep_their_contents_and_positions() {
        let tui = Tui::new();
        let reasoning = |name: &str| ReasoningOutput {
            collapsed: vec![Line::from(format!("> {name}"))],
            expanded: vec![Line::from(format!("v {name}")), Line::from("full trace")],
        };
        tui.append_lines(&[Line::from("before")]).unwrap();
        tui.append_reasoning_output(reasoning("one"), false)
            .unwrap();
        tui.append_tool_output(fold("tool")).unwrap();
        tui.append_reasoning_output(reasoning("two"), false)
            .unwrap();
        tui.append_lines(&[Line::from("after")]).unwrap();
        let collapsed = text(&tui.lock_state().unwrap());

        assert!(tui.toggle_reasoning_at(1).unwrap());
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            [
                "before",
                "v one",
                "full trace",
                "tool",
                "hint",
                "> two",
                "after"
            ]
        );
        assert!(!tui.toggle_reasoning_at(2).unwrap());
        tui.set_tools_expanded(true).unwrap();
        assert!(tui.toggle_reasoning_at(7).unwrap());
        tui.set_reasoning_expanded(false).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            ["before", "> one", "tool", "a", "b", "c", "> two", "after"]
        );
        tui.set_reasoning_expanded(true).unwrap();
        tui.set_tools_expanded(false).unwrap();
        assert_eq!(
            text(&tui.lock_state().unwrap()),
            [
                "before",
                "v one",
                "full trace",
                "tool",
                "hint",
                "v two",
                "full trace",
                "after"
            ]
        );
        tui.set_reasoning_expanded(false).unwrap();
        assert_eq!(text(&tui.lock_state().unwrap()), collapsed);
        tui.clear_transcript().unwrap();
        assert!(!tui.toggle_reasoning_at(1).unwrap());
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
