//! Frame presentation that scrolls moved rows instead of rewriting them.
//!
//! When the transcript grows, every visible row moves up. A plain cell diff
//! then rewrites the whole screen, which flickers on terminals and
//! multiplexers without synchronized output. The presenter finds the block
//! of rows that only moved, asks the terminal to scroll it, and writes the
//! cells that still differ.

use std::io::{self, Write};
use std::ops::Range;

use ratatui::backend::{Backend, ClearType, CrosstermBackend, WindowSize};
use ratatui::buffer::{Buffer, Cell};
use ratatui::crossterm::queue;
use ratatui::crossterm::terminal::{
    self, BeginSynchronizedUpdate, EndSynchronizedUpdate, ScrollUp,
};
use ratatui::layout::{Position, Rect, Size};
use ratatui::style::Color;
use ratatui::{Frame, Terminal};

use crate::hyperlinks::ScreenLinks;

/// Fewest rows worth scrolling instead of rewriting.
const MIN_SCROLL_ROWS: u16 = 4;

/// Cells a moved row may differ by, such as a scrollbar edge passing it.
const MOVED_ROW_SLACK: usize = 2;

/// Renders frames offscreen and writes the changes to `W`.
pub(crate) struct Presenter<W: Write> {
    offscreen: Terminal<Offscreen>,
    output: CrosstermBackend<W>,
    /// What the terminal shows; `None` until the first frame or after a resize.
    screen: Option<Buffer>,
    links: ScreenLinks,
}

/// Rows of a region that moved up by `amount`.
#[derive(Debug, PartialEq, Eq)]
struct Shift {
    rows: Range<u16>,
    amount: u16,
}

impl<W: Write> Presenter<W> {
    pub(crate) fn new(output: W) -> io::Result<Self> {
        Ok(Self {
            offscreen: Terminal::new(Offscreen::default())?,
            output: CrosstermBackend::new(output),
            screen: None,
            links: ScreenLinks::new(),
        })
    }

    /// Draw one frame at `width` x `height` as a single synchronized update.
    #[cfg(test)]
    pub(crate) fn draw(
        &mut self,
        (width, height): (u16, u16),
        render: impl FnOnce(&mut Frame<'_>),
    ) -> io::Result<()> {
        self.draw_with_links((width, height), |frame| {
            render(frame);
            ScreenLinks::new()
        })
    }

    pub(crate) fn draw_with_links(
        &mut self,
        (width, height): (u16, u16),
        render: impl FnOnce(&mut Frame<'_>) -> ScreenLinks,
    ) -> io::Result<()> {
        let area = Rect::new(0, 0, width, height);
        if self.offscreen.backend().buffer.area != area {
            self.offscreen.backend_mut().buffer = Buffer::empty(area);
            self.screen = None;
            self.links.clear();
        }
        let mut links = ScreenLinks::new();
        self.offscreen.draw(|frame| links = render(frame))?;
        let next = self.offscreen.backend().buffer.clone();

        queue!(self.output, BeginSynchronizedUpdate)?;
        let mut screen = match self.screen.take() {
            Some(mut screen) => {
                if let Some(shift) = find_shift(&screen, &next) {
                    // Scroll only the moved rows, then restore the full region.
                    write!(
                        self.output,
                        "\x1b[{};{}r",
                        shift.rows.start + 1,
                        shift.rows.end
                    )?;
                    queue!(self.output, ScrollUp(shift.amount))?;
                    write!(self.output, "\x1b[r")?;
                    shift_rows(&mut screen, &shift);
                    self.links = self
                        .links
                        .drain()
                        .filter_map(|(mut position, target)| {
                            if shift.rows.contains(&position.y) {
                                if position.y < shift.rows.start + shift.amount {
                                    return None;
                                }
                                position.y -= shift.amount;
                            }
                            Some((position, target))
                        })
                        .collect();
                }
                screen
            }
            None => {
                queue!(self.output, terminal::Clear(terminal::ClearType::All))?;
                Buffer::empty(next.area)
            }
        };
        // Invalidate changed targets before diffing, so Ratatui still handles
        // wide characters and skipped cells when a label needs rewriting.
        for position in self.links.keys().chain(links.keys()) {
            if next.area.contains(*position) && self.links.get(position) != links.get(position) {
                screen[*position].fg = if next[*position].fg == Color::Reset {
                    Color::Black
                } else {
                    Color::Reset
                };
            }
        }
        let changed = screen.diff(&next);
        for run in changed.chunk_by(|a, b| {
            links.get(&Position::new(a.0, a.1)) == links.get(&Position::new(b.0, b.1))
        }) {
            let target = links.get(&Position::new(run[0].0, run[0].1));
            if let Some(target) = target {
                write!(self.output, "\x1b]8;;{target}\x1b\\")?;
            }
            self.output.draw(run.iter().copied())?;
            if target.is_some() {
                write!(self.output, "\x1b]8;;\x1b\\")?;
            }
        }
        match self.offscreen.backend().cursor {
            Some(position) => {
                self.output.set_cursor_position(position)?;
                self.output.show_cursor()?;
            }
            None => self.output.hide_cursor()?,
        }
        queue!(self.output, EndSynchronizedUpdate)?;
        Backend::flush(&mut self.output)?;
        self.screen = Some(next);
        self.links = links;
        Ok(())
    }
}

/// In-memory backend the frame renders into before it is diffed.
struct Offscreen {
    buffer: Buffer,
    /// Cursor position, or `None` while hidden.
    cursor: Option<Position>,
    position: Position,
}

impl Default for Offscreen {
    fn default() -> Self {
        Self {
            buffer: Buffer::empty(Rect::new(0, 0, 1, 1)),
            cursor: None,
            position: Position::ORIGIN,
        }
    }
}

impl Backend for Offscreen {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        for (x, y, cell) in content {
            if self.buffer.area.contains(Position::new(x, y)) {
                self.buffer[(x, y)] = cell.clone();
            }
        }
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        self.cursor = None;
        Ok(())
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        self.cursor = Some(self.position);
        Ok(())
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        Ok(self.position)
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        self.position = position.into();
        if self.cursor.is_some() {
            self.cursor = Some(self.position);
        }
        Ok(())
    }

    fn clear(&mut self) -> io::Result<()> {
        self.buffer.reset();
        Ok(())
    }

    fn clear_region(&mut self, _clear_type: ClearType) -> io::Result<()> {
        self.buffer.reset();
        Ok(())
    }

    fn size(&self) -> io::Result<Size> {
        Ok(self.buffer.area.as_size())
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        Ok(WindowSize {
            columns_rows: self.buffer.area.as_size(),
            pixels: Size::default(),
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn row(buffer: &Buffer, y: u16) -> &[Cell] {
    let width = buffer.area.width as usize;
    let start = y as usize * width;
    &buffer.content[start..start + width]
}

/// The block of rows in `next` that sits `amount` rows higher than in
/// `screen`, for the amount whose block has the most exactly matching rows.
/// Rows that differ in a few cells, such as where the scrollbar thumb
/// moved, still extend the block.
fn find_shift(screen: &Buffer, next: &Buffer) -> Option<Shift> {
    if screen.area != next.area {
        return None;
    }
    let height = next.area.height;
    let mut best: Option<(u16, Shift)> = None;
    for amount in 1..=height / 2 {
        let mut start = 0;
        let mut exact = 0;
        for y in 0..=height - amount {
            if y < height - amount {
                let (now, before) = (row(next, y), row(screen, y + amount));
                if now != row(screen, y) && rows_match(now, before) {
                    exact += u16::from(now == before);
                    continue;
                }
            }
            if exact >= MIN_SCROLL_ROWS && best.as_ref().is_none_or(|(most, _)| exact > *most) {
                best = Some((
                    exact,
                    Shift {
                        rows: start..y + amount,
                        amount,
                    },
                ));
            }
            start = y + 1;
            exact = 0;
        }
    }
    best.map(|(_, shift)| shift)
}

/// Whether two rows are equal apart from a few cells.
fn rows_match(a: &[Cell], b: &[Cell]) -> bool {
    a.iter().zip(b).filter(|(a, b)| a != b).count() <= MOVED_ROW_SLACK
}

/// Apply a terminal scroll to the model of the screen: rows move up and the
/// rows uncovered at the bottom of the region are blank.
fn shift_rows(screen: &mut Buffer, shift: &Shift) {
    let width = screen.area.width as usize;
    let start = shift.rows.start as usize * width;
    let end = shift.rows.end as usize * width;
    let moved = shift.amount as usize * width;
    screen.content[start..end].rotate_left(moved);
    for cell in &mut screen.content[end - moved..end] {
        cell.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::Line;
    use ratatui::widgets::Paragraph;
    use std::sync::{Arc, Mutex};
    use url::Url;

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("the test holds no other lock")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn lines(range: Range<usize>) -> Vec<Line<'static>> {
        range
            .map(|index| Line::from(format!("transcript line {index}")))
            .collect()
    }

    #[test]
    fn a_new_line_scrolls_the_transcript_and_rewrites_one_row() {
        let screen = Buffer::with_lines(lines(0..20));
        let next = Buffer::with_lines(lines(1..21));
        let shift = find_shift(&screen, &next).expect("the transcript moved up");
        assert_eq!(
            shift,
            Shift {
                rows: 0..20,
                amount: 1
            }
        );
        let mut shifted = screen.clone();
        shift_rows(&mut shifted, &shift);
        let changed = shifted.diff(&next);
        assert!(changed.iter().all(|(_, y, _)| *y == 19));
    }

    #[test]
    fn fixed_rows_below_the_transcript_stay_in_place() {
        let mut before = lines(0..16);
        before.extend([Line::from("input"), Line::from("status")]);
        let mut after = lines(2..18);
        after.extend([Line::from("input"), Line::from("status")]);
        let screen = Buffer::with_lines(before);
        let next = Buffer::with_lines(after);
        let shift = find_shift(&screen, &next).unwrap();
        assert_eq!(
            shift,
            Shift {
                rows: 0..16,
                amount: 2
            }
        );
        let mut shifted = screen.clone();
        shift_rows(&mut shifted, &shift);
        assert!(
            shifted
                .diff(&next)
                .iter()
                .all(|(_, y, _)| (14..16).contains(y))
        );
    }

    #[test]
    fn unchanged_frames_do_not_scroll() {
        let screen = Buffer::with_lines(lines(0..20));
        assert_eq!(find_shift(&screen, &screen.clone()), None);
    }

    #[test]
    fn link_labels_emit_osc8_and_target_changes_redraw_them() {
        let sink = Sink::default();
        let mut presenter = Presenter::new(sink.clone()).unwrap();
        for target in ["https://example.com/one", "https://example.com/two"] {
            sink.0.lock().unwrap().clear();
            presenter
                .draw_with_links((20, 3), |frame| {
                    frame.render_widget(Paragraph::new("docs after"), frame.area());
                    (0..4)
                        .map(|x| (Position::new(x, 0), Arc::new(Url::parse(target).unwrap())))
                        .collect()
                })
                .unwrap();
            let output = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
            let open = format!("\x1b]8;;{target}\x1b\\");
            let start = output.find(&open).expect("the label opens its hyperlink");
            let rest = &output[start + open.len()..];
            let end = rest
                .find("\x1b]8;;\x1b\\")
                .expect("the hyperlink is closed");
            assert!(rest[..end].contains("docs"));
            assert!(!rest[..end].contains("after"));
        }
        sink.0.lock().unwrap().clear();
        presenter
            .draw((20, 3), |frame| {
                frame.render_widget(Paragraph::new("docs after"), frame.area());
            })
            .unwrap();
        let output = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        assert!(
            output.contains("docs"),
            "removing a target rewrites its cells"
        );
        assert!(!output.contains("\x1b]8;;https://"));
    }

    #[test]
    fn scrolling_preserves_links_without_rewriting_moved_labels() {
        let sink = Sink::default();
        let mut presenter = Presenter::new(sink.clone()).unwrap();
        for start in [0, 1] {
            sink.0.lock().unwrap().clear();
            presenter
                .draw_with_links((30, 20), |frame| {
                    frame.render_widget(Paragraph::new(lines(start..start + 20)), frame.area());
                    [(
                        Position::new(0, 10 - start as u16),
                        Arc::new(Url::parse("https://example.com/").unwrap()),
                    )]
                    .into_iter()
                    .collect()
                })
                .unwrap();
        }
        let output = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        assert!(
            output.contains("\x1b[1S"),
            "the transcript uses terminal scrolling"
        );
        assert!(
            !output.contains("\x1b]8;;https://"),
            "the existing link moves with its row"
        );
        assert_eq!(
            presenter.links[&Position::new(0, 9)].as_str(),
            "https://example.com/"
        );
    }

    #[test]
    fn growing_transcript_writes_far_less_than_a_full_redraw() {
        let sink = Sink::default();
        let mut presenter = Presenter::new(sink.clone()).unwrap();
        let mut history = lines(0..60);
        let mut sizes = Vec::new();
        for _ in 0..2 {
            sink.0.lock().unwrap().clear();
            let visible = history[history.len() - 40..].to_vec();
            presenter
                .draw((120, 40), |frame| {
                    frame.render_widget(Paragraph::new(visible), frame.area())
                })
                .unwrap();
            sizes.push(sink.0.lock().unwrap().len());
            history.push(Line::from(format!("transcript line {}", history.len())));
        }
        assert!(sizes[1] * 10 < sizes[0], "{sizes:?}");
    }
}
