//! Mouse selection over rendered cells, shared by chat and menus.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::layout::Position;
use unicode_width::UnicodeWidthStr;

use crate::styles;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TextPoint {
    /// Row in the screen or laid-out transcript, depending on the owner.
    pub row: usize,
    pub column: u16,
}

pub(crate) struct TextSelection {
    pub start: TextPoint,
    pub end: TextPoint,
}

impl TextSelection {
    pub fn contains(&self, point: TextPoint) -> bool {
        self.start.min(self.end) <= point && point <= self.start.max(self.end)
    }

    pub fn bounds(&self) -> (TextPoint, TextPoint) {
        (self.start.min(self.end), self.start.max(self.end))
    }

    pub fn text(&self, buffer: &Buffer, first_row: usize) -> String {
        let area = buffer.area;
        let (start, end) = self.bounds();
        let mut lines = Vec::new();
        for y in area.y..area.bottom() {
            let row = first_row + (y - area.y) as usize;
            if row < start.row || row > end.row {
                continue;
            }
            let mut line = String::new();
            let mut x = area.x;
            while x < area.right() {
                let symbol = buffer[(x, y)].symbol();
                let footprint = (symbol.width() as u16).max(1);
                if self.contains(TextPoint { row, column: x })
                    || self.contains(TextPoint {
                        row,
                        column: x + footprint - 1,
                    })
                {
                    line.push_str(symbol);
                }
                x += footprint;
            }
            lines.push(line.trim_end().to_string());
        }
        lines.join("\n")
    }
}

pub(crate) enum SelectionAction {
    Redraw,
    Copy(String),
}

#[derive(Default)]
pub(crate) struct ScreenSelection {
    frame: Option<Buffer>,
    source: Option<Buffer>,
    selection: Option<TextSelection>,
    dragging: bool,
}

impl ScreenSelection {
    pub fn is_dragging(&self) -> bool {
        self.dragging
    }

    pub fn contains(&self, position: Position) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|frame| frame.area.contains(position))
    }

    pub fn clear(&mut self) {
        self.source = None;
        self.selection = None;
        self.dragging = false;
    }

    pub fn handle_event(&mut self, event: &Event) -> Option<SelectionAction> {
        match event {
            Event::Mouse(mouse) => {
                let point = Position::new(mouse.column, mouse.row);
                match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        self.clear();
                        let frame = self
                            .frame
                            .as_ref()
                            .filter(|frame| frame.area.contains(point))?;
                        self.source = Some(frame.clone());
                        let point = TextPoint {
                            row: point.y as usize,
                            column: point.x,
                        };
                        self.selection = Some(TextSelection {
                            start: point,
                            end: point,
                        });
                        self.dragging = true;
                        Some(SelectionAction::Redraw)
                    }
                    MouseEventKind::Drag(MouseButton::Left) if self.dragging => {
                        let area = self.source.as_ref()?.area;
                        if let Some(selection) = &mut self.selection {
                            selection.end = TextPoint {
                                row: point.y.clamp(area.y, area.bottom() - 1) as usize,
                                column: point.x.clamp(area.x, area.right() - 1),
                            };
                        }
                        Some(SelectionAction::Redraw)
                    }
                    MouseEventKind::Up(MouseButton::Left) if self.dragging => {
                        self.dragging = false;
                        if self
                            .selection
                            .as_ref()
                            .is_some_and(|selection| selection.start == selection.end)
                        {
                            self.clear();
                            Some(SelectionAction::Redraw)
                        } else {
                            self.copy_action()
                        }
                    }
                    MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                        self.clear();
                        None
                    }
                    _ => None,
                }
            }
            Event::Key(key) if self.selection.is_some() => {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.copy_action()
                } else if key.code == KeyCode::Esc {
                    self.clear();
                    Some(SelectionAction::Redraw)
                } else {
                    self.clear();
                    None
                }
            }
            Event::Resize(_, _) => {
                self.clear();
                None
            }
            _ => None,
        }
    }

    fn copy_action(&self) -> Option<SelectionAction> {
        let source = self.source.as_ref()?;
        let selection = self.selection.as_ref()?;
        if selection.start == selection.end {
            return Some(SelectionAction::Redraw);
        }
        let text = selection.text(source, source.area.y as usize);
        Some(if text.trim().is_empty() {
            SelectionAction::Redraw
        } else {
            SelectionAction::Copy(text)
        })
    }

    /// Capture the drawn text and overlay the selected cells from the press.
    /// The copied text stays stable while other parts of the screen update.
    pub fn render(&mut self, buffer: &mut Buffer) {
        if self
            .frame
            .as_ref()
            .is_some_and(|frame| frame.area != buffer.area)
        {
            self.clear();
        }
        self.frame = Some(buffer.clone());
        let (Some(selection), Some(source)) = (&self.selection, &self.source) else {
            return;
        };
        if selection.start == selection.end {
            return;
        }
        for y in source.area.y..source.area.bottom() {
            let mut x = source.area.x;
            while x < source.area.right() {
                let width = (source[(x, y)].symbol().width() as u16).max(1);
                if selection.contains(TextPoint {
                    row: y as usize,
                    column: x,
                }) || selection.contains(TextPoint {
                    row: y as usize,
                    column: x + width - 1,
                }) {
                    for column in x..(x + width).min(source.area.right()) {
                        buffer[(column, y)] = source[(column, y)].clone();
                        buffer[(column, y)].set_style(styles::S_SELECTION);
                    }
                }
                x += width;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyEvent, MouseEvent};

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> Event {
        Event::Mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
    }

    fn copied(action: Option<SelectionAction>) -> String {
        match action {
            Some(SelectionAction::Copy(text)) => text,
            _ => panic!("a dragged selection produces text to copy"),
        }
    }

    #[test]
    fn screen_selection_copies_across_rows_in_both_directions() {
        let mut selection = ScreenSelection::default();
        let mut buffer = Buffer::with_lines(["header text", "input draft", "status line"]);
        selection.render(&mut buffer);
        for (start, end) in [
            (Position::new(7, 0), Position::new(5, 2)),
            (Position::new(5, 2), Position::new(7, 0)),
        ] {
            selection.handle_event(&mouse(
                MouseEventKind::Down(MouseButton::Left),
                start.x,
                start.y,
            ));
            selection.handle_event(&mouse(
                MouseEventKind::Drag(MouseButton::Left),
                end.x,
                end.y,
            ));
            assert_eq!(
                copied(selection.handle_event(&mouse(
                    MouseEventKind::Up(MouseButton::Left),
                    end.x,
                    end.y
                ))),
                "text\ninput draft\nstatus"
            );
            assert_eq!(
                copied(selection.handle_event(&Event::Key(KeyEvent::new(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL
                )))),
                "text\ninput draft\nstatus"
            );
            assert!(matches!(
                selection.handle_event(&Event::Key(KeyEvent::from(KeyCode::Esc))),
                Some(SelectionAction::Redraw)
            ));
        }
    }

    #[test]
    fn selection_keeps_copied_text_and_highlight_stable_during_updates() {
        let mut selection = ScreenSelection::default();
        let mut before = Buffer::with_lines(["draft text", "status old"]);
        selection.render(&mut before);
        selection.handle_event(&mouse(MouseEventKind::Down(MouseButton::Left), 0, 0));
        selection.handle_event(&mouse(MouseEventKind::Drag(MouseButton::Left), 4, 0));
        let mut after = Buffer::with_lines(["other text", "status new"]);
        selection.render(&mut after);
        assert_eq!(after[(0, 0)].symbol(), "d");
        assert_eq!(after[(0, 0)].bg, styles::S_SELECTION.bg.unwrap());
        assert_eq!(after[(7, 1)].symbol(), "n");
        assert_eq!(
            copied(selection.handle_event(&mouse(MouseEventKind::Up(MouseButton::Left), 4, 0))),
            "draft"
        );
        assert!(
            selection
                .handle_event(&Event::Key(KeyEvent::from(KeyCode::Char('x'))))
                .is_none()
        );
        selection.render(&mut after);
        assert!(selection.selection.is_none());
    }

    #[test]
    fn wide_and_combining_characters_copy_once_and_highlight_whole_cells() {
        let mut selection = ScreenSelection::default();
        let mut buffer = Buffer::with_lines(["a\u{754c}e\u{301}z"]);
        selection.render(&mut buffer);
        // Start on the second cell of the wide character.
        selection.handle_event(&mouse(MouseEventKind::Down(MouseButton::Left), 2, 0));
        selection.handle_event(&mouse(MouseEventKind::Drag(MouseButton::Left), 3, 0));
        selection.render(&mut buffer);
        assert_eq!(buffer[(1, 0)].bg, styles::S_SELECTION.bg.unwrap());
        assert_eq!(buffer[(2, 0)].bg, styles::S_SELECTION.bg.unwrap());
        assert_eq!(
            copied(selection.handle_event(&mouse(MouseEventKind::Up(MouseButton::Left), 3, 0))),
            "\u{754c}e\u{301}"
        );
    }

    #[test]
    fn bare_click_resize_and_outside_click_leave_no_selection_to_copy() {
        let mut selection = ScreenSelection::default();
        let mut buffer = Buffer::with_lines(["some text"]);
        selection.render(&mut buffer);
        selection.handle_event(&mouse(MouseEventKind::Down(MouseButton::Left), 1, 0));
        assert!(matches!(
            selection.handle_event(&mouse(MouseEventKind::Up(MouseButton::Left), 1, 0)),
            Some(SelectionAction::Redraw)
        ));
        assert!(
            selection
                .handle_event(&Event::Key(KeyEvent::new(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL
                )))
                .is_none()
        );
        selection.handle_event(&mouse(MouseEventKind::Down(MouseButton::Left), 1, 0));
        selection.handle_event(&mouse(MouseEventKind::Drag(MouseButton::Left), 5, 0));
        let mut resized = Buffer::with_lines(["short"]);
        selection.render(&mut resized);
        assert!(selection.selection.is_none());
        assert!(
            selection
                .handle_event(&mouse(MouseEventKind::Down(MouseButton::Left), 20, 20))
                .is_none()
        );
        assert!(
            selection
                .handle_event(&mouse(MouseEventKind::Drag(MouseButton::Left), 0, 0))
                .is_none()
        );
    }
}
