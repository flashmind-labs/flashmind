use std::ops::Range;

use crate::event_render::{COLLAPSED_DESCRIPTION, expand_hint, fit_tool_line, truncate_display};
use crate::styles::S_HINT;

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use unicode_width::UnicodeWidthStr;

/// Widest leading span that becomes a hanging indent.
const MAX_HANG: usize = 4;

/// Columns transcript text wraps at in a viewport `width` columns wide:
/// 95% of the width, but never narrower than 60 columns (or the whole
/// viewport when it is narrower than that).
pub(crate) fn text_width(width: u16) -> u16 {
    ((u32::from(width) * 19 / 20) as u16).max(width.min(60))
}

/// Lay out one transcript entry for a viewport `width` columns wide.
///
/// Text wraps at [`text_width`]; lines with a background, such as user
/// messages, span the full width and fill every row they wrap onto. Tool
/// headers are fitted to the text width. A line whose first span is a short
/// marker ending in a space, such as `"● "` or `"❯ "`, wraps under the text
/// after the marker instead of under the marker.
pub(crate) fn layout_line(line: &Line<'static>, width: u16) -> Vec<Line<'static>> {
    let filled = line.style.bg.is_some();
    let wrap = if filled { width } else { text_width(width) };
    if let Some(rows) = layout_command(line, wrap) {
        return rows;
    }
    let line = fit_tool_line(line, wrap as usize);
    let hang = line
        .spans
        .first()
        .filter(|marker| {
            let hang = marker.content.width();
            line.spans.len() >= 2
                && (1..=MAX_HANG).contains(&hang)
                && marker.content.ends_with(' ')
                && (wrap as usize) > hang * 2
        })
        .map_or(0, |marker| marker.content.width());
    if hang == 0 && (line.width() <= wrap as usize && !filled || wrap == 0) {
        return vec![line];
    }
    let inner = wrap - hang as u16;
    let rest = Line {
        spans: line.spans[usize::from(hang > 0)..].to_vec(),
        ..line.clone()
    };
    let paragraph = Paragraph::new(rest).wrap(Wrap { trim: false });
    let height = paragraph.line_count(inner).clamp(1, u16::MAX as usize) as u16;
    let mut buffer = Buffer::empty(Rect::new(0, 0, inner, height));
    if filled {
        buffer.set_style(buffer.area, line.style);
    }
    paragraph.render(buffer.area, &mut buffer);
    (0..height)
        .map(|y| {
            let mut spans = match (hang, y) {
                (0, _) => Vec::new(),
                (_, 0) => vec![line.spans[0].clone()],
                _ => vec![Span::raw(" ".repeat(hang))],
            };
            spans.extend(buffer_row(&buffer, y, filled));
            Line {
                spans,
                ..line.clone()
            }
        })
        .collect()
}

fn layout_command(line: &Line<'static>, width: u16) -> Option<Vec<Line<'static>>> {
    let [indent, marker, space, command, elapsed] = line.spans.as_slice() else {
        return None;
    };
    if indent.content != "  "
        || space.content != " "
        || !matches!(marker.content.as_ref(), "\u{2022}" | "\u{2717}")
    {
        return None;
    }
    let width_usize = usize::from(width);
    let duration = truncate_display(elapsed.content.trim(), width_usize);
    let prefix = format!("{}{}{}", indent.content, marker.content, space.content);
    let reserved = duration.width().max(8).min(width_usize);
    let available = width_usize.saturating_sub(prefix.width() + reserved + 2);
    if available == 0 {
        return Some(vec![fit_tool_line(
            &Line::from(vec![
                Span::styled(prefix, marker.style),
                command.clone(),
                elapsed.clone(),
            ]),
            width_usize,
        )]);
    }
    let inner =
        u16::try_from(available).expect("The command width is bounded by the u16 viewport width.");
    let paragraph = Paragraph::new(command.content.as_ref())
        .style(command.style)
        .wrap(Wrap { trim: false });
    let total = paragraph.line_count(inner);
    let shown = total.clamp(1, COLLAPSED_DESCRIPTION) as u16;
    let mut buffer = Buffer::empty(Rect::new(0, 0, inner, shown));
    paragraph.render(buffer.area, &mut buffer);
    let mut rows = Vec::new();
    for y in 0..shown {
        let mut spans = if y == 0 {
            vec![indent.clone(), marker.clone(), space.clone()]
        } else {
            vec![Span::raw(" ".repeat(prefix.width()))]
        };
        spans.extend(buffer_row(&buffer, y, false));
        if y == 0 {
            let used = spans.iter().map(Span::width).sum::<usize>() + duration.width();
            spans.push(Span::raw(" ".repeat(width_usize.saturating_sub(used))));
            spans.push(Span::styled(duration.clone(), elapsed.style));
        }
        rows.push(Line {
            spans,
            ..line.clone()
        });
    }
    if total > COLLAPSED_DESCRIPTION {
        rows.push(Line::from(vec![
            Span::raw(" ".repeat(prefix.width())),
            Span::styled(
                truncate_display(
                    &expand_hint(total - COLLAPSED_DESCRIPTION, true),
                    width_usize.saturating_sub(prefix.width()),
                ),
                S_HINT,
            ),
        ]));
    }
    Some(rows)
}

/// Read one buffer row back as styled spans. Trailing blank cells are dropped
/// unless `filled` keeps them to carry a background to the right edge.
fn buffer_row(buffer: &Buffer, y: u16, filled: bool) -> Vec<Span<'static>> {
    let mut cells: Vec<(String, Style)> = Vec::new();
    let mut x = 0;
    while x < buffer.area.width {
        let cell = &buffer[(x, y)];
        let symbol = cell.symbol();
        cells.push((symbol.to_owned(), cell.style()));
        x += (symbol.width() as u16).max(1);
    }
    if !filled {
        while cells.last().is_some_and(|(symbol, style)| {
            symbol == " " && matches!(style.bg, None | Some(Color::Reset))
        }) {
            cells.pop();
        }
    }
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (symbol, style) in cells {
        match spans.last_mut() {
            Some(span) if span.style == style => span.content.to_mut().push_str(&symbol),
            _ => spans.push(Span::styled(symbol, style)),
        }
    }
    spans
}

/// Cached row offsets for append-only transcript entries at one terminal width.
#[derive(Default)]
pub(crate) struct TranscriptLayout {
    width: u16,
    offsets: Vec<usize>,
}

pub(crate) struct TranscriptWindow {
    pub entries: Range<usize>,
    pub row_offset: u16,
}

impl TranscriptLayout {
    pub fn invalidate(&mut self) {
        self.offsets.clear();
    }

    pub fn width(&self) -> u16 {
        self.width
    }
    pub fn measure(&mut self, lines: &[Line<'static>], width: u16) {
        if width != self.width || self.offsets.len() > lines.len() + 1 || self.offsets.is_empty() {
            self.width = width;
            self.offsets = vec![0];
        }
        for line in &lines[self.offsets.len() - 1..] {
            let rows = Paragraph::new(layout_line(line, width))
                .wrap(Wrap { trim: false })
                .line_count(width.max(1));
            self.offsets.push(self.rows() + rows);
        }
    }

    pub fn rows(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    pub fn entry_row(&self, index: usize) -> usize {
        self.offsets.get(index).copied().unwrap_or(self.rows())
    }

    pub fn window(&self, top: usize, height: usize) -> TranscriptWindow {
        let entries = self.offsets.len().saturating_sub(1);
        if top >= self.rows() {
            return TranscriptWindow {
                entries: entries..entries,
                row_offset: top.saturating_sub(self.rows()).min(u16::MAX as usize) as u16,
            };
        }
        let start = self
            .offsets
            .partition_point(|&offset| offset <= top)
            .saturating_sub(1)
            .min(entries);
        let end = self
            .offsets
            .partition_point(|&offset| offset < top + height)
            .min(entries);
        TranscriptWindow {
            entries: start..end,
            row_offset: top
                .saturating_sub(self.offsets[start])
                .min(u16::MAX as usize) as u16,
        }
    }
}

pub(crate) struct TranscriptScroll {
    pub top: usize,
    pub max: usize,
    pub following: bool,
    visible: usize,
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            top: 0,
            max: 0,
            following: true,
            visible: 0,
        }
    }
}

impl TranscriptScroll {
    pub fn jump_to(&mut self, row: usize) {
        self.top = row.min(self.max);
        self.following = false;
    }

    pub fn latest(&mut self) {
        self.top = self.max;
        self.following = true;
    }
    pub fn update(&mut self, rows: usize, height: u16) {
        let was_at_bottom = self.top == self.max;
        self.visible = height as usize;
        self.max = rows.saturating_sub(self.visible);
        self.top = if self.following || was_at_bottom {
            self.max
        } else {
            self.top.min(self.max)
        };
        self.following |= was_at_bottom;
    }

    fn up(&mut self, rows: usize) {
        if self.max > 0 {
            self.following = false;
        }
        self.top = self.top.saturating_sub(rows);
    }

    fn down(&mut self, rows: usize) {
        self.top = (self.top + rows).min(self.max);
        self.following = self.top == self.max;
    }

    pub fn key(&mut self, key: KeyEvent) -> bool {
        let page = (self.visible / 2).max(1);
        match key.code {
            KeyCode::PageUp => self.up(page),
            KeyCode::PageDown => self.down(page),
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => self.up(self.max),
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => self.down(self.max),
            _ => return false,
        }
        true
    }

    pub fn mouse(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => self.up(3),
            MouseEventKind::ScrollDown => self.down(3),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_pauses_follow_and_keeps_position_as_output_arrives() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(100, 20);
        assert_eq!(scroll.top, 80);
        scroll.key(KeyEvent::from(KeyCode::PageUp));
        assert_eq!(scroll.top, 70);
        scroll.update(150, 15);
        assert_eq!(scroll.top, 70);
        assert!(!scroll.following);
        scroll.key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        assert_eq!(scroll.top, 135);
        assert!(scroll.following);
    }

    #[test]
    fn mouse_can_read_history_and_return_to_following() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(30, 10);
        scroll.mouse(MouseEventKind::ScrollUp);
        assert_eq!(scroll.top, 17);
        scroll.mouse(MouseEventKind::ScrollDown);
        assert!(scroll.following);
    }

    #[test]
    fn new_output_follows_when_position_is_already_at_bottom() {
        let mut scroll = TranscriptScroll::default();
        scroll.update(30, 10);
        scroll.jump_to(scroll.max);
        assert!(!scroll.following);

        scroll.update(35, 10);
        assert_eq!(scroll.top, 25);
        assert!(scroll.following);

        scroll.key(KeyEvent::from(KeyCode::PageUp));
        scroll.update(40, 10);
        assert_eq!(scroll.top, 20);
        assert!(!scroll.following);
    }

    #[test]
    fn wide_viewports_wrap_text_early_but_fill_shaded_lines() {
        let text = Line::from("example ".repeat(40));
        assert_eq!(text_width(200), 190);
        assert_eq!(text_width(64), 60);
        assert_eq!(text_width(40), 40);
        let rows = layout_line(&text, 120);
        assert!(rows.iter().all(|row| row.width() <= 114));
        assert!(rows.len() > 1);
        let shaded = Line::from(vec![Span::raw("> "), Span::raw("short")])
            .style(Style::new().bg(ratatui::style::Color::Blue));
        assert_eq!(layout_line(&shaded, 120)[0].width(), 120);
    }

    #[test]
    fn visible_window_handles_wrapping_growth_and_resize() {
        let mut lines = vec![
            Line::from("界".repeat(10)),
            Line::from("second"),
            Line::from("third"),
        ];
        let mut layout = TranscriptLayout::default();
        layout.measure(&lines, 10);
        assert_eq!(layout.rows(), 4);
        let window = layout.window(1, 2);
        assert_eq!(window.entries, 0..2);
        assert_eq!(window.row_offset, 1);
        lines.push(Line::from("fourth"));
        layout.measure(&lines, 10);
        assert_eq!(layout.rows(), 5);
        layout.measure(&lines, 20);
        assert_eq!(layout.rows(), 4);
    }
}
