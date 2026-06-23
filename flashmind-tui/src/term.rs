//! Low-level terminal output helpers.
//!
//! Provides functions for printing styled [`ratatui::text::Line`] values and
//! rendering any [`ratatui::widgets::Widget`] directly to a write stream
//! (typically stdout) without requiring a full ratatui terminal backend.
//!
// TODO: Kitty graphics protocol (with Sixel fallback) for inline image rendering
//!
//! This module bridges the gap between ratatui's buffer-based rendering model
//! and the append-mode output style used by the REPL — output is written as
//! crossterm escape sequences rather than through a full-screen refresh cycle.
//!
//! # How it works
//!
//! Ratatui widgets normally render into an off-screen [`Buffer`] that is then
//! diffed against the previous frame and applied to a full-screen terminal.
//! This module instead takes the rendered buffer and emits each cell as a
//! crossterm escape sequence directly to stdout, preserving the "chat log"
//! feel of incremental output.
//!
//! Style changes are optimized: consecutive cells with the same style share a
//! single style-setting sequence, minimizing escape sequence overhead.
//!
//! # Public API
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`print_line`] | Print a styled [`Line`] followed by a newline |
//! | [`render_widget_to_stdout`] | Render any [`Widget`] to stdout in append mode |
//! | [`visual_height`] | Calculate wrapped line height at a given width |

use std::io::{self, Write};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Widget;
use unicode_width::UnicodeWidthStr;

use ratatui::crossterm::{
    cursor::MoveToColumn,
    queue,
    style::{
        Attribute, Color as CColor, ContentStyle, Print, ResetColor, SetAttribute,
        SetBackgroundColor, SetForegroundColor,
    },
    terminal::{Clear, ClearType},
};

// ---------------------------------------------------------------------------
// Color conversion

fn to_ct_color(c: Color) -> CColor {
    match c {
        Color::Reset => CColor::Reset,
        Color::Black => CColor::Black,
        Color::Red => CColor::DarkRed,
        Color::Green => CColor::DarkGreen,
        Color::Yellow => CColor::DarkYellow,
        Color::Blue => CColor::DarkBlue,
        Color::Magenta => CColor::DarkMagenta,
        Color::Cyan => CColor::DarkCyan,
        Color::Gray => CColor::Grey,
        Color::DarkGray => CColor::DarkGrey,
        Color::LightRed => CColor::Red,
        Color::LightGreen => CColor::Green,
        Color::LightYellow => CColor::Yellow,
        Color::LightBlue => CColor::Blue,
        Color::LightMagenta => CColor::Magenta,
        Color::LightCyan => CColor::Cyan,
        Color::White => CColor::White,
        Color::Rgb(r, g, b) => CColor::Rgb { r, g, b },
        Color::Indexed(i) => CColor::AnsiValue(i),
    }
}

// ---------------------------------------------------------------------------
// Style application

fn apply_style<W: Write>(w: &mut W, style: Style) -> io::Result<()> {
    queue!(w, ResetColor)?;
    if let Some(fg) = style.fg {
        queue!(w, SetForegroundColor(to_ct_color(fg)))?;
    }
    if let Some(bg) = style.bg {
        queue!(w, SetBackgroundColor(to_ct_color(bg)))?;
    }
    if style.add_modifier.contains(Modifier::BOLD) {
        queue!(w, SetAttribute(Attribute::Bold))?;
    }
    if style.add_modifier.contains(Modifier::DIM) {
        queue!(w, SetAttribute(Attribute::Dim))?;
    }
    if style.add_modifier.contains(Modifier::ITALIC) {
        queue!(w, SetAttribute(Attribute::Italic))?;
    }
    if style.add_modifier.contains(Modifier::UNDERLINED) {
        queue!(w, SetAttribute(Attribute::Underlined))?;
    }
    if style.add_modifier.contains(Modifier::REVERSED) {
        queue!(w, SetAttribute(Attribute::Reverse))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API

/// Print a styled [`Line`] to the given writer, followed by a newline (`\r\n`).
///
/// Each span in the line has its style applied via crossterm escape sequences
/// before its content is printed.  The writer is reset to default style at the
/// end of each span and at the end of the line.
///
/// # Arguments
/// * `w` — any `Write` implementor (typically `io::stdout()`)
/// * `line` — the styled line containing one or more spans
///
/// # Note
/// The caller is responsible for flushing the writer after this call returns.
pub fn print_line<W: Write>(w: &mut W, line: &Line<'_>) -> io::Result<()> {
    let mut last_has_bg = false;
    for span in &line.spans {
        apply_style(w, span.style)?;
        queue!(w, Print(&span.content))?;
        last_has_bg = span.style.bg.is_some();
    }
    if last_has_bg {
        queue!(w, Clear(ClearType::UntilNewLine))?;
    }
    queue!(w, ResetColor, Print("\r\n"))?;
    Ok(())
}

/// Like [`print_line`], but flushes the writer immediately after writing.
pub fn println<W: Write>(w: &mut W, line: &Line<'_>) -> io::Result<()> {
    print_line(w, line)?;
    w.flush()
}

/// Render any [`Widget`] to a write stream by rasterizing it to a buffer first.
///
/// This is the core function that allows ratatui widgets to be rendered in
/// append mode (directly to stdout) instead of in a full-screen terminal
/// backend.  The widget is rendered into an off-screen [`Buffer`], then the
/// buffer cells are emitted as crossterm escape sequences row by row.
///
/// # Arguments
/// * `w` — the write stream (typically `io::stdout()`)
/// * `widget` — any type implementing [`Widget`]
/// * `width` — terminal width in columns
/// * `height` — number of rows to render
///
/// # Note
/// Output does **not** include a trailing newline after the last row.  Callers
/// are responsible for positioning the cursor after this call returns.
pub fn render_widget_to_stdout<W: Write, R: Widget>(
    w: &mut W,
    widget: R,
    width: u16,
    height: u16,
) -> io::Result<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    widget.render(area, &mut buf);
    emit_buffer_to_stdout(w, &buf, width, height)
}

/// Render a buffer built by `fill` to a write stream in append mode.
///
/// `fill` receives a freshly-emptied buffer covering `width × height` and may
/// render multiple widgets into sub-regions of it (e.g. a textarea on the left
/// and a pet on the right). The composited buffer is then emitted row by row.
///
/// Like [`render_widget_to_stdout`], output does **not** include a trailing
/// newline after the last row.
pub fn render_composited_to_stdout<W: Write, F>(
    w: &mut W,
    width: u16,
    height: u16,
    fill: F,
) -> io::Result<()>
where
    F: FnOnce(&mut Buffer),
{
    if width == 0 || height == 0 {
        return Ok(());
    }
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    fill(&mut buf);
    emit_buffer_to_stdout(w, &buf, width, height)
}

/// Emit an off-screen buffer's cells as crossterm escape sequences row by row.
///
/// Style changes are coalesced: consecutive cells with the same style share a
/// single style-setting sequence. Each row ends with `EL` (Erase in Line) to
/// cancel any pending auto-wrap state, keeping cursor movement predictable.
fn emit_buffer_to_stdout<W: Write>(
    w: &mut W,
    buf: &Buffer,
    width: u16,
    height: u16,
) -> io::Result<()> {
    let mut current_style: Option<ContentStyle> = None;
    for y in 0..height {
        queue!(w, MoveToColumn(0))?;
        let mut x: u16 = 0;
        while x < width {
            let cell = &buf[(x, y)];
            let sym = cell.symbol();
            let sw = UnicodeWidthStr::width(sym) as u16;

            let footprint = sw.max(1);
            if x + footprint > width {
                break;
            }

            let style = cell_to_content_style(cell);
            if Some(style) != current_style {
                queue!(w, ResetColor)?;
                if let Some(fg) = style.foreground_color {
                    queue!(w, SetForegroundColor(fg))?;
                }
                if let Some(bg) = style.background_color {
                    queue!(w, SetBackgroundColor(bg))?;
                }
                for attr in [
                    Attribute::Bold,
                    Attribute::Dim,
                    Attribute::Italic,
                    Attribute::Underlined,
                    Attribute::Reverse,
                ] {
                    if style.attributes.has(attr) {
                        queue!(w, SetAttribute(attr))?;
                    }
                }
                current_style = Some(style);
            }
            queue!(w, Print(sym))?;
            x += footprint;
        }
        queue!(w, ResetColor)?;
        current_style = None;
        // EL cancels pending auto-wrap state from full-width rows, keeping
        // cursor movement predictable for callers that track row counts.
        queue!(
            w,
            ratatui::crossterm::terminal::Clear(
                ratatui::crossterm::terminal::ClearType::UntilNewLine
            ),
        )?;
        if y + 1 < height {
            queue!(w, Print("\r\n"))?;
        }
    }
    Ok(())
}

/// Calculate the visual height (number of wrapped lines) a [`Line`] would
/// occupy when rendered at the given terminal width.
///
/// Returns at least `1` even for empty lines.
///
/// This accounts for terminal wrapping, which is critical for cursor/row
/// tracking in the streaming partial-text path: a long streamed line (e.g. a
/// note prefixed with an emoji) can wrap to multiple terminal rows, and
/// undercounting those rows leads to a stale anchor row after the terminal
/// scrolls — producing duplicated partial lines on screen.
pub fn visual_height(line: &Line<'_>, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    ratatui::widgets::Paragraph::new(vec![line.clone()])
        .wrap(ratatui::widgets::Wrap { trim: false })
        .line_count(width)
        .max(1) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::{Line, Span};

    #[test]
    fn visual_height_wraps_long_emoji_line() {
        // A long streamed note prefixed with an emoji presentation sequence
        // (U+26A0 + U+FE0F).  `unicode_width` counts the emoji as 2 cells
        // (matching terminal behaviour), so wrapping must be predicted
        // accurately — otherwise the streaming row tracker undercounts and
        // the terminal scrolls past the tracked anchor, duplicating the line.
        let line = Line::from(Span::raw(
            "⚠️ Note: dumper.trades has TTL toDateTime(time) + toIntervalHour(1), meaning rows are deleted 1 hour after their time. So \"today\" here = the last 24 hours of recent data.",
        ));
        // At 40 cols this ~170-cell line must wrap to several rows.
        assert!(visual_height(&line, 40) > 1, "long line should wrap");
        // Wider terminal -> fewer wrapped rows (monotonic non-increasing).
        let h40 = visual_height(&line, 40);
        let h80 = visual_height(&line, 80);
        let h200 = visual_height(&line, 200);
        assert!(h80 <= h40, "wider should not wrap more: {h80} vs {h40}");
        assert!(h200 <= h80, "wider should not wrap more: {h200} vs {h80}");
        assert_eq!(h200, 1, "at 200 cols the line fits on one row");
    }

    #[test]
    fn visual_height_emoji_counts_as_two_cells() {
        // 10 emoji at width 4 -> each emoji is 2 cells -> 20 cells / 4 = 5 rows.
        // (If the emoji were miscounted as 1 cell, this would be 10 cells / 4 = 3 rows.)
        let line = Line::from(Span::raw("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️"));
        let h = visual_height(&line, 4);
        assert!(h >= 5, "emoji should be 2 cells each: got {h}");
    }
}

// ---------------------------------------------------------------------------
// Tui — append-mode terminal wrapper

/// Append-mode terminal for drawing styled lines and widgets.
///
/// Unlike ratatui's `Terminal` which uses a full-screen alternate buffer, `Tui`
/// writes output incrementally to stdout, preserving scrollback history.
///
/// ```rust,ignore
/// let mut tui = Tui::new();
/// tui.println(&Line::from("hello"))?;
///
/// let _raw = tui.raw_mode()?;
/// let drawn = tui.draw_lines(&picker.lines(width))?;
/// // ... handle keys ...
/// tui.erase(drawn)?;
/// ```
#[derive(Debug)]
pub struct Tui {
    stdout: io::Stdout,
}

impl Tui {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
        }
    }

    /// Terminal width and height in columns/rows.
    pub fn size(&self) -> io::Result<(u16, u16)> {
        use ratatui::crossterm::terminal::size;
        size()
    }

    /// Terminal width in columns.
    pub fn width(&self) -> io::Result<u16> {
        self.size().map(|(w, _)| w)
    }

    /// Print a styled line and flush.
    pub fn println(&mut self, line: &Line<'_>) -> io::Result<()> {
        println(&mut self.stdout, line)
    }

    /// Print multiple lines and flush. Returns the number of terminal rows drawn
    /// (accounting for line wrapping).
    pub fn draw_lines(&mut self, lines: &[Line<'_>]) -> io::Result<u16> {
        let width = self.width().unwrap_or(80);
        let mut rows = 0u16;
        for line in lines {
            print_line(&mut self.stdout, line)?;
            rows += visual_height(line, width);
        }
        self.stdout.flush()?;
        Ok(rows)
    }

    /// Erase `n` lines above the cursor.
    pub fn erase(&mut self, n: u16) -> io::Result<()> {
        if n > 0 {
            use ratatui::crossterm::execute;
            execute!(
                self.stdout,
                ratatui::crossterm::cursor::MoveUp(n),
                MoveToColumn(0),
                ratatui::crossterm::terminal::Clear(
                    ratatui::crossterm::terminal::ClearType::FromCursorDown
                ),
            )?;
        }
        Ok(())
    }

    /// Redraw lines in place.
    ///
    /// Erases the previous `prev_rows` terminal rows, then draws the new lines.
    /// Both `prev_rows` and the return value count actual terminal rows
    /// (accounting for line wrapping), not logical lines.
    pub fn redraw_lines(&mut self, lines: &[Line<'_>], prev_rows: u16) -> io::Result<u16> {
        self.erase(prev_rows)?;
        self.draw_lines(lines)
    }

    /// Enter raw terminal mode. Returns a guard that restores normal mode on drop.
    pub fn raw_mode(&self) -> io::Result<crate::widgets::RawModeGuard> {
        crate::widgets::RawModeGuard::enable()
    }

    /// Render a ratatui [`Widget`] in append mode.
    pub fn draw_widget<W: Widget>(&mut self, widget: W, width: u16, height: u16) -> io::Result<()> {
        render_widget_to_stdout(&mut self.stdout, widget, width, height)?;
        self.stdout.flush()
    }

    /// Get a mutable reference to the underlying stdout.
    pub fn stdout(&mut self) -> &mut io::Stdout {
        &mut self.stdout
    }
}

impl Default for Tui {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cell-to-style conversion

fn cell_to_content_style(cell: &ratatui::buffer::Cell) -> ContentStyle {
    let mut cs = ContentStyle::new();
    if cell.fg != Color::Reset {
        cs.foreground_color = Some(to_ct_color(cell.fg));
    }
    if cell.bg != Color::Reset {
        cs.background_color = Some(to_ct_color(cell.bg));
    }
    let m = cell.modifier;
    if m.contains(Modifier::BOLD) {
        cs.attributes.set(Attribute::Bold);
    }
    if m.contains(Modifier::DIM) {
        cs.attributes.set(Attribute::Dim);
    }
    if m.contains(Modifier::ITALIC) {
        cs.attributes.set(Attribute::Italic);
    }
    if m.contains(Modifier::UNDERLINED) {
        cs.attributes.set(Attribute::Underlined);
    }
    if m.contains(Modifier::REVERSED) {
        cs.attributes.set(Attribute::Reverse);
    }
    cs
}
