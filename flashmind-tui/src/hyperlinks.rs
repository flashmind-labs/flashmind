//! Link targets for the cells written to the terminal.

use std::collections::HashMap;
use std::sync::Arc;

use ratatui::layout::Position;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use unicode_width::UnicodeWidthStr;
use url::Url;

use crate::styles::S_HINT;

pub(crate) type ScreenLinks = HashMap<Position, Arc<Url>>;

struct Label {
    remaining: usize,
    target: Option<Arc<Url>>,
}

fn is_label(style: Style) -> bool {
    style.fg == S_HINT.fg && style.add_modifier.contains(Modifier::UNDERLINED)
}

fn link_target(target: &str) -> Option<Arc<Url>> {
    if target.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return None;
    }
    // Malformed and unsupported destinations keep their visible fallback.
    let Ok(target) = Url::parse(target) else {
        return None;
    };
    matches!(target.scheme(), "https" | "http" | "mailto" | "file").then(|| Arc::new(target))
}

/// Recover targets from the renderer's label followed by ` (target)` format.
/// Work on logical lines, before wrapping can split a target across rows.
pub(crate) fn layout_links(source: &Line<'_>, rows: &[Line<'_>]) -> ScreenLinks {
    let mut targets = Vec::new();
    for (index, span) in source.spans.iter().enumerate() {
        if !is_label(span.style) {
            continue;
        }
        let target = source.spans.get(index + 1).and_then(|next| {
            next.content
                .strip_prefix(" (")
                .and_then(|text| text.split_once(')').map(|(target, _)| target))
        });
        let target = target.unwrap_or(span.content.as_ref());
        // Keep an entry for unsupported targets so later labels stay aligned.
        targets.push(Label {
            remaining: span.content.chars().filter(|c| !c.is_whitespace()).count(),
            target: link_target(target),
        });
    }
    if targets.is_empty() {
        return ScreenLinks::new();
    }
    let mut targets = targets.into_iter();
    let mut current = targets.next();
    let mut links = ScreenLinks::new();
    for (y, row) in rows.iter().enumerate() {
        let Ok(y) = u16::try_from(y) else { break };
        let mut x = 0u16;
        for span in &row.spans {
            if is_label(span.style) {
                for grapheme in span.styled_graphemes(Style::default()) {
                    while current.as_ref().is_some_and(|label| label.remaining == 0) {
                        current = targets.next();
                    }
                    let width = grapheme.symbol.width() as u16;
                    if let Some(label) = &mut current {
                        if width > 0
                            && let Some(target) = &label.target
                        {
                            links.insert(Position::new(x, y), target.clone());
                        }
                        let consumed = grapheme
                            .symbol
                            .chars()
                            .filter(|c| !c.is_whitespace())
                            .count();
                        label.remaining = label.remaining.saturating_sub(consumed);
                    }
                    x = x.saturating_add(width);
                }
            } else {
                x = x.saturating_add(u16::try_from(span.width()).unwrap_or(u16::MAX));
            }
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::layout_line;
    use ratatui::text::Span;

    fn link(label: &str, target: &str) -> Line<'static> {
        Line::from(vec![
            Span::styled(label.to_owned(), S_HINT.add_modifier(Modifier::UNDERLINED)),
            Span::raw(format!(" ({target})")),
        ])
    }

    #[test]
    fn wrapped_labels_keep_their_target() {
        let source = link("some long link label", "https://example.com/reference");
        let rows = layout_line(&source, 12);
        let links = layout_links(&source, &rows);
        assert!(links.keys().any(|position| position.y > 0));
        for (y, row) in rows.iter().enumerate() {
            for (x, character) in row.to_string().chars().enumerate() {
                if let Some(target) = links.get(&Position::new(x as u16, y as u16)) {
                    assert_eq!(target.as_str(), "https://example.com/reference");
                    assert!(!"():/".contains(character));
                }
            }
        }
        assert_eq!(links.len(), 19);
    }

    #[test]
    fn repeated_labels_keep_distinct_targets() {
        let mut source = link("docs", "https://example.com/one");
        source.spans.push(Span::raw(" "));
        source
            .spans
            .extend(link("docs", "https://example.com/two").spans);
        let links = layout_links(&source, std::slice::from_ref(&source));
        assert_eq!(
            links[&Position::new(0, 0)].as_str(),
            "https://example.com/one"
        );
        assert_eq!(links.values().filter(|url| url.path() == "/two").count(), 4);
    }

    #[test]
    fn unsupported_or_control_character_targets_are_not_links() {
        for target in [
            "relative.pdf",
            "javascript:alert(1)",
            "https://example.com/\x1b\\",
            "https://example.com/\x07",
        ] {
            let source = link("unsafe", target);
            assert!(layout_links(&source, std::slice::from_ref(&source)).is_empty());
        }
    }

    #[test]
    fn wide_characters_only_mark_the_cell_that_draws_them() {
        let source = link("文e\u{301}", "https://example.com");
        let links = layout_links(&source, std::slice::from_ref(&source));
        assert!(links.contains_key(&Position::new(0, 0)));
        assert!(!links.contains_key(&Position::new(1, 0)));
        assert!(links.contains_key(&Position::new(2, 0)));
        assert_eq!(links.len(), 2);

        let source = link("a👩‍💻b", "https://example.com");
        let links = layout_links(&source, std::slice::from_ref(&source));
        assert!(links.contains_key(&Position::new(3, 0)));
        assert_eq!(links.len(), 3);
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn markdown_links_reach_the_terminal_cell_map() {
        let lines = crate::markdown_render::render_reflowable_lines(
            "Read [the reference](https://example.com/reference) for details.",
            80,
        );
        let line = &lines[0];
        let links = layout_links(line, &layout_line(line, 80));
        assert_eq!(
            links[&Position::new(5, 0)].as_str(),
            "https://example.com/reference"
        );
        assert!(!links.contains_key(&Position::new(0, 0)));
        assert!(!links.contains_key(&Position::new(18, 0)));
    }
}
