//! Terminal-specific rendering — converts parsed markdown to ANSI-formatted output.
//!
//! Features:
//! - Bold, italic, strikethrough via ANSI codes
//! - Color-coded headings (H1=cyan, H2=blue, H3=magenta)
//! - Error/warning indicators (bright foreground colors)
//! - Tables via comfy-table with box-drawing characters
//! - Code blocks with simple syntax highlighting
//! - Respects terminal width for wrapping

use comfy_table::modifiers::{UTF8_ROUND_CORNERS, UTF8_SOLID_INNER_BORDERS};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, Table};
use nu_ansi_term::{Color, Style};
use ratatui::text::Line;
use regex::Regex;
use std::sync::LazyLock;

use crate::markdown::{
    DocBlock, InlineToken, latex_to_unicode, maybe_latex, parse_document, tokenize_inline,
};

const DEFAULT_WIDTH: u16 = 80;

const MAX_TABLE_WIDTH: u16 = 120;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render markdown text to ANSI-formatted terminal output.
pub fn render_terminal(text: &str) -> String {
    let blocks = parse_document(text).blocks;
    let (width, _) = ratatui::crossterm::terminal::size().unwrap_or((DEFAULT_WIDTH, 0));

    let mut out = String::with_capacity(text.len() * 2);
    for block in blocks.iter() {
        render_block(&mut out, block, width);
    }

    out
}

/// Render markdown text directly to ratatui [`Line`]s.
pub fn render_to_lines(text: &str) -> Vec<Line<'static>> {
    use ansi_to_tui::IntoText;

    let ansi = render_terminal(text);
    match ansi.as_str().into_text() {
        Ok(text) => text.lines.into_iter().collect(),
        Err(_) => vec![Line::raw(ansi)],
    }
}

// ---------------------------------------------------------------------------
// ANSI stripping
// ---------------------------------------------------------------------------

#[cfg(test)]
fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape && c == 'm' {
            in_escape = false;
        } else if !in_escape {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ANSI-aware text helpers
// ---------------------------------------------------------------------------

fn visible_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthChar;

    let mut width = 0;
    let mut in_escape = false;

    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else {
            width += c.width().unwrap_or(0);
        }
    }

    width
}

fn wrap_text(text: &str, width: usize) -> String {
    if width == 0 {
        return text.to_string();
    }

    let mut result = String::new();
    let mut current_line = String::new();
    let mut current_width = 0usize;

    let mut words: Vec<String> = Vec::new();
    let mut current_word = String::new();
    let mut in_escape = false;
    let mut escape_buffer = String::new();

    for c in text.chars() {
        if c == '\x1b' {
            in_escape = true;
            escape_buffer.push(c);
        } else if in_escape {
            escape_buffer.push(c);
            if c == 'm' {
                in_escape = false;
                current_word.push_str(&escape_buffer);
                escape_buffer.clear();
            }
        } else if c == '\n' {
            if !current_word.is_empty() {
                words.push(std::mem::take(&mut current_word));
            }
            words.push("\n".to_string());
        } else if c.is_whitespace() {
            if !current_word.is_empty() {
                words.push(std::mem::take(&mut current_word));
            }
            words.push(c.to_string());
        } else {
            current_word.push(c);
        }
    }
    if !current_word.is_empty() {
        words.push(current_word);
    }

    let mut active_style = String::new();

    for word in words {
        if word == "\n" {
            if !active_style.is_empty() {
                current_line.push_str("\x1b[0m");
            }
            result.push_str(&current_line);
            result.push('\n');
            current_line.clear();
            current_width = 0;

            if !active_style.is_empty() {
                current_line.push_str(&active_style);
            }
            continue;
        }

        let word_width = visible_width(&word);

        let needs_space = current_width > 0
            && !word.chars().all(|c| c.is_whitespace())
            && !current_line.ends_with(char::is_whitespace);
        let space_cost = if needs_space { 1 } else { 0 };

        if current_width + space_cost + word_width > width && current_width > 0 {
            if !active_style.is_empty() {
                current_line.push_str("\x1b[0m");
            }
            result.push_str(&current_line);
            result.push('\n');
            current_line.clear();

            if !active_style.is_empty() {
                current_line.push_str(&active_style);
            }

            current_line.push_str(&word);
            current_width = word_width;
        } else {
            if needs_space {
                current_line.push(' ');
                current_width += 1;
            }
            current_line.push_str(&word);
            current_width += word_width;
        }

        for escape in extract_ansi_sequences(&word) {
            if escape == "\x1b[0m" {
                active_style.clear();
            } else {
                active_style = escape;
            }
        }
    }

    result.push_str(&current_line);
    result
}

fn extract_ansi_sequences(s: &str) -> Vec<String> {
    let mut sequences = Vec::new();
    let mut in_escape = false;
    let mut escape = String::new();

    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            escape.push(c);
        } else if in_escape {
            escape.push(c);
            if c == 'm' {
                in_escape = false;
                sequences.push(std::mem::take(&mut escape));
            }
        }
    }

    sequences
}

// ---------------------------------------------------------------------------
// Block rendering
// ---------------------------------------------------------------------------

fn render_block(out: &mut String, block: &DocBlock, width: u16) {
    match block {
        DocBlock::Paragraph(text) => {
            let joined = text.replace('\n', " ");
            let wrapped = wrap_text(&render_inline(&joined), width as usize);
            out.push_str(&wrapped);
            out.push('\n');
        }

        DocBlock::Heading { level, text } => {
            out.push('\n');
            let style = heading_style(*level);
            let styled = style.paint(render_inline(text));
            out.push_str(&styled.to_string());
            out.push('\n');
        }

        DocBlock::CodeBlock { lang, code } => {
            let highlighted = highlight_code(code, *lang);
            out.push_str(&highlighted);
            out.push_str("\n\n");
        }

        DocBlock::Table { header, rows } => {
            out.push_str(&render_table(header, rows, width));
            out.push_str("\n\n");
        }

        DocBlock::BulletList(items) => {
            let indent = "  ";
            let avail = (width as usize).saturating_sub(indent.len());
            for item in items {
                let rendered = render_inline(item);
                let wrapped = wrap_text(&rendered, avail);
                for (i, line) in wrapped.split('\n').enumerate() {
                    if i == 0 {
                        out.push_str("- ");
                    } else {
                        out.push_str(indent);
                    }
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        DocBlock::OrderedList(start, items) => {
            let indent = "    ";
            let avail = (width as usize).saturating_sub(indent.len());
            for (i, item) in items.iter().enumerate() {
                let rendered = render_inline(item);
                let wrapped = wrap_text(&rendered, avail);
                for (j, line) in wrapped.split('\n').enumerate() {
                    if j == 0 {
                        out.push_str(&format!("{:>2}. ", start + i));
                    } else {
                        out.push_str(indent);
                    }
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        DocBlock::Blockquote(text) => {
            let style = Style::new().fg(Color::Fixed(8)).italic();
            for (i, line) in text.split('\n').enumerate() {
                if i > 0 || !text.is_empty() {
                    out.push_str(&style.paint("│ ").to_string());
                }
                if line.is_empty() {
                    out.push('\n');
                } else {
                    out.push_str(&style.paint(render_inline(line)).to_string());
                    out.push('\n');
                }
            }
        }

        DocBlock::HorizontalRule => {
            let line = "─".repeat((width as usize).min(MAX_TABLE_WIDTH as usize));
            let dimmed = Style::new().fg(Color::Fixed(8)).paint(line);
            out.push_str(&dimmed.to_string());
            out.push('\n');
        }
    }
}

// ---------------------------------------------------------------------------
// Inline rendering
// ---------------------------------------------------------------------------

fn render_inline(text: &str) -> String {
    let tokens = tokenize_inline(text);
    let mut out = String::with_capacity(text.len());

    for token in tokens {
        match token {
            InlineToken::Text(t) => {
                let t = maybe_latex(t);
                if let Some(style) = detect_severity_pattern(&t) {
                    out.push_str(&style.paint(t.as_ref()).to_string());
                } else {
                    out.push_str(&t);
                }
            }
            InlineToken::Bold(t) => {
                let t = maybe_latex(t);
                let styled = Style::new().bold().paint(t.as_ref());
                out.push_str(&styled.to_string());
            }
            InlineToken::Italic(t) => {
                let t = maybe_latex(t);
                let styled = Style::new().italic().paint(t.as_ref());
                out.push_str(&styled.to_string());
            }
            InlineToken::Strike(t) => {
                let t = maybe_latex(t);
                let styled = Style::new().blink().paint(t.as_ref());
                out.push_str(&styled.to_string());
            }
            InlineToken::Code(t) => {
                let styled = Style::new().bold().fg(Color::Cyan).paint(t);
                out.push_str(&styled.to_string());
            }
            InlineToken::Link { text, url } => {
                let label = Style::new().fg(Color::Blue).underline().paint(text);
                out.push_str(&label.to_string());
                if !url.is_empty() && url != text {
                    let dim = Style::new().fg(Color::Fixed(8));
                    out.push_str(&dim.paint(format!(" ({})", url)).to_string());
                }
            }
            InlineToken::Emoji(name) => {
                out.push(':');
                out.push_str(name);
                out.push(':');
            }
            InlineToken::Math(t) => {
                let converted = latex_to_unicode(t);
                let styled = Style::new().fg(Color::Yellow).paint(converted);
                out.push_str(&styled.to_string());
            }
        }
    }

    out
}

fn heading_style(level: u8) -> Style {
    match level {
        1 => Style::new().fg(Color::Cyan).bold().underline(),
        2 => Style::new().fg(Color::Blue).bold(),
        3 => Style::new().fg(Color::Magenta).bold(),
        4 => Style::new().fg(Color::Green).bold(),
        _ => Style::new().bold(),
    }
}

fn detect_severity_pattern(text: &str) -> Option<Style> {
    let upper = text.to_uppercase();

    if upper.starts_with("ERROR:")
        || upper.contains("ERROR:")
        || text.contains("✗")
        || text.contains("❌")
    {
        return Some(Style::new().fg(Color::Red).bold());
    }

    if upper.starts_with("WARNING:")
        || upper.contains("WARNING:")
        || text.contains("⚠")
        || (text.contains("!") && upper.contains("WARN"))
    {
        return Some(Style::new().fg(Color::Yellow).bold());
    }

    None
}

// ---------------------------------------------------------------------------
// Code highlighting
// ---------------------------------------------------------------------------

fn highlight_code(code: &str, lang: Option<&str>) -> String {
    let normalized_lang = lang.map(|s| s.to_lowercase());
    let lang_name = normalized_lang.as_deref().unwrap_or("plaintext");

    let effective_lang = match lang_name {
        "rs" | "rust" => "rust",
        "py" | "python" => "python",
        "js" | "javascript" | "jsx" => "javascript",
        "ts" | "typescript" | "tsx" => "typescript",
        "json" => "json",
        "sql" => "sql",
        "sh" | "bash" | "zsh" | "shell" => "shell",
        "txt" | "text" | "" => "plaintext",
        _ => lang_name,
    };

    match effective_lang {
        "rust" => highlight_rust(code),
        "python" => highlight_python(code),
        "javascript" | "typescript" => highlight_js_ts(code),
        "json" => highlight_json(code),
        "sql" => highlight_sql(code),
        "shell" => highlight_shell(code),
        _ => code.to_string(),
    }
}

fn highlight_rust(code: &str) -> String {
    static KW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(as|async|await|break|const|continue|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while|abstract|become|do|final|override|priv|typeof|unsized|virtual|yield)\b").unwrap()
    });

    static TYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize|f32|f64|bool|char|str|Vec|Option|Result|HashMap|HashSet|String|Cow|Box|Rc|Arc|Mutex|Pin|PhantomData)\b").unwrap()
    });

    highlight_by_rules(code, &[&KW_RE, &TYPE_RE], &[Color::Purple, Color::Cyan])
}

fn highlight_python(code: &str) -> String {
    static KW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b").unwrap()
    });

    static BUILTIN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(abs|all|any|bin|bool|callable|chr|classmethod|compile|complex|delattr|dict|dir|divmod|enumerate|eval|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|int|isinstance|issubclass|iter|len|list|locals|max|min|next|object|oct|open|ord|pow|print|property|range|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|vars|zip)\b").unwrap()
    });

    highlight_by_rules(code, &[&KW_RE, &BUILTIN_RE], &[Color::Purple, Color::Cyan])
}

fn highlight_js_ts(code: &str) -> String {
    static KW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|false|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|package|private|protected|public|return|set|static|string|switch|symbol|this|throw|true|try|typeof|var|void|while|with|yield)\b").unwrap()
    });

    static BUILTIN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(Array|Boolean|Date|Error|Function|Intl|JSON|Map|Math|Number|Object|Promise|Proxy|Reflect|RegExp|Set|String|Symbol|WeakMap|WeakSet|console|document|navigator|performance|window)\b").unwrap()
    });

    highlight_by_rules(code, &[&KW_RE, &BUILTIN_RE], &[Color::Purple, Color::Cyan])
}

fn highlight_json(code: &str) -> String {
    static KEY_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([^"]+)"\s*:"#).unwrap());
    static STR_VAL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""[^"]*""#).unwrap());
    static NUM_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b").unwrap());
    static BOOL_NULL_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b(true|false|null)\b").unwrap());

    let mut result = String::new();

    for line in code.lines() {
        let mut keys: Vec<String> = Vec::new();
        let with_key_placeholders = KEY_RE
            .replace_all(line, |caps: &regex::Captures| {
                let idx = keys.len();
                let styled = format!("{}:", Color::Cyan.bold().paint(format!("\"{}\"", &caps[1])));
                keys.push(styled);
                format!("\x00KEY{}\x00", idx)
            })
            .to_string();

        let mut strings: Vec<String> = Vec::new();
        let with_placeholders = STR_VAL_RE
            .replace_all(&with_key_placeholders, |caps: &regex::Captures| {
                let idx = strings.len();
                strings.push(caps[0].to_string());
                format!("\x00STR{}\x00", idx)
            })
            .to_string();

        let mut styled = NUM_RE
            .replace_all(&with_placeholders, |caps: &regex::Captures| {
                Color::Green.paint(&caps[1]).to_string()
            })
            .to_string();

        styled = BOOL_NULL_RE
            .replace_all(&styled, |caps: &regex::Captures| {
                Color::Purple.bold().paint(&caps[1]).to_string()
            })
            .to_string();

        for (idx, original) in strings.iter().enumerate() {
            let placeholder = format!("\x00STR{}\x00", idx);
            let highlighted = Color::Green.paint(original.as_str()).to_string();
            styled = styled.replace(&placeholder, &highlighted);
        }

        for (idx, styled_key) in keys.iter().enumerate() {
            let placeholder = format!("\x00KEY{}\x00", idx);
            styled = styled.replace(&placeholder, styled_key);
        }

        result.push_str(&styled);
        result.push('\n');
    }

    result.trim_end().to_string()
}

fn highlight_sql(code: &str) -> String {
    static KW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|DISTINCT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|ON|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT|CASE|WHEN|THEN|ELSE|END|AS|AND|OR|NOT|NULL|IS|LIKE|IN|BETWEEN|ASC|DESC|INDEX|PRIMARY|FOREIGN|REFERENCES|CONSTRAINT|VALUES|INTO|SET|CASCADE|RESTRICT|WITH|RECURSIVE|CAST|CONVERT|COALESCE|IFNULL|NVL|COUNT|SUM|AVG|MIN|MAX|TRUE|FALSE|BOOLEAN)\b").unwrap()
    });

    static TYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(INT|INTEGER|BIGINT|SMALLINT|TINYINT|FLOAT|DOUBLE|DECIMAL|NUMERIC|REAL|CHAR|VARCHAR|TEXT|DATE|TIME|TIMESTAMP|DATETIME|BOOL|BLOB|BYTEA|SERIAL|UUID|JSON|JSONB|XML|ARRAY|ENUM)\b").unwrap()
    });

    highlight_by_rules(code, &[&KW_RE, &TYPE_RE], &[Color::LightBlue, Color::Cyan])
}

fn highlight_shell(code: &str) -> String {
    static VAR_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\$\{?[a-zA-Z_][a-zA-Z0-9_]*\}?").unwrap());
    static SHELL_KW_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b(if|then|else|elif|fi|for|while|do|done|case|esac|in|function|return|exit|local|export|source|alias|unset|readonly|declare|typeset|eval|exec|trap|set|shift)\b").unwrap()
    });

    let mut result = String::new();

    for line in code.lines() {
        let temp = if line.trim().starts_with('#') {
            Style::new().fg(Color::Fixed(8)).paint(line).to_string()
        } else {
            let styled = SHELL_KW_RE
                .replace_all(line, |caps: &regex::Captures| {
                    Color::Purple.bold().paint(&caps[0]).to_string()
                })
                .to_string();

            VAR_RE
                .replace_all(&styled, |caps: &regex::Captures| {
                    Color::Cyan.paint(&caps[0]).to_string()
                })
                .to_string()
        };

        result.push_str(&temp);
        result.push('\n');
    }

    result.trim_end().to_string()
}

static STRING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""[^"]*"|'[^']*'|`[^`]*`"#).unwrap());

fn highlight_by_rules(code: &str, patterns: &[&LazyLock<Regex>], colors: &[Color]) -> String {
    let mut result = String::new();

    for line in code.lines() {
        if line.trim().starts_with("//")
            || (line.trim().starts_with("#") && !line.trim().starts_with("#!"))
        {
            result.push_str(&Style::new().fg(Color::Fixed(8)).paint(line).to_string());
            result.push('\n');
            continue;
        }

        let mut strings: Vec<String> = Vec::new();
        let with_placeholders = STRING_RE
            .replace_all(line, |caps: &regex::Captures| {
                let idx = strings.len();
                strings.push(caps[0].to_string());
                format!("\x00STR{}\x00", idx)
            })
            .to_string();

        let mut styled = with_placeholders;
        for (idx, pattern) in patterns.iter().enumerate() {
            let color = colors[idx];
            styled = pattern
                .replace_all(&styled, |caps: &regex::Captures| {
                    color.bold().paint(&caps[0]).to_string()
                })
                .to_string();
        }

        for (idx, original) in strings.iter().enumerate() {
            let placeholder = format!("\x00STR{}\x00", idx);
            let highlighted = Color::Yellow.paint(original.as_str()).to_string();
            styled = styled.replace(&placeholder, &highlighted);
        }

        result.push_str(&styled);
        result.push('\n');
    }

    result.trim_end().to_string()
}

fn render_table(header: &[&str], rows: &[Vec<&str>], _width: u16) -> String {
    if header.is_empty() {
        return String::new();
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .apply_modifier(UTF8_SOLID_INNER_BORDERS);

    let header_cells: Vec<Cell> = header
        .iter()
        .map(|h| Cell::new(render_inline(h)).add_attribute(Attribute::Bold))
        .collect();

    // A table with no body rows (e.g. an LLM response truncated right after the
    // `| h | h |` / `|---|---|` separator) would otherwise render as a hollow box
    // with a dangling header divider and an empty gap. Draw the header as a plain
    // bold row so the result is a clean, closed single-row box instead.
    if rows.is_empty() {
        table.add_row(header_cells);
        return table.to_string();
    }

    table.set_header(header_cells);

    for row in rows {
        let cells: Vec<Cell> = row.iter().map(|c| Cell::new(render_inline(c))).collect();
        table.add_row(cells);
    }

    table.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headings_colors() {
        let h1 = heading_style(1);
        let h2 = heading_style(2);
        let h3 = heading_style(3);

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
    }

    #[test]
    fn test_detect_error_warning() {
        assert!(detect_severity_pattern("ERROR: Something went wrong").is_some());
        assert!(detect_severity_pattern("Warning: Low disk space").is_some());
        assert!(detect_severity_pattern("Normal text").is_none());
    }

    #[test]
    fn test_parse_document_integration() {
        let md = "**bold** and *italic*";
        let rendered = render_terminal(md);
        assert!(!rendered.is_empty());
    }

    #[test]
    fn test_link_preserves_text_and_url() {
        let rendered = render_terminal("[file](file.pdf)");
        let plain = strip_ansi(&rendered);

        assert!(
            plain.contains("file"),
            "Expected link text, got: {:?}",
            plain
        );
        assert!(
            plain.contains("(file.pdf)"),
            "Expected URL in parentheses, got: {:?}",
            plain
        );
    }

    #[test]
    fn test_link_hides_url_when_same_as_text() {
        let rendered = render_terminal("[report.pdf](report.pdf)");
        let plain = strip_ansi(&rendered);

        assert!(
            plain.contains("report.pdf"),
            "link text missing: {:?}",
            plain
        );
        assert_eq!(
            plain.matches("report.pdf").count(),
            1,
            "URL should be hidden when same as text: {:?}",
            plain
        );
    }

    #[test]
    fn test_table_rendering() {
        let header = vec!["Col1", "Col2"];
        let rows = vec![vec!["Val1", "Val2"], vec!["Val3", "Val4"]];
        let rendered = render_table(&header, &rows, 80);

        assert!(rendered.contains("│"));
        assert!(rendered.contains("Col1"));
        assert!(rendered.contains("Val1"));
    }

    #[test]
    fn test_header_only_table_renders_closed_box() {
        // A table truncated right after the separator has a header but no rows.
        // It must render as a clean closed box, not a hollow one with a dangling
        // header divider and an empty gap (the `╞══╪══╡` + blank body artifact).
        let rendered = render_table(&["Parte", "Quién"], &[], 80);

        assert!(rendered.contains("Parte"));
        assert!(rendered.contains("Quién"));
        // No header-body divider (double line) when there is no body.
        assert!(
            !rendered.contains('╞'),
            "unexpected header divider: {rendered}"
        );
        assert!(
            !rendered.contains('═'),
            "unexpected double rule: {rendered}"
        );
    }

    #[test]
    fn test_code_highlighting_known_languages() {
        let rust_code = "fn main() { println!(\"Hello\"); }";
        let highlighted = highlight_code(rust_code, Some("rust"));
        assert!(!highlighted.is_empty());

        let py_code = "def foo():\n    pass";
        let highlighted_py = highlight_code(py_code, Some("python"));
        assert!(!highlighted_py.is_empty());
    }

    #[test]
    fn test_unknown_language_passthrough() {
        let code = "some random stuff";
        let result = highlight_code(code, Some("unknown_ext"));
        assert_eq!(result, code);
    }

    #[test]
    fn test_block_spacing() {
        let md = "## Heading\n\nA paragraph.\n\n- Item one\n- Item two\n\n```\ncode\n```\n\nAnother paragraph.\n";
        let rendered = render_terminal(md);

        let plain = strip_ansi(&rendered);

        let double_newlines = plain.matches("\n\n").count();
        assert!(
            double_newlines >= 1,
            "Expected at least 1 double-newline separator (code block), got {}.\nPlain:\n{}",
            double_newlines,
            plain
        );
    }

    #[test]
    fn test_json_highlight_no_ansi_corruption() {
        let json = r#"{
  "type": "function",
  "count": 42,
  "enabled": true,
  "value": null
}"#;
        let highlighted = highlight_json(json);

        let bytes = highlighted.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == 0x1b {
                assert!(
                    i + 1 < bytes.len() && bytes[i + 1] == b'[',
                    "Malformed ANSI escape at byte {}: \\x1b not followed by '['",
                    i
                );
            }
        }

        let plain = strip_ansi(&highlighted);
        assert!(plain.contains(r#""type": "function""#));
        assert!(plain.contains(r#""count": 42"#));
        assert!(plain.contains(r#""enabled": true"#));
        assert!(plain.contains(r#""value": null"#));
    }

    #[test]
    fn test_bold_rendering() {
        let md = "**Exchanges WITH record_event:**\n\n- item one\n- item two\n";
        let rendered = render_terminal(md);
        assert!(
            rendered.contains("\x1b[1m"),
            "Expected bold ANSI code in: {:?}",
            rendered
        );
        let plain = strip_ansi(&rendered);
        assert!(
            plain.contains("Exchanges WITH record_event:"),
            "Bold text missing from plain: {}",
            plain
        );
        assert!(
            !plain.contains("**"),
            "Raw ** markers should be consumed: {}",
            plain
        );
    }

    #[test]
    fn test_bold_in_numbered_list() {
        let md = "1. **src/file.rs:14** - Unused import\n2. **src/other.rs:100** - Dead code\n";
        let rendered = render_terminal(md);
        assert!(
            rendered.contains("\x1b[1m"),
            "Expected bold ANSI in numbered list: {:?}",
            rendered
        );
        let plain = strip_ansi(&rendered);
        assert!(!plain.contains("**"), "Raw ** in numbered list: {}", plain);
    }

    #[test]
    fn test_bold_ansi_to_tui_roundtrip() {
        use ansi_to_tui::IntoText;

        let md = "**Bold heading** and normal text";
        let rendered = render_terminal(md);
        let text = rendered.as_str().into_text().expect("ansi_to_tui failed");

        let has_bold_span = text.lines.iter().any(|line| {
            line.spans.iter().any(|span| {
                span.style
                    .add_modifier
                    .contains(ratatui::style::Modifier::BOLD)
                    && span.content.contains("Bold heading")
            })
        });
        assert!(
            has_bold_span,
            "Expected a bold span with 'Bold heading' in {:?}",
            text
        );
    }

    #[test]
    fn test_render_to_lines() {
        let md = "**bold** and `code`";
        let lines = render_to_lines(md);
        assert!(!lines.is_empty());
    }
}
