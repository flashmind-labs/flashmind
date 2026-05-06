//! Shared Markdown parser — block-level + inline tokenizer.
//!
//! Produces a `Vec<DocBlock>` intermediate representation that platform-specific
//! renderers (Slack Block Kit, Slack mrkdwn, Telegram MarkdownV2) consume.
//! Inline formatting is parsed via nom into `InlineToken` spans.

use std::borrow::Cow;

use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::{tag, take_until, take_while, take_while1};
use nom::character::complete::char;
use nom::combinator::{not, opt, peek, recognize, verify};
use nom::error::Error as NomError;
use nom::sequence::{delimited, pair, preceded};

/// Inline formatting token produced by the markdown tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub enum InlineToken<'a> {
    Text(&'a str),
    Bold(&'a str),
    Italic(&'a str),
    Strike(&'a str),
    Code(&'a str),
    Link { text: &'a str, url: &'a str },
    Emoji(&'a str),
    Math(&'a str),
}

// ---------------------------------------------------------------------------
// Nom parsers for individual inline elements
// ---------------------------------------------------------------------------

fn bold<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    delimited(tag("**"), take_until("**"), tag("**"))
        .map(InlineToken::Bold)
        .parse(input)
}

fn strike<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    delimited(tag("~~"), take_until("~~"), tag("~~"))
        .map(InlineToken::Strike)
        .parse(input)
}

fn code<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    delimited(char('`'), take_until("`"), char('`'))
        .map(InlineToken::Code)
        .parse(input)
}

/// Parse `*content*` where the opening `*` is NOT followed by another `*`.
/// Content may contain `**bold**` pairs inside.
///
/// Follows the CommonMark left/right-flanking delimiter rules:
/// - Opening `*` must not be followed by whitespace
/// - Closing `*` must not be preceded by whitespace
fn italic<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    let mut open = preceded(char('*'), peek(not(char('*'))));
    let (_rest, _) = open.parse(input)?;

    let after_open = &input[1..];

    // Opening * must not be followed by whitespace (left-flanking delimiter rule)
    if after_open
        .as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_whitespace())
    {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Verify,
        )));
    }

    // Scan for closing single * (skipping ** pairs inside)
    let mut i = 0;
    let bytes = after_open.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'*' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                // Skip ** pair inside italic
                i += 2;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'*' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
                continue;
            }

            if i == 0 {
                return Err(nom::Err::Error(NomError::new(
                    input,
                    nom::error::ErrorKind::Verify,
                )));
            }

            // Closing * must not be preceded by whitespace (right-flanking delimiter rule)
            if bytes[i - 1].is_ascii_whitespace() {
                i += 1;
                continue;
            }

            return Ok((&after_open[i + 1..], InlineToken::Italic(&after_open[..i])));
        }
        i += 1;
    }

    Err(nom::Err::Error(NomError::new(
        input,
        nom::error::ErrorKind::TakeUntil,
    )))
}

fn link<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    let text = delimited(char('['), take_until("]"), char(']'));
    let url = delimited(char('('), take_until(")"), char(')'));

    pair(text, url)
        .map(|(text, url)| InlineToken::Link { text, url })
        .parse(input)
}

fn emoji<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    let name = verify(
        take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '-' || c == '+'),
        |s: &str| !s.is_empty(),
    );

    delimited(char(':'), name, char(':'))
        .map(InlineToken::Emoji)
        .parse(input)
}

fn math<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    alt((
        delimited(tag("$$"), take_until("$$"), tag("$$")),
        delimited(char('$'), take_until("$"), char('$')),
    ))
    .map(InlineToken::Math)
    .parse(input)
}

fn any_char<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    let (rest, c) = recognize(nom::character::complete::anychar).parse(input)?;
    Ok((rest, InlineToken::Text(c)))
}

fn inline_token<'a>(input: &'a str) -> nom::IResult<&'a str, InlineToken<'a>> {
    alt((bold, strike, code, italic, link, emoji, math, any_char)).parse(input)
}

/// Tokenize an inline markdown string into a sequence of [`InlineToken`]s.
///
/// Adjacent `Text` tokens are merged into single spans for efficiency.
pub fn tokenize_inline(input: &str) -> Vec<InlineToken<'_>> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::new();
    let mut remaining = input;

    while !remaining.is_empty() {
        match inline_token(remaining) {
            Ok((rest, token)) => {
                // Merge adjacent Text tokens by extending the previous slice
                if let InlineToken::Text(t) = &token
                    && let Some(InlineToken::Text(prev)) = tokens.last()
                {
                    let start = prev.as_ptr() as usize - input.as_ptr() as usize;
                    let end = t.as_ptr() as usize - input.as_ptr() as usize + t.len();
                    tokens.pop();
                    tokens.push(InlineToken::Text(&input[start..end]));
                    remaining = rest;
                    continue;
                }
                tokens.push(token);
                remaining = rest;
            }
            Err(_) => break,
        }
    }

    tokens
}

/// Split a table row's inner text (between outer pipes) by `|`,
/// respecting backtick-quoted spans.
pub fn split_table_cells(inner: &str) -> Vec<&str> {
    let mut cells = Vec::new();
    let mut start = 0;
    let mut in_backtick = false;

    for (i, c) in inner.char_indices() {
        if c == '`' {
            in_backtick = !in_backtick;
        } else if c == '|' && !in_backtick {
            cells.push(&inner[start..i]);
            start = i + 1;
        }
    }

    cells.push(&inner[start..]);
    cells
}

/// Strip inline Markdown markers, returning plain text.
pub fn strip_inline_markers(s: &str) -> String {
    let tokens = tokenize_inline(s);
    let mut out = String::with_capacity(s.len());

    for token in tokens {
        match token {
            InlineToken::Text(t)
            | InlineToken::Bold(t)
            | InlineToken::Italic(t)
            | InlineToken::Strike(t)
            | InlineToken::Code(t)
            | InlineToken::Math(t) => out.push_str(t),
            InlineToken::Link { text, .. } => out.push_str(text),
            InlineToken::Emoji(name) => {
                out.push(':');
                out.push_str(name);
                out.push(':');
            }
        }
    }

    out
}

/// Convert LaTeX math symbols to Unicode and unwrap text-wrapping commands.
pub fn latex_to_unicode(input: &str) -> String {
    const SYMBOLS: &[(&str, &str)] = &[
        (r"\rightarrow", "→"),
        (r"\leftarrow", "←"),
        (r"\Rightarrow", "⇒"),
        (r"\Leftarrow", "⇐"),
        (r"\leftrightarrow", "↔"),
        (r"\approx", "≈"),
        (r"\neq", "≠"),
        (r"\leq", "≤"),
        (r"\geq", "≥"),
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\gamma", "γ"),
        (r"\delta", "δ"),
        (r"\epsilon", "ε"),
        (r"\zeta", "ζ"),
        (r"\eta", "η"),
        (r"\theta", "θ"),
        (r"\iota", "ι"),
        (r"\kappa", "κ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\nu", "ν"),
        (r"\xi", "ξ"),
        (r"\pi", "π"),
        (r"\rho", "ρ"),
        (r"\sigma", "σ"),
        (r"\tau", "τ"),
        (r"\phi", "φ"),
        (r"\chi", "χ"),
        (r"\psi", "ψ"),
        (r"\omega", "ω"),
        (r"\infty", "∞"),
        (r"\pm", "±"),
        (r"\times", "×"),
        (r"\div", "÷"),
        (r"\partial", "∂"),
        (r"\nabla", "∇"),
        (r"\forall", "∀"),
        (r"\exists", "∃"),
    ];

    let mut result = strip_latex_text_commands(input);
    for (latex, unicode) in SYMBOLS {
        result = result.replace(latex, unicode);
    }
    result
}

/// Strip LaTeX text-wrapping commands like `\text{...}`, `\mathrm{...}`, `\mathbf{...}`.
fn strip_latex_text_commands(input: &str) -> String {
    const TEXT_COMMANDS: &[&str] = &[
        r"\text",
        r"\mathrm",
        r"\mathbf",
        r"\mathit",
        r"\mathsf",
        r"\mathtt",
        r"\mathcal",
        r"\mathbb",
        r"\mathfrak",
        r"\operatorname",
        r"\textrm",
        r"\textbf",
        r"\textit",
        r"\textsf",
        r"\texttt",
    ];

    let mut result = input.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for cmd in TEXT_COMMANDS {
            if let Some(pos) = result.find(cmd) {
                let after_cmd = pos + cmd.len();
                let rest = &result[after_cmd..];
                if let Some(inner) = extract_braced_content(rest) {
                    result = format!(
                        "{}{}{}",
                        &result[..pos],
                        inner,
                        &result[after_cmd + inner.len() + 2..]
                    );
                    changed = true;
                    break;
                }
            }
        }
    }
    result
}

/// Extract content inside balanced `{...}` braces, handling nesting.
fn extract_braced_content(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    if bytes.is_empty() || bytes[0] != b'{' {
        return None;
    }

    let mut depth = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(input[1..i].to_string());
                }
            }
            b'\\' => {
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Apply [`latex_to_unicode`] only if the input contains a backslash (heuristic
/// for LaTeX commands). Returns `Cow::Borrowed` when no conversion is needed.
pub fn maybe_latex(input: &str) -> Cow<'_, str> {
    if input.contains('\\') {
        Cow::Owned(latex_to_unicode(input))
    } else {
        Cow::Borrowed(input)
    }
}

// ===========================================================================
// Block-level document parsing (nom on raw &str)
// ===========================================================================

/// A block-level element in a parsed markdown document.
///
/// Text fields store raw markdown — renderers call `tokenize_inline()` to parse
/// inline formatting as needed.
#[derive(Debug, Clone, PartialEq)]
pub enum DocBlock<'a> {
    /// One or more non-blank lines of text.
    Paragraph(&'a str),
    /// `# Heading` — level is the number of `#` chars.
    Heading { level: u8, text: &'a str },
    /// Fenced code block with optional language identifier.
    CodeBlock {
        lang: Option<&'a str>,
        code: &'a str,
    },
    /// Markdown table — header row + data rows, each a vec of cell text.
    Table {
        header: Vec<&'a str>,
        rows: Vec<Vec<&'a str>>,
    },
    /// `- item` or `* item` lines.
    BulletList(Vec<&'a str>),
    /// `1. item`, `2. item` lines (prefix stripped). First element is the starting number.
    OrderedList(usize, Vec<&'a str>),
    /// `> quoted text` (prefix stripped, lines joined with newline).
    Blockquote(String),
    /// `---`, `***`, or `___`.
    HorizontalRule,
}

/// Result of parsing a markdown document.
///
/// Always returns both the successfully parsed blocks AND the remaining
/// unparsed input. This lets callers (especially streaming renderers)
/// know exactly how much was consumed and what's left.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseResult<'a> {
    /// Blocks successfully parsed.
    pub blocks: Vec<DocBlock<'a>>,
    /// Remaining input that was not consumed (after all parsed blocks).
    pub rest: &'a str,
    /// Input position just before the last block was parsed.
    /// Useful for streaming: render `input[..last_block_start]` to flush
    /// all blocks except the trailing one (which may still be accumulating).
    /// Empty slice when there are 0 or 1 blocks.
    pub before_last: &'a str,
    /// Whether parsing stopped due to an incomplete block (e.g. unclosed code fence).
    pub incomplete: bool,
}

/// Parse a markdown document into block-level elements.
///
/// Returns a [`ParseResult`] containing the parsed blocks, the remaining
/// unparsed input, and whether the parse ended due to an incomplete block
/// (e.g. unclosed code fence from a truncated stream).
pub fn parse_document(input: &str) -> ParseResult<'_> {
    let mut blocks = Vec::new();
    let mut rest = input;
    // Track the position before the most recently parsed block
    let mut before_last_block = input;

    while !rest.is_empty() {
        match alt((
            block_code_fence,
            block_horizontal_rule,
            block_table,
            block_bullet_list,
            block_ordered_list,
            block_blockquote,
            block_heading,
            block_blank_line,
            block_paragraph,
        ))
        .parse(rest)
        {
            Ok((remaining, DocBlock::Paragraph(""))) => {
                // Blank line sentinel — skip
                rest = remaining;
            }
            Ok((remaining, block)) => {
                before_last_block = rest;
                blocks.push(block);
                rest = remaining;
            }
            Err(nom::Err::Failure(_)) => {
                // Propagated from incomplete blocks (e.g. unclosed code fence)
                return ParseResult {
                    blocks,
                    rest,
                    before_last: before_last_block,
                    incomplete: true,
                };
            }
            Err(_) => break,
        }
    }

    let before_last = if blocks.len() >= 2 {
        before_last_block
    } else {
        input
    };

    ParseResult {
        blocks,
        rest,
        before_last,
        incomplete: false,
    }
}

// ---------------------------------------------------------------------------
// Helpers: line consumption
// ---------------------------------------------------------------------------

/// Trim only horizontal whitespace (spaces and tabs), preserving newlines.
fn trim_h(s: &str) -> &str {
    s.trim_start_matches([' ', '\t'])
}

/// Consume a single line (up to and including `\n`, or the rest if no newline).
fn take_line(input: &str) -> nom::IResult<&str, &str> {
    if input.is_empty() {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Eof,
        )));
    }

    match input.find('\n') {
        Some(pos) => Ok((&input[pos + 1..], &input[..pos])),
        None => Ok(("", input)),
    }
}

/// Consume and discard optional trailing `\n`.
fn opt_newline(input: &str) -> nom::IResult<&str, ()> {
    let (rest, _) = opt(char('\n')).parse(input)?;
    Ok((rest, ()))
}

// ---------------------------------------------------------------------------
// Block-level nom parsers
// ---------------------------------------------------------------------------

/// ``` lang?\n ... \n ```
fn block_code_fence<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let trimmed = trim_h(input);

    // Opening fence
    let (after_fence, _) = tag("```").parse(trimmed)?;
    let (after_lang, lang_line) = take_while(|c| c != '\n').parse(after_fence)?;
    let (mut rest, _) = opt_newline(after_lang)?;

    let lang = lang_line.trim();
    let lang = if lang.is_empty() { None } else { Some(lang) };

    // Collect body until closing ```
    let body_start = rest;
    let body_end;

    loop {
        if rest.is_empty() {
            // Unclosed fence — input is incomplete (truncated stream)
            return Err(nom::Err::Failure(NomError::new(
                input,
                nom::error::ErrorKind::Tag,
            )));
        }

        let line_trimmed = rest.trim_start();
        if line_trimmed.starts_with("```") {
            body_end = rest;
            // Skip closing fence line
            let (after_close, _) = take_while(|c| c != '\n').parse(rest)?;
            let (after_nl, _) = opt_newline(after_close)?;
            rest = after_nl;
            break;
        }

        let (after_line, _) = take_line(rest)?;
        rest = after_line;
    }

    // Code is everything between opening and closing fence
    let code_len = body_start.len() - body_end.len();
    let code = &body_start[..code_len];
    let code = code.strip_suffix('\n').unwrap_or(code);

    Ok((rest, DocBlock::CodeBlock { lang, code }))
}

/// `---`, `***`, `___` (3+ chars, only those chars and spaces)
fn block_horizontal_rule<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let (rest, line) = take_line(input)?;
    let trimmed = line.trim();

    let is_hr = trimmed.len() >= 3
        && (trimmed.starts_with("---") || trimmed.starts_with("***") || trimmed.starts_with("___"))
        && trimmed
            .chars()
            .all(|c| c == '-' || c == '*' || c == '_' || c == ' ');

    if !is_hr {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    Ok((rest, DocBlock::HorizontalRule))
}

/// Table rows — supports both piped (`| cell | cell |`) and pipe-less (`cell | cell`) formats.
fn block_table<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let first_trimmed = trim_h(input);

    // Detect table format: piped (outer |) or pipe-less
    // Minimum `| x |` = 5 chars for a valid piped table line
    let first_line_len = first_trimmed.find('\n').map_or(first_trimmed.len(), |pos| {
        first_trimmed[..pos].trim_end().len()
    });
    let piped = first_line_len >= 5
        && first_trimmed.starts_with('|')
        && first_trimmed
            .find('\n')
            .map_or(first_trimmed.ends_with('|'), |pos| {
                first_trimmed[..pos].trim_end().ends_with('|')
            });

    // Pipe-less: first line must contain ` | ` and second line must be separator
    let pipeless = !piped && {
        let first_line = first_trimmed
            .find('\n')
            .map_or(first_trimmed, |pos| first_trimmed[..pos].trim());
        first_line.contains(" | ")
            && first_trimmed
                .find('\n')
                .map(|pos| {
                    let second = first_trimmed[pos + 1..].lines().next().unwrap_or("");
                    is_table_separator_line(second)
                })
                .unwrap_or(false)
    };

    if !piped && !pipeless {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let mut all_rows: Vec<Vec<&str>> = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        let (after_line, line) = take_line(rest)?;
        let t = line.trim();

        if t.is_empty() {
            break;
        }

        if piped {
            // Piped format: require outer pipes and at least `| |`
            if t.len() < 3 || !(t.starts_with('|') && t.ends_with('|')) {
                break;
            }
            let inner = &t[1..t.len() - 1];
            if !is_table_separator(inner) {
                let cells = split_table_cells(inner).iter().map(|c| c.trim()).collect();
                all_rows.push(cells);
            }
        } else {
            // Pipe-less format: split on |, skip separator lines
            if !t.contains('|') {
                break;
            }
            if is_table_separator_line(t) {
                rest = after_line;
                continue;
            }
            let cells: Vec<&str> = split_table_cells(t).iter().map(|c| c.trim()).collect();
            all_rows.push(cells);
        }

        rest = after_line;
    }

    if all_rows.is_empty() {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Many1,
        )));
    }

    let header = all_rows.remove(0);
    Ok((
        rest,
        DocBlock::Table {
            header,
            rows: all_rows,
        },
    ))
}

/// `- item` or `* item` (not `**`)
fn block_bullet_list<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    if !is_bullet_item(trim_h(input)) {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let mut items = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        let (after_line, line) = take_line(rest)?;
        let t = line.trim_start();

        if !is_bullet_item(t) {
            // Allow a single blank line between bullet items (common in LLM output)
            if t.is_empty() || t == "\n" {
                // Peek past the blank line
                let next = after_line.trim_start_matches([' ', '\t']);
                if is_bullet_item(next.trim_start()) {
                    rest = after_line;
                    continue;
                }
            }
            break;
        }

        items.push(&t[2..]);
        rest = after_line;
    }

    Ok((rest, DocBlock::BulletList(items)))
}

/// `1. item`, `2. item`, etc.
///
/// Requires at least two consecutive ordered items. A lone `1. text` line
/// is treated as a paragraph so that numbered headings followed by bullet
/// sub-items don't get mis-rendered as a single-item ordered list.
fn block_ordered_list<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    if !is_ordered_item(trim_h(input)) {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    // Peek ahead: require at least two ordered items (possibly separated by a blank line)
    let (after_first, _) = take_line(input)?;
    let peek = if !after_first.is_empty() && trim_h(after_first).starts_with('\n') {
        // Skip one blank line for the peek check
        if let Ok((after_blank, _)) = take_line(after_first) {
            after_blank
        } else {
            after_first
        }
    } else {
        after_first
    };
    if peek.is_empty() || !is_ordered_item(trim_h(peek)) {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Many1,
        )));
    }

    let mut items = Vec::new();
    let mut rest = input;
    let mut start = 1usize;

    while !rest.is_empty() {
        let (after_line, line) = take_line(rest)?;
        let t = line.trim_start();

        if !is_ordered_item(t) {
            // Allow a single blank line between ordered items (common in LLM output)
            if t.is_empty() || t == "\n" {
                let next = after_line.trim_start_matches([' ', '\t']);
                if is_ordered_item(next.trim_start()) {
                    rest = after_line;
                    continue;
                }
            }
            break;
        }

        if items.is_empty() {
            // Extract starting number from first item
            if let Some(pos) = t.find(". ") {
                start = t[..pos].parse().unwrap_or(1);
            }
        }

        items.push(strip_ordered_prefix(t));
        rest = after_line;
    }

    Ok((rest, DocBlock::OrderedList(start, items)))
}

/// `> quoted text`
fn block_blockquote<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    if !trim_h(input).starts_with("> ") {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let mut parts: Vec<&str> = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        let (after_line, line) = take_line(rest)?;
        let t = line.trim_start();

        if !t.starts_with("> ") {
            break;
        }

        parts.push(&t[2..]);
        rest = after_line;
    }

    Ok((rest, DocBlock::Blockquote(parts.join("\n"))))
}

/// `# Heading`, `## Heading`, etc.
fn block_heading<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let trimmed = trim_h(input);
    if !trimmed.starts_with('#') {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let (rest, line) = take_line(input)?;
    let t = line.trim_start();

    let level = t.bytes().take_while(|&b| b == b'#').count() as u8;
    // '#' is ASCII, so byte count == char count — safe to slice
    let text = t.get(level as usize..).unwrap_or("").trim();

    if text.is_empty() {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Verify,
        )));
    }

    Ok((rest, DocBlock::Heading { level, text }))
}

/// Empty/whitespace-only line — consumed and discarded (no block produced).
/// We return a sentinel that `parse_document` will skip.
fn block_blank_line<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let (rest, line) = take_line(input)?;

    if !line.trim().is_empty() {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Verify,
        )));
    }

    // Return empty paragraph — filtered out by parse_document
    Ok((rest, DocBlock::Paragraph("")))
}

/// Fallback: collect consecutive non-blank, non-block lines as a paragraph.
fn block_paragraph<'a>(input: &'a str) -> nom::IResult<&'a str, DocBlock<'a>> {
    let mut rest = input;
    let start = input;

    while !rest.is_empty() {
        // Blank line ends the paragraph
        if rest.starts_with('\n') {
            break;
        }

        // Peek: would any block parser match here?
        let trimmed = trim_h(rest);
        if trimmed.is_empty()
            || trimmed.starts_with("```")
            || trimmed.starts_with('#')
            || trimmed.starts_with("> ")
            || trimmed.starts_with('|')
            || is_bullet_item(trimmed)
            || is_horizontal_rule_line(trimmed)
            || is_table_line(trimmed)
        {
            break;
        }

        // Only break for ordered items if they'd actually form a list (2+ consecutive)
        if is_ordered_item(trimmed)
            && let Ok((after, _)) = take_line(rest)
            && is_ordered_item(trim_h(after))
        {
            break;
        }

        let (after_line, _) = take_line(rest)?;
        rest = after_line;
    }

    let consumed = start.len() - rest.len();
    if consumed == 0 {
        return Err(nom::Err::Error(NomError::new(
            input,
            nom::error::ErrorKind::Many1,
        )));
    }

    let text = &start[..consumed];
    let text = text.strip_suffix('\n').unwrap_or(text);

    // Strip ordered-item prefix from lone numbered lines (e.g. "1. **Bold heading**")
    // so they render as plain paragraphs without the number.
    let text = if !text.contains('\n') && is_ordered_item(text.trim_start()) {
        strip_ordered_prefix(text.trim_start())
    } else {
        text
    };

    Ok((rest, DocBlock::Paragraph(text)))
}

// ---------------------------------------------------------------------------
// Block-level predicates
// ---------------------------------------------------------------------------

fn is_horizontal_rule_line(trimmed: &str) -> bool {
    trimmed.len() >= 3
        && (trimmed.starts_with("---") || trimmed.starts_with("***") || trimmed.starts_with("___"))
        && trimmed
            .chars()
            .all(|c| c == '-' || c == '*' || c == '_' || c == ' ')
}

fn is_table_line(trimmed: &str) -> bool {
    let first = if let Some(nl) = trimmed.find('\n') {
        trimmed[..nl].trim()
    } else {
        trimmed
    };

    // Piped table: | col | col | (needs at least `| x |`)
    if first.len() >= 5 && first.starts_with('|') && first.ends_with('|') {
        return true;
    }

    // Pipe-less table: look for "header | header\nsep" pattern
    // Need at least one pipe in first line AND a separator line following
    if first.contains(" | ")
        && let Some(nl) = trimmed.find('\n')
    {
        let second_line = trimmed[nl + 1..].lines().next().unwrap_or("");
        let second_trimmed = second_line.trim();
        return is_table_separator_line(second_trimmed);
    }

    false
}

fn is_bullet_item(trimmed: &str) -> bool {
    (trimmed.starts_with("- ") || trimmed.starts_with("* ")) && !trimmed.starts_with("**")
}

fn is_ordered_item(trimmed: &str) -> bool {
    if let Some(pos) = trimmed.find(". ") {
        pos <= 3 && pos > 0 && trimmed[..pos].bytes().all(|b| b.is_ascii_digit())
    } else {
        false
    }
}

fn strip_ordered_prefix(s: &str) -> &str {
    match s.find(". ") {
        Some(pos) => &s[pos + 2..],
        None => s,
    }
}

fn is_table_separator(inner: &str) -> bool {
    inner
        .chars()
        .all(|c| c == '-' || c == '|' || c == ':' || c == ' ')
}

/// Check if a full line is a table separator (e.g. `---|---|---` or `| --- | --- |`)
fn is_table_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Strip outer pipes if present (min length 2 covers the edge case
    // where trimmed is just `"|"` and both starts_with/ends_with match).
    let inner = if trimmed.len() >= 2 && trimmed.starts_with('|') && trimmed.ends_with('|') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };
    // Must contain at least one dash and only separator chars
    inner.contains('-') && is_table_separator(inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text() {
        assert_eq!(
            tokenize_inline("hello world"),
            vec![InlineToken::Text("hello world")]
        );
    }

    #[test]
    fn test_bold() {
        assert_eq!(tokenize_inline("**bold**"), vec![InlineToken::Bold("bold")]);
    }

    #[test]
    fn test_italic() {
        assert_eq!(
            tokenize_inline("*italic*"),
            vec![InlineToken::Italic("italic")]
        );
    }

    #[test]
    fn test_code() {
        assert_eq!(tokenize_inline("`code`"), vec![InlineToken::Code("code")]);
    }

    #[test]
    fn test_strike() {
        assert_eq!(
            tokenize_inline("~~strike~~"),
            vec![InlineToken::Strike("strike")]
        );
    }

    #[test]
    fn test_link() {
        assert_eq!(
            tokenize_inline("[click](https://example.com)"),
            vec![InlineToken::Link {
                text: "click",
                url: "https://example.com"
            }]
        );
    }

    #[test]
    fn test_emoji() {
        assert_eq!(
            tokenize_inline(":rocket:"),
            vec![InlineToken::Emoji("rocket")]
        );
    }

    #[test]
    fn test_mixed() {
        assert_eq!(
            tokenize_inline("Hello **world** and `code` here"),
            vec![
                InlineToken::Text("Hello "),
                InlineToken::Bold("world"),
                InlineToken::Text(" and "),
                InlineToken::Code("code"),
                InlineToken::Text(" here"),
            ]
        );
    }

    #[test]
    fn test_italic_containing_bold() {
        assert_eq!(
            tokenize_inline("*4. **EOSDA** - Free*"),
            vec![InlineToken::Italic("4. **EOSDA** - Free")]
        );
    }

    #[test]
    fn test_bold_and_italic() {
        assert_eq!(
            tokenize_inline("**bold** and *italic*"),
            vec![
                InlineToken::Bold("bold"),
                InlineToken::Text(" and "),
                InlineToken::Italic("italic"),
            ]
        );
    }

    #[test]
    fn test_strip_markers() {
        assert_eq!(
            strip_inline_markers("**bold** `code` *italic*"),
            "bold code italic"
        );
    }

    #[test]
    fn test_split_table_cells_basic() {
        assert_eq!(split_table_cells("a|b|c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_table_cells_backtick() {
        assert_eq!(split_table_cells("`a|b`|c"), vec!["`a|b`", "c"]);
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(tokenize_inline(""), Vec::<InlineToken>::new());
    }

    #[test]
    fn test_unclosed_bold_is_text() {
        assert_eq!(
            tokenize_inline("**unclosed"),
            vec![InlineToken::Text("**unclosed")]
        );
    }

    #[test]
    fn test_emoji_with_special_chars() {
        assert_eq!(
            tokenize_inline(":heavy_plus_sign:"),
            vec![InlineToken::Emoji("heavy_plus_sign")]
        );
    }

    #[test]
    fn test_colon_with_space_not_emoji() {
        // Space inside breaks emoji parsing
        let tokens = tokenize_inline("time: value");
        assert_eq!(tokens, vec![InlineToken::Text("time: value")]);
    }

    // =======================================================================
    // parse_document tests
    // =======================================================================

    #[test]
    fn doc_paragraph() {
        let r = parse_document("Hello world");
        assert_eq!(r.blocks, vec![DocBlock::Paragraph("Hello world")]);
        assert!(r.rest.is_empty());
    }

    #[test]
    fn doc_multi_line_paragraph() {
        let r = parse_document("line one\nline two");
        assert_eq!(r.blocks, vec![DocBlock::Paragraph("line one\nline two")]);
    }

    #[test]
    fn doc_two_paragraphs() {
        let r = parse_document("para one\n\npara two");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("para one"),
                DocBlock::Paragraph("para two"),
            ]
        );
    }

    #[test]
    fn doc_heading() {
        let r = parse_document("## Summary\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::Heading {
                level: 2,
                text: "Summary"
            }]
        );
    }

    #[test]
    fn doc_code_block() {
        let r = parse_document("```rust\nfn main() {}\n```\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::CodeBlock {
                lang: Some("rust"),
                code: "fn main() {}"
            }]
        );
    }

    #[test]
    fn doc_code_block_no_lang() {
        let r = parse_document("```\ncode\n```\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::CodeBlock {
                lang: None,
                code: "code"
            }]
        );
    }

    #[test]
    fn doc_horizontal_rule() {
        let r = parse_document("above\n\n---\n\nbelow");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("above"),
                DocBlock::HorizontalRule,
                DocBlock::Paragraph("below"),
            ]
        );
    }

    #[test]
    fn doc_bullet_list() {
        let r = parse_document("- one\n- two\n- three\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::BulletList(vec!["one", "two", "three"])]
        );
    }

    #[test]
    fn doc_ordered_list() {
        let r = parse_document("1. first\n2. second\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::OrderedList(1, vec!["first", "second"])]
        );
    }

    #[test]
    fn doc_blockquote() {
        let r = parse_document("> line one\n> line two\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::Blockquote("line one\nline two".to_string())]
        );
    }

    #[test]
    fn doc_table() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n";
        let r = parse_document(input);
        assert_eq!(
            r.blocks,
            vec![DocBlock::Table {
                header: vec!["A", "B"],
                rows: vec![vec!["1", "2"], vec!["3", "4"]],
            }]
        );
    }

    #[test]
    fn doc_mixed_blocks() {
        let input = "## Title\n\nSome text.\n\n- a\n- b\n\n```\ncode\n```\n";
        let r = parse_document(input);
        assert_eq!(r.blocks.len(), 4);
        assert!(matches!(r.blocks[0], DocBlock::Heading { level: 2, .. }));
        assert!(matches!(r.blocks[1], DocBlock::Paragraph(_)));
        assert!(matches!(r.blocks[2], DocBlock::BulletList(_)));
        assert!(matches!(r.blocks[3], DocBlock::CodeBlock { .. }));
    }

    #[test]
    fn doc_empty_input() {
        let r = parse_document("");
        assert!(r.blocks.is_empty());
        assert!(r.rest.is_empty());
    }

    #[test]
    fn doc_unclosed_code_fence() {
        // Simulates truncated streaming output — incomplete with blocks parsed so far
        let r = parse_document("```rust\nfn main() {\n    println!(\"hi\");\n");
        assert!(r.incomplete);
        assert!(r.blocks.is_empty());
        assert!(r.rest.starts_with("```rust"));
    }

    #[test]
    fn doc_unclosed_code_fence_after_content() {
        // Paragraph parsed successfully, then truncated code fence
        let input = "Hello world\n\n```rust\nfn main() {\n";
        let r = parse_document(input);
        assert!(r.incomplete);
        assert_eq!(r.blocks, vec![DocBlock::Paragraph("Hello world")]);
        assert!(r.rest.starts_with("```rust"));
    }

    #[test]
    fn doc_single_pipe_not_table() {
        // A lone "|" should not crash the table parser.
        // It stays as unparsed rest (could be a partial table row in streaming).
        let r = parse_document("|\n");
        assert!(r.blocks.is_empty());
        assert_eq!(r.rest, "|\n");
    }

    #[test]
    fn doc_partial_table() {
        // Only header row, no separator or data — still valid (complete)
        let r = parse_document("| A | B |\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::Table {
                header: vec!["A", "B"],
                rows: vec![],
            }]
        );
    }

    // =======================================================================
    // Additional parse_document tests — blank line splitting & edge cases
    // =======================================================================

    #[test]
    fn doc_three_paragraphs() {
        let r = parse_document("one\n\ntwo\n\nthree");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("one"),
                DocBlock::Paragraph("two"),
                DocBlock::Paragraph("three"),
            ]
        );
    }

    #[test]
    fn doc_paragraph_then_heading() {
        let r = parse_document("intro text\n\n## Section");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("intro text"),
                DocBlock::Heading {
                    level: 2,
                    text: "Section"
                },
            ]
        );
    }

    #[test]
    fn doc_heading_then_paragraph() {
        let r = parse_document("# Title\n\nBody text here.");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Heading {
                    level: 1,
                    text: "Title"
                },
                DocBlock::Paragraph("Body text here."),
            ]
        );
    }

    #[test]
    fn doc_paragraph_then_bullet_list() {
        let r = parse_document("Here are items:\n\n- one\n- two");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("Here are items:"),
                DocBlock::BulletList(vec!["one", "two"]),
            ]
        );
    }

    #[test]
    fn doc_paragraph_then_ordered_list() {
        let r = parse_document("Steps:\n\n1. first\n2. second");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("Steps:"),
                DocBlock::OrderedList(1, vec!["first", "second"]),
            ]
        );
    }

    #[test]
    fn doc_paragraph_then_code_block() {
        let r = parse_document("Example:\n\n```\nfoo()\n```\n");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("Example:"),
                DocBlock::CodeBlock {
                    lang: None,
                    code: "foo()"
                },
            ]
        );
    }

    #[test]
    fn doc_paragraph_then_blockquote() {
        let r = parse_document("He said:\n\n> hello world");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("He said:"),
                DocBlock::Blockquote("hello world".to_string()),
            ]
        );
    }

    #[test]
    fn doc_multiple_blank_lines_between_paragraphs() {
        let r = parse_document("first\n\n\n\nsecond");
        assert_eq!(
            r.blocks,
            vec![DocBlock::Paragraph("first"), DocBlock::Paragraph("second"),]
        );
    }

    #[test]
    fn doc_heading_then_list_then_code() {
        let input = "# Setup\n\n- install\n- configure\n\n```bash\nmake build\n```\n";
        let r = parse_document(input);
        assert_eq!(r.blocks.len(), 3);
        assert!(matches!(r.blocks[0], DocBlock::Heading { level: 1, .. }));
        assert!(matches!(r.blocks[1], DocBlock::BulletList(_)));
        assert!(matches!(r.blocks[2], DocBlock::CodeBlock { .. }));
    }

    #[test]
    fn doc_hr_between_paragraphs() {
        let r = parse_document("above\n\n---\n\nbelow");
        assert_eq!(
            r.blocks,
            vec![
                DocBlock::Paragraph("above"),
                DocBlock::HorizontalRule,
                DocBlock::Paragraph("below"),
            ]
        );
    }

    #[test]
    fn doc_hr_standalone() {
        let r = parse_document("---");
        assert_eq!(r.blocks, vec![DocBlock::HorizontalRule]);
    }

    #[test]
    fn doc_table_between_paragraphs() {
        let input = "Results:\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nEnd.";
        let r = parse_document(input);
        assert_eq!(r.blocks.len(), 3);
        assert!(matches!(r.blocks[0], DocBlock::Paragraph("Results:")));
        assert!(matches!(r.blocks[1], DocBlock::Table { .. }));
        assert!(matches!(r.blocks[2], DocBlock::Paragraph("End.")));
    }

    #[test]
    fn doc_blockquote_multiline() {
        let r = parse_document("> line one\n> line two\n> line three\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::Blockquote(
                "line one\nline two\nline three".to_string()
            )]
        );
    }

    #[test]
    fn doc_only_blank_lines() {
        let r = parse_document("\n\n\n");
        assert!(r.blocks.is_empty());
    }

    #[test]
    fn doc_code_block_multiline() {
        let input = "```python\ndef foo():\n    return 42\n```\n";
        let r = parse_document(input);
        assert_eq!(
            r.blocks,
            vec![DocBlock::CodeBlock {
                lang: Some("python"),
                code: "def foo():\n    return 42"
            }]
        );
    }

    #[test]
    fn doc_bullet_list_star_prefix() {
        let r = parse_document("* one\n* two\n");
        assert_eq!(r.blocks, vec![DocBlock::BulletList(vec!["one", "two"])]);
    }

    #[test]
    fn doc_parse_result_rest() {
        // Verify that rest is properly returned
        let r = parse_document("# Title\n\nsome trailing");
        assert_eq!(r.blocks.len(), 2);
        assert!(r.rest.is_empty());
        assert!(!r.incomplete);
    }

    // =======================================================================
    // Inline tokenizer edge cases
    // =======================================================================

    #[test]
    fn test_nested_bold_in_italic() {
        assert_eq!(
            tokenize_inline("*text **bold** more*"),
            vec![InlineToken::Italic("text **bold** more")]
        );
    }

    #[test]
    fn test_adjacent_bold_italic() {
        assert_eq!(
            tokenize_inline("**bold***italic*"),
            vec![InlineToken::Bold("bold"), InlineToken::Italic("italic"),]
        );
    }

    #[test]
    fn test_link_with_parens_in_text() {
        assert_eq!(
            tokenize_inline("[click (here)](https://example.com)"),
            vec![InlineToken::Link {
                text: "click (here)",
                url: "https://example.com"
            }]
        );
    }

    #[test]
    fn test_emoji_in_text() {
        assert_eq!(
            tokenize_inline("hello :wave: world"),
            vec![
                InlineToken::Text("hello "),
                InlineToken::Emoji("wave"),
                InlineToken::Text(" world"),
            ]
        );
    }

    #[test]
    fn test_multiple_code_spans() {
        assert_eq!(
            tokenize_inline("`foo` and `bar`"),
            vec![
                InlineToken::Code("foo"),
                InlineToken::Text(" and "),
                InlineToken::Code("bar"),
            ]
        );
    }

    #[test]
    fn test_unclosed_italic_is_text() {
        let tokens = tokenize_inline("*unclosed");
        assert_eq!(tokens, vec![InlineToken::Text("*unclosed")]);
    }

    #[test]
    fn test_unclosed_strike_is_text() {
        let tokens = tokenize_inline("~~unclosed");
        assert_eq!(tokens, vec![InlineToken::Text("~~unclosed")]);
    }

    #[test]
    fn test_double_star_not_bold_when_empty() {
        // "****" — two bold delimiters with empty content
        let tokens = tokenize_inline("****");
        assert_eq!(tokens, vec![InlineToken::Bold("")]);
    }

    #[test]
    fn test_strip_markers_preserves_emoji() {
        assert_eq!(strip_inline_markers(":rocket: Launch"), ":rocket: Launch");
    }

    #[test]
    fn test_strip_markers_link() {
        assert_eq!(strip_inline_markers("[click](https://x.com)"), "click");
    }

    #[test]
    fn doc_pipeless_table() {
        let input = "Subagent ID | Theme | Status\n---|---|---\n`abc-123` | Sea | Completed\n`def-456` | Mountains | Completed";
        let r = parse_document(input);
        assert_eq!(r.blocks.len(), 1, "blocks: {:?}", r.blocks);
        match &r.blocks[0] {
            DocBlock::Table { header, rows } => {
                assert_eq!(header, &vec!["Subagent ID", "Theme", "Status"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], "`abc-123`");
                assert_eq!(rows[0][1], "Sea");
            }
            other => panic!("Expected Table, got: {:?}", other),
        }
    }

    #[test]
    fn doc_pipeless_table_in_document() {
        let input = "Summary\n\nSubagent ID | Theme | Status\n---|---|---\n`abc` | Sea | Done\n\nMore text.";
        let r = parse_document(input);
        assert_eq!(r.blocks.len(), 3, "blocks: {:?}", r.blocks);
        assert!(matches!(r.blocks[0], DocBlock::Paragraph("Summary")));
        assert!(matches!(r.blocks[1], DocBlock::Table { .. }));
        assert!(matches!(r.blocks[2], DocBlock::Paragraph("More text.")));
    }

    #[test]
    fn doc_lone_ordered_item_is_paragraph() {
        let input = "1. **Airport Access Restrictions:**\n\n- Item one\n- Item two\n\n1. **Operational Status:**\n\n- Item three";
        let r = parse_document(input);

        assert_eq!(
            r.blocks[0],
            DocBlock::Paragraph("**Airport Access Restrictions:**"),
            "Lone ordered item should become paragraph with prefix stripped"
        );
        assert!(matches!(r.blocks[1], DocBlock::BulletList(_)));
        assert_eq!(r.blocks[2], DocBlock::Paragraph("**Operational Status:**"),);
        assert!(matches!(r.blocks[3], DocBlock::BulletList(_)));
    }

    #[test]
    fn doc_consecutive_ordered_items_still_list() {
        let r = parse_document("1. first\n2. second\n3. third");
        assert_eq!(
            r.blocks,
            vec![DocBlock::OrderedList(1, vec!["first", "second", "third"])]
        );
    }

    #[test]
    fn doc_spaced_ordered_list() {
        // LLMs commonly emit ordered lists with blank lines between items
        let r = parse_document("1. first\n\n2. second\n\n3. third\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::OrderedList(1, vec!["first", "second", "third"])]
        );
    }

    #[test]
    fn doc_spaced_bullet_list() {
        let r = parse_document("- alpha\n\n- beta\n\n- gamma\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::BulletList(vec!["alpha", "beta", "gamma"])]
        );
    }

    #[test]
    fn doc_spaced_ordered_list_preserves_start() {
        let r = parse_document("3. third item\n\n4. fourth item\n");
        assert_eq!(
            r.blocks,
            vec![DocBlock::OrderedList(3, vec!["third item", "fourth item"])]
        );
    }

    #[test]
    fn latex_symbols_in_plain_text() {
        assert_eq!(
            latex_to_unicode("Input \\rightarrow Processing \\rightarrow Output"),
            "Input → Processing → Output",
        );
    }

    #[test]
    fn latex_text_command_stripping() {
        assert_eq!(
            latex_to_unicode("\\text{Signal Handler} \\rightarrow \\text{Order Router}"),
            "Signal Handler → Order Router",
        );
    }

    #[test]
    fn latex_mathbf_command_stripping() {
        assert_eq!(latex_to_unicode("\\mathbf{P} \\approx 0.95"), "P ≈ 0.95",);
    }

    #[test]
    fn latex_nested_text_commands() {
        assert_eq!(
            latex_to_unicode("\\text{\\text{inner value}}"),
            "inner value",
        );
    }

    #[test]
    fn latex_multiple_text_commands() {
        assert_eq!(
            latex_to_unicode(
                "\\mathrm{Cache} \\rightarrow \\text{Dispatcher} \\rightarrow \\mathtt{Worker}"
            ),
            "Cache → Dispatcher → Worker",
        );
    }

    #[test]
    fn latex_mixed_symbols_and_text_commands() {
        assert_eq!(
            latex_to_unicode(
                "\\text{Latency} \\leq \\text{Threshold} \\forall \\sigma \\leq \\epsilon"
            ),
            "Latency ≤ Threshold ∀ σ ≤ ε",
        );
    }

    #[test]
    fn latex_no_backslash_untouched() {
        assert_eq!(
            latex_to_unicode("plain text stays the same"),
            "plain text stays the same",
        );
    }

    #[test]
    fn latex_braced_content_with_nesting() {
        assert_eq!(
            latex_to_unicode("\\operatorname{rank}(A) \\times \\alpha"),
            "rank(A) × α",
        );
    }

    #[test]
    fn latex_unclosed_brace_not_stripped() {
        assert_eq!(
            latex_to_unicode("\\text{missing close"),
            "\\text{missing close",
        );
    }

    #[test]
    fn extract_braced_simple() {
        assert_eq!(extract_braced_content("{hello}"), Some("hello".to_string()));
    }

    #[test]
    fn extract_braced_nested() {
        assert_eq!(extract_braced_content("{a{b}c}"), Some("a{b}c".to_string()));
    }

    #[test]
    fn extract_braced_no_open() {
        assert_eq!(extract_braced_content("hello}"), None);
    }

    #[test]
    fn extract_braced_escaped() {
        assert_eq!(extract_braced_content(r"{a\}b}"), Some(r"a\}b".to_string()));
    }

    #[test]
    fn latex_math_token_and_bare_text() {
        let tokens = tokenize_inline("$x \\rightarrow y$ and \\text{done} \\rightarrow ok");
        let math_count = tokens
            .iter()
            .filter(|t| matches!(t, InlineToken::Math(_)))
            .count();
        assert_eq!(math_count, 1, "should parse exactly one $...$ math token");

        let rendered: String = tokens
            .iter()
            .map(|t| match t {
                InlineToken::Math(s) => latex_to_unicode(s),
                InlineToken::Text(s) => {
                    if s.contains('\\') {
                        latex_to_unicode(s)
                    } else {
                        s.to_string()
                    }
                }
                other => strip_inline_markers(&format!("{:?}", other)),
            })
            .collect();
        assert!(
            rendered.contains("→"),
            "arrows should be converted: {:?}",
            rendered
        );
        assert!(
            rendered.contains("done"),
            "\\text{{done}} should be stripped: {:?}",
            rendered
        );
        assert!(
            rendered.contains("ok"),
            "trailing text preserved: {:?}",
            rendered
        );
    }

    #[test]
    fn latex_inside_bold() {
        let tokens = tokenize_inline("**\\text{Pipeline} \\rightarrow \\text{Output}**");
        assert_eq!(tokens.len(), 1, "should be a single bold token");
        match &tokens[0] {
            InlineToken::Bold(t) => {
                assert_eq!(latex_to_unicode(t), "Pipeline → Output");
            }
            other => panic!("Expected Bold, got {:?}", other),
        }
    }

    #[test]
    fn latex_inside_italic() {
        let tokens = tokenize_inline("*\\alpha \\leq \\beta*");
        match &tokens[0] {
            InlineToken::Italic(t) => {
                assert_eq!(latex_to_unicode(t), "α ≤ β");
            }
            other => panic!("Expected Italic, got {:?}", other),
        }
    }

    #[test]
    fn latex_bold_mixed_with_plain() {
        let tokens = tokenize_inline("Result: **\\text{Mean} \\pm \\sigma** confirmed");
        assert_eq!(tokens.len(), 3);
        match (&tokens[0], &tokens[1], &tokens[2]) {
            (InlineToken::Text(pre), InlineToken::Bold(bold), InlineToken::Text(post)) => {
                assert_eq!(*pre, "Result: ");
                assert_eq!(latex_to_unicode(bold), "Mean ± σ");
                assert_eq!(*post, " confirmed");
            }
            other => panic!("Unexpected token layout: {:?}", other),
        }
    }

    #[test]
    fn maybe_latex_passthrough() {
        use std::borrow::Cow;
        let result = maybe_latex("no latex here");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, "no latex here");
    }

    #[test]
    fn maybe_latex_converts() {
        use std::borrow::Cow;
        let result = maybe_latex("\\text{hello} \\rightarrow world");
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(&*result, "hello → world");
    }

    #[test]
    fn latex_flow_diagram_bare() {
        let input = "\\text{Quotes} \\rightarrow \\text{Mid Price} \\rightarrow \\text{Spread Calc} \\rightarrow \\text{Signal}";
        assert_eq!(
            latex_to_unicode(input),
            "Quotes → Mid Price → Spread Calc → Signal",
        );
    }

    #[test]
    fn latex_flow_diagram_dollar() {
        let input = "$\\text{Quotes} \\rightarrow \\text{Mid Price} \\rightarrow \\text{Spread Calc} \\rightarrow \\text{Signal}$";
        let tokens = tokenize_inline(input);
        match &tokens[0] {
            InlineToken::Math(t) => {
                assert_eq!(
                    latex_to_unicode(t),
                    "Quotes → Mid Price → Spread Calc → Signal",
                );
            }
            other => panic!("Expected Math, got {:?}", other),
        }
    }
}
