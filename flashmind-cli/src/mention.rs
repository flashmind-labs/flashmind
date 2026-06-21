//! `@mention` file-path autocompletion for the REPL.
//!
//! [`CwdMentionProvider`] walks the current working directory (respecting
//! common ignore directories) and returns matching relative paths for the
//! [`flashmind_tui::MentionProvider`] trait.  [`extract_mentions`] parses
//! `@path` tokens out of submitted text so the caller can attach file contents
//! as context.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use flashmind_tui::MentionProvider;

/// Directories that are never offered as mention candidates (fallback when
/// `.gitignore` is absent or doesn't cover these).
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "Pods",
    ".gradle",
    ".idea",
    ".vscode",
    "coverage",
    ".cache",
];

/// Maximum number of candidates to surface.
const MAX_CANDIDATES: usize = 40;
/// Maximum file size (in bytes) we'll attach as context.
const MAX_CONTEXT_BYTES: usize = 256 * 1024;
/// Maximum number of content search results.
const MAX_SEARCH_RESULTS: usize = 20;

/// Cwd-scoped file-path completion provider.
///
/// Walks the working directory once (lazily, on first [`complete`](Self::complete)
/// call) and caches the relative path list.  Subsequent calls filter the cache
/// by case-insensitive substring match, preferring paths whose *file name*
/// contains the query.
#[derive(Debug)]
pub struct CwdMentionProvider {
    root: PathBuf,
    files: Mutex<Option<Vec<String>>>,
}

impl CwdMentionProvider {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            files: Mutex::new(None),
        }
    }

    fn ensure_indexed(&self, files: &mut Option<Vec<String>>) {
        if files.is_some() {
            return;
        }
        let mut out = Vec::new();
        let mut builder = ignore::WalkBuilder::new(&self.root);
        builder
            .max_depth(Some(7))
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true);
        for entry in builder.build().flatten() {
            if out.len() >= 2000 {
                break;
            }
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.starts_with('.') && name != ".well-known"
            {
                continue;
            }
            if let Ok(rel) = path.strip_prefix(&self.root) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if !IGNORED_DIRS
                    .iter()
                    .any(|d| rel_str.starts_with(d) || rel_str.contains(&format!("/{d}/")))
                {
                    out.push(rel_str);
                }
            }
        }
        out.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        *files = Some(out);
    }
}

impl MentionProvider for CwdMentionProvider {
    fn complete(&self, query: &str) -> Vec<String> {
        let mut guard = self.files.lock().unwrap();
        self.ensure_indexed(&mut guard);
        let all = guard.as_ref().expect("indexed");

        // Strip `:N` line-number suffix from query before matching.
        let (path_query, _line) = split_line_ref(query);
        let q = path_query.to_lowercase();

        if q.is_empty() {
            return all.iter().take(MAX_CANDIDATES).cloned().collect();
        }

        // Two buckets: file-name matches (preferred) then full-path matches.
        let mut name_hits: Vec<&String> = Vec::new();
        let mut path_hits: Vec<&String> = Vec::new();
        for f in all {
            let lower = f.to_lowercase();
            if !lower.contains(&q) {
                continue;
            }
            let basename = f.rsplit(['/', '\\']).next().unwrap_or(f);
            if basename.to_lowercase().contains(&q) {
                name_hits.push(f);
            } else {
                path_hits.push(f);
            }
            if name_hits.len() + path_hits.len() >= MAX_CANDIDATES {
                break;
            }
        }
        name_hits
            .iter()
            .chain(path_hits.iter())
            .map(|s| (*s).clone())
            .collect()
    }

    fn search_content(&self, query: &str) -> Vec<String> {
        search_content(&self.root, query)
    }
}

/// A resolved `@path` or `@path:line` mention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionRef {
    pub path: String,
    pub line: Option<usize>,
}

/// Split a path reference into `(path, optional_line)`.
fn split_line_ref(s: &str) -> (&str, Option<usize>) {
    if let Some(colon) = s.rfind(':') {
        let after = &s[colon + 1..];
        if !after.is_empty()
            && after.chars().all(|c| c.is_ascii_digit())
            && let Ok(n) = after.parse::<usize>()
        {
            return (&s[..colon], Some(n));
        }
    }
    (s, None)
}

/// Parse `@path` and `@path:line` tokens out of `text`.
///
/// A mention is an `@` at the start of the text or preceded by ASCII
/// whitespace, followed by a non-empty run of non-whitespace characters.  The
/// leading `@` is stripped.  Duplicates are removed while preserving order.
pub fn extract_mentions(text: &str) -> Vec<MentionRef> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            if end > start {
                let raw = &text[start..end];
                let trimmed = raw.trim_end_matches(|c: char| {
                    matches!(c, ',' | '.' | ';' | '!' | '?' | ')')
                });
                if !trimmed.is_empty() {
                    let (path, line) = split_line_ref(trimmed);
                    let path_s = path.to_string();
                    if !out.iter().any(|m: &MentionRef| m.path == path_s && m.line == line) {
                        out.push(MentionRef {
                            path: path_s,
                            line,
                        });
                    }
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Parse `#query` tokens out of `text` for content search.
pub fn extract_content_searches(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            if end > start {
                let raw = &text[start..end];
                let trimmed = raw.trim_end_matches(|c: char| {
                    matches!(c, ',' | '.' | ';' | '!' | '?' | ')')
                });
                if !trimmed.is_empty() && !out.contains(&trimmed.to_string()) {
                    out.push(trimmed.to_string());
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Search file contents under `root` for lines matching `query`.
fn search_content(root: &Path, query: &str) -> Vec<String> {
    let q_lower = query.to_lowercase();
    let mut results = Vec::new();
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .max_depth(Some(7))
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    for entry in builder.build().flatten() {
        if results.len() >= MAX_SEARCH_RESULTS {
            break;
        }
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy();
        if IGNORED_DIRS
            .iter()
            .any(|d| rel_str.starts_with(d) || rel_str.contains(&format!("/{d}/")))
        {
            continue;
        }

        let Ok(file) = std::fs::File::open(path) else {
            continue;
        };
        let reader = BufReader::new(file);
        for (line_num, line) in reader.lines().enumerate() {
            if results.len() >= MAX_SEARCH_RESULTS {
                break;
            }
            let Ok(line) = line else { break };
            if line.to_lowercase().contains(&q_lower) {
                let preview: String = line.chars().take(80).collect();
                results.push(format!("{}:{}: {}", rel_str, line_num + 1, preview));
            }
        }
    }
    results
}

/// Read a mentioned file's contents, capped at [`MAX_CONTEXT_BYTES`].
///
/// Returns `None` for missing files, binary/non-UTF-8 files, or oversized
/// files (in which case a placeholder is returned instead so the model knows
/// the file exists but is too large).
pub fn read_mentioned_file(cwd: &Path, rel: &str) -> Option<String> {
    let path = cwd.join(rel);
    let meta = std::fs::metadata(&path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    // Binary detection: NUL byte in the first 8KB ⇒ skip.
    let sniff = &bytes[..bytes.len().min(8192)];
    if sniff.contains(&0u8) {
        return Some(format!("<binary file, {} bytes>", bytes.len()));
    }
    if bytes.len() > MAX_CONTEXT_BYTES {
        return Some(format!(
            "<file too large: {} bytes; truncated>",
            bytes.len()
        ));
    }
    let content = String::from_utf8(bytes).ok()?;
    Some(content)
}

/// Build a single context block string for a set of mentioned files.
///
/// Files that cannot be read are noted by name so the model is aware they were
/// referenced but inaccessible.  Returns `None` if no mentions resolved to any
/// content (readable or not).
pub fn build_context_block(cwd: &Path, mentions: &[MentionRef]) -> Option<String> {
    if mentions.is_empty() {
        return None;
    }
    let mut block = String::from(
        "The user referenced the following files with @mentions. Their contents are included below for context:\n",
    );
    let mut any = false;
    for mention in mentions {
        any = true;
        let rel = &mention.path;
        match (read_mentioned_file(cwd, rel), mention.line) {
            (Some(content), Some(line)) => {
                let lines: Vec<&str> = content.lines().collect();
                let start = line.saturating_sub(11);
                let end = (line + 20).min(lines.len());
                let window: String = lines
                    .get(start..end)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                    .map(|(i, l)| format!("{:>4} {l}", start + i + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                block.push_str(&format!(
                    "\n<file path=\"{rel}\" lines=\"{}-{}\">\n{window}\n</file>\n",
                    start + 1,
                    end
                ));
            }
            (Some(content), None) => {
                block.push_str(&format!("\n<file path=\"{rel}\">\n{content}\n</file>\n"));
            }
            (None, _) => {
                block.push_str(&format!(
                    "\n<file path=\"{rel}\">\n(unable to read: not found, not a file, or unreadable)\n</file>\n"
                ));
            }
        }
    }
    if any { Some(block) } else { None }
}

/// Build context from `#query` content search results.
pub fn build_search_context_block(cwd: &Path, queries: &[String]) -> Option<String> {
    if queries.is_empty() {
        return None;
    }
    let mut block = String::from(
        "The user searched for the following content with #queries:\n",
    );
    let mut any = false;
    for query in queries {
        let results = search_content(cwd, query);
        if results.is_empty() {
            continue;
        }
        any = true;
        block.push_str(&format!("\n## Results for #{query}\n"));
        for result in &results {
            block.push_str(&format!("  {result}\n"));
        }
    }
    if any { Some(block) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mref(path: &str) -> MentionRef {
        MentionRef {
            path: path.to_string(),
            line: None,
        }
    }

    fn mref_line(path: &str, line: usize) -> MentionRef {
        MentionRef {
            path: path.to_string(),
            line: Some(line),
        }
    }

    #[test]
    fn extracts_simple_mentions() {
        let text = "look at @src/main.rs and @README.md please";
        assert_eq!(
            extract_mentions(text),
            vec![mref("src/main.rs"), mref("README.md")]
        );
    }

    #[test]
    fn ignores_at_inside_words() {
        assert!(extract_mentions("paco@hello.com").is_empty());
        assert_eq!(extract_mentions("email me at @x"), vec![mref("x")]);
    }

    #[test]
    fn deduplicates() {
        assert_eq!(extract_mentions("@a @a @b"), vec![mref("a"), mref("b")]);
        assert_eq!(extract_mentions("@a @b @a"), vec![mref("a"), mref("b")]);
    }

    #[test]
    fn trims_trailing_punctuation() {
        assert_eq!(
            extract_mentions("see @src/main.rs, then @lib.rs."),
            vec![mref("src/main.rs"), mref("lib.rs")]
        );
    }

    #[test]
    fn start_of_text_mention() {
        assert_eq!(
            extract_mentions("@Cargo.toml is great"),
            vec![mref("Cargo.toml")]
        );
    }

    #[test]
    fn extracts_line_references() {
        assert_eq!(
            extract_mentions("@src/main.rs:42"),
            vec![mref_line("src/main.rs", 42)]
        );
        assert_eq!(
            extract_mentions("check @lib.rs:100 and @util.rs"),
            vec![mref_line("lib.rs", 100), mref("util.rs")]
        );
    }

    #[test]
    fn extracts_content_searches() {
        assert_eq!(
            extract_content_searches("look for #ToolContext in code"),
            vec!["ToolContext"]
        );
        assert!(extract_content_searches("no hash").is_empty());
        assert_eq!(
            extract_content_searches("#foo #bar"),
            vec!["foo", "bar"]
        );
    }
}
