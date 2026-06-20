//! `@mention` file-path autocompletion for the REPL.
//!
//! [`CwdMentionProvider`] walks the current working directory (respecting
//! common ignore directories) and returns matching relative paths for the
//! [`flashmind_tui::MentionProvider`] trait.  [`extract_mentions`] parses
//! `@path` tokens out of submitted text so the caller can attach file contents
//! as context.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use flashmind_tui::MentionProvider;

/// Directories that are never offered as mention candidates.
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
        walk_dir(&self.root, &self.root, &mut out, 0);
        // Stable, readable ordering: shorter paths first, then alphabetical.
        out.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        *files = Some(out);
    }
}

impl MentionProvider for CwdMentionProvider {
    fn complete(&self, query: &str) -> Vec<String> {
        let mut guard = self.files.lock().unwrap();
        self.ensure_indexed(&mut guard);
        let all = guard.as_ref().expect("indexed");

        if query.is_empty() {
            return all.iter().take(MAX_CANDIDATES).cloned().collect();
        }
        let q = query.to_lowercase();

        // Two buckets: file-name matches (preferred) then full-path matches.
        let mut name_hits: Vec<&String> = Vec::new();
        let mut path_hits: Vec<&String> = Vec::new();
        for f in all {
            let lower = f.to_lowercase();
            if !lower.contains(&q) {
                continue;
            }
            // File name = last path segment.
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
}

/// Recursively walk `dir`, appending relative paths (forward slashes) to `out`.
fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<String>, depth: usize) {
    if depth > 6 || out.len() >= 2000 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        // Skip hidden files/dirs and ignored dirs.
        if name_str.starts_with('.') && name_str != ".well-known" {
            continue;
        }
        if path.is_dir() {
            if IGNORED_DIRS.contains(&name_str) {
                continue;
            }
            walk_dir(root, &path, out, depth + 1);
        } else if path.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            out.push(rel_str);
        }
    }
}

/// Parse `@path` tokens out of `text`.
///
/// A mention is an `@` at the start of the text or preceded by ASCII
/// whitespace, followed by a non-empty run of non-whitespace characters.  The
/// leading `@` is stripped from each returned path.  Duplicates are removed
/// while preserving order.
pub fn extract_mentions(text: &str) -> Vec<String> {
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
                // Trim surrounding punctuation that shouldn't be part of a path
                // (e.g. a trailing comma or period in prose).
                let raw = &text[start..end];
                let trimmed = raw.trim_end_matches(|c: char| {
                    matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | ')')
                });
                if !trimmed.is_empty() && !out.iter().any(|p: &String| p == trimmed) {
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
pub fn build_context_block(cwd: &Path, mentions: &[String]) -> Option<String> {
    if mentions.is_empty() {
        return None;
    }
    let mut block = String::from(
        "The user referenced the following files with @mentions. Their contents are included below for context:\n",
    );
    let mut any = false;
    for rel in mentions {
        any = true;
        match read_mentioned_file(cwd, rel) {
            Some(content) => {
                block.push_str(&format!("\n<file path=\"{rel}\">\n{content}\n</file>\n"));
            }
            None => {
                block.push_str(&format!(
                    "\n<file path=\"{rel}\">\n(unable to read: not found, not a file, or unreadable)\n</file>\n"
                ));
            }
        }
    }
    if any { Some(block) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_simple_mentions() {
        let text = "look at @src/main.rs and @README.md please";
        assert_eq!(extract_mentions(text), vec!["src/main.rs", "README.md"]);
    }

    #[test]
    fn ignores_at_inside_words() {
        assert!(extract_mentions("paco@hello.com").is_empty());
        assert_eq!(extract_mentions("email me at @x"), vec!["x"]);
    }

    #[test]
    fn deduplicates() {
        assert_eq!(extract_mentions("@a @a @b"), vec!["a", "b"]);
        assert_eq!(extract_mentions("@a @b @a"), vec!["a", "b"]);
    }

    #[test]
    fn trims_trailing_punctuation() {
        assert_eq!(
            extract_mentions("see @src/main.rs, then @lib.rs."),
            vec!["src/main.rs", "lib.rs"]
        );
    }

    #[test]
    fn start_of_text_mention() {
        assert_eq!(extract_mentions("@Cargo.toml is great"), vec!["Cargo.toml"]);
    }
}
