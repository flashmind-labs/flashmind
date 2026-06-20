//! In-memory cache for file contents with unified diff computation.
//!
//! Enables tracking of file state across agent turns so that tools can compute
//! diffs between the cached (previously seen) version and new content. Useful
//! for `str_replace`, `file_write`, and similar tools that modify files.
//!
//! [`FileCache`] is cheap to clone — it wraps an [`Arc`] internally, so all
//! clones share the same underlying store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use similar::{ChangeTag, TextDiff};

use crate::file_ops::to_relative_path;
use flashmind_types::tool::{DiffLine, FileDiff};

/// Shared in-memory cache mapping canonical file paths to their contents.
///
/// Clone is `O(1)` — all clones share the same [`DashMap`] via [`Arc`].
#[derive(Clone, Debug, Default)]
pub struct FileCache {
    inner: Arc<DashMap<PathBuf, String>>,
}

impl FileCache {
    /// Create a new, empty [`FileCache`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Store (or overwrite) the content for `path`.
    pub fn store(&self, path: &Path, content: String) {
        self.inner.insert(path.to_path_buf(), content);
    }

    /// Retrieve the cached content for `path`, or `None` if not cached.
    pub fn get(&self, path: &Path) -> Option<String> {
        self.inner.get(path).map(|v| v.clone())
    }

    /// Compute structured diff lines between the cached content and `new_content`.
    ///
    /// Returns `None` if:
    /// - `path` is not in the cache, or
    /// - the cached content equals `new_content` (no changes).
    ///
    /// Number of unchanged context lines to keep on each side of a change.
    const CONTEXT: usize = 2;

    pub fn diff(&self, path: &Path, new_content: &str) -> Option<Vec<DiffLine>> {
        let old_content = self.get(path)?;

        if old_content == new_content {
            return None;
        }

        let text_diff = TextDiff::from_lines(old_content.as_str(), new_content);

        // First pass: tag every change with its index and keep Equal lines
        // as Context variants so we can window them afterwards.
        #[derive(Clone, Copy)]
        enum Tag {
            Add,
            Del,
            Equal,
        }
        let raw: Vec<(Tag, u64, String)> = text_diff
            .iter_all_changes()
            .map(|change| {
                let line_no = change.old_index().or(change.new_index()).unwrap_or(0) as u64 + 1;
                let content = change.to_string_lossy().trim_end_matches('\n').to_string();
                let tag = match change.tag() {
                    ChangeTag::Insert => Tag::Add,
                    ChangeTag::Delete => Tag::Del,
                    ChangeTag::Equal => Tag::Equal,
                };
                (tag, line_no, content)
            })
            .collect();

        // Determine which raw indices are "interesting": an Add or Del, or an
        // Equal line within `CONTEXT` lines of one.  Two-sided window.
        let n = raw.len();
        let mut keep = vec![false; n];
        for (i, &(tag, _, _)) in raw.iter().enumerate() {
            if matches!(tag, Tag::Add | Tag::Del) {
                let lo = i.saturating_sub(Self::CONTEXT);
                let hi = (i + Self::CONTEXT).min(n - 1);
                for w in keep.iter_mut().take(hi + 1).skip(lo) {
                    *w = true;
                }
            }
        }

        let mut lines = Vec::new();
        let mut i = 0;
        let n = raw.len();
        while i < n {
            if keep[i] {
                // Emit a leading elision marker if we skipped lines to get
                // here (start of file or after a gap).
                if i > 0 && !keep[i - 1] {
                    lines.push(DiffLine::Context {
                        line: 0,
                        content: String::new(),
                    });
                }
                let (tag, line_no, content) = &raw[i];
                lines.push(match *tag {
                    Tag::Add => DiffLine::Added {
                        line: *line_no,
                        content: content.clone(),
                    },
                    Tag::Del => DiffLine::Removed {
                        line: *line_no,
                        content: content.clone(),
                    },
                    Tag::Equal => DiffLine::Context {
                        line: *line_no,
                        content: content.clone(),
                    },
                });
                i += 1;
            } else {
                // Skip the unkept run.  If we've already emitted something, a
                // trailing marker is added below when the next kept run starts
                // (handled by the leading-marker branch above).  If this is the
                // final run, emit a trailing marker now.
                let gap_start = i;
                while i < n && !keep[i] {
                    i += 1;
                }
                if !lines.is_empty() && i == n {
                    // Trailing gap at end of file.
                    lines.push(DiffLine::Context {
                        line: 0,
                        content: String::new(),
                    });
                }
                let _ = gap_start; // retained for clarity
            }
        }

        if lines.is_empty() {
            return None;
        }

        Some(lines)
    }

    /// Compute diff and return as a `Vec<FileDiff>` (empty if no diff).
    pub fn diff_vec(
        &self,
        path: &Path,
        new_content: &str,
        working_dir: Option<&PathBuf>,
    ) -> Vec<FileDiff> {
        self.diff(path, new_content)
            .map(|diff| {
                let path = to_relative_path(path, working_dir);
                vec![FileDiff { path, diff }]
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_is_empty() {
        let cache = FileCache::new();
        assert!(cache.get(Path::new("/tmp/foo.rs")).is_none());
    }

    #[test]
    fn store_and_get_roundtrip() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/hello.txt");

        cache.store(path, "hello world\n".to_string());

        assert_eq!(cache.get(path).unwrap(), "hello world\n");
    }

    #[test]
    fn store_overwrites_existing_entry() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/file.txt");

        cache.store(path, "version 1\n".to_string());
        cache.store(path, "version 2\n".to_string());

        assert_eq!(cache.get(path).unwrap(), "version 2\n");
    }

    #[test]
    fn get_returns_none_for_missing_path() {
        let cache = FileCache::new();
        assert!(cache.get(Path::new("/nonexistent/path.rs")).is_none());
    }

    #[test]
    fn diff_returns_none_when_not_cached() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/uncached.txt");

        assert!(cache.diff(path, "new content\n").is_none());
    }

    #[test]
    fn diff_returns_none_when_content_unchanged() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/same.txt");
        let content = "no changes here\n";

        cache.store(path, content.to_string());

        assert!(cache.diff(path, content).is_none());
    }

    #[test]
    fn diff_produces_structured_diff_on_change() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/changed.txt");

        cache.store(path, "line one\nline two\nline three\n".to_string());

        let result = cache
            .diff(path, "line one\nline TWO\nline three\n")
            .expect("expected a diff");

        // 3 lines total, change in the middle (index 1).  With CONTEXT=2 the
        // window covers all three lines, so we get: Context, Removed, Added,
        // Context.
        assert_eq!(result.len(), 4);
        assert!(matches!(&result[0], DiffLine::Context { content, .. } if content == "line one"));
        assert!(matches!(&result[1], DiffLine::Removed { content, .. } if content == "line two"));
        assert!(matches!(&result[2], DiffLine::Added { content, .. } if content == "line TWO"));
        assert!(matches!(&result[3], DiffLine::Context { content, .. } if content == "line three"));
    }

    #[test]
    fn diff_captures_added_and_removed_lines() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/ctx.txt");

        let old: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=10)
            .map(|i| {
                if i == 5 {
                    "line FIVE\n".to_string()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();

        cache.store(path, old);

        let diff = cache.diff(path, &new).unwrap();

        assert!(
            diff.iter()
                .any(|d| matches!(d, DiffLine::Removed { content, .. } if content == "line 5"))
        );
        assert!(
            diff.iter()
                .any(|d| matches!(d, DiffLine::Added { content, .. } if content == "line FIVE"))
        );
        // Change at line 5 (index 4).  CONTEXT=2 keeps indices 2..=6
        // (lines 3–7): Context(3), Context(4), Removed(5), Added(FIVE),
        // Context(6), Context(7).  Lines 1–2 are a leading gap and lines
        // 8–10 a trailing gap → one marker each.  Total = 6 + 2 = 8.
        assert_eq!(diff.len(), 8);
        assert_eq!(
            diff.iter()
                .filter(|d| matches!(d, DiffLine::Context { content, line: 0, .. } if content.is_empty()))
                .count(),
            2,
            "expected leading + trailing elision markers"
        );
    }

    #[test]
    fn diff_context_window_trims_distant_unchanged_lines() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/win.txt");
        // 20 lines; change only line 10.  Only lines 8–12 should survive plus
        // elision markers on each side — the far-away lines must be elided.
        let old: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=20)
            .map(|i| {
                if i == 10 {
                    "line TEN\n".to_string()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();
        cache.store(path, old);
        let diff = cache.diff(path, &new).unwrap();

        // Context lines 8,9 + Removed(10) + Added(TEN) + Context 11,12 = 6 lines
        // plus leading + trailing elision markers = 8 total.
        assert_eq!(diff.len(), 8);
        assert!(
            diff.iter()
                .any(|d| matches!(d, DiffLine::Context { content, .. } if content == "line 8"))
        );
        assert!(
            diff.iter()
                .any(|d| matches!(d, DiffLine::Context { content, .. } if content == "line 12"))
        );
        // No far-away lines leaked through.
        assert!(
            !diff
                .iter()
                .any(|d| matches!(d, DiffLine::Context { content, .. } if content == "line 1"))
        );
        assert!(
            !diff
                .iter()
                .any(|d| matches!(d, DiffLine::Context { content, .. } if content == "line 20"))
        );
    }

    #[test]
    fn clone_shares_underlying_store() {
        let cache1 = FileCache::new();
        let cache2 = cache1.clone();
        let path = Path::new("/tmp/shared.txt");

        cache1.store(path, "shared content\n".to_string());

        // cache2 sees the entry written via cache1
        assert_eq!(cache2.get(path).unwrap(), "shared content\n");
    }

    #[test]
    fn multiple_paths_stored_independently() {
        let cache = FileCache::new();
        let path_a = Path::new("/tmp/a.txt");
        let path_b = Path::new("/tmp/b.txt");

        cache.store(path_a, "alpha\n".to_string());
        cache.store(path_b, "beta\n".to_string());

        assert_eq!(cache.get(path_a).unwrap(), "alpha\n");
        assert_eq!(cache.get(path_b).unwrap(), "beta\n");
    }
}
