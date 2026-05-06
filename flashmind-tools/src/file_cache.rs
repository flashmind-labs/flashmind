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
    pub fn diff(&self, path: &Path, new_content: &str) -> Option<Vec<DiffLine>> {
        let old_content = self.get(path)?;

        if old_content == new_content {
            return None;
        }

        let text_diff = TextDiff::from_lines(old_content.as_str(), new_content);
        let mut lines = Vec::new();

        for change in text_diff.iter_all_changes() {
            let line_no = change.old_index().or(change.new_index()).unwrap_or(0) as u64 + 1;
            let content = change.to_string_lossy().trim_end_matches('\n').to_string();
            match change.tag() {
                ChangeTag::Insert => lines.push(DiffLine::Added {
                    line: line_no,
                    content,
                }),
                ChangeTag::Delete => lines.push(DiffLine::Removed {
                    line: line_no,
                    content,
                }),
                ChangeTag::Equal => {}
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

        assert_eq!(result.len(), 2);
        assert!(matches!(&result[0], DiffLine::Removed { content, .. } if content == "line two"));
        assert!(matches!(&result[1], DiffLine::Added { content, .. } if content == "line TWO"));
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
        assert_eq!(diff.len(), 2);
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
