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
use similar::TextDiff;

use crate::file_ops::to_relative_path;
use flashmind_types::tool::FileDiff;

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

    /// Compute a unified diff between the cached content and `new_content`.
    ///
    /// Returns `None` if:
    /// - `path` is not in the cache, or
    /// - the cached content equals `new_content` (no changes).
    ///
    /// The diff uses a 3-line context radius and includes `a/<path>` /
    /// `b/<path>` headers that mirror `git diff` output.
    ///
    /// When `working_dir` is provided, paths are shown relative to it.
    pub fn diff(
        &self,
        path: &Path,
        new_content: &str,
        working_dir: Option<&PathBuf>,
    ) -> Option<String> {
        let old_content = self.get(path)?;

        if old_content == new_content {
            return None;
        }

        let path_str = to_relative_path(path, working_dir);
        let header_old = format!("a/{path_str}");
        let header_new = format!("b/{path_str}");

        let diff = TextDiff::from_lines(old_content.as_str(), new_content);
        let output = diff
            .unified_diff()
            .header(&header_old, &header_new)
            .context_radius(3)
            .to_string();

        Some(output)
    }

    /// Compute diff and return as a `Vec<FileDiff>` (empty if no diff).
    pub fn diff_vec(
        &self,
        path: &Path,
        new_content: &str,
        working_dir: Option<&PathBuf>,
    ) -> Vec<FileDiff> {
        self.diff(path, new_content, working_dir)
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

        assert!(cache.diff(path, "new content\n", None).is_none());
    }

    #[test]
    fn diff_returns_none_when_content_unchanged() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/same.txt");
        let content = "no changes here\n";

        cache.store(path, content.to_string());

        assert!(cache.diff(path, content, None).is_none());
    }

    #[test]
    fn diff_produces_unified_diff_on_change() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/changed.txt");

        cache.store(path, "line one\nline two\nline three\n".to_string());

        let result = cache
            .diff(path, "line one\nline TWO\nline three\n", None)
            .expect("expected a diff");

        assert!(result.contains("--- a//tmp/changed.txt"));
        assert!(result.contains("+++ b//tmp/changed.txt"));
        assert!(result.contains("-line two"));
        assert!(result.contains("+line TWO"));
        // unchanged lines appear as context
        assert!(result.contains(" line one"));
        assert!(result.contains(" line three"));
    }

    #[test]
    fn diff_includes_context_radius() {
        let cache = FileCache::new();
        let path = Path::new("/tmp/ctx.txt");

        // 10 lines; change line 5 (index 4)
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

        let diff = cache.diff(path, &new, None).unwrap();

        // Lines 2-4 and 6-8 should appear as context (within 3 of line 5)
        assert!(diff.contains(" line 2"));
        assert!(diff.contains(" line 4"));
        assert!(diff.contains("-line 5"));
        assert!(diff.contains("+line FIVE"));
        assert!(diff.contains(" line 6"));
        assert!(diff.contains(" line 8"));
        // Line 1 and 9+ may be outside the 3-line radius — that's fine
    }

    #[test]
    fn diff_uses_relative_path_with_working_dir() {
        let cache = FileCache::new();
        let path = Path::new("/home/user/project/src/lib.rs");
        let working_dir = PathBuf::from("/home/user/project");

        cache.store(path, "fn main() {}\n".to_string());

        let diff = cache
            .diff(path, "fn hello() {}\n", Some(&working_dir))
            .expect("expected a diff");

        assert!(diff.contains("--- a/src/lib.rs"));
        assert!(diff.contains("+++ b/src/lib.rs"));
        assert!(!diff.contains("/home/user/project"));
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
