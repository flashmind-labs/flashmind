//! In-memory cache for web search and crawl results with line-based pagination.
//!
//! Enables agents to store large search/crawl results and paginate through them
//! without loading full content into LLM context. Each agent owns its own cache
//! via [`SearchCacheRef`].

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

pub type SearchCacheRef = Arc<RwLock<SearchResultCache>>;

/// Maximum number of cached entries before LRU eviction.
const MAX_ENTRIES: usize = 100;

/// Whether a cached entry came from a search or a crawl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKind {
    Search,
    Crawl,
}

/// A single result item within a cached entry.
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub title: String,
    pub url: String,
    pub description: Option<String>,
    pub markdown: Option<String>,
}

/// A cached search or crawl entry containing multiple results.
#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub query_or_url: String,
    pub results: Vec<CachedResult>,
    pub kind: CacheKind,
}

/// In-memory LRU cache for search and crawl results.
#[derive(Debug)]
pub struct SearchResultCache {
    entries: HashMap<String, CachedEntry>,
    /// Insertion-order tracking for LRU eviction (oldest first).
    order: Vec<String>,
    /// Monotonic counter for generating unique IDs.
    counter: u64,
}

impl Default for SearchResultCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchResultCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            counter: 0,
        }
    }

    /// Store search results and return a cache ID like `s_000001`.
    pub fn store_search(&mut self, query: &str, results: Vec<CachedResult>) -> String {
        self.counter += 1;
        let id = format!("s_{:06}", self.counter);

        let entry = CachedEntry {
            query_or_url: query.to_string(),
            results,
            kind: CacheKind::Search,
        };

        self.insert(id.clone(), entry);
        id
    }

    /// Store crawl results and return a cache ID like `c_000001`.
    pub fn store_crawl(&mut self, url: &str, pages: Vec<CachedResult>) -> String {
        self.counter += 1;
        let id = format!("c_{:06}", self.counter);

        let entry = CachedEntry {
            query_or_url: url.to_string(),
            results: pages,
            kind: CacheKind::Crawl,
        };

        self.insert(id.clone(), entry);
        id
    }

    /// Retrieve a cached entry by ID.
    pub fn get(&self, id: &str) -> Option<&CachedEntry> {
        self.entries.get(id)
    }

    /// Retrieve a specific result by index with line-based pagination.
    ///
    /// Returns `(result_ref, paginated_content)` where content is the markdown
    /// sliced by line offset/limit, with a pagination hint appended if more
    /// lines are available.
    pub fn get_result(
        &self,
        id: &str,
        index: usize,
        offset: usize,
        limit: usize,
    ) -> Option<(&CachedResult, String)> {
        let entry = self.entries.get(id)?;
        let result = entry.results.get(index)?;

        let markdown = result.markdown.as_deref().unwrap_or("");
        let lines: Vec<&str> = markdown.split('\n').collect();
        let total = lines.len();

        let start = offset.min(total);
        let end = (start + limit).min(total);
        let mut content = lines[start..end].join("\n");

        if end < total {
            content.push_str(&format!(
                "\n\n--- Showing lines {}-{} of {} total. Use offset={} to see more. ---",
                start + 1,
                end,
                total,
                end,
            ));
        }

        Some((result, content))
    }

    /// Insert an entry, evicting the oldest if at capacity.
    fn insert(&mut self, id: String, entry: CachedEntry) {
        if self.entries.len() >= MAX_ENTRIES
            && let Some(oldest_id) = self.order.first().cloned()
        {
            self.entries.remove(&oldest_id);
            self.order.remove(0);
        }

        self.entries.insert(id.clone(), entry);
        self.order.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results(count: usize) -> Vec<CachedResult> {
        (0..count)
            .map(|i| CachedResult {
                title: format!("Result {i}"),
                url: format!("https://example.com/{i}"),
                description: Some(format!("Description {i}")),
                markdown: Some(format!("Line 1 of {i}\nLine 2 of {i}\nLine 3 of {i}")),
            })
            .collect()
    }

    #[test]
    fn store_and_retrieve_search() {
        let mut cache = SearchResultCache::new();
        let id = cache.store_search("rust async", sample_results(3));

        assert_eq!(id, "s_000001");

        let entry = cache.get(&id).unwrap();
        assert_eq!(entry.query_or_url, "rust async");
        assert_eq!(entry.results.len(), 3);
        assert_eq!(entry.kind, CacheKind::Search);
    }

    #[test]
    fn store_and_retrieve_crawl() {
        let mut cache = SearchResultCache::new();
        let id = cache.store_crawl("https://example.com", sample_results(2));

        assert_eq!(id, "c_000001");

        let entry = cache.get(&id).unwrap();
        assert_eq!(entry.query_or_url, "https://example.com");
        assert_eq!(entry.results.len(), 2);
        assert_eq!(entry.kind, CacheKind::Crawl);
    }

    #[test]
    fn get_result_by_index() {
        let mut cache = SearchResultCache::new();
        let id = cache.store_search("test", sample_results(3));

        let (result, content) = cache.get_result(&id, 1, 0, 100).unwrap();
        assert_eq!(result.title, "Result 1");
        assert_eq!(content, "Line 1 of 1\nLine 2 of 1\nLine 3 of 1");
    }

    #[test]
    fn get_result_invalid_index() {
        let mut cache = SearchResultCache::new();
        let id = cache.store_search("test", sample_results(1));

        assert!(cache.get_result(&id, 5, 0, 100).is_none());
    }

    #[test]
    fn line_based_pagination_with_offset_and_limit() {
        let mut cache = SearchResultCache::new();

        let results = vec![CachedResult {
            title: "Big page".into(),
            url: "https://example.com".into(),
            description: None,
            markdown: Some(
                "line0\nline1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9".into(),
            ),
        }];

        let id = cache.store_search("paged", results);

        // First page: lines 0..3
        let (_, content) = cache.get_result(&id, 0, 0, 3).unwrap();
        assert_eq!(
            content,
            "line0\nline1\nline2\n\n--- Showing lines 1-3 of 10 total. Use offset=3 to see more. ---"
        );

        // Second page: lines 3..6
        let (_, content) = cache.get_result(&id, 0, 3, 3).unwrap();
        assert_eq!(
            content,
            "line3\nline4\nline5\n\n--- Showing lines 4-6 of 10 total. Use offset=6 to see more. ---"
        );

        // Last page: lines 8..10 (no hint)
        let (_, content) = cache.get_result(&id, 0, 8, 3).unwrap();
        assert_eq!(content, "line8\nline9");
    }

    #[test]
    fn pagination_offset_beyond_content() {
        let mut cache = SearchResultCache::new();

        let results = vec![CachedResult {
            title: "Short".into(),
            url: "https://example.com".into(),
            description: None,
            markdown: Some("only\ntwo".into()),
        }];

        let id = cache.store_search("short", results);
        let (_, content) = cache.get_result(&id, 0, 10, 5).unwrap();
        assert_eq!(content, "");
    }

    #[test]
    fn eviction_at_max_entries() {
        let mut cache = SearchResultCache::new();

        // Fill to capacity
        for i in 0..MAX_ENTRIES {
            cache.store_search(&format!("query {i}"), sample_results(1));
        }

        assert_eq!(cache.entries.len(), MAX_ENTRIES);

        // The first entry should still exist
        assert!(cache.get("s_000001").is_some());

        // Adding one more should evict the oldest
        cache.store_search("overflow", sample_results(1));
        assert_eq!(cache.entries.len(), MAX_ENTRIES);
        assert!(cache.get("s_000001").is_none());
        assert!(cache.get("s_000002").is_some());
        assert!(cache.get("s_000101").is_some());
    }

    #[test]
    fn empty_markdown_pagination() {
        let mut cache = SearchResultCache::new();

        let results = vec![CachedResult {
            title: "No content".into(),
            url: "https://example.com".into(),
            description: None,
            markdown: None,
        }];

        let id = cache.store_search("empty", results);
        let (_, content) = cache.get_result(&id, 0, 0, 10).unwrap();
        assert_eq!(content, "");
    }

    #[test]
    fn counter_increments_across_kinds() {
        let mut cache = SearchResultCache::new();

        let s1 = cache.store_search("q1", sample_results(1));
        let c1 = cache.store_crawl("https://a.com", sample_results(1));
        let s2 = cache.store_search("q2", sample_results(1));

        assert_eq!(s1, "s_000001");
        assert_eq!(c1, "c_000002");
        assert_eq!(s2, "s_000003");
    }
}
