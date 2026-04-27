//! Tool for reading cached web search/crawl results with line-based pagination.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::search_cache::SearchCacheRef;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

/// Reads cached web search or crawl results by ID with line-based offset/limit pagination.
pub struct WebSearchReadTool {
    cache: SearchCacheRef,
}

impl WebSearchReadTool {
    pub fn new(cache: SearchCacheRef) -> Self {
        Self { cache }
    }
}

#[derive(Deserialize)]
struct SearchReadArgs {
    id: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for WebSearchReadTool {
    fn name(&self) -> &str {
        "web_search_read"
    }

    fn description(&self) -> &str {
        "Read the full content of a web search or crawl result by ID. Use after web_search/web_crawl — their results include IDs for items with truncated bodies. Use offset/limit for pagination. IDs expire when newer searches replace the cache."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Result set ID from web_search/web_crawl (e.g. \"s_000001\")"
                },
                "index": {
                    "type": "integer",
                    "description": "Result index within the set (default: 0)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset for pagination (default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (default: 200)"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchReadArgs = ctx.parse_args(self.name())?;

        let index = args.index.unwrap_or(0);
        let offset = args.offset.unwrap_or(0);
        let limit = args.limit.unwrap_or(200);

        tracing::debug!(
            id = %args.id,
            index,
            offset,
            limit,
            "web_search_read: reading cached result"
        );

        let cache = self.cache.read().await;

        // Check if the ID exists at all
        let entry = match cache.get(&args.id) {
            Some(e) => e,
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!(
                        "No cached results for ID '{}'. IDs expire after newer searches.",
                        args.id
                    ),
                ));
            }
        };

        let count = entry.results.len();

        // Try to get the specific result with pagination
        match cache.get_result(&args.id, index, offset, limit) {
            Some((result, content)) => {
                let output = format!("## {} ({})\n\n{}", result.title, result.url, content);

                Ok(ToolResult::success(ctx.tool_call_id, output))
            }
            None => {
                // ID exists but index is out of range
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!(
                        "Index {} out of range. This result set has {} entries (0-{}).",
                        index,
                        count,
                        count - 1
                    ),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
        format!("Reading {}", url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_cache::{CachedResult, SearchResultCache};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_cache() -> SearchCacheRef {
        let mut cache = SearchResultCache::new();
        cache.store_search(
            "rust async",
            vec![
                CachedResult {
                    title: "Async Rust".into(),
                    url: "https://example.com/async".into(),
                    description: Some("Async programming in Rust".into()),
                    markdown: Some("line0\nline1\nline2\nline3\nline4".into()),
                },
                CachedResult {
                    title: "Tokio Guide".into(),
                    url: "https://tokio.rs/guide".into(),
                    description: None,
                    markdown: Some("tokio line 0\ntokio line 1".into()),
                },
            ],
        );
        Arc::new(RwLock::new(cache))
    }

    #[test]
    fn tool_metadata() {
        let cache = Arc::new(RwLock::new(SearchResultCache::new()));
        let tool = WebSearchReadTool::new(cache);

        assert_eq!(tool.name(), "web_search_read");
        assert!(!tool.description().is_empty());

        let params = tool.parameters();
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "id");
    }

    #[tokio::test]
    async fn read_missing_id() {
        let cache = Arc::new(RwLock::new(SearchResultCache::new()));
        let tool = WebSearchReadTool::new(cache);

        let result = crate::tests::execute_tool(&tool, "call_1", json!({"id": "s_999999"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("No cached results for ID"));
    }

    #[tokio::test]
    async fn read_valid_result() {
        let cache = make_cache();
        let tool = WebSearchReadTool::new(cache);

        let result =
            crate::tests::execute_tool(&tool, "call_2", json!({"id": "s_000001", "index": 0}))
                .await
                .unwrap();

        assert!(result.success);
        assert!(
            result
                .output
                .contains("## Async Rust (https://example.com/async)")
        );
        assert!(result.output.contains("line0"));
    }

    #[tokio::test]
    async fn read_index_out_of_range() {
        let cache = make_cache();
        let tool = WebSearchReadTool::new(cache);

        let result =
            crate::tests::execute_tool(&tool, "call_3", json!({"id": "s_000001", "index": 5}))
                .await
                .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Index 5 out of range"));
        assert!(result.output.contains("2 entries (0-1)"));
    }

    #[tokio::test]
    async fn read_with_pagination() {
        let cache = make_cache();
        let tool = WebSearchReadTool::new(cache);

        let result = crate::tests::execute_tool(
            &tool,
            "call_4",
            json!({"id": "s_000001", "index": 0, "offset": 2, "limit": 2}),
        )
        .await
        .unwrap();

        assert!(result.success);
        assert!(result.output.contains("line2"));
        assert!(result.output.contains("line3"));
        // Should have pagination hint since there's a line4 remaining
        assert!(result.output.contains("offset=4"));
    }
}
