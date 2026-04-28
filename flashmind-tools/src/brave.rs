//! Brave Search API tools for web, news, image, and video search.
//!
//! <https://api.search.brave.com/app#/documentation>

use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use crate::utils::{http_client, send_with_retry, truncate_utf8};
use anyhow::Result;
use flashmind_types::event::Source;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

const BASE_URL: &str = "https://api.search.brave.com/";

// ============================================================================
// Brave Search API Response Types
// ============================================================================

/// Top-level response for web search (results nested under `web.results`).
#[derive(Deserialize)]
struct WebSearchResponse {
    web: Option<WebResults>,
}

#[derive(Deserialize)]
struct WebResults {
    results: Vec<WebResult>,
}

#[derive(Deserialize)]
struct WebResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
}

/// Generic response for news/images/videos (results at top level).
#[derive(Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct GenericSearchResponse<T> {
    #[serde(default)]
    results: Vec<T>,
}

#[derive(Deserialize)]
struct NewsResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    age: String,
    #[serde(default)]
    meta_url: MetaUrl,
}

#[derive(Deserialize, Default)]
struct MetaUrl {
    #[serde(default)]
    hostname: String,
}

#[derive(Deserialize)]
struct ImageResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    properties: ImageProperties,
}

#[derive(Deserialize, Default)]
struct ImageProperties {
    #[serde(default)]
    width: u64,
    #[serde(default)]
    height: u64,
}

#[derive(Deserialize)]
struct VideoResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    meta_url: MetaUrl,
    #[serde(default)]
    video: VideoMeta,
}

#[derive(Deserialize, Default)]
struct VideoMeta {
    #[serde(default)]
    duration: String,
    views: Option<u64>,
}

/// Brave Search tool supporting web, news, image, and video search.
pub struct BraveSearchTool {
    client: reqwest::Client,
    api_key: String,
}

impl BraveSearchTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn search(
        &self,
        search_type: &str,
        query: &str,
        count: u64,
        offset: u64,
        freshness: Option<&str>,
        country: Option<&str>,
        lang: Option<&str>,
    ) -> Result<(String, Vec<Source>)> {
        let path = match search_type {
            "web" => "res/v1/web/search",
            "news" => "res/v1/news/search",
            "images" => "res/v1/images/search",
            "videos" => "res/v1/videos/search",
            other => anyhow::bail!("{}: Unknown type: {}", "brave_search", other),
        };
        let mut endpoint = Url::parse(BASE_URL).unwrap();
        endpoint.set_path(path);

        // Build query params - collect only non-None values
        let count_str = count.to_string();
        let offset_str = offset.to_string();
        let mut params: Vec<(&str, &str)> = vec![("q", query), ("count", &count_str)];

        if let Some(f) = freshness {
            params.push(("freshness", f));
        }
        if let Some(c) = country {
            params.push(("country", c));
        }
        if let Some(l) = lang {
            params.push(("search_lang", l));
        }
        if offset > 0 {
            params.push(("offset", &offset_str));
        }

        let response = send_with_retry(|| {
            self.client
                .get(endpoint.clone())
                .header("Accept", "application/json")
                .header("Accept-Encoding", "gzip")
                .header("X-Subscription-Token", &self.api_key)
                .query(&params)
        })
        .await?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            anyhow::bail!("brave_search: HTTP {}: {}", status.as_u16(), body);
        }

        format_results(search_type, &body)
    }
}

#[derive(Deserialize)]
struct BraveSearchArgs {
    query: String,
    #[serde(rename = "type")]
    search_type: Option<String>,
    count: Option<u64>,
    offset: Option<u64>,
    freshness: Option<String>,
    country: Option<String>,
    lang: Option<String>,
}

/// Deserialize and format search results by type, returning formatted text and source URLs.
fn format_results(search_type: &str, body: &str) -> Result<(String, Vec<Source>)> {
    let parse_err = |e: serde_json::Error| {
        anyhow::anyhow!("brave_search: JSON error: {}. Preview: {:.200}", e, body)
    };

    match search_type {
        "web" => {
            let resp: WebSearchResponse = serde_json::from_str(body).map_err(parse_err)?;
            let results = resp.web.map(|w| w.results).unwrap_or_default();
            if results.is_empty() {
                return Ok(("No web results found.".into(), Vec::new()));
            }
            let sources = results
                .iter()
                .map(|r| Source {
                    url: r.url.clone(),
                    title: Some(r.title.clone()),
                })
                .collect();
            Ok((format_items(&results, format_web_item), sources))
        }
        "news" => {
            let resp: GenericSearchResponse<NewsResult> =
                serde_json::from_str(body).map_err(parse_err)?;
            if resp.results.is_empty() {
                return Ok(("No news results found.".into(), Vec::new()));
            }
            let sources = resp
                .results
                .iter()
                .map(|r| Source {
                    url: r.url.clone(),
                    title: Some(r.title.clone()),
                })
                .collect();
            Ok((format_items(&resp.results, format_news_item), sources))
        }
        "images" => {
            let resp: GenericSearchResponse<ImageResult> =
                serde_json::from_str(body).map_err(parse_err)?;
            if resp.results.is_empty() {
                return Ok(("No image results found.".into(), Vec::new()));
            }
            Ok((format_items(&resp.results, format_image_item), Vec::new()))
        }
        "videos" => {
            let resp: GenericSearchResponse<VideoResult> =
                serde_json::from_str(body).map_err(parse_err)?;
            if resp.results.is_empty() {
                return Ok(("No video results found.".into(), Vec::new()));
            }
            let sources = resp
                .results
                .iter()
                .map(|r| Source {
                    url: r.url.clone(),
                    title: Some(r.title.clone()),
                })
                .collect();
            Ok((format_items(&resp.results, format_video_item), sources))
        }
        _ => Ok(("No results.".into(), Vec::new())),
    }
}

/// Format a slice of results using a per-item formatter.
fn format_items<T>(items: &[T], formatter: fn(usize, &T) -> String) -> String {
    let mut output = String::with_capacity(items.len() * 100);
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            output.push_str("\n\n");
        }
        output.push_str(&formatter(i + 1, item));
    }
    output
}

fn format_web_item(i: usize, r: &WebResult) -> String {
    format!("{}. {}\n   {}\n   {}", i, r.title, r.url, r.description)
}

fn format_news_item(i: usize, r: &NewsResult) -> String {
    format!(
        "{}. {} [{}]\n   {} | {}\n   {}",
        i, r.title, r.age, r.meta_url.hostname, r.url, r.description
    )
}

fn format_image_item(i: usize, r: &ImageResult) -> String {
    format!(
        "{}. {} ({}x{})\n   {}\n   Source: {}",
        i, r.title, r.properties.width, r.properties.height, r.url, r.source
    )
}

fn format_video_item(i: usize, r: &VideoResult) -> String {
    let views = r
        .video
        .views
        .map(|v| format!("{} views", v))
        .unwrap_or_default();
    format!(
        "{}. {} [{}]\n   {}\n   {} | {}",
        i, r.title, r.video.duration, r.url, r.meta_url.hostname, views
    )
}

#[async_trait]
impl Tool for BraveSearchTool {
    fn name(&self) -> &str {
        "brave_search"
    }

    fn description(&self) -> &str {
        "Search using Brave Search API. Supports web, news, images, and videos."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" },
                "type": { "type": "string", "enum": ["web", "news", "images", "videos"], "description": "Search type (default: web)" },
                "count": { "type": "integer", "description": "Results per page (default: 5, max: 20/50)" },
                "offset": { "type": "integer", "description": "Pagination offset (0-9)" },
                "freshness": { "type": "string", "description": "Time filter: pd/pw/pm/py or YYYY-MM-DDtoYYYY-MM-DD" },
                "country": { "type": "string", "description": "2-letter country code or ALL" },
                "lang": { "type": "string", "description": "Content language code (en, es, fr)" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();
        let args: BraveSearchArgs = ctx.parse_args(self.name())?;

        let search_type = args.search_type.as_deref().unwrap_or("web");
        let max_count = if matches!(search_type, "images" | "videos") {
            50
        } else {
            20
        };
        let count = args.count.unwrap_or(5).min(max_count);
        let offset = args.offset.unwrap_or(0).min(9);

        match self
            .search(
                search_type,
                &args.query,
                count,
                offset,
                args.freshness.as_deref(),
                args.country.as_deref(),
                args.lang.as_deref(),
            )
            .await
        {
            Ok((output, sources)) => {
                metrics::counter!("tools.searches").increment(1);
                metrics::histogram!("tools.search.duration_seconds").record(start.elapsed().as_secs_f64());
                Ok(ToolResult::success(ctx.tool_call_id, output).with_sources(sources))
            }
            Err(e) => {
                metrics::counter!("tools.searches").increment(1);
                metrics::histogram!("tools.search.duration_seconds").record(start.elapsed().as_secs_f64());
                Ok(ToolResult::failure(ctx.tool_call_id, e.to_string()))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        format!("Searching '{}' on Brave", truncate_utf8(query, 64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name() {
        assert_eq!(BraveSearchTool::new("k".into()).name(), "brave_search");
    }

    #[test]
    fn test_format_empty() {
        let body = r#"{"web": {"results": []}}"#;
        let (text, sources) = format_results("web", body).unwrap();
        assert_eq!(text, "No web results found.");
        assert!(sources.is_empty());
    }

    #[test]
    fn test_format_web() {
        let body = r#"{"web": {"results": [{"title": "Test", "url": "https://x.com", "description": "Desc"}]}}"#;
        let (text, sources) = format_results("web", body).unwrap();
        assert!(text.contains("Test") && text.contains("https://x.com"));
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].url, "https://x.com");
        assert_eq!(sources[0].title.as_deref(), Some("Test"));
    }
}
