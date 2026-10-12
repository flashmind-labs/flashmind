//! Firecrawl API tools for web search, scraping, crawling, and mapping.
//!
//! <https://docs.firecrawl.dev/api-reference/v2-introduction>

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;
use tracing::{Instrument, info_span};

use crate::search_cache::{CachedResult, SearchCacheRef};
use crate::utils::http_client;
use flashmind_types::event::Source;
use flashmind_types::tool::Tool;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::ToolResult;

const FIRECRAWL_BASE_V2: &str = "https://api.firecrawl.dev/v2";
const CRAWL_POLL_INTERVAL: Duration = Duration::from_secs(3);
const CRAWL_TIMEOUT: Duration = Duration::from_secs(120);
/// Most characters of a scraped page's markdown or HTML handed to the model.
const MAX_SCRAPE_CHARS: usize = 30_000;

/// What the model sees when Firecrawl refuses a request. Only the short
/// `error` field of a client error is kept, never the raw body, which can
/// echo the request.
fn api_error(status: reqwest::StatusCode, body: &str) -> String {
    let code = status.as_u16();
    let reason = match code {
        401 => "credentials were rejected".to_owned(),
        402 => "credits are exhausted".to_owned(),
        429 => "rate limited, try again shortly".to_owned(),
        400..=499 => serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|v| {
                v.get("error")?
                    .as_str()
                    .map(|e| e.chars().take(300).collect())
            })
            .unwrap_or_else(|| "the request was rejected".to_owned()),
        _ => "temporarily unavailable".to_owned(),
    };
    format!("Firecrawl HTTP {code}: {reason}")
}

/// Cuts `text` to [`MAX_SCRAPE_CHARS`], saying how long it was.
fn cap(text: &str) -> String {
    let total = text.chars().count();
    if total <= MAX_SCRAPE_CHARS {
        return text.to_owned();
    }
    let kept: String = text.chars().take(MAX_SCRAPE_CHARS).collect();
    format!("{kept}\n\n[truncated: {total} characters in total]")
}

/// Names one Firecrawl call: `error` prefixes errors returned to the caller,
/// `log` prefixes log lines.
#[derive(Clone, Copy)]
struct Tags {
    error: &'static str,
    log: &'static str,
}

/// How a polled Firecrawl job ended.
enum PollOutcome<T> {
    Completed(T),
    Failed,
    TimedOut,
}

/// Authenticated client for the Firecrawl v2 API, shared by every tool here.
struct FirecrawlApi {
    client: reqwest::Client,
    api_key: String,
}

impl FirecrawlApi {
    fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }

    /// POST `body` to `path` and return the response text. A refusal from
    /// Firecrawl comes back as `Ok(Err(..))` holding the [`api_error`] text.
    async fn post(
        &self,
        path: &str,
        body: &Value,
        tags: Tags,
        detail: String,
    ) -> anyhow::Result<Result<String, String>> {
        let response = self
            .client
            .post(format!("{FIRECRAWL_BASE_V2}{path}"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(body)
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = detail.as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("{}: {e}", tags.error))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("{}: rate limited (429)", tags.log);
            } else {
                tracing::error!(status = status.as_u16(), "{}: API error", tags.log);
            }
            return Ok(Err(api_error(status, &body_text)));
        }

        Ok(Ok(body_text))
    }

    /// Poll the job `id` under `path` every [`CRAWL_POLL_INTERVAL`] until it
    /// completes, fails, or [`CRAWL_TIMEOUT`] passes. Failed or unparsable
    /// polls are retried.
    async fn poll<T: DeserializeOwned>(
        &self,
        path: &str,
        id: &str,
        tags: Tags,
        status_of: fn(&T) -> &str,
    ) -> anyhow::Result<PollOutcome<T>> {
        let poll_url = format!("{FIRECRAWL_BASE_V2}{path}/{id}");
        let deadline = tokio::time::Instant::now() + CRAWL_TIMEOUT;
        let mut poll_count = 0u32;

        loop {
            tokio::time::sleep(CRAWL_POLL_INTERVAL).await;

            if tokio::time::Instant::now() > deadline {
                tracing::warn!(job_id = %id, polls = poll_count, "{}: timed out after 120s", tags.log);
                return Ok(PollOutcome::TimedOut);
            }

            poll_count += 1;
            tracing::debug!(job_id = %id, poll = poll_count, "{}: polling status", tags.log);

            let resp = self
                .client
                .get(&poll_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .instrument(info_span!(target: "prompt_trace", "step",
                    step = "api_call",
                    detail = format!("GET {path}/{id} poll={poll_count}").as_str(),
                ))
                .await
                .map_err(|e| anyhow::anyhow!("{}: {e}", tags.error))?;

            if !resp.status().is_success() {
                tracing::warn!(
                    status = resp.status().as_u16(),
                    "{}: poll request failed",
                    tags.log
                );
                continue;
            }

            let poll_body = resp.text().await.unwrap_or_default();
            let status_resp: T = match serde_json::from_str(&poll_body) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(err = %e, "{}: failed to parse poll response", tags.log);
                    continue;
                }
            };

            let status = status_of(&status_resp);
            tracing::debug!(status = %status, "{}: poll result", tags.log);

            match status {
                "completed" => return Ok(PollOutcome::Completed(status_resp)),
                "failed" => return Ok(PollOutcome::Failed),
                _ => continue, // "scraping", "processing", "queued", etc.
            }
        }
    }
}

// ============================================================================
// Web Search (Firecrawl v2)
// ============================================================================

/// Web search tool using Firecrawl v2 API.
pub struct WebSearchTool {
    api: FirecrawlApi,
    cache: SearchCacheRef,
}

impl WebSearchTool {
    pub fn new(api_key: String, cache: SearchCacheRef) -> Self {
        Self {
            api: FirecrawlApi::new(api_key),
            cache,
        }
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    limit: Option<u64>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    country: Option<String>,
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    data: SearchData,
    #[serde(default)]
    credits_used: u64,
}

#[derive(Deserialize, Default)]
struct SearchData {
    #[serde(default)]
    web: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    category: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "firecrawl_search"
    }

    fn description(&self) -> &str {
        "Search the web using Firecrawl. Returns results with full page content."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" },
                "limit": { "type": "integer", "description": "Max results (default: 5, max: 20)" },
                "category": { "type": "string", "enum": ["github", "research", "pdf"], "description": "Filter by category" },
                "location": { "type": "string", "description": "Location for geo-targeted results (e.g., 'San Francisco,California,United States')" },
                "country": { "type": "string", "description": "ISO country code (e.g., 'US', 'DE', 'JP')" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(5).min(20);

        tracing::debug!(query = %args.query, limit, "web_search: executing search");

        // Build request body with optional parameters
        let mut body = json!({
            "query": args.query,
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"]
            }
        });

        if let Some(ref cat) = args.category {
            body["categories"] = json!([{"type": cat}]);
        }
        if let Some(ref loc) = args.location {
            body["location"] = json!(loc);
        }
        if let Some(ref country) = args.country {
            body["country"] = json!(country);
        }

        let tags = Tags {
            error: "firecrawl_search",
            log: "web_search",
        };
        let detail = format!("POST /search query={} limit={}", args.query, limit);
        let body_text = match self.api.post("/search", &body, tags, detail).await? {
            Ok(text) => text,
            Err(msg) => return Ok(ToolResult::failure(ctx.tool_call_id, msg)),
        };

        let resp: SearchResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_search: failed to parse response");
            anyhow::anyhow!("firecrawl_search: {e}")
        })?;

        tracing::debug!(
            success = resp.success,
            results = resp.data.web.len(),
            credits = resp.credits_used,
            "web_search: parsed response"
        );

        if !resp.success || resp.data.web.is_empty() {
            tracing::debug!("web_search: no results found");
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No search results found.",
            ));
        }

        let sources: Vec<Source> = resp
            .data
            .web
            .iter()
            .filter(|r| !r.url.is_empty())
            .map(|r| Source {
                url: r.url.clone(),
                title: Some(r.title.clone()),
            })
            .collect();

        // Cache full results for later retrieval via web_search_read
        let cached: Vec<CachedResult> = resp
            .data
            .web
            .iter()
            .map(|r| CachedResult {
                title: r.title.clone(),
                url: r.url.clone(),
                description: r.description.clone(),
                markdown: r.markdown.clone(),
            })
            .collect();

        let cache_id = self.cache.write().await.store_search(&args.query, cached);

        // Build output with line-based truncation for large markdown bodies
        const MAX_BODY_LINES: usize = 100;

        let mut output = format!(
            "Search results (id: {}, use web_search_read to read full content):\n\n",
            cache_id
        );

        for (i, r) in resp.data.web.iter().enumerate() {
            if i > 0 {
                output.push_str("\n\n");
            }
            output.push_str(&format!("{}. {}\n   {}", i + 1, r.title, r.url));
            if let Some(ref desc) = r.description {
                output.push_str(&format!("\n   {}", desc));
            }
            if let Some(ref cat) = r.category {
                output.push_str(&format!(" [{}]", cat));
            }
            if let Some(ref md) = r.markdown {
                let lines: Vec<&str> = md.lines().collect();
                if lines.len() > MAX_BODY_LINES {
                    let truncated: String = lines[..MAX_BODY_LINES].join("\n");
                    output.push_str(&format!(
                        "\n   ---\n   {}\n   [truncated — {} lines total. Use web_search_read(id=\"{}\", index={}) for full content]",
                        truncated, lines.len(), cache_id, i
                    ));
                } else {
                    output.push_str(&format!("\n   ---\n   {}", md));
                }
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, output).with_sources(sources))
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Searching the web for '{}'", query)
    }
}

// ============================================================================
// Web Scrape (Firecrawl v2)
// ============================================================================

/// Web scrape tool for extracting content from a single URL.
pub struct WebScrapeTool {
    api: FirecrawlApi,
}

impl WebScrapeTool {
    pub fn new(api_key: String) -> Self {
        Self {
            api: FirecrawlApi::new(api_key),
        }
    }
}

#[derive(Deserialize)]
struct ScrapeArgs {
    url: String,
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    formats: Option<Vec<String>>,
    #[serde(default)]
    actions: Option<Vec<Value>>,
    #[serde(default = "default_timeout")]
    timeout: u64,
}

fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a string or array of strings")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_owned()]))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(if v.is_empty() { None } else { Some(v) })
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

fn default_timeout() -> u64 {
    30000
}

#[derive(Deserialize, Serialize)]
struct ScrapeResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    data: ScrapeData,
    #[serde(default)]
    warning: Option<String>,
}

#[derive(Deserialize, Serialize, Default)]
struct ScrapeData {
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    html: Option<String>,
    #[serde(default)]
    links: Option<Vec<String>>,
    #[serde(default)]
    screenshot: Option<String>,
    #[serde(default)]
    metadata: ScrapeMetadata,
}

#[derive(Deserialize, Serialize, Default)]
struct ScrapeMetadata {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "sourceURL", default)]
    source_url: Option<String>,
    #[serde(default)]
    keywords: Option<String>,
}

#[async_trait]
impl Tool for WebScrapeTool {
    fn name(&self) -> &str {
        "web_scrape"
    }

    fn description(&self) -> &str {
        "Scrape a single URL and extract content in various formats (markdown, html, links, screenshot)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to scrape" },
                "formats": {
                    "type": "array",
                    "items": { "type": "string", "enum": ["markdown", "html", "links", "screenshot"] },
                    "description": "Output formats to include (default: ['markdown'])"
                },
                "actions": {
                    "type": "array",
                    "items": { "type": "object" },
                    "description": "Browser actions to perform before scraping (e.g., wait, click, scroll)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000, max: 300000)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ScrapeArgs = ctx.parse_args(self.name())?;
        let timeout = args.timeout.min(300000);

        tracing::debug!(url = %args.url, timeout, "web_scrape: starting scrape");

        // Build formats array
        let formats = args
            .formats
            .as_ref()
            .map(|f| {
                f.iter()
                    .map(|fmt| match fmt.as_str() {
                        "markdown" => json!({"type": "markdown"}),
                        "html" => json!({"type": "html"}),
                        "rawHtml" => json!({"type": "rawHtml"}),
                        "links" => json!({"type": "links"}),
                        "screenshot" => json!({"type": "screenshot"}),
                        _ => json!({"type": "markdown"}),
                    })
                    .collect::<Vec<Value>>()
            })
            .unwrap_or_else(|| vec![json!({"type": "markdown"})]);

        // Build request body
        let mut body = json!({
            "url": args.url,
            "formats": formats,
            "timeout": timeout,
            "onlyMainContent": true
        });

        if let Some(ref actions) = args.actions {
            body["actions"] = json!(actions);
        }

        let tags = Tags {
            error: "web_scrape",
            log: "web_scrape",
        };
        let detail = format!("POST /scrape url={}", args.url);
        let body_text = match self.api.post("/scrape", &body, tags, detail).await? {
            Ok(text) => text,
            Err(msg) => return Ok(ToolResult::failure(ctx.tool_call_id, msg)),
        };

        let resp: ScrapeResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_scrape: failed to parse response");
            anyhow::anyhow!("web_scrape: {e}")
        })?;

        if !resp.success {
            let err_msg = resp.warning.unwrap_or_else(|| "Scrape failed".into());
            return Ok(ToolResult::failure(ctx.tool_call_id, err_msg));
        }

        tracing::debug!(url = %args.url, "web_scrape: completed successfully");

        // Build output based on requested formats
        let mut output = String::new();

        // Metadata header
        let title = resp
            .data
            .metadata
            .title
            .clone()
            .unwrap_or_else(|| "Untitled".into());
        output.push_str(&format!("# {}\n\n", title));

        if let Some(ref desc) = resp.data.metadata.description {
            output.push_str(&format!("**Description:** {}\n\n", desc));
        }

        output.push_str(&format!(
            "**URL:** {}\n\n",
            resp.data
                .metadata
                .source_url
                .unwrap_or_else(|| args.url.clone())
        ));

        // Content sections
        if let Some(ref md) = resp.data.markdown {
            output.push_str("## Content\n\n");
            output.push_str(&cap(md));
            output.push('\n');
        }

        if let Some(ref html) = resp.data.html {
            output.push_str("## HTML\n\n```html\n");
            output.push_str(&cap(html));
            output.push_str("\n```\n\n");
        }

        if let Some(ref links) = resp.data.links {
            output.push_str(&format!("## Links ({} found)\n\n", links.len()));
            for link in links.iter().take(20) {
                output.push_str(&format!("- {}\n", link));
            }
            if links.len() > 20 {
                output.push_str(&format!("\n... and {} more links\n", links.len() - 20));
            }
            output.push('\n');
        }

        if let Some(ref screenshot) = resp.data.screenshot {
            output.push_str(&format!("## Screenshot\n\n![Screenshot]({})\n", screenshot));
        }

        let sources = vec![Source {
            url: args.url,
            title: resp.data.metadata.title,
        }];

        Ok(ToolResult::success(ctx.tool_call_id, output).with_sources(sources))
    }

    fn humanize(&self, args: &Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Scraping: {}", url)
    }
}

// ============================================================================
// Web Crawl (Firecrawl v2)
// ============================================================================

/// Web crawl tool for multi-page website extraction.
pub struct WebCrawlTool {
    api: FirecrawlApi,
    cache: SearchCacheRef,
}

impl WebCrawlTool {
    pub fn new(api_key: String, cache: SearchCacheRef) -> Self {
        Self {
            api: FirecrawlApi::new(api_key),
            cache,
        }
    }
}

#[derive(Deserialize)]
struct CrawlArgs {
    url: String,
    limit: Option<u64>,
}

#[derive(Deserialize)]
struct CrawlStartResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    id: String,
}

#[derive(Deserialize)]
struct CrawlStatusResponse {
    #[serde(default)]
    status: String,
    #[serde(default)]
    data: Vec<CrawlPage>,
}

#[derive(Deserialize)]
struct CrawlPage {
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    metadata: Option<CrawlMetadata>,
}

#[derive(Deserialize)]
struct CrawlMetadata {
    #[serde(default, rename = "sourceURL")]
    source_url: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[async_trait]
impl Tool for WebCrawlTool {
    fn name(&self) -> &str {
        "web_crawl"
    }

    fn description(&self) -> &str {
        "Crawl a website and return markdown content of multiple pages."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to start crawling from" },
                "limit": { "type": "integer", "description": "Max pages to crawl (default: 10, max: 50)" }
            },
            "required": ["url"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(180) // Crawling can be slow
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CrawlArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(10).min(50);

        tracing::debug!(url = %args.url, limit, "web_crawl: starting crawl");

        // Start the crawl
        let tags = Tags {
            error: "web_crawl",
            log: "web_crawl",
        };
        let body = json!({
            "url": args.url,
            "limit": limit,
        });
        let detail = format!("POST /crawl url={} limit={}", args.url, limit);
        let body_text = match self.api.post("/crawl", &body, tags, detail).await? {
            Ok(text) => text,
            Err(msg) => return Ok(ToolResult::failure(ctx.tool_call_id, msg)),
        };

        let start: CrawlStartResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_crawl: failed to parse start response");
            anyhow::anyhow!("web_crawl: {e}")
        })?;

        if !start.success || start.id.is_empty() {
            tracing::error!("web_crawl: failed to start crawl");
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Firecrawl did not start the crawl.",
            ));
        }

        tracing::debug!(crawl_id = %start.id, "web_crawl: job submitted, polling");

        let status_resp = match self
            .api
            .poll("/crawl", &start.id, tags, |r: &CrawlStatusResponse| {
                &r.status
            })
            .await?
        {
            PollOutcome::Completed(status_resp) => status_resp,
            PollOutcome::Failed => {
                tracing::error!(crawl_id = %start.id, "web_crawl: crawl failed");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Crawl failed. Try web_scrape on the pages you need.",
                ));
            }
            PollOutcome::TimedOut => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Crawl timed out after 120 seconds. Try web_scrape on the pages you need.",
                ));
            }
        };

        tracing::debug!(pages = status_resp.data.len(), "web_crawl: completed");

        let cached: Vec<CachedResult> = status_resp
            .data
            .iter()
            .map(|p| {
                let title = p
                    .metadata
                    .as_ref()
                    .and_then(|m| m.title.clone())
                    .unwrap_or_else(|| "Untitled".into());
                let url = p
                    .metadata
                    .as_ref()
                    .and_then(|m| m.source_url.clone())
                    .unwrap_or_else(|| "unknown".into());
                CachedResult {
                    title,
                    url,
                    description: None,
                    markdown: p.markdown.clone(),
                }
            })
            .collect();

        let cache_id = self.cache.write().await.store_crawl(&args.url, cached);
        let output = format_crawl_results_truncated(&status_resp.data, &cache_id);

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Crawling: {}", url)
    }
}

fn format_crawl_results_truncated(pages: &[CrawlPage], cache_id: &str) -> String {
    const MAX_BODY_LINES: usize = 100;

    if pages.is_empty() {
        return "No pages crawled.".into();
    }

    let mut output = format!(
        "# Crawl Results ({} pages, id: {}, use web_search_read to access full content)\n\n",
        pages.len(),
        cache_id
    );

    for (i, page) in pages.iter().enumerate() {
        if i > 0 {
            output.push_str("\n\n---\n\n");
        }

        let title = page
            .metadata
            .as_ref()
            .and_then(|m| m.title.as_deref())
            .unwrap_or("Untitled");
        let url = page
            .metadata
            .as_ref()
            .and_then(|m| m.source_url.as_deref())
            .unwrap_or("unknown");

        output.push_str(&format!("## {} ({})\n\n", title, url));

        if let Some(ref md) = page.markdown {
            let lines: Vec<&str> = md.lines().collect();
            if lines.len() > MAX_BODY_LINES {
                let truncated: String = lines[..MAX_BODY_LINES].join("\n");
                output.push_str(&format!(
                    "{}\n---\n[truncated — {} lines total. Use web_search_read(id=\"{}\", index={}) for full content]",
                    truncated,
                    lines.len(),
                    cache_id,
                    i
                ));
            } else {
                output.push_str(md);
            }
        }
    }
    output
}

// ============================================================================
// Web Map (Firecrawl v2)
// ============================================================================

/// Web map tool for discovering URLs on a website.
pub struct WebMapTool {
    api: FirecrawlApi,
}

impl WebMapTool {
    pub fn new(api_key: String) -> Self {
        Self {
            api: FirecrawlApi::new(api_key),
        }
    }
}

#[derive(Deserialize)]
struct MapArgs {
    url: String,
    search: Option<String>,
    limit: Option<u64>,
}

#[derive(Deserialize)]
struct MapResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    links: Vec<MapLink>,
}

#[derive(Deserialize)]
struct MapLink {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[async_trait]
impl Tool for WebMapTool {
    fn name(&self) -> &str {
        "web_map"
    }

    fn description(&self) -> &str {
        "Discover all URLs on a website. Returns up to 100 links by default."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The base URL to map" },
                "search": { "type": "string", "description": "Optional search term to filter URLs (e.g., 'blog' for blog posts)" },
                "limit": { "type": "integer", "description": "Max URLs to return (default: 100, max: 100)" }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MapArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(100).min(100);

        tracing::debug!(url = %args.url, limit, "web_map: starting mapping");

        // Build request body
        let mut body = json!({
            "url": args.url,
            "limit": limit,
            "ignoreQueryParameters": true,
            "includeSubdomains": true
        });

        if let Some(ref search) = args.search {
            body["search"] = json!(search);
        }

        let tags = Tags {
            error: "web_map",
            log: "web_map",
        };
        let detail = format!("POST /map url={} limit={}", args.url, limit);
        let body_text = match self.api.post("/map", &body, tags, detail).await? {
            Ok(text) => text,
            Err(msg) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("{msg}. Try web_scrape with formats=[\"links\"] instead."),
                ));
            }
        };

        let resp: MapResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_map: failed to parse response");
            anyhow::anyhow!("web_map: {e}. Try web_scrape with formats=[\"links\"] instead.")
        })?;

        if !resp.success {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Map request failed. Try web_scrape with formats=[\"links\"] instead.",
            ));
        }

        let total_found = resp.links.len();
        let limit_applied = limit.min(total_found as u64);

        tracing::debug!(url = %args.url, total = total_found, limit = limit_applied, "web_map: completed");

        // Build output
        let mut output = String::new();
        output.push_str(&format!("# Website Map for {}\n\n", args.url));
        output.push_str(&format!(
            "**Total URLs found:** {} (showing first {})\n\n",
            total_found, limit_applied
        ));

        if resp.links.is_empty() {
            output.push_str("No URLs found on this website.\n");
        } else {
            output.push_str("## Discovered URLs\n\n");
            for (i, link) in resp.links.iter().enumerate() {
                output.push_str(&format!("{}. {}", i + 1, link.url));
                if let Some(ref title) = link.title {
                    output.push_str(&format!(" - **{}**", title));
                }
                if let Some(ref desc) = link.description {
                    output.push_str(&format!("\n   {}", desc));
                }
                output.push('\n');
            }

            if total_found as u64 > limit_applied {
                output.push_str(&format!(
                    "\n... and {} more URLs (limit was {})\n",
                    total_found - limit_applied as usize,
                    limit_applied
                ));
            }
        }

        let sources: Vec<Source> = resp
            .links
            .iter()
            .take(10)
            .map(|l| Source {
                url: l.url.clone(),
                title: l.title.clone(),
            })
            .collect();

        Ok(ToolResult::success(ctx.tool_call_id, output).with_sources(sources))
    }

    fn humanize(&self, args: &Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Mapping: {}", url)
    }
}

// ============================================================================
// Web Extract (Firecrawl v2)
// ============================================================================

/// Structured data extraction tool using Firecrawl v2 extract API.
pub struct WebExtractTool {
    api: FirecrawlApi,
}

impl WebExtractTool {
    pub fn new(api_key: String) -> Self {
        Self {
            api: FirecrawlApi::new(api_key),
        }
    }
}

#[derive(Deserialize)]
struct ExtractArgs {
    urls: Vec<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    enable_web_search: Option<bool>,
}

#[derive(Deserialize)]
struct ExtractStartResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    id: String,
}

#[derive(Deserialize)]
struct ExtractStatusResponse {
    #[serde(default)]
    status: String,
    #[serde(default)]
    data: Option<Value>,
}

#[async_trait]
impl Tool for WebExtractTool {
    fn name(&self) -> &str {
        "web_extract"
    }

    fn description(&self) -> &str {
        "Extract structured data from one or more URLs using a JSON schema or natural language prompt."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "URLs to extract data from (supports wildcards like example.com/*)"
                },
                "schema": {
                    "type": "object",
                    "description": "JSON Schema defining the structure of extracted data"
                },
                "prompt": {
                    "type": "string",
                    "description": "Natural language instructions for what data to extract"
                },
                "enable_web_search": {
                    "type": "boolean",
                    "description": "Allow following external links during extraction"
                }
            },
            "required": ["urls"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(180) // Extraction can be slow
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ExtractArgs = ctx.parse_args(self.name())?;

        if args.urls.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "web_extract: urls array must not be empty.",
            ));
        }

        tracing::debug!(
            urls = ?args.urls,
            has_schema = args.schema.is_some(),
            has_prompt = args.prompt.is_some(),
            "web_extract: starting extraction"
        );

        // Build request body
        let mut body = json!({ "urls": args.urls });

        if let Some(ref schema) = args.schema {
            body["schema"] = schema.clone();
        }
        if let Some(ref prompt) = args.prompt {
            body["prompt"] = json!(prompt);
        }
        if let Some(enable) = args.enable_web_search {
            body["enableWebSearch"] = json!(enable);
        }

        // Submit extraction job
        let tags = Tags {
            error: "web_extract",
            log: "web_extract",
        };
        let detail = format!("POST /extract urls={}", args.urls.join(", "));
        let body_text = match self.api.post("/extract", &body, tags, detail).await? {
            Ok(text) => text,
            Err(msg) => return Ok(ToolResult::failure(ctx.tool_call_id, msg)),
        };

        let start: ExtractStartResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_extract: failed to parse start response");
            anyhow::anyhow!("web_extract: {e}")
        })?;

        if !start.success || start.id.is_empty() {
            tracing::error!("web_extract: failed to start extraction");
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Firecrawl did not start the extraction.",
            ));
        }

        tracing::debug!(extract_id = %start.id, "web_extract: job submitted, polling");

        match self
            .api
            .poll("/extract", &start.id, tags, |r: &ExtractStatusResponse| {
                &r.status
            })
            .await?
        {
            PollOutcome::Completed(status_resp) => {
                tracing::debug!(extract_id = %start.id, "web_extract: completed");

                let output = match status_resp.data {
                    Some(data) => {
                        serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string())
                    }
                    None => "Extraction completed but returned no data.".into(),
                };

                Ok(ToolResult::success(ctx.tool_call_id, output))
            }
            PollOutcome::Failed => {
                tracing::error!(extract_id = %start.id, "web_extract: extraction failed");
                Ok(ToolResult::failure(ctx.tool_call_id, "Extraction failed."))
            }
            PollOutcome::TimedOut => Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Extraction timed out after 120 seconds.",
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let urls = args
            .get("urls")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "?".into());
        format!("Extracting data from: {}", urls)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    use crate::search_cache::SearchResultCache;

    fn test_cache() -> SearchCacheRef {
        Arc::new(RwLock::new(SearchResultCache::new()))
    }

    #[test]
    fn api_error_keeps_the_body_out() {
        let body = r#"{"success":false,"error":"Invalid URL","echo":"Bearer fc-secret"}"#;
        let msg = api_error(reqwest::StatusCode::BAD_REQUEST, body);
        assert_eq!(msg, "Firecrawl HTTP 400: Invalid URL");
        let msg = api_error(reqwest::StatusCode::UNAUTHORIZED, body);
        assert!(!msg.contains("fc-secret"));
        let unsupported = r#"{"success":false,"error":"We do not support this site."}"#;
        let msg = api_error(reqwest::StatusCode::FORBIDDEN, unsupported);
        assert_eq!(msg, "Firecrawl HTTP 403: We do not support this site.");
        let msg = api_error(reqwest::StatusCode::BAD_GATEWAY, "Bearer fc-secret");
        assert_eq!(msg, "Firecrawl HTTP 502: temporarily unavailable");
    }

    #[test]
    fn cap_cuts_long_pages() {
        assert_eq!(cap("short"), "short");
        let long = "é".repeat(MAX_SCRAPE_CHARS + 5);
        let capped = cap(&long);
        assert!(capped.starts_with(&"é".repeat(MAX_SCRAPE_CHARS)));
        assert!(capped.ends_with(&format!(
            "[truncated: {} characters in total]",
            MAX_SCRAPE_CHARS + 5
        )));
    }

    #[test]
    fn test_web_search_tool_name() {
        let tool = WebSearchTool::new("key".into(), test_cache());
        assert_eq!(tool.name(), "firecrawl_search");
    }

    #[test]
    fn test_web_scrape_tool_name() {
        let tool = WebScrapeTool::new("key".into());
        assert_eq!(tool.name(), "web_scrape");
    }

    #[test]
    fn test_web_crawl_tool_name() {
        let tool = WebCrawlTool::new("key".into(), test_cache());
        assert_eq!(tool.name(), "web_crawl");
    }

    #[test]
    fn test_web_map_tool_name() {
        let tool = WebMapTool::new("key".into());
        assert_eq!(tool.name(), "web_map");
    }

    #[test]
    fn test_format_crawl_empty() {
        assert_eq!(
            format_crawl_results_truncated(&[], "c_000001"),
            "No pages crawled."
        );
    }

    #[test]
    fn test_format_crawl_results() {
        let pages = vec![CrawlPage {
            markdown: Some("# Hello\nWorld".into()),
            metadata: Some(CrawlMetadata {
                source_url: Some("https://example.com".into()),
                title: Some("Example".into()),
            }),
        }];
        let result = format_crawl_results_truncated(&pages, "c_000001");
        assert!(result.contains("Example"));
        assert!(result.contains("https://example.com"));
        assert!(result.contains("# Hello"));
        assert!(result.contains("c_000001"));
    }

    #[test]
    fn test_search_response_parsing() {
        let json = r#"{"success": true, "data": {"web": [{"title": "Test", "url": "https://x.com", "description": "Desc"}]}, "creditsUsed": 1}"#;
        let resp: SearchResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.data.web.len(), 1);
        assert_eq!(resp.data.web[0].title, "Test");
    }

    #[test]
    fn test_crawl_start_response_parsing() {
        let json = r#"{"success": true, "id": "abc123"}"#;
        let resp: CrawlStartResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.id, "abc123");
    }

    #[test]
    fn test_map_response_parsing() {
        let json = r#"{"success": true, "links": [{"url": "https://example.com/page1", "title": "Page 1"}, {"url": "https://example.com/page2"}]}"#;
        let resp: MapResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.links.len(), 2);
        assert_eq!(resp.links[0].title, Some("Page 1".into()));
    }

    #[test]
    fn test_web_extract_tool_name() {
        let tool = WebExtractTool::new("key".into());
        assert_eq!(tool.name(), "web_extract");
    }

    #[test]
    fn test_extract_start_response_parsing() {
        let json = r#"{"success": true, "id": "ext_123"}"#;
        let resp: ExtractStartResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.id, "ext_123");
    }

    #[test]
    fn test_extract_status_response_parsing() {
        let json = r#"{"success": true, "status": "completed", "data": {"name": "Firecrawl", "pricing": [{"plan": "Free", "price": 0}]}}"#;
        let resp: ExtractStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "completed");
        assert!(resp.data.is_some());
        let data = resp.data.unwrap();
        assert_eq!(data["name"], "Firecrawl");
        assert_eq!(data["pricing"][0]["plan"], "Free");
    }
}
