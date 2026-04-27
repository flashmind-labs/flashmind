//! Fetch web page content as text via agent-browser.
//!
//! Complements `web_search` — search returns URLs, `web_fetch` retrieves
//! page content with pagination (limit/offset) for on-demand reading.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;


use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

/// Default number of lines returned per fetch.
const DEFAULT_LIMIT: usize = 200;

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    /// Number of lines to return (default: 200).
    limit: Option<usize>,
    /// Line offset to start from (default: 0).
    offset: Option<usize>,

    /// Optional CDP port to attach to existing browser.
    #[serde(alias = "cdp_port", alias = "cdpPort", alias = "cdp-port")]
    cdp_port: Option<u16>,
}

/// Fetch a web page's text content via agent-browser, with pagination.
pub struct WebFetchTool {
    pub browser_engine: String,
}

impl WebFetchTool {
    pub fn new(browser_engine: String) -> Self {
        Self { browser_engine }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and return its text content, extracted by a headless browser. \
         Long pages are paginated — use limit/offset to read more. \
         For searching the web, use web_search first, then web_search_read to read results."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "URL to fetch" },
                "limit": { "type": "integer", "description": "Max lines to return (default: 200)" },
                "offset": { "type": "integer", "description": "Line offset to start from (default: 0)" },
                "cdp_port": {
                    "type": "integer",
                    "description": "Optional CDP port to attach to existing browser instance (e.g., 9222)."
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: WebFetchArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(DEFAULT_LIMIT);
        let offset = args.offset.unwrap_or(0);

        tracing::debug!(url = %args.url, limit, offset, "web_fetch: fetching page");

        // Build base command with engine flag via env var (--engine arg is deprecated)
        let mut open_cmd = Command::new("agent-browser");
        if self.browser_engine != "chrome" {
            open_cmd.env("AGENT_BROWSER_ENGINE", &self.browser_engine);
        }
        if let Some(port) = args.cdp_port {
            open_cmd.arg("connect").arg(port.to_string());
        }
        open_cmd.arg("open").arg(&args.url);

        let open_output = open_cmd.output().await;

        match open_output {
            Ok(out) if !out.status.success() => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!(url = %args.url, "web_fetch: failed to open URL");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Failed to open URL: {}", stderr),
                ));
            }
            Err(e) => {
                tracing::warn!(error = %e, "web_fetch: agent-browser not available");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Failed to run agent-browser: {}. Is it installed?", e),
                ));
            }
            _ => {}
        }

        // Get text content with same engine/CDP settings
        let mut get_text_cmd = Command::new("agent-browser");
        if self.browser_engine != "chrome" {
            get_text_cmd.env("AGENT_BROWSER_ENGINE", &self.browser_engine);
        }
        if let Some(port) = args.cdp_port {
            get_text_cmd.arg("connect").arg(port.to_string());
        }
        get_text_cmd.args(["get", "text", "body"]);

        let text = match get_text_cmd.output().await {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Failed to get text: {}", stderr),
                ));
            }
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Failed to run agent-browser: {}", e),
                ));
            }
        };

        // Paginate by lines
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        tracing::debug!(
            total_lines = total,
            start = offset.min(total),
            "web_fetch: page content retrieved"
        );
        let start = offset.min(total);
        let end = (start + limit).min(total);
        let page = &lines[start..end];

        let mut output = page.join("\n");

        // Append pagination hint if there's more content
        if end < total {
            output.push_str(&format!(
                "\n\n--- Showing lines {}-{} of {} total. Use offset={} to see more. ---",
                start + 1,
                end,
                total,
                end
            ));
        } else if start > 0 {
            output.push_str(&format!(
                "\n\n--- Showing lines {}-{} of {} total. ---",
                start + 1,
                end,
                total
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");

        // Build connection info string
        let mut connection_info = Vec::new();
        if let Some(port) = args.get("cdp_port").and_then(|v| v.as_u64()) {
            connection_info.push(format!("CDP:{}", port));
        }
        let conn_prefix = if !connection_info.is_empty() {
            format!("[{}] ", connection_info.join(", "))
        } else {
            String::new()
        };

        format!("WebFetch{}: {}", conn_prefix, url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> WebFetchTool {
        WebFetchTool::new("chrome".into())
    }

    #[test]
    fn test_tool_name() {
        assert_eq!(tool().name(), "web_fetch");
    }

    #[test]
    fn test_parameters_has_url() {
        let params = tool().parameters();
        assert!(params["properties"]["url"].is_object());
        assert!(params["properties"]["limit"].is_object());
        assert!(params["properties"]["offset"].is_object());
    }
}
