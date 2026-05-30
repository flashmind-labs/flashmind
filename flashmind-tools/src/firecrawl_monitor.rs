//! Firecrawl monitoring tools for tracking web content changes.
//!
//! <https://docs.firecrawl.dev/api-reference/v2-introduction>

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{Instrument, info_span};

use crate::utils::http_client;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

const FIRECRAWL_BASE_V2: &str = "https://api.firecrawl.dev/v2";

// ---------------------------------------------------------------------------
// Web Monitor Create
// ---------------------------------------------------------------------------

/// Create a web monitor to track changes on a page.
pub struct WebMonitorCreateTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorCreateTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorCreateArgs {
    url: String,
    schedule: String,
    #[serde(default)]
    goal: Option<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    webhook_url: Option<String>,
    #[serde(default)]
    email: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct MonitorCreateResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    id: Option<String>,
}

#[async_trait]
impl Tool for WebMonitorCreateTool {
    fn name(&self) -> &str {
        "web_monitor_create"
    }

    fn description(&self) -> &str {
        "Create a monitor to track changes on a web page. Supports scheduled checks with intelligent change detection."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "URL to monitor" },
                "schedule": { "type": "string", "description": "Schedule as natural language ('every 30 minutes', 'daily at 9am') or cron expression" },
                "goal": { "type": "string", "description": "Plain language description of what constitutes a meaningful change (enables LLM judgment)" },
                "schema": { "type": "object", "description": "JSON schema for structured field-level change tracking (json diff mode)" },
                "webhook_url": { "type": "string", "description": "URL to receive change notifications" },
                "email": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Email addresses for change alerts"
                }
            },
            "required": ["url", "schedule"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorCreateArgs = ctx.parse_args(self.name())?;

        tracing::debug!(url = %args.url, schedule = %args.schedule, "web_monitor_create: creating monitor");

        // Build scrape options based on whether a schema is provided
        let scrape_options = if let Some(ref schema) = args.schema {
            json!({
                "formats": [{ "type": "changeTracking", "modes": ["json"], "schema": schema }]
            })
        } else {
            json!({
                "formats": [{ "type": "markdown" }, { "type": "changeTracking", "modes": ["git-diff"] }]
            })
        };

        let mut body = json!({
            "urls": [args.url],
            "schedule": args.schedule,
            "scrapeOptions": scrape_options,
        });

        if let Some(ref goal) = args.goal {
            body["goal"] = json!(goal);
        }

        // Build notifications object
        let mut notifications = json!({});
        if let Some(ref webhook_url) = args.webhook_url {
            notifications["webhook"] = json!({ "url": webhook_url });
        }
        if let Some(ref recipients) = args.email {
            notifications["email"] = json!({ "recipients": recipients });
        }
        if notifications.as_object().is_some_and(|o| !o.is_empty()) {
            body["notifications"] = notifications;
        }

        let response = self
            .client
            .post(format!("{}/monitor", FIRECRAWL_BASE_V2))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("POST /monitor url={}", args.url).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_create: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_create: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_create: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: MonitorCreateResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_create: failed to parse response");
            anyhow::anyhow!("web_monitor_create: {e}")
        })?;

        if !resp.success {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Monitor creation failed: {}", body_text),
            ));
        }

        let id = resp.id.unwrap_or_else(|| "unknown".into());
        tracing::debug!(monitor_id = %id, "web_monitor_create: monitor created");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Monitor created successfully.\n\nMonitor ID: {}\nURL: {}\nSchedule: {}", id, args.url, args.schedule),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Creating monitor for {}", url)
    }
}

// ---------------------------------------------------------------------------
// Web Monitor List
// ---------------------------------------------------------------------------

/// List all active web monitors.
pub struct WebMonitorListTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorListTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorListArgs {
    #[serde(default)]
    limit: Option<u64>,
}

#[derive(Deserialize)]
struct MonitorListResponse {
    #[serde(default)]
    monitors: Vec<MonitorSummary>,
}

#[derive(Deserialize)]
struct MonitorSummary {
    #[serde(default)]
    id: String,
    #[serde(default)]
    urls: Vec<String>,
    #[serde(default)]
    schedule: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

#[async_trait]
impl Tool for WebMonitorListTool {
    fn name(&self) -> &str {
        "web_monitor_list"
    }

    fn description(&self) -> &str {
        "List all active web monitors."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "description": "Max monitors to return" }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorListArgs = ctx.parse_args(self.name())?;

        tracing::debug!("web_monitor_list: listing monitors");

        let mut url = format!("{}/monitor", FIRECRAWL_BASE_V2);
        if let Some(limit) = args.limit {
            url.push_str(&format!("?limit={}", limit));
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = "GET /monitor",
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_list: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_list: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_list: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: MonitorListResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_list: failed to parse response");
            anyhow::anyhow!("web_monitor_list: {e}")
        })?;

        if resp.monitors.is_empty() {
            return Ok(ToolResult::success(ctx.tool_call_id, "No monitors found."));
        }

        tracing::debug!(count = resp.monitors.len(), "web_monitor_list: found monitors");

        let mut output = format!("# Web Monitors ({} total)\n\n", resp.monitors.len());
        output.push_str("| ID | URL(s) | Schedule | Status |\n");
        output.push_str("|---|---|---|---|\n");

        for m in &resp.monitors {
            let urls = if m.urls.is_empty() {
                "-".to_string()
            } else {
                m.urls.join(", ")
            };
            let schedule = m.schedule.as_deref().unwrap_or("-");
            let status = m.status.as_deref().unwrap_or("-");
            output.push_str(&format!("| {} | {} | {} | {} |\n", m.id, urls, schedule, status));
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing web monitors".into()
    }
}

// ---------------------------------------------------------------------------
// Web Monitor Get
// ---------------------------------------------------------------------------

/// Get details of a specific web monitor.
pub struct WebMonitorGetTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorGetTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorGetArgs {
    id: String,
}

#[async_trait]
impl Tool for WebMonitorGetTool {
    fn name(&self) -> &str {
        "web_monitor_get"
    }

    fn description(&self) -> &str {
        "Get details of a specific web monitor."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "The monitor ID" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorGetArgs = ctx.parse_args(self.name())?;

        tracing::debug!(monitor_id = %args.id, "web_monitor_get: fetching monitor");

        let response = self
            .client
            .get(format!("{}/monitor/{}", FIRECRAWL_BASE_V2, args.id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("GET /monitor/{}", args.id).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_get: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_get: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_get: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        // Pretty-print the full JSON response
        let parsed: Value = serde_json::from_str(&body_text).unwrap_or_else(|_| json!(body_text));
        let pretty = serde_json::to_string_pretty(&parsed).unwrap_or(body_text);

        tracing::debug!(monitor_id = %args.id, "web_monitor_get: fetched successfully");

        Ok(ToolResult::success(ctx.tool_call_id, pretty))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Getting monitor {}", id)
    }
}

// ---------------------------------------------------------------------------
// Web Monitor Delete
// ---------------------------------------------------------------------------

/// Delete a web monitor.
pub struct WebMonitorDeleteTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorDeleteTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorDeleteArgs {
    id: String,
}

#[derive(Deserialize)]
struct MonitorDeleteResponse {
    #[serde(default)]
    success: bool,
}

#[async_trait]
impl Tool for WebMonitorDeleteTool {
    fn name(&self) -> &str {
        "web_monitor_delete"
    }

    fn description(&self) -> &str {
        "Delete a web monitor."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "The monitor ID to delete" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorDeleteArgs = ctx.parse_args(self.name())?;

        tracing::debug!(monitor_id = %args.id, "web_monitor_delete: deleting monitor");

        let response = self
            .client
            .delete(format!("{}/monitor/{}", FIRECRAWL_BASE_V2, args.id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("DELETE /monitor/{}", args.id).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_delete: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_delete: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_delete: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: MonitorDeleteResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_delete: failed to parse response");
            anyhow::anyhow!("web_monitor_delete: {e}")
        })?;

        if !resp.success {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Delete failed: {}", body_text),
            ));
        }

        tracing::debug!(monitor_id = %args.id, "web_monitor_delete: deleted successfully");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Monitor {} deleted successfully.", args.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Deleting monitor {}", id)
    }
}

// ---------------------------------------------------------------------------
// Web Monitor Run
// ---------------------------------------------------------------------------

/// Trigger an immediate check for a web monitor.
pub struct WebMonitorRunTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorRunTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorRunArgs {
    id: String,
}

#[derive(Deserialize)]
struct MonitorRunResponse {
    #[serde(default)]
    success: bool,
    #[serde(default, rename = "checkId")]
    check_id: Option<String>,
}

#[async_trait]
impl Tool for WebMonitorRunTool {
    fn name(&self) -> &str {
        "web_monitor_run"
    }

    fn description(&self) -> &str {
        "Trigger an immediate check for a web monitor."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "The monitor ID to trigger" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorRunArgs = ctx.parse_args(self.name())?;

        tracing::debug!(monitor_id = %args.id, "web_monitor_run: triggering check");

        let response = self
            .client
            .post(format!("{}/monitor/{}/run", FIRECRAWL_BASE_V2, args.id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("POST /monitor/{}/run", args.id).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_run: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_run: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_run: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: MonitorRunResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_run: failed to parse response");
            anyhow::anyhow!("web_monitor_run: {e}")
        })?;

        if !resp.success {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Run failed: {}", body_text),
            ));
        }

        let check_id = resp.check_id.unwrap_or_else(|| "unknown".into());
        tracing::debug!(monitor_id = %args.id, check_id = %check_id, "web_monitor_run: check triggered");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Check triggered successfully.\n\nMonitor ID: {}\nCheck ID: {}\n\nUse web_monitor_checks to view results.", args.id, check_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Running monitor check {}", id)
    }
}

// ---------------------------------------------------------------------------
// Web Monitor Checks
// ---------------------------------------------------------------------------

/// Get check results for a web monitor.
pub struct WebMonitorChecksTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebMonitorChecksTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct MonitorChecksArgs {
    id: String,
    #[serde(default)]
    check_id: Option<String>,
}

#[derive(Deserialize)]
struct ChecksListResponse {
    #[serde(default)]
    checks: Vec<CheckSummary>,
}

#[derive(Deserialize)]
struct CheckSummary {
    #[serde(default)]
    id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    summary: Option<CheckSummaryDetail>,
}

#[derive(Deserialize)]
struct CheckSummaryDetail {
    #[serde(default)]
    changed: u64,
    #[serde(default)]
    same: u64,
    #[serde(default)]
    new: u64,
    #[serde(default)]
    removed: u64,
}

#[derive(Deserialize)]
struct CheckDetailResponse {
    #[serde(default)]
    id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    summary: Option<CheckSummaryDetail>,
    #[serde(default)]
    results: Vec<CheckPageResult>,
}

#[derive(Deserialize)]
struct CheckPageResult {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    diff: Option<String>,
    #[serde(default)]
    changes: Option<Value>,
}

#[async_trait]
impl Tool for WebMonitorChecksTool {
    fn name(&self) -> &str {
        "web_monitor_checks"
    }

    fn description(&self) -> &str {
        "Get check results for a web monitor. Lists recent checks or retrieves a specific check with change details."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "The monitor ID" },
                "check_id": { "type": "string", "description": "Specific check ID to retrieve (omit to list recent checks)" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: MonitorChecksArgs = ctx.parse_args(self.name())?;

        if let Some(ref check_id) = args.check_id {
            self.get_check_detail(&ctx, &args.id, check_id).await
        } else {
            self.list_checks(&ctx, &args.id).await
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        if let Some(check_id) = args.get("check_id").and_then(|v| v.as_str()) {
            format!("Getting check {} for monitor {}", check_id, id)
        } else {
            format!("Listing checks for monitor {}", id)
        }
    }
}

impl WebMonitorChecksTool {
    async fn list_checks(
        &self,
        ctx: &ToolContext<'_>,
        monitor_id: &str,
    ) -> anyhow::Result<ToolResult> {
        tracing::debug!(monitor_id = %monitor_id, "web_monitor_checks: listing checks");

        let response = self
            .client
            .get(format!("{}/monitor/{}/checks", FIRECRAWL_BASE_V2, monitor_id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("GET /monitor/{}/checks", monitor_id).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_checks: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_checks: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_checks: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: ChecksListResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_checks: failed to parse response");
            anyhow::anyhow!("web_monitor_checks: {e}")
        })?;

        if resp.checks.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("No checks found for monitor {}.", monitor_id),
            ));
        }

        tracing::debug!(monitor_id = %monitor_id, count = resp.checks.len(), "web_monitor_checks: found checks");

        let mut output = format!(
            "# Checks for Monitor {} ({} total)\n\n",
            monitor_id,
            resp.checks.len()
        );

        for check in &resp.checks {
            let status = check.status.as_deref().unwrap_or("unknown");
            let created = check.created_at.as_deref().unwrap_or("-");

            output.push_str(&format!("## Check {}\n", check.id));
            output.push_str(&format!("- **Status:** {}\n", status));
            output.push_str(&format!("- **Created:** {}\n", created));

            if let Some(ref summary) = check.summary {
                output.push_str(&format!(
                    "- **Changes:** {} changed, {} same, {} new, {} removed\n",
                    summary.changed, summary.same, summary.new, summary.removed
                ));
            }
            output.push('\n');
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    async fn get_check_detail(
        &self,
        ctx: &ToolContext<'_>,
        monitor_id: &str,
        check_id: &str,
    ) -> anyhow::Result<ToolResult> {
        tracing::debug!(monitor_id = %monitor_id, check_id = %check_id, "web_monitor_checks: fetching check detail");

        let response = self
            .client
            .get(format!(
                "{}/monitor/{}/checks/{}",
                FIRECRAWL_BASE_V2, monitor_id, check_id
            ))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .instrument(info_span!(target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("GET /monitor/{}/checks/{}", monitor_id, check_id).as_str(),
            ))
            .await
            .map_err(|e| anyhow::anyhow!("web_monitor_checks: {e}"))?;

        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            if status.as_u16() == 429 {
                tracing::warn!("web_monitor_checks: rate limited (429)");
            } else {
                tracing::error!(status = status.as_u16(), "web_monitor_checks: API error");
            }
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("HTTP {}: {}", status.as_u16(), body_text),
            ));
        }

        let resp: CheckDetailResponse = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(err = %e, "web_monitor_checks: failed to parse check detail");
            anyhow::anyhow!("web_monitor_checks: {e}")
        })?;

        tracing::debug!(check_id = %resp.id, "web_monitor_checks: fetched check detail");

        let check_status = resp.status.as_deref().unwrap_or("unknown");

        let mut output = format!("# Check {} (Monitor {})\n\n", resp.id, monitor_id);
        output.push_str(&format!("**Status:** {}\n\n", check_status));

        if let Some(ref summary) = resp.summary {
            output.push_str("## Summary\n\n");
            output.push_str(&format!("- **Changed:** {}\n", summary.changed));
            output.push_str(&format!("- **Same:** {}\n", summary.same));
            output.push_str(&format!("- **New:** {}\n", summary.new));
            output.push_str(&format!("- **Removed:** {}\n\n", summary.removed));
        }

        if !resp.results.is_empty() {
            output.push_str("## Page Results\n\n");
            for (i, result) in resp.results.iter().enumerate() {
                let url = result.url.as_deref().unwrap_or("unknown");
                let page_status = result.status.as_deref().unwrap_or("unknown");

                output.push_str(&format!("### {}. {} ({})\n\n", i + 1, url, page_status));

                if let Some(ref diff) = result.diff {
                    output.push_str("```diff\n");
                    output.push_str(diff);
                    output.push_str("\n```\n\n");
                }

                if let Some(ref changes) = result.changes {
                    let pretty =
                        serde_json::to_string_pretty(changes).unwrap_or_else(|_| changes.to_string());
                    output.push_str("**Field changes:**\n```json\n");
                    output.push_str(&pretty);
                    output.push_str("\n```\n\n");
                }
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_monitor_create_tool_name() {
        let tool = WebMonitorCreateTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_create");
    }

    #[test]
    fn test_web_monitor_list_tool_name() {
        let tool = WebMonitorListTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_list");
    }

    #[test]
    fn test_web_monitor_get_tool_name() {
        let tool = WebMonitorGetTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_get");
    }

    #[test]
    fn test_web_monitor_delete_tool_name() {
        let tool = WebMonitorDeleteTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_delete");
    }

    #[test]
    fn test_web_monitor_run_tool_name() {
        let tool = WebMonitorRunTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_run");
    }

    #[test]
    fn test_web_monitor_checks_tool_name() {
        let tool = WebMonitorChecksTool::new("key".into());
        assert_eq!(tool.name(), "web_monitor_checks");
    }

    #[test]
    fn test_monitor_create_response_parsing() {
        let json = r#"{"success": true, "id": "mon_abc123"}"#;
        let resp: MonitorCreateResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.id, Some("mon_abc123".into()));
    }

    #[test]
    fn test_monitor_list_response_parsing() {
        let json = r#"{"monitors": [{"id": "mon_1", "urls": ["https://example.com"], "schedule": "every 30 minutes", "status": "active"}]}"#;
        let resp: MonitorListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.monitors.len(), 1);
        assert_eq!(resp.monitors[0].id, "mon_1");
        assert_eq!(resp.monitors[0].urls, vec!["https://example.com"]);
        assert_eq!(resp.monitors[0].schedule, Some("every 30 minutes".into()));
        assert_eq!(resp.monitors[0].status, Some("active".into()));
    }

    #[test]
    fn test_monitor_delete_response_parsing() {
        let json = r#"{"success": true}"#;
        let resp: MonitorDeleteResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
    }

    #[test]
    fn test_monitor_run_response_parsing() {
        let json = r#"{"success": true, "checkId": "chk_xyz789"}"#;
        let resp: MonitorRunResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.check_id, Some("chk_xyz789".into()));
    }

    #[test]
    fn test_checks_list_response_parsing() {
        let json = r#"{"checks": [{"id": "chk_1", "status": "completed", "created_at": "2026-01-01T00:00:00Z", "summary": {"changed": 2, "same": 1, "new": 0, "removed": 1}}]}"#;
        let resp: ChecksListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.checks.len(), 1);
        assert_eq!(resp.checks[0].id, "chk_1");
        let summary = resp.checks[0].summary.as_ref().unwrap();
        assert_eq!(summary.changed, 2);
        assert_eq!(summary.same, 1);
        assert_eq!(summary.new, 0);
        assert_eq!(summary.removed, 1);
    }

    #[test]
    fn test_check_detail_response_parsing() {
        let json = r#"{
            "id": "chk_1",
            "status": "completed",
            "summary": {"changed": 1, "same": 0, "new": 0, "removed": 0},
            "results": [{"url": "https://example.com", "status": "changed", "diff": "- old line\n+ new line"}]
        }"#;
        let resp: CheckDetailResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "chk_1");
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].url, Some("https://example.com".into()));
        assert_eq!(resp.results[0].status, Some("changed".into()));
        assert!(resp.results[0].diff.as_ref().unwrap().contains("+ new line"));
    }

    #[test]
    fn test_check_detail_with_field_changes() {
        let json = r#"{
            "id": "chk_2",
            "status": "completed",
            "results": [{"url": "https://example.com", "status": "changed", "changes": {"price": {"old": "$10", "new": "$15"}}}]
        }"#;
        let resp: CheckDetailResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.results.len(), 1);
        let changes = resp.results[0].changes.as_ref().unwrap();
        assert_eq!(changes["price"]["new"], "$15");
    }
}
