//! OpenRouter account tools for activity metrics and credit balance.
//!
//! Requires a management API key from <https://openrouter.ai/settings/keys>.

use async_trait::async_trait;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::utils::{http_client, send_with_retry};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

const ACTIVITY_URL: &str = "https://openrouter.ai/api/v1/activity";
const CREDITS_URL: &str = "https://openrouter.ai/api/v1/credits";

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ActivityResponse {
    data: Vec<ActivityItem>,
}

#[derive(Deserialize)]
struct ActivityItem {
    date: String,
    model: String,
    #[allow(dead_code)]
    provider_name: String,
    usage: Decimal,
    #[serde(default)]
    byok_usage_inference: Decimal,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

#[derive(Deserialize)]
struct CreditsResponse {
    data: CreditsData,
}

#[derive(Deserialize)]
struct CreditsData {
    total_credits: Decimal,
    total_usage: Decimal,
}

// ---------------------------------------------------------------------------
// Activity tool
// ---------------------------------------------------------------------------

pub struct OpenRouterActivityTool {
    client: reqwest::Client,
    api_key: String,
}

impl OpenRouterActivityTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct ActivityArgs {
    date: Option<String>,
    api_key_hash: Option<String>,
}

#[async_trait]
impl Tool for OpenRouterActivityTool {
    fn name(&self) -> &str {
        "openrouter_activity"
    }

    fn description(&self) -> &str {
        "Get OpenRouter usage activity for the last 30 days, grouped by model and endpoint. \
         Returns request counts, token usage, and costs. Requires a management API key."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Filter to a single UTC date (YYYY-MM-DD). Must be within the last 30 days."
                },
                "api_key_hash": {
                    "type": "string",
                    "description": "Filter by API key hash (SHA-256 hex). Omit to see all keys."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ActivityArgs = ctx.parse_args(self.name())?;

        let mut params: Vec<(&str, &str)> = Vec::new();
        if let Some(ref date) = args.date {
            params.push(("date", date));
        }
        if let Some(ref hash) = args.api_key_hash {
            params.push(("api_key_hash", hash));
        }

        let response = send_with_retry(|| {
            self.client
                .get(ACTIVITY_URL)
                .bearer_auth(&self.api_key)
                .query(&params)
        })
        .await?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("OpenRouter API error (HTTP {}): {}", status.as_u16(), body),
            ));
        }

        let resp: ActivityResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse activity response: {}. Body: {:.200}",
                e,
                body
            )
        })?;

        if resp.data.is_empty() {
            return Ok(ToolResult::success(ctx.tool_call_id, "No activity found."));
        }

        let output = format_activity(&resp.data);
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        match args.get("date").and_then(|v| v.as_str()) {
            Some(date) => format!("Fetching OpenRouter activity for {}", date),
            None => "Fetching OpenRouter activity".into(),
        }
    }
}

fn format_activity(items: &[ActivityItem]) -> String {
    let mut total_cost = Decimal::ZERO;
    let mut total_requests = 0u64;
    let mut total_prompt = 0u64;
    let mut total_completion = 0u64;
    let mut total_reasoning = 0u64;

    let mut lines = Vec::with_capacity(items.len() + 4);
    lines.push(format!(
        "{:<12} {:<40} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "Date", "Model", "Requests", "Prompt", "Completion", "Reasoning", "Cost"
    ));
    lines.push("-".repeat(110));

    for item in items {
        let cost = item.usage + item.byok_usage_inference;
        total_cost += cost;
        total_requests += item.requests;
        total_prompt += item.prompt_tokens;
        total_completion += item.completion_tokens;
        total_reasoning += item.reasoning_tokens;

        lines.push(format!(
            "{:<12} {:<40} {:>8} {:>12} {:>12} {:>12} {:>10}",
            item.date,
            truncate_model(&item.model, 40),
            item.requests,
            item.prompt_tokens,
            item.completion_tokens,
            item.reasoning_tokens,
            cost.round_dp(4),
        ));
    }

    lines.push("-".repeat(110));
    lines.push(format!(
        "{:<12} {:<40} {:>8} {:>12} {:>12} {:>12} ${:>9}",
        "TOTAL",
        "",
        total_requests,
        total_prompt,
        total_completion,
        total_reasoning,
        total_cost.round_dp(4),
    ));

    lines.join("\n")
}

fn truncate_model(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max - 1).collect();
        format!("{}…", truncated)
    }
}

// ---------------------------------------------------------------------------
// Credits tool
// ---------------------------------------------------------------------------

pub struct OpenRouterCreditsTool {
    client: reqwest::Client,
    api_key: String,
}

impl OpenRouterCreditsTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: http_client(),
            api_key,
        }
    }
}

#[async_trait]
impl Tool for OpenRouterCreditsTool {
    fn name(&self) -> &str {
        "openrouter_credits"
    }

    fn description(&self) -> &str {
        "Get remaining OpenRouter credit balance. Returns total credits purchased, \
         total used, and remaining balance. Requires a management API key."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let response =
            send_with_retry(|| self.client.get(CREDITS_URL).bearer_auth(&self.api_key)).await?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("OpenRouter API error (HTTP {}): {}", status.as_u16(), body),
            ));
        }

        let resp: CreditsResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse credits response: {}. Body: {:.200}",
                e,
                body
            )
        })?;

        let remaining = resp.data.total_credits - resp.data.total_usage;
        let output = format!(
            "Credits purchased: ${}\nCredits used:      ${}\nRemaining:         ${}",
            resp.data.total_credits.round_dp(4),
            resp.data.total_usage.round_dp(4),
            remaining.round_dp(4),
        );

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Checking OpenRouter credits".into()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names() {
        assert_eq!(
            OpenRouterActivityTool::new("k".into()).name(),
            "openrouter_activity"
        );
        assert_eq!(
            OpenRouterCreditsTool::new("k".into()).name(),
            "openrouter_credits"
        );
    }

    #[test]
    fn format_activity_empty() {
        let output = format_activity(&[]);
        assert!(output.contains("Date"));
        assert!(output.contains("TOTAL"));
    }

    #[test]
    fn format_activity_single() {
        use rust_decimal_macros::dec;
        let items = vec![ActivityItem {
            date: "2025-05-20".into(),
            model: "anthropic/claude-sonnet-4".into(),
            provider_name: "Anthropic".into(),
            usage: dec!(1.2345),
            byok_usage_inference: Decimal::ZERO,
            requests: 42,
            prompt_tokens: 10000,
            completion_tokens: 5000,
            reasoning_tokens: 0,
        }];
        let output = format_activity(&items);
        assert!(output.contains("claude-sonnet-4"));
        assert!(output.contains("42"));
        assert!(output.contains("10000"));
        assert!(output.contains("1.2345"));
    }

    #[test]
    fn truncate_model_short() {
        assert_eq!(truncate_model("openai/gpt-4o", 40), "openai/gpt-4o");
    }

    #[test]
    fn truncate_model_long() {
        let long = "a".repeat(50);
        let result = truncate_model(&long, 40);
        assert_eq!(result.chars().count(), 40);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn parse_credits_response() {
        use rust_decimal_macros::dec;
        let json = r#"{"data":{"total_credits":100.5,"total_usage":25.75}}"#;
        let resp: CreditsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.total_credits, dec!(100.5));
        assert_eq!(resp.data.total_usage, dec!(25.75));
    }

    #[test]
    fn parse_activity_response() {
        let json = r#"{"data":[{
            "date":"2025-05-20","model":"openai/gpt-4.1","model_permaslug":"openai/gpt-4.1-2025-04-14",
            "endpoint_id":"550e8400-e29b-41d4-a716-446655440000","provider_name":"OpenAI",
            "usage":0.015,"byok_usage_inference":0.0,"requests":5,
            "prompt_tokens":50,"completion_tokens":125,"reasoning_tokens":25
        }]}"#;
        let resp: ActivityResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].requests, 5);
    }
}
