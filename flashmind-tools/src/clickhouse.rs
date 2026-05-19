//! ClickHouse query tool using the HTTP API.
//!
//! Uses ClickHouse's built-in HTTP interface with `JSONCompact` output format,
//! so no native driver crate is needed — just `reqwest` (already a non-optional
//! dependency of this crate).

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::db_common;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Connection configuration for a ClickHouse instance.
pub struct ClickHouseConfig {
    /// Base URL of the ClickHouse HTTP interface (e.g. `http://localhost:8123`).
    pub url: String,
    /// ClickHouse user. Defaults to `"default"` when `None`.
    pub user: Option<String>,
    /// Password for the user.
    pub password: Option<String>,
    /// Target database. Defaults to `"default"` when `None`.
    pub database: Option<String>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClickHouseResponse {
    meta: Vec<ColumnMeta>,
    data: Vec<Vec<Value>>,
    rows: usize,
}

#[derive(Deserialize)]
struct ColumnMeta {
    name: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    column_type: String,
}

// ---------------------------------------------------------------------------
// Value → String conversion
// ---------------------------------------------------------------------------

fn value_to_string(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.chars().count() > 200 {
                let truncated: String = s.chars().take(200).collect();
                format!("{truncated}...")
            } else {
                s.clone()
            }
        }
        Value::Bool(b) => b.to_string(),
        Value::Array(a) => format!("{a:?}"),
        Value::Object(o) => format!("{o:?}"),
    }
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClickHouseQueryArgs {
    query: String,
    limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Execute SQL queries against a ClickHouse instance via its HTTP API.
pub struct ClickHouseQueryTool {
    /// Connection configuration.
    pub config: ClickHouseConfig,
    /// When `true`, write/DDL statements are rejected and the HTTP request
    /// includes `&readonly=1` as defense-in-depth.
    pub readonly: bool,
}

#[async_trait]
impl Tool for ClickHouseQueryTool {
    fn name(&self) -> &str {
        "clickhouse_query"
    }

    fn description(&self) -> &str {
        "Execute a SQL query against a ClickHouse database via its HTTP API. \
         Returns results as a formatted table. Useful for analytics queries, \
         exploring schemas with 'SHOW TABLES' or 'DESCRIBE TABLE <name>'."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 50, max: 200)."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Query cancelled"));
        }

        let args: ClickHouseQueryArgs = ctx.parse_args(self.name())?;
        let query = args.query.trim().to_string();
        let limit = db_common::clamp_limit(args.limit);

        // Block write queries when in readonly mode.
        if self.readonly && db_common::is_write_query(&query) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Write operations not allowed. Only SELECT queries.",
            ));
        }

        let database = self.config.database.as_deref().unwrap_or("default");

        let mut url = format!(
            "{}/?database={}&default_format=JSONCompact",
            self.config.url.trim_end_matches('/'),
            database,
        );
        if self.readonly {
            url.push_str("&readonly=1");
        }

        debug!(query = %query, url = %url, "clickhouse_query: sending request");

        let client = reqwest::Client::new();
        let mut req = client.post(&url).body(query);

        if let Some(user) = &self.config.user {
            req = req.basic_auth(user, self.config.password.as_deref());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("clickhouse_query: HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("ClickHouse error ({status}): {body}"),
            ));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("clickhouse_query: failed to read response body: {e}"))?;

        let parsed: ClickHouseResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow::anyhow!("clickhouse_query: failed to parse response: {e}"))?;

        debug!(rows = parsed.rows, "clickhouse_query: response received");

        let columns: Vec<String> = parsed.meta.iter().map(|m| m.name.clone()).collect();
        let rows: Vec<Vec<String>> = parsed
            .data
            .iter()
            .take(limit)
            .map(|row| row.iter().map(value_to_string).collect())
            .collect();

        let output = db_common::format_table(&columns, &rows, 60);
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
        let short: String = query.chars().take(80).collect();
        format!("Running ClickHouse query: {short}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_to_string_null() {
        assert_eq!(value_to_string(&Value::Null), "NULL");
    }

    #[test]
    fn test_value_to_string_number() {
        assert_eq!(value_to_string(&json!(42)), "42");
        assert_eq!(value_to_string(&json!(3.14)), "3.14");
    }

    #[test]
    fn test_value_to_string_string() {
        assert_eq!(value_to_string(&json!("hello")), "hello");
    }

    #[test]
    fn test_value_to_string_long_string_truncated() {
        let long = "a".repeat(300);
        let result = value_to_string(&json!(long));
        assert!(result.ends_with("..."));
        // 200 chars + "..."
        assert_eq!(result.chars().count(), 203);
    }

    #[test]
    fn test_value_to_string_bool() {
        assert_eq!(value_to_string(&json!(true)), "true");
        assert_eq!(value_to_string(&json!(false)), "false");
    }

    #[test]
    fn test_value_to_string_array() {
        let result = value_to_string(&json!([1, 2, 3]));
        assert!(result.contains("1"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_value_to_string_object() {
        let result = value_to_string(&json!({"key": "val"}));
        assert!(result.contains("key"));
    }

    #[tokio::test]
    async fn test_readonly_blocks_write() {
        let tool = ClickHouseQueryTool {
            config: ClickHouseConfig {
                url: "http://localhost:8123".to_string(),
                user: None,
                password: None,
                database: None,
            },
            readonly: true,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(
            "test-id",
            json!({"query": "INSERT INTO t VALUES (1)"}),
            None,
            &cancel,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Write operations not allowed"));
    }

    #[tokio::test]
    async fn test_readonly_allows_select() {
        // We can't actually connect, but we can verify it gets past the
        // write-check and fails on the HTTP request instead.
        let tool = ClickHouseQueryTool {
            config: ClickHouseConfig {
                url: "http://127.0.0.1:19999".to_string(),
                user: None,
                password: None,
                database: None,
            },
            readonly: true,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("test-id", json!({"query": "SELECT 1"}), None, &cancel);

        let result = tool.execute(ctx).await;
        // Should fail with connection error, not a write-block error.
        assert!(result.is_err() || !result.unwrap().output().contains("Write operations"));
    }

    #[test]
    fn test_response_parsing() {
        let json_str = r#"{
            "meta": [
                {"name": "id", "type": "UInt64"},
                {"name": "name", "type": "String"}
            ],
            "data": [[1, "alice"], [2, "bob"]],
            "rows": 2
        }"#;

        let parsed: ClickHouseResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.rows, 2);
        assert_eq!(parsed.meta.len(), 2);
        assert_eq!(parsed.meta[0].name, "id");
        assert_eq!(parsed.meta[1].name, "name");
        assert_eq!(parsed.data.len(), 2);
        assert_eq!(parsed.data[0][0], json!(1));
        assert_eq!(parsed.data[0][1], json!("alice"));
    }

    #[test]
    fn test_response_to_table() {
        let json_str = r#"{
            "meta": [
                {"name": "id", "type": "UInt64"},
                {"name": "name", "type": "String"}
            ],
            "data": [[1, "alice"], [2, "bob"]],
            "rows": 2
        }"#;

        let parsed: ClickHouseResponse = serde_json::from_str(json_str).unwrap();
        let columns: Vec<String> = parsed.meta.iter().map(|m| m.name.clone()).collect();
        let rows: Vec<Vec<String>> = parsed
            .data
            .iter()
            .map(|row| row.iter().map(value_to_string).collect())
            .collect();

        let table = db_common::format_table(&columns, &rows, 60);
        assert!(table.contains("id"));
        assert!(table.contains("alice"));
        assert!(table.contains("bob"));
        assert!(table.contains("(2 rows)"));
    }

    #[tokio::test]
    async fn test_cancelled_returns_early() {
        let tool = ClickHouseQueryTool {
            config: ClickHouseConfig {
                url: "http://localhost:8123".to_string(),
                user: None,
                password: None,
                database: None,
            },
            readonly: false,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new("test-id", json!({"query": "SELECT 1"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("cancelled"));
    }

    #[test]
    fn test_humanize() {
        let tool = ClickHouseQueryTool {
            config: ClickHouseConfig {
                url: "http://localhost:8123".to_string(),
                user: None,
                password: None,
                database: None,
            },
            readonly: false,
        };

        let msg = tool.humanize(&json!({"query": "SELECT count() FROM events"}));
        assert!(msg.contains("SELECT count() FROM events"));
    }

    #[test]
    fn test_response_with_nulls() {
        let json_str = r#"{
            "meta": [{"name": "val", "type": "Nullable(String)"}],
            "data": [[null], ["hello"]],
            "rows": 2
        }"#;

        let parsed: ClickHouseResponse = serde_json::from_str(json_str).unwrap();
        let rows: Vec<Vec<String>> = parsed
            .data
            .iter()
            .map(|row| row.iter().map(value_to_string).collect())
            .collect();

        assert_eq!(rows[0][0], "NULL");
        assert_eq!(rows[1][0], "hello");
    }
}
