//! MySQL query tool.
//!
//! Allows agents to execute SQL queries against a MySQL database. Supports
//! read-only mode that blocks write operations both client-side and via
//! `SET SESSION TRANSACTION READ ONLY` as defense-in-depth.

use async_trait::async_trait;
use mysql_async::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::db_common;
use crate::utils::truncate_utf8;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for connecting to a MySQL database.
pub struct MysqlConfig {
    /// MySQL connection URL, e.g. `mysql://user:pass@host:port/dbname`.
    pub connection_string: String,
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MysqlQueryArgs {
    query: String,
    limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Value conversion
// ---------------------------------------------------------------------------

fn value_to_string(v: &mysql_async::Value) -> String {
    match v {
        mysql_async::Value::NULL => "NULL".to_string(),
        mysql_async::Value::Int(n) => n.to_string(),
        mysql_async::Value::UInt(n) => n.to_string(),
        mysql_async::Value::Float(f) => format!("{f:.4}"),
        mysql_async::Value::Double(f) => format!("{f:.4}"),
        mysql_async::Value::Bytes(b) => {
            let s = String::from_utf8_lossy(b);
            if s.len() > 200 {
                format!("{}...", truncate_utf8(&s, 200))
            } else {
                s.to_string()
            }
        }
        mysql_async::Value::Date(y, m, d, h, min, s, _us) => {
            format!("{y:04}-{m:02}-{d:02} {h:02}:{min:02}:{s:02}")
        }
        mysql_async::Value::Time(neg, d, h, m, s, _us) => {
            let sign = if *neg { "-" } else { "" };
            if *d > 0 {
                format!("{sign}{d}d {h:02}:{m:02}:{s:02}")
            } else {
                format!("{sign}{h:02}:{m:02}:{s:02}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Execute SQL queries against a MySQL database.
pub struct MysqlQueryTool {
    /// MySQL connection URL.
    pub connection_string: String,
    /// When `true`, write/DDL queries are rejected and the session is set to
    /// read-only mode as defense-in-depth.
    pub readonly: bool,
}

#[async_trait]
impl Tool for MysqlQueryTool {
    fn name(&self) -> &str {
        "mysql_query"
    }

    fn description(&self) -> &str {
        "Execute a SQL query against a MySQL database. \
         Returns results as a formatted table. Use for inspecting databases, \
         running queries, or exploring schemas with 'SHOW TABLES' or \
         'SELECT * FROM information_schema.tables'."
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

        let args: MysqlQueryArgs = ctx.parse_args(self.name())?;
        let limit = db_common::clamp_limit(args.limit);
        let query = args.query.trim().to_string();

        if self.readonly && db_common::is_write_query(&query) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Write operations not allowed. Only SELECT queries.",
            ));
        }

        debug!(query = %query, limit, readonly = self.readonly, "mysql_query: executing");

        let opts = mysql_async::Opts::from_url(&self.connection_string)
            .map_err(|e| anyhow::anyhow!("Invalid MySQL connection string: {e}"))?;

        let cancel = ctx.cancel_token().clone();
        let readonly = self.readonly;

        let result = tokio::spawn(async move {
            let mut conn = mysql_async::Conn::new(opts)
                .await
                .map_err(|e| format!("Failed to connect to MySQL: {e}"))?;

            // Defense-in-depth: enforce read-only at the session level.
            if readonly {
                conn.query_drop("SET SESSION TRANSACTION READ ONLY")
                    .await
                    .map_err(|e| format!("Failed to set read-only mode: {e}"))?;
            }

            if cancel.is_cancelled() {
                return Err("Query cancelled".to_string());
            }

            let mut result = conn
                .query_iter(&query)
                .await
                .map_err(|e| format!("Query error: {e}"))?;

            let columns: Vec<String> = result
                .columns_ref()
                .iter()
                .map(|c| c.name_str().to_string())
                .collect();

            let rows: Vec<mysql_async::Row> = result
                .collect()
                .await
                .map_err(|e| format!("Failed to collect results: {e}"))?;

            drop(result);
            conn.disconnect().await.ok();

            let string_rows: Vec<Vec<String>> = rows
                .iter()
                .take(limit)
                .map(|row| {
                    (0..columns.len())
                        .map(|i| {
                            row.as_ref(i)
                                .map(value_to_string)
                                .unwrap_or_else(|| "NULL".to_string())
                        })
                        .collect()
                })
                .collect();

            Ok(db_common::format_table(&columns, &string_rows, 60))
        })
        .await
        .map_err(|e| anyhow::anyhow!("mysql_query: {e}"))?;

        match result {
            Ok(output) => Ok(ToolResult::success(ctx.tool_call_id, output)),
            Err(err) => {
                if ctx.cancel_token().is_cancelled() || err.contains("cancel") {
                    Ok(ToolResult::failure(ctx.tool_call_id, "Query cancelled"))
                } else {
                    Ok(ToolResult::failure(ctx.tool_call_id, err))
                }
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let preview: String = query.chars().take(80).collect();
        format!("Running MySQL query: {preview}")
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
        assert_eq!(value_to_string(&mysql_async::Value::NULL), "NULL");
    }

    #[test]
    fn test_value_to_string_int() {
        assert_eq!(value_to_string(&mysql_async::Value::Int(42)), "42");
        assert_eq!(value_to_string(&mysql_async::Value::Int(-7)), "-7");
    }

    #[test]
    fn test_value_to_string_uint() {
        assert_eq!(value_to_string(&mysql_async::Value::UInt(100)), "100");
    }

    #[test]
    fn test_value_to_string_float() {
        assert_eq!(value_to_string(&mysql_async::Value::Float(3.14)), "3.1400");
        assert_eq!(
            value_to_string(&mysql_async::Value::Double(2.71828)),
            "2.7183"
        );
    }

    #[test]
    fn test_value_to_string_bytes() {
        let v = mysql_async::Value::Bytes(b"hello".to_vec());
        assert_eq!(value_to_string(&v), "hello");
    }

    #[test]
    fn test_value_to_string_bytes_truncated() {
        let long = "x".repeat(300);
        let v = mysql_async::Value::Bytes(long.into_bytes());
        let s = value_to_string(&v);
        assert!(s.ends_with("..."));
        assert!(s.len() <= 210);
    }

    #[test]
    fn test_value_to_string_date() {
        let v = mysql_async::Value::Date(2026, 5, 19, 14, 30, 0, 0);
        assert_eq!(value_to_string(&v), "2026-05-19 14:30:00");
    }

    #[test]
    fn test_value_to_string_time() {
        let v = mysql_async::Value::Time(false, 0, 1, 30, 45, 0);
        assert_eq!(value_to_string(&v), "01:30:45");
    }

    #[test]
    fn test_value_to_string_time_negative_with_days() {
        let v = mysql_async::Value::Time(true, 2, 3, 0, 0, 0);
        assert_eq!(value_to_string(&v), "-2d 03:00:00");
    }

    #[tokio::test]
    async fn test_readonly_blocks_write() {
        let tool = MysqlQueryTool {
            connection_string: "mysql://fake:fake@localhost/test".to_string(),
            readonly: true,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(
            "id",
            json!({"query": "INSERT INTO t VALUES (1)"}),
            None,
            &cancel,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Write operations not allowed"));
    }

    #[tokio::test]
    async fn test_readonly_blocks_drop() {
        let tool = MysqlQueryTool {
            connection_string: "mysql://fake:fake@localhost/test".to_string(),
            readonly: true,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("id", json!({"query": "DROP TABLE t"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Write operations not allowed"));
    }

    #[tokio::test]
    async fn test_writable_allows_write_keyword() {
        // With readonly=false, write queries are not blocked client-side.
        // They will fail at connection time since we use a fake URL, but
        // the point is they pass the write-check gate.
        let tool = MysqlQueryTool {
            connection_string: "mysql://fake:fake@localhost:13306/test".to_string(),
            readonly: false,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(
            "id",
            json!({"query": "INSERT INTO t VALUES (1)"}),
            None,
            &cancel,
        );

        let result = tool.execute(ctx).await.unwrap();
        // Should fail with a connection error, NOT a write-block error.
        assert!(!result.is_success());
        assert!(
            !result.output().contains("Write operations not allowed"),
            "writable tool should not block writes, got: {}",
            result.output()
        );
    }

    #[tokio::test]
    async fn test_cancelled_returns_immediately() {
        let tool = MysqlQueryTool {
            connection_string: "mysql://fake:fake@localhost/test".to_string(),
            readonly: true,
        };

        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new("id", json!({"query": "SELECT 1"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().to_lowercase().contains("cancel"));
    }

    #[test]
    fn test_humanize() {
        let tool = MysqlQueryTool {
            connection_string: String::new(),
            readonly: true,
        };
        let h = tool.humanize(&json!({"query": "SELECT * FROM users"}));
        assert!(h.contains("SELECT * FROM users"));
    }
}
