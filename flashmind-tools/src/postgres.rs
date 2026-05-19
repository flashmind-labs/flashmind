//! Postgres query tool.
//!
//! Allows agents to query a PostgreSQL database via `tokio-postgres`.
//! Connections are opened per query and optionally forced read-only via
//! `SET default_transaction_read_only = on`.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::db_common;
use crate::utils::truncate_utf8;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Postgres integration.
pub struct PostgresConfig {
    /// A `postgresql://user:pass@host:port/dbname` connection string.
    pub connection_string: String,
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PostgresQueryArgs {
    query: String,
    limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to extract a human-readable string from a single cell.
fn cell_to_string(row: &tokio_postgres::Row, idx: usize) -> String {
    // Try common types in order of likelihood.
    if let Ok(v) = row.try_get::<_, Option<String>>(idx) {
        return v.map_or("NULL".to_string(), |s| {
            if s.len() > 200 {
                format!("{}...", truncate_utf8(&s, 200))
            } else {
                s
            }
        });
    }
    if let Ok(v) = row.try_get::<_, Option<i64>>(idx) {
        return v.map_or("NULL".to_string(), |n| n.to_string());
    }
    if let Ok(v) = row.try_get::<_, Option<i32>>(idx) {
        return v.map_or("NULL".to_string(), |n| n.to_string());
    }
    if let Ok(v) = row.try_get::<_, Option<f64>>(idx) {
        return v.map_or("NULL".to_string(), |n| format!("{n:.4}"));
    }
    if let Ok(v) = row.try_get::<_, Option<f32>>(idx) {
        return v.map_or("NULL".to_string(), |n| format!("{n:.4}"));
    }
    if let Ok(v) = row.try_get::<_, Option<bool>>(idx) {
        return v.map_or("NULL".to_string(), |b| b.to_string());
    }
    "?".to_string()
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Execute SQL queries against a PostgreSQL database.
pub struct PostgresQueryTool {
    /// Connection string (`postgresql://…`).
    pub connection_string: String,
    /// When `true`, write queries are rejected and the session is forced
    /// read-only as defense-in-depth.
    pub readonly: bool,
}

#[async_trait]
impl Tool for PostgresQueryTool {
    fn name(&self) -> &str {
        "postgres_query"
    }

    fn description(&self) -> &str {
        "Execute a SQL query against a PostgreSQL database and return results \
         as a formatted table. Useful for inspecting data, running analytical \
         queries, or exploring schemas with queries like \
         'SELECT tablename FROM pg_tables WHERE schemaname = \\'public\\''."
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

        let args: PostgresQueryArgs = ctx.parse_args(self.name())?;
        let limit = db_common::clamp_limit(args.limit);
        let query = args.query.trim().to_string();

        if self.readonly && db_common::is_write_query(&query) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Write operations not allowed. Only SELECT queries.",
            ));
        }

        // Determine TLS mode from the connection string.
        let use_tls = !self.connection_string.contains("sslmode=disable");

        let conn_string = self.connection_string.clone();
        let readonly = self.readonly;

        let result = if use_tls {
            Self::run_query_tls(conn_string, query, limit, readonly).await
        } else {
            Self::run_query_notls(conn_string, query, limit, readonly).await
        };

        match result {
            Ok(output) => Ok(ToolResult::success(ctx.tool_call_id, output)),
            Err(err) => Ok(ToolResult::failure(ctx.tool_call_id, err.to_string())),
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("(unknown)");
        let preview: String = query.chars().take(80).collect();
        format!("Querying Postgres: {preview}")
    }
}

impl PostgresQueryTool {
    /// Connect with RusTLS and execute the query.
    async fn run_query_tls(
        conn_string: String,
        query: String,
        limit: usize,
        readonly: bool,
    ) -> anyhow::Result<String> {
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let tls = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config);

        let (client, connection) = tokio_postgres::connect(&conn_string, tls).await?;

        // tokio-postgres requires the connection future to be driven.
        let conn_handle = tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "postgres connection task ended with error");
            }
        });

        let result = Self::execute_on_client(&client, &query, limit, readonly).await;
        drop(client);
        let _ = conn_handle.await;
        result
    }

    /// Connect without TLS and execute the query.
    async fn run_query_notls(
        conn_string: String,
        query: String,
        limit: usize,
        readonly: bool,
    ) -> anyhow::Result<String> {
        let (client, connection) =
            tokio_postgres::connect(&conn_string, tokio_postgres::NoTls).await?;

        let conn_handle = tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "postgres connection task ended with error");
            }
        });

        let result = Self::execute_on_client(&client, &query, limit, readonly).await;
        drop(client);
        let _ = conn_handle.await;
        result
    }

    /// Run the actual query on an already-connected client.
    async fn execute_on_client(
        client: &tokio_postgres::Client,
        query: &str,
        limit: usize,
        readonly: bool,
    ) -> anyhow::Result<String> {
        if readonly {
            client
                .execute("SET default_transaction_read_only = on", &[])
                .await?;
        }

        let rows = client.query(query, &[]).await?;

        // Extract column names from the first row or the statement metadata.
        let columns: Vec<String> = if rows.is_empty() {
            Vec::new()
        } else {
            rows[0]
                .columns()
                .iter()
                .map(|c| c.name().to_string())
                .collect()
        };

        if columns.is_empty() {
            return Ok("(no results)".to_string());
        }

        let col_count = columns.len();
        let string_rows: Vec<Vec<String>> = rows
            .iter()
            .take(limit)
            .map(|row| (0..col_count).map(|i| cell_to_string(row, i)).collect())
            .collect();

        Ok(db_common::format_table(&columns, &string_rows, 60))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn readonly_blocks_write_queries() {
        let tool = PostgresQueryTool {
            connection_string: "postgresql://localhost/test".to_string(),
            readonly: true,
        };

        let cancel = CancellationToken::new();

        for stmt in &[
            "INSERT INTO t VALUES (1)",
            "  DELETE FROM t",
            "DROP TABLE t",
            "UPDATE t SET x = 1",
            "create table t (id int)",
        ] {
            let ctx = ToolContext::new("id", json!({"query": stmt}), None, &cancel);
            let result = tool.execute(ctx).await.unwrap();
            assert!(
                !result.is_success(),
                "expected failure for write query: {stmt}",
            );
            assert!(
                result.output().contains("Write operations not allowed"),
                "unexpected message for '{stmt}': {}",
                result.output(),
            );
        }
    }

    #[tokio::test]
    async fn readonly_allows_select() {
        // This test doesn't actually connect — it will fail at the connection
        // stage, but importantly it should NOT fail at the write-check stage.
        let tool = PostgresQueryTool {
            connection_string: "postgresql://localhost:1/nonexistent?sslmode=disable".to_string(),
            readonly: true,
        };

        let cancel = CancellationToken::new();
        let ctx = ToolContext::new("id", json!({"query": "SELECT 1"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        // It should fail with a connection error, not a "write not allowed" error.
        assert!(
            !result.output().contains("Write operations not allowed"),
            "SELECT should not be blocked: {}",
            result.output(),
        );
    }

    #[tokio::test]
    async fn cancelled_before_execute() {
        let tool = PostgresQueryTool {
            connection_string: "postgresql://localhost/test".to_string(),
            readonly: false,
        };

        let cancel = CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new("id", json!({"query": "SELECT 1"}), None, &cancel);

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().to_lowercase().contains("cancel"));
    }

    #[test]
    fn humanize_truncates_long_queries() {
        let tool = PostgresQueryTool {
            connection_string: String::new(),
            readonly: false,
        };

        let long_query = "SELECT ".to_string() + &"x".repeat(200);
        let desc = tool.humanize(&json!({"query": long_query}));
        assert!(desc.len() < 200, "humanize should truncate: {desc}");
    }
}
