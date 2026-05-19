//! Read-only SQLite query tool.
//!
//! Allows agents to query any SQLite database on disk. All queries
//! run in read-only mode — no writes, no schema changes.

use async_trait::async_trait;
use rusqlite::types::ValueRef;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::db_common;
use crate::utils::truncate_utf8;

#[derive(Deserialize)]
struct SqliteQueryArgs {
    path: String,
    query: String,
    limit: Option<usize>,
}

/// Read-only SQLite query tool.
pub struct SqliteQueryTool;

#[async_trait]
impl Tool for SqliteQueryTool {
    fn name(&self) -> &str {
        "sqlite_query"
    }

    fn description(&self) -> &str {
        "Execute a read-only SQL query against a SQLite database file. \
         Returns results as a formatted table. Use for inspecting databases, \
         running SELECT queries, or exploring schemas with '.tables' or \
         'SELECT * FROM sqlite_master'."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the SQLite database file."
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to execute (SELECT only). Use 'SELECT * FROM sqlite_master WHERE type=\"table\"' to list tables."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 50, max: 200)."
                }
            },
            "required": ["path", "query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Query cancelled"));
        }

        let args: SqliteQueryArgs = ctx.parse_args(self.name())?;
        let limit = db_common::clamp_limit(args.limit);

        let db_path = PathBuf::from(&args.path);
        if !db_path.exists() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Database not found: {}", args.path),
            ));
        }

        let query = args.query.trim().to_string();
        if db_common::is_write_query(&query) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Write operations not allowed. Only SELECT queries.",
            ));
        }

        // Wire up cancellation: pass the connection's interrupt handle out of
        // the blocking task so a watchdog on the async runtime can abort a
        // long-running statement when cancel_token fires. We always await the
        // blocking thread to completion (after firing interrupt, SQLite
        // returns SQLITE_INTERRUPT promptly) so the OS thread doesn't outlive
        // this function.
        let (handle_tx, handle_rx) = tokio::sync::oneshot::channel::<rusqlite::InterruptHandle>();
        let watchdog_cancel = ctx.cancel_token().clone();
        let watchdog = tokio::spawn(async move {
            if let Ok(interrupt_handle) = handle_rx.await {
                watchdog_cancel.cancelled().await;
                interrupt_handle.interrupt();
            }
        });

        let query_handle =
            tokio::task::spawn_blocking(move || -> std::result::Result<String, String> {
                let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;

                let conn = rusqlite::Connection::open_with_flags(&db_path, flags)
                    .map_err(|e| format!("Failed to open database: {e}"))?;

                // Hand the interrupt handle to the watchdog. Ignore send errors —
                // the receiver only goes away if the tool was already cancelled.
                let _ = handle_tx.send(conn.get_interrupt_handle());

                let mut stmt = conn
                    .prepare(&query)
                    .map_err(|e| format!("Query error: {e}"))?;

                let col_count = stmt.column_count();
                let col_names: Vec<String> = (0..col_count)
                    .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
                    .collect();

                let rows: Vec<Vec<String>> = stmt
                    .query_map([], |row| {
                        let mut values = Vec::with_capacity(col_count);
                        for i in 0..col_count {
                            let val: String = match row.get_ref(i) {
                                Ok(ValueRef::Null) => "NULL".to_string(),
                                Ok(ValueRef::Integer(n)) => n.to_string(),
                                Ok(ValueRef::Real(f)) => format!("{f:.4}"),
                                Ok(ValueRef::Text(t)) => {
                                    let s = String::from_utf8_lossy(t);
                                    if s.len() > 200 {
                                        format!("{}...", truncate_utf8(&s, 200))
                                    } else {
                                        s.to_string()
                                    }
                                }
                                Ok(ValueRef::Blob(b)) => {
                                    format!("<blob {}B>", b.len())
                                }
                                Err(_) => "?".to_string(),
                            };
                            values.push(val);
                        }
                        Ok(values)
                    })
                    .map_err(|e| format!("Query error: {e}"))?
                    .take(limit)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| format!("Query error: {e}"))?;

                Ok(db_common::format_table(&col_names, &rows, 60))
            });

        // Always await the blocking thread. If cancellation fires, the
        // watchdog interrupts the connection, causing SQLite to return
        // SQLITE_INTERRUPT within its next progress check — typically
        // milliseconds. Awaiting (rather than racing-and-returning) ensures
        // the OS thread doesn't outlive this function.
        let joined = query_handle
            .await
            .map_err(|e| anyhow::anyhow!("sqlite_query: {e}"))?;
        watchdog.abort();
        let was_cancelled = ctx.cancel_token().is_cancelled();

        match joined {
            Ok(output) => Ok(ToolResult::success(ctx.tool_call_id, output)),
            Err(err) => {
                // Normalize the SQLITE_INTERRUPT error to a clean cancellation
                // message when we know we triggered it.
                if was_cancelled || err.contains("interrupted") || err.contains("Interrupt") {
                    Ok(ToolResult::failure(ctx.tool_call_id, "Query cancelled"))
                } else {
                    Ok(ToolResult::failure(ctx.tool_call_id, err))
                }
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Querying database: {}", path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    fn make_test_db(path: &std::path::Path) {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
            .unwrap();
    }

    #[tokio::test]
    async fn test_sqlite_basic_select() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        make_test_db(&db);

        let tool = SqliteQueryTool;
        let cancel = CancellationToken::new();
        let ctx = ToolContext::new(
            "id",
            json!({"path": db.to_str().unwrap(), "query": "SELECT * FROM t"}),
            None,
            &cancel,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.is_success(), "got: {}", result.output());
        assert!(result.output().contains("id"));
    }

    #[tokio::test]
    async fn test_sqlite_returns_failure_when_pre_cancelled() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        make_test_db(&db);

        let tool = SqliteQueryTool;
        let cancel = CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new(
            "id",
            json!({"path": db.to_str().unwrap(), "query": "SELECT * FROM t"}),
            None,
            &cancel,
        );

        // Should return promptly with cancellation failure even though the
        // spawn_blocking task may still be opening the connection.
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), tool.execute(ctx))
            .await
            .expect("sqlite_query did not return within 2s of cancellation")
            .unwrap();

        assert!(!result.is_success(), "expected failure on cancel");
        assert!(
            result.output().to_lowercase().contains("cancel"),
            "expected cancellation message, got: {}",
            result.output()
        );
    }

    #[tokio::test]
    async fn test_sqlite_cancel_interrupts_slow_query() {
        // SELECT count(*) over a recursive CTE must visit every row before
        // emitting its single result, so the user-side LIMIT can't cut it
        // short. Without interrupt, this runs for many seconds; with the
        // watchdog wired up, cancellation surfaces in under a second.
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        make_test_db(&db);

        let tool = SqliteQueryTool;
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let path_str = db.to_str().unwrap().to_string();

        let exec = async move {
            let ctx = ToolContext::new(
                "id",
                json!({
                    "path": path_str,
                    "query": "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 1000000000) SELECT count(*) FROM r"
                }),
                None,
                &cancel,
            );
            tool.execute(ctx).await
        };

        // Give the query ~100ms to start, then cancel.
        let canceller = async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            cancel_clone.cancel();
        };

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (result, _) = tokio::join!(exec, canceller);
            result
        })
        .await
        .expect("sqlite did not respond to cancellation within 5s")
        .unwrap();

        assert!(
            !result.is_success(),
            "expected failure, got: {}",
            result.output()
        );
        assert!(
            result.output().to_lowercase().contains("cancel"),
            "expected cancellation message, got: {}",
            result.output()
        );
    }
}
