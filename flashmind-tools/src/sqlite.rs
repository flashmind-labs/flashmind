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

use crate::utils::truncate_utf8;

#[derive(Deserialize)]
struct SqliteQueryArgs {
    path: String,
    query: String,
    limit: Option<usize>,
}

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
        let args: SqliteQueryArgs = ctx.parse_args(self.name())?;
        let limit = args.limit.unwrap_or(50).min(200);

        let db_path = PathBuf::from(&args.path);
        if !db_path.exists() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Database not found: {}", args.path),
            ));
        }

        let query = args.query.trim().to_string();
        let query_upper = query.to_uppercase();
        let forbidden = [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH", "VACUUM",
            "REINDEX", "REPLACE",
        ];
        for keyword in &forbidden {
            if query_upper.starts_with(keyword) {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Write operations not allowed. Only SELECT queries.",
                ));
            }
        }

        let result = tokio::task::spawn_blocking(move || -> std::result::Result<String, String> {
            let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;

            let conn = rusqlite::Connection::open_with_flags(&db_path, flags)
                .map_err(|e| format!("Failed to open database: {e}"))?;

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
                .filter_map(|r| r.ok())
                .take(limit)
                .collect();

            if rows.is_empty() {
                return Ok("(no results)".to_string());
            }

            let mut widths: Vec<usize> = col_names.iter().map(|n| n.len()).collect();
            for row in &rows {
                for (i, val) in row.iter().enumerate() {
                    widths[i] = widths[i].max(val.len()).min(60);
                }
            }

            let mut out = String::new();

            let header: Vec<String> = col_names
                .iter()
                .zip(&widths)
                .map(|(n, w)| format!("{:<width$}", n, width = w))
                .collect();
            out.push_str(&header.join(" | "));
            out.push('\n');
            out.push_str(
                &widths
                    .iter()
                    .map(|w| "-".repeat(*w))
                    .collect::<Vec<_>>()
                    .join("-+-"),
            );
            out.push('\n');

            for row in &rows {
                let formatted: Vec<String> = row
                    .iter()
                    .zip(&widths)
                    .map(|(v, w)| {
                        if v.chars().count() > *w {
                            format!("{}...", truncate_utf8(v, w.saturating_sub(3)))
                        } else {
                            format!("{:<width$}", v, width = w)
                        }
                    })
                    .collect();
                out.push_str(&formatted.join(" | "));
                out.push('\n');
            }

            out.push_str(&format!("\n({} rows)", rows.len()));
            Ok(out)
        })
        .await
        .map_err(|e| anyhow::anyhow!("sqlite_query: {e}"))?;

        match result {
            Ok(output) => Ok(ToolResult::success(ctx.tool_call_id, output)),
            Err(err) => Ok(ToolResult::failure(ctx.tool_call_id, err)),
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Querying database: {}", path)
    }
}
