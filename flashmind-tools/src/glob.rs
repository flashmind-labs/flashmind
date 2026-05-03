//! Glob pattern matching for file discovery.
//!
//! Provides `GlobTool` that finds files matching glob patterns like `**/*.rs`.

use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

use crate::protected::ProtectedPaths;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

#[derive(Deserialize)]
struct GlobArgs {
    pattern: String,
}

/// Pattern-based file discovery tool.
pub struct GlobTool {
    pub protected: Arc<ProtectedPaths>,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files by name pattern. Use ** for recursive matching (e.g. src/**/*.rs). Returns matching file paths. Use grep to search file contents instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., **/*.rs, src/**/*.toml). Relative to working directory."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();
        let args: GlobArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.pattern) {
            return Ok(r);
        }

        // Use working directory as base for relative patterns, or current dir if no workspace
        let base: PathBuf = if let Some(wd) = ctx.working_dir {
            wd.clone()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };

        tracing::debug!(pattern = %args.pattern, "glob requested");

        // Build glob pattern from base + pattern
        let full_pattern = if args.pattern.starts_with('/') {
            args.pattern.clone()
        } else {
            format!("{}/{}", base.display(), args.pattern).replace("//", "/")
        };

        // Run glob in spawn_blocking so the outer select! can cancel around it.
        let base_clone = base.clone();
        let pattern_clone = full_pattern.clone();
        let matches = match tokio::task::spawn_blocking(move || {
            let mut results: Vec<String> = Vec::new();
            let entries = glob::glob(&pattern_clone).map_err(|e| e.to_string())?;

            for entry in entries.flatten() {
                if entry.is_file() {
                    let rel_path = entry
                        .strip_prefix(&base_clone)
                        .unwrap_or(&entry)
                        .to_string_lossy()
                        .to_string();
                    results.push(rel_path);
                }
            }

            results.sort();
            Ok::<_, String>(results)
        })
        .await
        {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid glob pattern: {}", e),
                ));
            }
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Glob task failed: {}", e),
                ));
            }
        };

        let result = if matches.is_empty() {
            format!("No files found matching pattern: {}", args.pattern)
        } else {
            let count = matches.len();
            let preview: String = matches
                .iter()
                .take(20)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if count > 20 {
                format!(
                    "Found {} files:\n{}\n...and {} more",
                    count,
                    preview,
                    count - 20
                )
            } else {
                format!("Found {} files:\n{}", count, preview)
            }
        };

        metrics::counter!("tools.glob.calls").increment(1);
        metrics::histogram!("tools.glob.duration_seconds").record(start.elapsed().as_secs_f64());
        metrics::histogram!("tools.glob.matches_found").record(matches.len() as f64);
        Ok(ToolResult::success(ctx.tool_call_id, result))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Finding files matching {}", pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_glob_basic() {
        let dir = tempdir().unwrap();

        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let tool = GlobTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
        };

        // Create a custom context with working_dir set
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();
        let ctx = ToolContext::new(
            "test-id",
            json!({ "pattern": "**/*.rs" }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("main.rs"));
        assert!(result.output.contains("lib.rs"));
    }
}
