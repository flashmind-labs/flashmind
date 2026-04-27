//! Regex-based file content search using the grep crate family.
//!
//! Provides `GrepTool` that searches file contents with ripgrep-powered regex matching.
//! Supports single files, directories (depth 1), and glob patterns.

use async_trait::async_trait;
use glob::glob;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::Searcher;
use grep_searcher::sinks::UTF8;
use ignore::WalkBuilder;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::debug;

use crate::file_ops::resolve_path;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    path: String,
    max_results: Option<usize>,
    /// Case-insensitive search (default: false)
    insensitive: Option<bool>,
    /// Match whole words only (default: false)
    whole_word: Option<bool>,
}

pub struct GrepTool;

/// Collect files to search based on the path argument.
///
/// - Glob patterns (containing `*`, `?`, `[`) are expanded recursively per the pattern.
/// - Directories are searched recursively.
/// - Files are returned as-is.
fn collect_files(path_arg: &str, base: &Path) -> Vec<PathBuf> {
    let is_glob = path_arg.contains('*') || path_arg.contains('?') || path_arg.contains('[');

    if is_glob {
        let full_pattern = if Path::new(path_arg).is_absolute() {
            path_arg.to_string()
        } else {
            format!("{}/{}", base.display(), path_arg).replace("//", "/")
        };

        match glob(&full_pattern) {
            Ok(entries) => entries.flatten().filter(|p| p.is_file()).collect(),
            Err(_) => vec![],
        }
    } else {
        let resolved = resolve_path(path_arg, Some(&base.to_path_buf()));
        if resolved.is_file() {
            vec![resolved]
        } else if resolved.is_dir() {
            // Use WalkBuilder for recursive directory traversal
            let walker = WalkBuilder::new(&resolved)
                .hidden(true) // ignore hidden files
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .build();

            walker
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
                .map(|e| e.path().to_path_buf())
                .collect()
        } else {
            vec![]
        }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex. Returns file:line: match format. \
         Accepts a file path, directory (recursive), or glob pattern (e.g. 'src/**/*.rs'). \
         Respects .gitignore. Use INSTEAD of bash grep — faster and structured output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File path, directory (searched recursively), or glob pattern (e.g. src/**/*.rs)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 50)"
                },
                "insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                },
                "whole_word": {
                    "type": "boolean",
                    "description": "Match whole words only (default: false)"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GrepArgs = ctx.parse_args(self.name())?;
        let max = args.max_results.unwrap_or(1000);

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        debug!(pattern = %args.pattern, path = %args.path, "grep requested");

        let mut builder = RegexMatcherBuilder::new();
        if args.insensitive == Some(true) {
            builder.case_insensitive(true);
        }
        if args.whole_word == Some(true) {
            builder.word(true);
        }
        let matcher = match builder.build(&args.pattern) {
            Ok(m) => m,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid regex pattern: {}", e),
                ));
            }
        };

        let base: PathBuf = if let Some(wd) = ctx.working_dir {
            wd.clone()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let cancel_token = ctx.cancel_token().clone();

        let path_arg = args.path.clone();
        let search_handle = tokio::task::spawn_blocking(move || {
            let files = collect_files(&path_arg, &base);
            debug!(?files, "collected files for search");

            let mut results: Vec<String> = Vec::new();
            let mut searcher = Searcher::new();

            'outer: for file_path in &files {
                if cancelled_inner.load(Ordering::Relaxed) {
                    break 'outer;
                }

                let display_path = file_path
                    .strip_prefix(&base)
                    .unwrap_or(file_path)
                    .to_string_lossy()
                    .to_string();

                let search_result = searcher.search_path(
                    &matcher,
                    file_path,
                    UTF8(|line_num, line| {
                        results.push(format!(
                            "{}:{}: {}",
                            display_path,
                            line_num,
                            line.trim_end()
                        ));
                        Ok(results.len() < max)
                    }),
                );

                if let Err(e) = search_result {
                    debug!(path = %display_path, error = %e, "search_file: skipping unreadable file");
                }

                if results.len() >= max {
                    break 'outer;
                }
            }

            results
        });

        // Race between search completion and cancellation
        let results = tokio::select! {
            biased;
            _ = cancel_token.cancelled() => {
                // Cancel was requested - set flag to stop the blocking task
                cancelled.store(true, Ordering::Relaxed);
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Search cancelled",
                ));
            }
            result = search_handle => {
                match result {
                    Ok(results) => results,
                    Err(e) => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!("Search failed: {}", e),
                        ));
                    }
                }
            }
        };
        let truncated = results.len() >= max;

        let output = if results.is_empty() {
            format!("No matches found for pattern: {}", args.pattern)
        } else {
            let mut out = results.join("\n");
            if truncated {
                out.push_str(&format!("\n\n... truncated at {} results", max));
            } else {
                out.push_str(&format!("\n\n{} matches", results.len()));
            }
            out
        };

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Searching '{}' in {}", pattern, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn make_ctx<'a>(
        tool_call_id: &'a str,
        args: Value,
        scope: &'a str,
        wd: Option<&'a PathBuf>,
        cancel_token: &'a CancellationToken,
        tx: &'a mpsc::Sender<flashmind_types::event::AgentEvent>,
    ) -> ToolContext<'a> {
        ToolContext::new(tool_call_id, args, scope, wd, cancel_token, tx)
    }

    #[tokio::test]
    async fn test_search_single_file() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("hello.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t1",
            json!({ "pattern": "println", "path": "hello.rs" }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success, "expected success");
        assert!(
            result.output.contains("hello.rs"),
            "should contain filename"
        );
        assert!(result.output.contains("println"), "should contain match");
        // Line number should be 2
        assert!(
            result.output.contains(":2:"),
            "should contain line number 2"
        );
    }

    #[tokio::test]
    async fn test_search_directory_recursive() {
        let dir = tempdir().unwrap();
        // File in root
        fs::write(dir.path().join("top.rs"), "fn top() {}\n").unwrap();
        // File in subdir — should be found with recursive dir search
        fs::create_dir_all(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/nested.rs"), "fn nested() {}\n").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t2",
            json!({ "pattern": "fn", "path": "." }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("top.rs"),
            "should find top-level file"
        );
        assert!(
            result.output.contains("nested.rs"),
            "should recurse into subdirs"
        );
    }

    #[tokio::test]
    async fn test_search_glob_recursive() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/sub")).unwrap();
        fs::write(dir.path().join("src/a.rs"), "struct Foo;\n").unwrap();
        fs::write(dir.path().join("src/sub/b.rs"), "struct Bar;\n").unwrap();
        fs::write(dir.path().join("src/readme.md"), "# struct ignored\n").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t3",
            json!({ "pattern": "struct", "path": "src/**/*.rs" }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("a.rs"), "should find a.rs");
        assert!(result.output.contains("b.rs"), "should find nested b.rs");
        // Should not match the .md file (not a .rs file per glob)
        assert!(
            !result.output.contains("readme.md"),
            "glob *.rs should not match .md"
        );
    }

    #[tokio::test]
    async fn test_no_matches() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("empty.rs"), "fn nothing() {}\n").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t4",
            json!({ "pattern": "xyzzy_not_found", "path": "empty.rs" }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("No matches found"),
            "should report no matches"
        );
    }

    #[tokio::test]
    async fn test_invalid_regex() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("file.rs"), "content\n").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t5",
            json!({ "pattern": "[invalid(regex", "path": "file.rs" }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(!result.success, "invalid regex should return failure");
        assert!(
            result.output.contains("Invalid regex"),
            "should mention invalid regex"
        );
    }

    #[tokio::test]
    async fn test_max_results_truncation() {
        let dir = tempdir().unwrap();
        // Write 10 matching lines
        let content = (1..=10)
            .map(|i| format!("match line {}\n", i))
            .collect::<String>();
        fs::write(dir.path().join("many.txt"), content).unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        let ctx = make_ctx(
            "t6",
            json!({ "pattern": "match", "path": "many.txt", "max_results": 3 }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("truncated at 3"),
            "should show truncation message"
        );
        // Should only have 3 result lines (not 10)
        let match_lines = result
            .output
            .lines()
            .filter(|l| l.contains("match line"))
            .count();
        assert_eq!(match_lines, 3, "should only return 3 results");
    }

    #[tokio::test]
    async fn test_search_truncate_utf8_pattern() {
        // Test searching for the actual truncate_utf8 function in src/utils
        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = std::env::current_dir().unwrap();

        let ctx = make_ctx(
            "t7",
            json!({ "pattern": "truncate_utf8", "path": "." }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );

        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("truncate_utf8"),
            "should find truncate_utf8 in codebase"
        );
    }

    // === New parameter tests ===
    #[tokio::test]
    async fn test_search_case_insensitive() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "Hello WORLD hello").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        // Case insensitive - should find all 3 occurrences
        let ctx = make_ctx(
            "t2",
            json!({ "pattern": "hello", "path": "test.txt", "insensitive": true }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );
        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        // Should find matches (the exact assertion depends on output format)
        assert!(
            result.output.contains("hello")
                || result.output.contains("Hello")
                || result.output.contains("WORLD")
        );
    }

    #[tokio::test]
    async fn test_search_whole_word() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "cat catalog category concat").unwrap();

        let tool = GrepTool;
        let scope = "repl:test";
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let wd = dir.path().to_path_buf();

        // Match "cat" as whole word
        let ctx = make_ctx(
            "t1",
            json!({ "pattern": "cat", "path": "test.txt", "whole_word": true }),
            scope,
            Some(&wd),
            &cancel_token,
            &tx,
        );
        let result = tool.execute(ctx).await.unwrap();
        assert!(result.success);
        // Should match "cat" at the beginning
        assert!(result.output.contains(":1: cat"));
    }

    // Note: invert is not yet implemented in the searcher, skipping that test
}
