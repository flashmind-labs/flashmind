//! Regex-based string replacement tool with capture group support.
//!
//! Provides `StrReplaceRegexTool` that finds and replaces text matching a regex
//! pattern in files. Supports capture groups ($1, $2) in the replacement string.

use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::{debug, warn};

use crate::file_cache::FileCache;
use crate::file_ops::resolve_path;
use crate::protected::ProtectedPaths;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

#[derive(Deserialize)]
struct StrReplaceRegexArgs {
    path: String,
    pattern: String,
    replacement: String,
    count: Option<usize>,
}

pub struct StrReplaceRegexTool {
    pub protected: Arc<ProtectedPaths>,
    pub file_cache: FileCache,
}

#[async_trait]
impl Tool for StrReplaceRegexTool {
    fn name(&self) -> &str {
        "str_replace_regex"
    }

    fn description(&self) -> &str {
        "Replace all text matching a regex pattern in a file. Replaces EVERY match (not just the first). Supports capture groups ($1, $2) in replacement. Fails if no matches found. For exact string replacement, use str_replace instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement string; use $1, $2 etc. for capture groups"
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum number of replacements to perform (default: all matches)"
                }
            },
            "required": ["path", "pattern", "replacement"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StrReplaceRegexArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let resolved_path = resolve_path(&args.path, ctx.working_dir);

        debug!(path = %resolved_path.display(), pattern = %args.pattern, "str_replace_regex requested");

        // Validate regex at parse time
        let re = match Regex::new(&args.pattern) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid regex pattern: {}", e),
                ));
            }
        };

        // Check protected paths
        if self.protected.is_write_protected(&resolved_path) {
            warn!(path = %resolved_path.display(), "str_replace_regex blocked: write-protected path");
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Error: Cannot modify write-protected file: {}", args.path),
            ));
        }

        // Read existing content
        let content = match tokio::fs::read_to_string(&resolved_path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error reading file: {}", e),
                ));
            }
        };

        self.file_cache.store(&resolved_path, content.clone());

        // Count total matches
        let total_matches = re.find_iter(&content).count();

        if total_matches == 0 {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Pattern not found in file.\n\nPattern: {}\n\nFile preview (first 500 chars):\n{}",
                    args.pattern,
                    content.chars().take(500).collect::<String>()
                ),
            ));
        }

        // Perform replacements
        let new_content = match args.count {
            None => re
                .replace_all(&content, args.replacement.as_str())
                .into_owned(),
            Some(limit) => replace_n(&re, &content, &args.replacement, limit),
        };

        let replacements_made = match args.count {
            None => total_matches,
            Some(limit) => total_matches.min(limit),
        };

        match tokio::fs::write(&resolved_path, &new_content).await {
            Ok(()) => {
                debug!(path = %resolved_path.display(), replacements = replacements_made, "str_replace_regex success");
                let diffs = self
                    .file_cache
                    .diff_vec(&resolved_path, &new_content, ctx.working_dir);
                self.file_cache.store(&resolved_path, new_content);
                Ok(ToolResult::success_with_diffs(
                    ctx.tool_call_id,
                    format!("OK ({} replacements)", replacements_made),
                    diffs,
                ))
            }
            Err(e) => {
                warn!(path = %resolved_path.display(), error = %e, "str_replace_regex failed");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error writing file: {}", e),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Editing {} (regex)", path)
    }
}

/// Replace up to `limit` non-overlapping matches, expanding capture groups.
fn replace_n(re: &Regex, text: &str, replacement: &str, limit: usize) -> String {
    if limit == 0 {
        return text.to_owned();
    }

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (count, mat) in re.find_iter(text).enumerate() {
        if count >= limit {
            break;
        }

        result.push_str(&text[last_end..mat.start()]);

        // Expand capture groups using caps.expand()
        if let Some(caps) = re.captures(&text[mat.start()..]) {
            caps.expand(replacement, &mut result);
        } else {
            result.push_str(replacement);
        }

        last_end = mat.end();
    }

    result.push_str(&text[last_end..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_tool(dir: &std::path::Path) -> StrReplaceRegexTool {
        StrReplaceRegexTool {
            protected: Arc::new(ProtectedPaths::new(dir)),
            file_cache: FileCache::new(),
        }
    }

    #[tokio::test]
    async fn test_basic_regex_replace_all() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "foo",
            "replacement": "qux"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "qux bar qux baz qux"
        );
        assert!(result.output.contains("3 replacements"));
    }

    #[tokio::test]
    async fn test_replace_with_count_limit() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "aaa aaa aaa").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "aaa",
            "replacement": "bbb",
            "count": 2
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "bbb bbb aaa");
        assert!(result.output.contains("2 replacements"));
    }

    #[tokio::test]
    async fn test_capture_groups() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "2024-01-15 and 2025-03-20").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": r"(\d{4})-(\d{2})-(\d{2})",
            "replacement": "$3/$2/$1"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "15/01/2024 and 20/03/2025"
        );
    }

    #[tokio::test]
    async fn test_capture_groups_with_count() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world hello world").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": r"(hello) (world)",
            "replacement": "$2 $1",
            "count": 1
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "world hello hello world"
        );
    }

    #[tokio::test]
    async fn test_pattern_not_found() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": r"\d+",
            "replacement": "NUM"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Pattern not found"));
        assert!(result.output.contains("hello world"));
    }

    #[tokio::test]
    async fn test_invalid_regex() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "[invalid",
            "replacement": "x"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Invalid regex pattern"));
    }

    #[tokio::test]
    async fn test_protected_path() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config.toml");
        std::fs::write(&file_path, "key = \"value\"").unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "value",
            "replacement": "new_value"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("protected"));
        // File must be unchanged
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "key = \"value\""
        );
    }

    #[tokio::test]
    async fn test_multiline_content() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(
            &file_path,
            "line 1: foo\nline 2: bar\nline 3: foo\nline 4: baz\n",
        )
        .unwrap();

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "foo",
            "replacement": "FOO"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "line 1: FOO\nline 2: bar\nline 3: FOO\nline 4: baz\n"
        );
        assert!(result.output.contains("2 replacements"));
    }

    #[tokio::test]
    async fn test_count_exceeds_matches() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "x y x").unwrap();

        let tool = make_tool(dir.path());
        // count=10 but only 2 matches — should replace all 2
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "x",
            "replacement": "z",
            "count": 10
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success, "expected success, got: {}", result.output);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "z y z");
        assert!(result.output.contains("2 replacements"));
    }

    #[tokio::test]
    async fn test_file_not_found() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nonexistent.txt");

        let tool = make_tool(dir.path());
        let args = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "pattern": "foo",
            "replacement": "bar"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Error reading file"));
    }
}
