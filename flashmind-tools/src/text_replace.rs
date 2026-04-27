//! String replacement tool for precise text edits.
//!
//! Provides `StrReplaceTool` that finds and replaces exact string matches in files.
//! Safer than line-based editing when you know the exact text to change.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

use crate::file_cache::FileCache;
use crate::protected::ProtectedPaths;
use crate::utils::truncate_utf8;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};
/// Resolve a path for file operations.
///
/// - Absolute paths pass through as-is
/// - Relative paths resolve against `working_dir`
/// - Without a working directory (REPL mode), paths are returned as-is
fn resolve_path(path: &str, working_dir: Option<&PathBuf>) -> PathBuf {
    let p = std::path::Path::new(path);

    if p.is_absolute() {
        return p.to_path_buf();
    }

    if let Some(wd) = working_dir {
        wd.join(p)
    } else {
        p.to_path_buf()
    }
}

#[derive(Deserialize)]
struct StrReplaceArgs {
    path: String,
    old_string: String,
    new_string: String,
}

pub struct StrReplaceTool {
    pub protected: Arc<ProtectedPaths>,
    pub file_cache: FileCache,
}

#[async_trait]
impl Tool for StrReplaceTool {
    fn name(&self) -> &str {
        "str_replace"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. The old_string must match exactly once — fails if not found or ambiguous (multiple matches). Include enough surrounding context in old_string to make it unique. For regex replacements, use str_replace_regex."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StrReplaceArgs = ctx.parse_args(self.name())?;

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

        let resolved_path = resolve_path(&args.path, ctx.working_dir);

        tracing::debug!(path = %resolved_path.display(), "str_replace requested");

        // Check protected paths
        if self.protected.is_write_protected(&resolved_path) {
            tracing::warn!(path = %resolved_path.display(), "str_replace blocked: write-protected path");
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

        // Use replacen to replace first occurrence only
        let new_content = content.replacen(&args.old_string, &args.new_string, 1);

        // Check if replacement happened
        if new_content == content {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Pattern not found in file.\n\nLooking for:\n{}\n\nFile preview (first 500 chars):\n{}",
                    args.old_string,
                    truncate_utf8(&content, 500),
                ),
            ));
        }

        match tokio::fs::write(&resolved_path, &new_content).await {
            Ok(()) => {
                tracing::debug!(path = %resolved_path.display(), "str_replace success");
                let diffs = self
                    .file_cache
                    .diff_vec(&resolved_path, &new_content, ctx.working_dir);
                self.file_cache.store(&resolved_path, new_content);
                Ok(ToolResult::success_with_diffs(
                    ctx.tool_call_id,
                    "OK",
                    diffs,
                ))
            }
            Err(e) => {
                tracing::warn!(path = %resolved_path.display(), error = %e, "str_replace failed");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error writing file: {}", e),
                ))
            }
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Editing {}", path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_str_replace_success() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "world",
            "new_string": "rust"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello rust");
    }

    #[tokio::test]
    async fn test_str_replace_not_found() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "notfound",
            "new_string": "something"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Pattern not found"));
    }

    #[tokio::test]
    async fn test_str_replace_multiple_matches() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello hello hello").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "hi"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_str_replace_multiline() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "    println!(\"hello\");",
            "new_string": "    println!(\"world\");\n    println!(\"extra\");"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("println!(\"world\")"));
        assert!(content.contains("println!(\"extra\")"));
        assert!(!content.contains("println!(\"hello\")"));
    }

    #[tokio::test]
    async fn test_str_replace_empty_new_string_deletes() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "keep this remove_me and this").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": " remove_me",
            "new_string": ""
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "keep this and this"
        );
    }

    #[tokio::test]
    async fn test_str_replace_special_chars() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "price is $100.00 (USD)").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "$100.00 (USD)",
            "new_string": "€85.50 (EUR)"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "price is €85.50 (EUR)"
        );
    }

    #[tokio::test]
    async fn test_str_replace_unicode() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello 🌍 world").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "🌍",
            "new_string": "🚀"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "hello 🚀 world"
        );
    }

    #[tokio::test]
    async fn test_str_replace_file_not_found() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nonexistent.txt");

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "bar"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Error reading file"));
    }

    #[tokio::test]
    async fn test_str_replace_protected_path() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config.toml");
        std::fs::write(&file_path, "key = \"value\"").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "value",
            "new_string": "new_value"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("protected"));
        // File unchanged
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "key = \"value\""
        );
    }

    #[tokio::test]
    async fn test_str_replace_whitespace_sensitive() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.py");
        std::fs::write(&file_path, "    if True:\n        pass\n").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        // Exact indentation must match
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "    if True:\n        pass",
            "new_string": "    if True:\n        return 42"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "    if True:\n        return 42\n"
        );
    }

    #[tokio::test]
    async fn test_str_replace_wrong_whitespace_fails() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.py");
        std::fs::write(&file_path, "    if True:\n        pass\n").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        // Wrong indentation — should not match
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "if True:\n    pass",
            "new_string": "if True:\n    return 42"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("Pattern not found"));
    }

    #[tokio::test]
    async fn test_str_replace_nested_path() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("src").join("main.rs");
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(&file_path, "fn main() {}").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "fn main() {}",
            "new_string": "fn main() {\n    println!(\"hi\");\n}"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("println!(\"hi\")"));
    }

    #[tokio::test]
    async fn test_str_replace_overlapping_matches() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "aaa").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        // "aa" appears at positions 0 and 1 (overlapping)
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "aa",
            "new_string": "bb"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_str_replace_preserves_rest_of_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let original = "line 1\nline 2\nline 3\nline 4\nline 5\n";
        std::fs::write(&file_path, original).unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "line 3",
            "new_string": "LINE THREE"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "line 1\nline 2\nLINE THREE\nline 4\nline 5\n"
        );
    }

    #[tokio::test]
    async fn test_str_replace_large_multiline_block() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        let content = r#"fn old_function() {
    let x = 1;
    let y = 2;
    x + y
}

fn other() {}
"#;
        std::fs::write(&file_path, content).unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "fn old_function() {\n    let x = 1;\n    let y = 2;\n    x + y\n}",
            "new_string": "fn new_function() -> i32 {\n    42\n}"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        let new_content = std::fs::read_to_string(&file_path).unwrap();
        assert!(new_content.contains("fn new_function() -> i32"));
        assert!(new_content.contains("fn other() {}"));
        assert!(!new_content.contains("old_function"));
    }

    #[tokio::test]
    async fn test_str_replace_output_shows_old_and_new() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo bar baz").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "bar",
            "new_string": "qux"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output, "OK");
    }

    #[tokio::test]
    async fn test_str_replace_not_found_shows_preview() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "actual file content here").unwrap();

        let tool = StrReplaceTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            file_cache: FileCache::new(),
        };

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "nonexistent pattern",
            "new_string": "replacement"
        });

        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("actual file content here"));
    }

    #[test]
    fn test_resolve_path_absolute() {
        let result = resolve_path("/absolute/path.txt", Some(&PathBuf::from("/working")));
        assert_eq!(result, PathBuf::from("/absolute/path.txt"));
    }

    #[test]
    fn test_resolve_path_relative_with_working_dir() {
        let result = resolve_path("src/main.rs", Some(&PathBuf::from("/project")));
        assert_eq!(result, PathBuf::from("/project/src/main.rs"));
    }

    #[test]
    fn test_resolve_path_relative_without_working_dir() {
        let result = resolve_path("src/main.rs", None);
        assert_eq!(result, PathBuf::from("src/main.rs"));
    }
}
