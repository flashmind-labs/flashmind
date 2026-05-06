//! String diff tool - compares two strings and returns unified diff.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

#[derive(Deserialize)]
struct StrDiffArgs {
    /// The original/string to compare from.
    old: String,
    /// The new string to compare to.
    new: String,
    /// Optional context radius (number of unchanged lines to show around changes).
    /// Default: 3.
    #[serde(default)]
    context_radius: Option<usize>,
}

/// Unified diff generation tool.
pub struct StrDiffTool;

#[async_trait]
impl Tool for StrDiffTool {
    fn name(&self) -> &str {
        "str_diff"
    }

    fn description(&self) -> &str {
        "Compare two strings and return a unified diff showing the differences. \
        Use this to see exactly what changed between two text versions."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "old": {
                    "type": "string",
                    "description": "The original/string to compare from"
                },
                "new": {
                    "type": "string",
                    "description": "The new string to compare to"
                },
                "context_radius": {
                    "type": "integer",
                    "description": "Optional context radius (number of unchanged lines to show around changes). Default: 3.",
                    "minimum": 0,
                    "maximum": 20
                }
            },
            "required": ["old", "new"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StrDiffArgs = ctx.parse_args(self.name())?;

        let diff = TextDiff::from_lines(&args.old, &args.new);

        // Check if there are any changes
        if diff.ops().is_empty()
            || diff
                .iter_all_changes()
                .all(|change| change.tag() == ChangeTag::Equal)
        {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No differences found - strings are identical.",
            ));
        }

        let context = args.context_radius.unwrap_or(3);

        let output = diff
            .unified_diff()
            .header("a/old", "b/new")
            .context_radius(context)
            .to_string();

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let radius = args
            .get("context_radius")
            .and_then(|v| v.as_u64())
            .unwrap_or(3);
        format!("Comparing strings (radius: {})", radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name() {
        assert_eq!(StrDiffTool.name(), "str_diff");
    }

    #[tokio::test]
    async fn test_str_diff_identical() {
        let result = crate::tests::execute_tool(
            &StrDiffTool,
            "call1",
            json!({"old": "hello\nworld", "new": "hello\nworld"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert!(result.output().contains("No differences found"));
    }

    #[tokio::test]
    async fn test_str_diff_with_changes() {
        let result = crate::tests::execute_tool(
            &StrDiffTool,
            "call2",
            json!({"old": "line one\nline two\nline three", "new": "line one\nline TWO\nline three"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert!(result.output().contains("-line two"));
        assert!(result.output().contains("+line TWO"));
    }

    #[tokio::test]
    async fn test_str_diff_additions() {
        let result = crate::tests::execute_tool(
            &StrDiffTool,
            "call3",
            json!({"old": "hello\nworld", "new": "hello\nbeautiful\nworld"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert!(result.output().contains("+beautiful"));
    }

    #[tokio::test]
    async fn test_str_diff_deletions() {
        let result = crate::tests::execute_tool(
            &StrDiffTool,
            "call4",
            json!({"old": "hello\nbeautiful\nworld", "new": "hello\nworld"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert!(result.output().contains("-beautiful"));
    }

    #[tokio::test]
    async fn test_str_diff_custom_context() {
        // 10 lines with change in middle, context_radius=1 should only show 1 line around change
        let old = (1..=10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut new = old.clone();
        new = new.replace("line 5", "line FIVE");

        let result = crate::tests::execute_tool(
            &StrDiffTool,
            "call5",
            json!({"old": old, "new": new, "context_radius": 1}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert!(result.output().contains("-line 5"));
        assert!(result.output().contains("+line FIVE"));
    }
}
