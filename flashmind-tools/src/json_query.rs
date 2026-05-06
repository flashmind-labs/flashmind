//! JSON query tool using dot-notation paths.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

#[derive(Deserialize)]
struct JsonQueryArgs {
    json: String,
    path: String,
}

/// JSON dot-notation extraction tool.
pub struct JsonQueryTool;

/// Convert a dot-notation path to a JSON pointer string.
///
/// Splits on `.` and handles array indexing like `users[0]` by splitting into
/// the key and index segments. Returns a `/`-prefixed pointer.
fn dot_to_pointer(path: &str) -> String {
    let mut parts = Vec::new();
    for segment in path.split('.') {
        if let Some((key, rest)) = segment.split_once('[') {
            // Key before the bracket
            parts.push(key);
            // Extract indices — handles chained brackets like arr[0][1]
            for piece in rest.split('[') {
                if piece.is_empty() {
                    continue;
                }
                // Strip trailing ']'
                let index = piece.trim_end_matches(']');
                parts.push(index);
            }
        } else {
            parts.push(segment);
        }
    }
    format!("/{}", parts.join("/"))
}

#[async_trait]
impl Tool for JsonQueryTool {
    fn name(&self) -> &str {
        "json_query"
    }

    fn description(&self) -> &str {
        "Extract values from JSON data using dot-notation paths. Supports nested objects and array indexing (e.g., 'data.users[0].name')."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "json": { "type": "string", "description": "JSON string to query" },
                "path": { "type": "string", "description": "Dot-notation path like 'data.users[0].name'" }
            },
            "required": ["json", "path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: JsonQueryArgs = ctx.parse_args(self.name())?;

        let value: Value = match serde_json::from_str(&args.json) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid JSON: {}", e),
                ));
            }
        };

        let pointer = dot_to_pointer(&args.path);

        match value.pointer(&pointer) {
            Some(result) => {
                let pretty = serde_json::to_string_pretty(result).unwrap_or_default();
                Ok(ToolResult::success(ctx.tool_call_id, pretty))
            }
            None => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Path not found: {}", args.path),
            )),
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Querying JSON: {}", path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name() {
        assert_eq!(JsonQueryTool.name(), "json_query");
    }

    #[test]
    fn test_dot_to_pointer() {
        assert_eq!(dot_to_pointer("data.users[0].name"), "/data/users/0/name");
        assert_eq!(dot_to_pointer("simple"), "/simple");
        assert_eq!(dot_to_pointer("a.b.c"), "/a/b/c");
        assert_eq!(dot_to_pointer("arr[2]"), "/arr/2");
    }

    #[tokio::test]
    async fn test_json_query_simple() {
        let result = crate::tests::execute_tool(
            &JsonQueryTool,
            "call1",
            json!({"json": r#"{"name":"Alice"}"#, "path": "name"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert_eq!(result.output().trim(), r#""Alice""#);
    }

    #[tokio::test]
    async fn test_json_query_nested() {
        let data = r#"{"data":{"user":{"age":30}}}"#;
        let result = crate::tests::execute_tool(
            &JsonQueryTool,
            "call2",
            json!({"json": data, "path": "data.user.age"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert_eq!(result.output().trim(), "30");
    }

    #[tokio::test]
    async fn test_json_query_array_index() {
        let data = r#"{"users":[{"name":"Alice"},{"name":"Bob"}]}"#;
        let result = crate::tests::execute_tool(
            &JsonQueryTool,
            "call3",
            json!({"json": data, "path": "users[1].name"}),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert_eq!(result.output().trim(), r#""Bob""#);
    }

    #[tokio::test]
    async fn test_json_query_invalid_json() {
        let result = crate::tests::execute_tool(
            &JsonQueryTool,
            "call4",
            json!({"json": "not json", "path": "x"}),
        )
        .await
        .unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_json_query_path_not_found() {
        let result = crate::tests::execute_tool(
            &JsonQueryTool,
            "call5",
            json!({"json": r#"{"a":1}"#, "path": "b.c"}),
        )
        .await
        .unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Path not found"));
    }
}
