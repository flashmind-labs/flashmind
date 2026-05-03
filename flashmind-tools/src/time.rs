//! Time tool — returns current date and time in UTC.

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

/// UTC date/time retrieval tool.
pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str {
        "get_time"
    }

    fn description(&self) -> &str {
        "Get the current date and time in UTC. If you need the user's local time, ask them for their timezone offset."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let now = chrono::Utc::now();
        let output = format!(
            "UTC: {}\nUnix: {}\nDay: {}",
            now.format("%Y-%m-%d %H:%M:%S+00:00"),
            now.timestamp(),
            now.format("%A"),
        );
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &serde_json::Value) -> String {
        "Getting time".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_time_returns_success() {
        let tool = TimeTool;
        let result = crate::tests::execute_tool(&tool, "test-id", json!({}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("UTC:"));
        assert!(result.output.contains("Unix:"));
        assert!(result.output.contains("Day:"));
    }

    #[test]
    fn test_time_tool_metadata() {
        let tool = TimeTool;
        assert_eq!(tool.name(), "get_time");
        assert!(!tool.description().is_empty());
    }
}
