use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

pub struct McpListTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpListTool {
    fn name(&self) -> &str {
        "mcp_list"
    }

    fn description(&self) -> &str {
        "List all registered MCP servers and their available tools."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        crate::utils::validate_tool_args(&ctx.args, &self.parameters())?;
        let servers = self.mcp.list().await;

        if servers.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No MCP servers registered.",
            ));
        }

        let mut output = String::new();
        for (name, tools, error) in &servers {
            output.push_str(&format!("## {name}\n"));
            if let Some(err) = error {
                output.push_str(&format!("  Error: {err}\n"));
            } else if tools.is_empty() {
                output.push_str("  (no tools)\n");
            } else {
                for tool in tools {
                    output.push_str(&format!("  - {tool}\n"));
                }
            }
            output.push('\n');
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing MCP servers".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_unknown_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let registry = McpRegistry::new(
            crate::mcp::config::McpDiskConfig::new(dir.path().to_path_buf()),
            None,
        );
        let tool = McpListTool { mcp: registry };
        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("call", json!({"typo": true}), None, &cancel);
        let error = tool
            .execute(ctx)
            .await
            .expect_err("unknown argument must fail");
        assert!(error.to_string().contains("unknown argument `typo`"));
    }
}
