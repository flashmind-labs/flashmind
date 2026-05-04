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
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
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
