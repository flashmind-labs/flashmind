use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

#[derive(Deserialize)]
struct McpRemoveArgs {
    name: String,
}

pub struct McpRemoveTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpRemoveTool {
    fn name(&self) -> &str {
        "mcp_remove"
    }

    fn description(&self) -> &str {
        "Remove a registered MCP server. Disconnects and deletes its configuration."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the MCP server to remove"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpRemoveArgs = ctx.parse_args(self.name())?;

        match self.mcp.remove(&args.name).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Removed MCP server '{}'", args.name),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to remove '{}': {e}", args.name),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Removing MCP server '{name}'")
    }
}
