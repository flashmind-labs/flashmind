use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

#[derive(Deserialize)]
struct McpRemoveArgs {
    name: String,
    confirm: Option<bool>,
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
        "Remove an MCP server. Disconnects if active and deletes the saved configuration. \
         Only use when the user explicitly asks to remove a server — never as a reaction to \
         transient errors (auth failures, timeouts, connection issues). \
         Requires confirm=true to execute."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the MCP server to remove"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to actually remove. First call should omit this — the tool will ask you to reconsider."
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpRemoveArgs = ctx.parse_args(self.name())?;
        let name = &args.name;

        if !args.confirm.unwrap_or(false) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Are you sure you want to permanently remove MCP server '{name}'? \
                     This deletes its saved configuration. If the server is just failing \
                     temporarily (auth errors, timeouts, connection issues), do NOT remove it — \
                     troubleshoot the issue instead. Only remove if the user explicitly asked. \
                     To proceed, call mcp_remove again with confirm=true."
                ),
            ));
        }

        match self.mcp.remove(name).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Removed MCP server '{name}'"),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to remove '{name}': {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Removing MCP server '{name}'")
    }
}
