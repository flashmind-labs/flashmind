use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

pub struct McpExecuteTool {
    mcp: McpRegistry,
}

impl McpExecuteTool {
    pub fn new(mcp: McpRegistry) -> Self {
        Self { mcp }
    }
}

#[async_trait]
impl Tool for McpExecuteTool {
    fn name(&self) -> &str {
        "mcp_execute"
    }

    fn description(&self) -> &str {
        "Execute a tool on a connected MCP server. \
         Use mcp_search to discover available tools first."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "MCP server name"
                },
                "tool": {
                    "type": "string",
                    "description": "Tool name (from mcp_search results)"
                },
                "params": {
                    "type": "object",
                    "description": "Tool parameters (varies per tool)"
                }
            },
            "required": ["server", "tool"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let Some(server) = ctx.args.get("server").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure(ctx.tool_call_id, "server is required"));
        };
        let Some(tool) = ctx.args.get("tool").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure(ctx.tool_call_id, "tool is required"));
        };
        let params = ctx.args.get("params").cloned().unwrap_or_else(|| json!({}));

        match self.mcp.call_tool(server, tool, params).await {
            Ok(result) => {
                let text: String = result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("\n");
                if result.is_error {
                    Ok(ToolResult::failure(ctx.tool_call_id, text))
                } else {
                    Ok(ToolResult::success(ctx.tool_call_id, text))
                }
            }
            Err(e) => {
                tracing::warn!(server, tool, error = %e, "MCP tool call failed");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Failed to call {server}/{tool}: {e}"),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("?");
        let tool = args.get("tool").and_then(|v| v.as_str()).unwrap_or("?");
        format!("[mcp:{server}] {tool}")
    }
}
