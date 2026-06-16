use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

/// Single proxy tool for calling any tool on any connected MCP server.
///
/// Instead of registering one tool per MCP server tool (which can easily
/// reach 50+ entries for a single Grafana server), this tool exposes a
/// `server` + `tool` + `params` interface.  Its [`parameters()`] schema
/// dynamically lists connected servers and their tool names so the LLM
/// knows what's available without a separate discovery step.
pub struct McpExecuteTool {
    pub(crate) mcp: McpRegistry,
}

impl McpExecuteTool {
    pub fn new(mcp: McpRegistry) -> Self {
        Self { mcp }
    }
}

#[async_trait]
impl Tool for McpExecuteTool {
    fn name(&self) -> &str {
        "mcp"
    }

    fn description(&self) -> &str {
        "Call a tool on a connected MCP server. Use mcp_search to see tool descriptions and parameters."
    }

    fn parameters(&self) -> Value {
        let all_tools = self.mcp.current_mcp_tools();

        let (server_desc, tool_desc) = if all_tools.is_empty() {
            (
                "MCP server name".to_string(),
                "Tool name on the server".to_string(),
            )
        } else {
            let mut servers: Vec<&String> = all_tools.keys().collect();
            servers.sort();

            let mut catalog = String::from("Tool name. Available tools per server:");
            for server in &servers {
                let tools = &all_tools[*server];
                let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                names.sort();
                catalog.push_str(&format!("\n- {}: {}", server, names.join(", ")));
            }

            let server_names: Vec<&str> = servers.iter().map(|s| s.as_str()).collect();
            (
                format!("MCP server name. Connected: {}", server_names.join(", ")),
                catalog,
            )
        };

        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": server_desc
                },
                "tool": {
                    "type": "string",
                    "description": tool_desc
                },
                "params": {
                    "type": "object",
                    "description": "Tool parameters (use mcp_search to see parameter schemas)"
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

    fn max_output_bytes(&self) -> usize {
        usize::MAX
    }

    fn max_output_lines(&self) -> usize {
        usize::MAX
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("?");
        let tool = args.get("tool").and_then(|v| v.as_str()).unwrap_or("?");
        format!("[mcp:{server}] {tool}")
    }
}
