use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;

pub struct McpSearchTool {
    mcp: McpRegistry,
}

impl McpSearchTool {
    pub fn new(mcp: McpRegistry) -> Self {
        Self { mcp }
    }
}

#[async_trait]
impl Tool for McpSearchTool {
    fn name(&self) -> &str {
        "mcp_search"
    }

    fn description(&self) -> &str {
        "Search available tools across connected MCP servers. \
         Returns tool names and descriptions. Use mcp_execute to run a tool."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Filter by server name"
                },
                "query": {
                    "type": "string",
                    "description": "Search keyword to match against tool names and descriptions"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let server_filter = ctx.args.get("server").and_then(|v| v.as_str());
        let query_filter = ctx.args.get("query").and_then(|v| v.as_str());
        let query_lower = query_filter.map(|q| q.to_lowercase());

        let all_tools = self.mcp.current_mcp_tools();
        if all_tools.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No MCP servers connected.",
            ));
        }

        let mut results: Vec<Value> = Vec::new();
        for (server_name, tool_defs) in &all_tools {
            if let Some(sf) = server_filter
                && !server_name.eq_ignore_ascii_case(sf)
            {
                continue;
            }
            for tool in tool_defs {
                if let Some(ref q) = query_lower {
                    let name_lower = tool.name.to_lowercase();
                    let desc_lower = tool.description.as_deref().unwrap_or("").to_lowercase();
                    if !name_lower.contains(q.as_str()) && !desc_lower.contains(q.as_str()) {
                        continue;
                    }
                }
                results.push(json!({
                    "server": server_name,
                    "tool": tool.name,
                    "description": tool.description.as_deref().unwrap_or(""),
                }));
            }
        }

        if results.is_empty() {
            let server_names: Vec<&String> = all_tools.keys().collect();
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "No matching tools found. Connected servers: {}",
                    server_names
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }

        let output = serde_json::to_string_pretty(&results).unwrap_or_default();
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("all");
        let query = args.get("query").and_then(|v| v.as_str());
        match query {
            Some(q) => format!("Searching MCP tools (server={server}, query=\"{q}\")"),
            None => format!("Listing MCP tools (server={server})"),
        }
    }
}
