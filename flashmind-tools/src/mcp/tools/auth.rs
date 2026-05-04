use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::auth::AuthOutcome;
use crate::mcp::registry::McpRegistry;

#[derive(Deserialize)]
struct McpAuthArgs {
    server: String,
}

pub struct McpAuthTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpAuthTool {
    fn name(&self) -> &str {
        "mcp_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with an MCP server that requires OAuth. Returns an authorization URL \
         for the user to visit in their browser."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the MCP server to authenticate with"
                }
            },
            "required": ["server"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpAuthArgs = ctx.parse_args(self.name())?;

        match self.mcp.authenticate(&args.server).await {
            Ok(AuthOutcome::Completed) => {
                let tools = self.mcp.current_mcp_tools();
                let tool_names: Vec<&str> = tools
                    .get(&args.server)
                    .map(|t| t.iter().map(|d| d.name.as_str()).collect())
                    .unwrap_or_default();
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!(
                        "Authenticated and reconnected to '{}' with {} tools: {}",
                        args.server,
                        tool_names.len(),
                        tool_names.join(", ")
                    ),
                ))
            }
            Ok(AuthOutcome::InteractionRequired { message }) => {
                Ok(ToolResult::interrupt(ctx.tool_call_id, message))
            }
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Authentication failed for '{}': {e}", args.server),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Authenticating MCP server '{server}'")
    }
}
