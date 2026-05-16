use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{InterruptPayload, Tool, ToolContext, ToolResult};

use crate::mcp::auth::AuthOutcome;
use crate::mcp::registry::McpRegistry;

#[derive(Debug)]
pub struct McpOAuthInterrupt {
    pub server: String,
    pub message: String,
}

impl InterruptPayload for McpOAuthInterrupt {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn display_output(&self) -> String {
        self.message.clone()
    }
}

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
        "Authenticate with an MCP server that requires OAuth or a reauth command."
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

        // Check for a reauth command before attempting OAuth.
        let reauth_cmd = self
            .mcp
            .get_config(&args.server)
            .await
            .and_then(|cfg| cfg.reauth.clone());

        if let Some(cmd) = reauth_cmd {
            return self.run_reauth(ctx.tool_call_id, &args.server, &cmd).await;
        }

        match self.mcp.authenticate(&args.server).await {
            Ok(AuthOutcome::Completed) => Ok(self.success_result(ctx.tool_call_id, &args.server)),
            Ok(AuthOutcome::InteractionRequired { message }) => Ok(ToolResult::interrupt(
                ctx.tool_call_id,
                Arc::new(McpOAuthInterrupt {
                    server: args.server.clone(),
                    message,
                }),
            )),
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

impl McpAuthTool {
    async fn run_reauth(
        &self,
        tool_call_id: &str,
        server: &str,
        cmd: &str,
    ) -> anyhow::Result<ToolResult> {
        tracing::info!(server, cmd, "running reauth command");

        let output = tokio::process::Command::new("sh")
            .args(["-c", cmd])
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => {
                if let Err(e) = self.mcp.reconnect(server).await {
                    tracing::warn!(server, error = %e, "reconnect after reauth failed");
                }
                Ok(self.success_result(tool_call_id, server))
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                Ok(ToolResult::failure(
                    tool_call_id,
                    format!(
                        "Reauth command exited with {}: {}",
                        out.status,
                        stderr.trim()
                    ),
                ))
            }
            Err(e) => Ok(ToolResult::failure(
                tool_call_id,
                format!("Failed to run reauth command: {e}"),
            )),
        }
    }

    fn success_result(&self, tool_call_id: &str, server: &str) -> ToolResult {
        let tools = self.mcp.current_mcp_tools();
        let tool_names: Vec<&str> = tools
            .get(server)
            .map(|t| t.iter().map(|d| d.name.as_str()).collect())
            .unwrap_or_default();
        ToolResult::success(
            tool_call_id,
            format!(
                "Authenticated and reconnected to '{}' with {} tools: {}",
                server,
                tool_names.len(),
                tool_names.join(", ")
            ),
        )
    }
}
