use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::config::McpServerConfig;
use crate::mcp::registry::McpRegistry;
use crate::mcp::types::McpAuthRequired;

#[derive(Deserialize)]
struct McpAddArgs {
    name: String,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    url: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    #[serde(default)]
    scopes: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

pub struct McpAddTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpAddTool {
    fn name(&self) -> &str {
        "mcp_add"
    }

    fn description(&self) -> &str {
        "Register and connect an MCP server. Supports two transports:\n\
         - **stdio**: local process via `command` + `args`. Pass API tokens as `env` vars.\n\
         - **HTTP/SSE**: remote server via `url`. OAuth is handled automatically — if the \
         server returns 401, the user is prompted to authenticate via a browser link."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for this server (e.g. 'fastmail', 'github')"
                },
                "command": {
                    "type": "string",
                    "description": "Command to spawn the MCP server (stdio transport)"
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Arguments for the command"
                },
                "url": {
                    "type": "string",
                    "description": "HTTP/SSE endpoint URL (alternative to command)"
                },
                "client_id": {
                    "type": "string",
                    "description": "OAuth client ID (for providers that require pre-registered credentials)"
                },
                "client_secret": {
                    "type": "string",
                    "description": "OAuth client secret"
                },
                "scopes": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "OAuth scopes to request"
                },
                "env": {
                    "type": "object",
                    "additionalProperties": { "type": "string" },
                    "description": "Environment variables for the server process (key-value pairs)"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpAddArgs = ctx.parse_args(self.name())?;

        if args.command.is_none() && args.url.is_none() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Must provide either 'command' or 'url'",
            ));
        }

        let config = McpServerConfig {
            name: args.name.clone(),
            command: args.command,
            args: args.args,
            url: args.url,
            env: args.env,
            client_id: args.client_id,
            client_secret: args.client_secret,
            scopes: args.scopes,
            credentials: None,
            reauth: None,
            cached_tools: vec![],
        };

        if let Err(e) = self.mcp.add(config.clone()).await {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to save config: {e}"),
            ));
        }

        let name = &args.name;
        match self.mcp.connect(config).await {
            Ok(tools) => {
                let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!(
                        "Connected to MCP server '{name}' with {} tools: {}",
                        tools.len(),
                        tool_names.join(", ")
                    ),
                ))
            }
            Err(e) if e.downcast_ref::<McpAuthRequired>().is_some() => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Server '{name}' registered but requires authentication. \
                     Call mcp_auth with server=\"{name}\" to authorize."
                ),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Registered '{name}' but failed to connect: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
            let cmd_args: Vec<&str> = args
                .get("args")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            if cmd_args.is_empty() {
                format!("Adding MCP server '{name}' ({cmd})")
            } else {
                format!("Adding MCP server '{name}' ({cmd} {})", cmd_args.join(" "))
            }
        } else if let Some(url) = args.get("url").and_then(|v| v.as_str()) {
            format!("Adding MCP server '{name}' ({url})")
        } else {
            format!("Adding MCP server '{name}'")
        }
    }
}
