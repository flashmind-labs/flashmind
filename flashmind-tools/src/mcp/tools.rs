//! LLM-visible tools for MCP server management and tool invocation.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

use super::{McpRegistry, McpServerConfig};

/// Arguments for registering an MCP server (`mcp_add`).
#[derive(Deserialize)]
struct McpAddArgs {
    name: String,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    url: Option<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

/// Arguments for removing an MCP server (`mcp_remove`).
#[derive(Deserialize)]
struct McpRemoveArgs {
    name: String,
}

/// Arguments for calling a tool on an MCP server (`mcp_run`).
#[derive(Deserialize)]
struct McpRunArgs {
    server: String,
    tool: Option<String>,
    #[serde(default)]
    arguments: Value,
}

/// Register an MCP server and connect immediately.
/// If OAuth is required, opens the browser and blocks until the user authorizes.
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
            credentials_json: None,
        };

        // Save config first so it persists even if connect fails
        if let Err(e) = self.mcp.add(config.clone()).await {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to save config: {e}"),
            ));
        }

        // Now connect — this may trigger OAuth (opens browser, blocks until authorized)
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
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Registered '{name}' but failed to connect: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Adding MCP server '{name}'")
    }
}

/// Remove an MCP server — disconnect if active and delete the saved config.
pub struct McpRemoveTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpRemoveTool {
    fn name(&self) -> &str {
        "mcp_remove"
    }

    fn description(&self) -> &str {
        "Remove an MCP server. Disconnects if active and deletes the saved configuration."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the server to remove"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpRemoveArgs = ctx.parse_args(self.name())?;
        let name = &args.name;

        match self.mcp.remove(name).await {
            Ok(_) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Removed MCP server '{name}'."),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to remove: {e}"),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Removing MCP server '{name}'")
    }
}

/// List registered MCP servers and their tools.
pub struct McpListTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpListTool {
    fn name(&self) -> &str {
        "mcp_list"
    }

    fn description(&self) -> &str {
        "List registered MCP servers and their available tools. Auto-connects if needed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let servers = self.mcp.list().await;
        if servers.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No MCP servers registered. Use mcp_add to register one.",
            ));
        }

        let mut output = String::new();
        for (name, tool_names, error) in &servers {
            if let Some(err) = error {
                output.push_str(&format!("**{name}** [error: {err}]\n"));
            } else if tool_names.is_empty() {
                output.push_str(&format!("**{name}** (no tools)\n"));
            } else {
                output.push_str(&format!("**{name}**: {}\n", tool_names.join(", ")));
            }
        }
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing MCP servers".to_string()
    }
}

/// Call a tool on an MCP server. Auto-connects if needed.
pub struct McpRunTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpRunTool {
    fn name(&self) -> &str {
        "mcp_run"
    }

    fn description(&self) -> &str {
        "Call a tool on an MCP server. Auto-connects if needed. \
         Call with just `server` (no `tool`) to list the server's available tools."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the MCP server (as used in mcp_add)"
                },
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to call on the server. Omit to list available tools."
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool (matching its input schema)"
                }
            },
            "required": ["server"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let parsed: McpRunArgs = ctx.parse_args(self.name())?;
        let server = &parsed.server;

        if let Some(tool) = &parsed.tool {
            let arguments = if parsed.arguments.is_null() {
                json!({})
            } else {
                parsed.arguments
            };

            match self.mcp.call_tool(server, tool, arguments).await {
                Ok(result) => {
                    let output: String = result
                        .content
                        .iter()
                        .filter_map(|c| c.as_text())
                        .collect::<Vec<_>>()
                        .join("\n");

                    if result.is_error {
                        Ok(ToolResult::failure(ctx.tool_call_id, output))
                    } else {
                        Ok(ToolResult::success(ctx.tool_call_id, output))
                    }
                }
                Err(e) => Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("MCP error: {}", e),
                )),
            }
        } else {
            // No tool specified — list available tools for this server
            match self.mcp.list_tools(server).await {
                Ok(tools) => {
                    let list: Vec<String> = tools
                        .iter()
                        .map(|t| {
                            if let Some(ref desc) = t.description {
                                format!("- **{}**: {}", t.name, desc)
                            } else {
                                format!("- **{}**", t.name)
                            }
                        })
                        .collect();
                    Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!(
                            "Server '{}' has {} tools:\n{}",
                            server,
                            list.len(),
                            list.join("\n")
                        ),
                    ))
                }
                Err(e) => Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("MCP error: {e}"),
                )),
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
