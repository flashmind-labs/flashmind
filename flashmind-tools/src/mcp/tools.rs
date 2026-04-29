//! LLM-visible tools for MCP server management and tool invocation.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

use super::{McpRegistry, McpServerConfig};

/// Add/register an MCP server configuration (saves to disk, doesn't connect yet).
pub struct McpAddTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpAddTool {
    fn name(&self) -> &str {
        "mcp_add"
    }

    fn description(&self) -> &str {
        "Register an MCP (Model Context Protocol) server. Saves config to disk. \
         Use mcp_run to auto-connect and call tools on it."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for this server (e.g. 'blender', 'github')"
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
        let name = match ctx.args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Missing required 'name' parameter",
                ));
            }
        };

        let command = ctx
            .args
            .get("command")
            .and_then(|v| v.as_str())
            .map(String::from);
        let url = ctx
            .args
            .get("url")
            .and_then(|v| v.as_str())
            .map(String::from);

        if command.is_none() && url.is_none() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Must provide either 'command' or 'url'",
            ));
        }

        let args: Vec<String> = ctx
            .args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let env: HashMap<String, String> = ctx
            .args
            .get("env")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let config = McpServerConfig {
            name: name.clone(),
            command,
            args,
            url,
            env,
        };

        match self.mcp.add(config).await {
            Ok(_) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Registered MCP server '{name}'. Use mcp_run to auto-connect and call tools."
                ),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to register: {e}"),
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
        let name = match ctx.args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Missing required 'name' parameter",
                ));
            }
        };

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
        let server = match ctx.args.get("server").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Missing required 'server' parameter",
                ));
            }
        };

        // No tool specified — list available tools on the server
        let tool = match ctx.args.get("tool").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return match self.mcp.list_tools(server).await {
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
                };
            }
        };

        let arguments = ctx.args.get("arguments").cloned().unwrap_or(json!({}));

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

/// Check OAuth authentication status for MCP servers.
pub struct McpAuthTool;

#[async_trait]
impl Tool for McpAuthTool {
    fn name(&self) -> &str {
        "mcp_auth"
    }

    fn description(&self) -> &str {
        "Check or manage OAuth authentication for MCP servers. \
         Shows auth status and available providers. \
         Use `flash mcp auth <server>` CLI command for interactive OAuth flows."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "MCP server name to check auth for (omit to list all)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let server = ctx.args.get("server").and_then(|v| v.as_str());

        let msg = match server {
            Some(s) => format!(
                "To authenticate MCP server '{s}', run:\n  flash mcp auth {s} --provider <google|microsoft> --scopes <scopes>\n\n\
                 Or use the API:\n  POST /api/v2/oauth/authorize\n  GET /api/v2/oauth/status/{s}"
            ),
            None => "To check OAuth status for all MCP servers:\n  GET /api/v2/oauth/status\n\n\
                 To authenticate a specific server:\n  flash mcp auth <server> --provider <google|microsoft> --scopes <scopes>".to_string(),
        };

        Ok(ToolResult::success(ctx.tool_call_id, msg))
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("all");
        format!("[mcp:auth] {server}")
    }
}
