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

/// Wrapper that exposes a single MCP server tool as a first-class Tool.
pub struct McpToolWrapper {
    pub mcp: McpRegistry,
    pub server_name: String,
    pub tool_name: String,
    pub full_name: String,
    pub tool_description: String,
    pub tool_schema: Value,
}

pub fn make_mcp_tool_wrappers(
    mcp: &McpRegistry,
    server_name: &str,
    tool_defs: &[super::McpToolDef],
) -> Vec<std::sync::Arc<dyn Tool>> {
    tool_defs
        .iter()
        .map(|t| {
            std::sync::Arc::new(McpToolWrapper {
                mcp: mcp.clone(),
                server_name: server_name.to_string(),
                tool_name: t.name.clone(),
                full_name: format!("{}_{}", server_name, t.name),
                tool_description: t.description.clone().unwrap_or_default(),
                tool_schema: t.input_schema.clone(),
            }) as std::sync::Arc<dyn Tool>
        })
        .collect()
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.full_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters(&self) -> Value {
        self.tool_schema.clone()
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let arguments = if ctx.args.is_null() {
            json!({})
        } else {
            ctx.args.clone()
        };

        match self
            .mcp
            .call_tool(&self.server_name, &self.tool_name, arguments)
            .await
        {
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
                format!("MCP error: {e}"),
            )),
        }
    }

    fn max_output_bytes(&self) -> usize {
        usize::MAX
    }

    fn max_output_lines(&self) -> usize {
        usize::MAX
    }

    fn humanize(&self, _args: &Value) -> String {
        format!("[mcp:{}] {}", self.server_name, self.tool_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_mcp_tool_wrappers() {
        let dir = tempfile::tempdir().unwrap();
        let registry = McpRegistry::new(dir.path().to_path_buf());

        let tool_defs = vec![
            super::super::McpToolDef {
                name: "search_emails".into(),
                description: Some("Search emails by query".into()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            },
            super::super::McpToolDef {
                name: "send_email".into(),
                description: None,
                input_schema: json!({"type": "object"}),
            },
        ];

        let wrappers = make_mcp_tool_wrappers(&registry, "gmail", &tool_defs);
        assert_eq!(wrappers.len(), 2);
        assert_eq!(wrappers[0].name(), "gmail_search_emails");
        assert_eq!(wrappers[0].description(), "Search emails by query");
        assert_eq!(wrappers[1].name(), "gmail_send_email");
        assert_eq!(wrappers[1].description(), "");

        let params = wrappers[0].parameters();
        assert_eq!(params["properties"]["query"]["type"], "string");
    }

    #[test]
    fn test_wrapper_humanize() {
        let dir = tempfile::tempdir().unwrap();
        let registry = McpRegistry::new(dir.path().to_path_buf());

        let wrapper = McpToolWrapper {
            mcp: registry,
            server_name: "gmail".into(),
            tool_name: "search".into(),
            full_name: "gmail_search".into(),
            tool_description: "Search".into(),
            tool_schema: json!({}),
        };

        assert_eq!(wrapper.humanize(&json!({})), "[mcp:gmail] search");
    }
}
