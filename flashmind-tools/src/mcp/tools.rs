//! MCP management tools — `mcp_add`, `mcp_list`, `mcp_auth`, `mcp_remove`.
//!
//! These tools allow the agent to dynamically manage MCP server connections
//! at runtime without restarting.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

use super::{McpAuthRequired, McpRegistry, McpServerConfig, McpToolDef};

// ---------------------------------------------------------------------------
// McpAddTool
// ---------------------------------------------------------------------------

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
            credentials_json: None,
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
        format!("Adding MCP server '{name}'")
    }
}

// ---------------------------------------------------------------------------
// McpRemoveTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct McpRemoveArgs {
    name: String,
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
        "Remove a registered MCP server. Disconnects and deletes its configuration."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the MCP server to remove"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: McpRemoveArgs = ctx.parse_args(self.name())?;

        match self.mcp.remove(&args.name).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Removed MCP server '{}'", args.name),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to remove '{}': {e}", args.name),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Removing MCP server '{name}'")
    }
}

// ---------------------------------------------------------------------------
// McpListTool
// ---------------------------------------------------------------------------

pub struct McpListTool {
    pub mcp: McpRegistry,
}

#[async_trait]
impl Tool for McpListTool {
    fn name(&self) -> &str {
        "mcp_list"
    }

    fn description(&self) -> &str {
        "List all registered MCP servers and their available tools."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let servers = self.mcp.list().await;

        if servers.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No MCP servers registered.",
            ));
        }

        let mut output = String::new();
        for (name, tools, error) in &servers {
            output.push_str(&format!("## {name}\n"));
            if let Some(err) = error {
                output.push_str(&format!("  Error: {err}\n"));
            } else if tools.is_empty() {
                output.push_str("  (no tools)\n");
            } else {
                for tool in tools {
                    output.push_str(&format!("  - {tool}\n"));
                }
            }
            output.push('\n');
        }

        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing MCP servers".to_string()
    }
}

// ---------------------------------------------------------------------------
// McpAuthTool
// ---------------------------------------------------------------------------

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

        let output = json!({
            "server": args.server,
        });
        Ok(ToolResult::interrupt(ctx.tool_call_id, output.to_string()))
    }

    fn humanize(&self, args: &Value) -> String {
        let server = args.get("server").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Authenticating MCP server '{server}'")
    }
}

// ---------------------------------------------------------------------------
// McpToolWrapper
// ---------------------------------------------------------------------------

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
    tool_defs: &[McpToolDef],
) -> Vec<Arc<dyn Tool>> {
    tool_defs
        .iter()
        .map(|t| {
            Arc::new(McpToolWrapper {
                mcp: mcp.clone(),
                server_name: server_name.to_string(),
                tool_name: t.name.clone(),
                full_name: format!("{}_{}", server_name, t.name),
                tool_description: t.description.clone().unwrap_or_default(),
                tool_schema: t.input_schema.clone(),
            }) as Arc<dyn Tool>
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
                format!("MCP error: {e:#}"),
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
        struct InMemoryProvider;

        #[async_trait]
        impl super::super::McpConfigProvider for InMemoryProvider {
            async fn list_configs(&self) -> anyhow::Result<Vec<McpServerConfig>> {
                Ok(vec![])
            }
            async fn save_config(&self, _config: &McpServerConfig) -> anyhow::Result<()> {
                Ok(())
            }
            async fn delete_config(&self, _name: &str) -> anyhow::Result<()> {
                Ok(())
            }
            async fn save_credentials(
                &self,
                _name: &str,
                _credentials_json: &str,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            async fn load_credentials(&self, _name: &str) -> anyhow::Result<Option<String>> {
                Ok(None)
            }
        }

        let provider: Arc<dyn super::super::McpConfigProvider> = Arc::new(InMemoryProvider);
        let registry = McpRegistry::new(provider);

        let tool_defs = vec![
            McpToolDef {
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
            McpToolDef {
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
}
