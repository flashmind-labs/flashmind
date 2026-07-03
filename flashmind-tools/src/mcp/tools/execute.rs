use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;
use crate::mcp::tools::wrapper::McpToolApproval;

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
        "Call a tool on a connected MCP server. Before calling a tool you have not already \
         inspected, run mcp_search filtered to that server (add a query if the server has many \
         tools) to load its parameter schema, then pass exactly those arguments as `params`, a \
         single JSON object, \
         e.g. {\"server\": \"notion\", \"tool\": \"notion-search\", \"params\": {\"query\": \"portfolio\"}}."
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
                    "description": "The tool's arguments as a JSON object (not a string), \
                         e.g. {\"query\": \"portfolio\"}. Run mcp_search filtered to the server \
                         to load the tool's parameter schema and use those exact field names. \
                         Omit or pass {} if the tool takes no arguments."
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

        let params = coerce_params(ctx.args.get("params"));

        if self.mcp.is_tool_restricted(server, tool)
            && !self
                .mcp
                .session_approved_for(server)
                .read()
                .unwrap()
                .contains(tool)
        {
            return Ok(ToolResult::interrupt(
                ctx.tool_call_id,
                Arc::new(McpToolApproval {
                    server_name: server.to_string(),
                    tool_name: tool.to_string(),
                    full_name: format!("{server}_{tool}"),
                    args: params,
                }),
            ));
        }

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

// ---------------------------------------------------------------------------

/// Normalize the `params` value from an LLM tool call into a JSON object.
///
/// The `mcp` proxy tool declares `params` as a nested object, but weaker models
/// (MiniMax, GLM, and others) routinely fail to emit a real nested object. They
/// send it JSON-encoded as a string, as an empty or whitespace-only string, or
/// they leak their native `<arg_key>k</arg_key> <arg_value>v</arg_value>`
/// tool-call template as raw text. In every case the object arrives as a plain
/// string and required fields (for example Notion's `query`) reach the server
/// as `undefined`. This recovers those shapes so the call still succeeds.
fn coerce_params(raw: Option<&Value>) -> Value {
    match raw {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(s)) => coerce_params_str(s),
        Some(other) => other.clone(),
    }
}

fn coerce_params_str(s: &str) -> Value {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    // Common case: the model JSON-encoded the object as a string.
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(trimmed) {
        return v;
    }
    // MiniMax/GLM native tool-call markup leaked as text.
    if let Some(obj) = parse_arg_markup(trimmed) {
        tracing::debug!("recovered MCP params from arg_key/arg_value markup");
        return obj;
    }
    json!({})
}

/// Parse the `<arg_key>k</arg_key> <arg_value>v</arg_value>` markup that some
/// models leak in place of a JSON object. Closing `</arg_value>` tags are
/// optional: a value runs until the next `<arg_key>` or the end of the string.
/// Each value is parsed as JSON when possible, otherwise kept as a string.
fn parse_arg_markup(s: &str) -> Option<Value> {
    if !s.contains("<arg_key>") {
        return None;
    }
    let mut obj = serde_json::Map::new();
    let mut rest = s;
    while let Some(kstart) = rest.find("<arg_key>") {
        let after_key = &rest[kstart + "<arg_key>".len()..];
        let Some(kend) = after_key.find("</arg_key>") else {
            break;
        };
        let key = after_key[..kend].trim().to_string();
        let after_kclose = &after_key[kend + "</arg_key>".len()..];
        let Some(vstart) = after_kclose.find("<arg_value>") else {
            break;
        };
        let after_vopen = &after_kclose[vstart + "<arg_value>".len()..];
        // Value ends at the closing tag or the next key, whichever comes first.
        let vclose = after_vopen.find("</arg_value>");
        let next_key = after_vopen.find("<arg_key>");
        let vend = match (vclose, next_key) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => after_vopen.len(),
        };
        let val = after_vopen[..vend].trim();
        let parsed =
            serde_json::from_str::<Value>(val).unwrap_or_else(|_| Value::String(val.to_string()));
        if !key.is_empty() {
            obj.insert(key, parsed);
        }
        rest = &after_vopen[vend..];
    }
    if obj.is_empty() {
        None
    } else {
        Some(Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_params_pass_through() {
        let v = json!({"query": "portfolio"});
        assert_eq!(coerce_params(Some(&v)), v);
    }

    #[test]
    fn missing_or_null_params_become_empty_object() {
        assert_eq!(coerce_params(None), json!({}));
        assert_eq!(coerce_params(Some(&Value::Null)), json!({}));
    }

    #[test]
    fn whitespace_string_becomes_empty_object() {
        assert_eq!(coerce_params(Some(&json!("\n"))), json!({}));
        assert_eq!(coerce_params(Some(&json!("   "))), json!({}));
    }

    #[test]
    fn json_encoded_string_is_parsed() {
        let raw = json!("{\"query\": \"portfolio\"}");
        assert_eq!(coerce_params(Some(&raw)), json!({"query": "portfolio"}));
    }

    #[test]
    fn arg_markup_without_closing_tag_is_recovered() {
        // Exact shape seen from GLM/MiniMax via OpenRouter for notion-search.
        let raw = json!("<arg_key>query</arg_key> <arg_value>portfolio");
        assert_eq!(coerce_params(Some(&raw)), json!({"query": "portfolio"}));
    }

    #[test]
    fn arg_markup_with_closing_tags_is_recovered() {
        let raw = json!("<arg_key>query</arg_key> <arg_value>hello world</arg_value>");
        assert_eq!(coerce_params(Some(&raw)), json!({"query": "hello world"}));
    }

    #[test]
    fn arg_markup_multiple_pairs() {
        let raw = json!(
            "<arg_key>query</arg_key> <arg_value>docs</arg_value> <arg_key>limit</arg_key> <arg_value>5</arg_value>"
        );
        assert_eq!(
            coerce_params(Some(&raw)),
            json!({"query": "docs", "limit": 5})
        );
    }

    #[test]
    fn arg_markup_json_value_is_parsed() {
        let raw = json!("<arg_key>query</arg_key> <arg_value>{\"limit\": 10, \"q\": \"HSBC\"}");
        assert_eq!(
            coerce_params(Some(&raw)),
            json!({"query": {"limit": 10, "q": "HSBC"}})
        );
    }

    #[test]
    fn unrecoverable_string_becomes_empty_object() {
        assert_eq!(coerce_params(Some(&json!("just some text"))), json!({}));
    }
}
