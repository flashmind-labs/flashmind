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
/// `server` + `tool` + `params` interface. Its `parameters()` schema
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
            "additionalProperties": false,
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
        crate::utils::validate_tool_args(&ctx.args, &self.parameters())?;
        let Some(server) = ctx.args.get("server").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure(ctx.tool_call_id, "server is required"));
        };
        let Some(tool) = ctx.args.get("tool").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure(ctx.tool_call_id, "tool is required"));
        };

        let params = coerce_params(ctx.args.get("params"))?;

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
/// Accept JSON-encoded objects and argument markup emitted by some models.
/// Invalid input fails instead of becoming an empty remote request.
fn coerce_params(raw: Option<&Value>) -> anyhow::Result<Value> {
    match raw {
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(s)) => coerce_params_str(s),
        Some(other @ Value::Object(_)) => Ok(other.clone()),
        Some(_) => anyhow::bail!("params must be an object"),
    }
}

fn coerce_params_str(s: &str) -> anyhow::Result<Value> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }
    // Common case: the model JSON-encoded the object as a string.
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(trimmed) {
        return Ok(v);
    }
    // MiniMax/GLM native tool-call markup leaked as text.
    if let Some(obj) = parse_arg_markup(trimmed) {
        tracing::debug!("recovered MCP params from arg_key/arg_value markup");
        return Ok(obj);
    }
    anyhow::bail!("params must contain a JSON object or argument markup")
}

/// Parse the `<arg_key>k</arg_key> <arg_value>v</arg_value>` markup that some
/// models leak in place of a JSON object. Closing `</arg_value>` tags are
/// optional: a value runs until the next `<arg_key>` or the end of the string.
/// Each value is parsed as JSON when possible, otherwise kept as a string.
fn parse_arg_markup(s: &str) -> Option<Value> {
    let mut obj = serde_json::Map::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        let after_key = rest.strip_prefix("<arg_key>")?;
        let (key, after_key) = after_key.split_once("</arg_key>")?;
        let key = key.trim();
        if key.is_empty() || obj.contains_key(key) {
            return None;
        }
        let value = after_key.trim_start().strip_prefix("<arg_value>")?;
        let closing = value.find("</arg_value>");
        let next_key = value.find("<arg_key>");
        let end = match (closing, next_key) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => value.len(),
        };
        let text = value[..end].trim();
        let parsed =
            serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.to_string()));
        obj.insert(key.to_string(), parsed);
        rest = &value[end..];
        if let Some(after_close) = rest.strip_prefix("</arg_value>") {
            rest = after_close;
        }
        rest = rest.trim();
    }
    (!obj.is_empty()).then_some(Value::Object(obj))
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn rejects_unknown_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let registry = McpRegistry::new(
            crate::mcp::config::McpDiskConfig::new(dir.path().to_path_buf()),
            None,
        );
        let tool = McpExecuteTool { mcp: registry };
        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new("call", json!({"typo": true}), None, &cancel);
        let error = tool
            .execute(ctx)
            .await
            .expect_err("unknown argument must fail");
        assert!(error.to_string().contains("unknown argument `typo`"));
    }

    use super::*;

    #[test]
    fn object_params_pass_through() {
        let v = json!({"query": "portfolio"});
        assert_eq!(coerce_params(Some(&v)).unwrap(), v);
    }

    #[test]
    fn missing_or_null_params_become_empty_object() {
        assert_eq!(coerce_params(None).unwrap(), json!({}));
        assert_eq!(coerce_params(Some(&Value::Null)).unwrap(), json!({}));
    }

    #[test]
    fn whitespace_string_becomes_empty_object() {
        assert_eq!(coerce_params(Some(&json!("\n"))).unwrap(), json!({}));
        assert_eq!(coerce_params(Some(&json!("   "))).unwrap(), json!({}));
    }

    #[test]
    fn json_encoded_string_is_parsed() {
        let raw = json!("{\"query\": \"portfolio\"}");
        assert_eq!(
            coerce_params(Some(&raw)).unwrap(),
            json!({"query": "portfolio"})
        );
    }

    #[test]
    fn arg_markup_without_closing_tag_is_recovered() {
        // Exact shape seen from GLM/MiniMax via OpenRouter for notion-search.
        let raw = json!("<arg_key>query</arg_key> <arg_value>portfolio");
        assert_eq!(
            coerce_params(Some(&raw)).unwrap(),
            json!({"query": "portfolio"})
        );
    }

    #[test]
    fn arg_markup_with_closing_tags_is_recovered() {
        let raw = json!("<arg_key>query</arg_key> <arg_value>hello world</arg_value>");
        assert_eq!(
            coerce_params(Some(&raw)).unwrap(),
            json!({"query": "hello world"})
        );
    }

    #[test]
    fn arg_markup_multiple_pairs() {
        let raw = json!(
            "<arg_key>query</arg_key> <arg_value>docs</arg_value> <arg_key>limit</arg_key> <arg_value>5</arg_value>"
        );
        assert_eq!(
            coerce_params(Some(&raw)).unwrap(),
            json!({"query": "docs", "limit": 5})
        );
    }

    #[test]
    fn arg_markup_json_value_is_parsed() {
        let raw = json!("<arg_key>query</arg_key> <arg_value>{\"limit\": 10, \"q\": \"example\"}");
        assert_eq!(
            coerce_params(Some(&raw)).unwrap(),
            json!({"query": {"limit": 10, "q": "example"}})
        );
    }

    #[test]
    fn malformed_markup_is_rejected_without_partial_forwarding() {
        for raw in [
            "<arg_key>query</arg_key><arg_value>docs</arg_value><arg_key>limit</arg_key>",
            "<arg_key>query</arg_key><arg_value>first</arg_value><arg_key>query</arg_key><arg_value>second",
            "<arg_key>query</arg_key><arg_value>docs</arg_value>orphan text",
        ] {
            assert!(coerce_params(Some(&json!(raw))).is_err());
        }
    }

    #[test]
    fn invalid_params_are_rejected() {
        for raw in [json!("just some text"), json!([1]), json!(5), json!(true)] {
            assert!(coerce_params(Some(&raw)).is_err());
        }
    }
}
