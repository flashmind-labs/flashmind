use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::Host;
use crate::mcp::registry::McpRegistry;
use crate::mcp::types::McpToolDef;

/// Above this many matched tools, `mcp_search` returns names and descriptions only and
/// omits parameter schemas, which would otherwise bloat context (a single Notion tool
/// schema is ~8 KB). Narrowing the search by server or query brings the schemas back.
const SCHEMA_THRESHOLD: usize = 8;

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
        "Search available tools across connected MCP servers. Returns tool names and \
         descriptions. When the search is narrow (filter by `server`, or add a `query` \
         that matches only a few tools) it also returns each tool's parameter schema. \
         Run it before calling a tool with the `mcp` tool so you pass the right arguments."
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

        let output = render_search(&all_tools, server_filter, query_lower.as_deref());
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

// ---------------------------------------------------------------------------

/// Filter the connected servers' tools and render the search output.
///
/// `server_filter` matches a server name case-insensitively; `query` matches
/// case-insensitively against tool names and descriptions (both already lowercased by
/// the caller). Each tool's `input_schema` is included only when the match count is at
/// or below [`SCHEMA_THRESHOLD`], so a broad listing stays lean; above it, the output
/// carries a note telling the caller how to narrow the search to get the schemas.
fn render_search(
    all_tools: &HashMap<Host, Vec<McpToolDef>>,
    server_filter: Option<&str>,
    query_lower: Option<&str>,
) -> String {
    let mut matched: Vec<(&String, &McpToolDef)> = Vec::new();
    for (server_name, tool_defs) in all_tools {
        if let Some(sf) = server_filter
            && !server_name.eq_ignore_ascii_case(sf)
        {
            continue;
        }
        for tool in tool_defs {
            if let Some(q) = query_lower {
                let name_lower = tool.name.to_lowercase();
                let desc_lower = tool.description.as_deref().unwrap_or("").to_lowercase();
                if !name_lower.contains(q) && !desc_lower.contains(q) {
                    continue;
                }
            }
            matched.push((server_name, tool));
        }
    }

    if matched.is_empty() {
        let mut server_names: Vec<&str> = all_tools.keys().map(|s| s.as_str()).collect();
        server_names.sort_unstable();
        return format!(
            "No matching tools found. Connected servers: {}",
            server_names.join(", ")
        );
    }

    // Include each tool's parameter schema only when the search is already narrow. A
    // broad, unfiltered listing across every server would inline hundreds of KB of
    // schema (Notion alone is ~84 KB), so above the threshold we return names and
    // descriptions only and tell the caller how to narrow to get the schemas.
    let include_schemas = matched.len() <= SCHEMA_THRESHOLD;

    let results: Vec<Value> = matched
        .iter()
        .map(|(server_name, tool)| {
            let mut entry = json!({
                "server": server_name,
                "tool": tool.name,
                "description": tool.description.as_deref().unwrap_or(""),
            });
            if include_schemas {
                entry["input_schema"] = tool.input_schema.clone();
            }
            entry
        })
        .collect();

    let mut output = serde_json::to_string_pretty(&results).unwrap_or_default();
    if !include_schemas {
        output.push_str(&format!(
            "\n\n{} tools matched; filter by server or add a query to see parameter schemas.",
            matched.len()
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> McpToolDef {
        McpToolDef {
            name: name.to_string(),
            description: Some(format!("does {name}")),
            input_schema: json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"],
            }),
        }
    }

    fn servers(counts: &[(&str, usize)]) -> HashMap<Host, Vec<McpToolDef>> {
        counts
            .iter()
            .map(|(server, n)| {
                let tools = (0..*n).map(|i| tool(&format!("t{i}"))).collect();
                (server.to_string(), tools)
            })
            .collect()
    }

    #[test]
    fn narrow_search_includes_input_schema() {
        let all = servers(&[("cloudflare", 3), ("notion", 40)]);
        let out = render_search(&all, Some("cloudflare"), None);
        let parsed: Vec<Value> = serde_json::from_str(&out).expect("valid JSON array");
        assert_eq!(parsed.len(), 3);
        for entry in &parsed {
            assert_eq!(entry["input_schema"]["required"][0], json!("code"));
        }
        // The filtered-out server's fat tool list must not leak in.
        assert!(!out.contains("notion"));
    }

    #[test]
    fn broad_search_omits_schema_and_hints() {
        let all = servers(&[("notion", 40)]);
        let out = render_search(&all, None, None);
        // Above the threshold: trailing note, not a bare JSON array.
        assert!(out.contains("filter by server or add a query"));
        let json_part = out.split("\n\n").next().unwrap();
        let parsed: Vec<Value> = serde_json::from_str(json_part).expect("valid JSON array");
        assert_eq!(parsed.len(), 40);
        assert!(parsed.iter().all(|e| e.get("input_schema").is_none()));
    }

    #[test]
    fn threshold_boundary_includes_schema() {
        let all = servers(&[("srv", SCHEMA_THRESHOLD)]);
        let out = render_search(&all, None, None);
        assert!(!out.contains("filter by server or add a query"));
        let parsed: Vec<Value> = serde_json::from_str(&out).expect("valid JSON array");
        assert!(parsed.iter().all(|e| e.get("input_schema").is_some()));
    }

    #[test]
    fn no_match_lists_connected_servers() {
        let all = servers(&[("cloudflare", 3), ("notion", 2)]);
        let out = render_search(&all, None, Some("nonexistent-xyz"));
        assert!(out.starts_with("No matching tools found."));
        assert!(out.contains("cloudflare"));
        assert!(out.contains("notion"));
    }
}
