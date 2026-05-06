use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;
use crate::mcp::types::McpToolDef;

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

    fn humanize(&self, args: &Value) -> String {
        let base = format!("[mcp:{}] {}", self.server_name, self.tool_name);
        let Some(obj) = args.as_object() else {
            return base;
        };
        if obj.is_empty() {
            return base;
        }
        let summary: Vec<String> = obj
            .iter()
            .take(4)
            .map(|(k, v)| {
                let val = match v {
                    Value::String(s) => {
                        if s.chars().count() > 60 {
                            let truncated: String = s.chars().take(57).collect();
                            format!("\"{truncated}…\"")
                        } else {
                            format!("\"{s}\"")
                        }
                    }
                    Value::Array(a) => format!("[{} items]", a.len()),
                    Value::Object(o) => format!("{{{} keys}}", o.len()),
                    other => other.to_string(),
                };
                format!("{k}={val}")
            })
            .collect();
        let extra = if obj.len() > 4 {
            format!(", +{} more", obj.len() - 4)
        } else {
            String::new()
        };
        format!("{base} ({}{})", summary.join(", "), extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_mcp_tool_wrappers() {
        struct InMemoryProvider;

        #[async_trait]
        impl crate::mcp::config::McpConfigProvider for InMemoryProvider {
            async fn list_configs(
                &self,
            ) -> anyhow::Result<Vec<crate::mcp::config::McpServerConfig>> {
                Ok(vec![])
            }
            async fn save_config(
                &self,
                _config: &crate::mcp::config::McpServerConfig,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            async fn delete_config(&self, _name: &str) -> anyhow::Result<()> {
                Ok(())
            }
            async fn save_credentials(
                &self,
                _name: &str,
                _credentials: &Value,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            async fn load_credentials(&self, _name: &str) -> anyhow::Result<Option<Value>> {
                Ok(None)
            }
        }

        let provider: Arc<dyn crate::mcp::config::McpConfigProvider> = Arc::new(InMemoryProvider);
        let registry = McpRegistry::new(provider, None);

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
