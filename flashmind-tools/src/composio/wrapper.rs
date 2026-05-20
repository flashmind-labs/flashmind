//! Tool wrapper that presents a remote Composio tool as a local [`Tool`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::client::ComposioClient;
use super::types::ComposioToolDef;

/// Adapter that presents a remote Composio tool as a local [`Tool`].
///
/// Each tool fetched from `GET /tools` is wrapped in a `ComposioToolWrapper`
/// and registered into the agent's tool registry.  Execution forwards
/// arguments to `POST /tools/execute/{slug}` via the [`ComposioClient`].
///
/// Tool names are prefixed with `composio_` and lowercased to avoid collision
/// with native tools (e.g. `GITHUB_CREATE_ISSUE` → `composio_github_create_issue`).
pub struct ComposioToolWrapper {
    /// HTTP client for the Composio API.
    pub client: Arc<ComposioClient>,
    /// Original Composio tool slug (e.g. `"GITHUB_CREATE_ISSUE"`).
    pub slug: String,
    /// Fully-qualified tool name preserving original case (e.g. `"composio_GITHUB_CREATE_ISSUE"`).
    pub full_name: String,
    /// Lowercased version of `full_name` returned by the `Tool` trait.
    lowered_name: String,
    /// Human-readable description for the LLM.
    pub tool_description: String,
    /// JSON Schema for the tool's input parameters.
    pub tool_schema: Value,
    /// Toolkit this tool belongs to (e.g. `"github"`).
    pub toolkit_slug: String,
}

/// Create [`ComposioToolWrapper`] instances for every tool in `tool_defs`.
pub fn make_composio_tool_wrappers(
    client: &Arc<ComposioClient>,
    tool_defs: &[ComposioToolDef],
) -> Vec<Arc<dyn Tool>> {
    tool_defs
        .iter()
        .map(|t| {
            let full_name = t.slug.clone();
            let lowered_name = full_name.to_lowercase();
            Arc::new(ComposioToolWrapper {
                client: Arc::clone(client),
                slug: t.slug.clone(),
                full_name,
                lowered_name,
                tool_description: t
                    .description
                    .clone()
                    .or_else(|| t.display_name.clone())
                    .unwrap_or_default(),
                tool_schema: t.input_schema.clone(),
                toolkit_slug: t.toolkit_slug.clone().unwrap_or_default(),
            }) as Arc<dyn Tool>
        })
        .collect()
}

#[async_trait]
impl Tool for ComposioToolWrapper {
    fn name(&self) -> &str {
        &self.lowered_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters(&self) -> Value {
        self.tool_schema.clone()
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let arguments = if ctx.args.is_null() {
            serde_json::json!({})
        } else {
            ctx.args.clone()
        };

        let api_key = self.client.api_key();
        let redact = |s: String| -> String {
            if !api_key.is_empty() && s.contains(api_key) {
                s.replace(api_key, "[REDACTED]")
            } else {
                s
            }
        };

        match self.client.execute_tool(&self.slug, arguments).await {
            Ok(resp) => {
                if resp.successful == Some(false) {
                    let msg = resp.error.unwrap_or_else(|| "tool execution failed".into());
                    Ok(ToolResult::failure(ctx.tool_call_id, redact(msg)))
                } else if let Some(err) = resp.error {
                    Ok(ToolResult::failure(ctx.tool_call_id, redact(err)))
                } else {
                    let output = if resp.data.is_string() {
                        resp.data.as_str().unwrap_or_default().to_string()
                    } else {
                        serde_json::to_string_pretty(&resp.data).unwrap_or_default()
                    };
                    Ok(ToolResult::success(ctx.tool_call_id, redact(output)))
                }
            }
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                redact(format!("Composio error: {e:#}")),
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
        let tool_name = self.slug.to_lowercase();
        let base = format!("[composio:{}] {}", self.toolkit_slug, tool_name);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_defs() -> Vec<ComposioToolDef> {
        vec![
            ComposioToolDef {
                slug: "GITHUB_CREATE_ISSUE".into(),
                display_name: Some("Create Issue".into()),
                description: Some("Create a new GitHub issue".into()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "owner": { "type": "string" },
                        "repo": { "type": "string" },
                        "title": { "type": "string" }
                    },
                    "required": ["owner", "repo", "title"]
                }),
                toolkit_slug: Some("github".into()),
            },
            ComposioToolDef {
                slug: "SLACK_SEND_MESSAGE".into(),
                display_name: None,
                description: None,
                input_schema: json!({"type": "object"}),
                toolkit_slug: Some("slack".into()),
            },
        ]
    }

    #[test]
    fn wrapper_naming() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        assert_eq!(wrappers.len(), 2);
        assert_eq!(wrappers[0].name(), "github_create_issue");
        assert_eq!(wrappers[1].name(), "slack_send_message");
    }

    #[test]
    fn wrapper_description_fallback() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        assert_eq!(wrappers[0].description(), "Create a new GitHub issue");
        assert_eq!(wrappers[1].description(), "");
    }

    #[test]
    fn wrapper_schema_passthrough() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        let params = wrappers[0].parameters();
        assert_eq!(params["properties"]["owner"]["type"], "string");
        assert_eq!(params["required"], json!(["owner", "repo", "title"]));
    }

    #[test]
    fn humanize_with_args() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        let args = json!({"owner": "org", "repo": "myrepo", "title": "Bug report"});
        let h = wrappers[0].humanize(&args);
        assert!(h.starts_with("[composio:github]"));
        assert!(h.contains("owner=\"org\""));
        assert!(h.contains("title=\"Bug report\""));
    }

    #[test]
    fn humanize_empty_args() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        let h = wrappers[0].humanize(&json!({}));
        assert_eq!(h, "[composio:github] github_create_issue");
    }

    #[test]
    fn humanize_truncates_long_strings() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        let long = "a".repeat(100);
        let args = json!({"title": long});
        let h = wrappers[0].humanize(&args);
        assert!(h.contains("…"));
    }

    #[test]
    fn humanize_extra_args() {
        let client = Arc::new(ComposioClient::new("key".into(), None, None));
        let wrappers = make_composio_tool_wrappers(&client, &sample_defs());

        let args = json!({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5});
        let h = wrappers[0].humanize(&args);
        assert!(h.contains("+1 more"));
    }
}
