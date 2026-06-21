use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde_json::{Value, json};

use flashmind_types::tool::{InterruptPayload, Tool, ToolContext, ToolResult};

use crate::mcp::registry::McpRegistry;
use crate::mcp::types::McpToolDef;

// ---------------------------------------------------------------------------
// McpToolApproval — typed interrupt payload for MCP tool permission prompts
// ---------------------------------------------------------------------------

/// Interrupt payload emitted when a restricted MCP tool is called without
/// prior session approval.
///
/// The CLI downcasts this to show a choice picker (allow once / allow for
/// session / deny).
#[derive(Debug)]
pub struct McpToolApproval {
    /// MCP server name (e.g. `"gmail"`).
    pub server_name: String,
    /// Tool name as reported by the MCP server (e.g. `"send_email"`).
    pub tool_name: String,
    /// Fully-qualified tool name (`"{server}_{tool}"`).
    pub full_name: String,
    /// Arguments the agent passed to the tool call.
    pub args: Value,
}

impl InterruptPayload for McpToolApproval {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn display_output(&self) -> String {
        format!(
            "MCP tool '{}/{}' requires approval. The user will be prompted.",
            self.server_name, self.tool_name
        )
    }
}

/// Adapter that presents a remote MCP server tool as a local [`Tool`].
///
/// When an MCP server connects, each of its tools (described by [`McpToolDef`])
/// is wrapped in an `McpToolWrapper` instance and registered into the agent's
/// [`ToolRegistry`](flashmind_types::tool::ToolRegistry).  This lets the LLM
/// invoke remote tools through the same function-calling path it uses for
/// built-in tools like `grep` or `file_read`.
///
/// # Naming convention
///
/// Every wrapper prefixes the original tool name with the MCP server name and
/// an underscore so that tools from different servers never collide:
///
/// ```text
///   server "gmail", tool "search"  →  registered as "gmail_search"
///   server "github", tool "search" →  registered as "github_search"
/// ```
///
/// # Construction
///
/// Wrappers are created in bulk by [`make_mcp_tool_wrappers`], which converts
/// a slice of [`McpToolDef`]s into `Vec<Arc<dyn Tool>>`.  Individual instances
/// are rarely constructed by hand.
///
/// # Execution flow
///
/// 1. The agent calls the tool via the normal function-calling mechanism.
/// 2. [`execute`](Self::execute) forwards arguments to the remote server
///    through [`McpRegistry::call_tool`].
/// 3. Text content items are joined with newlines and returned as the result.
/// 4. If the server flags the response as an error, a [`ToolResult::Failure`]
///    is returned instead.
///
/// # Output limits
///
/// Because MCP servers can return arbitrary payloads, this wrapper returns
/// `usize::MAX` for both [`max_output_bytes`](Self::max_output_bytes) and
/// [`max_output_lines`](Self::max_output_lines), effectively disabling output
/// truncation for wrapped tools.
pub struct McpToolWrapper {
    /// Handle to the MCP registry that manages server connections.
    pub mcp: McpRegistry,
    /// Logical name of the MCP server this tool belongs to (e.g. `"gmail"`).
    pub server_name: String,
    /// Original tool name as reported by the MCP server (e.g. `"search"`).
    pub tool_name: String,
    /// Fully-qualified tool name used for registration (`"{server}_{tool}"`).
    pub full_name: String,
    /// Human-readable description taken from the server's tool definition.
    pub tool_description: String,
    /// JSON Schema describing the tool's input parameters, forwarded from the
    /// MCP server's [`McpToolDef::input_schema`].
    pub tool_schema: Value,
    /// Tool names that require approval (loaded from config at connection time).
    pub restricted: Arc<HashSet<String>>,
    /// Tools approved by the user for this session (shared across all wrappers
    /// for the same server).
    pub session_approved: Arc<RwLock<HashSet<String>>>,
}

/// Create [`McpToolWrapper`] instances for every tool in `tool_defs`.
///
/// Each wrapper is namespaced with `server_name` so the resulting tool names
/// are unique within the agent's tool registry.  The returned vectors contain
/// `Arc<dyn Tool>` handles ready to be registered.
///
/// # Arguments
///
/// * `mcp` — The [`McpRegistry`] that will handle actual RPC calls to the
///   remote server.  Cloned into each wrapper so they remain valid after the
///   caller drops its reference.
/// * `server_name` — Logical identifier for the MCP server (used as a name
///   prefix for every generated tool).
/// * `tool_defs` — Tool definitions received from the MCP server via
///   [`list_tools`](rmcp::model::Tool).
///
/// # Returns
///
/// A `Vec<Arc<dyn Tool>>` — one wrapper per input definition, each ready to be
/// registered with [`ToolRegistry::register`](flashmind_types::tool::ToolRegistry::register).
///
/// # Example
///
/// ```ignore
/// let restricted = Arc::new(HashSet::new());
/// let session_approved = Arc::new(RwLock::new(HashSet::new()));
/// let wrappers = make_mcp_tool_wrappers(&registry, "gmail", &tool_defs, restricted, session_approved);
/// for wrapper in wrappers {
///     tool_registry.register(wrapper);
/// }
/// ```
pub fn make_mcp_tool_wrappers(
    mcp: &McpRegistry,
    server_name: &str,
    tool_defs: &[McpToolDef],
    restricted: Arc<HashSet<String>>,
    session_approved: Arc<RwLock<HashSet<String>>>,
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
                restricted: restricted.clone(),
                session_approved: session_approved.clone(),
            }) as Arc<dyn Tool>
        })
        .collect()
}

#[async_trait]
impl Tool for McpToolWrapper {
    /// Returns the fully-qualified tool name (`"{server}_{tool}"`).
    ///
    /// This is the name the LLM sees in function definitions and uses to
    /// invoke the tool.
    fn name(&self) -> &str {
        &self.full_name
    }

    /// Returns the tool's description as provided by the MCP server.
    ///
    /// If the server did not supply a description, an empty string is returned.
    fn description(&self) -> &str {
        &self.tool_description
    }

    /// Returns the tool's JSON Schema parameter definition.
    ///
    /// This is the `inputSchema` field from the MCP server's tool definition,
    /// cloned on each call so the caller owns the returned [`Value`].
    fn parameters(&self) -> Value {
        self.tool_schema.clone()
    }

    /// Execute the wrapped tool on the remote MCP server.
    ///
    /// Forwards `ctx.args` to the server via [`McpRegistry::call_tool`].  If
    /// the arguments are `null`, an empty JSON object is sent instead.
    ///
    /// The response text content items are joined with newline separators and
    /// returned as a single string.  If the server marks the response as an
    /// error, a [`ToolResult::Failure`] is produced; otherwise
    /// [`ToolResult::Success`] is returned.
    ///
    /// Network-level errors from [`McpRegistry::call_tool`] are caught and
    /// converted into a [`ToolResult::Failure`] with a formatted message so
    /// that `execute` always returns `Ok(...)`.
    ///
    /// # Errors
    ///
    /// This method itself does not propagate errors — failures are encoded as
    /// [`ToolResult::Failure`] variants inside an `Ok`.  The outer
    /// `anyhow::Result` is only used for unexpected panics or allocation
    /// failures.
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if self.restricted.contains(&self.tool_name)
            && !self.session_approved.read().unwrap().contains(&self.tool_name)
        {
            return Ok(ToolResult::interrupt(
                ctx.tool_call_id,
                Arc::new(McpToolApproval {
                    server_name: self.server_name.clone(),
                    tool_name: self.tool_name.clone(),
                    full_name: self.full_name.clone(),
                    args: ctx.args.clone(),
                }),
            ));
        }

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

    /// Returns `usize::MAX` — no output size limit.
    ///
    /// MCP servers may return arbitrarily large responses, so the wrapper
    /// opts out of the default 256 KiB truncation.
    fn max_output_bytes(&self) -> usize {
        usize::MAX
    }

    /// Returns `usize::MAX` — no output line limit.
    ///
    /// MCP servers may return arbitrarily large responses, so the wrapper
    /// opts out of the default 10,000-line truncation.
    fn max_output_lines(&self) -> usize {
        usize::MAX
    }

    /// Generate a human-readable summary of the tool call for display.
    ///
    /// Produces a string of the form `[mcp:{server}] {tool}` followed by a
    /// parenthesised summary of up to four argument key-value pairs:
    ///
    /// ```text
    /// [mcp:gmail] search (query="unread emails", limit=10, +2 more)
    /// ```
    ///
    /// Long string values (more than 60 characters) are truncated with an
    /// ellipsis.  Arrays and objects show their length instead of contents.
    ///
    /// If `args` is not a JSON object or is empty, only the base label is
    /// returned.
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
    use std::collections::HashSet;
    use std::sync::RwLock;

    use tokio_util::sync::CancellationToken;

    use super::*;

    struct InMemoryProvider;

    #[async_trait]
    impl crate::mcp::config::McpConfigProvider for InMemoryProvider {
        async fn list_configs(&self) -> anyhow::Result<Vec<crate::mcp::config::McpServerConfig>> {
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

    fn make_registry() -> McpRegistry {
        let provider: Arc<dyn crate::mcp::config::McpConfigProvider> = Arc::new(InMemoryProvider);
        McpRegistry::new(provider, None)
    }

    #[test]
    fn test_make_mcp_tool_wrappers() {
        let registry = make_registry();

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

        let restricted = Arc::new(HashSet::new());
        let session_approved = Arc::new(RwLock::new(HashSet::new()));
        let wrappers =
            make_mcp_tool_wrappers(&registry, "gmail", &tool_defs, restricted, session_approved);
        assert_eq!(wrappers.len(), 2);
        assert_eq!(wrappers[0].name(), "gmail_search_emails");
        assert_eq!(wrappers[0].description(), "Search emails by query");
        assert_eq!(wrappers[1].name(), "gmail_send_email");
        assert_eq!(wrappers[1].description(), "");

        let params = wrappers[0].parameters();
        assert_eq!(params["properties"]["query"]["type"], "string");
    }

    #[tokio::test]
    async fn test_restricted_tool_returns_interrupt() {
        let registry = make_registry();

        let restricted: Arc<HashSet<String>> =
            Arc::new(["send_email".to_string()].into_iter().collect());
        let session_approved: Arc<RwLock<HashSet<String>>> =
            Arc::new(RwLock::new(HashSet::new()));

        let tool_defs = vec![McpToolDef {
            name: "send_email".into(),
            description: Some("Send an email".into()),
            input_schema: json!({"type": "object"}),
        }];

        let wrappers =
            make_mcp_tool_wrappers(&registry, "gmail", &tool_defs, restricted, session_approved);

        let cancel = CancellationToken::new();
        let ctx = ToolContext::new("call_1", json!({"to": "test@example.com"}), None, &cancel);
        let result = wrappers[0].execute(ctx).await.unwrap();
        assert!(result.is_interrupt());

        let payload = result.payload().unwrap();
        let approval = payload.as_any().downcast_ref::<McpToolApproval>().unwrap();
        assert_eq!(approval.server_name, "gmail");
        assert_eq!(approval.tool_name, "send_email");
        assert_eq!(approval.full_name, "gmail_send_email");
        assert_eq!(approval.args, json!({"to": "test@example.com"}));
    }

    #[tokio::test]
    async fn test_session_approved_tool_passes() {
        let registry = make_registry();

        let restricted: Arc<HashSet<String>> =
            Arc::new(["send_email".to_string()].into_iter().collect());
        let session_approved: Arc<RwLock<HashSet<String>>> =
            Arc::new(RwLock::new(["send_email".to_string()].into_iter().collect()));

        let tool_defs = vec![McpToolDef {
            name: "send_email".into(),
            description: Some("Send an email".into()),
            input_schema: json!({"type": "object"}),
        }];

        let wrappers =
            make_mcp_tool_wrappers(&registry, "gmail", &tool_defs, restricted, session_approved);

        // This will try to call the MCP server (which isn't connected), so it will
        // return a failure — but the important thing is it does NOT return an interrupt.
        let cancel = CancellationToken::new();
        let ctx = ToolContext::new("call_2", json!({}), None, &cancel);
        let result = wrappers[0].execute(ctx).await.unwrap();
        assert!(!result.is_interrupt());
    }
}
