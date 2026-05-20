//! Per-toolkit Composio OAuth authentication tools.
//!
//! For each toolkit in the whitelist, a `{slug}_auth` tool is registered.
//! Phase 1 (no `session_id`) creates a session and returns the OAuth URL.
//! Phase 2 (with `session_id`) polls the session, and on success registers
//! the toolkit's service tools via [`PendingTools`].

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{InterruptPayload, ToolContext, ToolResult};

use super::client::ComposioClient;
use super::types::SessionToolkits;
use super::wrapper::make_composio_tool_wrappers;
use crate::builder::PendingTools;

// ---------------------------------------------------------------------------

#[derive(Debug)]
struct OAuthInterrupt {
    message: String,
}

impl InterruptPayload for OAuthInterrupt {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn display_output(&self) -> String {
        self.message.clone()
    }
}

// ---------------------------------------------------------------------------

/// Auth tool for a single Composio toolkit (e.g. `github_auth`, `slack_auth`).
///
/// Registered when `ComposioConfig` has toolkits listed but no
/// `connected_account_id`. After the user completes OAuth, the toolkit's
/// service tools are pushed into [`PendingTools`].
pub struct ComposioAuthTool {
    client: Arc<ComposioClient>,
    toolkit: String,
    pending_tools: PendingTools,
    tool_name: String,
    tool_description: String,
}

impl ComposioAuthTool {
    pub fn new(client: Arc<ComposioClient>, toolkit: String, pending_tools: PendingTools) -> Self {
        let tool_name = format!("{toolkit}_auth");
        let tool_description = format!(
            "Authenticate with {toolkit} via Composio. Call without arguments to get the \
             authorization URL, then call again with the session_id to complete sign-in."
        );
        Self {
            client,
            toolkit,
            pending_tools,
            tool_name,
            tool_description,
        }
    }
}

#[derive(Deserialize)]
struct AuthArgs {
    session_id: Option<String>,
}

#[async_trait]
impl Tool for ComposioAuthTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID from the first call (omit to start OAuth)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: AuthArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        match args.session_id {
            None => {
                let session = self
                    .client
                    .create_session(
                        &format!("agent_{}", self.toolkit),
                        Some(SessionToolkits {
                            enable: Some(vec![self.toolkit.clone()]),
                            disable: None,
                        }),
                        None,
                    )
                    .await
                    .context("failed to create Composio session")?;

                let url = session
                    .connection_urls
                    .get(&self.toolkit)
                    .cloned()
                    .unwrap_or_else(|| format!("(no OAuth URL returned for {})", self.toolkit));

                Ok(ToolResult::interrupt(
                    ctx.tool_call_id,
                    Arc::new(OAuthInterrupt {
                        message: format!(
                            "Please visit this URL to connect {toolkit}:\n\n{url}\n\n\
                             After authorizing, call this tool again with \
                             session_id=\"{session_id}\".",
                            toolkit = self.toolkit,
                            session_id = session.session_id,
                        ),
                    }),
                ))
            }
            Some(session_id) => {
                let session = self
                    .client
                    .get_session(&session_id)
                    .await
                    .context("failed to poll Composio session")?;

                let account_ids = session.connected_accounts.get(&self.toolkit);
                let Some(ids) = account_ids.filter(|v| !v.is_empty()) else {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!(
                            "{} is not connected yet. Ask the user to complete \
                             authorization, then try again with the same session_id.",
                            self.toolkit
                        ),
                    ));
                };

                let account_id = &ids[0];

                let scoped_client = Arc::new(ComposioClient::new(
                    self.client.api_key().into(),
                    Some(account_id.clone()),
                    None,
                ));

                let tools = self
                    .client
                    .list_session_tools(&session_id)
                    .await
                    .context("failed to fetch session tools after auth")?;

                let wrappers = make_composio_tool_wrappers(&scoped_client, &tools);
                let count = wrappers.len();

                let mut pending = self.pending_tools.lock().unwrap();
                pending.extend(wrappers);
                drop(pending);

                info!(toolkit = %self.toolkit, count, account_id, "registered Composio tools after auth");

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!(
                        "{} authenticated successfully. {count} tools are now available.",
                        self.toolkit
                    ),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("session_id").is_some() {
            format!("Completing {} OAuth authentication", self.toolkit)
        } else {
            format!("Getting {} OAuth authorization URL", self.toolkit)
        }
    }
}
