//! CalDAV OAuth authentication tool.
//!
//! Presents the user with a provider authorization URL, accepts the code back,
//! exchanges it for tokens, and registers the CalDAV tools dynamically.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{InterruptPayload, ToolContext, ToolResult};

use super::CalDavConfig;
use super::auth;
use crate::oauth;

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

use crate::builder::PendingTools;

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Auth tool that handles the CalDAV OAuth2 authorization code flow.
///
/// Only registered when `CalDavAuth::OAuth` is used. Credentials are
/// provided inline via `CalDavConfig`.
pub struct CalDavAuthTool {
    /// CalDAV configuration including OAuth credentials.
    pub config: CalDavConfig,
    /// When `true`, only read tools are registered after auth.
    pub readonly: bool,
    /// Queue for dynamically registering tools after auth completes.
    pub pending_tools: PendingTools,
}

#[derive(Deserialize)]
struct AuthArgs {
    code: Option<String>,
}

#[async_trait]
impl Tool for CalDavAuthTool {
    fn name(&self) -> &str {
        "caldav_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with a CalDAV server via OAuth. Call without arguments to get the \
         authorization URL, then call again with the code to complete sign-in."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Authorization code from the OAuth redirect (omit to get the auth URL)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: AuthArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        match args.code {
            None => {
                let creds = match &self.config.auth {
                    super::CalDavAuth::OAuth { credentials } => credentials,
                    _ => anyhow::bail!("caldav_auth tool requires OAuth auth mode"),
                };
                let url = auth::auth_url(creds);
                Ok(ToolResult::interrupt(
                    ctx.tool_call_id,
                    Arc::new(OAuthInterrupt {
                        message: format!(
                            "Please visit this URL to authorize CalDAV access:\n\n{url}\n\n\
                             After authorizing, you'll be redirected. Copy the `code` parameter \
                             from the redirect URL and call this tool again with that code."
                        ),
                    }),
                ))
            }
            Some(code) => {
                let creds = match &self.config.auth {
                    super::CalDavAuth::OAuth { credentials } => credentials,
                    _ => anyhow::bail!("caldav_auth tool requires OAuth auth mode"),
                };

                let token = auth::exchange_code(creds, &code)
                    .await
                    .context("failed to exchange CalDAV authorization code")?;

                if let Some(path) = &self.config.token_path {
                    oauth::save_token(path, &token)?;
                    info!("CalDAV OAuth token saved to {}", path.display());
                }

                self.register_service_tools()?;

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    "Authentication successful. CalDAV tools are now available.".to_string(),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("code").is_some() {
            "Completing CalDAV OAuth authentication".to_string()
        } else {
            "Getting CalDAV OAuth authorization URL".to_string()
        }
    }
}

impl CalDavAuthTool {
    fn register_service_tools(&self) -> Result<()> {
        use super::CalDavClient;
        use super::tools::*;

        let client = Arc::new(CalDavClient::new(&self.config)?);
        let readonly = self.readonly;

        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(CalDavListCalendarsTool {
                client: client.clone(),
            }),
            Arc::new(CalDavListEventsTool {
                client: client.clone(),
            }),
            Arc::new(CalDavGetEventTool {
                client: client.clone(),
            }),
            Arc::new(CalDavSearchEventsTool {
                client: client.clone(),
            }),
        ];

        if !readonly {
            tools.push(Arc::new(CalDavCreateEventTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CalDavUpdateEventTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CalDavDeleteEventTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CalDavCreateCalendarTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CalDavDeleteCalendarTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CalDavRenameCalendarTool {
                client: client.clone(),
            }));
        }

        let mut pending = self.pending_tools.lock().unwrap();
        pending.extend(tools);
        Ok(())
    }
}
