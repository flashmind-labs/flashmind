//! Outlook OAuth authentication tool.
//!
//! Presents the user with a Microsoft authorization URL, accepts the code back,
//! exchanges it for tokens, and registers the service tools dynamically.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{InterruptPayload, ToolContext, ToolResult};

use super::OutlookConfig;
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

/// Auth tool that handles the Microsoft OAuth2 authorization code flow.
///
/// Only registered when no valid cached token exists. Credentials are
/// provided inline via `OutlookConfig`.
pub struct OutlookAuthTool {
    pub config: OutlookConfig,
    pub pending_tools: PendingTools,
}

#[derive(Deserialize)]
struct AuthArgs {
    code: Option<String>,
}

#[async_trait]
impl Tool for OutlookAuthTool {
    fn name(&self) -> &str {
        "outlook_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with Microsoft Outlook. Call without arguments to get the \
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
                let url = auth::auth_url(&self.config.credentials);
                Ok(ToolResult::interrupt(
                    ctx.tool_call_id,
                    Arc::new(OAuthInterrupt {
                        message: format!(
                            "Please visit this URL to authorize Outlook access:\n\n{url}\n\n\
                             After authorizing, you'll be redirected. Copy the `code` parameter \
                             from the redirect URL and call this tool again with that code."
                        ),
                    }),
                ))
            }
            Some(code) => {
                let token = auth::exchange_code(&self.config.credentials, &code)
                    .await
                    .context("failed to exchange authorization code")?;

                oauth::save_token(&self.config.token_path, &token)?;
                info!(
                    "Outlook OAuth token saved to {}",
                    self.config.token_path.display()
                );

                self.register_service_tools()?;

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    "Authentication successful. Outlook tools are now available.".to_string(),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("code").is_some() {
            "Completing Outlook OAuth authentication".to_string()
        } else {
            "Getting Outlook OAuth authorization URL".to_string()
        }
    }
}

impl OutlookAuthTool {
    fn register_service_tools(&self) -> Result<()> {
        use super::OutlookClient;
        use super::calendar::tools::*;
        use super::contacts::tools::*;
        use super::mail::tools::*;

        let readonly = self.config.readonly;

        let mut scopes: Vec<&str> = vec!["offline_access"];
        if readonly {
            scopes.extend_from_slice(&["Mail.Read", "Calendars.Read", "Contacts.Read"]);
        } else {
            scopes.extend_from_slice(&[
                "Mail.Read",
                "Mail.Send",
                "Calendars.ReadWrite",
                "Contacts.Read",
                "Contacts.ReadWrite",
            ]);
        }

        let client = Arc::new(OutlookClient::new(self.config.clone(), &scopes)?);

        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(OutlookListMessagesTool {
                client: client.clone(),
            }),
            Arc::new(OutlookGetMessageTool {
                client: client.clone(),
            }),
            Arc::new(OutlookListFoldersTool {
                client: client.clone(),
            }),
            Arc::new(OutlookListEventsTool {
                client: client.clone(),
            }),
            Arc::new(OutlookGetEventTool {
                client: client.clone(),
            }),
            Arc::new(OutlookListContactsTool {
                client: client.clone(),
            }),
            Arc::new(OutlookGetContactTool {
                client: client.clone(),
            }),
        ];

        if !readonly {
            // Mail (write)
            tools.push(Arc::new(OutlookSendMailTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(OutlookCreateDraftTool {
                client: client.clone(),
            }));

            // Calendar (write)
            tools.push(Arc::new(OutlookCreateEventTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(OutlookUpdateEventTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(OutlookDeleteEventTool {
                client: client.clone(),
            }));

            // Contacts (write)
            tools.push(Arc::new(OutlookCreateContactTool {
                client: client.clone(),
            }));
        }

        let mut pending = self.pending_tools.lock().unwrap();
        pending.extend(tools);
        Ok(())
    }
}
