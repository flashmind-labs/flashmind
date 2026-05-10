//! Google OAuth authentication tool.
//!
//! Presents the user with an authorization URL, accepts the code back,
//! exchanges it for tokens, and registers the service tools dynamically.
//!
//! Reads credentials lazily at execute time so the tool can be registered
//! even before the credentials file exists on disk.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult};

use super::auth::{self, Credentials, OAuthClientCredentials};
use super::client::GoogleConfig;
use crate::oauth;

use crate::builder::PendingTools;

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Auth tool that handles the Google OAuth2 authorization code flow.
///
/// Registered by the builder when no valid cached token exists. Reads
/// credentials from disk at execute time, so it can be registered even
/// before the user has created a credentials file.
pub struct GoogleAuthTool {
    pub config: GoogleConfig,
    pub scopes: Vec<&'static str>,
    pub readonly: bool,
    pub pending_tools: PendingTools,
}

#[derive(Deserialize)]
struct AuthArgs {
    code: Option<String>,
}

impl GoogleAuthTool {
    fn load_credentials(&self) -> Result<OAuthClientCredentials> {
        let data = std::fs::read_to_string(&self.config.credentials_path).with_context(|| {
            format!(
                "Google credentials file not found at {}.\n\n\
                 To set up Google API access:\n\
                 1. Go to https://console.cloud.google.com/apis/credentials\n\
                 2. Create an OAuth 2.0 Client ID (Desktop app)\n\
                 3. Download the JSON and save it to: {}",
                self.config.credentials_path.display(),
                self.config.credentials_path.display(),
            )
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&data).context("invalid credentials JSON")?;

        match Credentials::from_json(&json, None)? {
            Credentials::UserOAuth(creds) => Ok(creds),
            Credentials::ServiceAccount { .. } => {
                bail!(
                    "Service account credentials don't need interactive auth. \
                     Remove the google_auth tool or switch to OAuth client credentials."
                )
            }
        }
    }
}

#[async_trait]
impl Tool for GoogleAuthTool {
    fn name(&self) -> &str {
        "google_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with Google APIs. Call without arguments to get the \
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

        let credentials = self.load_credentials()?;
        let combined_scope = self.scopes.join(" ");

        match args.code {
            None => {
                let url = auth::auth_url(&credentials, &combined_scope);
                Ok(ToolResult::interrupt(
                    ctx.tool_call_id,
                    format!(
                        "Please visit this URL to authorize Google access:\n\n{url}\n\n\
                         After authorizing, you'll receive a code. Call this tool again \
                         with that code to complete authentication."
                    ),
                ))
            }
            Some(code) => {
                let token = auth::exchange_code(&credentials, &code)
                    .await
                    .context("failed to exchange authorization code")?;

                oauth::save_token(&self.config.token_path, &token)?;
                info!(
                    "Google OAuth token saved to {}",
                    self.config.token_path.display()
                );

                self.register_service_tools()?;

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    "Authentication successful. Google tools are now available.".to_string(),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("code").is_some() {
            "Completing Google OAuth authentication".to_string()
        } else {
            "Getting Google OAuth authorization URL".to_string()
        }
    }
}

impl GoogleAuthTool {
    fn register_service_tools(&self) -> Result<()> {
        let tools = self.build_service_tools()?;
        let mut pending = self.pending_tools.lock().unwrap();
        pending.extend(tools);
        Ok(())
    }

    fn build_service_tools(&self) -> Result<Vec<Arc<dyn Tool>>> {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

        #[cfg(feature = "gmail")]
        {
            use super::gmail::{self, tools::*};

            let client = Arc::new(gmail::new_client(&self.config)?);

            tools.push(Arc::new(GmailSearchThreadsTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GmailGetThreadTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GmailListDraftsTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GmailListLabelsTool {
                client: client.clone(),
            }));

            if !self.readonly {
                tools.push(Arc::new(GmailSendTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailCreateDraftTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailCreateLabelTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailLabelMessageTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailUnlabelMessageTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailLabelThreadTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GmailUnlabelThreadTool {
                    client: client.clone(),
                }));
            }
        }

        #[cfg(feature = "google-calendar")]
        {
            use super::calendar::{self, tools::*};

            let client = Arc::new(calendar::new_client(&self.config)?);

            tools.push(Arc::new(GcalListCalendarsTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GcalListEventsTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GcalGetEventTool {
                client: client.clone(),
            }));

            if !self.readonly {
                tools.push(Arc::new(GcalCreateEventTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GcalUpdateEventTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GcalDeleteEventTool {
                    client: client.clone(),
                }));
            }
        }

        #[cfg(feature = "google-contacts")]
        {
            use super::contacts::{self, tools::*};

            let client = Arc::new(contacts::new_client(&self.config)?);

            tools.push(Arc::new(GcontactsListTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GcontactsSearchTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GcontactsGetTool {
                client: client.clone(),
            }));

            if !self.readonly {
                tools.push(Arc::new(GcontactsCreateTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GcontactsUpdateTool {
                    client: client.clone(),
                }));
                tools.push(Arc::new(GcontactsDeleteTool {
                    client: client.clone(),
                }));
            }
        }

        Ok(tools)
    }
}
