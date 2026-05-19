//! GitHub OAuth authentication tool.
//!
//! Presents the user with a GitHub authorization URL, accepts the code back,
//! exchanges it for a token, and registers the service tools dynamically.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{InterruptPayload, ToolContext, ToolResult};

use super::GitHubConfig;
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

/// Auth tool that handles the GitHub OAuth2 authorization code flow.
///
/// Only registered when no valid cached token exists. Credentials are
/// provided inline via `GitHubConfig`.
pub struct GitHubAuthTool {
    /// Configuration including credentials and token path.
    pub config: GitHubConfig,
    /// When `true`, only read tools are registered after auth.
    pub readonly: bool,
    /// Shared queue for dynamically registering tools after auth completes.
    pub pending_tools: PendingTools,
}

#[derive(Deserialize)]
struct AuthArgs {
    code: Option<String>,
}

#[async_trait]
impl Tool for GitHubAuthTool {
    fn name(&self) -> &str {
        "github_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with GitHub. Call without arguments to get the \
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
                            "Please visit this URL to authorize GitHub access:\n\n{url}\n\n\
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
                    "GitHub OAuth token saved to {}",
                    self.config.token_path.display()
                );

                self.register_service_tools()?;

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    "Authentication successful. GitHub tools are now available.".to_string(),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("code").is_some() {
            "Completing GitHub OAuth authentication".to_string()
        } else {
            "Getting GitHub OAuth authorization URL".to_string()
        }
    }
}

impl GitHubAuthTool {
    fn register_service_tools(&self) -> Result<()> {
        use super::GitHubClient;
        use super::tools::*;

        let client = Arc::new(GitHubClient::new(self.config.clone())?);
        let readonly = self.readonly;

        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(GitHubListReposTool {
                client: client.clone(),
            }),
            Arc::new(GitHubSearchIssuesTool {
                client: client.clone(),
            }),
            Arc::new(GitHubGetIssueTool {
                client: client.clone(),
            }),
            Arc::new(GitHubListPrsTool {
                client: client.clone(),
            }),
            Arc::new(GitHubGetPrTool {
                client: client.clone(),
            }),
            Arc::new(GitHubListNotificationsTool {
                client: client.clone(),
            }),
        ];

        if !readonly {
            tools.push(Arc::new(GitHubCreateIssueTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GitHubCommentOnIssueTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GitHubCommentOnPrTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(GitHubMergePrTool {
                client: client.clone(),
            }));
        }

        let mut pending = self.pending_tools.lock().unwrap();
        pending.extend(tools);
        Ok(())
    }
}
