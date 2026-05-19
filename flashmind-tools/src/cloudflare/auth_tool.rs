//! Cloudflare API token authentication tool.
//!
//! Unlike OAuth-based integrations, Cloudflare uses a simple API token.
//! The user creates a token at dash.cloudflare.com and provides it directly.
//! On successful verification the token is persisted and service tools are
//! registered dynamically via `PendingTools`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult};

use super::CloudflareConfig;
use super::types::TokenVerifyResult;
use crate::builder::PendingTools;
use crate::oauth;

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Auth tool that validates a Cloudflare API token and registers service tools.
///
/// Only registered when no valid cached token exists. Once the user supplies a
/// valid token it is persisted to disk and the full set of Cloudflare tools
/// becomes available.
pub struct CloudflareAuthTool {
    /// Configuration including the token persistence path.
    pub config: CloudflareConfig,
    /// When `true`, only read tools are registered after auth.
    pub readonly: bool,
    /// Shared queue for dynamically registering tools after auth completes.
    pub pending_tools: PendingTools,
}

#[derive(Deserialize)]
struct AuthArgs {
    api_token: Option<String>,
}

#[async_trait]
impl Tool for CloudflareAuthTool {
    fn name(&self) -> &str {
        "cloudflare_auth"
    }

    fn description(&self) -> &str {
        "Authenticate with the Cloudflare API. Call without arguments for \
         instructions, or provide your API token to complete sign-in."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "api_token": {
                    "type": "string",
                    "description": "Cloudflare API token (omit to get setup instructions)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: AuthArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        match args.api_token {
            None => Ok(ToolResult::success(
                ctx.tool_call_id,
                "To authenticate with Cloudflare:\n\n\
                 1. Visit https://dash.cloudflare.com/profile/api-tokens\n\
                 2. Click \"Create Token\"\n\
                 3. Choose a template (e.g. \"Edit zone DNS\") or create a custom token\n\
                 4. Copy the generated token\n\
                 5. Call this tool again with the `api_token` parameter\n\n\
                 The token will be saved locally for future use."
                    .to_string(),
            )),
            Some(token) => {
                // Validate the token by calling the verify endpoint
                let client = super::CloudflareClient::new(token.clone())?;
                let verify: TokenVerifyResult = client.get("user/tokens/verify").await?;

                if verify.status != "active" {
                    return Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!(
                            "Token verification failed: status is '{}'. \
                             Please provide an active API token.",
                            verify.status
                        ),
                    ));
                }

                // Save as a CachedToken with far-future expiry (10 years)
                let cached = oauth::CachedToken {
                    access_token: token,
                    refresh_token: None,
                    expires_at: Utc::now().timestamp() + 10 * 365 * 24 * 3600,
                };
                oauth::save_token(&self.config.token_path, &cached)?;
                info!(
                    "Cloudflare API token saved to {}",
                    self.config.token_path.display()
                );

                self.register_service_tools()?;

                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    "Authentication successful. Cloudflare tools are now available.".to_string(),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        if args.get("api_token").is_some() {
            "Verifying Cloudflare API token".to_string()
        } else {
            "Getting Cloudflare authentication instructions".to_string()
        }
    }
}

impl CloudflareAuthTool {
    fn register_service_tools(&self) -> Result<()> {
        use super::CloudflareClient;
        use super::tools::*;

        let cached = oauth::load_token(&self.config.token_path)?
            .ok_or_else(|| anyhow::anyhow!("token was just saved but could not be loaded"))?;
        let client = Arc::new(CloudflareClient::new(cached.access_token)?);
        let readonly = self.readonly;

        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(CloudflareListZonesTool {
                client: client.clone(),
            }),
            Arc::new(CloudflareListDnsRecordsTool {
                client: client.clone(),
            }),
            Arc::new(CloudflareGetDnsRecordTool {
                client: client.clone(),
            }),
            Arc::new(CloudflareListWorkerRoutesTool {
                client: client.clone(),
            }),
        ];

        if !readonly {
            tools.push(Arc::new(CloudflareCreateDnsRecordTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CloudflareUpdateDnsRecordTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CloudflareDeleteDnsRecordTool {
                client: client.clone(),
            }));
            tools.push(Arc::new(CloudflarePurgeCacheTool {
                client: client.clone(),
            }));
        }

        let mut pending = self.pending_tools.lock().unwrap();
        pending.extend(tools);
        Ok(())
    }
}
