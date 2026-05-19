//! Slack OAuth2 implementation.
//!
//! Supports the OAuth v2 Authorization Code flow for user token authentication
//! via the Slack Web API.

use anyhow::{Context, Result, bail};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::oauth::CachedToken;
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// OAuth2 credentials for a Slack app.
///
/// Create an app at <https://api.slack.com/apps> and configure OAuth scopes
/// under "User Token Scopes".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackCredentials {
    /// Client ID from the Slack app settings.
    pub client_id: String,
    /// Client secret from the Slack app settings.
    pub client_secret: String,
    /// Redirect URI registered in the app.
    pub redirect_uri: String,
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

/// Build the Slack OAuth v2 authorization URL for the interactive consent flow.
///
/// Uses `user_scope` (not `scope`) to request a user token rather than a bot token.
pub fn auth_url(creds: &SlackCredentials) -> String {
    let user_scope = "channels:read,channels:history,groups:read,groups:history,search:read,chat:write,reactions:write";

    format!(
        "https://slack.com/oauth/v2/authorize?\
         client_id={}&user_scope={}&redirect_uri={}",
        urlencoding::encode(&creds.client_id),
        urlencoding::encode(user_scope),
        urlencoding::encode(&creds.redirect_uri),
    )
}

/// Response from `oauth.v2.access` containing the authed user token.
#[derive(Debug, Deserialize)]
struct OAuthV2AccessResponse {
    ok: bool,
    error: Option<String>,
    authed_user: Option<AuthedUser>,
}

/// The `authed_user` block within the Slack OAuth v2 access response.
#[derive(Debug, Deserialize)]
struct AuthedUser {
    #[allow(dead_code)]
    id: String,
    access_token: String,
}

/// Exchange an authorization code for a user access token.
///
/// Slack user tokens do not expire, so we set a far-future expiry (10 years).
/// The response structure is non-standard — the user token lives under
/// `authed_user.access_token` rather than at the top level.
pub async fn exchange_code(creds: &SlackCredentials, code: &str) -> Result<CachedToken> {
    let resp: OAuthV2AccessResponse = http_client()
        .post("https://slack.com/api/oauth.v2.access")
        .form(&[
            ("client_id", creds.client_id.as_str()),
            ("client_secret", creds.client_secret.as_str()),
            ("code", code),
            ("redirect_uri", creds.redirect_uri.as_str()),
        ])
        .send()
        .await?
        .json()
        .await
        .context("parsing Slack OAuth response")?;

    if !resp.ok {
        let error = resp.error.unwrap_or_else(|| "unknown_error".into());
        bail!("Slack OAuth exchange failed: {error}");
    }

    let authed_user = resp
        .authed_user
        .context("Slack OAuth response missing authed_user field")?;

    // Slack user tokens don't expire — use 10 years
    Ok(CachedToken {
        access_token: authed_user.access_token,
        refresh_token: None,
        expires_at: Utc::now().timestamp() + 315_360_000,
    })
}
