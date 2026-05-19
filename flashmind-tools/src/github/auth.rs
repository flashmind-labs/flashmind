//! GitHub OAuth2 implementation.
//!
//! Supports the Authorization Code flow for user authentication
//! via GitHub's OAuth Apps.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::oauth::{CachedToken, TokenResponse};
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// OAuth2 credentials for a GitHub OAuth App.
///
/// Create an app at <https://github.com/settings/developers> -> OAuth Apps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubCredentials {
    /// Client ID from the GitHub OAuth App registration.
    pub client_id: String,
    /// Client secret from the GitHub OAuth App registration.
    pub client_secret: String,
    /// Redirect URI registered in the app.
    pub redirect_uri: String,
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

/// Build the GitHub OAuth2 authorization URL for the interactive consent flow.
pub fn auth_url(creds: &GitHubCredentials) -> String {
    format!(
        "https://github.com/login/oauth/authorize?\
         client_id={}&redirect_uri={}&scope=repo,notifications",
        urlencoding::encode(&creds.client_id),
        urlencoding::encode(&creds.redirect_uri),
    )
}

/// Exchange an authorization code for an access token.
///
/// GitHub classic OAuth tokens typically do not expire, so we set a
/// far-future expiry (10 years) when no `expires_in` is returned.
pub async fn exchange_code(creds: &GitHubCredentials, code: &str) -> Result<CachedToken> {
    let resp: TokenResponse = http_client()
        .post("https://github.com/login/oauth/access_token")
        .header("Accept", "application/json")
        .form(&[
            ("client_id", creds.client_id.as_str()),
            ("client_secret", creds.client_secret.as_str()),
            ("code", code),
            ("redirect_uri", creds.redirect_uri.as_str()),
        ])
        .send()
        .await?
        .error_for_status()
        .context("GitHub token exchange failed")?
        .json()
        .await?;

    // GitHub classic tokens don't expire — use 10 years if no expires_in
    let expires_at = if let Some(secs) = resp.expires_in {
        Utc::now().timestamp() + secs
    } else {
        Utc::now().timestamp() + 315_360_000 // ~10 years
    };

    Ok(CachedToken {
        access_token: resp.access_token,
        refresh_token: resp.refresh_token,
        expires_at,
    })
}

/// Refresh an expired access token using a refresh token.
///
/// Only applicable to GitHub Apps (not classic OAuth Apps). Classic tokens
/// do not expire and have no refresh flow.
pub async fn refresh_token(creds: &GitHubCredentials, refresh: &str) -> Result<CachedToken> {
    let resp: TokenResponse = http_client()
        .post("https://github.com/login/oauth/access_token")
        .header("Accept", "application/json")
        .form(&[
            ("client_id", creds.client_id.as_str()),
            ("client_secret", creds.client_secret.as_str()),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
        ])
        .send()
        .await?
        .error_for_status()
        .context("GitHub token refresh failed")?
        .json()
        .await?;

    let expires_at = if let Some(secs) = resp.expires_in {
        Utc::now().timestamp() + secs
    } else {
        Utc::now().timestamp() + 315_360_000
    };

    Ok(CachedToken {
        access_token: resp.access_token,
        refresh_token: resp.refresh_token,
        expires_at,
    })
}
