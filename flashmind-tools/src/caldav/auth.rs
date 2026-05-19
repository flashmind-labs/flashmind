//! CalDAV OAuth2 implementation.
//!
//! Supports the Authorization Code flow for CalDAV providers that use OAuth
//! (e.g. Google CalDAV, Fastmail with OAuth). Endpoints are fully configurable
//! since CalDAV is provider-agnostic.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::oauth::{CachedToken, TokenResponse};
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// OAuth2 credentials for a CalDAV provider.
///
/// All endpoints are configurable since CalDAV is a standard protocol
/// implemented by many providers with different OAuth configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalDavOAuthCredentials {
    /// OAuth2 client ID.
    pub client_id: String,
    /// OAuth2 client secret (optional for public clients using PKCE).
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Authorization endpoint URL.
    pub authorize_url: String,
    /// Token endpoint URL.
    pub token_url: String,
    /// Redirect URI registered with the provider.
    pub redirect_uri: String,
    /// OAuth2 scopes to request.
    #[serde(default)]
    pub scopes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

/// Build the OAuth2 authorization URL for the interactive consent flow.
pub fn auth_url(creds: &CalDavOAuthCredentials) -> String {
    let scope = creds.scopes.join(" ");

    format!(
        "{}?client_id={}&redirect_uri={}&scope={}&response_type=code",
        creds.authorize_url,
        urlencoding::encode(&creds.client_id),
        urlencoding::encode(&creds.redirect_uri),
        urlencoding::encode(&scope),
    )
}

/// Exchange an authorization code for access + refresh tokens.
pub async fn exchange_code(creds: &CalDavOAuthCredentials, code: &str) -> Result<CachedToken> {
    let scope = creds.scopes.join(" ");

    let mut form: Vec<(&str, &str)> = vec![
        ("client_id", &creds.client_id),
        ("code", code),
        ("redirect_uri", &creds.redirect_uri),
        ("grant_type", "authorization_code"),
        ("scope", &scope),
    ];
    if let Some(secret) = &creds.client_secret {
        form.push(("client_secret", secret));
    }

    let resp: TokenResponse = http_client()
        .post(&creds.token_url)
        .form(&form)
        .send()
        .await?
        .error_for_status()
        .context("CalDAV token exchange failed")?
        .json()
        .await?;

    Ok(resp.into())
}

/// Refresh an expired access token using a refresh token.
///
/// Preserves the existing refresh token if the provider does not return a
/// new one in the response.
pub async fn refresh_token(creds: &CalDavOAuthCredentials, refresh: &str) -> Result<CachedToken> {
    let scope = creds.scopes.join(" ");
    let refresh_owned = refresh.to_string();
    let grant_type = "refresh_token".to_string();

    let mut form: Vec<(&str, &str)> = vec![
        ("client_id", &creds.client_id),
        ("refresh_token", &refresh_owned),
        ("grant_type", &grant_type),
        ("scope", &scope),
    ];
    if let Some(secret) = &creds.client_secret {
        form.push(("client_secret", secret));
    }

    let resp: TokenResponse = http_client()
        .post(&creds.token_url)
        .form(&form)
        .send()
        .await?
        .error_for_status()
        .context("CalDAV token refresh failed")?
        .json()
        .await?;

    let mut token: CachedToken = resp.into();
    if token.refresh_token.is_none() {
        token.refresh_token = Some(refresh.to_string());
    }
    Ok(token)
}
