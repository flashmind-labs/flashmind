//! Shared Google OAuth2 token management.
//!
//! Supports two authentication flows, auto-detected from the credentials JSON:
//! - **User OAuth** (`"installed"` or `"web"` key): interactive authorization code flow
//!   with persistent refresh tokens.
//! - **Service account** (`"type": "service_account"`): JWT-based token exchange with
//!   optional domain-wide delegation.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Deserialize;
use tracing::debug;

use crate::oauth::{CachedToken, TokenResponse};
use crate::utils::http_client;

const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

// ---------------------------------------------------------------------------
// Credential types
// ---------------------------------------------------------------------------

/// Google Cloud service account private key loaded from a JSON key file.
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceAccountKey {
    pub client_email: String,
    pub private_key: String,
    pub token_uri: Option<String>,
}

/// OAuth2 client credentials downloaded from the Google Cloud Console.
#[derive(Debug, Clone, Deserialize)]
pub struct OAuthClientCredentials {
    pub client_id: String,
    pub client_secret: String,
    pub auth_uri: Option<String>,
    pub token_uri: Option<String>,
    #[serde(default)]
    pub redirect_uris: Vec<String>,
}

/// Credential variant for Google API authentication.
#[derive(Debug, Clone)]
pub enum Credentials {
    /// JWT-based service account with optional domain-wide delegation.
    ServiceAccount {
        key: ServiceAccountKey,
        impersonate: Option<String>,
    },
    /// Interactive OAuth2 authorization code flow.
    UserOAuth(OAuthClientCredentials),
}

impl Credentials {
    /// Parse credentials from a Google Cloud credentials JSON file.
    pub fn from_json(json: &serde_json::Value, impersonate: Option<String>) -> Result<Self> {
        if json.get("type").and_then(|v| v.as_str()) == Some("service_account") {
            let key: ServiceAccountKey =
                serde_json::from_value(json.clone()).context("invalid service account key JSON")?;
            return Ok(Self::ServiceAccount { key, impersonate });
        }

        let inner = json
            .get("installed")
            .or_else(|| json.get("web"))
            .context("credentials JSON must contain 'installed', 'web', or be a service account")?;

        let creds: OAuthClientCredentials =
            serde_json::from_value(inner.clone()).context("invalid OAuth client credentials")?;

        Ok(Self::UserOAuth(creds))
    }
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

/// Exchange an authorization code for access + refresh tokens.
pub async fn exchange_code(creds: &OAuthClientCredentials, code: &str) -> Result<CachedToken> {
    let redirect = creds
        .redirect_uris
        .first()
        .map(|s| s.as_str())
        .unwrap_or("urn:ietf:wg:oauth:2.0:oob");

    let token_uri = creds.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let resp: TokenResponse = http_client()
        .post(token_uri)
        .form(&[
            ("code", code),
            ("client_id", &creds.client_id),
            ("client_secret", &creds.client_secret),
            ("redirect_uri", redirect),
            ("grant_type", "authorization_code"),
        ])
        .send()
        .await?
        .error_for_status()
        .context("token exchange failed")?
        .json()
        .await?;

    Ok(resp.into())
}

/// Refresh an expired access token using a refresh token.
pub async fn refresh_token(creds: &OAuthClientCredentials, refresh: &str) -> Result<CachedToken> {
    let token_uri = creds.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let grant_type = "refresh_token".to_string();
    let refresh_owned = refresh.to_string();
    let resp: TokenResponse = http_client()
        .post(token_uri)
        .form(&[
            ("client_id", &creds.client_id),
            ("client_secret", &creds.client_secret),
            ("refresh_token", &refresh_owned),
            ("grant_type", &grant_type),
        ])
        .send()
        .await?
        .error_for_status()
        .context("token refresh failed")?
        .json()
        .await?;

    let mut token: CachedToken = resp.into();
    if token.refresh_token.is_none() {
        token.refresh_token = Some(refresh.to_string());
    }
    Ok(token)
}

/// Mint a fresh access token using a service account's private key (JWT assertion).
pub async fn service_account_token(
    key: &ServiceAccountKey,
    impersonate: Option<&str>,
    scope: &str,
) -> Result<CachedToken> {
    let now = Utc::now().timestamp();

    let mut claims = serde_json::json!({
        "iss": key.client_email,
        "scope": scope,
        "aud": key.token_uri.as_deref().unwrap_or(TOKEN_URL),
        "iat": now,
        "exp": now + 3600,
    });
    if let Some(sub) = impersonate {
        claims["sub"] = serde_json::json!(sub);
    }

    let encoding_key = jsonwebtoken::EncodingKey::from_rsa_pem(key.private_key.as_bytes())
        .context("invalid RSA private key in service account JSON")?;

    let jwt_header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    let jwt = jsonwebtoken::encode(&jwt_header, &claims, &encoding_key)
        .context("failed to sign service account JWT")?;

    let token_uri = key.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let resp: TokenResponse = http_client()
        .post(token_uri)
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", &jwt),
        ])
        .send()
        .await?
        .error_for_status()
        .context("service account token exchange failed")?
        .json()
        .await?;

    let expires_in = resp.expires_in.unwrap_or(3600);
    debug!("acquired service account token, expires in {expires_in}s");

    Ok(resp.into())
}

/// Build the Google OAuth2 authorization URL for the interactive consent flow.
pub fn auth_url(creds: &OAuthClientCredentials, scope: &str) -> String {
    let redirect = creds
        .redirect_uris
        .first()
        .map(|s| s.as_str())
        .unwrap_or("urn:ietf:wg:oauth:2.0:oob");

    let auth_uri = creds
        .auth_uri
        .as_deref()
        .unwrap_or("https://accounts.google.com/o/oauth2/auth");

    format!(
        "{auth_uri}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
        creds.client_id,
        urlencoding::encode(redirect),
        urlencoding::encode(scope),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_service_account() {
        let json = serde_json::json!({
            "type": "service_account",
            "client_email": "test@project.iam.gserviceaccount.com",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
        });
        let creds = Credentials::from_json(&json, None).unwrap();
        assert!(matches!(creds, Credentials::ServiceAccount { .. }));
    }

    #[test]
    fn detect_user_oauth() {
        let json = serde_json::json!({
            "installed": {
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"]
            }
        });
        let creds = Credentials::from_json(&json, None).unwrap();
        assert!(matches!(creds, Credentials::UserOAuth(_)));
    }

    #[test]
    fn invalid_json_errors() {
        let json = serde_json::json!({"foo": "bar"});
        assert!(Credentials::from_json(&json, None).is_err());
    }
}
