//! OAuth2 token management for the Gmail API.
//!
//! Supports two authentication flows, auto-detected from the credentials JSON:
//! - **User OAuth** (`"installed"` or `"web"` key): interactive authorization code flow
//!   with persistent refresh tokens.
//! - **Service account** (`"type": "service_account"`): JWT-based token exchange with
//!   optional domain-wide delegation.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::utils::http_client;

const GMAIL_SCOPE: &str = "https://mail.google.com/";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

// ---------------------------------------------------------------------------
// Credential types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct ServiceAccountKey {
    pub(crate) client_email: String,
    pub(crate) private_key: String,
    pub(crate) token_uri: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OAuthClientCredentials {
    pub(crate) client_id: String,
    pub(crate) client_secret: String,
    pub(crate) auth_uri: Option<String>,
    pub(crate) token_uri: Option<String>,
    #[serde(default)]
    pub(crate) redirect_uris: Vec<String>,
}

pub(crate) enum Credentials {
    ServiceAccount {
        key: ServiceAccountKey,
        impersonate: Option<String>,
    },
    UserOAuth(OAuthClientCredentials),
}

impl Credentials {
    /// Parse credentials from a Google Cloud JSON file.
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
// Cached token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: i64,
}

impl CachedToken {
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() >= self.expires_at - 60
    }
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

/// Exchange an authorization code for tokens (user OAuth flow).
#[allow(dead_code)]
pub(crate) async fn exchange_code(
    creds: &OAuthClientCredentials,
    code: &str,
) -> Result<CachedToken> {
    let redirect = creds
        .redirect_uris
        .first()
        .map(|s| s.as_str())
        .unwrap_or("urn:ietf:wg:oauth:2.0:oob");

    let token_uri = creds.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let resp: serde_json::Value = http_client()
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

    parse_token_response(&resp)
}

/// Refresh an expired access token using a refresh token.
pub(crate) async fn refresh_token(
    creds: &OAuthClientCredentials,
    refresh: &str,
) -> Result<CachedToken> {
    let token_uri = creds.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let grant_type = "refresh_token".to_string();
    let refresh_owned = refresh.to_string();
    let resp: serde_json::Value = http_client()
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

    let mut token = parse_token_response(&resp)?;
    if token.refresh_token.is_none() {
        token.refresh_token = Some(refresh.to_string());
    }
    Ok(token)
}

/// Acquire a token using a service account JWT.
pub(crate) async fn service_account_token(
    key: &ServiceAccountKey,
    impersonate: Option<&str>,
) -> Result<CachedToken> {
    let now = Utc::now().timestamp();

    let mut claims = serde_json::json!({
        "iss": key.client_email,
        "scope": GMAIL_SCOPE,
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

    let resp: serde_json::Value = http_client()
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

    let access_token = resp["access_token"]
        .as_str()
        .context("missing access_token in response")?
        .to_string();
    let expires_in = resp["expires_in"].as_i64().unwrap_or(3600);

    debug!("acquired service account token, expires in {expires_in}s");

    Ok(CachedToken {
        access_token,
        refresh_token: None,
        expires_at: Utc::now().timestamp() + expires_in,
    })
}

/// Build the authorization URL for the interactive OAuth flow.
pub(crate) fn auth_url(creds: &OAuthClientCredentials) -> String {
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
        urlencoding::encode(GMAIL_SCOPE),
    )
}

/// Load a persisted token from disk.
pub(crate) fn load_token(path: &Path) -> Result<Option<CachedToken>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path).context("reading cached token")?;
    let token: CachedToken = serde_json::from_str(&data).context("parsing cached token")?;
    Ok(Some(token))
}

/// Persist a token to disk.
pub(crate) fn save_token(path: &Path, token: &CachedToken) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("creating token directory")?;
    }
    let data = serde_json::to_string_pretty(token)?;
    std::fs::write(path, data).context("writing cached token")?;
    debug!("persisted token to {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_token_response(resp: &serde_json::Value) -> Result<CachedToken> {
    let access_token = resp["access_token"]
        .as_str()
        .context("missing access_token in token response")?
        .to_string();
    let refresh_token = resp["refresh_token"].as_str().map(|s| s.to_string());
    let expires_in = resp["expires_in"].as_i64().unwrap_or(3600);

    Ok(CachedToken {
        access_token,
        refresh_token,
        expires_at: Utc::now().timestamp() + expires_in,
    })
}

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

    #[test]
    fn token_expiry() {
        let fresh = CachedToken {
            access_token: "tok".into(),
            refresh_token: None,
            expires_at: Utc::now().timestamp() + 3600,
        };
        assert!(!fresh.is_expired());

        let stale = CachedToken {
            access_token: "tok".into(),
            refresh_token: None,
            expires_at: Utc::now().timestamp() - 10,
        };
        assert!(stale.is_expired());
    }
}
