//! Native Slack tools via Slack Web API.
//!
//! Provides tools for channels, messages, threads, search, and reactions
//! using OAuth2 Authorization Code flow with user tokens.

pub mod auth;
pub mod auth_tool;
pub mod tools;
pub mod types;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::debug;

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

const BASE_URL: &str = "https://slack.com/api";

// ---------------------------------------------------------------------------
// Slack API response wrapper
// ---------------------------------------------------------------------------

/// Wrapper for Slack API responses which always return HTTP 200.
///
/// Slack signals errors via the `ok` field rather than HTTP status codes.
/// The actual response data is flattened into `data` via `#[serde(flatten)]`.
#[derive(Deserialize)]
struct SlackApiResponse<T> {
    ok: bool,
    error: Option<String>,
    #[serde(flatten)]
    data: T,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for authenticating with the Slack Web API.
///
/// Holds OAuth credentials inline — the app developer passes client_id/secret
/// directly rather than pointing to a file on disk.
#[derive(Debug, Clone)]
pub struct SlackConfig {
    /// OAuth credentials for the Slack app.
    pub credentials: auth::SlackCredentials,
    /// Path where the cached OAuth token is persisted between runs.
    pub token_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for the Slack Web API.
///
/// Handles token acquisition and caching. Slack user tokens do not expire,
/// so no refresh flow is needed.
pub struct SlackClient {
    http: Client,
    config: SlackConfig,
    token: RwLock<Option<CachedToken>>,
}

impl SlackClient {
    /// Create a new client with the given config and cached token (if any).
    pub fn new(config: SlackConfig) -> Result<Self> {
        let cached = oauth::load_token(&config.token_path).context("loading cached token")?;

        Ok(Self {
            http: http_client(),
            config,
            token: RwLock::new(cached),
        })
    }

    async fn access_token(&self) -> Result<String> {
        // Fast path: read lock
        {
            let guard = self.token.read().await;
            if let Some(tok) = guard.as_ref()
                && !tok.is_expired()
            {
                return Ok(tok.access_token.clone());
            }
        }

        // Slow path: write lock with double-check
        let guard = self.token.write().await;
        if let Some(tok) = guard.as_ref()
            && !tok.is_expired()
        {
            return Ok(tok.access_token.clone());
        }

        bail!(
            "No cached Slack token found. Run the authorization flow first:\n\
             1. Visit: {}\n\
             2. Authorize and copy the code\n\
             3. Use the slack_auth tool to exchange the code for a token",
            auth::auth_url(&self.config.credentials)
        );
    }

    /// Send an authenticated GET request to a Slack API method.
    ///
    /// Slack always returns HTTP 200 — errors are signalled via the `ok` field
    /// in the JSON response body.
    pub async fn api_get<T: DeserializeOwned>(
        &self,
        method: &str,
        params: &[(&str, &str)],
    ) -> Result<T> {
        let token = self.access_token().await?;
        let url = format!("{BASE_URL}/{method}");

        debug!("Slack GET {method}");

        let resp = self
            .http
            .get(&url)
            .bearer_auth(&token)
            .query(params)
            .send()
            .await
            .context("Slack API request failed")?;

        let text = resp.text().await.context("reading Slack response body")?;
        let wrapper: SlackApiResponse<T> =
            serde_json::from_str(&text).context("parsing Slack API response")?;

        if !wrapper.ok {
            let error = wrapper.error.unwrap_or_else(|| "unknown_error".into());
            if error == "token_expired" || error == "invalid_auth" || error == "token_revoked" {
                bail!(
                    "Slack API error: {error}. Your token is no longer valid — \
                     please re-authenticate using the slack_auth tool."
                );
            }
            bail!("Slack API error: {error}");
        }

        Ok(wrapper.data)
    }

    /// Send an authenticated POST request with a JSON body to a Slack API method.
    ///
    /// Slack always returns HTTP 200 — errors are signalled via the `ok` field
    /// in the JSON response body.
    pub async fn api_post<T: DeserializeOwned>(&self, method: &str, body: &Value) -> Result<T> {
        let token = self.access_token().await?;
        let url = format!("{BASE_URL}/{method}");

        debug!("Slack POST {method}");

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&token)
            .json(body)
            .send()
            .await
            .context("Slack API request failed")?;

        let text = resp.text().await.context("reading Slack response body")?;
        let wrapper: SlackApiResponse<T> =
            serde_json::from_str(&text).context("parsing Slack API response")?;

        if !wrapper.ok {
            let error = wrapper.error.unwrap_or_else(|| "unknown_error".into());
            if error == "token_expired" || error == "invalid_auth" || error == "token_revoked" {
                bail!(
                    "Slack API error: {error}. Your token is no longer valid — \
                     please re-authenticate using the slack_auth tool."
                );
            }
            bail!("Slack API error: {error}");
        }

        Ok(wrapper.data)
    }
}

#[cfg(test)]
impl SlackClient {
    pub fn new_for_test() -> Self {
        let credentials = auth::SlackCredentials {
            client_id: "test-client-id".into(),
            client_secret: "test-client-secret".into(),
            redirect_uri: "http://localhost".into(),
        };
        Self {
            http: http_client(),
            config: SlackConfig {
                credentials,
                token_path: "/dev/null".into(),
            },
            token: RwLock::new(None),
        }
    }
}
