//! Native Gmail API tools.
//!
//! Provides 10 tools for reading, searching, drafting, and labelling Gmail
//! messages — equivalent to the MCP Gmail server but without process overhead.
//!
//! Gated behind the `gmail` feature flag.

pub mod auth;
pub mod tools;
pub mod types;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::utils::http_client;
use auth::{CachedToken, Credentials};

const BASE_URL: &str = "https://gmail.googleapis.com/gmail/v1/users/me";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Gmail tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmailConfig {
    /// Path to Google OAuth client credentials JSON or service account key JSON.
    pub credentials_path: PathBuf,
    /// Path where the OAuth refresh/access token is cached between runs.
    pub token_path: PathBuf,
    /// For service accounts with domain-wide delegation: the user to impersonate.
    pub impersonate: Option<String>,
    /// When true, only register read-only tools (search, get_thread, list_drafts,
    /// list_labels). Excludes send, create_draft, create_label, and label mutations.
    #[serde(default)]
    pub readonly: bool,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Shared Gmail API client holding HTTP state and OAuth tokens.
pub struct GmailClient {
    http: Client,
    config: GmailConfig,
    credentials: Credentials,
    token: RwLock<Option<CachedToken>>,
}

impl GmailClient {
    /// Create a new client. Parses the credentials file eagerly so
    /// configuration errors surface at startup.
    pub fn new(config: GmailConfig) -> Result<Self> {
        let creds_json: serde_json::Value = {
            let data = std::fs::read_to_string(&config.credentials_path).with_context(|| {
                format!(
                    "reading credentials from {}",
                    config.credentials_path.display()
                )
            })?;
            serde_json::from_str(&data).context("parsing credentials JSON")?
        };

        let credentials = Credentials::from_json(&creds_json, config.impersonate.clone())?;

        let cached = auth::load_token(&config.token_path).context("loading cached token")?;

        Ok(Self {
            http: http_client(),
            config,
            credentials,
            token: RwLock::new(cached),
        })
    }

    /// Get a valid access token, refreshing or acquiring one as needed.
    async fn access_token(&self) -> Result<String> {
        {
            let guard = self.token.read().await;
            if let Some(tok) = guard.as_ref()
                && !tok.is_expired()
            {
                return Ok(tok.access_token.clone());
            }
        }

        let mut guard = self.token.write().await;
        if let Some(tok) = guard.as_ref()
            && !tok.is_expired()
        {
            return Ok(tok.access_token.clone());
        }

        let new_token = match &self.credentials {
            Credentials::ServiceAccount { key, impersonate } => {
                auth::service_account_token(key, impersonate.as_deref()).await?
            }
            Credentials::UserOAuth(creds) => {
                if let Some(existing) = guard.as_ref() {
                    if let Some(refresh) = &existing.refresh_token {
                        debug!("refreshing OAuth token");
                        let mut refreshed = auth::refresh_token(creds, refresh).await?;
                        if refreshed.refresh_token.is_none() {
                            refreshed.refresh_token = existing.refresh_token.clone();
                        }
                        refreshed
                    } else {
                        bail!(
                            "OAuth token expired and no refresh token available. \
                             Delete {} and re-authenticate.",
                            self.config.token_path.display()
                        );
                    }
                } else {
                    bail!(
                        "No cached Gmail token found. Run the authorization flow first:\n\
                         1. Visit: {}\n\
                         2. Authorize and copy the code\n\
                         3. Use the gmail_auth tool to exchange the code for a token",
                        auth::auth_url(creds)
                    );
                }
            }
        };

        auth::save_token(&self.config.token_path, &new_token)
            .unwrap_or_else(|e| warn!("failed to persist token: {e}"));

        let access = new_token.access_token.clone();
        *guard = Some(new_token);
        Ok(access)
    }

    /// Make an authenticated GET request to the Gmail API.
    pub(crate) async fn get(&self, path: &str) -> Result<serde_json::Value> {
        self.request(reqwest::Method::GET, path, None).await
    }

    /// Make an authenticated POST request to the Gmail API.
    pub(crate) async fn post(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.request(reqwest::Method::POST, path, Some(body)).await
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let token = self.access_token().await?;
        let url = format!("{BASE_URL}/{path}");

        let mut req = self.http.request(method.clone(), &url).bearer_auth(&token);

        if let Some(b) = &body {
            req = req.json(b);
        }

        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401, forcing token refresh and retrying");
            {
                let mut guard = self.token.write().await;
                *guard = None;
            }
            let new_token = self.access_token().await?;
            let mut retry = self.http.request(method, &url).bearer_auth(&new_token);
            if let Some(b) = body {
                retry = retry.json(&b);
            }
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("Gmail API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Gmail API response");
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Gmail API error ({}): {}", status, text);
        }
        serde_json::from_str(&text).context("parsing Gmail API response")
    }
}
