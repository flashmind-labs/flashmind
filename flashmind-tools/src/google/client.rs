//! Generic authenticated Google API HTTP client.
//!
//! Parameterized by `base_url` and `scope`, shared across Gmail, Calendar,
//! and Contacts services.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

use super::auth::{self, Credentials};
#[cfg(test)]
use super::auth::ServiceAccountKey;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleConfig {
    pub credentials_path: PathBuf,
    pub token_path: PathBuf,
    pub impersonate: Option<String>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

pub struct GoogleClient {
    http: Client,
    base_url: &'static str,
    scope: &'static str,
    config: GoogleConfig,
    credentials: Credentials,
    token: RwLock<Option<CachedToken>>,
}

impl GoogleClient {
    pub fn new(config: GoogleConfig, base_url: &'static str, scope: &'static str) -> Result<Self> {
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
        let cached = oauth::load_token(&config.token_path).context("loading cached token")?;

        Ok(Self {
            http: http_client(),
            base_url,
            scope,
            config,
            credentials,
            token: RwLock::new(cached),
        })
    }

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
                auth::service_account_token(key, impersonate.as_deref(), self.scope).await?
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
                        "No cached token found. Run the authorization flow first:\n\
                         1. Visit: {}\n\
                         2. Authorize and copy the code\n\
                         3. Use the auth tool to exchange the code for a token",
                        auth::auth_url(creds, self.scope)
                    );
                }
            }
        };

        oauth::save_token(&self.config.token_path, &new_token)
            .unwrap_or_else(|e| warn!("failed to persist token: {e}"));

        let access = new_token.access_token.clone();
        *guard = Some(new_token);
        Ok(access)
    }

    pub async fn get(&self, path: &str) -> Result<Value> {
        self.request(reqwest::Method::GET, path, None).await
    }

    pub async fn post(&self, path: &str, body: Value) -> Result<Value> {
        self.request(reqwest::Method::POST, path, Some(body)).await
    }

    pub async fn put(&self, path: &str, body: Value) -> Result<Value> {
        self.request(reqwest::Method::PUT, path, Some(body)).await
    }

    pub async fn patch(&self, path: &str, body: Value) -> Result<Value> {
        self.request(reqwest::Method::PATCH, path, Some(body)).await
    }

    pub async fn delete(&self, path: &str) -> Result<Value> {
        self.request(reqwest::Method::DELETE, path, None).await
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let token = self.access_token().await?;
        let url = format!("{}/{path}", self.base_url);

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
            if status == reqwest::StatusCode::NO_CONTENT {
                return Ok(serde_json::json!({}));
            }
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("Google API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Google API response");
        }

        let status = resp.status();
        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(serde_json::json!({}));
        }
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Google API error ({}): {}", status, text);
        }
        serde_json::from_str(&text).context("parsing Google API response")
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
impl GoogleClient {
    pub fn new_for_test(base_url: &'static str, scope: &'static str) -> Self {
        Self {
            http: http_client(),
            base_url,
            scope,
            config: GoogleConfig {
                credentials_path: "/dev/null".into(),
                token_path: "/dev/null".into(),
                impersonate: None,
            },
            credentials: Credentials::ServiceAccount {
                key: ServiceAccountKey {
                    client_email: "test@test.iam.gserviceaccount.com".into(),
                    private_key: String::new(),
                    token_uri: None,
                },
                impersonate: None,
            },
            token: RwLock::new(None),
        }
    }
}
