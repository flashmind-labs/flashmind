//! Generic authenticated Google API HTTP client.
//!
//! Parameterized by `base_url` and `scope`, shared across Gmail, Calendar,
//! and Contacts services.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

#[cfg(test)]
use super::auth::ServiceAccountKey;
use super::auth::{self, Credentials};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for authenticating with Google APIs (Gmail, Calendar, Contacts).
///
/// Holds OAuth credentials inline — the app developer embeds client_id/secret
/// directly rather than pointing to a file on disk.
#[derive(Debug, Clone)]
pub struct GoogleConfig {
    /// OAuth credentials for authentication.
    pub credentials: Credentials,
    /// Path where the cached OAuth token is persisted between runs.
    pub token_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for a single Google API service.
///
/// Handles token acquisition, caching, refresh, and automatic retry on 401.
/// Parameterized by `base_url` and `scope` — each Google service (Gmail,
/// Calendar, Contacts) creates its own instance via [`GoogleClient::new`].
pub struct GoogleClient {
    http: Client,
    base_url: &'static str,
    scope: &'static str,
    config: GoogleConfig,
    credentials: Credentials,
    token: RwLock<Option<CachedToken>>,
}

impl GoogleClient {
    /// Create a new client with the given credentials and cached token (if any).
    pub fn new(config: GoogleConfig, base_url: &'static str, scope: &'static str) -> Result<Self> {
        let cached = oauth::load_token(&config.token_path).context("loading cached token")?;
        let credentials = config.credentials.clone();

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

    /// Send an authenticated GET request and deserialize the JSON response.
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.request(reqwest::Method::GET, path, None::<&()>).await
    }

    /// Send an authenticated POST request with a JSON body.
    pub async fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        self.request(reqwest::Method::POST, path, Some(body)).await
    }

    /// Send an authenticated PUT request with a JSON body.
    pub async fn put<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        self.request(reqwest::Method::PUT, path, Some(body)).await
    }

    /// Send an authenticated PATCH request with a JSON body.
    pub async fn patch<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        self.request(reqwest::Method::PATCH, path, Some(body)).await
    }

    /// Send an authenticated POST request that returns no content (204).
    pub async fn post_no_content<B: Serialize + Sync>(&self, path: &str, body: &B) -> Result<()> {
        let token = self.access_token().await?;
        let url = format!("{}/{path}", self.base_url);

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&token)
            .json(body)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401, forcing token refresh and retrying");
            {
                let mut guard = self.token.write().await;
                *guard = None;
            }
            let new_token = self.access_token().await?;
            let retry_resp = self
                .http
                .post(&url)
                .bearer_auth(&new_token)
                .json(body)
                .send()
                .await?;
            let status = retry_resp.status();
            if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
                let text = retry_resp.text().await?;
                bail!("Google API error ({}): {}", status, text);
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
            let text = resp.text().await?;
            bail!("Google API error ({}): {}", status, text);
        }
        Ok(())
    }

    /// Send an authenticated DELETE request (expects 204 No Content on success).
    pub async fn delete(&self, path: &str) -> Result<()> {
        let token = self.access_token().await?;
        let url = format!("{}/{path}", self.base_url);

        let resp = self
            .http
            .request(reqwest::Method::DELETE, &url)
            .bearer_auth(&token)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401, forcing token refresh and retrying");
            {
                let mut guard = self.token.write().await;
                *guard = None;
            }
            let new_token = self.access_token().await?;
            let retry_resp = self
                .http
                .request(reqwest::Method::DELETE, &url)
                .bearer_auth(&new_token)
                .send()
                .await?;
            let status = retry_resp.status();
            if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
                let text = retry_resp.text().await?;
                bail!("Google API error ({}): {}", status, text);
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
            let text = resp.text().await?;
            bail!("Google API error ({}): {}", status, text);
        }
        Ok(())
    }

    async fn request<B: Serialize, T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&B>,
    ) -> Result<T> {
        let token = self.access_token().await?;
        let url = format!("{}/{path}", self.base_url);

        let mut req = self.http.request(method.clone(), &url).bearer_auth(&token);
        if let Some(b) = body {
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
                retry = retry.json(b);
            }
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("Google API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Google API response");
        }

        let status = resp.status();
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
                credentials: Credentials::ServiceAccount {
                    key: ServiceAccountKey {
                        client_email: "test@test.iam.gserviceaccount.com".into(),
                        private_key: String::new(),
                        token_uri: None,
                    },
                    impersonate: None,
                },
                token_path: "/dev/null".into(),
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
