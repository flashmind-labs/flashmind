//! Native GitHub tools via GitHub REST API.
//!
//! Provides tools for repositories, issues, pull requests, and notifications
//! using OAuth2 Authorization Code flow.

pub mod auth;
pub mod auth_tool;
pub mod tools;
pub mod types;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

const BASE_URL: &str = "https://api.github.com";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for authenticating with the GitHub REST API.
///
/// Holds OAuth credentials inline — the app developer passes client_id/secret
/// directly rather than pointing to a file on disk.
#[derive(Debug, Clone)]
pub struct GitHubConfig {
    /// OAuth credentials for GitHub.
    pub credentials: auth::GitHubCredentials,
    /// Path where the cached OAuth token is persisted between runs.
    pub token_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for the GitHub REST API.
///
/// Handles token acquisition, caching, and automatic retry on 401.
pub struct GitHubClient {
    http: Client,
    config: GitHubConfig,
    token: RwLock<Option<CachedToken>>,
}

impl GitHubClient {
    /// Create a new client with the given config and cached token (if any).
    pub fn new(config: GitHubConfig) -> Result<Self> {
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
        let mut guard = self.token.write().await;
        if let Some(tok) = guard.as_ref()
            && !tok.is_expired()
        {
            return Ok(tok.access_token.clone());
        }

        if let Some(existing) = guard.as_ref() {
            // GitHub tokens typically don't expire; only refresh if we have a refresh_token
            if let Some(refresh) = &existing.refresh_token {
                debug!("refreshing GitHub OAuth token");
                let mut refreshed = auth::refresh_token(&self.config.credentials, refresh).await?;
                if refreshed.refresh_token.is_none() {
                    refreshed.refresh_token = existing.refresh_token.clone();
                }

                oauth::save_token(&self.config.token_path, &refreshed)
                    .unwrap_or_else(|e| warn!("failed to persist token: {e}"));

                let access = refreshed.access_token.clone();
                *guard = Some(refreshed);
                return Ok(access);
            }

            bail!(
                "GitHub token expired and no refresh token available. \
                 Delete {} and re-authenticate.",
                self.config.token_path.display()
            );
        }

        bail!(
            "No cached GitHub token found. Run the authorization flow first:\n\
             1. Visit: {}\n\
             2. Authorize and copy the code\n\
             3. Use the github_auth tool to exchange the code for a token",
            auth::auth_url(&self.config.credentials)
        );
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

    /// Send an authenticated DELETE request (expects 204 No Content on success).
    pub async fn delete(&self, path: &str) -> Result<()> {
        let token = self.access_token().await?;
        let url = format!("{BASE_URL}/{path}");

        let resp = self
            .http
            .request(reqwest::Method::DELETE, &url)
            .bearer_auth(&token)
            .header("Accept", "application/vnd.github+json")
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
                .header("Accept", "application/vnd.github+json")
                .send()
                .await?;
            let status = retry_resp.status();
            if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
                let text = retry_resp.text().await?;
                bail!("GitHub API error ({}): {}", status, text);
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
            let text = resp.text().await?;
            bail!("GitHub API error ({}): {}", status, text);
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
        let url = format!("{BASE_URL}/{path}");

        let mut req = self
            .http
            .request(method.clone(), &url)
            .bearer_auth(&token)
            .header("Accept", "application/vnd.github+json");
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
            let mut retry = self
                .http
                .request(method, &url)
                .bearer_auth(&new_token)
                .header("Accept", "application/vnd.github+json");
            if let Some(b) = body {
                retry = retry.json(b);
            }
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("GitHub API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing GitHub API response");
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("GitHub API error ({}): {}", status, text);
        }
        serde_json::from_str(&text).context("parsing GitHub API response")
    }
}

#[cfg(test)]
impl GitHubClient {
    pub fn new_for_test() -> Self {
        let credentials = auth::GitHubCredentials {
            client_id: "test-client-id".into(),
            client_secret: "test-client-secret".into(),
            redirect_uri: "http://localhost".into(),
        };
        Self {
            http: http_client(),
            config: GitHubConfig {
                credentials,
                token_path: "/dev/null".into(),
            },
            token: RwLock::new(None),
        }
    }
}
