//! Native Microsoft Outlook tools via Microsoft Graph API.
//!
//! Provides tools for Mail, Calendar, and Contacts using OAuth2
//! Authorization Code flow with PKCE.

pub mod auth;
pub mod auth_tool;
pub mod calendar;
pub mod contacts;
pub mod mail;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use serde::Deserialize;

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Batch types
// ---------------------------------------------------------------------------

/// A single request in a JSON batch.
#[derive(Debug, Clone, Serialize)]
pub struct BatchRequest {
    pub id: String,
    pub method: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Response from a JSON batch request.
#[derive(Debug, Deserialize)]
pub struct BatchResponse {
    pub responses: Vec<BatchResponseItem>,
}

/// Individual response within a batch.
#[derive(Debug, Deserialize)]
pub struct BatchResponseItem {
    pub id: String,
    pub status: u16,
    pub body: Option<serde_json::Value>,
}

const BASE_URL: &str = "https://graph.microsoft.com/v1.0/me";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for authenticating with Microsoft Graph API (Outlook Mail,
/// Calendar, Contacts).
///
/// Holds OAuth credentials inline — the app developer passes client_id/secret
/// directly rather than pointing to a file on disk.
#[derive(Debug, Clone)]
pub struct OutlookConfig {
    /// OAuth credentials for Microsoft Entra ID (Azure AD).
    pub credentials: auth::OutlookCredentials,
    /// Path where the cached OAuth token is persisted between runs.
    pub token_path: PathBuf,
    /// When `true`, only read-scopes are requested and write tools are not registered.
    pub readonly: bool,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for Microsoft Graph API.
///
/// Handles token acquisition, caching, refresh, and automatic retry on 401.
pub struct OutlookClient {
    http: Client,
    credentials: auth::OutlookCredentials,
    config: OutlookConfig,
    token: RwLock<Option<CachedToken>>,
}

impl OutlookClient {
    /// Create a new client with the given credentials and cached token (if any).
    pub fn new(config: OutlookConfig, scopes: &[&str]) -> Result<Self> {
        let mut credentials = config.credentials.clone();

        // Store the scopes needed for token refresh
        if credentials.scopes.is_empty() {
            credentials.scopes = scopes.iter().map(|s| s.to_string()).collect();
        }

        let cached = oauth::load_token(&config.token_path).context("loading cached token")?;

        Ok(Self {
            http: http_client(),
            credentials,
            config,
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

        if let Some(existing) = guard.as_ref() {
            if let Some(refresh) = &existing.refresh_token {
                debug!("refreshing Outlook OAuth token");
                let mut refreshed = auth::refresh_token(&self.credentials, refresh).await?;
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
                "OAuth token expired and no refresh token available. \
                 Delete {} and re-authenticate.",
                self.config.token_path.display()
            );
        }

        bail!(
            "No cached Outlook token found. Run the authorization flow first:\n\
             1. Visit: {}\n\
             2. Authorize and copy the code\n\
             3. Use the outlook_auth tool to exchange the code for a token",
            auth::auth_url(&self.credentials)
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

    /// Send an authenticated PATCH request with a JSON body.
    pub async fn patch<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        self.request(reqwest::Method::PATCH, path, Some(body)).await
    }

    /// Send a JSON batch request (`POST /$batch`) and return individual responses.
    ///
    /// Microsoft Graph limits each batch to 20 requests. This method
    /// automatically chunks larger batches and concatenates results.
    pub async fn batch(&self, requests: Vec<BatchRequest>) -> Result<Vec<BatchResponseItem>> {
        let mut all_responses = Vec::with_capacity(requests.len());

        for chunk in requests.chunks(20) {
            let body = serde_json::json!({ "requests": chunk });
            let resp: BatchResponse = self.batch_post(&body).await?;
            all_responses.extend(resp.responses);
        }

        Ok(all_responses)
    }

    async fn batch_post<B: Serialize, T: DeserializeOwned>(&self, body: &B) -> Result<T> {
        const BATCH_URL: &str = "https://graph.microsoft.com/v1.0/$batch";
        let token = self.access_token().await?;

        let resp = self
            .http
            .post(BATCH_URL)
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
                .post(BATCH_URL)
                .bearer_auth(&new_token)
                .json(body)
                .send()
                .await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("Microsoft Graph API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Graph API batch response");
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Microsoft Graph API error ({}): {}", status, text);
        }
        serde_json::from_str(&text).context("parsing Graph API batch response")
    }

    /// Send an authenticated DELETE request (expects 204 No Content on success).
    pub async fn delete(&self, path: &str) -> Result<()> {
        let token = self.access_token().await?;
        let url = format!("{BASE_URL}/{path}");

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
                bail!("Microsoft Graph API error ({}): {}", status, text);
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
            let text = resp.text().await?;
            bail!("Microsoft Graph API error ({}): {}", status, text);
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
                bail!("Microsoft Graph API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Graph API response");
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Microsoft Graph API error ({}): {}", status, text);
        }
        serde_json::from_str(&text).context("parsing Graph API response")
    }
}

#[cfg(test)]
impl OutlookClient {
    pub fn new_for_test() -> Self {
        let credentials = auth::OutlookCredentials {
            client_id: "test-client-id".into(),
            client_secret: None,
            tenant: "common".into(),
            redirect_uri: "http://localhost".into(),
            scopes: vec!["Mail.Read".into()],
        };
        Self {
            http: http_client(),
            credentials: credentials.clone(),
            config: OutlookConfig {
                credentials: credentials,
                token_path: "/dev/null".into(),
                readonly: false,
            },
            token: RwLock::new(None),
        }
    }
}
