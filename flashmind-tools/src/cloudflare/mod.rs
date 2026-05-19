//! Cloudflare API tools.
//!
//! Provides tools for DNS management, zone listing, worker routes, and cache
//! purging using a Cloudflare API token for authentication.

pub mod auth_tool;
pub mod tools;
pub mod types;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::utils::http_client;

const BASE_URL: &str = "https://api.cloudflare.com/client/v4";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for authenticating with the Cloudflare API.
///
/// Only requires a path to persist the API token on disk.
pub struct CloudflareConfig {
    /// Path where the cached API token is persisted between runs.
    pub token_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Envelope types
// ---------------------------------------------------------------------------

/// Standard Cloudflare API response wrapper.
#[derive(Deserialize)]
struct CloudflareEnvelope<T> {
    success: bool,
    errors: Vec<CloudflareError>,
    result: Option<T>,
}

/// An error returned by the Cloudflare API.
#[derive(Deserialize)]
struct CloudflareError {
    #[allow(dead_code)]
    code: u32,
    message: String,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for the Cloudflare REST API.
///
/// All requests use Bearer token authentication with the user-supplied API
/// token. Unlike OAuth flows, Cloudflare API tokens do not expire or need
/// refreshing.
pub struct CloudflareClient {
    http: Client,
    api_token: String,
}

impl CloudflareClient {
    /// Create a new client with the given API token.
    pub fn new(api_token: String) -> Result<Self> {
        Ok(Self {
            http: http_client(),
            api_token,
        })
    }

    /// Send an authenticated GET request and deserialize the JSON response.
    ///
    /// The Cloudflare envelope is unwrapped automatically; only the `result`
    /// field is returned.
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{BASE_URL}/{path}");

        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Cloudflare API error ({}): {}", status, text);
        }

        let envelope: CloudflareEnvelope<T> =
            serde_json::from_str(&text).context("parsing Cloudflare API response")?;
        Self::unwrap_envelope(envelope)
    }

    /// Send an authenticated POST request with a JSON body.
    pub async fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        self.request_with_body(reqwest::Method::POST, path, body)
            .await
    }

    /// Send an authenticated PUT request with a JSON body.
    pub async fn put<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        self.request_with_body(reqwest::Method::PUT, path, body)
            .await
    }

    /// Send an authenticated DELETE request (expects success status).
    pub async fn delete_req(&self, path: &str) -> Result<()> {
        let url = format!("{BASE_URL}/{path}");

        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("Cloudflare API error ({}): {}", status, text);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    async fn request_with_body<B: Serialize, T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{BASE_URL}/{path}");

        let resp = self
            .http
            .request(method, &url)
            .bearer_auth(&self.api_token)
            .json(body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Cloudflare API error ({}): {}", status, text);
        }

        let envelope: CloudflareEnvelope<T> =
            serde_json::from_str(&text).context("parsing Cloudflare API response")?;
        Self::unwrap_envelope(envelope)
    }

    fn unwrap_envelope<T>(envelope: CloudflareEnvelope<T>) -> Result<T> {
        if !envelope.success {
            let msg = envelope
                .errors
                .first()
                .map(|e| e.message.as_str())
                .unwrap_or("unknown Cloudflare API error");
            bail!("Cloudflare API error: {msg}");
        }
        envelope
            .result
            .context("Cloudflare API returned success but no result")
    }
}

#[cfg(test)]
impl CloudflareClient {
    pub fn new_for_test() -> Self {
        Self {
            http: http_client(),
            api_token: "test-token".into(),
        }
    }
}
