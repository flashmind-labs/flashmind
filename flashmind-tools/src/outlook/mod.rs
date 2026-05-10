//! Native Microsoft Outlook tools via Microsoft Graph API.
//!
//! Provides tools for Mail, Calendar, and Contacts using OAuth2
//! Authorization Code flow with PKCE.

pub mod auth;
pub mod mail;
pub mod calendar;
pub mod contacts;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

const BASE_URL: &str = "https://graph.microsoft.com/v1.0/me";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlookConfig {
    pub credentials_path: PathBuf,
    pub token_path: PathBuf,
    #[serde(default)]
    pub readonly: bool,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

pub struct OutlookClient {
    http: Client,
    credentials: auth::OutlookCredentials,
    config: OutlookConfig,
    token: RwLock<Option<CachedToken>>,
}

impl OutlookClient {
    pub fn new(config: OutlookConfig, scopes: &[&str]) -> Result<Self> {
        let creds_json = std::fs::read_to_string(&config.credentials_path).with_context(|| {
            format!(
                "reading Outlook credentials from {}",
                config.credentials_path.display()
            )
        })?;
        let mut credentials: auth::OutlookCredentials =
            serde_json::from_str(&creds_json).context("parsing Outlook credentials JSON")?;

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

    pub async fn get(&self, path: &str) -> Result<Value> {
        self.request(reqwest::Method::GET, path, None).await
    }

    pub async fn post(&self, path: &str, body: Value) -> Result<Value> {
        self.request(reqwest::Method::POST, path, Some(body)).await
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
            if status == reqwest::StatusCode::NO_CONTENT {
                return Ok(serde_json::json!({}));
            }
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("Microsoft Graph API error ({}): {}", status, text);
            }
            return serde_json::from_str(&text).context("parsing Graph API response");
        }

        let status = resp.status();
        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(serde_json::json!({}));
        }
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
        Self {
            http: http_client(),
            credentials: auth::OutlookCredentials {
                client_id: "test-client-id".into(),
                client_secret: None,
                tenant: "common".into(),
                redirect_uri: "http://localhost".into(),
                scopes: vec!["Mail.Read".into()],
            },
            config: OutlookConfig {
                credentials_path: "/dev/null".into(),
                token_path: "/dev/null".into(),
                readonly: false,
            },
            token: RwLock::new(None),
        }
    }
}
