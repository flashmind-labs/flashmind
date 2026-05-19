//! Generic CalDAV integration for any CalDAV-compliant server.
//!
//! Supports Fastmail, Nextcloud, iCloud, and any other CalDAV provider.
//! Provides both Basic auth and OAuth2 authentication modes.

pub mod auth;
pub mod auth_tool;
pub mod tools;
pub mod types;
pub mod xml;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::oauth::{self, CachedToken};
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

/// Authentication mode for connecting to a CalDAV server.
#[derive(Debug, Clone)]
pub enum CalDavAuth {
    /// HTTP Basic authentication (username + password or app-specific password).
    Basic {
        /// CalDAV username.
        username: String,
        /// CalDAV password or app-specific password.
        password: String,
    },
    /// OAuth2 Authorization Code flow.
    OAuth {
        /// Provider-specific OAuth2 credentials.
        credentials: auth::CalDavOAuthCredentials,
    },
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for connecting to a CalDAV server.
#[derive(Debug, Clone)]
pub struct CalDavConfig {
    /// Base URL of the CalDAV server (e.g. `https://caldav.fastmail.com/dav`).
    pub server_url: String,
    /// Authentication mode.
    pub auth: CalDavAuth,
    /// Path where the cached OAuth token is persisted (only used for OAuth mode).
    pub token_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated HTTP client for CalDAV servers.
///
/// Handles token acquisition, caching, refresh, and automatic retry on 401.
/// Supports both Basic and OAuth authentication.
pub struct CalDavClient {
    http: Client,
    server_url: String,
    auth: CalDavAuth,
    token_path: Option<PathBuf>,
    token: RwLock<Option<CachedToken>>,
}

impl CalDavClient {
    /// Create a new client from the given configuration.
    ///
    /// For Basic auth, no token is loaded. For OAuth, attempts to load a
    /// cached token from `token_path`.
    pub fn new(config: &CalDavConfig) -> Result<Self> {
        let cached = match (&config.auth, &config.token_path) {
            (CalDavAuth::OAuth { .. }, Some(path)) => {
                oauth::load_token(path).context("loading cached CalDAV token")?
            }
            _ => None,
        };

        Ok(Self {
            http: http_client(),
            server_url: config.server_url.clone(),
            auth: config.auth.clone(),
            token_path: config.token_path.clone(),
            token: RwLock::new(cached),
        })
    }

    /// Retrieve a valid OAuth access token, refreshing if necessary.
    ///
    /// Uses double-checked locking to avoid unnecessary refreshes under
    /// concurrent access.
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
                debug!("refreshing CalDAV OAuth token");
                let creds = match &self.auth {
                    CalDavAuth::OAuth { credentials } => credentials,
                    _ => bail!("access_token called but auth mode is not OAuth"),
                };
                let mut refreshed = auth::refresh_token(creds, refresh).await?;
                if refreshed.refresh_token.is_none() {
                    refreshed.refresh_token = existing.refresh_token.clone();
                }

                if let Some(path) = &self.token_path {
                    oauth::save_token(path, &refreshed)
                        .unwrap_or_else(|e| warn!("failed to persist CalDAV token: {e}"));
                }

                let access = refreshed.access_token.clone();
                *guard = Some(refreshed);
                return Ok(access);
            }
            bail!(
                "CalDAV OAuth token expired and no refresh token available. \
                 Re-authenticate using the caldav_auth tool."
            );
        }

        bail!(
            "No cached CalDAV token found. Run the authorization flow first \
             using the caldav_auth tool."
        );
    }

    /// Apply authentication to an outgoing request.
    async fn apply_auth(&self, req: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder> {
        match &self.auth {
            CalDavAuth::Basic { username, password } => {
                Ok(req.basic_auth(username, Some(password)))
            }
            CalDavAuth::OAuth { .. } => {
                let token = self.access_token().await?;
                Ok(req.bearer_auth(token))
            }
        }
    }

    /// Build the full URL for a CalDAV resource path.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.server_url.trim_end_matches('/'), path)
    }

    /// Invalidate the cached token so the next request triggers a refresh.
    async fn invalidate_token(&self) {
        let mut guard = self.token.write().await;
        if let Some(tok) = guard.as_mut() {
            tok.expires_at = 0;
        }
    }

    // -----------------------------------------------------------------------
    // CalDAV HTTP methods
    // -----------------------------------------------------------------------

    /// Send a PROPFIND request to discover resources and their properties.
    pub async fn propfind(&self, path: &str, depth: u8, body: &str) -> Result<String> {
        let url = self.url(path);
        let method = reqwest::Method::from_bytes(b"PROPFIND").unwrap();

        let req = self
            .http
            .request(method.clone(), &url)
            .header("Content-Type", "application/xml")
            .header("Depth", depth.to_string())
            .body(body.to_string());
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on PROPFIND, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self
                .http
                .request(method, &url)
                .header("Content-Type", "application/xml")
                .header("Depth", depth.to_string())
                .body(body.to_string());
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(text);
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(text)
    }

    /// Send a REPORT request (calendar-query or calendar-multiget).
    pub async fn report(&self, path: &str, body: &str) -> Result<String> {
        let url = self.url(path);
        let method = reqwest::Method::from_bytes(b"REPORT").unwrap();

        let req = self
            .http
            .request(method.clone(), &url)
            .header("Content-Type", "application/xml")
            .header("Depth", "1")
            .body(body.to_string());
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on REPORT, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self
                .http
                .request(method, &url)
                .header("Content-Type", "application/xml")
                .header("Depth", "1")
                .body(body.to_string());
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(text);
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(text)
    }

    /// Fetch raw iCalendar data for a single resource.
    pub async fn get_raw(&self, path: &str) -> Result<String> {
        let url = self.url(path);

        let req = self.http.get(&url);
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on GET, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self.http.get(&url);
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            let text = retry_resp.text().await?;
            if !status.is_success() {
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(text);
        }

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(text)
    }

    /// Upload iCalendar data to a resource path.
    pub async fn put_ical(&self, path: &str, ical: &str) -> Result<()> {
        let url = self.url(path);

        let req = self
            .http
            .put(&url)
            .header("Content-Type", "text/calendar; charset=utf-8")
            .body(ical.to_string());
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on PUT, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self
                .http
                .put(&url)
                .header("Content-Type", "text/calendar; charset=utf-8")
                .body(ical.to_string());
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            if !status.is_success() {
                let text = retry_resp.text().await?;
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(())
    }

    /// Delete a CalDAV resource.
    pub async fn delete(&self, path: &str) -> Result<()> {
        let url = self.url(path);

        let req = self.http.delete(&url);
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on DELETE, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self.http.delete(&url);
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
                let text = retry_resp.text().await?;
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NO_CONTENT {
            let text = resp.text().await?;
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(())
    }

    /// Create a new calendar collection via MKCALENDAR.
    pub async fn mkcalendar(&self, path: &str, body: &str) -> Result<()> {
        let url = self.url(path);
        let method = reqwest::Method::from_bytes(b"MKCALENDAR").unwrap();

        let req = self
            .http
            .request(method.clone(), &url)
            .header("Content-Type", "application/xml")
            .body(body.to_string());
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on MKCALENDAR, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self
                .http
                .request(method, &url)
                .header("Content-Type", "application/xml")
                .body(body.to_string());
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            if !status.is_success() {
                let text = retry_resp.text().await?;
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(())
    }

    /// Modify properties on a CalDAV resource via PROPPATCH.
    pub async fn proppatch(&self, path: &str, body: &str) -> Result<()> {
        let url = self.url(path);
        let method = reqwest::Method::from_bytes(b"PROPPATCH").unwrap();

        let req = self
            .http
            .request(method.clone(), &url)
            .header("Content-Type", "application/xml")
            .body(body.to_string());
        let req = self.apply_auth(req).await?;
        let resp = req.send().await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            debug!("got 401 on PROPPATCH, forcing token refresh and retrying");
            self.invalidate_token().await;
            let retry = self
                .http
                .request(method, &url)
                .header("Content-Type", "application/xml")
                .body(body.to_string());
            let retry = self.apply_auth(retry).await?;
            let retry_resp = retry.send().await?;
            let status = retry_resp.status();
            if !status.is_success() {
                let text = retry_resp.text().await?;
                bail!("CalDAV error ({status}): {text}");
            }
            return Ok(());
        }

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(())
    }
}

#[cfg(test)]
impl CalDavClient {
    /// Create a client for unit tests (no real server connection).
    pub fn new_for_test() -> Self {
        Self {
            http: http_client(),
            server_url: "https://caldav.example.com/dav".into(),
            auth: CalDavAuth::Basic {
                username: "test".into(),
                password: "test".into(),
            },
            token_path: None,
            token: RwLock::new(None),
        }
    }
}
