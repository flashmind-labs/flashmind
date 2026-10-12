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
use reqwest::{Client, Method, RequestBuilder, Response, StatusCode};
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
                let refreshed = auth::refresh_token(creds, refresh).await?;

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
    async fn apply_auth(&self, req: RequestBuilder) -> Result<RequestBuilder> {
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

    /// Send an authenticated request, retrying once after a 401.
    ///
    /// `build` is called again for the retry so the request is rebuilt with
    /// fresh authentication.
    async fn send(&self, verb: &str, build: impl Fn() -> RequestBuilder) -> Result<Response> {
        let resp = self.apply_auth(build()).await?.send().await?;
        if resp.status() != StatusCode::UNAUTHORIZED {
            return Ok(resp);
        }
        debug!("got 401 on {verb}, forcing token refresh and retrying");
        self.invalidate_token().await;
        Ok(self.apply_auth(build()).await?.send().await?)
    }

    /// Return the response body, or an error if the status is not a success.
    async fn text_or_error(resp: Response) -> Result<String> {
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(text)
    }

    /// Return an error carrying the response body if the status is not a success.
    async fn ok_or_error(resp: Response) -> Result<()> {
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("CalDAV error ({status}): {text}");
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // CalDAV HTTP methods
    // -----------------------------------------------------------------------

    /// Send a PROPFIND request to discover resources and their properties.
    pub async fn propfind(&self, path: &str, depth: u8, body: &str) -> Result<String> {
        let url = self.url(path);
        let method = Method::from_bytes(b"PROPFIND").unwrap();

        let resp = self
            .send("PROPFIND", || {
                self.http
                    .request(method.clone(), &url)
                    .header("Content-Type", "application/xml")
                    .header("Depth", depth.to_string())
                    .body(body.to_string())
            })
            .await?;
        Self::text_or_error(resp).await
    }

    /// Send a REPORT request (calendar-query or calendar-multiget).
    pub async fn report(&self, path: &str, body: &str) -> Result<String> {
        let url = self.url(path);
        let method = Method::from_bytes(b"REPORT").unwrap();

        let resp = self
            .send("REPORT", || {
                self.http
                    .request(method.clone(), &url)
                    .header("Content-Type", "application/xml")
                    .header("Depth", "1")
                    .body(body.to_string())
            })
            .await?;
        Self::text_or_error(resp).await
    }

    /// Fetch raw iCalendar data for a single resource.
    pub async fn get_raw(&self, path: &str) -> Result<String> {
        let url = self.url(path);
        let resp = self.send("GET", || self.http.get(&url)).await?;
        Self::text_or_error(resp).await
    }

    /// Upload iCalendar data to a resource path.
    pub async fn put_ical(&self, path: &str, ical: &str) -> Result<()> {
        let url = self.url(path);

        let resp = self
            .send("PUT", || {
                self.http
                    .put(&url)
                    .header("Content-Type", "text/calendar; charset=utf-8")
                    .body(ical.to_string())
            })
            .await?;
        Self::ok_or_error(resp).await
    }

    /// Delete a CalDAV resource.
    pub async fn delete(&self, path: &str) -> Result<()> {
        let url = self.url(path);
        let resp = self.send("DELETE", || self.http.delete(&url)).await?;
        Self::ok_or_error(resp).await
    }

    /// Create a new calendar collection via MKCALENDAR.
    pub async fn mkcalendar(&self, path: &str, body: &str) -> Result<()> {
        let url = self.url(path);
        let method = Method::from_bytes(b"MKCALENDAR").unwrap();

        let resp = self
            .send("MKCALENDAR", || {
                self.http
                    .request(method.clone(), &url)
                    .header("Content-Type", "application/xml")
                    .body(body.to_string())
            })
            .await?;
        Self::ok_or_error(resp).await
    }

    /// Modify properties on a CalDAV resource via PROPPATCH.
    pub async fn proppatch(&self, path: &str, body: &str) -> Result<()> {
        let url = self.url(path);
        let method = Method::from_bytes(b"PROPPATCH").unwrap();

        let resp = self
            .send("PROPPATCH", || {
                self.http
                    .request(method.clone(), &url)
                    .header("Content-Type", "application/xml")
                    .body(body.to_string())
            })
            .await?;
        Self::ok_or_error(resp).await
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
