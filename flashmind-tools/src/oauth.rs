//! Shared OAuth2 token persistence utilities.
//!
//! Provides [`CachedToken`] and file-based load/save used by both Google and
//! Microsoft auth implementations.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::debug;

// ---------------------------------------------------------------------------
// CachedToken
// ---------------------------------------------------------------------------

/// An OAuth2 access token persisted to disk for reuse across runs.
///
/// Shared by both Google and Microsoft auth flows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToken {
    /// The bearer token sent in `Authorization` headers.
    pub access_token: String,
    /// Long-lived token used to obtain a new access token after expiry.
    /// `None` for service account credentials (which mint fresh JWTs).
    pub refresh_token: Option<String>,
    /// Unix timestamp (seconds) when `access_token` expires.
    pub expires_at: i64,
}

impl CachedToken {
    /// Returns `true` if the token expires within the next 60 seconds.
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() >= self.expires_at - 60
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Load a cached OAuth token from disk, returning `None` if the file does not exist.
pub fn load_token(path: &Path) -> Result<Option<CachedToken>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path).context("reading cached token")?;
    let token: CachedToken = serde_json::from_str(&data).context("parsing cached token")?;
    Ok(Some(token))
}

/// Persist an OAuth token to disk, creating parent directories as needed.
pub fn save_token(path: &Path, token: &CachedToken) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("creating token directory")?;
    }
    let data = serde_json::to_string_pretty(token)?;
    std::fs::write(path, data).context("writing cached token")?;
    debug!("persisted token to {}", path.display());
    Ok(())
}

/// Deserialized OAuth2 token response from any provider.
///
/// Used by both Google and Microsoft auth flows to parse the JSON body
/// returned by the token endpoint.
#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<i64>,
}

impl From<TokenResponse> for CachedToken {
    fn from(resp: TokenResponse) -> Self {
        Self {
            access_token: resp.access_token,
            refresh_token: resp.refresh_token,
            expires_at: Utc::now().timestamp() + resp.expires_in.unwrap_or(3600),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_expiry() {
        let fresh = CachedToken {
            access_token: "tok".into(),
            refresh_token: None,
            expires_at: Utc::now().timestamp() + 3600,
        };
        assert!(!fresh.is_expired());

        let stale = CachedToken {
            access_token: "tok".into(),
            refresh_token: None,
            expires_at: Utc::now().timestamp() - 10,
        };
        assert!(stale.is_expired());
    }
}
