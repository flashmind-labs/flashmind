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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: i64,
}

impl CachedToken {
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() >= self.expires_at - 60
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

pub fn load_token(path: &Path) -> Result<Option<CachedToken>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path).context("reading cached token")?;
    let token: CachedToken = serde_json::from_str(&data).context("parsing cached token")?;
    Ok(Some(token))
}

pub fn save_token(path: &Path, token: &CachedToken) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("creating token directory")?;
    }
    let data = serde_json::to_string_pretty(token)?;
    std::fs::write(path, data).context("writing cached token")?;
    debug!("persisted token to {}", path.display());
    Ok(())
}

pub fn parse_token_response(resp: &serde_json::Value) -> Result<CachedToken> {
    let access_token = resp["access_token"]
        .as_str()
        .context("missing access_token in token response")?
        .to_string();
    let refresh_token = resp["refresh_token"].as_str().map(|s| s.to_string());
    let expires_in = resp["expires_in"].as_i64().unwrap_or(3600);

    Ok(CachedToken {
        access_token,
        refresh_token,
        expires_at: Utc::now().timestamp() + expires_in,
    })
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
