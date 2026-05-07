//! Dual-mode Tailscale client: local API (Unix socket) with CLI fallback.

use crate::error::TailscaleError;
use crate::local_api::LocalApi;
use crate::status::{RawStatus, TailscaleStatus};
use crate::types::{PingResult, WhoisResult};

// ---------------------------------------------------------------------------

/// Client for the Tailscale daemon.
///
/// Tries the local API (Unix socket) first for all operations. Falls back to
/// the `tailscale` CLI when the socket is unavailable.
pub struct TailscaleClient {
    local_api: Option<LocalApi>,
}

impl TailscaleClient {
    /// Auto-detect: tries to find the local API socket, falls back to CLI-only.
    pub fn connect() -> Self {
        Self {
            local_api: LocalApi::detect(),
        }
    }

    /// Create a client using an explicit socket path.
    pub fn with_socket(path: std::path::PathBuf) -> Self {
        Self {
            local_api: Some(LocalApi::new(path)),
        }
    }

    /// CLI-only mode — never uses the Unix socket.
    pub fn cli_only() -> Self {
        Self { local_api: None }
    }

    /// Access the raw local API client, if available.
    pub fn local_api(&self) -> Option<&LocalApi> {
        self.local_api.as_ref()
    }

    /// Returns `true` if the local API socket is configured.
    pub fn has_local_api(&self) -> bool {
        self.local_api.is_some()
    }

    // -------------------------------------------------------------------
    // Methods with CLI fallback
    // -------------------------------------------------------------------

    /// Returns `true` if Tailscale is reachable (via socket or CLI).
    pub async fn is_available(&self) -> bool {
        if let Some(api) = &self.local_api
            && api.is_reachable().await
        {
            return true;
        }
        cli_is_available().await
    }

    /// Query full tailnet status.
    pub async fn status(&self) -> Result<TailscaleStatus, TailscaleError> {
        if let Some(api) = &self.local_api {
            match api.status().await {
                Ok(s) => return Ok(s),
                Err(TailscaleError::SocketError(e)) => {
                    tracing::debug!("local API failed, falling back to CLI: {e}");
                }
                Err(e) => return Err(e),
            }
        }
        cli_status().await
    }

    /// The hostname of this device on the tailnet.
    pub async fn device_name(&self) -> Result<String, TailscaleError> {
        Ok(self.status().await?.self_device.device_name)
    }

    /// The first Tailscale IP address of this device.
    pub async fn ip(&self) -> Result<String, TailscaleError> {
        let status = self.status().await?;
        status
            .self_device
            .tailscale_ips
            .into_iter()
            .next()
            .ok_or_else(|| TailscaleError::Api("no Tailscale IPs found".into()))
    }

    /// The MagicDNS name of this device.
    pub async fn dns_name(&self) -> Result<String, TailscaleError> {
        Ok(self.status().await?.self_device.dns_name)
    }

    // -------------------------------------------------------------------
    // Local API-only methods
    // -------------------------------------------------------------------

    /// Identify a Tailscale peer by IP:port (local API only).
    pub async fn whois(&self, addr: &str) -> Result<WhoisResult, TailscaleError> {
        self.require_local_api()?.whois(addr).await
    }

    /// Current daemon preferences (local API only).
    pub async fn prefs(&self) -> Result<serde_json::Value, TailscaleError> {
        self.require_local_api()?.prefs().await
    }

    /// Ping a Tailscale peer (local API only).
    pub async fn ping(&self, ip: &str) -> Result<PingResult, TailscaleError> {
        self.require_local_api()?.ping(ip, "disco").await
    }

    /// Current DNS configuration (local API only).
    pub async fn dns_config(&self) -> Result<serde_json::Value, TailscaleError> {
        self.require_local_api()?.dns_config().await
    }

    /// Domains for which TLS certs can be issued (local API only).
    pub async fn cert_domains(&self) -> Result<Vec<String>, TailscaleError> {
        self.require_local_api()?.cert_domains().await
    }

    // -------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------

    fn require_local_api(&self) -> Result<&LocalApi, TailscaleError> {
        self.local_api
            .as_ref()
            .ok_or_else(|| TailscaleError::SocketError("local API not available".into()))
    }
}

// ---------------------------------------------------------------------------
// CLI fallback functions
// ---------------------------------------------------------------------------

async fn cli_is_available() -> bool {
    tokio::process::Command::new("tailscale")
        .arg("version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn cli_status() -> Result<TailscaleStatus, TailscaleError> {
    let output = tokio::process::Command::new("tailscale")
        .args(["status", "--json"])
        .output()
        .await
        .map_err(|_| TailscaleError::NotInstalled)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("not running") || stderr.contains("not connected") {
            return Err(TailscaleError::NotRunning);
        }
        return Err(TailscaleError::Api(stderr.trim().to_string()));
    }

    let raw: RawStatus = serde_json::from_slice(&output.stdout)
        .map_err(|e| TailscaleError::Api(format!("failed to parse status JSON: {e}")))?;

    Ok(raw.into_status())
}
