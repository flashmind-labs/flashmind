//! Tailscale-specific error types.

// ---------------------------------------------------------------------------

/// Errors from Tailscale operations.
#[derive(Debug, thiserror::Error)]
pub enum TailscaleError {
    #[error("tailscale CLI not found")]
    NotInstalled,
    #[error("tailscale daemon not running")]
    NotRunning,
    #[error("funnel permission denied by Tailscale operator policy")]
    PermissionDenied,
    #[error("local API unavailable: {0}")]
    SocketError(String),
    #[error("{0}")]
    Api(String),
}
