//! Tailscale Funnel route management.
//!
//! Wraps `tailscale funnel` CLI commands to add, remove, and query Funnel
//! routes. Caches permission-denied state to avoid repeated failed attempts.

use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------

static FUNNEL_DENIED: AtomicBool = AtomicBool::new(false);

/// Manages Tailscale Funnel routes for exposing local services to the internet.
pub struct FunnelManager;

impl FunnelManager {
    /// Add a Funnel route mapping `path` to a local `bind` address.
    ///
    /// Rejects the root path `/` to avoid hijacking the entire hostname.
    /// Returns the public HTTPS URL on success, or `None` if denied or failed.
    pub async fn add_route(bind: &str, path: &str) -> Option<String> {
        if Self::is_denied() {
            return None;
        }

        let prefix = extract_prefix(path)?;
        let target = format!("http://{}{}", bind, prefix);

        let output = tokio::process::Command::new("tailscale")
            .args(["funnel", "--bg", "--set-path", &prefix, &target])
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => {
                tracing::info!(path = %prefix, "Tailscale Funnel route added");
                Self::get_url(&prefix)
                    .await
                    .or_else(|| Some(format!("(funnel active at {})", prefix)))
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                if stderr.contains("Access denied") || stderr.contains("denied") {
                    mark_denied_once(&stderr);
                } else {
                    tracing::warn!(path = %prefix, "Tailscale Funnel failed: {}", stderr.trim());
                }
                None
            }
            Err(_) => None,
        }
    }

    /// Remove a Funnel route for the given path prefix.
    pub async fn remove_route(path: &str) {
        let Some(prefix) = extract_prefix(path) else {
            return;
        };

        let _ = tokio::process::Command::new("tailscale")
            .args(["funnel", "--set-path", &prefix, "off"])
            .output()
            .await;

        tracing::info!(path = %prefix, "Tailscale Funnel route removed");
    }

    /// Query the current Funnel status and extract the public URL for `path`.
    pub async fn get_url(path: &str) -> Option<String> {
        let prefix = extract_prefix(path)?;

        let status = tokio::process::Command::new("tailscale")
            .args(["funnel", "status"])
            .output()
            .await
            .ok()?;

        let stdout = String::from_utf8_lossy(&status.stdout);
        let base_url = stdout
            .lines()
            .find_map(|l| l.split_whitespace().find(|w| w.starts_with("https://")))?;

        let base_url = base_url.trim_end_matches('/');
        Some(format!("{}{}", base_url, prefix))
    }

    /// Returns `true` if Funnel access has been denied by Tailscale policy.
    pub fn is_denied() -> bool {
        FUNNEL_DENIED.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------

fn mark_denied_once(stderr: &str) {
    if !FUNNEL_DENIED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "Tailscale Funnel disabled (operator permission denied): {}. \
             Funnel routes will be skipped; services remain reachable on localhost.",
            stderr.trim()
        );
    }
}

/// Extract the first path segment as `/segment`. Returns `None` for root `/`.
fn extract_prefix(path: &str) -> Option<String> {
    path.trim_start_matches('/')
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(|s| format!("/{}", s))
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_prefix_simple() {
        assert_eq!(extract_prefix("/webhooks"), Some("/webhooks".into()));
    }

    #[test]
    fn extract_prefix_nested() {
        assert_eq!(extract_prefix("/api/v1/hooks"), Some("/api".into()));
    }

    #[test]
    fn extract_prefix_no_leading_slash() {
        assert_eq!(extract_prefix("hooks/abc"), Some("/hooks".into()));
    }

    #[test]
    fn extract_prefix_root_rejected() {
        assert_eq!(extract_prefix("/"), None);
    }

    #[test]
    fn extract_prefix_empty_rejected() {
        assert_eq!(extract_prefix(""), None);
    }

    #[test]
    fn denied_flag_default() {
        // Reset for test isolation — in practice this is process-global.
        FUNNEL_DENIED.store(false, Ordering::Relaxed);
        assert!(!FunnelManager::is_denied());
    }
}
