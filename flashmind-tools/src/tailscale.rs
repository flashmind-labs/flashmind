//! Tailscale Funnel helpers shared by webhook and http_serve tools.

use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{info, warn};

static FUNNEL_DENIED: AtomicBool = AtomicBool::new(false);

fn mark_denied_once(stderr: &str) {
    if !FUNNEL_DENIED.swap(true, Ordering::Relaxed) {
        warn!(
            "Tailscale Funnel disabled (operator permission denied): {}. Funnel routes will be skipped; services remain reachable on localhost.",
            stderr.trim()
        );
    }
}

pub fn funnel_denied() -> bool {
    FUNNEL_DENIED.load(Ordering::Relaxed)
}

/// Check if the `tailscale` CLI is available.
pub async fn tailscale_available() -> bool {
    tokio::process::Command::new("tailscale")
        .arg("version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Add a single Tailscale Funnel route for a path prefix.
///
/// Rejects the root path `/` to avoid hijacking the entire hostname.
pub async fn add_funnel_route(bind: &str, path: &str) -> Option<String> {
    if funnel_denied() {
        return None;
    }

    let prefix = path
        .trim_start_matches('/')
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(|s| format!("/{}", s))?;

    let target = format!("http://{}{}", bind, prefix);
    let output = tokio::process::Command::new("tailscale")
        .args(["funnel", "--bg", "--set-path", &prefix, &target])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            info!(path = %prefix, "Tailscale Funnel route added");

            if let Ok(status) = tokio::process::Command::new("tailscale")
                .args(["funnel", "status"])
                .output()
                .await
            {
                let stdout = String::from_utf8_lossy(&status.stdout);
                if let Some(url) = stdout
                    .lines()
                    .find_map(|l| l.split_whitespace().find(|w| w.starts_with("https://")))
                {
                    let url = url.trim_end_matches('/');
                    return Some(format!("{}{}", url, prefix));
                }
            }

            Some(format!("(funnel active at {})", prefix))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("Access denied") || stderr.contains("denied") {
                mark_denied_once(&stderr);
            } else {
                warn!(path = %prefix, "Tailscale Funnel failed: {}", stderr.trim());
            }
            None
        }
        Err(_) => None,
    }
}

/// Remove a Tailscale Funnel route for a path prefix.
pub async fn remove_funnel_route(path: &str) {
    let prefix = path
        .trim_start_matches('/')
        .split('/')
        .next()
        .map(|s| format!("/{}", s));

    if let Some(prefix) = prefix {
        let _ = tokio::process::Command::new("tailscale")
            .args(["funnel", "--set-path", &prefix, "off"])
            .output()
            .await;

        info!(path = %prefix, "Tailscale Funnel route removed");
    }
}
