//! Helper functions for formatting Kubernetes objects.
//!
//! These utilities turn raw k8s-openapi structs into human-readable strings
//! similar to `kubectl` output.

use chrono::Utc;
use k8s_openapi::api::core::v1::Pod;

// ---------------------------------------------------------------------------
// Age formatting
// ---------------------------------------------------------------------------

/// Format an RFC 3339 timestamp as a human-friendly relative age string.
///
/// Returns values like `"5m"`, `"2h"`, `"3d"`, or `"unknown"` if parsing
/// fails.
pub fn format_age(timestamp: &str) -> String {
    let Ok(ts) = chrono::DateTime::parse_from_rfc3339(timestamp) else {
        return "unknown".into();
    };
    let duration = Utc::now().signed_duration_since(ts);

    if duration.num_days() > 0 {
        format!("{}d", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{}h", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{}m", duration.num_minutes())
    } else {
        format!("{}s", duration.num_seconds().max(0))
    }
}

// ---------------------------------------------------------------------------
// Pod status
// ---------------------------------------------------------------------------

/// Derive the display status of a pod (e.g. `Running`, `Pending`,
/// `CrashLoopBackOff`).
///
/// Mirrors the logic `kubectl` uses: inspect container waiting reasons first,
/// then fall back to the phase field.
pub fn pod_status(pod: &Pod) -> String {
    let status = match &pod.status {
        Some(s) => s,
        None => return "Unknown".into(),
    };

    // Check container statuses for waiting reasons (e.g. CrashLoopBackOff).
    if let Some(containers) = &status.container_statuses {
        for cs in containers {
            if let Some(state) = &cs.state {
                if let Some(waiting) = &state.waiting
                    && let Some(reason) = &waiting.reason
                {
                    return reason.clone();
                }
                if let Some(terminated) = &state.terminated
                    && let Some(reason) = &terminated.reason
                {
                    return reason.clone();
                }
            }
        }
    }

    status.phase.clone().unwrap_or_else(|| "Unknown".into())
}

// ---------------------------------------------------------------------------
// Container ready count
// ---------------------------------------------------------------------------

/// Count how many containers in a pod are ready vs total.
///
/// Returns `(ready, total)`.
pub fn container_ready_count(pod: &Pod) -> (usize, usize) {
    let Some(status) = &pod.status else {
        let total = pod.spec.as_ref().map(|s| s.containers.len()).unwrap_or(0);
        return (0, total);
    };

    let total = pod.spec.as_ref().map(|s| s.containers.len()).unwrap_or(0);

    let ready = status
        .container_statuses
        .as_ref()
        .map(|cs| cs.iter().filter(|c| c.ready).count())
        .unwrap_or(0);

    (ready, total)
}

// ---------------------------------------------------------------------------
// Restart count
// ---------------------------------------------------------------------------

/// Total restart count across all containers in a pod.
pub fn total_restarts(pod: &Pod) -> i32 {
    pod.status
        .as_ref()
        .and_then(|s| s.container_statuses.as_ref())
        .map(|cs| cs.iter().map(|c| c.restart_count).sum())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_age_seconds() {
        let now = Utc::now();
        let ts = now.to_rfc3339();
        let age = format_age(&ts);
        assert!(age.ends_with('s'), "expected seconds, got {age}");
    }

    #[test]
    fn format_age_invalid() {
        assert_eq!(format_age("not-a-date"), "unknown");
    }
}
