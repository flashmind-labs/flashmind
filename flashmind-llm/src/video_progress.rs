//! Progress lines for video generation jobs that poll a provider for minutes.

use std::time::Duration;

use tokio::time::Instant;

/// Minimum gap between progress lines for a job whose status has not changed.
const REPEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Tracks a polled video job so each status is reported once, with elapsed time.
pub(crate) struct JobProgress {
    started: Instant,
    reported: Option<(String, Instant)>,
}

impl JobProgress {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            reported: None,
        }
    }

    /// A line describing `status`, or `None` when the same status was reported
    /// less than [`REPEAT_INTERVAL`] ago.
    pub(crate) fn line(&mut self, status: &str) -> Option<String> {
        let status = status.to_ascii_lowercase();
        let now = Instant::now();
        if let Some((last, at)) = &self.reported
            && *last == status
            && now.duration_since(*at) < REPEAT_INTERVAL
        {
            return None;
        }
        self.reported = Some((status.clone(), now));
        Some(format!(
            "Video generation {}... ({} elapsed)",
            status.replace('_', " "),
            format_secs(now.duration_since(self.started).as_secs())
        ))
    }
}

/// Whole seconds as `12s`, `3m 15s` or `1h 02m 03s`.
fn format_secs(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {:02}s", secs / 60, secs % 60),
        _ => format!("{}h {:02}m {:02}s", secs / 3600, secs / 60 % 60, secs % 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_each_status_once() {
        let mut progress = JobProgress::new();
        let first = progress.line("in_progress").unwrap();
        assert!(first.starts_with("Video generation in progress... ("));
        assert_eq!(progress.line("in_progress"), None);
        assert!(progress.line("completed").unwrap().contains("completed"));
    }

    #[test]
    fn formats_short_and_long_durations() {
        assert_eq!(format_secs(0), "0s");
        assert_eq!(format_secs(59), "59s");
        assert_eq!(format_secs(75), "1m 15s");
        assert_eq!(format_secs(3723), "1h 02m 03s");
    }
}
