//! Append-only JSONL log for cron job executions.
//!
//! Each time a job fires, the handler appends a [`CronLogEntry`] to
//! `~/.flashmind/cron.log`. The [`CronLog`] struct provides write and
//! query access so the agent can inspect recent runs.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

// ---------------------------------------------------------------------------
// CronLogEntry
// ---------------------------------------------------------------------------

/// A single recorded cron job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronLogEntry {
    /// Job ID that fired.
    pub job_id: uuid::Uuid,
    /// Task description from the job.
    pub task: String,
    /// When the job started executing.
    pub started_at: DateTime<Utc>,
    /// When execution finished (None if still running or crashed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Whether execution succeeded.
    pub success: bool,
    /// Error message if execution failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Session key if the job spawned an agent session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_key: Option<String>,
}

// ---------------------------------------------------------------------------
// CronLog
// ---------------------------------------------------------------------------

/// Append-only log backed by a JSONL file.
#[derive(Clone)]
pub struct CronLog {
    path: PathBuf,
}

impl CronLog {
    /// Create a log writer/reader for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Path to the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a completed entry to the log.
    pub async fn append(&self, entry: &CronLogEntry) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(entry)?;
        line.push('\n');

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        Ok(())
    }

    /// Read the most recent `n` entries (newest first).
    pub async fn recent(&self, n: usize) -> anyhow::Result<Vec<CronLogEntry>> {
        let content = match tokio::fs::read_to_string(&self.path).await {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let entries: Vec<CronLogEntry> = content
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str(line).ok())
            .take(n)
            .collect();

        Ok(entries)
    }

    /// Read entries for a specific job ID (newest first, capped at `n`).
    pub async fn for_job(&self, job_id: uuid::Uuid, n: usize) -> anyhow::Result<Vec<CronLogEntry>> {
        let content = match tokio::fs::read_to_string(&self.path).await {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let entries: Vec<CronLogEntry> = content
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str(line).ok())
            .filter(|e: &CronLogEntry| e.job_id == job_id)
            .take(n)
            .collect();

        Ok(entries)
    }
}
