//! Cron job definition and schedule types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JobSchedule
// ---------------------------------------------------------------------------

/// When a job should fire: either on a recurring cron schedule or once at a
/// specific datetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobSchedule {
    /// Recurring schedule specified as a POSIX 5-field cron expression string.
    Cron(String),
    /// One-shot execution at a specific UTC datetime.
    Once(DateTime<Utc>),
}

// ---------------------------------------------------------------------------
// CronJob
// ---------------------------------------------------------------------------

/// A scheduled job with its metadata and execution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob<M = serde_json::Value> {
    /// Unique identifier for this job.
    pub id: uuid::Uuid,
    /// When this job should fire.
    pub schedule: JobSchedule,
    /// Description of the task to perform when the job fires.
    pub task: String,
    /// Whether this job is currently active.
    pub enabled: bool,
    /// If true, the job is deleted after its first execution.
    #[serde(default)]
    pub once: bool,
    /// Arbitrary metadata attached to this job.
    #[serde(default)]
    pub metadata: M,
    /// When this job was created.
    pub created_at: DateTime<Utc>,
    /// When this job last executed, if ever.
    pub last_run: Option<DateTime<Utc>>,
}
