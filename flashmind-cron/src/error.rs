//! Error types for the cron scheduling system.

/// Errors that can occur during cron scheduling operations.
#[derive(Debug, thiserror::Error)]
pub enum CronError {
    /// The cron expression string could not be parsed.
    #[error("invalid cron expression: {0}")]
    InvalidExpression(String),

    /// No job exists with the given ID.
    #[error("job not found: {0}")]
    NotFound(uuid::Uuid),

    /// The runner has reached its maximum number of concurrent jobs.
    #[error("maximum concurrent jobs reached ({0})")]
    MaxJobs(usize),

    /// A one-shot schedule refers to a time that has already passed.
    #[error("schedule is in the past")]
    PastSchedule,

    /// An error occurred in the storage backend.
    #[error("storage error: {0}")]
    Storage(#[from] anyhow::Error),
}
