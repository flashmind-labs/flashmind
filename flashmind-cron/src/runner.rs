//! Async cron job runner that spawns per-job tasks and manages their lifecycle.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

pub use crate::log::CronLogEntry;

use crate::job::{CronJob, JobSchedule};
use crate::log::CronLog;
use crate::registry::CronRegistry;
use crate::schedule::CronSchedule;

// ---------------------------------------------------------------------------
// CronHandler
// ---------------------------------------------------------------------------

/// Callback invoked when a cron job fires.
#[async_trait]
pub trait CronHandler: Send + Sync + 'static {
    /// Execute the job's task. Errors are logged but do not stop the runner.
    async fn execute(&self, job: &CronJob) -> anyhow::Result<()>;
}

// ---------------------------------------------------------------------------
// CronRunner
// ---------------------------------------------------------------------------

/// Async scheduler that loads jobs from a [`CronRegistry`], spawns a tokio
/// task for each enabled job, and manages their lifecycle via cancellation
/// tokens.
pub struct CronRunner {
    registry: Arc<CronRegistry>,
    handler: Arc<dyn CronHandler>,
    cancel: CancellationToken,
    notify: Arc<Notify>,
    handles: Arc<AsyncMutex<HashMap<uuid::Uuid, CancellationToken>>>,
    log: Option<Arc<CronLog>>,
}

impl CronRunner {
    /// Create a new runner.
    pub fn new(
        registry: Arc<CronRegistry>,
        handler: Arc<dyn CronHandler>,
        cancel: CancellationToken,
    ) -> Self {
        let notify = registry.notify();
        Self {
            registry,
            handler,
            cancel,
            notify,
            handles: Arc::new(AsyncMutex::new(HashMap::new())),
            log: None,
        }
    }

    /// Attach an execution log. When set, every job execution is recorded.
    pub fn with_log(mut self, log: Arc<CronLog>) -> Self {
        self.log = Some(log);
        self
    }

    /// Load all enabled jobs and spawn a task for each. Re-spawns on
    /// registry mutations. Blocks until the cancellation token is cancelled.
    pub async fn start(&self) -> anyhow::Result<()> {
        tracing::info!("cron runner starting");
        self.spawn_all().await?;

        loop {
            tokio::select! {
                () = self.cancel.cancelled() => break,
                () = self.notify.notified() => {
                    tracing::debug!("cron registry changed, reloading jobs");
                    if let Err(e) = self.reload().await {
                        tracing::error!(error = %e, "failed to reload cron jobs");
                    }
                }
            }
        }

        tracing::info!("cron runner shutting down");
        let handles = self.handles.lock().await;
        for token in handles.values() {
            token.cancel();
        }

        Ok(())
    }

    /// Re-read jobs from the store. Cancels all existing tasks and re-spawns
    /// enabled jobs so that schedule/task edits take effect immediately.
    pub async fn reload(&self) -> anyhow::Result<()> {
        let jobs = self.registry.list().await?;
        let mut handles = self.handles.lock().await;

        // Cancel all existing tasks
        for token in handles.values() {
            token.cancel();
        }
        handles.clear();

        // Spawn enabled jobs with fresh state
        for job in &jobs {
            if !job.enabled {
                continue;
            }
            let child_token = self.cancel.child_token();
            self.spawn_job(job, child_token.clone());
            handles.insert(job.id, child_token);
        }

        Ok(())
    }

    async fn spawn_all(&self) -> anyhow::Result<()> {
        let jobs = self.registry.list().await?;
        let mut handles = self.handles.lock().await;

        let enabled_count = jobs.iter().filter(|j| j.enabled).count();
        tracing::info!(
            total = jobs.len(),
            enabled = enabled_count,
            "loaded cron jobs"
        );

        for job in &jobs {
            if !job.enabled {
                tracing::debug!(job_id = %job.id, task = %job.task, "skipping disabled job");
                continue;
            }
            tracing::info!(job_id = %job.id, task = %job.task, schedule = ?job.schedule, "spawning job");
            let child_token = self.cancel.child_token();
            self.spawn_job(job, child_token.clone());
            handles.insert(job.id, child_token);
        }

        Ok(())
    }

    fn spawn_job(&self, job: &CronJob, token: CancellationToken) {
        let handler = self.handler.clone();
        let registry = self.registry.clone();
        let log = self.log.clone();
        let job = job.clone();

        tokio::spawn(async move {
            match &job.schedule {
                JobSchedule::Once(at) => {
                    run_once_job(&job, *at, &handler, &registry, &token, log.as_deref()).await;
                }
                JobSchedule::Cron(expr) => {
                    run_recurring_job(&job, expr, &handler, &registry, &token, log.as_deref())
                        .await;
                }
                JobSchedule::OnWake { .. } => {
                    // OnWake jobs are triggered externally by the wake detector,
                    // not by the timer-based runner. Wait for cancellation.
                    token.cancelled().await;
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Per-job task functions
// ---------------------------------------------------------------------------

async fn run_once_job(
    job: &CronJob,
    at: chrono::DateTime<chrono::Utc>,
    handler: &Arc<dyn CronHandler>,
    registry: &Arc<CronRegistry>,
    token: &CancellationToken,
    log: Option<&CronLog>,
) {
    let now = chrono::Utc::now();
    if at > now {
        let delay = (at - now).to_std().unwrap_or_default();
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = token.cancelled() => return,
        }
    }

    tracing::info!(job_id = %job.id, task = %job.task, "executing one-shot cron job");
    let started_at = chrono::Utc::now();
    let result = handler.execute(job).await;
    let finished_at = chrono::Utc::now();

    let (success, error) = match &result {
        Ok(()) => {
            tracing::info!(job_id = %job.id, "one-shot job completed");
            (true, None)
        }
        Err(e) => {
            tracing::error!(job_id = %job.id, error = %e, "one-shot job failed");
            (false, Some(e.to_string()))
        }
    };

    if let Some(log) = log {
        let entry = CronLogEntry {
            job_id: job.id,
            task: job.task.clone(),
            started_at,
            finished_at: Some(finished_at),
            success,
            error,
            session_key: None,
        };
        if let Err(e) = log.append(&entry).await {
            tracing::warn!(error = %e, "failed to write cron log entry");
        }
    }

    if let Err(e) = registry.delete(job.id).await {
        tracing::error!(job_id = %job.id, error = %e, "failed to delete one-shot job");
    }
}

async fn run_recurring_job(
    job: &CronJob,
    expr: &str,
    handler: &Arc<dyn CronHandler>,
    registry: &Arc<CronRegistry>,
    token: &CancellationToken,
    log: Option<&CronLog>,
) {
    let schedule = match CronSchedule::parse(expr) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(job_id = %job.id, error = %e, "invalid cron expression");
            return;
        }
    };

    let mut current_job = job.clone();

    loop {
        let now = chrono::Utc::now();
        let Some(next) = schedule.next_after(&now) else {
            tracing::warn!(job_id = %job.id, "no future match found, stopping");
            break;
        };

        tracing::debug!(job_id = %job.id, next = %next, "sleeping until next fire");
        let delay = (next - now).to_std().unwrap_or_default();
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = token.cancelled() => return,
        }

        tracing::info!(job_id = %job.id, task = %current_job.task, "executing recurring cron job");
        let started_at = chrono::Utc::now();
        let result = handler.execute(&current_job).await;
        let finished_at = chrono::Utc::now();

        let (success, error) = match &result {
            Ok(()) => {
                tracing::info!(job_id = %job.id, elapsed_ms = (finished_at - started_at).num_milliseconds(), "recurring job completed");
                (true, None)
            }
            Err(e) => {
                tracing::error!(job_id = %job.id, error = %e, "recurring job failed");
                (false, Some(e.to_string()))
            }
        };

        if let Some(log) = log {
            let entry = CronLogEntry {
                job_id: job.id,
                task: current_job.task.clone(),
                started_at,
                finished_at: Some(finished_at),
                success,
                error,
                session_key: None,
            };
            if let Err(e) = log.append(&entry).await {
                tracing::warn!(error = %e, "failed to write cron log entry");
            }
        }

        current_job.last_run = Some(chrono::Utc::now());
        if let Err(e) = registry.update(&current_job).await {
            tracing::error!(job_id = %job.id, error = %e, "failed to update last_run");
        }

        if current_job.once {
            if let Err(e) = registry.delete(job.id).await {
                tracing::error!(job_id = %job.id, error = %e, "failed to delete once-job");
            }
            break;
        }
    }
}
