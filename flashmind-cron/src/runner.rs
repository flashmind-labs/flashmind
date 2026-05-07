//! Async cron job runner that spawns per-job tasks and manages their lifecycle.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::job::{CronJob, JobSchedule};
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
    handles: Arc<tokio::sync::Mutex<HashMap<uuid::Uuid, CancellationToken>>>,
}

impl CronRunner {
    /// Create a new runner.
    pub fn new(
        registry: Arc<CronRegistry>,
        handler: Arc<dyn CronHandler>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            registry,
            handler,
            cancel,
            handles: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Load all enabled jobs and spawn a task for each. Blocks until the
    /// cancellation token is cancelled.
    pub async fn start(&self) -> anyhow::Result<()> {
        self.spawn_all().await?;

        self.cancel.cancelled().await;

        let handles = self.handles.lock().await;
        for token in handles.values() {
            token.cancel();
        }

        Ok(())
    }

    /// Re-read jobs from the store. Cancel tasks for removed/disabled jobs
    /// and spawn tasks for new/re-enabled ones.
    pub async fn reload(&self) -> anyhow::Result<()> {
        let jobs = self.registry.list().await?;
        let mut handles = self.handles.lock().await;

        let active_ids: std::collections::HashSet<uuid::Uuid> =
            jobs.iter().filter(|j| j.enabled).map(|j| j.id).collect();

        // Cancel removed or disabled jobs
        let to_remove: Vec<uuid::Uuid> = handles
            .keys()
            .filter(|id| !active_ids.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            if let Some(token) = handles.remove(&id) {
                token.cancel();
            }
        }

        // Spawn new jobs
        for job in &jobs {
            if job.enabled && !handles.contains_key(&job.id) {
                let child_token = self.cancel.child_token();
                self.spawn_job(job, child_token.clone());
                handles.insert(job.id, child_token);
            }
        }

        Ok(())
    }

    async fn spawn_all(&self) -> anyhow::Result<()> {
        let jobs = self.registry.list().await?;
        let mut handles = self.handles.lock().await;

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

    fn spawn_job(&self, job: &CronJob, token: CancellationToken) {
        let handler = self.handler.clone();
        let registry = self.registry.clone();
        let job = job.clone();

        tokio::spawn(async move {
            match &job.schedule {
                JobSchedule::Once(at) => {
                    run_once_job(&job, *at, &handler, &registry, &token).await;
                }
                JobSchedule::Cron(expr) => {
                    run_recurring_job(&job, expr, &handler, &registry, &token).await;
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
) {
    let now = chrono::Utc::now();
    if at > now {
        let delay = (at - now).to_std().unwrap_or_default();
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = token.cancelled() => return,
        }
    }

    if let Err(e) = handler.execute(job).await {
        tracing::error!(job_id = %job.id, error = %e, "one-shot job failed");
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

        let delay = (next - now).to_std().unwrap_or_default();
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = token.cancelled() => return,
        }

        if let Err(e) = handler.execute(&current_job).await {
            tracing::error!(job_id = %job.id, error = %e, "recurring job failed");
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
