//! CRUD wrapper around a [`CronStore`] for managing cron jobs.

use std::sync::Arc;

use chrono::{Timelike, Utc};
use tokio::sync::Notify;
use uuid::Uuid;

use crate::job::{CronJob, JobSchedule};
use crate::store::CronStore;

// ---------------------------------------------------------------------------
// CronRegistry
// ---------------------------------------------------------------------------

/// High-level API for creating, listing, updating, and deleting cron jobs.
///
/// Wraps a [`CronStore`] and handles ID generation and timestamp management.
/// Mutations notify the runner to reload via [`Notify`].
pub struct CronRegistry {
    store: Arc<dyn CronStore>,
    notify: Arc<Notify>,
}

impl CronRegistry {
    /// Create a new registry backed by the given store.
    pub fn new(store: Arc<dyn CronStore>) -> Self {
        Self {
            store,
            notify: Arc::new(Notify::new()),
        }
    }

    /// Returns a handle the runner can use to watch for mutations.
    pub fn notify(&self) -> Arc<Notify> {
        self.notify.clone()
    }

    /// Create a new job with a generated ID and current timestamp.
    pub async fn create(
        &self,
        schedule: JobSchedule,
        task: String,
        once: bool,
    ) -> anyhow::Result<CronJob> {
        let job = CronJob {
            id: Uuid::new_v4(),
            schedule,
            task,
            enabled: true,
            once,
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            last_run: None,
        };
        self.store.upsert(&job).await?;
        self.notify.notify_waiters();
        Ok(job)
    }

    /// List all stored jobs.
    pub async fn list(&self) -> anyhow::Result<Vec<CronJob>> {
        self.store.list().await
    }

    /// Get a single job by ID.
    pub async fn get(&self, id: Uuid) -> anyhow::Result<Option<CronJob>> {
        self.store.get(id).await
    }

    /// Update an existing job in the store.
    pub async fn update(&self, job: &CronJob) -> anyhow::Result<()> {
        self.store.upsert(job).await?;
        self.notify.notify_waiters();
        Ok(())
    }

    /// Returns all enabled `OnWake` jobs whose time-window and gap constraints
    /// are satisfied given the current local time and the sleep duration.
    pub async fn matching_wake_jobs(
        &self,
        sleep_duration_secs: u64,
    ) -> anyhow::Result<Vec<CronJob>> {
        let jobs = self.list().await?;
        let local_hour = chrono::Local::now().hour();
        Ok(jobs
            .into_iter()
            .filter(|j| {
                if !j.enabled {
                    return false;
                }
                match &j.schedule {
                    JobSchedule::OnWake {
                        from_hour,
                        min_gap_secs,
                        max_gap_secs,
                    } => {
                        if sleep_duration_secs < *min_gap_secs {
                            return false;
                        }
                        if let Some(max) = max_gap_secs
                            && sleep_duration_secs > *max
                        {
                            return false;
                        }
                        if let Some(hour) = from_hour
                            && local_hour < *hour
                        {
                            return false;
                        }
                        true
                    }
                    _ => false,
                }
            })
            .collect())
    }

    /// Delete a job by ID. Returns true if the job existed.
    pub async fn delete(&self, id: Uuid) -> anyhow::Result<bool> {
        let removed = self.store.delete(id).await?;
        if removed {
            self.notify.notify_waiters();
        }
        Ok(removed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::TomlCronStore;

    #[tokio::test]
    async fn test_registry_create_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(TomlCronStore::new(dir.path().join("cron.toml")));
        let registry = CronRegistry::new(store);

        let job = registry
            .create(
                JobSchedule::Cron("0 9 * * *".into()),
                "morning check".into(),
                false,
            )
            .await
            .unwrap();
        assert_eq!(job.task, "morning check");
        assert!(job.enabled);

        let jobs = registry.list().await.unwrap();
        assert_eq!(jobs.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_update() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(TomlCronStore::new(dir.path().join("cron.toml")));
        let registry = CronRegistry::new(store);

        let mut job = registry
            .create(JobSchedule::Cron("* * * * *".into()), "task".into(), false)
            .await
            .unwrap();

        job.enabled = false;
        registry.update(&job).await.unwrap();

        let fetched = registry.get(job.id).await.unwrap().unwrap();
        assert!(!fetched.enabled);
    }

    #[tokio::test]
    async fn test_registry_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(TomlCronStore::new(dir.path().join("cron.toml")));
        let registry = CronRegistry::new(store);

        let job = registry
            .create(JobSchedule::Cron("* * * * *".into()), "task".into(), false)
            .await
            .unwrap();

        assert!(registry.delete(job.id).await.unwrap());
        assert!(registry.get(job.id).await.unwrap().is_none());
    }
}
