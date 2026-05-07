//! Pluggable storage backend for cron jobs.
//!
//! Provides the [`CronStore`] trait for custom backends and a built-in
//! [`TomlCronStore`] that persists jobs to a TOML file.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::job::CronJob;

// ---------------------------------------------------------------------------
// CronStore trait
// ---------------------------------------------------------------------------

/// Async storage backend for cron jobs.
#[async_trait]
pub trait CronStore: Send + Sync {
    /// List all stored jobs.
    async fn list(&self) -> anyhow::Result<Vec<CronJob>>;

    /// Get a job by ID, returning `None` if it does not exist.
    async fn get(&self, id: uuid::Uuid) -> anyhow::Result<Option<CronJob>>;

    /// Insert or update a job.
    async fn upsert(&self, job: &CronJob) -> anyhow::Result<()>;

    /// Delete a job by ID. Returns true if a job was actually removed.
    async fn delete(&self, id: uuid::Uuid) -> anyhow::Result<bool>;
}

// ---------------------------------------------------------------------------
// TomlCronStore
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default)]
struct CronJobs {
    #[serde(default)]
    jobs: Vec<CronJob>,
}

/// File-backed cron job store using TOML serialization.
///
/// Each mutation reads the file, applies the change, and writes it back.
/// Suitable for low-throughput use cases (agent cron jobs).
pub struct TomlCronStore {
    path: PathBuf,
}

impl TomlCronStore {
    /// Create a new store backed by the given file path.
    ///
    /// The file does not need to exist yet; it will be created on the first write.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    async fn read_file(&self) -> anyhow::Result<CronJobs> {
        match tokio::fs::read_to_string(&self.path).await {
            Ok(contents) => {
                let jobs: CronJobs = toml::from_str(&contents)?;
                Ok(jobs)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(CronJobs::default()),
            Err(e) => Err(e.into()),
        }
    }

    async fn write_file(&self, data: &CronJobs) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(data)?;
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&self.path, contents).await?;
        Ok(())
    }
}

#[async_trait]
impl CronStore for TomlCronStore {
    async fn list(&self) -> anyhow::Result<Vec<CronJob>> {
        Ok(self.read_file().await?.jobs)
    }

    async fn get(&self, id: uuid::Uuid) -> anyhow::Result<Option<CronJob>> {
        let data = self.read_file().await?;
        Ok(data.jobs.into_iter().find(|j| j.id == id))
    }

    async fn upsert(&self, job: &CronJob) -> anyhow::Result<()> {
        let mut data = self.read_file().await?;
        if let Some(existing) = data.jobs.iter_mut().find(|j| j.id == job.id) {
            *existing = job.clone();
        } else {
            data.jobs.push(job.clone());
        }
        self.write_file(&data).await
    }

    async fn delete(&self, id: uuid::Uuid) -> anyhow::Result<bool> {
        let mut data = self.read_file().await?;
        let len_before = data.jobs.len();
        data.jobs.retain(|j| j.id != id);
        let removed = data.jobs.len() < len_before;
        if removed {
            self.write_file(&data).await?;
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
    use crate::job::JobSchedule;
    use chrono::Utc;

    fn make_job(task: &str) -> CronJob {
        CronJob {
            id: uuid::Uuid::new_v4(),
            schedule: JobSchedule::Cron("*/5 * * * *".into()),
            task: task.into(),
            enabled: true,
            once: false,
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            last_run: None,
        }
    }

    #[tokio::test]
    async fn test_toml_store_crud() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cron.toml");
        let store = TomlCronStore::new(&path);

        // Empty initially
        let jobs = store.list().await.unwrap();
        assert!(jobs.is_empty());

        // Insert
        let job = make_job("test task");
        let id = job.id;
        store.upsert(&job).await.unwrap();

        // Get
        let fetched = store.get(id).await.unwrap().unwrap();
        assert_eq!(fetched.task, "test task");

        // List
        let jobs = store.list().await.unwrap();
        assert_eq!(jobs.len(), 1);

        // Update
        let mut updated = fetched;
        updated.task = "updated task".into();
        store.upsert(&updated).await.unwrap();
        let fetched = store.get(id).await.unwrap().unwrap();
        assert_eq!(fetched.task, "updated task");

        // Delete
        assert!(store.delete(id).await.unwrap());
        assert!(store.get(id).await.unwrap().is_none());
        assert!(!store.delete(id).await.unwrap());
    }

    #[tokio::test]
    async fn test_toml_store_multiple_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let store = TomlCronStore::new(dir.path().join("cron.toml"));

        let j1 = make_job("job one");
        let j2 = make_job("job two");
        store.upsert(&j1).await.unwrap();
        store.upsert(&j2).await.unwrap();

        let jobs = store.list().await.unwrap();
        assert_eq!(jobs.len(), 2);

        store.delete(j1.id).await.unwrap();
        let jobs = store.list().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].task, "job two");
    }
}
