//! Generic access to Meshy tasks that have no dedicated client method.

use anyhow::{Context, Result, bail};
use reqwest::Method;
use serde_json::Value;
use tokio::time::Instant;

use super::{JOB_TIMEOUT, ThreeDClient, ThreeDProvider, response_json};

/// A finished Meshy task.
#[derive(Debug)]
pub struct MeshyTaskOutput {
    /// Task ID, usable as `input_task_id` or `rig_task_id` in later requests.
    pub id: String,
    /// Credits charged after refunds, when reported.
    pub consumed_credits: Option<u64>,
    /// The complete task object, including download URLs.
    pub task: Value,
}

impl ThreeDClient {
    /// Submit a Meshy task to `path` and wait for it to succeed.
    /// `on_task` receives the task ID once it is created.
    pub async fn meshy_task(
        &self,
        path: &str,
        body: Value,
        mut on_task: impl FnMut(&str),
    ) -> Result<MeshyTaskOutput> {
        self.require_meshy()?;
        let deadline = Instant::now() + JOB_TIMEOUT;
        let id = self.meshy_submit(path, body).await?;
        on_task(&id);
        let task = self.meshy_wait_json(path, &id, deadline).await?;
        Ok(MeshyTaskOutput {
            consumed_credits: task["consumed_credits"].as_u64(),
            id,
            task,
        })
    }

    /// GET a Meshy path that returns JSON immediately, such as a catalog or balance.
    pub async fn meshy_get(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        self.require_meshy()?;
        let mut url = self.endpoint(path);
        if !query.is_empty() {
            url.query_pairs_mut()
                .extend_pairs(query.iter().map(|(key, value)| (*key, value.as_str())));
        }
        let response = self
            .authed(Method::GET, url)
            .send()
            .await
            .map_err(reqwest::Error::without_url)
            .context("Meshy request failed")?;
        response_json(response, "Meshy request").await
    }

    fn require_meshy(&self) -> Result<()> {
        if !matches!(self.provider, ThreeDProvider::Meshy) {
            bail!("this operation is only supported for Meshy");
        }
        Ok(())
    }
}
