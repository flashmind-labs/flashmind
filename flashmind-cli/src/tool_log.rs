//! `--log-all-tools`: append every tool call to a file before it runs.

use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use flashmind_types::ToolCall;
use flashmind_types::tool::{GateDecision, ToolGate};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

type AsyncMutex<T> = tokio::sync::Mutex<T>;

/// A gate that records each call as one JSON line and always lets it run.
/// The line is flushed before the call starts, so a call that kills the
/// process is still in the file.
pub struct ToolCallLog {
    file: AsyncMutex<File>,
}

impl ToolCallLog {
    /// Open `path` for appending, creating it if needed.
    pub async fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("cannot open tool log {}", path.display()))?;
        Ok(Self {
            file: AsyncMutex::new(file),
        })
    }

    async fn write(&self, call: &ToolCall) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(&serde_json::json!({
            "time": chrono::Utc::now().to_rfc3339(),
            "pid": std::process::id(),
            "id": call.id,
            "tool": call.name,
            "arguments": call.arguments,
        }))?;
        line.push(b'\n');
        let mut file = self.file.lock().await;
        file.write_all(&line).await?;
        file.flush().await
    }
}

#[async_trait]
impl ToolGate for ToolCallLog {
    async fn check(&self, call: &ToolCall) -> GateDecision {
        if let Err(error) = self.write(call).await {
            tracing::warn!(%error, tool = %call.name, "cannot write tool log");
        }
        GateDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, command: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "exec".into(),
            arguments: serde_json::json!({ "command": command }),
        }
    }

    #[tokio::test]
    async fn logs_each_call_as_a_json_line_and_allows_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tools.jsonl");
        let log = ToolCallLog::open(&path).await.unwrap();

        assert!(matches!(
            log.check(&call("a", "ls")).await,
            GateDecision::Allow
        ));
        assert!(matches!(
            log.check(&call("b", "pkill flsh")).await,
            GateDecision::Allow
        ));

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["id"], "b");
        assert_eq!(lines[1]["tool"], "exec");
        assert_eq!(lines[1]["arguments"]["command"], "pkill flsh");
        assert_eq!(lines[1]["pid"], std::process::id());
    }

    #[tokio::test]
    async fn appends_to_an_existing_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tools.jsonl");
        std::fs::write(&path, "earlier\n").unwrap();

        let log = ToolCallLog::open(&path).await.unwrap();
        log.check(&call("a", "ls")).await;

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("earlier\n"));
        assert_eq!(text.lines().count(), 2);
    }
}
