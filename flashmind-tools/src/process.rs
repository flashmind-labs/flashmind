//! Background process management tool.
//!
//! Allows agents to interact with long-running processes started via the `exec`
//! tool with `background: true`. Supports reading stdout/stderr, writing to
//! stdin, and killing processes.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::Mutex;

use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

/// Buffered output lines collected by a background reader task.
type LineBuf = Arc<Mutex<Vec<String>>>;

/// A running background process with buffered output.
struct BackgroundProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout_buf: LineBuf,
    stderr_buf: LineBuf,
    /// How many lines of stdout have already been returned.
    stdout_cursor: usize,
    /// How many lines of stderr have already been returned.
    stderr_cursor: usize,
    command: String,
}

/// Shared registry of background processes, keyed by PID.
#[derive(Clone, Default)]
pub struct ProcessRegistry(Arc<Mutex<HashMap<u32, BackgroundProcess>>>);

impl ProcessRegistry {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())))
    }

    /// Kill all background processes (for cleanup in tests).
    #[cfg(test)]
    pub async fn kill_all(&self) {
        let mut map = self.0.lock().await;
        for (&pid, _) in map.iter() {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
        map.clear();
    }

    /// Register a newly spawned background process.
    pub async fn insert(&self, pid: u32, mut child: Child, command: String) {
        let stdout_buf: LineBuf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf: LineBuf = Arc::new(Mutex::new(Vec::new()));

        // Spawn reader tasks that buffer lines from stdout/stderr.
        if let Some(stdout) = child.stdout.take() {
            let buf = stdout_buf.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    buf.lock().await.push(line);
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let buf = stderr_buf.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    buf.lock().await.push(line);
                }
            });
        }

        let stdin = child.stdin.take();

        let mut map = self.0.lock().await;
        map.insert(
            pid,
            BackgroundProcess {
                child,
                stdin,
                stdout_buf,
                stderr_buf,
                stdout_cursor: 0,
                stderr_cursor: 0,
                command,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Process tool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ProcessArgs {
    /// Action to perform: "list", "poll", "write", "kill"
    action: String,
    /// PID of the process (required for poll/write/kill)
    pid: Option<u32>,
    /// Input to write to stdin (required for "write" action)
    input: Option<String>,
}

/// Background process management tool.
pub struct ProcessTool {
    pub registry: ProcessRegistry,
}

#[async_trait]
impl Tool for ProcessTool {
    fn name(&self) -> &str {
        "process"
    }

    fn description(&self) -> &str {
        "Manage background processes. Actions: list (show all), poll (read new stdout/stderr), write (send to stdin), kill (terminate)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "write", "kill"],
                    "description": "Action to perform"
                },
                "pid": {
                    "type": "integer",
                    "description": "Process ID (required for poll/write/kill)"
                },
                "input": {
                    "type": "string",
                    "description": "Text to write to process stdin (for 'write' action)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ProcessArgs = ctx.parse_args(self.name())?;

        match args.action.as_str() {
            "list" => self.list(ctx.tool_call_id).await,
            "poll" => {
                let pid = match args.pid {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            "pid is required for poll",
                        ));
                    }
                };
                self.poll(ctx.tool_call_id, pid).await
            }
            "write" => {
                let pid = match args.pid {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            "pid is required for write",
                        ));
                    }
                };
                let input = args.input.unwrap_or_default();
                self.write(ctx.tool_call_id, pid, &input).await
            }
            "kill" => {
                let pid = match args.pid {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            "pid is required for kill",
                        ));
                    }
                };
                self.kill(ctx.tool_call_id, pid).await
            }
            other => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Unknown action: {}. Use list/poll/write/kill", other),
            )),
        }
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
        match action {
            "list" => "Listing processes".to_string(),
            "kill" => {
                let pid = args.get("pid").and_then(|v| v.as_u64()).unwrap_or(0);
                format!("Killing process {}", pid)
            }
            _ => format!("Process: {}", action),
        }
    }
}

impl ProcessTool {
    async fn list(&self, call_id: &str) -> anyhow::Result<ToolResult> {
        let mut map = self.registry.0.lock().await;
        if map.is_empty() {
            return Ok(ToolResult::success(
                call_id,
                "No background processes running.",
            ));
        }

        // Check for exited processes and update status
        let mut lines = Vec::new();
        for (&pid, proc) in map.iter_mut() {
            let status = match proc.child.try_wait() {
                Ok(Some(exit)) => format!("exited ({})", exit.code().unwrap_or(-1)),
                Ok(None) => "running".to_string(),
                Err(e) => format!("error: {}", e),
            };
            lines.push(format!("PID {}: {} [{}]", pid, proc.command, status));
        }

        Ok(ToolResult::success(call_id, lines.join("\n")))
    }

    async fn poll(&self, call_id: &str, pid: u32) -> anyhow::Result<ToolResult> {
        let mut map = self.registry.0.lock().await;
        let proc = match map.get_mut(&pid) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::failure(
                    call_id,
                    format!("No process with PID {}", pid),
                ));
            }
        };

        // Check if process has exited
        let status = match proc.child.try_wait() {
            Ok(Some(exit)) => Some(exit.code().unwrap_or(-1)),
            _ => None,
        };

        // Drain new lines from the shared buffers
        let new_stdout = {
            let buf = proc.stdout_buf.lock().await;
            let new = buf[proc.stdout_cursor..].to_vec();
            proc.stdout_cursor = buf.len();
            new
        };

        let new_stderr = {
            let buf = proc.stderr_buf.lock().await;
            let new = buf[proc.stderr_cursor..].to_vec();
            proc.stderr_cursor = buf.len();
            new
        };

        let mut output = String::new();

        if !new_stdout.is_empty() {
            output.push_str("[stdout]\n");
            output.push_str(&new_stdout.join("\n"));
            output.push('\n');
        }

        if !new_stderr.is_empty() {
            output.push_str("[stderr]\n");
            output.push_str(&new_stderr.join("\n"));
            output.push('\n');
        }

        if let Some(code) = status {
            output.push_str(&format!("[exited with code {}]\n", code));
        } else if output.is_empty() {
            output.push_str("(no new output)\n");
        }

        Ok(ToolResult::success(call_id, output.trim()))
    }

    async fn write(&self, call_id: &str, pid: u32, input: &str) -> anyhow::Result<ToolResult> {
        let mut map = self.registry.0.lock().await;
        let proc = match map.get_mut(&pid) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::failure(
                    call_id,
                    format!("No process with PID {}", pid),
                ));
            }
        };

        let stdin = match proc.stdin.as_mut() {
            Some(s) => s,
            None => return Ok(ToolResult::failure(call_id, "Process has no stdin handle")),
        };

        let data = if input.ends_with('\n') {
            input.to_string()
        } else {
            format!("{}\n", input)
        };

        match stdin.write_all(data.as_bytes()).await {
            Ok(()) => Ok(ToolResult::success(call_id, "Written to stdin.")),
            Err(e) => Ok(ToolResult::failure(call_id, format!("Write error: {}", e))),
        }
    }

    async fn kill(&self, call_id: &str, pid: u32) -> anyhow::Result<ToolResult> {
        let mut map = self.registry.0.lock().await;
        let proc = match map.remove(&pid) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::failure(
                    call_id,
                    format!("No process with PID {}", pid),
                ));
            }
        };

        // Drain remaining output before killing
        let final_stdout = {
            let buf = proc.stdout_buf.lock().await;
            buf[proc.stdout_cursor..].to_vec()
        };
        let final_stderr = {
            let buf = proc.stderr_buf.lock().await;
            buf[proc.stderr_cursor..].to_vec()
        };

        // Kill the process
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }

        let mut output = format!("Process {} killed.", pid);

        if !final_stdout.is_empty() {
            output.push_str("\n[final stdout]\n");
            output.push_str(&final_stdout.join("\n"));
        }
        if !final_stderr.is_empty() {
            output.push_str("\n[final stderr]\n");
            output.push_str(&final_stderr.join("\n"));
        }

        Ok(ToolResult::success(call_id, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_empty() {
        let tool = ProcessTool {
            registry: ProcessRegistry::new(),
        };
        let result = crate::tests::execute_tool(&tool, "t", json!({"action": "list"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("No background"));
    }

    #[tokio::test]
    async fn test_poll_missing_pid() {
        let tool = ProcessTool {
            registry: ProcessRegistry::new(),
        };
        let result =
            crate::tests::execute_tool(&tool, "t", json!({"action": "poll", "pid": 99999}))
                .await
                .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("No process"));
    }

    #[tokio::test]
    async fn test_background_process_lifecycle() {
        let registry = ProcessRegistry::new();

        // Spawn a simple process
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c")
            .arg("echo hello; sleep 10")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::piped());

        let child = cmd.spawn().unwrap();
        let pid = child.id().unwrap();
        registry
            .insert(pid, child, "echo hello; sleep 10".into())
            .await;

        let tool = ProcessTool {
            registry: registry.clone(),
        };

        // List should show the process
        let result = crate::tests::execute_tool(&tool, "t", json!({"action": "list"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains(&pid.to_string()));

        // Wait a moment for output to buffer
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Poll should show stdout
        let result = crate::tests::execute_tool(&tool, "t", json!({"action": "poll", "pid": pid}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));

        // Kill the process
        let result = crate::tests::execute_tool(&tool, "t", json!({"action": "kill", "pid": pid}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("killed"));
    }
}
