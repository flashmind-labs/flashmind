//! Shell command execution tools — `bash`, `process_start`, `process_write`, `process_kill`.
//!
//! Provides [`BashTool`] for synchronous command execution with configurable timeout,
//! and process management tools (`ProcessStartTool`, `ProcessWriteTool`, etc.)
//! for long-running background processes registered in a [`ProcessRegistry`].
//!
//! # Security
//!
//! - Commands are subject to [`ForbiddenCmd`](flashmind_types::tool::ForbiddenCmd) patterns checked by the tool registry
//! - Protected paths (e.g., `/etc/shadow`, `~/.ssh/`) are validated before execution
//! - Unix commands are run in a new process group (via `setsid`) so child processes are killed on timeout
//! - Default timeout: 600 seconds; maximum: 1800 seconds

use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt::{self, Debug, Formatter};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use crate::process::ProcessRegistry;
use crate::protected::ProtectedPaths;
use flashmind_types::MIN_SECRET_REDACT_LEN;
use flashmind_types::tool::{CommandAllowList, InterruptPayload, ToolContext};
use flashmind_types::tool::{Tool, ToolResult};

/// Guard that kills the child's process group on drop unless the command
/// completed.
struct ChildGuard {
    pid: Option<u32>,
}

impl ChildGuard {
    fn new(child: &Child) -> Self {
        Self { pid: child.id() }
    }

    fn completed(&mut self) {
        self.pid = None;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        // Kill the entire process group (created by setsid) so child
        // processes don't outlive the command.
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
}

fn record_exec(start: Instant) {
    metrics::counter!("tools.exec.calls").increment(1);
    metrics::histogram!("tools.exec.duration_seconds").record(start.elapsed().as_secs_f64());
}

const DEFAULT_TIMEOUT_SECS: u64 = 600;
const MAX_TIMEOUT_SECS: u64 = 1800;

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// CommandApproval — typed interrupt payload for command permission prompts
// ---------------------------------------------------------------------------

/// Interrupt payload emitted when a command is not pre-approved.
///
/// The CLI downcasts this to show the approval prompt; after the user
/// approves, the LLM re-calls `exec` and the exact command passes the allowlist.
pub struct CommandApproval {
    pub command: String,
    allowlist: Arc<dyn CommandAllowList>,
}

impl Debug for CommandApproval {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandApproval")
            .field("command", &self.command)
            .finish_non_exhaustive()
    }
}

impl CommandApproval {
    /// Approve this exact command for the current session.
    pub fn approve_for_session(&self) {
        self.allowlist
            .add_session_pattern(&glob::Pattern::escape(&self.command));
    }
}

impl InterruptPayload for CommandApproval {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn display_output(&self) -> String {
        format!(
            "Command '{}' requires approval. The user will be prompted.",
            self.command
        )
    }
}

// ---------------------------------------------------------------------------
// BashTool
// ---------------------------------------------------------------------------

/// Shell command execution tool.
pub struct BashTool {
    pub protected: Arc<ProtectedPaths>,
    pub secrets: Vec<String>,
    pub process_registry: ProcessRegistry,
    pub forbidden_cmds: Vec<flashmind_types::tool::ForbiddenCmd>,
    pub allowlist: Option<Arc<dyn CommandAllowList>>,
}

fn redact_secrets(output: &str, secrets: &[String]) -> String {
    let mut result = output.to_string();
    for secret in secrets {
        if secret.len() >= MIN_SECRET_REDACT_LEN {
            result = result.replace(secret.as_str(), "[REDACTED]");
        }
    }
    result
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "exec"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Returns stdout+stderr combined. \
        Non-zero exit codes report as failure but output is still returned — \
        check the output to determine actual success. Set background=true for long-running processes \
        (returns PID, use 'process' tool to manage). \
        Foreground commands time out after 10 minutes by default (override with timeout_secs, max 1800)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background and return PID immediately. Use 'process' tool to interact with it."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Kill the foreground command after this many seconds (default 600, max 1800). Ignored when background=true."
                }
            },
            "required": ["command"]
        })
    }

    fn max_output_bytes(&self) -> usize {
        128 * 1024
    }

    fn max_output_lines(&self) -> usize {
        1000
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = Instant::now();
        let args: BashArgs = ctx.parse_args(self.name())?;

        for rule in &self.forbidden_cmds {
            if let Ok(re) = regex::Regex::new(&rule.command)
                && re.is_match(&args.command)
            {
                tracing::warn!(command = %args.command, reason = %rule.reason, "forbidden command blocked");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Forbidden: {}", rule.reason),
                ));
            }
        }

        if let Some(ref allowlist) = self.allowlist
            && !allowlist.is_allowed(&args.command)
        {
            return Ok(ToolResult::interrupt(
                ctx.tool_call_id,
                Arc::new(CommandApproval {
                    command: args.command.clone(),
                    allowlist: allowlist.clone(),
                }),
            ));
        }

        let command = if args.command.trim_start().starts_with("find ") {
            format!("timeout 15 {}", args.command)
        } else {
            args.command.clone()
        };

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&command)
            .env("NO_COLOR", "1")
            // Prevent interactive editors from blocking — git rebase, merge, etc.
            .env("GIT_EDITOR", "true")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("EDITOR", "true")
            .env("VISUAL", "true");

        // Always pipe stdin so interactive commands get EOF instead of hanging
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped());

        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }

        if let Some(ref wd) = ctx.working_dir {
            cmd.current_dir(wd);
        }

        if args.background {
            record_exec(start);
            return match cmd.spawn() {
                Ok(child) => {
                    let pid = child.id().unwrap_or(0);
                    tracing::debug!(pid, "exec: background process spawned");
                    self.process_registry.insert(pid, child, args.command).await;
                    Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!(
                            "Background process started (PID: {}). Use the 'process' tool to interact with it.",
                            pid
                        ),
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "exec: failed to spawn background command");
                    Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Error spawning command: {}", e),
                    ))
                }
            };
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error spawning command: {}", e),
                ));
            }
        };
        let mut guard = ChildGuard::new(&child);

        let timeout_secs = args
            .timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        let timeout = Duration::from_secs(timeout_secs);

        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        let stderr_task = tokio::spawn(async move {
            let mut stderr_lines = Vec::new();
            if let Some(stderr) = child_stderr {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    stderr_lines.push(line);
                }
            }
            stderr_lines
        });

        let mut stdout = String::new();
        let mut stdout_reader = child_stdout.map(|s| BufReader::new(s).lines());

        // Read stdout until it closes, racing cancellation and the timeout.
        loop {
            tokio::select! {
                biased;
                _ = ctx.cancel_token().cancelled() => {
                    record_exec(start);
                    return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
                }
                _ = tokio::time::sleep(timeout) => {
                    record_exec(start);
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!(
                            "Command timed out after {}s and was killed. Re-run with a larger timeout_secs, or use background=true for long-running processes.",
                            timeout_secs
                        ),
                    ));
                }
                line = async {
                    match stdout_reader.as_mut() {
                        Some(r) => r.next_line().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match line {
                        Ok(Some(line)) => {
                            ctx.progress(&line);
                            stdout.push_str(&line);
                            stdout.push('\n');
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::debug!(error = %e, "exec: stdout read error");
                            break;
                        }
                    }
                }
            }
        }

        // Close our end of the pipe so a child still writing gets SIGPIPE
        // instead of blocking, then wait for it to exit.
        drop(stdout_reader);
        let exit_code = match child.wait().await {
            Ok(status) => status.code().unwrap_or(-1),
            Err(e) => {
                tracing::warn!(error = %e, "exec: command execution error");
                record_exec(start);
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error executing command: {e}"),
                ));
            }
        };
        let stderr_lines = stderr_task.await.unwrap_or_default();
        guard.completed();
        let stderr = stderr_lines.join("\n");

        let combined = if stderr.is_empty() {
            stdout
        } else {
            format!("{}\n[stderr]\n{}", stdout, stderr)
        };

        let output_text = crate::utils::strip_ansi(&redact_secrets(
            &format!("[exit code: {}]\n{}", exit_code, combined.trim()),
            &self.secrets,
        ));
        record_exec(start);
        if exit_code == 0 {
            Ok(ToolResult::success(ctx.tool_call_id, output_text))
        } else {
            Ok(ToolResult::failure(ctx.tool_call_id, output_text))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        format!("Running `{command}`")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    #[derive(Default)]
    struct TestAllowList(Mutex<Vec<glob::Pattern>>);

    impl CommandAllowList for TestAllowList {
        fn is_allowed(&self, command: &str) -> bool {
            self.0
                .lock()
                .unwrap()
                .iter()
                .any(|pattern| pattern.matches(command))
        }

        fn add_session_pattern(&self, pattern: &str) {
            self.0
                .lock()
                .unwrap()
                .push(glob::Pattern::new(pattern).unwrap());
        }
    }

    #[tokio::test]
    async fn approval_allows_only_the_exact_command_and_then_executes() {
        let policy = Arc::new(TestAllowList::default());
        let mut tool = make_tool();
        tool.allowlist = Some(policy.clone());
        let args = json!({ "command": "printf '%s' 'file[1]*?'" });
        let result = crate::tests::execute_tool(&tool, "test-id", args.clone())
            .await
            .unwrap();
        assert!(result.is_interrupt());
        let approval = result
            .payload()
            .unwrap()
            .as_any()
            .downcast_ref::<CommandApproval>()
            .unwrap();
        assert!(!policy.is_allowed(&approval.command));
        approval.approve_for_session();
        assert!(policy.is_allowed(&approval.command));
        assert!(!policy.is_allowed("printf '%s' 'file1abcx'"));
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(result.is_success());
        assert_eq!(result.output(), "[exit code: 0]\nfile[1]*?");
    }

    fn make_tool() -> BashTool {
        let dir = tempdir().unwrap();
        BashTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            secrets: Vec::new(),
            process_registry: ProcessRegistry::new(),
            forbidden_cmds: Vec::new(),
            allowlist: None,
        }
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = make_tool();
        let args = json!({ "command": "echo hello" });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let tool = make_tool();
        let args = json!({ "command": "exit 1" });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("exit code: 1"));
    }

    #[tokio::test]
    async fn test_bash_working_dir() {
        let dir = tempdir().unwrap();
        let tool = make_tool();
        let args = json!({
            "command": "pwd",
            "working_dir": dir.path().to_str().unwrap()
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
    }

    #[test]
    fn test_redact_secrets() {
        let secrets = vec!["sk-abc123456789".to_string(), "xoxb-my-token".to_string()];
        let output = "API key: sk-abc123456789\nToken: xoxb-my-token\nSafe: hello";
        let redacted = redact_secrets(output, &secrets);
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("sk-abc123456789"));
        assert!(!redacted.contains("xoxb-my-token"));
        assert!(redacted.contains("Safe: hello"));
    }

    #[test]
    fn test_redact_skips_short_secrets() {
        let secrets = vec!["short".to_string()];
        let output = "value: short";
        let redacted = redact_secrets(output, &secrets);
        assert_eq!(redacted, output);
    }

    #[tokio::test]
    async fn test_bash_secrets_redacted_in_output() {
        let dir = tempdir().unwrap();
        let tool = BashTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            secrets: vec!["supersecretapikey123".to_string()],
            process_registry: ProcessRegistry::new(),
            forbidden_cmds: Vec::new(),
            allowlist: None,
        };
        let args = json!({ "command": "echo supersecretapikey123" });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(result.is_success());
        assert!(!result.output().contains("supersecretapikey123"));
        assert!(result.output().contains("[REDACTED]"));
    }

    #[tokio::test]
    async fn test_bash_forbidden_command_blocked() {
        let dir = tempdir().unwrap();
        let tool = BashTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            secrets: Vec::new(),
            process_registry: ProcessRegistry::new(),
            forbidden_cmds: vec![flashmind_types::tool::ForbiddenCmd {
                command: "^cat".to_string(),
                reason: "Use file_read tool instead".to_string(),
                reconsider: false,
            }],
            allowlist: None,
        };
        let args = json!({ "command": "cat /etc/hosts" });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("Forbidden"));
    }

    #[tokio::test]
    async fn test_bash_forbidden_allows_non_matching() {
        let dir = tempdir().unwrap();
        let tool = BashTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            secrets: Vec::new(),
            process_registry: ProcessRegistry::new(),
            forbidden_cmds: vec![flashmind_types::tool::ForbiddenCmd {
                command: "^cat".to_string(),
                reason: "Use file_read".to_string(),
                reconsider: false,
            }],
            allowlist: None,
        };
        let args = json!({ "command": "echo hello" });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_bash_timeout_kills_command() {
        let tool = make_tool();
        let args = json!({
            "command": "sleep 30",
            "timeout_secs": 1
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.output().contains("timed out"));
    }

    #[tokio::test]
    async fn test_bash_background() {
        let tool = make_tool();
        let args = json!({
            "command": "echo bg_test; sleep 60",
            "background": true
        });
        let result = crate::tests::execute_tool(&tool, "test-id", args)
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains("PID"));
        assert!(result.output().contains("Background process started"));

        tool.process_registry.kill_all().await;
    }
}
