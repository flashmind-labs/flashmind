//! Shell command execution with an optional timeout.
//!
//! # Security
//!
//! - Commands are subject to [`ForbiddenCmd`](flashmind_types::tool::ForbiddenCmd) patterns checked by the tool registry
//! - Protected paths (e.g., `/etc/shadow`, `~/.ssh/`) are validated before execution
//! - Unix commands run in a new process group; Linux cleanup also follows descendants that create separate sessions
//! - Commands have no timeout unless `timeout_secs` is supplied

use anyhow::Context;
use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::fmt::{self, Debug, Formatter};
#[cfg(target_os = "linux")]
use std::fs::File;
use std::future::pending;
use std::io;
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};

use crate::protected::ProtectedPaths;
use flashmind_types::MIN_SECRET_REDACT_LEN;
use flashmind_types::tool::{CommandAllowList, InterruptPayload, ToolContext};
use flashmind_types::tool::{Tool, ToolResult};

/// Kill an unfinished command and its descendants when execution is dropped.
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
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            let Ok(pid) = libc::pid_t::try_from(pid) else {
                tracing::warn!("command pid exceeds the platform process id range");
                return;
            };
            // Descendants can create their own sessions, escaping the root's
            // process group. Find them before killing their parents.
            #[cfg(target_os = "linux")]
            kill_descendants(pid);
            // SAFETY: The positive child pid identifies the process group
            // created by setsid; negating it addresses that group only.
            let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
            if result == -1 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    tracing::warn!(%error, "failed to kill command process group");
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
struct ProcessHandle {
    pid: libc::pid_t,
    descriptor: OwnedFd,
    directory: File,
}

#[cfg(target_os = "linux")]
impl ProcessHandle {
    fn open(pid: libc::pid_t) -> io::Result<Self> {
        // SAFETY: pidfd_open accepts a positive process id and zero flags.
        let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        if descriptor == -1 {
            return Err(io::Error::last_os_error());
        }
        let descriptor = i32::try_from(descriptor).map_err(io::Error::other)?;
        // SAFETY: The successful syscall returned a fresh descriptor, and
        // ownership is transferred into OwnedFd exactly once.
        let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor) };
        let directory = File::open(format!("/proc/{pid}"))?;
        let mut state = libc::pollfd {
            fd: descriptor.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: state is one initialized pollfd and its owned descriptor
        // remains open. The zero timeout only queries readiness.
        match unsafe { libc::poll(&mut state, 1, 0) } {
            -1 => return Err(io::Error::last_os_error()),
            0 => {}
            _ => return Err(io::Error::from_raw_os_error(libc::ESRCH)),
        }
        // If the old process exited and its pid was reused before opening
        // /proc, the pidfd is readable above. Pinning the live process's
        // directory keeps subsequent inspection independent of numeric reuse.
        Ok(Self {
            pid,
            descriptor,
            directory,
        })
    }

    fn proc_path(&self) -> PathBuf {
        PathBuf::from(format!("/proc/self/fd/{}", self.directory.as_raw_fd()))
    }

    fn signal(&self, signal: libc::c_int) {
        // SAFETY: The owned pidfd remains open and identifies this process
        // even if its numeric pid is subsequently reused. Null siginfo and
        // zero flags request an ordinary signal.
        let result = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.descriptor.as_raw_fd(),
                signal,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        };
        if result == -1 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                tracing::warn!(%error, signal, "failed to signal command descendant");
            }
        }
    }

    fn is_child_of(&self, parent: libc::pid_t) -> io::Result<bool> {
        let stat = std::fs::read_to_string(self.proc_path().join("stat"))?;
        let parent_pid = stat
            .rsplit_once(')')
            .and_then(|(_, fields)| fields.split_whitespace().nth(1))
            .and_then(|pid| pid.parse::<libc::pid_t>().ok());
        Ok(parent_pid == Some(parent))
    }
}

#[cfg(target_os = "linux")]
fn process_disappeared(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
}

#[cfg(target_os = "linux")]
fn kill_descendants(pid: libc::pid_t) {
    // Bound synchronous Drop work and retained descriptors. The process-group
    // fallback still runs if the tree cannot be completely inspected.
    // Daemons already reparented before cancellation are outside this tree.
    const MAX_PROCESSES: usize = 4096;
    let root = match ProcessHandle::open(pid) {
        Ok(process) => process,
        Err(error) => {
            if !process_disappeared(&error) {
                tracing::warn!(%error, "failed to inspect command descendants");
            }
            return;
        }
    };
    root.signal(libc::SIGSTOP);
    let mut processes = vec![root];
    let mut next = 0;
    let inspection = (|| -> io::Result<()> {
        while next < processes.len() {
            let parent = processes[next].pid;
            // Forks from any thread belong to that thread's children list.
            let threads = match std::fs::read_dir(processes[next].proc_path().join("task")) {
                Ok(threads) => threads,
                Err(error) if process_disappeared(&error) => {
                    next += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            for thread in threads {
                let children = match std::fs::read_to_string(thread?.path().join("children")) {
                    Ok(children) => children,
                    Err(error) if process_disappeared(&error) => continue,
                    Err(error) => return Err(error),
                };
                for pid in children.split_whitespace() {
                    if processes.len() == MAX_PROCESSES {
                        return Err(io::Error::other("command descendant limit reached"));
                    }
                    let pid = pid.parse::<libc::pid_t>().map_err(io::Error::other)?;
                    let child = match ProcessHandle::open(pid) {
                        Ok(child) => child,
                        Err(error) if process_disappeared(&error) => continue,
                        Err(error) => return Err(error),
                    };
                    match child.is_child_of(parent) {
                        Ok(true) => {
                            child.signal(libc::SIGSTOP);
                            processes.push(child);
                        }
                        Ok(false) => {}
                        Err(error) if process_disappeared(&error) => {}
                        Err(error) => return Err(error),
                    }
                }
            }
            next += 1;
        }
        Ok(())
    })();
    if let Err(error) = inspection
        && !process_disappeared(&error)
    {
        tracing::warn!(%error, "could not inspect all command descendants");
    }
    for process in processes.into_iter().rev() {
        process.signal(libc::SIGKILL);
    }
}

fn record_exec(start: Instant) {
    metrics::counter!("tools.exec.calls").increment(1);
    metrics::histogram!("tools.exec.duration_seconds").record(start.elapsed().as_secs_f64());
}

async fn capture_output<R: AsyncRead + Unpin>(
    pipe: Option<R>,
    ctx: &ToolContext<'_>,
    secrets: &[String],
    preview: Option<&Mutex<OutputPreview>>,
) -> io::Result<String> {
    let mut output = String::new();
    if let Some(pipe) = pipe {
        let mut reader = BufReader::new(pipe).lines();
        while let Some(line) = reader.next_line().await? {
            let visible = crate::utils::strip_ansi(&redact_secrets(&line, secrets));
            if let Some(preview) = preview {
                preview
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(visible.clone());
            } else {
                output.push_str(&line);
                output.push('\n');
            }
            ctx.progress(&visible);
        }
    }
    Ok(output)
}

struct OutputPreview {
    head_limit: usize,
    tail_limit: usize,
    head: Vec<String>,
    tail: VecDeque<String>,
    omitted: bool,
}

impl OutputPreview {
    fn push(&mut self, line: String) {
        if self.head.len() < self.head_limit {
            self.head.push(line);
        } else if self.tail_limit > 0 {
            if self.tail.len() == self.tail_limit {
                self.tail.pop_front();
                self.omitted = true;
            }
            self.tail.push_back(line);
        } else {
            self.omitted = true;
        }
    }

    fn finish(self) -> String {
        let mut lines = self.head;
        if self.omitted {
            lines.push("[output lines omitted]".to_string());
        }
        lines.extend(self.tail);
        lines.join("\n")
    }
}

const MAX_TIMEOUT_SECS: u64 = 1800;

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    tail: Option<usize>,
    #[serde(default)]
    head: Option<usize>,
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
    fn concurrent(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "exec"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Returns stdout+stderr combined. \
        Non-zero exit codes report as failure but output is still returned, \
        check the output to determine actual success. \
        Commands have no timeout by default. Set timeout_secs to limit a command (max 1800). \
        Set head and/or tail to limit the output returned to you while the user sees live stdout and stderr. \
        Use these arguments instead of piping a running command through head or tail to shorten its output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional limit in seconds (max 1800). Omit for no timeout."
                },
                "tail": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Return the last N combined stdout/stderr lines, plus any requested head and the exit code. Live progress remains visible. Head and tail may total up to 1000 lines."
                },
                "head": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Return the first N combined stdout/stderr lines, plus any requested tail and the exit code. The command still runs to completion and live progress remains visible."
                }
            },
            "required": ["command"]
        })
    }

    fn max_output_bytes(&self) -> usize {
        128 * 1024
    }

    fn max_output_lines(&self) -> usize {
        1002
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = Instant::now();
        let args: BashArgs = ctx.parse_args(self.name())?;
        let head = args.head.unwrap_or(0);
        let tail = args.tail.unwrap_or(0);
        if args.head == Some(0)
            || args.tail == Some(0)
            || head.checked_add(tail).is_none_or(|total| total > 1000)
        {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "head and tail must be positive and total at most 1000 lines",
            ));
        }
        let preview = (args.head.is_some() || args.tail.is_some()).then(|| {
            Mutex::new(OutputPreview {
                head_limit: head,
                tail_limit: tail,
                head: Vec::new(),
                tail: VecDeque::new(),
                omitted: false,
            })
        });

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

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&args.command)
            .env("NO_COLOR", "1")
            // Prevent interactive editors from blocking — git rebase, merge, etc.
            .env("GIT_EDITOR", "true")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("EDITOR", "true")
            .env("VISUAL", "true");

        // Always pipe stdin so interactive commands get EOF instead of hanging
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        // SAFETY: The child only calls the async-signal-safe setsid before exec.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }

        if let Some(ref wd) = ctx.working_dir {
            cmd.current_dir(wd);
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

        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        let execution = async {
            let (stdout, stderr) = tokio::try_join!(
                capture_output(child_stdout, &ctx, &self.secrets, preview.as_ref()),
                capture_output(child_stderr, &ctx, &self.secrets, preview.as_ref()),
            )
            .context("failed to read command output")?;
            let exit_code = match child.wait().await {
                Ok(status) => status.code().unwrap_or(-1),
                Err(error) => {
                    tracing::warn!(%error, "exec: command execution error");
                    record_exec(start);
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Error executing command: {error}"),
                    ));
                }
            };
            guard.completed();

            let combined = if let Some(preview) = preview {
                preview
                    .into_inner()
                    .unwrap_or_else(|error| error.into_inner())
                    .finish()
            } else if stderr.is_empty() {
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
        };
        let timeout = async {
            match args.timeout_secs {
                Some(seconds) => {
                    let seconds = seconds.clamp(1, MAX_TIMEOUT_SECS);
                    tokio::time::sleep(Duration::from_secs(seconds)).await;
                    seconds
                }
                None => pending().await,
            }
        };
        tokio::select! {
            biased;
            _ = ctx.cancel_token().cancelled() => {
                record_exec(start);
                Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"))
            }
            seconds = timeout => {
                record_exec(start);
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Command timed out after {seconds}s and was killed. Omit timeout_secs to run without a time limit."),
                ))
            }
            result = execution => result,
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        command.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn head_and_tail_limit_results_without_hiding_live_output_or_exit_status() {
        for (mut args, expected) in [
            (json!({"head": 2}), vec!["line-1", "line-2"]),
            (json!({"tail": 2}), vec!["line-4", "line-5"]),
            (
                json!({"head": 2, "tail": 2}),
                vec!["line-1", "line-2", "line-4", "line-5"],
            ),
            (
                json!({"head": 3, "tail": 3}),
                vec!["line-1", "line-2", "line-3", "line-4", "line-5"],
            ),
            (
                json!({}),
                vec!["line-1", "line-2", "line-3", "line-4", "line-5"],
            ),
        ] {
            args["command"] =
                json!("printf 'line-1\\nline-2\\nline-3\\nline-4\\nline-5\\n'; exit 7");
            let tool = make_tool();
            let cancel = CancellationToken::new();
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let result = tool
                .execute(ToolContext::new("test", args, None, &cancel).with_progress(sender))
                .await
                .unwrap();
            assert!(!result.is_success());
            let output = result.output();
            assert!(output.starts_with("[exit code: 7]\n"));
            assert_eq!(
                output
                    .lines()
                    .filter(|line| line.starts_with("line-"))
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                output.contains("[output lines omitted]"),
                expected.len() < 5
            );
            let mut progress = Vec::new();
            while let Some(line) = receiver.recv().await {
                progress.push(line);
            }
            assert_eq!(progress, ["line-1", "line-2", "line-3", "line-4", "line-5"]);
        }
    }

    #[tokio::test]
    async fn output_selection_combines_both_streams_and_redacts_the_result() {
        let mut tool = make_tool();
        tool.secrets = vec!["example-secret-value".to_string()];
        let cancel = CancellationToken::new();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let result = tool.execute(ToolContext::new("test", json!({
            "command": "printf 'stdout example-secret-value\\n'; printf 'stderr example-secret-value\\n' >&2",
            "head": 1, "tail": 1
        }), None, &cancel).with_progress(sender)).await.unwrap();
        assert!(result.is_success());
        let mut progress = Vec::new();
        while let Some(line) = receiver.recv().await {
            progress.push(line);
        }
        assert_eq!(progress.len(), 2);
        let output = result.output();
        assert_eq!(output.lines().skip(1).collect::<Vec<_>>(), progress);
        assert!(!output.contains("example-secret-value"));
        assert!(output.contains("stdout [REDACTED]"));
        assert!(output.contains("stderr [REDACTED]"));
    }

    #[tokio::test]
    async fn invalid_output_limits_do_not_execute_the_command() {
        let temp = tempdir().unwrap();
        let cwd = temp.path().to_path_buf();
        let cancel = CancellationToken::new();
        for mut args in [
            json!({"head": 0}),
            json!({"tail": 0}),
            json!({"head": 500, "tail": 501}),
            json!({"head": usize::MAX, "tail": 1}),
        ] {
            args["command"] = json!("printf executed > marker");
            let result = make_tool()
                .execute(ToolContext::new("test", args, Some(&cwd), &cancel))
                .await
                .unwrap();
            assert!(!result.is_success());
            assert!(result.output().contains("head and tail"));
        }
        assert!(!cwd.join("marker").exists());
    }

    #[test]
    fn command_label_is_the_unmodified_command() {
        let command = "cd /path/to/work && example \\\n  --first \\\n  --second";
        assert_eq!(make_tool().humanize(&json!({"command": command})), command);
    }

    #[tokio::test]
    async fn stderr_progress_is_delivered_while_the_command_is_running() {
        let tool = make_tool();
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let ctx = ToolContext::new(
                "test",
                json!({"command": "printf 'ready\\n' >&2; read answer", "head": 1, "tail": 1}),
                None,
                &task_cancel,
            )
            .with_progress(sender);
            tool.execute(ctx).await.unwrap()
        });
        let progress = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .unwrap();
        assert_eq!(progress.as_deref(), Some("ready"));
        assert!(!task.is_finished());
        cancel.cancel();
        assert_eq!(task.await.unwrap().output(), "Cancelled");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn process_inspection_stays_anchored_after_the_process_exits() {
        let mut child = Command::new("bash")
            .args(["-c", "read -r line"])
            .stdin(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let pid = libc::pid_t::try_from(child.id().unwrap()).unwrap();
        let mut process = ProcessHandle::open(pid).unwrap();
        let parent = libc::pid_t::try_from(std::process::id()).unwrap();
        assert!(process.is_child_of(parent).unwrap());
        child.kill().await.unwrap();
        // A replacement process can eventually receive the same numeric pid.
        // Substituting a live pid demonstrates that inspection never follows
        // that number once the original process directory has been pinned.
        process.pid = parent;
        assert!(matches!(
            process.is_child_of(parent).unwrap_err().raw_os_error(),
            Some(libc::ESRCH | libc::ENOENT)
        ));
        assert!(matches!(
            std::fs::read_dir(process.proc_path().join("task"))
                .unwrap_err()
                .raw_os_error(),
            Some(libc::ESRCH | libc::ENOENT)
        ));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn stopping_a_command_kills_descendants_in_separate_sessions() {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

        #[derive(Debug)]
        enum Stop {
            Cancel,
            Timeout,
            Drop,
        }

        struct DetachedChild(OwnedFd);

        impl Drop for DetachedChild {
            fn drop(&mut self) {
                // SAFETY: The owned pidfd identifies only this test's child.
                // Killing an already exited process is deliberately harmless.
                unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        self.0.as_raw_fd(),
                        libc::SIGKILL,
                        0,
                        0,
                    );
                }
            }
        }

        for stop in [Stop::Cancel, Stop::Timeout, Stop::Drop] {
            let directory = tempdir().unwrap();
            let cwd = directory.path().to_path_buf();
            let tool = make_tool();
            let cancel = CancellationToken::new();
            let task_cancel = cancel.clone();
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let task = tokio::spawn(async move {
                let context = ToolContext::new("test", json!({
                    "command": r#"mkfifo hold
exec 3<>hold
setsid bash -c 'setsid bash -c '\''printf "%s\n" "$BASHPID"; read -r line <&3'\'' & printf "%s\n" "$BASHPID"; read -r line <&3' &
read -r line <&3"#,
                    "timeout_secs": 60,
                }), Some(&cwd), &task_cancel).with_progress(sender);
                tool.execute(context).await.unwrap()
            });
            let mut descendants = Vec::new();
            for _ in 0..2 {
                let line = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                    .await
                    .unwrap()
                    .unwrap();
                let pid: libc::pid_t = line.parse().unwrap();
                // SAFETY: getsid only queries the process identified by pid.
                assert_eq!(unsafe { libc::getsid(pid) }, pid);
                // SAFETY: pidfd_open receives a child pid and valid zero flags.
                let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
                assert!(descriptor >= 0, "{}", io::Error::last_os_error());
                let descriptor = i32::try_from(descriptor).unwrap();
                // SAFETY: pidfd_open returned a new descriptor owned by this test.
                descendants.push(DetachedChild(unsafe { OwnedFd::from_raw_fd(descriptor) }));
            }
            match stop {
                Stop::Cancel => {
                    cancel.cancel();
                    assert_eq!(task.await.unwrap().output(), "Cancelled");
                }
                Stop::Timeout => {
                    tokio::time::pause();
                    tokio::time::advance(Duration::from_secs(60)).await;
                    assert!(task.await.unwrap().output().contains("timed out"));
                    tokio::time::resume();
                }
                Stop::Drop => {
                    task.abort();
                    assert!(task.await.unwrap_err().is_cancelled());
                }
            }
            let exited = tokio::task::spawn_blocking(move || {
                descendants.into_iter().all(|child| {
                    let mut descriptor = libc::pollfd {
                        fd: child.0.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // SAFETY: descriptor points to one initialized pollfd whose
                    // owned pidfd remains open for the duration of this call.
                    unsafe { libc::poll(&mut descriptor, 1, 2000) == 1 }
                })
            })
            .await
            .unwrap();
            assert!(exited, "{stop:?} left a detached descendant running");
        }
    }

    #[tokio::test]
    async fn live_output_from_both_streams_is_redacted_and_has_no_ansi_codes() {
        let mut tool = make_tool();
        tool.secrets = vec!["example-secret-value".into()];
        let cancel = CancellationToken::new();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let result = tool.execute(ToolContext::new("test", json!({
            "command": "printf '\\033[31mexample-secret-value\\033[0m\\n'; printf 'example-secret-value\\n' >&2"
        }), None, &cancel).with_progress(sender)).await.unwrap();
        assert!(result.is_success());
        for _ in 0..2 {
            let line = receiver.recv().await.unwrap();
            assert!(!line.contains("example-secret-value"));
            assert!(!line.contains('\u{1b}'));
            assert!(line.contains("REDACTED"));
        }
    }

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
    async fn commands_without_a_timeout_keep_running_until_cancelled() {
        for command in [
            "printf 'ready\\n'; sleep 30",
            "printf 'ready\\n'; exec 1>&-; sleep 30",
        ] {
            let tool = make_tool();
            let cancel = CancellationToken::new();
            let task_cancel = cancel.clone();
            let (progress, mut output) = tokio::sync::mpsc::unbounded_channel();
            let task = tokio::spawn(async move {
                let context =
                    ToolContext::new("test-id", json!({ "command": command }), None, &task_cancel)
                        .with_progress(progress);
                tool.execute(context).await.unwrap()
            });
            assert_eq!(output.recv().await.as_deref(), Some("ready"));
            tokio::time::pause();
            tokio::time::advance(Duration::from_secs(601)).await;
            tokio::task::yield_now().await;
            assert!(
                !task.is_finished(),
                "An omitted timeout must not stop the command."
            );
            cancel.cancel();
            let result = task.await.unwrap();
            assert!(!result.is_success());
            assert_eq!(result.output(), "Cancelled");
            tokio::time::resume();
        }
    }

    #[tokio::test]
    async fn find_commands_are_passed_to_the_shell_unchanged() {
        let result = crate::tests::execute_tool(
            &make_tool(),
            "test-id",
            json!({ "command": "find () { printf 'example\\n'; }; find" }),
        )
        .await
        .unwrap();
        assert!(result.is_success());
        assert_eq!(result.output(), "[exit code: 0]\nexample");
    }

    #[tokio::test]
    async fn explicit_timeout_applies_after_stdout_closes() {
        let result = crate::tests::execute_tool(
            &make_tool(),
            "test-id",
            json!({ "command": "exec 1>&-; sleep 30", "timeout_secs": 1 }),
        )
        .await
        .unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("timed out after 1s"));
    }

    #[test]
    fn shell_schema_exposes_command_timeout_head_and_tail() {
        let tool = make_tool();
        let schema = tool.parameters();
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(properties.len(), 4);
        assert!(properties.contains_key("command"));
        assert!(properties.contains_key("timeout_secs"));
        assert!(properties.contains_key("tail"));
        assert!(properties.contains_key("head"));
    }

    #[tokio::test]
    async fn obsolete_background_argument_does_not_detach_commands() {
        let tool = make_tool();
        let result = crate::tests::execute_tool(
            &tool,
            "test-id",
            json!({"command": "echo final-output; exit 7", "background": true}),
        )
        .await
        .unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("final-output"));
        assert!(result.output().contains("7"));
    }
}
