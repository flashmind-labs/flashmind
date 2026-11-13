//! Shell tasks that outlive a turn and remain owned by the tool registry.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::utils::truncate_utf8;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::bash::{BashTool, strip_redundant_cd};

const MAX_JOBS: usize = 32;
const MAX_LINES: usize = 100;
const MAX_LINE_BYTES: usize = 1000;

/// How a background job ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobEnd {
    Completed,
    Failed,
    /// Cancelled, by [`Jobs::stop`] or by the `process` tool.
    Stopped,
}

/// A change in the background jobs of a tool registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobEvent {
    /// A job runs in the background: it was started with
    /// `run_in_background`, or a foreground command moved there.
    Started {
        id: Uuid,
        tool_call_id: String,
        command: String,
    },
    /// A job that was announced as started has ended. Sent once, after
    /// `Started`.
    Ended {
        id: Uuid,
        end: JobEnd,
        /// The tail of the output, for a job that did not complete.
        summary: Option<String>,
    },
}

#[derive(Default)]
struct ProcessState {
    lines: VecDeque<String>,
    result: Option<ToolResult>,
    progress: Option<UnboundedSender<String>>,
    background: bool,
    /// `Started` was sent, so the end of the job is sent too.
    announced: bool,
    end: Option<JobEnd>,
}

impl ProcessState {
    fn output(&self, id: Uuid) -> String {
        let status = match &self.result {
            None => "running",
            Some(result) if result.is_success() => "completed",
            Some(_) => "failed or stopped",
        };
        let mut output = format!("Process {id}: {status}");
        for line in &self.lines {
            output.push('\n');
            output.push_str(line);
        }
        if let Some(result) = &self.result {
            output.push('\n');
            output.push_str(&result.output());
        }
        output
    }
}

struct BackgroundJob {
    id: Uuid,
    cancel: CancellationToken,
    state: Arc<Mutex<ProcessState>>,
    task: Option<JoinHandle<ToolResult>>,
}

impl BackgroundJob {
    fn is_finished(&self) -> bool {
        self.task.as_ref().is_some_and(JoinHandle::is_finished)
            || self
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .result
                .is_some()
    }
}

impl Drop for BackgroundJob {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .progress = None;
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[derive(Default)]
struct Processes {
    jobs: Mutex<Vec<BackgroundJob>>,
    observer: Mutex<Option<UnboundedSender<JobEvent>>>,
}

impl Processes {
    fn notify(&self, event: JobEvent) {
        let mut observer = self.observer.lock().unwrap_or_else(|e| e.into_inner());
        if observer
            .as_ref()
            .is_some_and(|observer| observer.send(event).is_err())
        {
            *observer = None;
        }
    }

    /// Announces `job` as a background job, and its end if it has come.
    /// The job's own task sends the end once this has run.
    fn announce(&self, job: &BackgroundJob, tool_call_id: &str, command: &str) {
        self.notify(JobEvent::Started {
            id: job.id,
            tool_call_id: tool_call_id.to_owned(),
            command: command.to_owned(),
        });
        let ended = {
            let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
            state.announced = true;
            ended_event(job.id, &state)
        };
        if let Some(event) = ended {
            self.notify(event);
        }
    }
}

/// The `Ended` event of a finished job, when it was announced.
fn ended_event(id: Uuid, state: &ProcessState) -> Option<JobEvent> {
    let (end, result) = (state.end?, state.result.as_ref()?);
    let summary = (end != JobEnd::Completed).then(|| result.output());
    state
        .announced
        .then_some(JobEvent::Ended { id, end, summary })
}

/// A handle on the background jobs of one tool registry, for a client that
/// reports them and stops them.
#[derive(Clone)]
pub struct Jobs {
    processes: Arc<Processes>,
}

impl Jobs {
    /// The job events from now on. Only the latest subscriber gets them.
    pub fn subscribe(&self) -> UnboundedReceiver<JobEvent> {
        // Unbounded: one small event per job change, drained by its owner
        // as they come, and at most 32 jobs run at once.
        let (events, receiver) = mpsc::unbounded_channel();
        *self
            .processes
            .observer
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(events);
        receiver
    }

    /// Stops a running job. False when `id` names no running job.
    pub fn stop(&self, id: Uuid) -> bool {
        let jobs = self
            .processes
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(job) = jobs.iter().find(|job| job.id == id) else {
            return false;
        };
        if job.is_finished() {
            return false;
        }
        job.cancel.cancel();
        true
    }
}

fn make_room(jobs: &mut Vec<BackgroundJob>) -> bool {
    if jobs.len() < MAX_JOBS {
        return true;
    }
    if let Some(index) = jobs.iter().position(BackgroundJob::is_finished) {
        jobs.remove(index);
        return true;
    }
    false
}

fn background_result(call_id: &str, id: Uuid) -> ToolResult {
    ToolResult::success(
        call_id,
        format!(
            "Started background process {id}. Use process with action=status to read output, or action=stop to stop it. Startup has not been verified."
        ),
    )
}

/// Shell execution with optional background tasks.
pub struct BackgroundExecTool {
    shell: Arc<BashTool>,
    processes: Arc<Processes>,
}

impl BackgroundExecTool {
    pub fn new(shell: Arc<BashTool>) -> Self {
        Self {
            shell,
            processes: Arc::new(Processes::default()),
        }
    }

    pub fn jobs(&self) -> Jobs {
        Jobs {
            processes: self.processes.clone(),
        }
    }

    pub fn process_tool(&self) -> ProcessTool {
        ProcessTool {
            processes: self.processes.clone(),
        }
    }

    fn spawn_job(&self, ctx: &ToolContext<'_>, cwd: PathBuf, foreground: bool) -> BackgroundJob {
        let id = Uuid::new_v4();
        let cancel = CancellationToken::new();
        let state = Arc::new(Mutex::new(ProcessState {
            progress: if foreground {
                ctx.progress_sender()
            } else {
                None
            },
            background: !foreground,
            ..ProcessState::default()
        }));
        let task_state = state.clone();
        // Weak: the jobs own their tasks, and a task must not keep them alive.
        let processes = Arc::downgrade(&self.processes);
        let job_id = id;
        let task_cancel = cancel.clone();
        let shell = self.shell.clone();
        let call_id = ctx.tool_call_id.to_owned();
        let args = ctx.args.clone();
        let retain_preview = args.get("head").is_some() || args.get("tail").is_some();
        let task = tokio::spawn(async move {
            let context = ToolContext::new(&call_id, args, Some(&cwd), &task_cancel);
            let output_state = task_state.clone();
            let capture = move |line: &str| {
                let mut state = output_state.lock().unwrap_or_else(|e| e.into_inner());
                if state.lines.len() == MAX_LINES {
                    state.lines.pop_front();
                }
                state
                    .lines
                    .push_back(truncate_utf8(line, MAX_LINE_BYTES).to_owned());
                if let Some(progress) = &state.progress
                    && progress.send(line.to_owned()).is_err()
                {
                    state.progress = None;
                }
                !state.background || retain_preview
            };
            let result = match shell.execute_with_output(context, Some(&capture)).await {
                Ok(result) => result,
                Err(error) => {
                    ToolResult::failure(&call_id, format!("Background command failed: {error:#}"))
                }
            };
            let output = result.output();
            let output = truncate_utf8(&output, MAX_LINE_BYTES).to_owned();
            let summary = if result.is_success() {
                ToolResult::success(&call_id, output)
            } else {
                ToolResult::failure(&call_id, output)
            };
            let ended = {
                let mut state = task_state.lock().unwrap_or_else(|e| e.into_inner());
                state.progress = None;
                state.end = Some(if task_cancel.is_cancelled() {
                    JobEnd::Stopped
                } else if summary.is_success() {
                    JobEnd::Completed
                } else {
                    JobEnd::Failed
                });
                state.result = Some(summary);
                ended_event(job_id, &state)
            };
            if let (Some(event), Some(processes)) = (ended, processes.upgrade()) {
                processes.notify(event);
            }
            result
        });
        BackgroundJob {
            id,
            cancel,
            state,
            task: Some(task),
        }
    }
}

#[derive(Deserialize)]
struct ExecArgs {
    command: String,
    #[serde(default)]
    run_in_background: bool,
}

#[async_trait]
impl Tool for BackgroundExecTool {
    fn name(&self) -> &str {
        self.shell.name()
    }

    fn concurrent(&self) -> bool {
        false
    }

    fn supports_background(&self) -> bool {
        true
    }

    fn description(&self) -> &str {
        "Execute a shell command using bash, or sh if bash is unavailable. Set run_in_background=true for servers, watchers, or other long-running commands. \
        Background execution returns a process ID immediately; use process to read output or stop it. \
        Run the command directly, without nohup, setsid, or a trailing &. \
        Background tasks survive turn cancellation and stop when this agent session exits. \
        Starting a task does not confirm readiness: check its output and verify the service before reporting success. \
        Foreground commands return stdout and stderr with the exit code and have no timeout by default. \
        Set timeout_secs for an optional limit (max 1800). \
        Set head and/or tail to limit returned output (at most 1000 lines total). \
        Commands already start in your working directory. Do not prefix them with cd to that directory."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.shell.parameters();
        schema["properties"]["run_in_background"] = json!({
            "type": "boolean",
            "description": "Start a managed background task and return immediately. Defaults to false."
        });
        schema
    }

    fn max_output_bytes(&self) -> usize {
        self.shell.max_output_bytes()
    }

    fn max_output_lines(&self) -> usize {
        self.shell.max_output_lines()
    }

    fn humanize(&self, args: &Value) -> String {
        self.shell.humanize(args)
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        crate::utils::validate_tool_args(&ctx.args, &self.parameters())?;
        let args: ExecArgs = ctx.parse_args(self.name())?;
        if let Some(arguments) = ctx.args.as_object_mut() {
            arguments.remove("run_in_background");
        }
        let request = ctx.background_request().cloned();
        if !args.run_in_background && request.is_none() {
            return self.shell.execute(ctx).await;
        }
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }
        let cwd = match ctx.working_dir {
            Some(cwd) => cwd.clone(),
            None => {
                std::env::current_dir().context("failed to determine command working directory")?
            }
        };
        let command = strip_redundant_cd(&args.command, &cwd);
        if let Some(result) = self.shell.check_command(ctx.tool_call_id, command) {
            return Ok(result);
        }
        ctx.args["command"] = json!(command);
        if args.run_in_background {
            // Supply a default only when the caller did not choose an output limit.
            if ctx.args.get("head").is_none() && ctx.args.get("tail").is_none() {
                ctx.args["tail"] = json!(MAX_LINES);
            }
            let mut jobs = self
                .processes
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if !make_room(&mut jobs) {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Background process limit reached (32). Stop a process first.",
                ));
            }
            let job = self.spawn_job(&ctx, cwd, false);
            let id = job.id;
            self.processes.announce(&job, ctx.tool_call_id, command);
            jobs.push(job);
            return Ok(background_result(ctx.tool_call_id, id));
        }
        let request = request.context("foreground execution requires a background control")?;
        let mut job = self.spawn_job(&ctx, cwd, true);
        let mut can_background = true;
        loop {
            tokio::select! {
                biased;
                () = ctx.cancel_token().cancelled() => {
                    return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
                }
                result = job.task.as_mut().context("foreground execution has no task")? => {
                    return result.context("foreground command task failed");
                }
                () = request.requested(), if can_background => {
                    let mut jobs = self.processes.jobs.lock().unwrap_or_else(|e| e.into_inner());
                    if !make_room(&mut jobs) {
                        ctx.progress("Background process limit reached (32). This command is still running in the foreground.");
                        can_background = false;
                        continue;
                    }
                    {
                        let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
                        state.progress = None;
                        state.background = true;
                    }
                    let id = job.id;
                    self.processes.announce(&job, ctx.tool_call_id, command);
                    jobs.push(job);
                    return Ok(background_result(ctx.tool_call_id, id));
                }
            };
        }
    }
}

/// Read output and stop shell tasks started with `exec`.
pub struct ProcessTool {
    processes: Arc<Processes>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    List,
    Status,
    Stop,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessArgs {
    action: Action,
    id: Option<Uuid>,
}

#[async_trait]
impl Tool for ProcessTool {
    fn name(&self) -> &str {
        "process"
    }

    fn concurrent(&self) -> bool {
        false
    }

    fn description(&self) -> &str {
        "List background shell tasks, read their latest output and completion status, or stop one. \
        Use the process ID returned by exec with run_in_background=true. Output retains the last 100 lines. \
        Up to 32 tasks are retained; completed tasks may be removed when new tasks start."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "action": { "type": "string", "enum": ["list", "status", "stop"] },
                "id": { "type": "string", "description": "Process ID, required for status and stop." }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ProcessArgs = ctx.parse_args(self.name())?;
        if matches!(args.action, Action::List) && ctx.args.get("id").is_some() {
            anyhow::bail!("id is not supported by list");
        }
        let state;
        let task;
        let id;
        {
            let mut jobs = self
                .processes
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if matches!(args.action, Action::List) {
                let output = jobs
                    .iter()
                    .map(|job| {
                        let status = if job.is_finished() {
                            "finished"
                        } else {
                            "running"
                        };
                        format!("{}: {status}", job.id)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(ToolResult::success(
                    ctx.tool_call_id,
                    if output.is_empty() {
                        "No background processes.".to_owned()
                    } else {
                        output
                    },
                ));
            }
            let Some(requested_id) = args.id else {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "id is required for status and stop",
                ));
            };
            id = requested_id;
            let Some(job) = jobs.iter_mut().find(|job| job.id == id) else {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Unknown background process ID",
                ));
            };
            state = job.state.clone();
            if matches!(args.action, Action::Stop) {
                job.cancel.cancel();
            }
            task = if matches!(args.action, Action::Stop) || job.is_finished() {
                job.task.take()
            } else {
                None
            };
        }
        if let Some(task) = task
            && let Err(error) = task.await
        {
            state.lock().unwrap_or_else(|e| e.into_inner()).result = Some(ToolResult::failure(
                ctx.tool_call_id,
                format!("Background process task failed: {error}"),
            ));
        }
        let output = state.lock().unwrap_or_else(|e| e.into_inner()).output(id);
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn exec_preserves_output_limits_and_rejects_unknown_arguments() {
        let dir = tempdir().unwrap();
        let exec = BackgroundExecTool::new(Arc::new(shell(dir.path())));
        let cancel = CancellationToken::new();
        let cwd = dir.path().to_path_buf();
        for args in [
            json!({"command":"true","run_in_background":false,"typo":1}),
            json!({"command":"true","run_in_background":true,"typo":1}),
        ] {
            assert!(
                exec.execute(ToolContext::new("test", args, Some(&cwd), &cancel))
                    .await
                    .is_err()
            );
        }
        let result = exec.execute(ToolContext::new("test",json!({"command":"printf 'first\nsecond\nthird\n'","run_in_background":true,"head":1}),Some(&cwd),&cancel)).await.unwrap();
        assert!(result.is_success());
        let task = exec
            .processes
            .jobs
            .lock()
            .unwrap()
            .last_mut()
            .unwrap()
            .task
            .take()
            .unwrap();
        let output = task.await.unwrap().output();
        assert!(output.contains("first"));
        assert!(!output.contains("second"));
        let process = exec.process_tool();
        assert!(
            process
                .execute(ToolContext::new(
                    "test",
                    json!({"action":"list","id":Uuid::new_v4()}),
                    Some(&cwd),
                    &cancel
                ))
                .await
                .is_err()
        );
    }

    #[test]
    fn process_rejects_unknown_fields() {
        let error = serde_json::from_value::<super::ProcessArgs>(
            serde_json::json!({"unexpected_parameter":true}),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("unknown field"));
    }

    use std::path::Path;
    use std::time::Duration;

    use flashmind_types::tool::{BackgroundRequest, CommandAllowList, ForbiddenCmd};
    use tempfile::tempdir;

    use super::*;
    use crate::protected::ProtectedPaths;

    fn shell(cwd: &Path) -> BashTool {
        BashTool {
            protected: Arc::new(ProtectedPaths::new(cwd)),
            secrets: vec!["example-secret".to_owned()],
            forbidden_cmds: Vec::new(),
            allowlist: None,
        }
    }

    fn process_id(result: &ToolResult) -> Uuid {
        assert!(result.is_success(), "{}", result.output());
        result
            .output()
            .split_whitespace()
            .nth(3)
            .unwrap()
            .trim_end_matches('.')
            .parse()
            .unwrap()
    }

    #[tokio::test]
    async fn foreground_commands_still_return_their_output_and_exit_status() {
        let cwd = tempdir().unwrap();
        let exec = BackgroundExecTool::new(Arc::new(shell(cwd.path())));
        let cancel = CancellationToken::new();
        let request = BackgroundRequest::default();
        let result = exec
            .execute(
                ToolContext::new(
                    "test",
                    json!({
                        "command": "printf 'example-secret'; exit 7"
                    }),
                    None,
                    &cancel,
                )
                .with_background_request(&request),
            )
            .await
            .unwrap();
        assert!(!result.is_success());
        assert_eq!(result.output(), "[exit code: 7]\n[REDACTED]");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn backgrounding_keeps_the_process_and_releases_live_output() {
        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;
        use tokio::sync::mpsc;

        let directory = tempdir().unwrap();
        let cwd = directory.path().to_path_buf();
        assert!(
            Command::new("mkfifo")
                .arg("control")
                .current_dir(&cwd)
                .status()
                .await
                .unwrap()
                .success()
        );
        let mut control = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(cwd.join("control"))
            .await
            .unwrap();
        let exec = Arc::new(BackgroundExecTool::new(Arc::new(shell(&cwd))));
        let process = exec.process_tool();
        let request = BackgroundRequest::default();
        let task_request = request.clone();
        let turn_cancel = CancellationToken::new();
        let task_cancel = turn_cancel.clone();
        let task_exec = exec.clone();
        let task_cwd = cwd.clone();
        let (progress, mut output) = mpsc::unbounded_channel();
        let foreground = tokio::spawn(async move {
            let command = "printf 'launched\\n' >> launches; exec 3<>control; printf 'ready\\n'; read -r line <&3; printf 'finished\\n'";
            task_exec
                .execute(
                    ToolContext::new(
                        "command",
                        json!({"command": command}),
                        Some(&task_cwd),
                        &task_cancel,
                    )
                    .with_background_request(&task_request)
                    .with_progress(progress),
                )
                .await
                .unwrap()
        });
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), output.recv())
                .await
                .unwrap()
                .as_deref(),
            Some("ready")
        );
        request.request();
        let result = tokio::time::timeout(Duration::from_secs(5), foreground)
            .await
            .unwrap()
            .unwrap();
        let id = process_id(&result);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), output.recv())
                .await
                .unwrap(),
            None
        );
        turn_cancel.cancel();
        control.write_all(b"finish\n").await.unwrap();
        let task = exec.processes.jobs.lock().unwrap()[0].task.take().unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert_eq!(
            tokio::fs::read_to_string(cwd.join("launches"))
                .await
                .unwrap(),
            "launched\n"
        );
        let result = crate::tests::execute_tool(
            &process,
            "status",
            json!({
                "action": "status", "id": id.to_string()
            }),
        )
        .await
        .unwrap();
        assert!(result.output().contains("finished"));
        assert!(result.output().contains("[exit code: 0]"));
    }

    #[tokio::test]
    async fn background_completion_retains_output_redaction_and_failure() {
        let cwd = tempdir().unwrap();
        let exec = BackgroundExecTool::new(Arc::new(shell(cwd.path())));
        let process = exec.process_tool();
        let result = crate::tests::execute_tool(
            &exec,
            "test",
            json!({
                "command": "printf 'example-secret\\n'; exit 7",
                "run_in_background": true
            }),
        )
        .await
        .unwrap();
        let id = process_id(&result);
        let task = exec.processes.jobs.lock().unwrap()[0].task.take().unwrap();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
        let result = crate::tests::execute_tool(
            &process,
            "status",
            json!({
                "action": "status", "id": id.to_string()
            }),
        )
        .await
        .unwrap();
        let output = result.output();
        assert!(output.contains("failed or stopped"));
        assert!(output.contains("[exit code: 7]"));
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("example-secret"));
    }

    #[tokio::test]
    async fn jobs_report_their_start_before_their_end_even_when_they_end_at_once() {
        let cwd = tempdir().unwrap();
        let exec = BackgroundExecTool::new(Arc::new(shell(cwd.path())));
        let mut events = exec.jobs().subscribe();
        let result = crate::tests::execute_tool(
            &exec,
            "test",
            json!({ "command": "exit 3", "run_in_background": true }),
        )
        .await
        .unwrap();
        let id = process_id(&result);
        assert_eq!(
            events.recv().await,
            Some(JobEvent::Started {
                id,
                tool_call_id: "test".to_owned(),
                command: "exit 3".to_owned(),
            })
        );
        let Some(JobEvent::Ended { id: ended, end, .. }) = events.recv().await else {
            panic!("the job should report its end");
        };
        assert_eq!((ended, end), (id, JobEnd::Failed));
    }

    #[tokio::test]
    async fn background_commands_require_approval_and_respect_forbidden_commands() {
        struct Deny;
        impl CommandAllowList for Deny {
            fn is_allowed(&self, _: &str) -> bool {
                false
            }
            fn add_session_pattern(&self, _: &str) {}
        }

        let cwd = tempdir().unwrap();
        let mut bash = shell(cwd.path());
        bash.allowlist = Some(Arc::new(Deny));
        bash.forbidden_cmds.push(ForbiddenCmd {
            command: "forbidden-command".to_owned(),
            reason: "blocked".to_owned(),
            reconsider: false,
        });
        let exec = BackgroundExecTool::new(Arc::new(bash));
        for command in ["printf ready", "forbidden-command"] {
            let result = crate::tests::execute_tool(
                &exec,
                "test",
                json!({
                    "command": command, "run_in_background": true
                }),
            )
            .await
            .unwrap();
            if command == "forbidden-command" {
                assert_eq!(result.output(), "Forbidden: blocked");
            } else {
                assert!(result.is_interrupt());
                assert!(
                    result
                        .payload()
                        .unwrap()
                        .as_any()
                        .downcast_ref::<crate::bash::CommandApproval>()
                        .is_some()
                );
            }
            assert!(exec.processes.jobs.lock().unwrap().is_empty());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_command_survives_turn_cancellation_and_stops_on_request_or_drop() {
        use std::process::Stdio;

        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::process::Command;

        for stop_with_tool in [true, false] {
            let directory = tempdir().unwrap();
            let cwd = directory.path().to_path_buf();
            assert!(
                Command::new("mkfifo")
                    .args(["ready", "hold"])
                    .current_dir(&cwd)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
            let mut control = tokio::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(cwd.join("hold"))
                .await
                .unwrap();
            let mut reader = Command::new("cat")
                .arg("ready")
                .current_dir(&cwd)
                .stdout(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .unwrap();
            let mut ready = BufReader::new(reader.stdout.take().unwrap()).lines();
            let exec = BackgroundExecTool::new(Arc::new(shell(&cwd)));
            let process = exec.process_tool();
            let turn_cancel = CancellationToken::new();
            let result = exec
            .execute(ToolContext::new(
                "start",
                json!({
                    "command": "exec 3<>hold; exec 4>ready; printf 'ready\\n' >&4; while read -r line <&3; do printf '%s\\n' \"$line\" >&4; done",
                    "run_in_background": true
                }),
                Some(&cwd),
                &turn_cancel,
            ))
            .await
            .unwrap();
            let id = process_id(&result);
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), ready.next_line())
                    .await
                    .unwrap()
                    .unwrap()
                    .as_deref(),
                Some("ready")
            );
            turn_cancel.cancel();
            control.write_all(b"alive\n").await.unwrap();
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), ready.next_line())
                    .await
                    .unwrap()
                    .unwrap()
                    .as_deref(),
                Some("alive")
            );
            let result = crate::tests::execute_tool(
                &process,
                "status",
                json!({
                    "action": "status", "id": id.to_string()
                }),
            )
            .await
            .unwrap();
            assert!(result.output().contains(": running"));
            if stop_with_tool {
                let stopped = tokio::time::timeout(
                    Duration::from_secs(5),
                    crate::tests::execute_tool(
                        &process,
                        "stop",
                        json!({
                            "action": "stop", "id": id.to_string()
                        }),
                    ),
                )
                .await
                .unwrap()
                .unwrap();
                assert!(stopped.output().contains("Cancelled"));
                let result = crate::tests::execute_tool(
                    &process,
                    "list",
                    json!({
                        "action": "list"
                    }),
                )
                .await
                .unwrap();
                assert!(result.output().contains(&format!("{id}: finished")));
            } else {
                drop(exec);
                drop(process);
            }
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), ready.next_line())
                    .await
                    .unwrap()
                    .unwrap(),
                None
            );
            assert!(reader.wait().await.unwrap().success());
        }
    }
}
