//! Tool implementations for cron job management.
//!
//! Provides five tools that implement the [`Tool`]
//! trait: `cron_create`, `cron_list`, `cron_edit`, `cron_delete`, and
//! `schedule_once`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult, parse_args};

use crate::job::JobSchedule;
use crate::log::CronLog;
use crate::registry::CronRegistry;
use crate::schedule::CronSchedule;

// ---------------------------------------------------------------------------
// CronCreateTool
// ---------------------------------------------------------------------------

/// Tool that creates a new recurring cron job.
pub struct CronCreateTool {
    registry: Arc<CronRegistry>,
    extra_metadata: Value,
}

impl CronCreateTool {
    /// Create a new instance backed by the given registry.
    pub fn new(registry: Arc<CronRegistry>) -> Self {
        Self {
            registry,
            extra_metadata: json!({}),
        }
    }

    /// Create a new instance that merges extra fields into every job's metadata.
    pub fn with_metadata(registry: Arc<CronRegistry>, extra_metadata: Value) -> Self {
        Self {
            registry,
            extra_metadata,
        }
    }
}

#[derive(Deserialize)]
struct CronCreateArgs {
    schedule: String,
    task: String,
}

#[async_trait]
impl Tool for CronCreateTool {
    fn name(&self) -> &str {
        "cron_create"
    }

    fn description(&self) -> &str {
        "Create a recurring cron job. The schedule is a POSIX 5-field cron expression (minute hour day_of_month month day_of_week)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "schedule": {
                    "type": "string",
                    "description": "POSIX 5-field cron expression, e.g. '*/15 * * * *' for every 15 minutes"
                },
                "task": {
                    "type": "string",
                    "description": "Description of the task to execute"
                }
            },
            "required": ["schedule", "task"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CronCreateArgs = parse_args("cron_create", ctx.args)?;

        // Validate the cron expression
        if let Err(e) = CronSchedule::parse(&args.schedule) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Invalid cron expression: {e}"),
            ));
        }

        let job = self
            .registry
            .create_with_metadata(
                JobSchedule::Cron(args.schedule),
                args.task,
                false,
                self.extra_metadata.clone(),
            )
            .await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Created cron job {}", job.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("job");
        format!("Creating cron job: {task}")
    }
}

// ---------------------------------------------------------------------------
// CronListTool
// ---------------------------------------------------------------------------

/// Tool that lists all cron jobs.
pub struct CronListTool {
    registry: Arc<CronRegistry>,
}

impl CronListTool {
    /// Create a new instance backed by the given registry.
    pub fn new(registry: Arc<CronRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for CronListTool {
    fn name(&self) -> &str {
        "cron_list"
    }

    fn description(&self) -> &str {
        "List all scheduled cron jobs with their IDs, schedules, tasks, and status."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let jobs = self.registry.list().await?;

        if jobs.is_empty() {
            return Ok(ToolResult::success(ctx.tool_call_id, "No cron jobs found."));
        }

        let mut lines = Vec::with_capacity(jobs.len() + 1);
        lines.push(format!(
            "{:<36}  {:<20}  {:<8}  {}",
            "ID", "Schedule", "Enabled", "Task"
        ));
        for job in &jobs {
            let sched = match &job.schedule {
                JobSchedule::Cron(expr) => expr.clone(),
                JobSchedule::Once(dt) => format!("once @ {}", dt.format("%Y-%m-%d %H:%M")),
                JobSchedule::OnWake { from_hour, .. } => match from_hour {
                    Some(h) => format!("on wake (from {h}:00)"),
                    None => "on wake".to_string(),
                },
            };
            lines.push(format!(
                "{:<36}  {:<20}  {:<8}  {}",
                job.id, sched, job.enabled, job.task
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, lines.join("\n")))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing cron jobs".to_string()
    }
}

// ---------------------------------------------------------------------------
// CronEditTool
// ---------------------------------------------------------------------------

/// Tool that edits an existing cron job's schedule, task, or enabled status.
pub struct CronEditTool {
    registry: Arc<CronRegistry>,
}

impl CronEditTool {
    /// Create a new instance backed by the given registry.
    pub fn new(registry: Arc<CronRegistry>) -> Self {
        Self { registry }
    }
}

#[derive(Deserialize)]
struct CronEditArgs {
    id: String,
    schedule: Option<String>,
    task: Option<String>,
    enabled: Option<bool>,
}

#[async_trait]
impl Tool for CronEditTool {
    fn name(&self) -> &str {
        "cron_edit"
    }

    fn description(&self) -> &str {
        "Edit an existing cron job. You can change the schedule, task description, or enabled status."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "UUID of the job to edit"
                },
                "schedule": {
                    "type": "string",
                    "description": "New POSIX 5-field cron expression"
                },
                "task": {
                    "type": "string",
                    "description": "New task description"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable or disable the job"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CronEditArgs = parse_args("cron_edit", ctx.args)?;

        let id: uuid::Uuid = match args.id.parse() {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid UUID: {}", args.id),
                ));
            }
        };

        let Some(mut job) = self.registry.get(id).await? else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Job not found: {id}"),
            ));
        };

        if let Some(schedule) = args.schedule {
            if let Err(e) = CronSchedule::parse(&schedule) {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid cron expression: {e}"),
                ));
            }
            job.schedule = JobSchedule::Cron(schedule);
        }
        if let Some(task) = args.task {
            job.task = task;
        }
        if let Some(enabled) = args.enabled {
            job.enabled = enabled;
        }

        self.registry.update(&job).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Updated job {id}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Editing cron job {id}")
    }
}

// ---------------------------------------------------------------------------
// CronDeleteTool
// ---------------------------------------------------------------------------

/// Tool that deletes a cron job by ID.
pub struct CronDeleteTool {
    registry: Arc<CronRegistry>,
}

impl CronDeleteTool {
    /// Create a new instance backed by the given registry.
    pub fn new(registry: Arc<CronRegistry>) -> Self {
        Self { registry }
    }
}

#[derive(Deserialize)]
struct CronDeleteArgs {
    id: String,
}

#[async_trait]
impl Tool for CronDeleteTool {
    fn name(&self) -> &str {
        "cron_delete"
    }

    fn description(&self) -> &str {
        "Delete a cron job by its UUID."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "UUID of the job to delete"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CronDeleteArgs = parse_args("cron_delete", ctx.args)?;

        let id: uuid::Uuid = match args.id.parse() {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid UUID: {}", args.id),
                ));
            }
        };

        if self.registry.delete(id).await? {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Deleted job {id}"),
            ))
        } else {
            Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Job not found: {id}"),
            ))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Deleting cron job {id}")
    }
}

// ---------------------------------------------------------------------------
// ScheduleOnceTool
// ---------------------------------------------------------------------------

/// Tool that creates a one-shot scheduled job at a specific datetime.
pub struct ScheduleOnceTool {
    registry: Arc<CronRegistry>,
    extra_metadata: Value,
}

impl ScheduleOnceTool {
    /// Create a new instance backed by the given registry.
    pub fn new(registry: Arc<CronRegistry>) -> Self {
        Self {
            registry,
            extra_metadata: json!({}),
        }
    }

    /// Create a new instance that merges extra fields into every job's metadata.
    pub fn with_metadata(registry: Arc<CronRegistry>, extra_metadata: Value) -> Self {
        Self {
            registry,
            extra_metadata,
        }
    }
}

#[derive(Deserialize)]
struct ScheduleOnceArgs {
    datetime: String,
    task: String,
}

#[async_trait]
impl Tool for ScheduleOnceTool {
    fn name(&self) -> &str {
        "schedule_once"
    }

    fn description(&self) -> &str {
        "Schedule a one-time task at a specific UTC datetime (ISO 8601 format)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "datetime": {
                    "type": "string",
                    "description": "ISO 8601 UTC datetime, e.g. '2025-12-31T23:59:00Z'"
                },
                "task": {
                    "type": "string",
                    "description": "Description of the task to execute"
                }
            },
            "required": ["datetime", "task"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ScheduleOnceArgs = parse_args("schedule_once", ctx.args)?;

        let dt: chrono::DateTime<chrono::Utc> = match args.datetime.parse() {
            Ok(dt) => dt,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Invalid datetime: {e}"),
                ));
            }
        };

        if dt <= chrono::Utc::now() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Schedule is in the past",
            ));
        }

        let job = self
            .registry
            .create_with_metadata(
                JobSchedule::Once(dt),
                args.task,
                true,
                self.extra_metadata.clone(),
            )
            .await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Scheduled one-time job {} at {}",
                job.id,
                dt.format("%Y-%m-%d %H:%M UTC")
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("task");
        format!("Scheduling one-time: {task}")
    }
}

// ---------------------------------------------------------------------------
// CronHistoryTool
// ---------------------------------------------------------------------------

/// Tool that shows recent cron job execution history for debugging.
pub struct CronHistoryTool {
    log: Arc<CronLog>,
}

impl CronHistoryTool {
    /// Create a new instance backed by the given log.
    pub fn new(log: Arc<CronLog>) -> Self {
        Self { log }
    }
}

#[derive(Deserialize)]
struct CronHistoryArgs {
    job_id: Option<String>,
    count: Option<usize>,
}

#[async_trait]
impl Tool for CronHistoryTool {
    fn name(&self) -> &str {
        "cron_history"
    }

    fn description(&self) -> &str {
        "Show recent cron job execution history. Use this to debug whether jobs ran, when they ran, and whether they succeeded or failed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Filter to a specific job UUID. Omit to see all jobs."
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent entries to return (default 20, max 100)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CronHistoryArgs = parse_args("cron_history", ctx.args)?;
        let count = args.count.unwrap_or(20).min(100);

        let entries = if let Some(id_str) = args.job_id {
            let id: uuid::Uuid = match id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Invalid UUID: {id_str}"),
                    ));
                }
            };
            self.log.for_job(id, count).await?
        } else {
            self.log.recent(count).await?
        };

        if entries.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No cron execution history found.",
            ));
        }

        let mut lines = Vec::with_capacity(entries.len() + 1);
        lines.push(format!(
            "{:<36}  {:<20}  {:<7}  {}",
            "Job ID", "Started", "Status", "Task"
        ));
        for entry in &entries {
            let status = if entry.success { "ok" } else { "FAILED" };
            let started = entry.started_at.format("%Y-%m-%d %H:%M UTC");
            lines.push(format!(
                "{:<36}  {:<20}  {:<7}  {}",
                entry.job_id, started, status, entry.task
            ));
            if let Some(err) = &entry.error {
                lines.push(format!("  └─ error: {err}"));
            }
            if let Some(key) = &entry.session_key {
                lines.push(format!("  └─ session: {key}"));
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, lines.join("\n")))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Checking cron history".to_string()
    }
}
