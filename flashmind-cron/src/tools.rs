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

use chrono::{DateTime, NaiveDateTime, Utc};
use chrono_tz::Tz;

use crate::job::JobSchedule;
use crate::log::CronLog;
use crate::registry::CronRegistry;
use crate::schedule::{CronSchedule, to_utc};

/// Parse a job UUID argument, or build the tool failure the caller returns.
fn parse_job_id(raw: &str, tool_call_id: &str) -> Result<uuid::Uuid, ToolResult> {
    raw.parse()
        .map_err(|_| ToolResult::failure(tool_call_id, format!("Invalid UUID: {raw}")))
}

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

        let metadata = self.extra_metadata.clone();
        let job = self
            .registry
            .create_with_metadata(JobSchedule::Cron(args.schedule), args.task, false, metadata)
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

        let id = match parse_job_id(&args.id, ctx.tool_call_id) {
            Ok(id) => id,
            Err(failure) => return Ok(failure),
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

        let id = match parse_job_id(&args.id, ctx.tool_call_id) {
            Ok(id) => id,
            Err(failure) => return Ok(failure),
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

/// Local datetimes the model writes, with or without seconds.
const NAIVE_FORMATS: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M",
];

/// `raw` as a UTC instant. A datetime with an offset is taken as is; one
/// without is local to `tz`, or UTC when there is none.
fn parse_datetime(raw: &str, tz: Option<Tz>) -> Result<DateTime<Utc>, String> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw)
        .or_else(|_| DateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%:z"))
        .or_else(|_| DateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M%:z"))
    {
        return Ok(dt.with_timezone(&Utc));
    }
    let naive = NAIVE_FORMATS
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(raw, format).ok())
        .ok_or_else(|| {
            format!("Invalid datetime {raw:?}: write it as ISO 8601, like 2025-12-31T23:59:00")
        })?;
    match tz {
        Some(tz) => to_utc(naive, tz).ok_or_else(|| {
            "Datetime falls in a DST gap (does not exist in that timezone)".to_owned()
        }),
        None => Ok(naive.and_utc()),
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
        "Schedule a one-time task at a specific datetime. The datetime is interpreted in the user's timezone if configured, otherwise as UTC."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "datetime": {
                    "type": "string",
                    "description": "ISO 8601 datetime, e.g. '2025-12-31T23:59:00'. Interpreted in user's timezone if set, otherwise UTC."
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

        let tz: Option<Tz> = self
            .extra_metadata
            .get("timezone")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());

        let dt = match parse_datetime(args.datetime.trim(), tz) {
            Ok(dt) => dt,
            Err(error) => return Ok(ToolResult::failure(ctx.tool_call_id, error)),
        };

        let now = Utc::now();
        if dt <= now {
            let now = match tz {
                Some(tz) => format!(
                    "{} {tz}",
                    now.with_timezone(&tz).format("%Y-%m-%dT%H:%M:%S")
                ),
                None => format!("{} UTC", now.format("%Y-%m-%dT%H:%M:%S")),
            };
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Schedule is in the past: it is now {now}. Pick a later time, or do the task now."
                ),
            ));
        }

        let metadata = self.extra_metadata.clone();
        let job = self
            .registry
            .create_with_metadata(JobSchedule::Once(dt), args.task, true, metadata)
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
            let id = match parse_job_id(&id_str, ctx.tool_call_id) {
                Ok(id) => id,
                Err(failure) => return Ok(failure),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(raw: &str) -> DateTime<Utc> {
        raw.parse().expect("test instants are valid RFC 3339")
    }

    #[test]
    fn local_datetimes_parse_with_or_without_seconds() {
        let tokyo: Tz = "Asia/Tokyo".parse().expect("a known zone");
        let nine = utc("2026-10-14T00:00:00Z");
        for raw in [
            "2026-10-14T09:00:00",
            "2026-10-14T09:00",
            "2026-10-14 09:00",
            "2026-10-14 09:00:00",
            "2026-10-14T09:00:00+09:00",
            "2026-10-14T00:00:00Z",
            "2026-10-14T09:00+09:00",
        ] {
            assert_eq!(parse_datetime(raw, Some(tokyo)), Ok(nine), "{raw}");
        }
        assert_eq!(parse_datetime("2026-10-14T00:00", None), Ok(nine));
    }

    #[test]
    fn unparseable_datetimes_say_the_format() {
        let error = parse_datetime("tomorrow 9am", None).expect_err("not a datetime");
        assert!(error.contains("2025-12-31T23:59:00"), "{error}");
    }
}
