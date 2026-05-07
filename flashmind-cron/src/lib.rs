//! Cron job scheduling with pluggable storage for Flashmind.
//!
//! This crate provides a complete cron scheduling system including:
//!
//! - **Parsing**: Nom-based POSIX 5-field cron expression parser with support
//!   for wildcards, steps, ranges, lists, and named months/days.
//! - **Scheduling**: Match datetimes against cron expressions and compute
//!   next fire times.
//! - **Jobs**: Recurring (`Cron`) and one-shot (`Once`) job types with
//!   pluggable metadata.
//! - **Storage**: [`CronStore`] trait for custom backends, with a built-in
//!   [`TomlCronStore`] for file-based persistence.
//! - **Registry**: CRUD operations for managing jobs.
//! - **Runner**: Async scheduler that spawns per-job tasks with cancellation
//!   support.
//! - **Tools**: Five [`Tool`](flashmind_types::tool::Tool) implementations
//!   for agent-driven cron management.

pub mod error;
pub mod job;
pub mod parse;
pub mod registry;
pub mod runner;
pub mod schedule;
pub mod store;
pub mod tools;

pub use error::CronError;
pub use job::{CronJob, JobSchedule};
pub use parse::{CronExpr, FieldSet};
pub use registry::CronRegistry;
pub use runner::{CronHandler, CronRunner};
pub use schedule::CronSchedule;
pub use store::{CronStore, TomlCronStore};
pub use tools::{CronCreateTool, CronDeleteTool, CronEditTool, CronListTool, ScheduleOnceTool};
