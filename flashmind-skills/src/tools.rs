//! Tool implementations for skill operations.
//!
//! Provides four tools that expose skill functionality to the agent:
//! [`SkillListTool`], [`SkillLoadTool`], [`SkillRunTool`], and [`SkillInstallTool`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::installer::SkillInstaller;
use crate::registry::SkillRegistry;
use crate::runner::SkillRunner;

// ---------------------------------------------------------------------------
// SkillListTool
// ---------------------------------------------------------------------------

/// Lists all discovered skills with name and description.
pub struct SkillListTool {
    pub registry: Arc<RwLock<SkillRegistry>>,
}

#[async_trait]
impl Tool for SkillListTool {
    fn name(&self) -> &str {
        "skill_list"
    }

    fn description(&self) -> &str {
        "List all discovered skills with their names and descriptions."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let registry = self.registry.read().await;
        let skills = registry.list();

        if skills.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No skills discovered.",
            ));
        }

        let listing: Vec<Value> = skills
            .iter()
            .map(|s| {
                json!({
                    "name": s.meta.name,
                    "description": s.meta.description,
                })
            })
            .collect();

        let output = serde_json::to_string_pretty(&listing)?;
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }
}

// ---------------------------------------------------------------------------
// SkillLoadTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SkillLoadArgs {
    name: String,
}

/// Loads a skill's body content by name.
pub struct SkillLoadTool {
    pub registry: Arc<RwLock<SkillRegistry>>,
}

#[async_trait]
impl Tool for SkillLoadTool {
    fn name(&self) -> &str {
        "skill_load"
    }

    fn description(&self) -> &str {
        "Load a skill's full content by name. Returns the skill's markdown body."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load."
                }
            },
            "required": ["name"]
        })
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Loading skill {name}")
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SkillLoadArgs = ctx.parse_args(self.name())?;
        let registry = self.registry.read().await;

        match registry.get(&args.name) {
            Some(skill) => Ok(ToolResult::success(ctx.tool_call_id, &skill.body)),
            None => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Skill not found: {}", args.name),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// SkillRunTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SkillRunArgs {
    name: String,
    command: String,
}

/// Executes a command in a skill's directory.
pub struct SkillRunTool {
    pub registry: Arc<RwLock<SkillRegistry>>,
    pub runner: Arc<SkillRunner>,
}

#[async_trait]
impl Tool for SkillRunTool {
    fn name(&self) -> &str {
        "skill_run"
    }

    fn description(&self) -> &str {
        "Execute a shell command in a skill's directory. The skill's .env is loaded and secrets are redacted from output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to run in."
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute in the skill's directory."
                }
            },
            "required": ["name", "command"]
        })
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Running {cmd} in skill {name}")
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SkillRunArgs = ctx.parse_args(self.name())?;
        let registry = self.registry.read().await;

        let skill = match registry.get(&args.name) {
            Some(s) => s.clone(),
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Skill not found: {}", args.name),
                ));
            }
        };
        drop(registry);

        match self.runner.run(&skill, &args.command).await {
            Ok(output) => {
                let result = json!({
                    "exit_code": output.exit_code,
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                });
                Ok(ToolResult::success(ctx.tool_call_id, result.to_string()))
            }
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// SkillInstallTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SkillInstallArgs {
    name: String,
    #[serde(default)]
    env: Option<HashMap<String, String>>,
}

/// Creates a new skill directory with optional environment variables.
pub struct SkillInstallTool {
    pub registry: Arc<RwLock<SkillRegistry>>,
}

#[async_trait]
impl Tool for SkillInstallTool {
    fn name(&self) -> &str {
        "skill_install"
    }

    fn description(&self) -> &str {
        "Create a new skill directory. Optionally provide env vars as key-value pairs to write a .env file."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new skill."
                },
                "env": {
                    "type": "object",
                    "description": "Optional environment variables as KEY:VALUE pairs for the skill's .env file.",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["name"]
        })
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Installing skill {name}")
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SkillInstallArgs = ctx.parse_args(self.name())?;

        let registry = self.registry.read().await;
        let search_dirs: Vec<_> = registry.list().iter().map(|s| s.dir.clone()).collect();
        drop(registry);

        // Use the first search dir's parent as the base, or fall back to cwd
        let base_dir = search_dirs
            .first()
            .and_then(|d| d.parent().map(|p| p.to_path_buf()))
            .or_else(|| ctx.working_dir.map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let env_vars = args.env.map(|m| m.into_iter().collect::<Vec<_>>());

        let skill_dir = SkillInstaller::install(&base_dir, &args.name, env_vars).await?;

        // Re-discover so the new skill is available
        let mut registry = self.registry.write().await;
        registry.discover().await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Skill '{}' installed at {}", args.name, skill_dir.display()),
        ))
    }
}
