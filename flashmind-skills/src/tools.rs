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
use crate::registry::DiskSkillProvider;
use crate::runner::SkillRunner;
use crate::skill::SkillProvider;

// ---------------------------------------------------------------------------
// SkillListTool
// ---------------------------------------------------------------------------

/// Lists all discovered skills with name and description.
pub struct SkillListTool {
    pub provider: Arc<RwLock<dyn SkillProvider>>,
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
        let provider = self.provider.read().await;
        let skills = provider.list();

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
    pub provider: Arc<RwLock<dyn SkillProvider>>,
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
        let provider = self.provider.read().await;

        match provider.get(&args.name) {
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
    pub provider: Arc<RwLock<dyn SkillProvider>>,
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
        let provider = self.provider.read().await;

        let skill = match provider.get(&args.name) {
            Some(s) => s.clone(),
            None => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Skill not found: {}", args.name),
                ));
            }
        };
        drop(provider);

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
// SkillSaveTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SkillSaveArgs {
    name: String,
    description: String,
    procedure: String,
}

/// Create or update a skill. Writes a `SKILL.md` file with the given
/// name, description, and procedure body. Use after figuring out a
/// multi-step workflow the user might want to repeat.
pub struct SkillSaveTool {
    pub provider: Arc<RwLock<DiskSkillProvider>>,
}

#[async_trait]
impl Tool for SkillSaveTool {
    fn name(&self) -> &str {
        "skill_save"
    }

    fn description(&self) -> &str {
        "Save a reusable skill — a multi-step procedure you've figured out. \
         Creates the skill if it doesn't exist, updates it if it does. \
         The name and description appear in your system prompt for future recall."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Kebab-case name (e.g. 'deploy-staging')"
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary of what the skill does"
                },
                "procedure": {
                    "type": "string",
                    "description": "Full step-by-step instructions including commands, parameters, pitfalls, and verification"
                }
            },
            "required": ["name", "description", "procedure"]
        })
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Saving skill '{name}'")
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SkillSaveArgs = ctx.parse_args(self.name())?;

        if args.name.is_empty() || args.description.is_empty() || args.procedure.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "name, description, and procedure are all required",
            ));
        }

        let provider = self.provider.read().await;
        let base_dir = provider
            .search_dirs()
            .first()
            .cloned()
            .or_else(|| ctx.working_dir.map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let existed = provider.get(&args.name).is_some();
        drop(provider);

        let skill_dir = base_dir.join(&args.name);
        tokio::fs::create_dir_all(&skill_dir).await?;

        let content = format!(
            "---\nname: {}\ndescription: {}\n---\n{}",
            args.name, args.description, args.procedure
        );
        tokio::fs::write(skill_dir.join("SKILL.md"), &content).await?;

        self.provider.write().await.refresh().await?;

        let verb = if existed { "updated" } else { "created" };
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Skill '{}' {verb}", args.name),
        ))
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
    pub provider: Arc<RwLock<DiskSkillProvider>>,
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

        let provider = self.provider.read().await;
        let base_dir = provider
            .search_dirs()
            .first()
            .cloned()
            .or_else(|| ctx.working_dir.map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        drop(provider);

        let env_vars = args.env.map(|m| m.into_iter().collect::<Vec<_>>());

        let skill_dir = SkillInstaller::install(&base_dir, &args.name, env_vars).await?;

        self.provider.write().await.refresh().await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Skill '{}' installed at {}", args.name, skill_dir.display()),
        ))
    }
}
