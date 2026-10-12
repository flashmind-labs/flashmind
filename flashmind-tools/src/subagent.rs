//! Agent tools — delegate tasks to and control spawned agents.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use flashmind_core::{AgentManager, SpawnBuilder};
use flashmind_types::{
    AgentLlmConfig, LlmProvider, ToolRegistry,
    tool::{Tool, ToolContext, ToolResult},
};

// ---------------------------------------------------------------------------
// DelegateTool
// ---------------------------------------------------------------------------

/// Spawn an agent to handle a task independently.
pub struct DelegateTool {
    manager: Arc<AgentManager>,
    provider: Arc<dyn LlmProvider>,
    llm: Option<AgentLlmConfig>,
    parent_tools: Option<Arc<RwLock<ToolRegistry>>>,
    instructions: String,
}

const SUBAGENT_TOOL_NAMES: &[&str] = &[
    "delegate",
    "agent_status",
    "agent_wait",
    "communicate",
    "agent_terminate",
];

impl DelegateTool {
    /// Create a new delegate tool.
    pub fn new(manager: Arc<AgentManager>, provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            manager,
            provider,
            llm: None,
            parent_tools: None,
            instructions: String::new(),
        }
    }

    /// Set the LLM config that spawned agents inherit.
    pub fn with_llm(mut self, llm: AgentLlmConfig) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the parent tool registry. Spawned agents inherit a clone of this
    /// registry (minus subagent management tools to prevent recursive spawning).
    pub fn with_tools(mut self, tools: Arc<RwLock<ToolRegistry>>) -> Self {
        self.parent_tools = Some(tools);
        self
    }

    /// Add reusable instructions to the spawned agent's context.
    pub fn with_instructions(mut self, instructions: String) -> Self {
        self.instructions = instructions;
        self
    }
}

#[derive(Deserialize)]
struct DelegateArgs {
    task: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
}

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a task to an agent that runs independently in the background. \
         Returns the agent ID immediately — use `agent_status` to check progress \
         or `agent_wait` to block until completion."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task to delegate. Be specific and self-contained."
                },
                "name": {
                    "type": "string",
                    "description": "Optional human-readable name for the agent (e.g. 'Researcher'). \
                                    Allows addressing the agent by name in communicate, agent_status, \
                                    agent_wait, and agent_terminate. Must be unique among active agents."
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt for the agent."
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: DelegateArgs = ctx.parse_args("delegate")?;

        let mut builder = SpawnBuilder::new(args.task.clone(), self.provider.clone())
            .instructions(self.instructions.clone());

        if let Some(name) = &args.name {
            builder = builder.name(name.clone());
        }

        if let Some(prompt) = args.system_prompt {
            builder = builder.system_prompt(prompt);
        }

        if let Some(ref llm) = self.llm {
            builder = builder.llm(llm.clone());
        }

        if let Some(ref parent_tools) = self.parent_tools {
            let mut tools = parent_tools.read().await.clone();
            let total = tools.list().len();
            for name in SUBAGENT_TOOL_NAMES {
                tools.remove(name);
            }
            tracing::info!(
                tool_count = tools.list().len(),
                total_before_strip = total,
                tools = ?tools.list(),
                "delegate: passing tools to child agent"
            );
            builder = builder.tools(tools);
        } else {
            tracing::warn!("delegate: no parent_tools set, child gets empty registry");
        }

        match self.manager.spawn(builder).await {
            Ok(id) => {
                let short_id = &id.simple().to_string()[..8];
                let label = match &args.name {
                    Some(name) => format!("{name} ({short_id})"),
                    None => short_id.to_string(),
                };
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("Agent spawned: {label}\nTask: {}", args.task),
                ))
            }
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("task");
        let truncated: String = task.chars().take(60).collect();
        format!("Delegating: {truncated}")
    }
}

// ---------------------------------------------------------------------------
// AgentStatusTool
// ---------------------------------------------------------------------------

/// Check the status of one or all agents.
pub struct AgentStatusTool {
    manager: Arc<AgentManager>,
}

impl AgentStatusTool {
    /// Create a new status tool.
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize)]
struct StatusArgs {
    #[serde(default)]
    agent: Option<String>,
}

#[async_trait]
impl Tool for AgentStatusTool {
    fn name(&self) -> &str {
        "agent_status"
    }

    fn description(&self) -> &str {
        "Check the status of an agent by name or ID, or list all active agents if omitted."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Optional agent name or ID. Omit to list all."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StatusArgs = ctx.parse_args("agent_status")?;

        if let Some(agent_str) = args.agent {
            let id = resolve_agent(&self.manager, &agent_str).await?;
            match self.manager.status(id).await {
                Ok(status) => Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("Agent {agent_str}: {status}"),
                )),
                Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
            }
        } else {
            let statuses = self.manager.all_statuses().await;
            if statuses.is_empty() {
                return Ok(ToolResult::success(ctx.tool_call_id, "No active agents."));
            }

            let mut output = String::from("Active agents:\n");
            for (id, name, task, status) in &statuses {
                let short_id = &id.simple().to_string()[..8];
                let label = match name {
                    Some(n) => format!("{n} ({short_id})"),
                    None => short_id.to_string(),
                };
                output.push_str(&format!("  {label}: {status} — {task}\n"));
            }
            Ok(ToolResult::success(ctx.tool_call_id, output))
        }
    }

    fn humanize(&self, _args: &Value) -> String {
        "Checking agent status".to_string()
    }
}

// ---------------------------------------------------------------------------
// AgentTerminateTool
// ---------------------------------------------------------------------------

/// Cancel a running agent.
pub struct AgentTerminateTool {
    manager: Arc<AgentManager>,
}

impl AgentTerminateTool {
    /// Create a new terminate tool.
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize)]
struct TerminateArgs {
    agent: String,
}

#[async_trait]
impl Tool for AgentTerminateTool {
    fn name(&self) -> &str {
        "agent_terminate"
    }

    fn description(&self) -> &str {
        "Cancel a running agent by name or ID. The agent will stop at the next cancellation check."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "The agent name or ID to terminate."
                }
            },
            "required": ["agent"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: TerminateArgs = ctx.parse_args("agent_terminate")?;

        let id = resolve_agent(&self.manager, &args.agent).await?;

        match self.manager.terminate(id).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Agent {} terminated.", args.agent),
            )),
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let agent = args.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Terminating agent {agent}")
    }
}

// ---------------------------------------------------------------------------
// AgentWaitTool
// ---------------------------------------------------------------------------

/// Wait for an agent to complete and return its result.
pub struct AgentWaitTool {
    manager: Arc<AgentManager>,
}

impl AgentWaitTool {
    /// Create a new wait tool.
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize)]
struct WaitArgs {
    agent: String,
    timeout_secs: u64,
}

#[async_trait]
impl Tool for AgentWaitTool {
    fn name(&self) -> &str {
        "agent_wait"
    }

    fn description(&self) -> &str {
        "Wait for an agent to complete and return its final response. \
         Accepts the agent's name or ID. Requires a timeout in seconds."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "The agent name or ID to wait for."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Maximum seconds to wait before giving up."
                }
            },
            "required": ["agent", "timeout_secs"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(1800)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: WaitArgs = ctx.parse_args("agent_wait")?;
        let timeout = std::time::Duration::from_secs(args.timeout_secs.min(1800));

        let id = resolve_agent(&self.manager, &args.agent).await?;

        tokio::select! {
            result = self.manager.wait(id) => {
                match result {
                    Ok(result) => Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!("Agent {} completed:\n\n{result}", args.agent),
                    )),
                    Err(e) => Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Agent {} failed: {e}", args.agent),
                    )),
                }
            }
            _ = tokio::time::sleep(timeout) => {
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Timed out waiting for agent {} after {}s", args.agent, args.timeout_secs),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let agent = args.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Waiting for agent {agent}")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve an agent reference (name, ID prefix, or full UUID) to a UUID.
async fn resolve_agent(manager: &AgentManager, agent: &str) -> anyhow::Result<uuid::Uuid> {
    manager.resolve(agent).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, Message, Provider, Role, StreamEvent,
    };

    use crate::ToolBuilder;
    use crate::tests::execute_tool;

    struct RecordingProvider {
        messages: Mutex<Vec<Message>>,
    }

    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            *self.messages.lock().unwrap() = request.messages;
            Box::pin(futures::stream::iter([
                Ok(StreamEvent::ContentDelta("Done.".to_string())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    #[tokio::test]
    async fn delegate_shares_instructions_without_changing_visible_task_or_system_prompt() {
        let skill_instructions =
            "Available skills: /example (source: /path/to/skills/example/SKILL.md)";
        for instructions in ["", skill_instructions] {
            for system_prompt in [None, Some("You are a test assistant.")] {
                let manager = Arc::new(AgentManager::new(1, 1));
                let provider = Arc::new(RecordingProvider {
                    messages: Mutex::new(Vec::new()),
                });
                let builder = ToolBuilder::new();
                let builder = if instructions.is_empty() {
                    builder.subagents(manager.clone(), provider.clone(), None)
                } else {
                    builder.subagents_with_instructions(
                        manager.clone(),
                        provider.clone(),
                        None,
                        instructions.to_string(),
                    )
                };
                let (registry, _) = builder.build_with_sync().await;
                let result = execute_tool(
                    registry.get("delegate").unwrap().as_ref(),
                    "call",
                    json!({
                        "task": "Inspect the code.",
                        "name": "worker",
                        "system_prompt": system_prompt,
                    }),
                )
                .await
                .unwrap();
                assert!(result.is_success());
                assert!(result.output().ends_with("\nTask: Inspect the code."));
                assert!(!result.output().contains(skill_instructions));

                let id = manager.resolve("worker").await.unwrap();
                assert_eq!(manager.all_statuses().await[0].2, "Inspect the code.");
                tokio::time::timeout(Duration::from_secs(5), manager.wait(id))
                    .await
                    .unwrap()
                    .unwrap();
                let messages = provider.messages.lock().unwrap();
                let system = &messages
                    .iter()
                    .find(|message| message.role == Role::System)
                    .unwrap()
                    .content;
                assert!(system.contains("# Task\n\nInspect the code."));
                assert_eq!(
                    system.contains(skill_instructions),
                    !instructions.is_empty()
                );
                match system_prompt {
                    Some(prompt) => assert!(system.starts_with(prompt)),
                    None => assert!(
                        system.starts_with("You are a background agent running autonomously.")
                    ),
                }
                if instructions.is_empty() {
                    assert!(system.ends_with("# Task\n\nInspect the code."));
                }
            }
        }
    }

    #[test]
    fn test_delegate_tool_metadata() {
        let provider: Arc<dyn flashmind_types::LlmProvider> =
            Arc::new(crate::tests::mock_provider());
        let manager = Arc::new(AgentManager::new(10, 3));
        let tool = DelegateTool::new(manager, provider);
        assert_eq!(tool.name(), "delegate");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_status_tool_metadata() {
        let manager = Arc::new(AgentManager::new(10, 3));
        let tool = AgentStatusTool::new(manager);
        assert_eq!(tool.name(), "agent_status");
    }

    #[test]
    fn test_terminate_tool_metadata() {
        let manager = Arc::new(AgentManager::new(10, 3));
        let tool = AgentTerminateTool::new(manager);
        assert_eq!(tool.name(), "agent_terminate");
    }

    #[test]
    fn test_wait_tool_metadata() {
        let manager = Arc::new(AgentManager::new(10, 3));
        let tool = AgentWaitTool::new(manager);
        assert_eq!(tool.name(), "agent_wait");
        assert_eq!(tool.timeout_secs(), Some(1800));
    }

    #[tokio::test]
    async fn test_resolve_agent_not_found() {
        let manager = AgentManager::new(10, 3);
        let result = resolve_agent(&manager, "Pacifist").await;
        assert!(result.is_err());
    }
}
