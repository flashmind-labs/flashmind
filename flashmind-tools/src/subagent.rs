//! Agent tools — delegate tasks, communicate with, and control spawned agents.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_core::AgentManager;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

// ---------------------------------------------------------------------------
// DelegateTool
// ---------------------------------------------------------------------------

/// Spawn an agent to handle a task independently.
pub struct DelegateTool {
    manager: Arc<AgentManager>,
    provider: Arc<dyn flashmind_types::LlmProvider>,
}

impl DelegateTool {
    /// Create a new delegate tool.
    pub fn new(
        manager: Arc<AgentManager>,
        provider: Arc<dyn flashmind_types::LlmProvider>,
    ) -> Self {
        Self { manager, provider }
    }
}

#[derive(Deserialize)]
struct DelegateArgs {
    task: String,
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

        let mut builder =
            flashmind_core::SpawnBuilder::new(args.task.clone(), self.provider.clone());

        if let Some(prompt) = args.system_prompt {
            builder = builder.system_prompt(prompt);
        }

        match self.manager.spawn(builder).await {
            Ok(id) => {
                let short_id = &id.simple().to_string()[..8];
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("Agent spawned: {short_id}\nTask: {}", args.task),
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
// CommunicateTool
// ---------------------------------------------------------------------------

/// Send a message to a running agent.
pub struct CommunicateTool {
    manager: Arc<AgentManager>,
}

impl CommunicateTool {
    /// Create a new communicate tool.
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize)]
struct CommunicateArgs {
    id: String,
    message: String,
}

#[async_trait]
impl Tool for CommunicateTool {
    fn name(&self) -> &str {
        "communicate"
    }

    fn description(&self) -> &str {
        "Send a message to a running agent. The message will be processed \
         on the agent's next turn as a user message."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The agent ID (8-character hex prefix from delegate)."
                },
                "message": {
                    "type": "string",
                    "description": "Message to send to the agent."
                }
            },
            "required": ["id", "message"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CommunicateArgs = ctx.parse_args("communicate")?;

        let id = parse_agent_id(&args.id)?;

        match self.manager.send(id, args.message).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Message sent to agent {}", args.id),
            )),
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Messaging agent {id}")
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
    id: Option<String>,
}

#[async_trait]
impl Tool for AgentStatusTool {
    fn name(&self) -> &str {
        "agent_status"
    }

    fn description(&self) -> &str {
        "Check the status of an agent by ID, or list all active agents if no ID is provided."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Optional agent ID. Omit to list all."
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: StatusArgs = ctx.parse_args("agent_status")?;

        if let Some(id_str) = args.id {
            let id = parse_agent_id(&id_str)?;
            match self.manager.status(id).await {
                Ok(status) => Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("Agent {id_str}: {status}"),
                )),
                Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
            }
        } else {
            let statuses = self.manager.all_statuses().await;
            if statuses.is_empty() {
                return Ok(ToolResult::success(ctx.tool_call_id, "No active agents."));
            }

            let mut output = String::from("Active agents:\n");
            for (id, task, status) in &statuses {
                let short_id = &id.simple().to_string()[..8];
                output.push_str(&format!("  {short_id}: {status} — {task}\n"));
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
    id: String,
}

#[async_trait]
impl Tool for AgentTerminateTool {
    fn name(&self) -> &str {
        "agent_terminate"
    }

    fn description(&self) -> &str {
        "Cancel a running agent by ID. The agent will stop at the next cancellation check."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The agent ID to terminate."
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: TerminateArgs = ctx.parse_args("agent_terminate")?;

        let id = parse_agent_id(&args.id)?;

        match self.manager.terminate(id).await {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Agent {} terminated.", args.id),
            )),
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Terminating agent {id}")
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
    id: String,
}

#[async_trait]
impl Tool for AgentWaitTool {
    fn name(&self) -> &str {
        "agent_wait"
    }

    fn description(&self) -> &str {
        "Wait for an agent to complete and return its final response. \
         This blocks until the agent finishes."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The agent ID to wait for."
                }
            },
            "required": ["id"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(1800)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: WaitArgs = ctx.parse_args("agent_wait")?;

        let id = parse_agent_id(&args.id)?;

        match self.manager.wait(id).await {
            Ok(result) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Agent {} completed:\n\n{result}", args.id),
            )),
            Err(e) => Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Agent {} failed: {e}", args.id),
            )),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Waiting for agent {id}")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an 8-char hex prefix or full UUID into a Uuid.
fn parse_agent_id(s: &str) -> anyhow::Result<uuid::Uuid> {
    if s.len() == 8 {
        let padded = format!("{s}0000-0000-0000-000000000000");
        uuid::Uuid::parse_str(&padded).map_err(|_| anyhow::anyhow!("invalid agent ID: {s}"))
    } else {
        uuid::Uuid::parse_str(s).map_err(|_| anyhow::anyhow!("invalid agent ID: {s}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delegate_tool_metadata() {
        let provider: Arc<dyn flashmind_types::LlmProvider> =
            Arc::new(crate::tests::mock_provider());
        let manager = Arc::new(AgentManager::new(
            flashmind_types::InjectQueue::new(),
            10,
            3,
        ));
        let tool = DelegateTool::new(manager, provider);
        assert_eq!(tool.name(), "delegate");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_communicate_tool_metadata() {
        let manager = Arc::new(AgentManager::new(
            flashmind_types::InjectQueue::new(),
            10,
            3,
        ));
        let tool = CommunicateTool::new(manager);
        assert_eq!(tool.name(), "communicate");
    }

    #[test]
    fn test_status_tool_metadata() {
        let manager = Arc::new(AgentManager::new(
            flashmind_types::InjectQueue::new(),
            10,
            3,
        ));
        let tool = AgentStatusTool::new(manager);
        assert_eq!(tool.name(), "agent_status");
    }

    #[test]
    fn test_terminate_tool_metadata() {
        let manager = Arc::new(AgentManager::new(
            flashmind_types::InjectQueue::new(),
            10,
            3,
        ));
        let tool = AgentTerminateTool::new(manager);
        assert_eq!(tool.name(), "agent_terminate");
    }

    #[test]
    fn test_wait_tool_metadata() {
        let manager = Arc::new(AgentManager::new(
            flashmind_types::InjectQueue::new(),
            10,
            3,
        ));
        let tool = AgentWaitTool::new(manager);
        assert_eq!(tool.name(), "agent_wait");
        assert_eq!(tool.timeout_secs(), Some(1800));
    }
}
