//! Agent tools — delegate tasks to and control spawned agents.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use flashmind_core::{AgentManager, ConversationEntry, SpawnBuilder};
use flashmind_types::{
    AgentLlmConfig, LlmProvider, Model, ToolRegistry,
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
    "agent_model",
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
    model: Option<Model>,
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
                },
                "model": {
                    "type": "string",
                    "description": "Optional provider:model ID from list_models. Must use the current provider. Inherits the parent's model when omitted; routing and other settings are preserved."
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: DelegateArgs = ctx.parse_args("delegate")?;

        let mut builder = SpawnBuilder::new(
            args.task.clone(),
            ctx.provider
                .clone()
                .unwrap_or_else(|| self.provider.clone()),
        )
        .instructions(self.instructions.clone());

        if let Some(path) = ctx.working_dir {
            builder = builder.working_dir(path.clone());
        }

        if let Some(name) = &args.name {
            builder = builder.name(name.clone());
        }

        if let Some(prompt) = args.system_prompt {
            builder = builder.system_prompt(prompt);
        }

        if let Some(llm) = ctx.llm.or(self.llm.as_ref()) {
            builder = builder.llm(llm.clone());
        }
        if let Some(model) = args.model {
            builder = builder.model(model);
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
                let label = agent_label(id, args.name.as_deref());
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
            let id = self.manager.resolve(&agent_str).await?;
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
                let label = agent_label(*id, name.as_deref());
                output.push_str(&format!("  {label}: {status} — {task}\n"));
            }
            Ok(ToolResult::success(ctx.tool_call_id, output))
        }
    }

    fn humanize(&self, _args: &Value) -> String {
        "Checking agent status".to_string()
    }
}

/// Change the model used by a retained subagent conversation.
pub struct AgentModelTool {
    manager: Arc<AgentManager>,
}

impl AgentModelTool {
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize)]
struct ModelArgs {
    agent: String,
    model: Model,
}

#[async_trait]
impl Tool for AgentModelTool {
    fn name(&self) -> &str {
        "agent_model"
    }

    fn description(&self) -> &str {
        "Change a subagent's model before its next model request. Current work finishes normally; history and settings are preserved."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": { "type": "string", "description": "Agent name or ID." },
                "model": { "type": "string", "description": "Provider:model ID from list_models. Must use the subagent's current provider." }
            },
            "required": ["agent", "model"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ModelArgs = ctx.parse_args("agent_model")?;
        let id = self.manager.resolve(&args.agent).await?;
        match self.manager.set_model(id, args.model.clone()) {
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Agent {} will use {} on its next model request.",
                    args.agent, args.model
                ),
            )),
            Err(error) => Ok(ToolResult::failure(ctx.tool_call_id, error.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let agent = args.get("agent").and_then(Value::as_str).unwrap_or("?");
        format!("Changing model for agent {agent}")
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

        let id = self.manager.resolve(&args.agent).await?;

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
// CommunicateTool
// ---------------------------------------------------------------------------

/// Send a message to one of this agent's own subagents.
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
    agent: String,
    message: String,
}

#[async_trait]
impl Tool for CommunicateTool {
    fn name(&self) -> &str {
        "communicate"
    }

    fn description(&self) -> &str {
        "Send a message to one of your agents by name or ID, to steer it, add details \
         or ask about its result. A running agent answers it once its current response \
         is done; a finished agent resumes with its full history. The reply arrives \
         like its first result."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "The agent name or ID."
                },
                "message": {
                    "type": "string",
                    "description": "What to tell the agent."
                }
            },
            "required": ["agent", "message"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CommunicateArgs = ctx.parse_args("communicate")?;
        if args.message.trim().is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "The message is empty.",
            ));
        }
        let id = match self.manager.resolve(&args.agent).await {
            Ok(id) => id,
            Err(error) => return Ok(ToolResult::failure(ctx.tool_call_id, error.to_string())),
        };
        let resumed = self.manager.conversation(id)?.is_finished();
        match self
            .manager
            .follow_up(id, ConversationEntry::user(args.message))
        {
            Ok(()) if resumed => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Agent {} resumed with your message.", args.agent),
            )),
            Ok(()) => Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Message queued for agent {}. It answers once its current response is done.",
                    args.agent
                ),
            )),
            Err(error) => Ok(ToolResult::failure(ctx.tool_call_id, error.to_string())),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let agent = args.get("agent").and_then(Value::as_str).unwrap_or("?");
        format!("Messaging agent {agent}")
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

        let id = self.manager.resolve(&args.agent).await?;

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

/// Display label for an agent: `name (short-id)`, or just the short ID.
fn agent_label(id: uuid::Uuid, name: Option<&str>) -> String {
    let short_id = &id.simple().to_string()[..8];
    match name {
        Some(name) => format!("{name} ({short_id})"),
        None => short_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, Provider, Role, StreamEvent,
    };

    use crate::ToolBuilder;
    use crate::tests::execute_tool;

    struct RecordingProvider {
        requests: Mutex<Vec<CompletionRequest>>,
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
            self.requests.lock().unwrap().push(request);
            Box::pin(futures_util::stream::iter([
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
                    requests: Mutex::new(Vec::new()),
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
                let requests = provider.requests.lock().unwrap();
                let messages = &requests[0].messages;
                let system = &messages
                    .iter()
                    .find(|message| message.role == Role::System)
                    .unwrap()
                    .content;
                assert!(!system.contains("Inspect the code."));
                assert_eq!(
                    messages
                        .iter()
                        .find(|message| message.role == Role::User)
                        .unwrap()
                        .content,
                    "Inspect the code."
                );
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
            }
        }
    }

    #[tokio::test]
    async fn delegate_uses_model_override_with_current_context() {
        use flashmind_types::{ProviderPreferences, SessionId};
        use tokio_util::sync::CancellationToken;

        let manager = Arc::new(AgentManager::new(1, 1));
        let fallback = Arc::new(RecordingProvider {
            requests: Mutex::new(Vec::new()),
        });
        let current = Arc::new(RecordingProvider {
            requests: Mutex::new(Vec::new()),
        });
        let llm = AgentLlmConfig::new("ollama:parent".parse().unwrap())
            .with_session_id(SessionId::new())
            .with_provider_preferences(ProviderPreferences {
                zdr: Some(true),
                ..Default::default()
            });
        let (tools, _) = ToolBuilder::new()
            .subagents(manager.clone(), fallback.clone(), None)
            .models()
            .build_with_sync()
            .await;
        let cancel = CancellationToken::new();
        let args = json!({"task": "Inspect the code.", "name": "worker", "model": "ollama:worker"});
        let result = tools
            .get("delegate")
            .unwrap()
            .execute(
                ToolContext::new("spawn", args, None, &cancel)
                    .with_llm(&llm)
                    .with_provider(current.clone()),
            )
            .await
            .unwrap();
        assert!(result.is_success(), "{}", result.output());
        manager
            .wait(manager.resolve("worker").await.unwrap())
            .await
            .unwrap();
        assert!(fallback.requests.lock().unwrap().is_empty());
        let requests = current.requests.lock().unwrap();
        assert_eq!(requests[0].model.to_string(), "ollama:worker");
        assert_eq!(requests[0].session_id, llm.session_id);
        assert_eq!(
            requests[0].provider_preferences.as_ref().unwrap().zdr,
            Some(true)
        );
        assert!(
            requests[0]
                .tools
                .iter()
                .any(|tool| tool.name == "list_models")
        );
        assert!(!requests[0].tools.iter().any(|tool| tool.name == "delegate"));
    }

    #[tokio::test]
    async fn registered_model_tool_changes_existing_conversations_and_rejects_other_providers() {
        use flashmind_core::ConversationEntry;

        let manager = Arc::new(AgentManager::new(1, 1));
        let provider = Arc::new(RecordingProvider {
            requests: Mutex::new(Vec::new()),
        });
        let (tools, _) = ToolBuilder::new()
            .subagents(manager.clone(), provider.clone(), None)
            .build_with_sync()
            .await;
        execute_tool(
            tools.get("delegate").unwrap().as_ref(),
            "spawn",
            json!({"task": "Inspect the code.", "name": "worker"}),
        )
        .await
        .unwrap();
        let id = manager.resolve("worker").await.unwrap();
        manager.wait(id).await.unwrap();
        let tool = tools.get("agent_model").unwrap();
        let result = execute_tool(
            tool.as_ref(),
            "switch",
            json!({"agent": "worker", "model": "ollama:replacement"}),
        )
        .await
        .unwrap();
        assert!(result.is_success(), "{}", result.output());
        let result = execute_tool(
            tool.as_ref(),
            "invalid",
            json!({"agent": "worker", "model": "openrouter:example/worker"}),
        )
        .await
        .unwrap();
        assert!(!result.is_success());
        manager
            .follow_up(id, ConversationEntry::user("Explain the result."))
            .unwrap();
        manager.wait(id).await.unwrap();
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[1].model.name(), "replacement");
        assert!(
            !requests[1]
                .tools
                .iter()
                .any(|tool| tool.name == "agent_model")
        );
    }

    struct GatedProvider {
        requests: Arc<Mutex<Vec<CompletionRequest>>>,
        gate: Arc<tokio::sync::Semaphore>,
    }

    #[async_trait]
    impl LlmProvider for GatedProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.requests.lock().unwrap().push(request);
            let gate = self.gate.clone();
            Box::pin(futures_util::StreamExt::chain(
                futures_util::stream::once(async move {
                    gate.acquire().await.unwrap().forget();
                    Ok(StreamEvent::ContentDelta("Done.".to_string()))
                }),
                futures_util::stream::once(async { Ok(StreamEvent::Finished(FinishReason::Stop)) }),
            ))
        }
    }

    fn last_user_message(request: &CompletionRequest) -> &str {
        &request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .unwrap()
            .content
    }

    #[tokio::test]
    async fn communicate_resumes_a_finished_agent_with_its_history() {
        let manager = Arc::new(AgentManager::new(1, 1));
        let provider = Arc::new(RecordingProvider {
            requests: Mutex::new(Vec::new()),
        });
        let (tools, _) = ToolBuilder::new()
            .subagents(manager.clone(), provider.clone(), None)
            .build_with_sync()
            .await;
        execute_tool(
            tools.get("delegate").unwrap().as_ref(),
            "spawn",
            json!({"task": "Inspect the code.", "name": "worker"}),
        )
        .await
        .unwrap();
        let id = manager.resolve("worker").await.unwrap();
        manager.wait(id).await.unwrap();

        let result = execute_tool(
            tools.get("communicate").unwrap().as_ref(),
            "send",
            json!({"agent": "worker", "message": "Explain the result."}),
        )
        .await
        .unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert!(result.output().contains("resumed"));
        manager.wait(id).await.unwrap();

        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(last_user_message(&requests[1]), "Explain the result.");
        assert!(
            requests[1]
                .messages
                .iter()
                .any(|message| message.content == "Inspect the code.")
        );
        assert!(
            !requests[1]
                .tools
                .iter()
                .any(|tool| tool.name == "communicate")
        );
    }

    #[tokio::test]
    async fn communicate_reaches_a_running_agent() {
        let manager = Arc::new(AgentManager::new(1, 1));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let provider = Arc::new(GatedProvider {
            requests: requests.clone(),
            gate: gate.clone(),
        });
        let id = manager
            .spawn(SpawnBuilder::new("Inspect the code.", provider.clone()).name("worker"))
            .await
            .unwrap();
        let tool = CommunicateTool::new(manager.clone());

        let result = execute_tool(
            &tool,
            "send",
            json!({"agent": "worker", "message": "Also check the tests."}),
        )
        .await
        .unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert!(result.output().contains("queued"));

        gate.add_permits(2);
        manager.wait(id).await.unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(last_user_message(&requests[1]), "Also check the tests.");
    }

    #[tokio::test]
    async fn communicate_refuses_unknown_agents_and_empty_messages() {
        let manager = Arc::new(AgentManager::new(1, 1));
        let tool = CommunicateTool::new(manager);
        for args in [
            json!({"agent": "nobody", "message": "Hello."}),
            json!({"agent": "nobody", "message": "  "}),
        ] {
            let result = execute_tool(&tool, "send", args).await.unwrap();
            assert!(!result.is_success());
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
        let result = manager.resolve("Pacifist").await;
        assert!(result.is_err());
    }
}
