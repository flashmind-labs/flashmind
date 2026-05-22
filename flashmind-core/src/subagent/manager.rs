//! Agent manager — orchestrates spawning, lifecycle, and control of child agents.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use flashmind_types::{AgentEvent, AgentInput};

use crate::agent::Agent;
use crate::conversation::{Conversation, ConversationEntry};

use super::builder::SpawnBuilder;
use super::handle::{AgentHandle, AgentStatus};

// ---------------------------------------------------------------------------
// AgentManager
// ---------------------------------------------------------------------------

/// Orchestrates spawned agents with concurrency and depth limits.
pub struct AgentManager {
    handles: Arc<AsyncMutex<HashMap<Uuid, AgentHandle>>>,
    max_concurrent: usize,
    max_depth: usize,
    current_depth: usize,
}

impl AgentManager {
    /// Create a new manager.
    ///
    /// - `max_concurrent` — maximum number of agents that can run simultaneously
    /// - `max_depth` — maximum nesting depth (agents spawning agents)
    pub fn new(max_concurrent: usize, max_depth: usize) -> Self {
        Self {
            handles: Arc::new(AsyncMutex::new(HashMap::new())),
            max_concurrent,
            max_depth,
            current_depth: 0,
        }
    }

    /// Set the current nesting depth. Agents spawned from this manager
    /// will have `depth + 1`.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.current_depth = depth;
        self
    }

    /// Return the number of currently active (running or awaiting) child agents.
    pub async fn active_count(&self) -> usize {
        let handles = self.handles.lock().await;
        handles.values().filter(|h| !h.is_finished()).count()
    }

    /// Spawn a new agent from the given builder.
    ///
    /// Returns the agent's UUID immediately — the agent runs asynchronously.
    pub async fn spawn(&self, builder: SpawnBuilder) -> anyhow::Result<Uuid> {
        let active = self.active_count().await;
        if active >= self.max_concurrent {
            anyhow::bail!(
                "maximum concurrent agents reached ({}/{})",
                active,
                self.max_concurrent
            );
        }

        if self.current_depth >= self.max_depth {
            anyhow::bail!(
                "maximum agent depth reached ({}/{})",
                self.current_depth,
                self.max_depth
            );
        }

        if let Some(ref name) = builder.name {
            let handles = self.handles.lock().await;
            let taken = handles
                .values()
                .any(|h| h.name.as_deref() == Some(name) && !h.is_finished());
            if taken {
                anyhow::bail!("agent name already in use: {name}");
            }
        }

        let id = Uuid::new_v4();
        let name = builder.name.clone();
        let task = builder.task.clone();
        let cancel_token = CancellationToken::new();
        let status = Arc::new(Mutex::new(AgentStatus::Running { turn: 0 }));

        let mut tools = builder.tools.unwrap_or_default();

        if !builder.strip_prefixes.is_empty() {
            let prefixes: Vec<&str> = builder.strip_prefixes.iter().map(|s| s.as_str()).collect();
            tools.strip_prefixes(&prefixes);
        }

        let mut agent = if let Some(llm) = builder.llm {
            Agent::builder(builder.provider)
                .tools(tools)
                .llm(llm)
                .build_sync()
        } else {
            Agent::builder(builder.provider).tools(tools).build_sync()
        };
        agent.refresh_features().await;

        let join_handle = tokio::spawn(run_agent(SpawnContext {
            agent,
            task: task.clone(),
            system_prompt: builder.system_prompt,
            max_iterations: builder.max_iterations,
            cancel_token: cancel_token.clone(),
            status: status.clone(),
        }));

        let handle = AgentHandle::new(id, name, task, cancel_token, join_handle, status);
        self.handles.lock().await.insert(id, handle);

        Ok(id)
    }

    /// Resolve an agent reference to its UUID.
    ///
    /// Tries, in order:
    /// 1. Exact name match among active agents
    /// 2. UUID hex prefix match (e.g. 8-char short ID)
    /// 3. Full UUID parse
    ///
    /// Returns an error if no agent matches or the string is invalid.
    pub async fn resolve(&self, agent: &str) -> anyhow::Result<Uuid> {
        let handles = self.handles.lock().await;

        // Try name match first.
        if let Some(h) = handles.values().find(|h| h.name.as_deref() == Some(agent)) {
            return Ok(h.id);
        }

        // Try UUID prefix match.
        if agent.len() <= 32 && agent.chars().all(|c| c.is_ascii_hexdigit()) {
            let matches: Vec<_> = handles
                .keys()
                .filter(|id| id.simple().to_string().starts_with(agent))
                .collect();

            match matches.len() {
                1 => return Ok(*matches[0]),
                n if n > 1 => {
                    anyhow::bail!("ambiguous agent ID prefix: {agent} matches {n} agents")
                }
                _ => {}
            }
        }

        // Try full UUID parse.
        if let Ok(id) = Uuid::parse_str(agent)
            && handles.contains_key(&id)
        {
            return Ok(id);
        }

        anyhow::bail!("agent not found: {agent}")
    }

    /// Get the status of an agent.
    pub async fn status(&self, id: Uuid) -> anyhow::Result<AgentStatus> {
        let handles = self.handles.lock().await;
        let handle = handles
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))?;
        Ok(handle.status())
    }

    /// Get the status of all agents.
    pub async fn all_statuses(&self) -> Vec<(Uuid, Option<String>, String, AgentStatus)> {
        let handles = self.handles.lock().await;
        handles
            .values()
            .map(|h| (h.id, h.name.clone(), h.task.clone(), h.status()))
            .collect()
    }

    /// Cancel a running agent.
    pub async fn terminate(&self, id: Uuid) -> anyhow::Result<()> {
        let handles = self.handles.lock().await;
        let handle = handles
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))?;
        handle.cancel();
        Ok(())
    }

    /// Wait for an agent to complete and return its final response.
    ///
    /// Removes the agent from the active set.
    pub async fn wait(&self, id: Uuid) -> anyhow::Result<String> {
        let mut handle = {
            let mut handles = self.handles.lock().await;
            handles
                .remove(&id)
                .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))?
        };

        handle.wait().await
    }

    /// Cancel all active agents.
    pub async fn cancel_all(&self) {
        let handles = self.handles.lock().await;
        for handle in handles.values() {
            handle.cancel();
        }
    }
}

// ---------------------------------------------------------------------------
// Agent task
// ---------------------------------------------------------------------------

struct SpawnContext {
    agent: Agent,
    task: String,
    system_prompt: Option<String>,
    max_iterations: Option<usize>,
    cancel_token: CancellationToken,
    status: Arc<Mutex<AgentStatus>>,
}

async fn run_agent(ctx: SpawnContext) -> anyhow::Result<String> {
    let SpawnContext {
        mut agent,
        task,
        system_prompt,
        max_iterations,
        cancel_token,
        status,
    } = ctx;
    let mut conversation = Conversation::new();

    let system = match system_prompt {
        Some(prompt) => format!("{prompt}\n\n# Task\n\n{task}"),
        None => format!(
            "You are a background agent running autonomously. Use your tools to complete \
             the task thoroughly, then respond with a concise summary of what you found or did.\n\
             \n\
             You cannot interact with the user — no questions, no confirmations, no clarifications. \
             Work with what you have. If something is ambiguous, make a reasonable choice and note \
             it in your response.\n\n# Task\n\n{task}"
        ),
    };
    conversation.set_system(&system);
    conversation.add(ConversationEntry::user("Proceed with the task."));

    let stream = agent.start(
        &mut conversation,
        cancel_token.clone(),
        AgentInput::Resume,
        max_iterations,
    );
    tokio::pin!(stream);

    let mut final_response = String::new();
    let mut turn_count = 0usize;

    loop {
        tokio::select! {
            event = stream.next() => {
                let Some(event) = event else { break };

                match &event {
                    AgentEvent::ToolResult { .. } => {
                        turn_count += 1;
                        *status.lock().unwrap() = AgentStatus::Running {
                            turn: turn_count,
                        };
                    }
                    AgentEvent::Done(text) => {
                        final_response = text.clone();
                        *status.lock().unwrap() = AgentStatus::Completed {
                            result: text.clone(),
                        };
                    }
                    AgentEvent::Error(err) => {
                        *status.lock().unwrap() = AgentStatus::Failed {
                            error: err.clone(),
                        };
                        anyhow::bail!("{err}");
                    }
                    _ => {}
                }
            }
            _ = cancel_token.cancelled() => {
                *status.lock().unwrap() = AgentStatus::Cancelled;
                break;
            }
        }
    }

    Ok(final_response)
}
