//! Agent manager — orchestrates spawning, lifecycle, and control of child agents.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use flashmind_types::{AgentEvent, AgentInput, InjectEvent, InjectQueue};

use crate::agent::Agent;
use crate::conversation::{Conversation, ConversationEntry};

use super::builder::SpawnBuilder;
use super::handle::{AgentHandle, AgentStatus};

// ---------------------------------------------------------------------------
// AgentManager
// ---------------------------------------------------------------------------

/// Orchestrates spawned agents with concurrency and depth limits.
///
/// Each manager instance tracks agents spawned by a single parent agent.
/// The parent's [`InjectQueue`] receives progress updates from children.
pub struct AgentManager {
    handles: Arc<AsyncMutex<HashMap<Uuid, AgentHandle>>>,
    parent_queue: Arc<InjectQueue>,
    max_concurrent: usize,
    max_depth: usize,
    current_depth: usize,
    progress_interval: usize,
}

impl AgentManager {
    /// Create a new manager.
    ///
    /// - `parent_queue` — the parent agent's inject queue for receiving progress
    /// - `max_concurrent` — maximum number of agents that can run simultaneously
    /// - `max_depth` — maximum nesting depth (agents spawning agents)
    pub fn new(parent_queue: Arc<InjectQueue>, max_concurrent: usize, max_depth: usize) -> Self {
        Self {
            handles: Arc::new(AsyncMutex::new(HashMap::new())),
            parent_queue,
            max_concurrent,
            max_depth,
            current_depth: 0,
            progress_interval: 3,
        }
    }

    /// Set the current nesting depth. Agents spawned from this manager
    /// will have `depth + 1`.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.current_depth = depth;
        self
    }

    /// Set how often (in turns) progress is forwarded to the parent.
    pub fn with_progress_interval(mut self, interval: usize) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Number of currently active (non-finished) agents.
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

        let child_id = name
            .clone()
            .unwrap_or_else(|| id.simple().to_string()[..8].to_string());

        let mut tools = builder.tools.unwrap_or_default();

        if !builder.strip_prefixes.is_empty() {
            let prefixes: Vec<&str> = builder.strip_prefixes.iter().map(|s| s.as_str()).collect();
            tools.strip_prefixes(&prefixes);
        }

        let agent = if let Some(llm) = builder.llm {
            Agent::new(builder.provider, tools, llm)
        } else {
            Agent::builder(builder.provider).tools(tools).build()
        };

        let join_handle = tokio::spawn(run_agent(SpawnContext {
            agent,
            task: task.clone(),
            system_prompt: builder.system_prompt,
            max_iterations: builder.max_iterations,
            cancel_token: cancel_token.clone(),
            status: status.clone(),
            parent_queue: self.parent_queue.clone(),
            child_id,
            progress_interval: self.progress_interval,
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
    parent_queue: Arc<InjectQueue>,
    child_id: String,
    progress_interval: usize,
}

async fn run_agent(ctx: SpawnContext) -> anyhow::Result<String> {
    let SpawnContext {
        mut agent,
        task,
        system_prompt,
        max_iterations,
        cancel_token,
        status,
        parent_queue,
        child_id,
        progress_interval,
    } = ctx;
    let mut conversation = Conversation::new();

    if let Some(prompt) = system_prompt {
        conversation.prepend(ConversationEntry::system(&prompt));
    }

    let stream = agent.start(&mut conversation, AgentInput::user(&task), max_iterations);
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

                        if progress_interval > 0 && turn_count.is_multiple_of(progress_interval) {
                            parent_queue.push(InjectEvent::AgentProgress {
                                id: child_id.clone(),
                                turn: turn_count,
                                content: format!(
                                    "Agent '{task}' completed {turn_count} tool calls"
                                ),
                            });
                        }
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
                        parent_queue.push(InjectEvent::AgentError {
                            id: child_id.clone(),
                            error: err.clone(),
                        });
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
