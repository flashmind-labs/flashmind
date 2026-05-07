//! Agent handle and status tracking.

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use flashmind_types::InjectQueue;

// ---------------------------------------------------------------------------
// AgentStatus
// ---------------------------------------------------------------------------

/// Current state of a spawned agent.
#[derive(Debug, Clone)]
pub enum AgentStatus {
    /// Agent is actively processing turns.
    Running {
        /// Number of turns completed so far.
        turn: usize,
    },
    /// Agent finished successfully.
    Completed {
        /// Final assistant response text.
        result: String,
    },
    /// Agent encountered an error.
    Failed {
        /// Error description.
        error: String,
    },
    /// Agent was cancelled by the parent.
    Cancelled,
}

impl AgentStatus {
    /// Whether the agent is still running.
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

    /// Whether the agent has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        !self.is_running()
    }
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Running { turn } => write!(f, "running (turn {turn})"),
            Self::Completed { .. } => write!(f, "completed"),
            Self::Failed { error } => write!(f, "failed: {error}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentHandle
// ---------------------------------------------------------------------------

/// Handle to a running agent. Provides control and status inspection.
pub struct AgentHandle {
    /// Unique identifier for this agent.
    pub id: Uuid,
    /// Human-readable task description.
    pub task: String,
    cancel_token: CancellationToken,
    join_handle: Option<JoinHandle<anyhow::Result<String>>>,
    inject_queue: Arc<InjectQueue>,
    status: Arc<Mutex<AgentStatus>>,
}

impl AgentHandle {
    pub(crate) fn new(
        id: Uuid,
        task: String,
        cancel_token: CancellationToken,
        join_handle: JoinHandle<anyhow::Result<String>>,
        inject_queue: Arc<InjectQueue>,
        status: Arc<Mutex<AgentStatus>>,
    ) -> Self {
        Self {
            id,
            task,
            cancel_token,
            join_handle: Some(join_handle),
            inject_queue,
            status,
        }
    }

    /// Current status of the agent.
    pub fn status(&self) -> AgentStatus {
        self.status.lock().unwrap().clone()
    }

    /// Shared status reference for external updates.
    pub fn status_ref(&self) -> &Arc<Mutex<AgentStatus>> {
        &self.status
    }

    /// Whether the underlying task has finished.
    pub fn is_finished(&self) -> bool {
        self.join_handle.as_ref().is_none_or(|h| h.is_finished())
    }

    /// Send a message to this agent. Processed on its next turn.
    pub fn send_message(&self, message: String) {
        self.inject_queue
            .push(flashmind_types::InjectEvent::UserMessage {
                text: message,
                parts: None,
            });
    }

    /// Cancel the agent. The task will stop at the next cancellation check.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
        *self.status.lock().unwrap() = AgentStatus::Cancelled;
    }

    /// Wait for the agent to complete and return its final response.
    ///
    /// Consumes the join handle — can only be called once.
    pub async fn wait(&mut self) -> anyhow::Result<String> {
        let handle = self
            .join_handle
            .take()
            .ok_or_else(|| anyhow::anyhow!("agent already awaited"))?;

        match handle.await {
            Ok(result) => result,
            Err(e) => anyhow::bail!("agent task panicked: {e}"),
        }
    }

    /// The inject queue for this agent (for direct injection).
    pub fn inject_queue(&self) -> &Arc<InjectQueue> {
        &self.inject_queue
    }
}

impl fmt::Debug for AgentHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentHandle")
            .field("id", &self.id)
            .field("task", &self.task)
            .field("finished", &self.is_finished())
            .finish()
    }
}
