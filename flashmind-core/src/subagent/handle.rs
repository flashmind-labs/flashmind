//! Agent handles, retained transcripts, and status tracking.

use std::fmt;
use std::sync::{Arc, Mutex};

use flashmind_types::{AgentEvent, Model};
use tokio::sync::{Notify, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::ConversationEntry;

/// Current state of a spawned agent.
#[derive(Debug, Clone)]
pub enum AgentStatus {
    /// Agent is actively processing turns.
    Running { turn: usize },
    /// Agent finished successfully.
    Completed { result: String },
    /// Agent encountered an error.
    Failed { error: String },
    /// Agent was cancelled by the parent.
    Cancelled,
}

impl AgentStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

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

/// An entry in a subagent's display transcript.
#[derive(Debug, Clone)]
pub enum SubagentEvent {
    User(ConversationEntry),
    Agent(AgentEvent),
}

pub(crate) struct SessionState {
    pub events: Vec<SubagentEvent>,
    pub pending: usize,
    pub cancel: CancellationToken,
    pub model: Model,
    pub pending_model: Option<Model>,
}

/// A retained subagent conversation. Follow-ups use the same runtime and history.
pub struct AgentHandle {
    pub id: Uuid,
    pub name: Option<String>,
    pub task: String,
    pub(crate) sender: UnboundedSender<ConversationEntry>,
    pub(crate) state: Arc<Mutex<SessionState>>,
    pub(crate) status: Arc<Mutex<AgentStatus>>,
    pub(crate) changed: Arc<Notify>,
}

/// A finished response and its position in the retained transcript.
pub struct AgentCompletion {
    pub event_count: usize,
    pub status: AgentStatus,
}

impl AgentHandle {
    /// Return a new completion after `offset`, keeping status and transcript consistent.
    pub fn completion_since(&self, offset: usize) -> Option<AgentCompletion> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.events.len() <= offset {
            return None;
        }
        let status = self.status();
        status.is_finished().then(|| AgentCompletion {
            event_count: state.events.len(),
            status,
        })
    }

    /// Model used for the current response, or the most recent model request.
    pub fn model(&self) -> Model {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model
            .clone()
    }

    pub(crate) fn set_model(&self, model: Model) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if model.provider != state.model.provider {
            anyhow::bail!(
                "subagent model must use the current provider ({})",
                state.model.provider
            );
        }
        if self.sender.is_closed() {
            anyhow::bail!("agent conversation is closed");
        }
        state.pending_model = Some(model);
        Ok(())
    }

    pub fn status(&self) -> AgentStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn status_ref(&self) -> &Arc<Mutex<AgentStatus>> {
        &self.status
    }

    pub fn is_finished(&self) -> bool {
        self.status().is_finished()
    }

    /// Return transcript entries after the given offset without consuming them.
    pub fn events_since(&self, offset: usize) -> Vec<SubagentEvent> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .events
            .get(offset..)
            .unwrap_or_default()
            .to_vec()
    }

    pub(crate) fn send(&self, entry: ConversationEntry) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.sender
            .send(entry)
            .map_err(|_| anyhow::anyhow!("agent conversation is closed"))?;
        state.pending += 1;
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = AgentStatus::Running { turn: 0 };
        Ok(())
    }

    /// Cancel the current response. The conversation remains available.
    pub fn cancel(&self) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending > 0 {
            state.cancel.cancel();
        }
    }

    /// Wait for the current response and any queued follow-ups, without consuming it.
    pub async fn wait(&self) -> anyhow::Result<String> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            match self.status() {
                AgentStatus::Running { .. } => changed.await,
                AgentStatus::Completed { result } => return Ok(result),
                AgentStatus::Failed { error } => anyhow::bail!("{error}"),
                AgentStatus::Cancelled => anyhow::bail!("agent cancelled"),
            }
        }
    }
}

impl fmt::Debug for AgentHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentHandle")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("task", &self.task)
            .field("finished", &self.is_finished())
            .finish()
    }
}
