//! Background commands over ACP, in the JetBrains AIR `asyncTasks`
//! extension that the Claude and Codex adapters speak: the client gets
//! `async_task_*` session updates, and can stop a task or move a running
//! command to the background with `_session/async_task/*` requests.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    ClientCapabilities, SessionId, SessionUpdate, ToolCallUpdate,
};
use agent_client_protocol::{
    Client, ConnectionTo, Error, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
use flashmind_tools::background::{JobEnd, JobEvent};
use flashmind_types::tool::BackgroundRequest;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedReceiver;
use uuid::Uuid;

/// The capability a client declares in `_meta.jetbrains.air.capabilities`.
const CAPABILITY: &str = "asyncTasks";

/// The longest task name, in characters.
const MAX_NAME: usize = 80;

/// Whether the client declared the `asyncTasks` capability: version 1 or
/// later of `_meta.jetbrains.air`, with the name in its capability list.
pub(super) fn client_supports(capabilities: &ClientCapabilities) -> bool {
    let Some(air) = capabilities
        .meta
        .as_ref()
        .and_then(|meta| meta.get("jetbrains"))
        .and_then(|jetbrains| jetbrains.get("air"))
    else {
        return false;
    };
    let versioned = air
        .get("version")
        .and_then(Value::as_u64)
        .is_some_and(|version| version >= 1);
    let listed = air
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|names| names.iter().any(|name| name == CAPABILITY));
    versioned && listed
}

/// The agent's side of the declaration, for the `initialize` response.
pub(super) fn agent_meta() -> Map<String, Value> {
    air_meta("capabilities", json!([CAPABILITY]))
}

/// `_meta.jetbrains.air` with the extension version and one payload.
fn air_meta(key: &str, payload: Value) -> Map<String, Value> {
    let mut air = Map::new();
    air.insert("version".to_owned(), json!(1));
    air.insert(key.to_owned(), payload);
    let mut jetbrains = Map::new();
    jetbrains.insert("air".to_owned(), Value::Object(air));
    let mut meta = Map::new();
    meta.insert("jetbrains".to_owned(), Value::Object(jetbrains));
    meta
}

/// The running foreground commands of a session that can move to the
/// background, by tool call id. The turn fills it; the handler for
/// `_session/async_task/background` empties it.
pub(super) type Foreground = Arc<Mutex<HashMap<String, BackgroundRequest>>>;

/// Moves the foreground commands named by `tool_call_id` (all of them
/// without one) to the background. True when at least one moved.
pub(super) fn background(foreground: &Foreground, tool_call_id: Option<&str>) -> bool {
    let mut running = foreground
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let ids: Vec<String> = running
        .keys()
        .filter(|id| tool_call_id.is_none_or(|wanted| wanted == id.as_str()))
        .cloned()
        .collect();
    for id in &ids {
        if let Some(request) = running.remove(id) {
            request.request();
        }
    }
    !ids.is_empty()
}

/// `_session/async_task/stop`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/async_task/stop", response = StopResponse)]
#[serde(rename_all = "camelCase")]
pub(super) struct StopRequest {
    pub session_id: SessionId,
    pub async_task_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
pub(super) struct StopResponse {
    pub stopped: bool,
}

/// `_session/async_task/background`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_session/async_task/background", response = BackgroundResponse)]
#[serde(rename_all = "camelCase")]
pub(super) struct BackgroundTaskRequest {
    pub session_id: SessionId,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
pub(super) struct BackgroundResponse {
    pub backgrounded: bool,
}

/// An `async_task_*` update. The `SessionUpdate` of the ACP crate has no
/// such variant, so it goes out as a `session/update` of its own.
#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcNotification)]
#[notification(method = "session/update")]
#[serde(rename_all = "camelCase")]
pub(super) struct TaskNotification {
    pub session_id: SessionId,
    pub update: TaskUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
#[serde(rename_all_fields = "camelCase")]
pub(super) enum TaskUpdate {
    AsyncTaskSpawned {
        async_task_id: String,
        name: String,
        task_type: String,
        description: String,
        show_in_transcript: bool,
        can_stop: bool,
        tool_call_id: String,
    },
    AsyncTaskStateUpdate {
        async_task_id: String,
        state: TaskState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TaskState {
    Completed,
    Failed,
    Stopped,
}

impl From<JobEnd> for TaskState {
    fn from(end: JobEnd) -> Self {
        match end {
            JobEnd::Completed => Self::Completed,
            JobEnd::Failed => Self::Failed,
            JobEnd::Stopped => Self::Stopped,
        }
    }
}

/// A session's background tasks as the client sees them.
pub(super) struct Tasks {
    /// The job events; none when the client did not ask for tasks.
    events: Option<UnboundedReceiver<JobEvent>>,
    /// Tool calls whose command went to the background.
    backgrounded: HashSet<String>,
}

impl Tasks {
    pub(super) fn new(events: Option<UnboundedReceiver<JobEvent>>) -> Self {
        Self {
            events,
            backgrounded: HashSet::new(),
        }
    }

    /// The next job event; never ready when there is no subscription. It
    /// is safe to drop before it is ready.
    pub(super) async fn next(&mut self) -> Option<JobEvent> {
        match &mut self.events {
            Some(events) => events.recv().await,
            None => std::future::pending().await,
        }
    }

    /// Sends every event already queued.
    pub(super) fn drain(
        &mut self,
        cx: &ConnectionTo<Client>,
        session: &SessionId,
    ) -> Result<(), Error> {
        while let Some(event) = self.events.as_mut().and_then(|e| e.try_recv().ok()) {
            self.publish(cx, session, event)?;
        }
        Ok(())
    }

    /// Reports one job event to the client.
    pub(super) fn publish(
        &mut self,
        cx: &ConnectionTo<Client>,
        session: &SessionId,
        event: JobEvent,
    ) -> Result<(), Error> {
        let update = match event {
            JobEvent::Started {
                id,
                tool_call_id,
                command,
            } => {
                self.backgrounded.insert(tool_call_id.clone());
                TaskUpdate::AsyncTaskSpawned {
                    async_task_id: id.to_string(),
                    name: name(&command),
                    task_type: "shell".to_owned(),
                    description: command,
                    show_in_transcript: true,
                    can_stop: true,
                    tool_call_id,
                }
            }
            JobEvent::Ended { id, end, summary } => TaskUpdate::AsyncTaskStateUpdate {
                async_task_id: id.to_string(),
                state: end.into(),
                summary,
            },
        };
        cx.send_notification(TaskNotification {
            session_id: session.clone(),
            update,
        })
    }

    /// Marks the update of a tool call whose command went to the background,
    /// so the client shows it as work still running.
    pub(super) fn mark(&self, update: &mut SessionUpdate) {
        if let SessionUpdate::ToolCallUpdate(ToolCallUpdate {
            tool_call_id, meta, ..
        }) = update
            && self.backgrounded.contains(tool_call_id.0.as_ref())
        {
            let marker = air_meta(CAPABILITY, json!({ "backgrounded": true }));
            *meta = Some(marker);
        }
    }
}

/// The task id a client sent, when it names a job.
pub(super) fn job_id(task: &str) -> Option<Uuid> {
    task.trim().parse().ok()
}

/// A short name for a command: its first line, cut to a readable length.
fn name(command: &str) -> String {
    let line = command.lines().next().unwrap_or_default().trim();
    line.chars().take(MAX_NAME).collect()
}
