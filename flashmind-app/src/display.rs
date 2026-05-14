//! Display log persistence for session restore.
//!
//! Stores display events as JSONL so resumed sessions can replay
//! the conversation through the TUI without re-streaming from the LLM.
//!
//! Uses `ServerMessage` as the wire format — a clean, serializable subset
//! of `AgentEvent` — matching the agent's session format.

use std::io::{self, BufRead, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use flashmind_types::AgentEvent;

// ---------------------------------------------------------------------------
// ServerMessage — serializable subset of AgentEvent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ServerMessage {
    TextDelta {
        content: String,
    },
    ReasoningDelta {
        content: String,
    },
    ToolStart {
        name: String,
        humanized: String,
    },
    ToolResult {
        success: bool,
        output: String,
    },
    Done {
        content: String,
    },
    Error {
        message: String,
    },
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    Status {
        status: String,
    },
    FileDiff {
        path: String,
        diff: Vec<flashmind_types::tool::DiffLine>,
    },
    Compacted {
        summary: String,
    },
    SubagentEvent {
        id: String,
        task: String,
        #[serde(default)]
        model: Option<String>,
        event: SubagentWireEvent,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SubagentWireEvent {
    ToolStart {
        name: String,
        humanized: String,
    },
    ToolResult {
        name: String,
        success: bool,
        elapsed_ms: u64,
        output: String,
    },
    TextDelta {
        content: String,
    },
    ReasoningDelta {
        content: String,
    },
    Done {
        content: String,
    },
    Error {
        message: String,
    },
}

// ---------------------------------------------------------------------------
// DisplayEvent — top-level log entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[serde(rename_all = "snake_case")]
pub enum DisplayEvent {
    /// An agent event (tool start, text delta, done, etc.).
    Server { msg: ServerMessage },
    /// A user prompt.
    User { text: String },
    /// A system/info message.
    System { text: String },
    /// Output was cleared (e.g. /clear command).
    Clear,
}

// ---------------------------------------------------------------------------
// Conversion: AgentEvent <-> ServerMessage
// ---------------------------------------------------------------------------

pub fn agent_event_to_server_msg(event: &AgentEvent) -> Option<ServerMessage> {
    match event {
        AgentEvent::TextDelta(content) => Some(ServerMessage::TextDelta {
            content: content.clone(),
        }),
        AgentEvent::ReasoningDelta(content) => Some(ServerMessage::ReasoningDelta {
            content: content.clone(),
        }),
        AgentEvent::ToolStart {
            name, humanized, ..
        } => Some(ServerMessage::ToolStart {
            name: name.clone(),
            humanized: humanized.clone(),
        }),
        AgentEvent::ToolResult {
            success, output, ..
        } => Some(ServerMessage::ToolResult {
            success: *success,
            output: output.clone(),
        }),
        AgentEvent::Done(content) => Some(ServerMessage::Done {
            content: content.clone(),
        }),
        AgentEvent::Error(message) => Some(ServerMessage::Error {
            message: message.clone(),
        }),
        AgentEvent::Usage(usage) => Some(ServerMessage::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }),
        AgentEvent::Status(status) => Some(ServerMessage::Status {
            status: status.clone(),
        }),
        AgentEvent::FileDiff { path, diff } => Some(ServerMessage::FileDiff {
            path: path.clone(),
            diff: diff.clone(),
        }),
        AgentEvent::Compacted(summary) => Some(ServerMessage::Compacted {
            summary: summary.clone(),
        }),
        AgentEvent::SpawnedEvent {
            id,
            task,
            model,
            event: inner,
        } => {
            let wire_event = match inner.as_ref() {
                AgentEvent::ToolStart {
                    name, humanized, ..
                } => SubagentWireEvent::ToolStart {
                    name: name.clone(),
                    humanized: humanized.clone(),
                },
                AgentEvent::ToolResult {
                    name,
                    success,
                    elapsed_ms,
                    output,
                    ..
                } => SubagentWireEvent::ToolResult {
                    name: name.clone(),
                    success: *success,
                    elapsed_ms: *elapsed_ms,
                    output: output.clone(),
                },
                AgentEvent::TextDelta(content) => SubagentWireEvent::TextDelta {
                    content: content.clone(),
                },
                AgentEvent::ReasoningDelta(content) => SubagentWireEvent::ReasoningDelta {
                    content: content.clone(),
                },
                AgentEvent::Done(content) => SubagentWireEvent::Done {
                    content: content.clone(),
                },
                AgentEvent::Error(message) => SubagentWireEvent::Error {
                    message: message.clone(),
                },
                _ => return None,
            };
            Some(ServerMessage::SubagentEvent {
                id: id.to_string(),
                task: task.clone(),
                model: model.as_ref().map(|m| m.to_string()),
                event: wire_event,
            })
        }
        _ => None,
    }
}

pub fn server_msg_to_event(msg: ServerMessage) -> Option<AgentEvent> {
    match msg {
        ServerMessage::TextDelta { content } => Some(AgentEvent::TextDelta(content)),
        ServerMessage::ReasoningDelta { content } => Some(AgentEvent::ReasoningDelta(content)),
        ServerMessage::ToolStart { name, humanized } => Some(AgentEvent::ToolStart {
            name,
            id: String::new(),
            humanized,
        }),
        ServerMessage::ToolResult { success, output } => Some(AgentEvent::ToolResult {
            name: String::new(),
            id: String::new(),
            output,
            success,
            elapsed_ms: 0,
            sources: Vec::new(),
        }),
        ServerMessage::Done { content } => Some(AgentEvent::Done(content)),
        ServerMessage::Error { message } => Some(AgentEvent::Error(message)),
        ServerMessage::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        } => Some(AgentEvent::Usage(flashmind_types::TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })),
        ServerMessage::Status { status } => Some(AgentEvent::Status(status)),
        ServerMessage::FileDiff { path, diff } => Some(AgentEvent::FileDiff { path, diff }),
        ServerMessage::Compacted { summary } => Some(AgentEvent::Compacted(summary)),
        ServerMessage::SubagentEvent {
            id,
            task,
            model,
            event: inner,
        } => {
            let inner_event = match inner {
                SubagentWireEvent::ToolStart { name, humanized } => AgentEvent::ToolStart {
                    name,
                    id: String::new(),
                    humanized,
                },
                SubagentWireEvent::ToolResult {
                    name,
                    success,
                    elapsed_ms,
                    output,
                } => AgentEvent::ToolResult {
                    name,
                    id: String::new(),
                    output,
                    success,
                    elapsed_ms,
                    sources: Vec::new(),
                },
                SubagentWireEvent::TextDelta { content } => AgentEvent::TextDelta(content),
                SubagentWireEvent::ReasoningDelta { content } => {
                    AgentEvent::ReasoningDelta(content)
                }
                SubagentWireEvent::Done { content } => AgentEvent::Done(content),
                SubagentWireEvent::Error { message } => AgentEvent::Error(message),
            };
            Some(AgentEvent::SpawnedEvent {
                id,
                task,
                model: model.and_then(|m| m.parse().ok()),
                event: Box::new(inner_event),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Display log
// ---------------------------------------------------------------------------

/// Accumulates display events for persistence.
pub struct DisplayLog {
    events: Vec<DisplayEvent>,
}

impl DisplayLog {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn log_agent_event(&mut self, event: &AgentEvent) {
        if let Some(msg) = agent_event_to_server_msg(event) {
            if matches!(msg, ServerMessage::Compacted { .. }) {
                self.events.clear();
            }
            self.events.push(DisplayEvent::Server { msg });
        }
    }

    pub fn log_user(&mut self, text: String) {
        self.events.push(DisplayEvent::User { text });
    }

    pub fn log_system(&mut self, text: String) {
        self.events.push(DisplayEvent::System { text });
    }

    pub fn log_clear(&mut self) {
        self.events.clear();
        self.events.push(DisplayEvent::Clear);
    }

    pub fn extend(&mut self, events: Vec<DisplayEvent>) {
        self.events.extend(events);
    }

    pub fn events(&self) -> &[DisplayEvent] {
        &self.events
    }
}

impl Default for DisplayLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Persistence (JSONL)
// ---------------------------------------------------------------------------

pub fn save(path: &Path, events: &[DisplayEvent]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for event in events {
        let line = serde_json::to_string(event).unwrap_or_default();
        writeln!(f, "{line}")?;
    }
    Ok(())
}

pub fn load(path: &Path) -> Vec<DisplayEvent> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}
