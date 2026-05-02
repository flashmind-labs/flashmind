//! Telemetry and control events for the agent turn loop.
//!
//! This module defines the event types, input structures, and injection mechanisms
//! that flow through the agent runtime during each turn.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`AgentEvent`] | Stream of events emitted by the agent loop (text deltas, tool results, status…) |
//! | [`AgentInput`] | Input that starts or resumes an agent session |
//! | [`TurnStatus`] | Result of one iteration: `Done`, `ToolCalls`, `Continue`, or `Interrupted` |
//! | [`TurnUsage`] | Token usage reported at the end of an LLM call |
//! | [`InjectEvent`] | Messages injected into a running turn from outside the loop |
//! | [`InjectQueue`] | Shared queue for mid-turn interjects (subagents, reminders, user commands) |
//! | [`Source`] | Web source attached to a tool result (URL + optional title) |
//!
//! # Event flow
//!
//! 1. Agent emits [`AgentEvent::Started`] with cancellation token and inject queue
//! 2. Text arrives incrementally as [`AgentEvent::TextDelta`] (and [`AgentEvent::ReasoningDelta`])
//! 3. Tool execution produces [`AgentEvent::ToolStart`] → [`AgentEvent::ToolResult`]
//! 4. File modifications produce [`AgentEvent::FileDiff`]
//! 5. The turn ends with [`AgentEvent::Done`] or [`AgentEvent::Error`]
//!
//! Listeners (REPL, Telegram, Slack, etc.) receive the event stream and render
//! incrementally. Unknown variants should be silently ignored for forward compatibility.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::llm::TokenUsage;
use crate::message::ContentPart;
use crate::model::{AgentLlmConfig, Model};

/// Messages that can be injected into a running agent turn from outside the loop.
///
/// Used by subagents, cron reminders, and interactive controls to append
/// user messages or report progress without restarting the turn.
#[derive(Debug)]
pub enum InjectEvent {
    /// Append an extra user message mid-turn (e.g. from a slash command).
    UserMessage {
        text: String,
        parts: Option<Vec<ContentPart>>,
    },
    /// Forward a live progress update from a running subagent.
    SubagentProgress {
        id: String,
        turn: usize,
        content: String,
    },
    /// Report that a delegated subagent finished with an error.
    SubagentError { id: String, error: String },
}

/// Shared queue for injecting messages into a running agent.
///
/// Replaces the previous `mpsc::channel<InjectEvent>` pattern. The queue is
/// visible to both producer (UI/subagents) and consumer (agent loop), which
/// enables:
/// - Cancelling a queued message before the agent consumes it
/// - Displaying pending messages in the TUI
/// - Peeking at queue state without consuming
pub struct InjectQueue {
    queue: Mutex<VecDeque<(u64, InjectEvent)>>,
    next_id: AtomicU64,
    notify: Notify,
}

impl InjectQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(0),
            notify: Notify::new(),
        })
    }

    /// Push an event into the queue, returning an ID that can be used to cancel it.
    pub fn push(&self, event: InjectEvent) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.queue.lock().unwrap().push_back((id, event));
        self.notify.notify_one();
        id
    }

    /// Remove a queued event by ID before the agent consumes it.
    /// Returns `None` if already consumed or not found.
    pub fn cancel(&self, id: u64) -> Option<InjectEvent> {
        let mut q = self.queue.lock().unwrap();
        if let Some(pos) = q.iter().position(|(eid, _)| *eid == id) {
            q.remove(pos).map(|(_, ev)| ev)
        } else {
            None
        }
    }

    /// Drain all pending events. Called by the agent between turns.
    pub fn drain(&self) -> Vec<InjectEvent> {
        self.queue
            .lock()
            .unwrap()
            .drain(..)
            .map(|(_, ev)| ev)
            .collect()
    }

    /// Check whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Snapshot of pending user message texts (for TUI display).
    pub fn pending_user_messages(&self) -> Vec<(u64, String)> {
        self.queue
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(id, ev)| match ev {
                InjectEvent::UserMessage { text, .. } => Some((*id, text.clone())),
                _ => None,
            })
            .collect()
    }

    /// Wait until something is pushed. Use in `tokio::select!` to replace `recv()`.
    pub async fn notified(&self) {
        self.notify.notified().await;
    }
}

impl Default for InjectQueue {
    fn default() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(0),
            notify: Notify::new(),
        }
    }
}

impl std::fmt::Debug for InjectQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.queue.lock().map(|q| q.len()).unwrap_or(0);
        f.debug_struct("InjectQueue").field("len", &len).finish()
    }
}

/// Input that starts or resumes an agent session.
#[derive(Debug, Clone, Default)]
pub enum AgentInput {
    /// A new user message to process.
    User {
        content: String,
        /// Additional system-level context injected before the user message.
        context: Option<String>,
        /// Multimodal attachments.
        parts: Option<Vec<ContentPart>>,
    },
    /// Resume a paused conversation without injecting new content.
    #[default]
    Resume,
}

impl AgentInput {
    /// Short-hand to construct a simple text-only [`AgentInput::User`].
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            context: None,
            parts: None,
        }
    }
}

/// Token usage reported at the end of an LLM call.
#[derive(Debug, Clone, Copy, Default)]
pub struct TurnUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Result of a single agent turn after the LLM has responded.
#[derive(Debug)]
pub enum TurnStatus {
    /// No tool calls but the turn should continue (e.g. empty response retry, compaction).
    Continue { content: String, usage: TurnUsage },
    /// No tool calls and no more content expected — this session turn is done.
    Done { content: String, usage: TurnUsage },
    /// The LLM returned tool calls — caller is responsible for executing them.
    ToolCalls {
        content: String,
        tool_calls: Vec<crate::ToolCall>,
        usage: TurnUsage,
    },
    /// A tool requested interactive user input. The agent loop has stopped;
    /// the caller should collect the response, add a `ConversationEntry::tool`
    /// with the result, and resume with `AgentInput::Resume`.
    Interrupted {
        tool_call_id: String,
        tool_name: String,
        /// Proposal data as JSON — the caller interprets based on `tool_name`.
        output: String,
        content: String,
        usage: TurnUsage,
    },
}

impl TurnStatus {
    /// Extract content and usage from any variant.
    pub fn content_and_usage(&self) -> (&str, &TurnUsage) {
        match self {
            Self::Continue { content, usage }
            | Self::Done { content, usage }
            | Self::ToolCalls { content, usage, .. }
            | Self::Interrupted { content, usage, .. } => (content, usage),
        }
    }
}

/// Alias for the final result of a turn.
pub type TurnResult = anyhow::Result<TurnStatus>;

/// A web source attached to a tool result (e.g. search result link).
#[derive(Debug, Clone)]
pub struct Source {
    pub url: String,
    pub title: Option<String>,
}

/// Events emitted by the agent turn loop as a `Stream<Item = AgentEvent>`.
///
/// Listeners (REPL, Slack, Telegram, etc.) receive these and render them
/// incrementally. Consumers should handle all variants; unknown variants
/// should be silently ignored so forward-compatibility is preserved.
#[derive(Debug)]
pub enum AgentEvent {
    /// Incremental reasoning token (for models that surface internal thinking).
    ReasoningDelta(String),
    /// Incremental text token from the assistant message.
    TextDelta(String),
    /// A tool call has started execution.
    ToolStart {
        name: String,
        id: String,
        /// Human-readable summary of the arguments (for display in TUI/cards).
        humanized: String,
    },
    /// A tool has finished. The `output` may be truncated for display.
    ToolResult {
        name: String,
        id: String,
        output: String,
        success: bool,
        elapsed_ms: u64,
        sources: Vec<Source>,
    },
    /// Incremental audio output chunk (base64-encoded) for real-time playback.
    AudioChunk {
        data: String,
        format: String,
    },
    /// Informational status message (compaction progress, retry attempts, etc.).
    Status(String),
    /// Conversation was compacted; payload is the summary text.
    Compacted(String),
    /// A file was modified by a tool (displayed as a unified diff).
    FileDiff { path: String, diff: String },
    /// Token usage telemetry from the provider.
    Usage(TokenUsage),
    /// Final terminal event — processing complete with `String` as the full response.
    Done(String),
    /// Terminal error event.
    Error(String),
    /// A subagent produced an event while running in parallel.
    SubagentEvent {
        id: String,
        task: String,
        model: Option<Model>,
        profile: Option<String>,
        role: Option<String>,
        event: Box<AgentEvent>,
    },
    /// Initial event emitted once when the loop starts.
    ///
    /// Contains the cancellation token (for external abort), the shared inject
    /// queue (for mid-turn interjects and cancellation), and current config snapshot.
    Started {
        cancel_token: CancellationToken,
        inject_queue: Arc<InjectQueue>,
        sampling: AgentLlmConfig,
        profile: Option<String>,
        role: Option<String>,
    },
    /// A tool requested interactive input. Emitted when a turn ends with
    /// `TurnStatus::Interrupted`. The caller should show the appropriate
    /// picker, inject the tool result, and resume.
    Interrupted {
        tool_call_id: String,
        tool_name: String,
        output: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_input_user_shorthand() {
        let input = AgentInput::user("hello");
        match input {
            AgentInput::User {
                content,
                context,
                parts,
            } => {
                assert_eq!(content, "hello");
                assert!(context.is_none());
                assert!(parts.is_none());
            }
            _ => panic!("expected User variant"),
        }
    }

    #[test]
    fn agent_input_default_is_resume() {
        let input = AgentInput::default();
        assert!(matches!(input, AgentInput::Resume));
    }

    #[test]
    fn turn_status_content_and_usage_done() {
        let status = TurnStatus::Done {
            content: "response".into(),
            usage: TurnUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            },
        };
        let (content, usage) = status.content_and_usage();
        assert_eq!(content, "response");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
    }

    #[test]
    fn turn_status_content_and_usage_tool_calls() {
        let status = TurnStatus::ToolCalls {
            content: "thinking".into(),
            tool_calls: vec![],
            usage: TurnUsage {
                prompt_tokens: 5,
                completion_tokens: 15,
            },
        };
        let (content, usage) = status.content_and_usage();
        assert_eq!(content, "thinking");
        assert_eq!(usage.completion_tokens, 15);
    }

    #[test]
    fn turn_usage_default_is_zero() {
        let usage = TurnUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
    }
}
