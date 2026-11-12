//! Ordered conversation history with a semantic intermediate representation.
//!
//! [`Conversation`] stores [`ConversationEntry`] values — a richer type than
//! the LLM wire-format [`Message`]. Each entry carries an [`EntryKind`] tag
//! that distinguishes system prompts, reminders, injected memories, summaries,
//! and the usual user/assistant/tool turns.
//!
//! The conversion to LLM wire format happens via [`Conversation::to_messages`],
//! which maps each entry kind to the appropriate role and content shape.
//!
//! # Entry lifecycle
//!
//! 1. **System prompt** is prepended at session start
//! 2. **User/Assistant turns** are appended as exchanges flow
//! 3. **Tool results** are inserted after tool calls resolve
//! 4. **Memories** can be injected from RAG lookups
//! 5. **Compaction** replaces earlier entries with a summary when context pressure hits

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::blobs::without_blobs;
use flashmind_types::{
    AgentLlmConfig, CompletionRequest, ContentPart, InteractionMode, LlmProvider, Message, Model,
    NativeReasoningFormat, ReasoningDetails, ReasoningLevel, Role, SamplingParams, StreamEvent,
    TokenUsage, ToolCall, ToolDefinition,
};

use crate::agent::DEFAULT_KEEP_RECENT_TOKENS;
use crate::compaction::{COMPACTION_PROMPT, COMPACTION_PROMPT_ITERATIVE, NATIVE_COMPACTION_PROMPT};

const NATIVE_COMPACTION_TAG: &str = "native_compaction";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeCheckpoint {
    model: Model,
    source_digest: String,
    context: ReasoningDetails,
}

#[derive(Debug, Clone)]
struct CompactionContext {
    llm: AgentLlmConfig,
    tools: Vec<ToolDefinition>,
}

/// Incrementally fingerprints API-relevant input, excluding display timestamps.
#[derive(Clone)]
pub(crate) struct RequestPrefix(Sha256);

impl RequestPrefix {
    fn new(tools: &[ToolDefinition]) -> Option<Self> {
        let mut prefix = Self(Sha256::new());
        prefix.update(&tools)?;
        Some(prefix)
    }

    fn update(&mut self, value: &impl Serialize) -> Option<()> {
        let bytes = serde_json::to_vec(value).ok()?;
        self.0
            .update(u64::try_from(bytes.len()).ok()?.to_le_bytes());
        self.0.update(bytes);
        Some(())
    }

    pub(crate) fn push(&mut self, message: &Message) -> Option<()> {
        self.update(&(
            &message.role,
            &message.content,
            &message.reasoning,
            &message.reasoning_details,
            &message.tool_calls,
            &message.tool_call_id,
            &message.parts,
        ))
    }

    pub(crate) fn digest(&self) -> String {
        self.0
            .clone()
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    pub(crate) fn from_request(request: &CompletionRequest) -> Option<Self> {
        let mut prefix = Self::new(&request.tools)?;
        for message in &request.messages {
            prefix.push(message)?;
        }
        Some(prefix)
    }
}

// ---------------------------------------------------------------------------
// CompactionResult
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CompactionResult {
    pub summary: String,
    pub usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// EntryKind
// ---------------------------------------------------------------------------

/// Semantic tag for a conversation entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EntryKind {
    /// Primary system prompt (always position 0, mapped to `Role::System`).
    SystemPrompt(String),
    /// Developer-role message. Covers system messages, reminders, memories,
    /// agent progress, and compaction summaries. Differentiated by `tag`.
    Developer {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// User turn, optionally with multimodal parts.
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parts: Option<Vec<ContentPart>>,
    },
    /// Assistant turn, optionally requesting tool calls.
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(default, skip_serializing_if = "ReasoningDetails::is_empty")]
        reasoning_details: ReasoningDetails,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_model: Option<Model>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_prefix: Option<String>,
    },
    /// Result returned from a tool execution.
    Tool {
        call_id: String,
        output: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        parts: Vec<ContentPart>,
    },
}

// ---------------------------------------------------------------------------
// ConversationEntry
// ---------------------------------------------------------------------------

/// A single entry in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    pub kind: EntryKind,
    pub timestamp: DateTime<Utc>,
}

impl ConversationEntry {
    // -- Convenience constructors ------------------------------------------

    /// Create a system prompt entry (`Role::System` in wire format).
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::SystemPrompt(content.into()),
            timestamp: Utc::now(),
        }
    }

    /// Create a developer/system-message entry. Maps to `Role::Developer` in wire format.
    pub fn system_message(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: None,
                metadata: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a reminder entry that prepends context to the next user turn.
    pub fn reminder(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: Some("reminder".into()),
                metadata: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Create the reminder that asks the model to carry on after compaction
    /// summarized away the request it was working on. Keeps the request in
    /// metadata so a second compaction in the same turn can repeat it.
    fn continuation(request: Option<String>) -> Self {
        let mut content = CONTINUATION_NOTE.to_string();
        if let Some(request) = &request {
            content.push_str("\n\nThe user's request, verbatim:\n\n");
            content.push_str(request);
        }
        Self {
            kind: EntryKind::Developer {
                content,
                tag: Some("reminder".into()),
                metadata: request.map(|request| serde_json::json!({ "continues": request })),
            },
            timestamp: Utc::now(),
        }
    }

    /// The request a continuation note repeats, if this entry is one.
    fn continued_request(&self) -> Option<&str> {
        match &self.kind {
            EntryKind::Developer {
                tag: Some(tag),
                metadata: Some(metadata),
                ..
            } if tag == "reminder" => metadata["continues"].as_str(),
            _ => None,
        }
    }

    /// Create a plain-text user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::User {
                content: content.into(),
                parts: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a user message with multimodal attachments (images, documents, etc.).
    pub fn user_with_parts(content: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            kind: EntryKind::User {
                content: content.into(),
                parts: if parts.is_empty() { None } else { Some(parts) },
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a plain assistant response.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Assistant {
                content: content.into(),
                tool_calls: None,
                reasoning: None,
                reasoning_details: ReasoningDetails::default(),
                reasoning_model: None,
                reasoning_prefix: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant response with optional reasoning trace.
    pub fn assistant_with_reasoning(content: impl Into<String>, reasoning: Option<String>) -> Self {
        Self {
            kind: EntryKind::Assistant {
                content: content.into(),
                tool_calls: None,
                reasoning: reasoning.filter(|r| !r.is_empty()),
                reasoning_details: ReasoningDetails::default(),
                reasoning_model: None,
                reasoning_prefix: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant response that includes tool call requests.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        reasoning: Option<String>,
    ) -> Self {
        Self {
            kind: EntryKind::Assistant {
                content: content.into(),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                reasoning: reasoning.filter(|r| !r.is_empty()),
                reasoning_details: ReasoningDetails::default(),
                reasoning_model: None,
                reasoning_prefix: None,
            },
            timestamp: Utc::now(),
        }
    }

    /// Attach complete reasoning state and its source model for safe replay.
    pub fn with_reasoning_details(mut self, details: ReasoningDetails, model: Model) -> Self {
        if let EntryKind::Assistant {
            reasoning_details,
            reasoning_model,
            ..
        } = &mut self.kind
        {
            *reasoning_details = details;
            *reasoning_model = Some(model);
        }
        self
    }

    pub(crate) fn with_reasoning_prefix(mut self, prefix: Option<String>) -> Self {
        if let EntryKind::Assistant {
            reasoning_prefix, ..
        } = &mut self.kind
        {
            *reasoning_prefix = prefix;
        }
        self
    }

    /// Estimated request bytes, including tool arguments and reasoning state.
    pub(crate) fn request_bytes(&self) -> usize {
        let mut bytes = self.content().len();
        if let EntryKind::Assistant {
            reasoning,
            reasoning_details,
            tool_calls,
            ..
        } = &self.kind
        {
            bytes = bytes.saturating_add(if reasoning_details.is_empty() {
                reasoning.as_ref().map_or(0, String::len)
            } else {
                reasoning_details.estimated_bytes()
            });
            for call in tool_calls.iter().flatten() {
                bytes = bytes
                    .saturating_add(call.arguments.to_string().len())
                    .saturating_add(call.id.len())
                    .saturating_add(call.name.len());
            }
        }
        bytes
    }

    fn summary_content(&self) -> String {
        let mut content = self.content().to_owned();
        if let EntryKind::Assistant {
            tool_calls: Some(calls),
            ..
        } = &self.kind
        {
            let names: Vec<&str> = calls.iter().map(|call| call.name.as_str()).collect();
            content.push_str(&format!("\n[tool calls: {}]", names.join(", ")));
            for call in calls {
                content.push_str(&format!(
                    "\n{} arguments: {}",
                    call.name,
                    without_blobs(&call.arguments.to_string())
                ));
            }
        }
        content
    }

    /// Create a tool execution result, linked to its originating call by `call_id`.
    /// File data printed in `output` is left out.
    pub fn tool(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        let output = output.into();
        let output = match without_blobs(&output) {
            Cow::Borrowed(_) => output,
            Cow::Owned(kept) => kept,
        };
        Self {
            kind: EntryKind::Tool {
                call_id: call_id.into(),
                output,
                parts: Vec::new(),
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a tool result with multimodal content.
    pub fn tool_with_parts(
        call_id: impl Into<String>,
        output: impl Into<String>,
        parts: Vec<ContentPart>,
    ) -> Self {
        let mut entry = Self::tool(call_id, output);
        if let EntryKind::Tool {
            parts: entry_parts, ..
        } = &mut entry.kind
        {
            *entry_parts = parts;
        }
        entry
    }

    /// Create a memory injection entry with search score metadata.
    pub fn memory(content: impl Into<String>, id: impl Into<String>, score: f64) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: Some("memory".into()),
                metadata: Some(serde_json::json!({"id": id.into(), "score": score})),
            },
            timestamp: Utc::now(),
        }
    }

    /// Create an agent progress entry for subagent status updates.
    pub fn agent_progress(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: Some("agent_progress".into()),
                metadata: Some(serde_json::json!({"id": id.into()})),
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a compaction summary entry that replaces truncated history.
    pub fn summary(content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: Some("summary".into()),
                metadata: None,
            },
            timestamp: Utc::now(),
        }
    }

    // -- Accessors ---------------------------------------------------------

    /// Get the effective role string of this entry for wire-format serialization.
    pub fn role(&self) -> &'static str {
        match &self.kind {
            EntryKind::SystemPrompt(_) => "system",
            EntryKind::Developer { .. } => "developer",
            EntryKind::User { .. } => "user",
            EntryKind::Assistant { .. } => "assistant",
            EntryKind::Tool { .. } => "tool",
        }
    }

    /// Convert this entry to a wire-format [`Message`].
    pub fn to_message(&self) -> Message {
        match &self.kind {
            EntryKind::SystemPrompt(text) => Message::system(text),
            EntryKind::Developer {
                content,
                tag,
                metadata,
            } => {
                if tag.as_deref() == Some("agent_progress") {
                    let id = metadata
                        .as_ref()
                        .and_then(|m| m["id"].as_str())
                        .unwrap_or("?");
                    Message::developer(format!("[Agent {id} progress]\n{content}"))
                } else {
                    Message::developer(content)
                }
            }
            EntryKind::User { content, parts } => {
                if let Some(parts) = parts {
                    Message::user_with_parts(content, parts.clone())
                } else {
                    Message::user(content)
                }
            }
            EntryKind::Assistant {
                content,
                tool_calls,
                reasoning,
                reasoning_details,
                ..
            } => {
                let mut message = if let Some(tc) = tool_calls {
                    Message::assistant_with_tool_calls(content, tc.clone())
                } else {
                    Message::assistant(content)
                };
                message.reasoning = reasoning.clone();
                message.reasoning_details = reasoning_details.clone();
                message
            }
            // Histories saved before tools kept file data out of their output.
            EntryKind::Tool {
                call_id, output, ..
            } => Message::tool_result(call_id, without_blobs(output)),
        }
    }

    /// Get a reference to the entry's text content.
    pub fn content(&self) -> &str {
        match &self.kind {
            EntryKind::SystemPrompt(s)
            | EntryKind::Developer { content: s, .. }
            | EntryKind::User { content: s, .. }
            | EntryKind::Assistant { content: s, .. } => s,
            EntryKind::Tool { output, .. } => output,
        }
    }

    /// Get the reasoning trace, if any.
    pub fn reasoning(&self) -> Option<&str> {
        match &self.kind {
            EntryKind::Assistant { reasoning, .. } => reasoning.as_deref(),
            _ => None,
        }
    }

    /// Replace the entry's text content.
    pub fn set_content(&mut self, new: String) {
        if self.content() != new {
            self.invalidate_native_reasoning();
        }
        match &mut self.kind {
            EntryKind::SystemPrompt(s)
            | EntryKind::Developer { content: s, .. }
            | EntryKind::User { content: s, .. }
            | EntryKind::Assistant { content: s, .. } => *s = new,
            EntryKind::Tool { output, .. } => *output = new,
        }
    }

    fn invalidate_native_reasoning(&mut self) {
        if let EntryKind::Assistant {
            reasoning_details,
            reasoning,
            reasoning_model,
            ..
        } = &mut self.kind
            && (reasoning_details
                .native_items(NativeReasoningFormat::Anthropic)
                .is_some()
                || reasoning_details
                    .native_items(NativeReasoningFormat::Responses)
                    .is_some()
                || reasoning_details
                    .native_items(NativeReasoningFormat::Mistral)
                    .is_some())
        {
            *reasoning_details = ReasoningDetails::default();
            *reasoning = None;
            *reasoning_model = None;
        }
    }

    /// Returns `true` if this entry is the system prompt (`EntryKind::SystemPrompt`). Distinct from [`Self::is_system_message`], which checks for developer messages.
    pub fn is_system(&self) -> bool {
        matches!(self.kind, EntryKind::SystemPrompt(_))
    }

    /// Returns `true` if this entry is a developer-injected message (`EntryKind::Developer`).
    pub fn is_developer(&self) -> bool {
        matches!(self.kind, EntryKind::Developer { .. })
    }

    fn has_tag(&self, t: &str) -> bool {
        matches!(&self.kind, EntryKind::Developer { tag: Some(tag), .. } if tag == t)
    }

    /// Returns `true` if this entry is a developer message without a tag (`EntryKind::Developer { tag: None, .. }`). This maps to `Role::Developer` in wire format. Distinct from [`Self::is_system`], which checks for the system prompt.
    pub fn is_system_message(&self) -> bool {
        matches!(&self.kind, EntryKind::Developer { tag: None, .. })
    }

    /// Returns `true` if this entry is a user message.
    pub fn is_user(&self) -> bool {
        matches!(self.kind, EntryKind::User { .. })
    }

    /// Returns `true` if this entry is an assistant response.
    pub fn is_assistant(&self) -> bool {
        matches!(self.kind, EntryKind::Assistant { .. })
    }

    /// Returns `true` if this entry is a tool result.
    pub fn is_tool(&self) -> bool {
        matches!(self.kind, EntryKind::Tool { .. })
    }

    /// Returns `true` if this entry is a memory injection.
    pub fn is_memory(&self) -> bool {
        self.has_tag("memory")
    }

    /// Returns `true` if this entry is a reminder.
    pub fn is_reminder(&self) -> bool {
        self.has_tag("reminder")
    }

    fn is_interaction_mode(&self) -> bool {
        self.has_tag("interaction_mode")
    }

    /// Returns `true` if this entry is a compaction summary.
    pub fn is_summary(&self) -> bool {
        self.has_tag("summary")
    }

    /// Returns `true` if this entry contains subagent progress information.
    pub fn is_agent_progress(&self) -> bool {
        self.has_tag("agent_progress")
    }
}

// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

/// Ordered sequence of conversation entries.
///
/// The conversation is the mutable IR that flows through each agent turn. It is
/// caller-owned (the [`Agent`](super::Agent) borrows it during a turn). Entries are
/// serializable to/from JSON for session persistence.
///
/// # Entry kinds and wire mapping
///
/// Each entry carries an [`EntryKind`] tag that determines how it maps to LLM
/// wire format in [`to_messages`](Self::to_messages):
///
/// | Kind | Wire role | Notes |
/// |------|-----------|-------|
/// | `SystemPrompt` | `system` | Always position 0 |
/// | `Developer` | `developer` | System messages, reminders, memories, summaries, agent progress (differentiated by `tag`) |
/// | `User` | `user` | Plain text or multimodal parts |
/// | `Assistant` | `assistant` | May include tool calls |
/// | `Tool` | `tool` | Linked to assistant by `call_id` |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    entries: Vec<ConversationEntry>,
    #[serde(skip)]
    compaction_context: Option<CompactionContext>,
    #[serde(default, skip)]
    last_turn_start: Option<usize>,
}

// ---------------------------------------------------------------------------
// Compaction tuning
// ---------------------------------------------------------------------------

/// Hard cap on how long the compaction LLM call may take before we give up.
const COMPACTION_TIMEOUT_SECS: u64 = 300;
/// Added after a compaction that lands mid-task, so the model carries on
/// instead of answering the summary.
const CONTINUATION_NOTE: &str = "The conversation was compacted while you were working. \
    The summary above covers what happened so far. Carry on with the task from where you \
    stopped: do not ask the user anything first, do not repeat finished work, and do not \
    stop to report the compaction. Keep replies and progress updates in the language of the user's \
    request unless they asked for another language (default to English when unclear), even if the \
    summary uses another language.";
/// Rough characters-per-token heuristic for sizing input.
pub(crate) const CHARS_PER_TOKEN: usize = 2;
/// Tokens we reserve for the compaction model's response.
const RESPONSE_BUDGET_TOKENS: u32 = 12_000;
/// Tokens we reserve for the system prompt and trailing instruction.
const SYSTEM_OVERHEAD_TOKENS: u32 = 2_000;
/// Floor on input budget so small-context models still get a usable window.
const MIN_INPUT_CHARS: usize = 4_096;
/// Conservative default when the provider can't report a context window.
const DEFAULT_CONTEXT_WINDOW: u32 = 32_000;

/// Tool names whose calls indicate a file was *read* during a span.
const READ_TOOLS: &[&str] = &[
    "file_read",
    "read_lines",
    "glob",
    "grep",
    "image_read",
    "attach",
];
/// Tool names whose calls indicate a file was *modified* during a span.
const MODIFY_TOOLS: &[&str] = &[
    "file_write",
    "file_delete",
    "str_replace",
    "str_replace_regex",
];

/// Build a deterministic `## Files Touched` markdown block from the tool calls
/// in a summarized span, classifying each touched path as read or modified.
///
/// Returns `None` when no file-touching tool calls are present. Paths are
/// de-duplicated, preserve first-seen order, and a path that was ever modified
/// is reported only under Modified.
fn files_touched_block(entries: &[&ConversationEntry]) -> Option<String> {
    let mut read: Vec<String> = Vec::new();
    let mut modified: Vec<String> = Vec::new();

    for entry in entries {
        let EntryKind::Assistant {
            tool_calls: Some(calls),
            ..
        } = &entry.kind
        else {
            continue;
        };
        for call in calls {
            let Some(path) = call
                .arguments
                .get("path")
                .or_else(|| call.arguments.get("file_path"))
                .and_then(|v| v.as_str())
            else {
                continue;
            };
            let name = call.name.as_str();
            if MODIFY_TOOLS.contains(&name) {
                if !modified.iter().any(|p| p == path) {
                    modified.push(path.to_string());
                }
            } else if READ_TOOLS.contains(&name) && !read.iter().any(|p| p == path) {
                read.push(path.to_string());
            }
        }
    }

    // A modified file is more salient than a read of the same path.
    read.retain(|p| !modified.iter().any(|m| m == p));

    if read.is_empty() && modified.is_empty() {
        return None;
    }

    let mut out = String::from("## Files Touched");
    if !modified.is_empty() {
        out.push_str("\n### Modified");
        for p in &modified {
            out.push_str("\n- ");
            out.push_str(p);
        }
    }
    if !read.is_empty() {
        out.push_str("\n### Read");
        for p in &read {
            out.push_str("\n- ");
            out.push_str(p);
        }
    }
    Some(out)
}

/// Whether an entry past the system-prompt prefix goes into an LLM summary.
/// Shared by [`Conversation::summarizable_entry_count`] and the compaction
/// body so the size gate counts exactly what would be summarized.
fn is_summarizable(e: &ConversationEntry) -> bool {
    !e.is_memory()
        && !e.is_reminder()
        && !e.is_interaction_mode()
        && !e.has_tag(NATIVE_COMPACTION_TAG)
        && !matches!(e.kind, EntryKind::SystemPrompt(_))
        && (!e.content().is_empty()
            || matches!(&e.kind, EntryKind::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty()))
}

impl Conversation {
    pub fn interaction_mode(&self) -> InteractionMode {
        self.entries
            .iter()
            .find_map(|entry| match &entry.kind {
                EntryKind::Developer {
                    tag: Some(tag),
                    metadata: Some(value),
                    ..
                } if tag == "interaction_mode" => match serde_json::from_value(value.clone()) {
                    Ok(mode) => Some(mode),
                    Err(error) => {
                        tracing::warn!(%error, "Invalid saved interaction mode; using Act mode");
                        Some(InteractionMode::default())
                    }
                },
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Replace the mode instruction without growing the conversation on each switch.
    pub fn set_interaction_mode(&mut self, mode: InteractionMode) {
        let kind = EntryKind::Developer {
            content: mode.developer_instruction().into(),
            tag: Some("interaction_mode".into()),
            metadata: Some(serde_json::json!(mode)),
        };
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.is_interaction_mode())
        {
            entry.kind = kind;
        } else {
            self.add(ConversationEntry {
                kind,
                timestamp: Utc::now(),
            });
        }
    }

    /// Create an empty conversation.
    pub fn new() -> Self {
        Self {
            compaction_context: None,
            entries: Vec::new(),
            last_turn_start: None,
        }
    }

    /// Create a conversation pre-seeded with a system prompt.
    pub fn with_system(content: impl Into<String>) -> Self {
        let mut conv = Self::new();
        conv.add(ConversationEntry::system(content));
        conv
    }

    /// Append an entry to the conversation.
    pub fn add(&mut self, entry: ConversationEntry) {
        self.entries.push(entry);
    }

    /// View the entry history.
    pub fn entries(&self) -> &[ConversationEntry] {
        &self.entries
    }

    /// Mutable access to the entry history.
    pub fn entries_mut(&mut self) -> &mut Vec<ConversationEntry> {
        &mut self.entries
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_turn_start = None;
    }

    /// Mark the current position as the start of the latest turn.
    pub fn mark_turn_start(&mut self) {
        self.last_turn_start = Some(self.entries.len());
    }

    /// Entries added since the last [`mark_turn_start`](Self::mark_turn_start) call.
    pub fn last_turn(&self) -> &[ConversationEntry] {
        match self.last_turn_start {
            Some(idx) if idx < self.entries.len() => &self.entries[idx..],
            _ => &[],
        }
    }

    /// Set or replace the system prompt. If the first entry is already a
    /// System entry, it is replaced. Otherwise a new one is prepended.
    pub fn set_system(&mut self, content: impl Into<String>) {
        let entry = ConversationEntry::system(content);
        if self.entries.first().is_some_and(|e| e.is_system()) {
            self.entries[0] = entry;
        } else {
            self.entries.insert(0, entry);
        }
    }

    /// Replace entry at index 0 or push if empty.
    pub fn replace_first(&mut self, entry: ConversationEntry) {
        if self.entries.is_empty() {
            self.entries.push(entry);
        } else {
            self.entries[0] = entry;
        }
    }

    /// Ensure every assistant tool call has a matching tool result and vice
    /// versa. Unpaired tool calls are stripped from their assistant entry (the
    /// assistant text is kept) and orphan tool-result entries are removed.
    ///
    /// Also enforces adjacency: tool-result entries are moved to immediately
    /// follow their corresponding assistant entry. OpenAI (and compatible
    /// APIs) require tool messages right after the assistant that made the
    /// call — any interleaved reminder/memory/progress entry would cause a
    /// 400 error.
    pub fn sanitize(&mut self) {
        // IDs can repeat across responses. Pair each result with the latest
        // preceding unmatched occurrence, then restore adjacency per exchange.
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        struct CallPosition {
            assistant: usize,
            call: usize,
        }
        let mut pending: HashMap<&str, Vec<CallPosition>> = HashMap::new();
        let mut pairs = HashMap::new();
        for (index, entry) in self.entries.iter().enumerate() {
            match &entry.kind {
                EntryKind::Assistant {
                    tool_calls: Some(calls),
                    ..
                } => {
                    for (call, tool) in calls.iter().enumerate() {
                        pending.entry(&tool.id).or_default().push(CallPosition {
                            assistant: index,
                            call,
                        });
                    }
                }
                EntryKind::Tool { call_id, .. } => {
                    if let Some(position) = pending.get_mut(call_id.as_str()).and_then(Vec::pop) {
                        pairs.insert(position, index);
                    }
                }
                _ => {}
            }
        }
        let mut entries: Vec<_> = std::mem::take(&mut self.entries)
            .into_iter()
            .map(Some)
            .collect();
        for index in 0..entries.len() {
            let Some(mut entry) = entries[index].take() else {
                continue;
            };
            if entry.is_tool() {
                continue;
            }
            let mut results = Vec::new();
            let mut changed = false;
            if let EntryKind::Assistant {
                tool_calls: Some(calls),
                ..
            } = &mut entry.kind
            {
                let before = calls.len();
                let mut call = 0;
                calls.retain(|_| {
                    let position = CallPosition {
                        assistant: index,
                        call,
                    };
                    call += 1;
                    if let Some(result) = pairs
                        .get(&position)
                        .and_then(|result| entries[*result].take())
                    {
                        results.push(result);
                        true
                    } else {
                        false
                    }
                });
                changed = before != calls.len();
                if calls.is_empty()
                    && let EntryKind::Assistant { tool_calls, .. } = &mut entry.kind
                {
                    *tool_calls = None;
                }
            }
            if changed {
                entry.invalidate_native_reasoning();
            }
            self.entries.push(entry);
            self.entries.extend(results);
        }
    }

    /// Remove all Memory entries.
    pub fn strip_memories(&mut self) {
        self.entries.retain(|e| !e.is_memory());
    }

    /// Remove unused reminders while preserving the input of signed responses.
    pub fn strip_reminders(&mut self) {
        self.remove_unbound_reminders(None);
    }

    fn remove_unbound_reminders(&mut self, content: Option<&str>) {
        let bound = self.entries.iter().rposition(|entry| {
            matches!(&entry.kind,
            EntryKind::Assistant { reasoning_details, reasoning_prefix: Some(_), .. }
                if reasoning_details.is_context_bound())
        });
        let mut index = 0;
        self.entries.retain(|entry| {
            let keep = bound.is_some_and(|bound| index < bound)
                || !entry.is_reminder()
                || content.is_some_and(|content| entry.content() != content);
            index += 1;
            keep
        });
    }

    /// Remove all agent progress entries.
    pub fn strip_agent_progress(&mut self) {
        self.entries.retain(|e| !e.is_agent_progress());
    }

    /// Remove unused reminders, then push a new reminder.
    pub fn replace_reminder(&mut self, content: impl Into<String>) {
        self.strip_reminders();
        self.add(ConversationEntry::reminder(content));
    }

    /// Move this reminder to the end without removing unrelated reminders.
    pub(crate) fn refresh_periodic_reminder(&mut self, content: &str) {
        self.remove_unbound_reminders(Some(content));
        self.add(ConversationEntry::reminder(content));
    }

    /// Replace or insert an agent progress entry for the given subagent ID.
    pub fn replace_agent_progress(&mut self, id: impl Into<String>, content: impl Into<String>) {
        let id = id.into();
        self.entries.retain(|e| {
            !matches!(
                &e.kind,
                EntryKind::Developer { tag: Some(t), metadata: Some(m), .. }
                if t == "agent_progress" && m["id"].as_str() == Some(&id)
            )
        });
        self.add(ConversationEntry::agent_progress(id, content));
    }

    /// Collect all memory IDs referenced in this conversation's entries.
    pub fn memory_ids(&self) -> HashSet<&str> {
        let mut set = HashSet::new();
        for entry in &self.entries {
            if let EntryKind::Developer {
                tag: Some(t),
                metadata: Some(m),
                ..
            } = &entry.kind
                && t == "memory"
                && let Some(id) = m["id"].as_str()
                && !id.is_empty()
            {
                set.insert(id);
            }
        }
        set
    }

    /// Remove memory entries whose similarity score falls below the threshold.
    pub fn strip_low_score_memories(&mut self, keep: usize) {
        let mut memory_entries: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let EntryKind::Developer {
                    tag: Some(t),
                    metadata: Some(m),
                    ..
                } = &e.kind
                    && t == "memory"
                {
                    Some((i, m["score"].as_f64().unwrap_or(0.0)))
                } else {
                    None
                }
            })
            .collect();

        if memory_entries.len() <= keep {
            return;
        }

        memory_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let to_remove: HashSet<usize> = memory_entries[keep..].iter().map(|(i, _)| *i).collect();

        let mut idx = 0;
        self.entries.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }

    /// Find the content of the most recent assistant entry, if any.
    pub fn last_assistant_content(&self) -> Option<&str> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.is_assistant())
            .map(|e| e.content())
    }

    // -- Wire format conversion -------------------------------------------

    /// Convert entries to LLM wire-format [`Message`] values.
    ///
    /// Entries map to messages according to their [`EntryKind`]. Tool images
    /// follow the corresponding group of tool results in a user message.
    ///
    /// Call [`sanitize`](Self::sanitize) before this if tool call/result pairing
    /// might be inconsistent (e.g. after session load with partial writes).
    pub fn to_messages(&self) -> Vec<Message> {
        self.messages_for_model(None, true)
    }

    /// Replay reasoning and native checkpoints only since the last model change.
    pub fn to_messages_for_model(&self, model: &Model) -> Vec<Message> {
        self.messages_for_model(Some(model), true)
    }

    fn reasoning_replay_start(&self, model: &Model) -> usize {
        self.entries
            .iter()
            .rposition(|entry| {
                matches!(&entry.kind, EntryKind::Assistant { reasoning_model, .. }
                if !reasoning_model.as_ref().is_some_and(|source| {
                    source.provider == model.provider
                        && source.capability_name() == model.capability_name()
                }))
            })
            .map_or(0, |index| index + 1)
    }

    /// Keep the exact tool definitions and reasoning settings used by this agent.
    pub fn set_compaction_context(&mut self, llm: &AgentLlmConfig, tools: Vec<ToolDefinition>) {
        self.compaction_context = Some(CompactionContext {
            llm: llm.clone(),
            tools,
        });
    }

    fn source_digest(&self, end: usize) -> Option<String> {
        let tools = self
            .compaction_context
            .as_ref()
            .map_or(&[][..], |context| context.tools.as_slice());
        let mut hash = RequestPrefix::new(tools)?;
        for entry in &self.entries[..end] {
            if !entry.has_tag(NATIVE_COMPACTION_TAG)
                && !entry.is_memory()
                && !entry.is_reminder()
                && !entry.is_agent_progress()
                && !entry.is_interaction_mode()
            {
                hash.update(&entry.kind)?;
            }
        }
        Some(hash.digest())
    }

    fn native_checkpoint(&self, model: &Model) -> Option<(usize, ReasoningDetails)> {
        let replay_start = self.reasoning_replay_start(model);
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| {
                if index < replay_start {
                    return None;
                }
                let EntryKind::Developer {
                    metadata: Some(metadata),
                    ..
                } = &entry.kind
                else {
                    return None;
                };
                if !entry.has_tag(NATIVE_COMPACTION_TAG) {
                    return None;
                }
                let source: Model = metadata.get("model")?.as_str()?.parse().ok()?;
                if source.provider != model.provider
                    || source.capability_name() != model.capability_name()
                {
                    return None;
                }
                if metadata.get("source_digest")?.as_str()? != self.source_digest(index)? {
                    return None;
                }
                let context = ReasoningDetails::deserialize(metadata.get("context")?).ok()?;
                Some((index, context))
            })
    }

    pub(crate) fn messages_without_checkpoint(&self, model: &Model) -> Vec<Message> {
        self.messages_for_model(Some(model), false)
    }

    fn messages_for_model(&self, model: Option<&Model>, use_checkpoint: bool) -> Vec<Message> {
        let mut msgs = Vec::with_capacity(self.entries.len());
        let replay_start = model.map_or(0, |model| self.reasoning_replay_start(model));
        let mut start = 0;
        if let Some((index, checkpoint)) = model
            .filter(|_| use_checkpoint)
            .and_then(|model| self.native_checkpoint(model))
        {
            msgs.extend(
                self.entries
                    .iter()
                    .take(self.compaction_prefix())
                    .map(ConversationEntry::to_message),
            );
            let mut snapshot = Message::assistant("");
            snapshot.reasoning_details = checkpoint;
            msgs.push(snapshot);
            msgs.extend(
                self.entries[..=index]
                    .iter()
                    .filter(|entry| entry.is_interaction_mode())
                    .map(ConversationEntry::to_message),
            );
            start = index + 1;
        }

        let tools = self
            .compaction_context
            .as_ref()
            .filter(|context| {
                model.is_some_and(|model| {
                    context.llm.model.provider == model.provider
                        && context.llm.model.capability_name() == model.capability_name()
                })
            })
            .map_or(&[][..], |context| context.tools.as_slice());
        let mut prefix = RequestPrefix::new(tools);
        let mut hashed = 0;
        let mut parts = Vec::new();
        for (index, entry) in self.entries.iter().enumerate().skip(start) {
            if entry.has_tag(NATIVE_COMPACTION_TAG) {
                continue;
            }
            if !matches!(entry.kind, EntryKind::Tool { .. }) && !parts.is_empty() {
                msgs.push(Message::user_with_parts(
                    "Attachments returned by the preceding tools:",
                    std::mem::take(&mut parts),
                ));
            }
            let mut message = entry.to_message();
            if index < replay_start {
                // Keep earlier traces for display without reviving them when
                // switching back to a model used before another response.
                message.reasoning = None;
                message.reasoning_details = ReasoningDetails::default();
            }
            if model.is_some() && message.reasoning_details.is_context_bound() {
                for preceding in &msgs[hashed..] {
                    prefix = prefix.and_then(|mut prefix| {
                        prefix.push(preceding)?;
                        Some(prefix)
                    });
                }
                hashed = msgs.len();
                let stored = match &entry.kind {
                    EntryKind::Assistant {
                        reasoning_prefix, ..
                    } => reasoning_prefix.as_deref(),
                    _ => None,
                };
                let expected = prefix.as_ref().and_then(|prefix| {
                    let mut prefix = prefix.clone();
                    prefix.push(&message)?;
                    Some(prefix.digest())
                });
                if stored.is_none() || stored != expected.as_deref() {
                    message.reasoning = None;
                    message.reasoning_details = ReasoningDetails::default();
                    tracing::debug!("Omitting reasoning with an unverified request prefix");
                }
            }
            msgs.push(message);
            if let EntryKind::Tool {
                parts: tool_parts, ..
            } = &entry.kind
            {
                parts.extend(tool_parts.iter().cloned());
            }
        }
        if !parts.is_empty() {
            msgs.push(Message::user_with_parts(
                "Attachments returned by the preceding tools:",
                parts,
            ));
        }

        msgs
    }

    // -- Compaction --------------------------------------------------------

    /// Prefer native compaction, otherwise summarize entries via an LLM call.
    ///
    /// Native checkpoints retain the transcript for model switches. The local
    /// fallback below replaces older entries after a nonempty summary succeeds.
    ///
    /// # Local fallback
    ///
    /// 1. Collects summarizable entries (excludes system prompt, memories, reminders, empty)
    /// 2. Sizes the input to fit the compaction model's context window
    /// 3. Sanitizes the input: drops `Assistant.tool_calls` and converts `Tool`
    ///    entries to developer messages so the request has no orphaned tool refs
    /// 4. Sends to the LLM with [`COMPACTION_PROMPT`] under a hard timeout
    /// 5. On success, replaces the conversation body with a single summary entry
    ///    (memories and reminders are stripped only after the LLM call succeeds)
    ///
    /// Returns `Ok(Some(summary))` on success, `Ok(None)` when there is nothing
    /// to summarize or the model returned empty output, and `Err` on stream
    /// errors or timeout. The conversation is **not mutated** unless the call
    /// succeeds with non-empty output.
    pub async fn compact_with_llm(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
    ) -> anyhow::Result<Option<CompactionResult>> {
        self.compact_with_llm_keeping_tokens(provider, model, DEFAULT_KEEP_RECENT_TOKENS)
            .await
    }

    /// Like [`compact_with_llm`](Self::compact_with_llm) but allows the caller
    /// to specify how many recent user turns to preserve verbatim after
    /// compaction. A "turn" is a user message plus all following
    /// assistant/tool/developer entries until the next user message.
    pub async fn compact_with_llm_keeping(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
        keep_recent_turns: usize,
    ) -> anyhow::Result<Option<CompactionResult>> {
        let prefix = self.compaction_prefix();

        // Walk backwards through the original entries counting user messages
        // as turn starts. Everything from the Nth-from-last user message
        // onward is kept verbatim.
        let keep_from = if keep_recent_turns == 0 {
            self.entries.len()
        } else {
            let mut turns_seen = 0usize;
            let mut split = prefix;
            for i in (prefix..self.entries.len()).rev() {
                if self.entries[i].is_user() {
                    turns_seen += 1;
                    if turns_seen >= keep_recent_turns {
                        split = i;
                        break;
                    }
                }
            }
            split
        };

        self.compact_from(provider, model, keep_from).await
    }

    /// Like [`compact_with_llm_keeping`](Self::compact_with_llm_keeping) but
    /// preserves recent history by a *token budget* rather than a turn count.
    ///
    /// Walks backward from the newest entry accumulating an estimated token
    /// count (via `CHARS_PER_TOKEN`) and snaps the kept boundary to a user
    /// message, so a partial turn is never kept and recent context stays stable
    /// even when individual turns vary wildly in size.
    pub async fn compact_with_llm_keeping_tokens(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
        keep_recent_tokens: u32,
    ) -> anyhow::Result<Option<CompactionResult>> {
        let keep_from = self.keep_recent_from(keep_recent_tokens);
        self.compact_from(provider, model, keep_from).await
    }

    pub(crate) fn has_compaction_input(&self, keep_recent_tokens: u32) -> bool {
        self.entries
            .iter()
            .take(self.keep_recent_from(keep_recent_tokens))
            .skip(self.compaction_prefix())
            .any(is_summarizable)
    }

    fn keep_recent_from(&self, keep_recent_tokens: u32) -> usize {
        let prefix = self.compaction_prefix();

        if keep_recent_tokens == 0 {
            self.entries.len()
        } else {
            let budget_chars = (keep_recent_tokens as usize).saturating_mul(CHARS_PER_TOKEN);
            let mut acc = 0usize;
            // Default: keep nothing extra (summarize everything) if no user
            // boundary is found. As we walk back, `split` tracks the oldest
            // user message we've decided to keep.
            let mut split = self.entries.len();
            let mut current_turn_chars = None;
            for i in (prefix..self.entries.len()).rev() {
                acc = acc.saturating_add(self.entries[i].request_bytes());
                if self.entries[i].is_user() {
                    split = i;
                    current_turn_chars.get_or_insert(acc);
                    if acc >= budget_chars {
                        break;
                    }
                }
            }
            // Keeping an unfinished turn that is over budget frees nothing
            // where it is needed most, in a long tool loop. Summarize it too;
            // the continuation note carries the request.
            if current_turn_chars.is_some_and(|chars| chars > budget_chars)
                && self.turn_in_progress()
            {
                self.entries.len()
            } else {
                split
            }
        }
    }

    /// Whether the model is mid-task: the newest message is not a final
    /// assistant reply, so another request follows.
    fn turn_in_progress(&self) -> bool {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.is_user() || entry.is_assistant() || entry.is_tool())
            .is_some_and(|entry| {
                !matches!(
                    &entry.kind,
                    EntryKind::Assistant { tool_calls, .. }
                        if tool_calls.as_ref().is_none_or(Vec::is_empty)
                )
            })
    }

    /// The note to add after a compaction that keeps no user message while
    /// the model is mid-task, so it does not stop at the summary.
    fn continuation_after(&self, keep_from: usize) -> Option<ConversationEntry> {
        if self.entries[keep_from..]
            .iter()
            .any(ConversationEntry::is_user)
            || !self.turn_in_progress()
        {
            return None;
        }
        let request = self.entries[..keep_from]
            .iter()
            .rev()
            .find(|entry| entry.is_user())
            .map(|entry| entry.content().to_owned())
            .or_else(|| {
                self.entries
                    .iter()
                    .rev()
                    .find_map(|entry| entry.continued_request().map(str::to_owned))
            });
        Some(ConversationEntry::continuation(request))
    }

    /// Number of leading entries (just the system prompt, if present) that are
    /// never eligible for summarization.
    fn compaction_prefix(&self) -> usize {
        if self.entries.first().is_some_and(|e| e.is_system()) {
            1
        } else {
            0
        }
    }

    fn compaction_request(&self, model: &Model) -> CompletionRequest {
        let messages = self.to_messages_for_model(model);
        let mut request = if let Some(context) =
            self.compaction_context.as_ref().filter(|context| {
                context.llm.model.provider == model.provider
                    && context.llm.model.capability_name() == model.capability_name()
            }) {
            let mut request = CompletionRequest::from_config(&context.llm, messages);
            request.tools = context.tools.clone();
            request
        } else {
            CompletionRequest::new(model.clone(), messages)
        };
        request.max_tokens = Some(RESPONSE_BUDGET_TOKENS);
        request
    }

    async fn compact_natively(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
    ) -> anyhow::Result<Option<CompactionResult>> {
        if !self
            .entries
            .iter()
            .any(|entry| entry.is_user() || entry.is_assistant())
        {
            return Ok(None);
        }
        let request = self.compaction_request(model);
        let result = tokio::time::timeout(
            Duration::from_secs(COMPACTION_TIMEOUT_SECS),
            provider.compact(request, NATIVE_COMPACTION_PROMPT),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Native compaction timed out"))??;
        let Some(result) = result else {
            return Ok(None);
        };
        anyhow::ensure!(
            !result.context.is_empty(),
            "Native compaction returned no context"
        );
        let checkpoint = NativeCheckpoint {
            model: model.clone(),
            source_digest: self
                .source_digest(self.entries.len())
                .ok_or_else(|| anyhow::anyhow!("Could not fingerprint compaction history"))?,
            context: result.context,
        };
        let metadata = serde_json::to_value(checkpoint)?;
        let continuation = self.continuation_after(self.entries.len());
        // Keep one opaque window per model and the original transcript.
        let mut removed_before_turn = 0;
        let turn_start = self.last_turn_start.unwrap_or(0);
        let mut index = 0;
        self.entries.retain(|entry| {
            let same_model = entry.has_tag(NATIVE_COMPACTION_TAG)
                && matches!(&entry.kind, EntryKind::Developer { metadata: Some(metadata), .. }
                    if metadata.get("model").and_then(serde_json::Value::as_str)
                        .and_then(|source| source.parse::<Model>().ok())
                        .is_some_and(|source| source.provider == model.provider
                            && source.capability_name() == model.capability_name()));
            if same_model && index < turn_start {
                removed_before_turn += 1;
            }
            index += 1;
            !same_model
        });
        if let Some(start) = &mut self.last_turn_start {
            *start -= removed_before_turn;
        }
        self.entries.push(ConversationEntry {
            kind: EntryKind::Developer {
                content: result.summary.clone(),
                tag: Some(NATIVE_COMPACTION_TAG.into()),
                metadata: Some(metadata),
            },
            timestamp: Utc::now(),
        });
        self.entries.extend(continuation);
        tracing::info!(provider = provider.name(), model = %model, entries = self.entries.len(), "Native context checkpoint saved");
        Ok(Some(CompactionResult {
            summary: result.summary,
            usage: result.usage,
        }))
    }

    /// Shared compaction body: summarize the entries before `keep_from` into a
    /// single summary entry, keeping `entries[keep_from..]` verbatim. Returns
    /// `Ok(None)` when there is nothing to summarize.
    async fn compact_from(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
        keep_from: usize,
    ) -> anyhow::Result<Option<CompactionResult>> {
        if let Some((index, _)) = self.native_checkpoint(model)
            && provider.supports_native_context(&self.compaction_request(model))
            && !self.entries[index + 1..]
                .iter()
                .any(|entry| entry.is_user() || entry.is_assistant() || entry.is_tool())
        {
            return Ok(None);
        }
        if let Some(result) = self.compact_natively(provider, model).await? {
            return Ok(Some(result));
        }
        let prefix = self.compaction_prefix();
        let has_system = prefix == 1;

        let to_summarize: Vec<&ConversationEntry> = self
            .entries
            .iter()
            .take(keep_from)
            .skip(prefix)
            .filter(|e| is_summarizable(e))
            .collect();

        if to_summarize.is_empty() {
            return Ok(None);
        }

        // Derive the input/output budget from the compaction model's context
        // window. Fall back to a conservative default if the provider doesn't
        // report one.
        let window = provider
            .context_window(model)
            .await
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        let available_tokens =
            window.saturating_sub(RESPONSE_BUDGET_TOKENS + SYSTEM_OVERHEAD_TOKENS);
        let max_input_chars = ((available_tokens as usize) * CHARS_PER_TOKEN).max(MIN_INPUT_CHARS);
        let max_output_tokens = RESPONSE_BUDGET_TOKENS.min((window / 4).max(512));

        // Walk in reverse, accumulating until we hit the input budget.
        let mut total_chars = 0usize;
        let mut start_idx = 0;
        for (i, entry) in to_summarize.iter().enumerate().rev() {
            total_chars = total_chars.saturating_add(entry.summary_content().len());
            if total_chars > max_input_chars {
                start_idx = i + 1;
                break;
            }
        }

        // Drop any `Tool` entries at the head whose paired `Assistant` was cut
        // by the budget walk — sending an orphan tool_result without its
        // tool_call is rejected by OpenAI/Anthropic.
        let mut head = start_idx;
        while head < to_summarize.len() && to_summarize[head].is_tool() {
            head += 1;
        }
        let entries_for_llm = &to_summarize[head..];
        if entries_for_llm.is_empty() {
            tracing::warn!(
                "All entries exceed compaction budget or are orphan tool results, skipping LLM summary"
            );
            return Ok(None);
        }
        if head > 0 {
            tracing::info!(
                "Compaction: trimmed {} oldest entries to fit compaction model context",
                head
            );
        }

        // Check if we're updating an existing summary (iterative compaction).
        let has_prior_summary = entries_for_llm.iter().any(|e| e.is_summary());
        let system_prompt = if has_prior_summary {
            COMPACTION_PROMPT_ITERATIVE
        } else {
            COMPACTION_PROMPT
        };

        // Build the message list, sanitizing entries to avoid provider errors:
        // - `Assistant { tool_calls: Some(_) }` is rewritten to plain assistant
        //   text (with a `[tool calls: …]` annotation) because the compaction
        //   request declares `tools: vec![]`, and providers reject `tool_calls`
        //   in messages when no tools are declared.
        // - `Tool { … }` entries become `developer` messages — `tool` messages
        //   require a preceding `Assistant` with matching `tool_calls`, which
        //   we just stripped.
        let mut messages = Vec::with_capacity(entries_for_llm.len() + 2);
        messages.push(Message::system(system_prompt));
        for entry in entries_for_llm {
            let mut msg = match &entry.kind {
                EntryKind::Assistant {
                    tool_calls: Some(_),
                    ..
                } => Message::assistant(entry.summary_content()),
                EntryKind::Tool { output, .. } => {
                    Message::developer(format!("[tool result]\n{output}"))
                }
                _ => entry.to_message(),
            };
            // Summaries retain outcomes; signed and encrypted reasoning belongs
            // only on the original assistant message in the active history.
            msg.reasoning = None;
            msg.reasoning_details = ReasoningDetails::default();
            messages.push(msg);
        }
        // Finish with a user turn. Some providers move developer messages into
        // the system instruction, leaving an unsupported assistant prefill.
        let instruction = "Now produce the summary as instructed in the system prompt.";
        if let Some(last) = messages.last_mut()
            && last.role == Role::User
        {
            last.content.push_str("\n\n");
            last.content.push_str(instruction);
        } else {
            messages.push(Message::user(instruction));
        }

        let request = CompletionRequest {
            model: model.clone(),
            messages,
            tools: vec![],
            max_tokens: Some(max_output_tokens),
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(Decimal::new(3, 1)),
                ..Default::default()
            },
            modalities: vec![],
            audio_config: None,
            image_config: None,
            user: None,
            provider_preferences: None,
            session_id: None,
        };

        tracing::info!(
            model = %model,
            entries = entries_for_llm.len(),
            messages = request.messages.len(),
            context_window = window,
            max_output_tokens,
            keep_from,
            "Compaction: sending LLM request"
        );

        // Bounded by a hard timeout so a stalled connection cannot block the
        // agent loop indefinitely.
        let (summary, usage) = match tokio::time::timeout(
            std::time::Duration::from_secs(COMPACTION_TIMEOUT_SECS),
            async {
                let mut stream = provider.complete(request);
                let mut buf = String::new();
                let mut usage = TokenUsage::default();
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(StreamEvent::ContentDelta(delta)) => buf.push_str(&delta),
                        Ok(StreamEvent::Usage(u)) => usage = u,
                        Ok(_) => {}
                        Err(e) => return Err(anyhow::anyhow!("compaction stream error: {e}")),
                    }
                }
                Ok::<_, anyhow::Error>((buf, usage))
            },
        )
        .await
        {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                tracing::warn!(model = %model, "Compaction LLM stream error: {e}");
                return Err(e);
            }
            Err(_) => {
                tracing::warn!(model = %model, "Compaction LLM call timed out after {COMPACTION_TIMEOUT_SECS}s");
                return Err(anyhow::anyhow!(
                    "compaction timed out after {COMPACTION_TIMEOUT_SECS}s"
                ));
            }
        };

        if summary.trim().is_empty() {
            tracing::warn!(model = %model, "Compaction LLM call returned empty summary, skipping");
            return Ok(None);
        }

        // Append a deterministic Files Touched section derived from the tool
        // calls in the summarized span — computed in code (not asked of the
        // LLM) so a compacted coding session keeps an accurate read/modified
        // map regardless of how the model phrased its summary.
        let summary = match files_touched_block(&to_summarize) {
            Some(block) => format!("{}\n\n{block}", summary.trim_end()),
            None => summary,
        };

        let kept_count = self.entries.len() - keep_from;
        tracing::info!(
            "Compacted {} entries into summary ({} chars), keeping {} recent entries",
            to_summarize.len(),
            summary.len(),
            kept_count,
        );

        let continuation = self.continuation_after(keep_from);

        // The LLM call succeeded — now we can safely mutate the conversation.
        // Collect the tail entries we want to keep before mutating.
        let tail: Vec<ConversationEntry> = self.entries[keep_from..]
            .iter()
            .filter(|e| !e.is_memory() && !e.is_reminder() && !e.is_interaction_mode())
            .cloned()
            .collect();

        let mut new_entries = Vec::new();
        if has_system {
            new_entries.push(self.entries[0].clone());
        }
        new_entries.extend(
            self.entries
                .iter()
                .filter(|entry| entry.is_interaction_mode())
                .cloned(),
        );
        new_entries.push(ConversationEntry::summary(summary.clone()));
        new_entries.extend(tail);
        new_entries.extend(continuation);
        self.entries = new_entries;

        Ok(Some(CompactionResult { summary, usage }))
    }

    /// Count entries eligible for LLM summarization (excludes system prompt, memories, reminders, and empty entries).
    pub fn summarizable_entry_count(&self) -> usize {
        self.entries
            .iter()
            .skip(self.compaction_prefix())
            .filter(|e| is_summarizable(e))
            .count()
    }

    /// Truncate tool result outputs longer than `max_bytes` to save context.
    /// Appends `\n[truncated]` to indicate the output was cut.
    /// Returns the number of entries truncated.
    pub fn truncate_long_tool_outputs(&mut self, max_bytes: usize) -> usize {
        let mut count = 0;

        for entry in &mut self.entries {
            if !entry.is_tool() {
                continue;
            }

            let content = entry.content();
            if content.len() <= max_bytes || content.ends_with("[truncated]") {
                continue;
            }

            let truncate_at = (0..=max_bytes)
                .rev()
                .find(|&i| content.is_char_boundary(i))
                .unwrap_or(0);

            let mut new_content = content[..truncate_at].to_string();
            new_content.push_str("\n[truncated]");
            entry.set_content(new_content);
            count += 1;
        }

        count
    }

    /// Replace binary content parts (images, documents, video, audio) with
    /// text placeholders across all user entries. Returns the number of parts
    /// replaced. This frees large base64 payloads that can cause context overflow.
    pub fn strip_binary_parts(&mut self) -> usize {
        let mut count = 0;

        for entry in &mut self.entries {
            let parts = match &mut entry.kind {
                EntryKind::User {
                    parts: Some(parts), ..
                }
                | EntryKind::Tool { parts, .. } => parts,
                _ => continue,
            };
            let mut stripped = 0;
            {
                for part in parts.iter_mut() {
                    let replacement = match part {
                        ContentPart::Image { media_type, .. } => {
                            Some(format!("[Image: {}]", media_type))
                        }
                        ContentPart::Document {
                            media_type,
                            filename,
                            ..
                        } => Some(format!("[Document: {} ({})]", filename, media_type)),
                        ContentPart::Video {
                            media_type,
                            filename,
                            ..
                        } => Some(format!("[Video: {} ({})]", filename, media_type)),
                        ContentPart::Audio {
                            media_type,
                            filename,
                            ..
                        } => Some(format!("[Audio: {} ({})]", filename, media_type)),
                        ContentPart::ImageUrl { url } => Some(format!("[Image: {}]", url)),
                        ContentPart::VideoUrl { url } => Some(format!("[Video: {}]", url)),
                        ContentPart::Text { .. } => None,
                    };

                    if let Some(text) = replacement {
                        *part = ContentPart::Text { text };
                        stripped += 1;
                    }
                }
            }
            // The agent learns why the file it got is gone.
            if stripped > 0
                && let EntryKind::Tool { output, .. } = &mut entry.kind
            {
                output.push_str("\n[The model rejected the attachment, so it was removed.]");
            }
            count += stripped;
        }

        count
    }

    /// Remove all tool-related entries: Tool entries and tool_calls from
    /// Assistant entries. Returns the number of entries removed.
    pub fn strip_tool_messages(&mut self) -> usize {
        let before = self.entries.len();

        self.entries.retain(|e| !e.is_tool());

        for entry in &mut self.entries {
            entry.invalidate_native_reasoning();
            if let EntryKind::Assistant { tool_calls, .. } = &mut entry.kind {
                *tool_calls = None;
            }
        }

        before - self.entries.len()
    }

    /// Keep only the system prompt and the last user + assistant entries.
    ///
    /// If no user entry exists (e.g. conversation is all tool calls), a
    /// synthetic "Continue." user message is inserted so the LLM always
    /// receives at least one user message — OpenAI rejects conversations
    /// without one.
    pub fn truncate_to_last_exchange(&mut self) {
        let system = self.entries.iter().find(|e| e.is_system()).cloned();
        let last_user = self.entries.iter().rev().find(|e| e.is_user()).cloned();
        let last_assistant = self
            .entries
            .iter()
            .rev()
            .find(|e| e.is_assistant())
            .cloned();

        self.entries.clear();

        if let Some(sys) = system {
            self.entries.push(sys);
        }
        if let Some(user) = last_user {
            self.entries.push(user);
        } else {
            self.entries
                .push(ConversationEntry::user("Continue.".to_string()));
        }
        if let Some(asst) = last_assistant {
            self.entries.push(asst);
        }
    }

    /// Hard-reset the conversation to just the system prompt (if any) and the
    /// last user entry.
    ///
    /// If no user entry exists, a synthetic "Continue." message is inserted
    /// so the LLM always receives at least one user message.
    pub fn truncate_to_latest(&mut self) {
        let system = self.entries.iter().find(|e| e.is_system()).cloned();
        let last_user = self.entries.iter().rev().find(|e| e.is_user()).cloned();

        self.entries.clear();

        if let Some(sys) = system {
            self.entries.push(sys);
        }
        if let Some(user) = last_user {
            self.entries.push(user);
        } else {
            self.entries
                .push(ConversationEntry::user("Continue.".to_string()));
        }
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::Role;

    #[test]
    fn tool_images_follow_all_results_and_survive_session_serialization() {
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::assistant_with_tool_calls(
            "",
            vec![
                ToolCall {
                    id: "one".into(),
                    name: "image_read".into(),
                    arguments: serde_json::json!({}),
                },
                ToolCall {
                    id: "two".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({}),
                },
            ],
            None,
        ));
        let entry = ConversationEntry::tool_with_parts(
            "one",
            "Image loaded",
            vec![ContentPart::Image {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
            }],
        );
        conversation.add(serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap());
        conversation.add(ConversationEntry::tool("two", "text"));
        let messages = conversation.to_messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, Role::Tool);
        assert_eq!(messages[2].role, Role::Tool);
        assert_eq!(messages[3].role, Role::User);
        assert!(
            matches!(&messages[3].parts.as_ref().unwrap()[0], ContentPart::Image { media_type, data } if media_type == "image/png" && data == "aW1hZ2U=")
        );
        assert_eq!(conversation.strip_binary_parts(), 1);
        assert!(matches!(
            &conversation.to_messages()[3].parts.as_ref().unwrap()[0],
            ContentPart::Text { .. }
        ));
    }

    #[test]
    fn tool_document_reaches_the_next_user_message() {
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::tool_with_parts(
            "call",
            "Attached file: report.pdf",
            vec![ContentPart::Document {
                media_type: "application/pdf".into(),
                filename: "report.pdf".into(),
                data: "cGRm".into(),
            }],
        ));
        let messages = conversation.to_messages();
        assert_eq!(
            messages[1].content,
            "Attachments returned by the preceding tools:"
        );
        assert!(
            matches!(&messages[1].parts.as_ref().unwrap()[0], ContentPart::Document { filename, data, .. } if filename == "report.pdf" && data == "cGRm")
        );
    }

    #[test]
    fn repeated_provider_tool_ids_keep_each_exchange() {
        let mut conversation = Conversation::new();
        for output in ["first result", "second result"] {
            conversation.add(ConversationEntry::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_0".into(),
                    name: "lookup".into(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ));
            conversation.add(ConversationEntry::tool("call_0", output));
        }
        conversation.sanitize();
        let messages = conversation.to_messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].content, "first result");
        assert_eq!(messages[3].content, "second result");
        conversation.sanitize();
        assert_eq!(conversation.to_messages().len(), 4);
    }

    #[test]
    fn assistant_reasoning_survives_conversion_and_session_serialization() {
        let entry = ConversationEntry::assistant_with_reasoning("answer", Some("thinking".into()));
        let restored: ConversationEntry =
            serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
        assert_eq!(restored.to_message().reasoning.as_deref(), Some("thinking"));
        assert!(
            ConversationEntry::user("question")
                .to_message()
                .reasoning
                .is_none()
        );
    }

    #[test]
    fn reasoning_replay_is_limited_to_the_source_model_and_survives_serialization() {
        let source: Model = "openrouter:example/model".parse().unwrap();
        let details = serde_json::json!([{"type":"reasoning.encrypted","data":"opaque","signature":"signature","index":0}]);
        let entry = ConversationEntry::assistant_with_reasoning("answer", Some("thinking".into()))
            .with_reasoning_details(
                serde_json::from_value(details.clone()).unwrap(),
                source.clone(),
            );
        let restored = serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
        let mut conversation = Conversation::new();
        let restored: ConversationEntry = restored;
        let mut prefix =
            RequestPrefix::from_request(&CompletionRequest::new(source.clone(), vec![])).unwrap();
        prefix.push(&restored.to_message()).unwrap();
        conversation.add(restored.with_reasoning_prefix(Some(prefix.digest())));
        for same in [source, "openrouter:alias,example/model".parse().unwrap()] {
            let original = conversation.to_messages_for_model(&same);
            assert_eq!(
                serde_json::to_value(&original[0].reasoning_details).unwrap(),
                details
            );
            assert_eq!(original[0].reasoning.as_deref(), Some("thinking"));
        }
        for other in ["openrouter:example/other", "openai:example/model"] {
            let messages = conversation.to_messages_for_model(&other.parse().unwrap());
            assert!(messages[0].reasoning.is_none());
            assert!(messages[0].reasoning_details.is_empty());
            assert_eq!(messages[0].content, "answer");
        }
    }

    #[test]
    fn switching_back_does_not_replay_thinking_from_before_the_model_change() {
        let source: Model = "openrouter:example/model".parse().unwrap();
        for other in ["openrouter:example/other", "openai:example/model"] {
            for details in [
                ReasoningDetails::chat_field("reasoning_content", "old thinking"),
                serde_json::from_value(serde_json::json!([
                    {"type":"reasoning.encrypted","data":"opaque","signature":"signed"}
                ]))
                .unwrap(),
            ] {
                let mut conversation = Conversation::new();
                let entry = ConversationEntry::assistant_with_tool_calls(
                    "first answer",
                    vec![ToolCall {
                        id: "call_example".into(),
                        name: "example".into(),
                        arguments: serde_json::json!({}),
                    }],
                    Some("old thinking".into()),
                )
                .with_reasoning_details(details, source.clone());
                let mut prefix = RequestPrefix::new(&[]).unwrap();
                prefix.push(&entry.to_message()).unwrap();
                conversation.add(entry.with_reasoning_prefix(Some(prefix.digest())));
                conversation.add(ConversationEntry::tool("call_example", "evidence"));
                conversation.add(
                    ConversationEntry::assistant("other answer").with_reasoning_details(
                        ReasoningDetails::default(),
                        other.parse().unwrap(),
                    ),
                );
                let restored: Conversation =
                    serde_json::from_str(&serde_json::to_string(&conversation).unwrap()).unwrap();
                let messages = restored.to_messages_for_model(&source);
                assert!(
                    messages.iter().all(|message| message.reasoning.is_none()
                        && message.reasoning_details.is_empty())
                );
                assert_eq!(messages[0].content, "first answer");
                assert_eq!(
                    messages[0].tool_calls.as_ref().unwrap()[0].id,
                    "call_example"
                );
                assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_example"));
                assert_eq!(messages[1].content, "evidence");
                assert_eq!(messages[2].content, "other answer");
                assert_eq!(restored.entries()[0].reasoning(), Some("old thinking"));

                let mut resumed = restored;
                let details = serde_json::from_value(serde_json::json!([
                    {"type":"reasoning.encrypted","data":"new opaque","signature":"new signed"}
                ]))
                .unwrap();
                let entry = ConversationEntry::assistant_with_reasoning(
                    "resumed answer",
                    Some("new thinking".into()),
                )
                .with_reasoning_details(details, source.clone());
                let mut prefix =
                    RequestPrefix::from_request(&CompletionRequest::new(source.clone(), messages))
                        .unwrap();
                prefix.push(&entry.to_message()).unwrap();
                resumed.add(entry.with_reasoning_prefix(Some(prefix.digest())));
                let messages = resumed.to_messages_for_model(&source);
                assert!(messages[0].reasoning.is_none());
                assert_eq!(messages[3].reasoning.as_deref(), Some("new thinking"));
                assert!(!messages[3].reasoning_details.is_empty());
            }
        }
    }

    #[test]
    fn editing_native_history_invalidates_replay_instead_of_restoring_removed_tool_calls() {
        for kind in [
            NativeReasoningFormat::Anthropic,
            NativeReasoningFormat::Responses,
            NativeReasoningFormat::Mistral,
        ] {
            let model: Model = "openrouter:example/model".parse().unwrap();
            let original = ConversationEntry::assistant_with_tool_calls("original", vec![ToolCall {
                id:"call_example".into(),name:"lookup".into(),arguments:serde_json::json!({}),
            }], Some("thinking".into())).with_reasoning_details(
                ReasoningDetails::native(kind, vec![serde_json::json!({"type":"thinking","thinking":"thinking","signature":"signature"})]), model.clone(),
            );
            let mut edited = original.clone();
            edited.set_content("original".into());
            assert!(!edited.to_message().reasoning_details.is_empty());
            edited.set_content("edited".into());
            assert!(edited.to_message().reasoning_details.is_empty());
            let mut conversation = Conversation::new();
            conversation.add(original.clone());
            conversation.sanitize();
            let message = &conversation.to_messages_for_model(&model)[0];
            assert!(message.tool_calls.is_none());
            assert!(message.reasoning_details.is_empty());
            assert_eq!(message.content, "original");
            let mut conversation = Conversation::new();
            conversation.add(original);
            conversation.add(ConversationEntry::tool("call_example", "evidence"));
            conversation.strip_tool_messages();
            assert!(
                conversation.to_messages_for_model(&model)[0]
                    .reasoning_details
                    .is_empty()
            );
        }
    }

    #[test]
    fn interrupted_or_unattributed_reasoning_is_retained_for_display_only() {
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::assistant_with_reasoning(
            "partial",
            Some("unfinished thinking".into()),
        ));
        assert_eq!(
            conversation.to_messages()[0].reasoning.as_deref(),
            Some("unfinished thinking")
        );
        let request =
            conversation.to_messages_for_model(&"openrouter:example/model".parse().unwrap());
        assert!(request[0].reasoning.is_none());
        assert!(request[0].reasoning_details.is_empty());
        assert_eq!(request[0].content, "partial");
    }

    #[test]
    fn reasoning_and_tool_arguments_count_toward_the_recent_history_budget() {
        let mut conversation = Conversation::new();
        conversation.add(ConversationEntry::user("task"));
        let entry = ConversationEntry::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_provider".into(),
                name: "example".into(),
                arguments: serde_json::json!({"input":"a".repeat(100)}),
            }],
            Some("thinking".repeat(100)),
        );
        assert!(entry.request_bytes() > 900);
        conversation.add(entry);
        conversation.add(ConversationEntry::tool("call_provider", "evidence"));
        assert_eq!(
            conversation.keep_recent_from(100),
            conversation.entries().len()
        );
    }

    #[test]
    fn test_conversation_add_entry() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("hello"));
        assert_eq!(conv.entries().len(), 1);
    }

    #[test]
    fn test_conversation_with_system() {
        let conv = Conversation::with_system("You are helpful");
        assert_eq!(conv.entries().len(), 1);
        assert!(conv.entries()[0].is_system());
    }

    #[test]
    fn test_last_assistant_content() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("hi"));
        assert!(conv.last_assistant_content().is_none());

        conv.add(ConversationEntry::assistant("hello"));
        conv.add(ConversationEntry::user("how are you?"));
        let last = conv.last_assistant_content().unwrap();
        assert_eq!(last, "hello");
    }

    #[test]
    fn test_truncate_to_latest() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("You are helpful"));
        conv.add(ConversationEntry::user("first question"));
        conv.add(ConversationEntry::assistant("first answer"));
        conv.add(ConversationEntry::user("second question"));
        conv.add(ConversationEntry::assistant("second answer"));
        conv.add(ConversationEntry::user("third question"));

        conv.truncate_to_latest();

        assert_eq!(conv.entries().len(), 2);
        assert!(conv.entries()[0].is_system());
        assert_eq!(conv.entries()[0].content(), "You are helpful");
        assert!(conv.entries()[1].is_user());
        assert_eq!(conv.entries()[1].content(), "third question");
    }

    #[test]
    fn test_truncate_to_latest_no_system() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("hello"));
        conv.add(ConversationEntry::assistant("hi"));
        conv.add(ConversationEntry::user("bye"));

        conv.truncate_to_latest();

        assert_eq!(conv.entries().len(), 1);
        assert!(conv.entries()[0].is_user());
        assert_eq!(conv.entries()[0].content(), "bye");
    }

    #[test]
    fn test_truncate_long_tool_outputs() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("do stuff"));
        conv.add(ConversationEntry::tool("c1", "x".repeat(2000)));
        conv.add(ConversationEntry::tool("c2", "short"));
        conv.add(ConversationEntry::tool("c3", "y".repeat(1500)));

        let truncated = conv.truncate_long_tool_outputs(1024);
        assert_eq!(truncated, 2);

        assert!(conv.entries()[1].content().ends_with("\n[truncated]"));
        assert!(conv.entries()[1].content().len() <= 1024 + "\n[truncated]".len());

        assert_eq!(conv.entries()[2].content(), "short");

        assert!(conv.entries()[3].content().ends_with("\n[truncated]"));
    }

    #[test]
    fn test_truncate_long_tool_outputs_leaves_non_tool_entries() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("u".repeat(5000)));
        conv.add(ConversationEntry::assistant("a".repeat(5000)));
        conv.add(ConversationEntry::tool("c1", "t".repeat(5000)));

        let truncated = conv.truncate_long_tool_outputs(1024);
        assert_eq!(truncated, 1);

        assert_eq!(conv.entries()[0].content().len(), 5000);
        assert_eq!(conv.entries()[1].content().len(), 5000);

        assert!(conv.entries()[2].content().ends_with("\n[truncated]"));
    }

    #[test]
    fn test_truncate_long_tool_outputs_unicode_boundary() {
        let mut conv = Conversation::new();
        let emoji_content: String = "🔥".repeat(257); // 1028 bytes, just over
        conv.add(ConversationEntry::tool("c1", &emoji_content));

        let truncated = conv.truncate_long_tool_outputs(1024);
        assert_eq!(truncated, 1);

        assert!(conv.entries()[0].content().ends_with("\n[truncated]"));
    }

    #[test]
    fn test_truncate_long_tool_outputs_idempotent() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::tool("c1", "x".repeat(2000)));

        let first = conv.truncate_long_tool_outputs(1024);
        assert_eq!(first, 1);

        let second = conv.truncate_long_tool_outputs(1024);
        assert_eq!(second, 0);
    }

    #[test]
    fn test_strip_tool_messages() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::user("do stuff"));
        conv.add(ConversationEntry::assistant_with_tool_calls(
            "I'll help",
            vec![ToolCall {
                id: "c1".into(),
                name: "exec".into(),
                arguments: serde_json::json!({}),
            }],
            None,
        ));
        conv.add(ConversationEntry::tool("c1", "result"));
        conv.add(ConversationEntry::assistant("done!"));
        conv.add(ConversationEntry::user("thanks"));

        let stripped = conv.strip_tool_messages();
        assert_eq!(stripped, 1); // 1 tool entry removed

        assert_eq!(conv.entries().len(), 5);
        // Assistant entry should have tool_calls stripped
        assert!(conv.entries()[2].is_assistant());
        if let EntryKind::Assistant { tool_calls, .. } = &conv.entries()[2].kind {
            assert!(tool_calls.is_none());
        }
        assert_eq!(conv.entries()[2].content(), "I'll help");
    }

    #[test]
    fn test_strip_binary_parts() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user("hello"));
        conv.add(ConversationEntry::user_with_parts(
            "look at this",
            vec![
                ContentPart::Image {
                    media_type: "image/png".into(),
                    data: "iVBORw0KGgo=".into(),
                },
                ContentPart::Text {
                    text: "caption".into(),
                },
            ],
        ));
        conv.add(ConversationEntry::assistant("nice image"));
        conv.add(ConversationEntry::user_with_parts(
            "another",
            vec![ContentPart::Document {
                media_type: "application/pdf".into(),
                filename: "report.pdf".into(),
                data: "JVBER=".into(),
            }],
        ));

        let stripped = conv.strip_binary_parts();
        assert_eq!(stripped, 2); // 1 image + 1 document

        // First user entry is plain text — unchanged
        if let EntryKind::User { parts, .. } = &conv.entries()[0].kind {
            assert!(parts.is_none());
        }

        // Second entry: image replaced with placeholder, text kept
        if let EntryKind::User { parts, .. } = &conv.entries()[1].kind {
            let parts = parts.as_ref().unwrap();
            assert_eq!(parts.len(), 2);
            assert!(
                matches!(&parts[0], ContentPart::Text { text } if text == "[Image: image/png]")
            );
            assert!(matches!(&parts[1], ContentPart::Text { text } if text == "caption"));
        }

        // Fourth entry: document replaced
        if let EntryKind::User { parts, .. } = &conv.entries()[3].kind {
            let parts = parts.as_ref().unwrap();
            assert_eq!(parts.len(), 1);
            assert!(matches!(&parts[0], ContentPart::Text { text } if text.contains("report.pdf")));
        }
    }

    #[test]
    fn test_truncate_to_last_exchange() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::user("q1"));
        conv.add(ConversationEntry::assistant("a1"));
        conv.add(ConversationEntry::user("q2"));
        conv.add(ConversationEntry::assistant_with_tool_calls(
            "working",
            vec![ToolCall {
                id: "c1".into(),
                name: "exec".into(),
                arguments: serde_json::json!({}),
            }],
            None,
        ));
        conv.add(ConversationEntry::tool("c1", "result"));
        conv.add(ConversationEntry::assistant("final answer"));

        conv.truncate_to_last_exchange();

        assert_eq!(conv.entries().len(), 3);
        assert!(conv.entries()[0].is_system());
        assert_eq!(conv.entries()[0].content(), "sys");
        assert!(conv.entries()[1].is_user());
        assert_eq!(conv.entries()[1].content(), "q2");
        assert!(conv.entries()[2].is_assistant());
        assert_eq!(conv.entries()[2].content(), "final answer");
    }

    #[test]
    fn test_truncate_to_last_exchange_no_assistant() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::user("hello"));

        conv.truncate_to_last_exchange();

        assert_eq!(conv.entries().len(), 2);
        assert!(conv.entries()[0].is_system());
        assert!(conv.entries()[1].is_user());
    }

    // -- New tests for new functionality -----------------------------------

    #[test]
    fn test_strip_memories() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::memory("remembered fact", "m1", 0.85));
        conv.add(ConversationEntry::user("hello"));
        conv.add(ConversationEntry::memory("another fact", "m2", 0.72));

        conv.strip_memories();

        assert_eq!(conv.entries().len(), 2);
        assert!(conv.entries()[0].is_system());
        assert!(conv.entries()[1].is_user());
    }

    #[test]
    fn test_strip_low_score_memories() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::memory("low", "m1", 0.5));
        conv.add(ConversationEntry::memory("high", "m2", 0.95));
        conv.add(ConversationEntry::memory("mid", "m3", 0.75));
        conv.add(ConversationEntry::user("hello"));

        conv.strip_low_score_memories(1);

        // Only the highest-scoring memory should remain
        let memories: Vec<_> = conv.entries().iter().filter(|e| e.is_memory()).collect();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content(), "high");

        // Non-memory entries untouched
        assert_eq!(conv.entries().len(), 3); // sys + memory + user
    }

    #[test]
    fn test_strip_low_score_memories_keeps_all_when_under_limit() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::memory("a", "m1", 0.9));
        conv.add(ConversationEntry::memory("b", "m2", 0.8));

        conv.strip_low_score_memories(5);

        let memories: Vec<_> = conv.entries().iter().filter(|e| e.is_memory()).collect();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_replace_reminder() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::reminder("old reminder"));
        conv.add(ConversationEntry::user("hello"));

        conv.replace_reminder("new reminder");

        // Old reminder gone, new one appended
        let reminders: Vec<_> = conv.entries().iter().filter(|e| e.is_reminder()).collect();
        assert_eq!(reminders.len(), 1);
        assert_eq!(reminders[0].content(), "new reminder");
    }

    #[test]
    fn periodic_reminder_preserves_other_reminders() {
        let mut conv = Conversation::new();
        conv.refresh_periodic_reminder("tool rules");
        conv.add(ConversationEntry::reminder("continue the reply"));
        conv.refresh_periodic_reminder("tool rules");
        let contents: Vec<_> = conv
            .entries()
            .iter()
            .map(ConversationEntry::content)
            .collect();
        assert_eq!(contents, ["continue the reply", "tool rules"]);
    }

    #[test]
    fn test_memory_ids() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::memory("fact 1", "id-a", 0.9));
        conv.add(ConversationEntry::user("hello"));
        conv.add(ConversationEntry::memory("fact 2", "id-c", 0.7));

        let ids = conv.memory_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("id-a"));
        assert!(ids.contains("id-c"));
    }

    #[test]
    fn test_to_messages_system() {
        let conv = Conversation::with_system("You are helpful");
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, "You are helpful");
    }

    #[test]
    fn test_to_messages_reminder() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::reminder("remember this"));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Developer);
        assert_eq!(msgs[0].content, "remember this");
    }

    #[test]
    fn test_to_messages_memory() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::memory("a fact", "m1", 0.85));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Developer);
        assert_eq!(msgs[0].content, "a fact");
    }

    #[test]
    fn test_to_messages_summary() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::summary("conversation summary here"));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Developer);
        assert_eq!(msgs[0].content, "conversation summary here");
    }

    #[test]
    fn test_to_messages_tool() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::tool("call-1", "result data"));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Tool);
        assert_eq!(msgs[0].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(msgs[0].content, "result data");
    }

    #[test]
    fn test_to_messages_user_with_parts() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::user_with_parts(
            "Look",
            vec![ContentPart::Text {
                text: "caption".into(),
            }],
        ));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
        assert!(msgs[0].parts.is_some());
    }

    #[test]
    fn test_to_messages_system_message() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("system prompt"));
        conv.add(ConversationEntry::system_message("injected info"));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, "system prompt");
        assert_eq!(msgs[1].role, Role::Developer);
        assert_eq!(msgs[1].content, "injected info");
    }

    #[test]
    fn test_set_content() {
        let mut entry = ConversationEntry::user("old");
        entry.set_content("new".to_string());
        assert_eq!(entry.content(), "new");

        let mut tool_entry = ConversationEntry::tool("c1", "old output");
        tool_entry.set_content("new output".to_string());
        assert_eq!(tool_entry.content(), "new output");
    }

    #[test]
    fn test_set_system() {
        let mut conv = Conversation::with_system("old");
        conv.set_system("new");
        assert_eq!(conv.entries().len(), 1);
        assert_eq!(conv.entries()[0].content(), "new");

        // When no system entry exists, prepend
        let mut conv2 = Conversation::new();
        conv2.add(ConversationEntry::user("hello"));
        conv2.set_system("injected");
        assert_eq!(conv2.entries().len(), 2);
        assert!(conv2.entries()[0].is_system());
        assert_eq!(conv2.entries()[0].content(), "injected");
    }

    #[test]
    fn test_entry_kind_predicates() {
        assert!(ConversationEntry::system("s").is_system());
        assert!(ConversationEntry::user("u").is_user());
        assert!(ConversationEntry::assistant("a").is_assistant());
        assert!(ConversationEntry::tool("id", "out").is_tool());
        assert!(ConversationEntry::memory("m", "id", 0.5).is_memory());
        assert!(ConversationEntry::reminder("r").is_reminder());
        assert!(ConversationEntry::summary("s").is_summary());
    }

    #[test]
    fn test_strip_reminders() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::reminder("r1"));
        conv.add(ConversationEntry::user("u"));
        conv.add(ConversationEntry::reminder("r2"));

        conv.strip_reminders();
        assert_eq!(conv.entries().len(), 1);
        assert!(conv.entries()[0].is_user());
    }

    // -- compact_with_llm --------------------------------------------------

    mod compact_with_llm_tests {
        use super::*;
        use async_trait::async_trait;
        use flashmind_types::{
            AgentEvent, AliasedModel, CompletionStream, FinishReason, Provider, Role,
        };
        use std::sync::{Arc, Mutex};

        /// Records each request passed to `complete()` and replays a scripted
        /// response stream. Optionally reports a custom context window.
        struct MockProvider {
            captured: Arc<Mutex<Vec<CompletionRequest>>>,
            response: Vec<Result<StreamEvent, String>>,
            context_window: Option<u32>,
        }

        impl MockProvider {
            fn new(response: Vec<Result<StreamEvent, String>>) -> Self {
                Self {
                    captured: Arc::new(Mutex::new(Vec::new())),
                    response,
                    context_window: None,
                }
            }

            fn with_context_window(mut self, window: u32) -> Self {
                self.context_window = Some(window);
                self
            }

            fn captured(&self) -> Arc<Mutex<Vec<CompletionRequest>>> {
                self.captured.clone()
            }
        }

        #[async_trait]
        impl LlmProvider for MockProvider {
            fn name(&self) -> &str {
                "mock-compaction"
            }

            fn provider(&self) -> Provider {
                Provider::Ollama
            }

            async fn context_window(&self, _model: &Model) -> Option<u32> {
                self.context_window
            }

            fn complete(&self, request: CompletionRequest) -> CompletionStream {
                self.captured.lock().unwrap().push(request);
                let events = self.response.clone();
                Box::pin(async_stream::stream! {
                    for ev in events {
                        match ev {
                            Ok(e) => yield Ok(e),
                            Err(msg) => yield Err(anyhow::anyhow!(msg)),
                        }
                    }
                })
            }
        }

        enum NativeOutcome {
            Success,
            Unsupported,
            Failed,
            Pending,
        }

        struct NativeProvider {
            outcome: NativeOutcome,
            captured: Mutex<Vec<CompletionRequest>>,
            local: MockProvider,
        }

        impl NativeProvider {
            fn new(outcome: NativeOutcome) -> Self {
                Self {
                    outcome,
                    captured: Mutex::new(Vec::new()),
                    local: MockProvider::new(vec![
                        Ok(StreamEvent::ContentDelta("local summary".into())),
                        Ok(StreamEvent::Finished(FinishReason::Stop)),
                    ]),
                }
            }
        }

        #[async_trait]
        impl LlmProvider for NativeProvider {
            fn name(&self) -> &str {
                "native"
            }
            fn provider(&self) -> Provider {
                Provider::OpenAi
            }
            fn complete(&self, request: CompletionRequest) -> CompletionStream {
                self.local.complete(request)
            }
            fn supports_native_context(&self, request: &CompletionRequest) -> bool {
                request.sampling.top_k.is_none()
            }
            async fn compact(
                &self,
                request: CompletionRequest,
                instructions: &str,
            ) -> anyhow::Result<Option<flashmind_types::NativeCompaction>> {
                if !self.supports_native_context(&request) {
                    return Ok(None);
                }
                assert!(instructions.contains("important reasoning"));
                self.captured.lock().unwrap().push(request);
                match self.outcome {
                    NativeOutcome::Success => Ok(Some(flashmind_types::NativeCompaction {
                        context: ReasoningDetails::native(
                            NativeReasoningFormat::Responses,
                            vec![
                                serde_json::json!({"type":"compaction","encrypted_content":"opaque","extra":"keep"}),
                            ],
                        ),
                        summary: "native checkpoint".into(),
                        usage: TokenUsage::default(),
                    })),
                    NativeOutcome::Unsupported => Ok(None),
                    NativeOutcome::Failed => anyhow::bail!("native failed"),
                    NativeOutcome::Pending => std::future::pending().await,
                }
            }
        }

        fn native_conversation() -> (Conversation, Model) {
            let model: Model = "openai:example-model".parse().unwrap();
            let mut conversation = Conversation::with_system("original instructions");
            conversation.add(ConversationEntry::user("original task"));
            conversation.add(
                ConversationEntry::assistant_with_reasoning(
                    "original answer",
                    Some("trace-only fact".into()),
                )
                .with_reasoning_details(ReasoningDetails::default(), model.clone()),
            );
            (conversation, model)
        }

        #[tokio::test]
        async fn native_compaction_precedes_local_summary_and_survives_reload_and_model_switches() {
            let (mut conversation, model) = native_conversation();
            let provider = NativeProvider::new(NativeOutcome::Success);
            let mut config = AgentLlmConfig::new(model.clone());
            config.reasoning = ReasoningLevel::High;
            conversation.set_compaction_context(
                &config,
                vec![ToolDefinition {
                    name: "lookup".into(),
                    description: "Find evidence".into(),
                    parameters: serde_json::json!({"type":"object"}),
                }],
            );
            let result = conversation
                .compact_with_llm_keeping_tokens(&provider, &model, u32::MAX)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result.summary, "native checkpoint");
            assert!(provider.local.captured.lock().unwrap().is_empty());
            {
                let captured = provider.captured.lock().unwrap();
                assert_eq!(captured[0].reasoning, ReasoningLevel::High);
                assert_eq!(captured[0].max_tokens, Some(RESPONSE_BUDGET_TOKENS));
                assert_eq!(captured[0].tools[0].name, "lookup");
                assert_eq!(
                    captured[0].messages[2].reasoning.as_deref(),
                    Some("trace-only fact")
                );
            }
            let mut restored: Conversation =
                serde_json::from_str(&serde_json::to_string(&conversation).unwrap()).unwrap();
            restored.set_compaction_context(
                &config,
                conversation
                    .compaction_context
                    .as_ref()
                    .unwrap()
                    .tools
                    .clone(),
            );
            let messages = restored.to_messages_for_model(&model);
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0].content, "original instructions");
            assert_eq!(
                messages[1]
                    .reasoning_details
                    .native_items(NativeReasoningFormat::Responses)
                    .unwrap(),
                [
                    serde_json::json!({"type":"compaction","encrypted_content":"opaque","extra":"keep"})
                ]
            );
            let other = restored.to_messages_for_model(&"openai:other-model".parse().unwrap());
            assert_eq!(
                other
                    .iter()
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>(),
                ["original instructions", "original task", "original answer"]
            );
            assert!(
                other
                    .iter()
                    .all(|message| message.reasoning_details.is_empty())
            );
            assert!(
                restored
                    .compact_with_llm_keeping_tokens(&provider, &model, 0)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(provider.captured.lock().unwrap().len(), 1);
            restored.add(ConversationEntry::user("next task"));
            assert_eq!(
                restored.to_messages_for_model(&model)[2].content,
                "next task"
            );
            restored
                .compact_with_llm_keeping_tokens(&provider, &model, 0)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(provider.captured.lock().unwrap()[1].messages.len(), 3);
            let messages = restored.to_messages_for_model(&model);
            assert_eq!(messages.len(), 3);
            assert!(messages[2].content.contains("next task"));
        }

        #[tokio::test]
        async fn switching_models_does_not_revive_an_earlier_native_window() {
            let (mut conversation, first) = native_conversation();
            let provider = NativeProvider::new(NativeOutcome::Success);
            conversation
                .compact_with_llm_keeping_tokens(&provider, &first, 0)
                .await
                .unwrap();
            conversation.add(ConversationEntry::user("second task"));
            let second: Model = "openai:second-model".parse().unwrap();
            conversation.add(
                ConversationEntry::assistant("second answer")
                    .with_reasoning_details(ReasoningDetails::default(), second.clone()),
            );
            conversation
                .compact_with_llm_keeping_tokens(&provider, &second, 0)
                .await
                .unwrap();
            let first_window = conversation.to_messages_for_model(&first);
            assert_eq!(first_window.len(), 5);
            assert!(
                first_window
                    .iter()
                    .all(|message| message.reasoning.is_none()
                        && message.reasoning_details.is_empty())
            );
            assert_eq!(first_window[1].content, "original task");
            assert_eq!(first_window[2].content, "original answer");
            assert_eq!(first_window[3].content, "second task");
            assert_eq!(first_window[4].content, "second answer");
            let second_window = conversation.to_messages_for_model(&second);
            assert_eq!(second_window.len(), 2);
            assert!(!second_window[1].reasoning_details.is_empty());
        }

        #[tokio::test]
        async fn transport_changes_allow_local_recompaction_without_a_new_turn() {
            let (mut conversation, model) = native_conversation();
            let provider = NativeProvider::new(NativeOutcome::Success);
            conversation
                .compact_with_llm_keeping_tokens(&provider, &model, 0)
                .await
                .unwrap();
            let mut config = AgentLlmConfig::new(model.clone());
            config.sampling.top_k = Some(5);
            conversation.set_compaction_context(&config, vec![]);
            let result = conversation
                .compact_with_llm_keeping_tokens(&provider, &model, 0)
                .await
                .unwrap()
                .expect("local fallback must run");
            assert_eq!(result.summary, "local summary");
        }

        #[tokio::test]
        async fn transport_fallback_restores_the_archived_transcript() {
            let (mut conversation, model) = native_conversation();
            let provider = NativeProvider::new(NativeOutcome::Success);
            conversation
                .compact_with_llm_keeping_tokens(&provider, &model, 0)
                .await
                .unwrap();
            let mut config = AgentLlmConfig::new(model);
            config.sampling.top_k = Some(5);
            let cancel = tokio_util::sync::CancellationToken::new();
            let mut stream = crate::streaming::stream_llm_response(
                &provider,
                &config,
                &cancel,
                &mut conversation,
                &[],
            );
            while stream.next().await.is_some() {}
            stream.take_result().unwrap().unwrap();
            let captured = provider.local.captured.lock().unwrap();
            assert_eq!(captured[0].messages[1].content, "original task");
            assert_eq!(captured[0].messages[2].content, "original answer");
            assert!(
                captured[0]
                    .messages
                    .iter()
                    .all(|message| message.reasoning_details.is_empty())
            );
        }

        #[tokio::test]
        async fn changed_tools_invalidate_native_checkpoints() {
            let (mut conversation, model) = native_conversation();
            conversation
                .compact_with_llm_keeping_tokens(
                    &NativeProvider::new(NativeOutcome::Success),
                    &model,
                    0,
                )
                .await
                .unwrap();
            conversation.set_compaction_context(
                &AgentLlmConfig::new(model.clone()),
                vec![ToolDefinition {
                    name: "new_tool".into(),
                    description: "Changed tools".into(),
                    parameters: serde_json::json!({}),
                }],
            );
            assert_eq!(
                conversation.to_messages_for_model(&model)[1].content,
                "original task"
            );
        }

        #[tokio::test]
        async fn edited_local_summaries_invalidate_native_checkpoints() {
            let (mut conversation, model) = native_conversation();
            conversation
                .entries_mut()
                .insert(1, ConversationEntry::summary("saved constraints"));
            conversation
                .compact_with_llm_keeping_tokens(
                    &NativeProvider::new(NativeOutcome::Success),
                    &model,
                    0,
                )
                .await
                .unwrap();
            conversation.entries_mut()[1].set_content("edited constraints".into());
            let messages = conversation.to_messages_for_model(&model);
            assert!(
                messages
                    .iter()
                    .any(|message| message.content == "edited constraints")
            );
            assert!(
                messages
                    .iter()
                    .all(|message| message.reasoning_details.is_empty())
            );
        }

        #[tokio::test]
        async fn edited_transcripts_invalidate_native_checkpoints() {
            let (mut conversation, model) = native_conversation();
            conversation
                .compact_with_llm_keeping_tokens(
                    &NativeProvider::new(NativeOutcome::Success),
                    &model,
                    0,
                )
                .await
                .unwrap();
            conversation.entries_mut()[2].set_content("edited answer".into());
            let messages = conversation.to_messages_for_model(&model);
            assert_eq!(messages.len(), 3);
            assert_eq!(messages[2].content, "edited answer");
            assert!(
                messages
                    .iter()
                    .all(|message| message.reasoning_details.is_empty())
            );
        }

        #[tokio::test]
        async fn unsupported_native_compaction_falls_back_but_failure_and_cancellation_keep_history()
         {
            let (mut conversation, model) = native_conversation();
            let provider = NativeProvider::new(NativeOutcome::Unsupported);
            assert_eq!(
                conversation
                    .compact_with_llm_keeping_tokens(&provider, &model, 0)
                    .await
                    .unwrap()
                    .unwrap()
                    .summary,
                "local summary"
            );
            assert_eq!(provider.local.captured.lock().unwrap().len(), 1);
            let (mut conversation, model) = native_conversation();
            let before = serde_json::to_value(&conversation).unwrap();
            assert!(
                conversation
                    .compact_with_llm_keeping_tokens(
                        &NativeProvider::new(NativeOutcome::Failed),
                        &model,
                        0
                    )
                    .await
                    .is_err()
            );
            assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
            let provider = NativeProvider::new(NativeOutcome::Pending);
            {
                use std::future::Future;
                use std::task::{Context, Waker};
                let mut future =
                    Box::pin(conversation.compact_with_llm_keeping_tokens(&provider, &model, 0));
                assert!(
                    future
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
            }
            assert_eq!(serde_json::to_value(&conversation).unwrap(), before);
        }

        fn test_model() -> Model {
            Model {
                provider: Provider::Ollama,
                model: AliasedModel {
                    name: "test".into(),
                    real_name: None,
                },
            }
        }

        #[tokio::test]
        async fn local_compaction_keeps_tool_exchanges_and_invalidates_bound_reasoning() {
            let mut conv = Conversation::with_system("sys");
            let details: ReasoningDetails = serde_json::from_value(serde_json::json!([
                {"type":"reasoning.encrypted","data":"opaque","index":0}
            ]))
            .unwrap();
            conv.add(ConversationEntry::user("old task"));
            conv.add(
                ConversationEntry::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "call_old".into(),
                        name: "file_read".into(),
                        arguments: serde_json::json!({"path":"/srv/project/example.rs"}),
                    }],
                    Some("old thinking".into()),
                )
                .with_reasoning_details(details.clone(), test_model()),
            );
            conv.add(ConversationEntry::tool("call_old", "old evidence"));
            conv.add(ConversationEntry::assistant("old answer"));
            conv.add(ConversationEntry::user("recent task"));
            let recent = ConversationEntry::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_recent".into(),
                    name: "example".into(),
                    arguments: serde_json::json!({}),
                }],
                None,
            )
            .with_reasoning_details(details.clone(), test_model());
            let mut prefix = RequestPrefix::from_request(&CompletionRequest::new(
                test_model(),
                conv.to_messages_for_model(&test_model()),
            ))
            .unwrap();
            prefix.push(&recent.to_message()).unwrap();
            conv.add(recent.with_reasoning_prefix(Some(prefix.digest())));
            assert!(
                !conv
                    .to_messages_for_model(&test_model())
                    .last()
                    .unwrap()
                    .reasoning_details
                    .is_empty()
            );
            conv.add(ConversationEntry::tool("call_recent", "recent evidence"));
            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            conv.compact_with_llm_keeping(&provider, &test_model(), 1)
                .await
                .unwrap()
                .unwrap();
            let captured = provider.captured.lock().unwrap();
            assert!(
                captured[0]
                    .messages
                    .iter()
                    .any(|message| message.content.contains("/srv/project/example.rs"))
            );
            assert!(
                captured[0]
                    .messages
                    .iter()
                    .all(|message| message.reasoning.is_none()
                        && message.reasoning_details.is_empty())
            );
            let messages = conv.to_messages_for_model(&test_model());
            let assistant = messages
                .iter()
                .find(|message| message.role == Role::Assistant)
                .unwrap();
            assert!(assistant.reasoning_details.is_empty());
            assert_eq!(conv.entries().iter().find(|entry| matches!(&entry.kind,
                EntryKind::Assistant { tool_calls: Some(calls), .. } if calls[0].id == "call_recent"))
                .unwrap().to_message().reasoning_details, details);
            assert_eq!(assistant.tool_calls.as_ref().unwrap()[0].id, "call_recent");
            assert!(
                messages
                    .iter()
                    .any(|message| message.tool_call_id.as_deref() == Some("call_recent"))
            );
            assert!(
                conv.entries().iter().any(|entry| entry.is_summary()
                    && entry.content().contains("/srv/project/example.rs"))
            );
        }

        #[tokio::test]
        async fn manual_compaction_summarizes_history_under_the_recent_budget() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("first question"));
            conv.add(ConversationEntry::assistant("first answer"));
            conv.add(ConversationEntry::user("recent question"));
            conv.add(ConversationEntry::assistant("recent answer"));
            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("conversation summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let model = test_model();
            let events: Vec<_> = crate::compaction::compact_manually(&mut conv, &provider, &model)
                .collect()
                .await;
            assert!(
                matches!(events.as_slice(), [AgentEvent::Compacted(summary)] if summary == "conversation summary")
            );
            assert_eq!(conv.entries().len(), 2);
            assert!(conv.entries()[0].is_system());
            assert!(conv.entries()[1].is_summary());
        }

        #[tokio::test]
        async fn manual_compaction_preserves_history_when_the_summary_fails() {
            for response in [vec![], vec![Err("example stream failure".into())]] {
                let mut conv = Conversation::with_system("sys");
                conv.add(ConversationEntry::user("question"));
                conv.add(ConversationEntry::assistant("answer"));
                conv.add(ConversationEntry::memory("remembered fact", "m1", 0.9));
                let before = conv.entries().to_vec();
                let provider = MockProvider::new(response);
                let model = test_model();
                let events: Vec<_> =
                    crate::compaction::compact_manually(&mut conv, &provider, &model)
                        .collect()
                        .await;
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::Compacted(_)))
                );
                assert_eq!(conv.entries().len(), before.len());
                for (entry, original) in conv.entries().iter().zip(&before) {
                    assert_eq!(entry.timestamp, original.timestamp);
                    assert_eq!(entry.content(), original.content());
                }
            }
        }

        #[tokio::test]
        async fn manual_compaction_without_history_skips_the_model() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::memory("remembered fact", "m1", 0.9));
            let provider = MockProvider::new(vec![]);
            let model = test_model();
            let events: Vec<_> = crate::compaction::compact_manually(&mut conv, &provider, &model)
                .collect()
                .await;
            assert!(
                matches!(events.as_slice(), [AgentEvent::Status(message)] if message == "Nothing to summarize.")
            );
            assert!(provider.captured().lock().unwrap().is_empty());
            assert_eq!(conv.entries().len(), 2);
        }

        #[tokio::test]
        async fn empty_conversation_returns_none_and_doesnt_mutate() {
            let mut conv = Conversation::new();
            let provider = MockProvider::new(vec![]);
            let result = conv.compact_with_llm(&provider, &test_model()).await;
            assert!(matches!(result, Ok(None)));
            assert_eq!(conv.entries().len(), 0);
        }

        #[tokio::test]
        async fn only_system_prompt_returns_none_and_doesnt_strip_memories() {
            let mut conv = Conversation::with_system("you are helpful");
            conv.add(ConversationEntry::memory("a fact", "m1", 0.9));
            conv.add(ConversationEntry::reminder("a reminder"));

            let provider = MockProvider::new(vec![]);
            let result = conv.compact_with_llm(&provider, &test_model()).await;

            assert!(matches!(result, Ok(None)));
            // Memories and reminders must NOT have been stripped — nothing
            // was summarizable, so the conversation is untouched.
            assert!(conv.entries().iter().any(|e| e.is_memory()));
            assert!(conv.entries().iter().any(|e| e.is_reminder()));
        }

        #[tokio::test]
        async fn happy_path_replaces_body_with_summary() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::memory("old fact", "m1", 0.9));
            conv.add(ConversationEntry::user("hello"));
            conv.add(ConversationEntry::assistant("hi there"));

            // keep_recent_turns=0 so everything is summarized
            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("a brief summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let captured = provider.captured();
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            assert_eq!(
                result.unwrap().map(|r| r.summary),
                Some("a brief summary".to_string())
            );
            assert_eq!(conv.entries().len(), 2);
            assert!(conv.entries()[0].is_system());
            assert!(conv.entries()[1].is_summary());
            assert_eq!(conv.entries()[1].content(), "a brief summary");

            // The captured request must NOT include any memory entries.
            let reqs = captured.lock().unwrap();
            assert_eq!(reqs.len(), 1);
            for m in &reqs[0].messages {
                assert_ne!(m.content, "old fact");
            }
        }

        #[tokio::test]
        async fn preserves_recent_turns_after_summary() {
            let mut conv = Conversation::with_system("sys");
            // Old turns (will be summarized)
            conv.add(ConversationEntry::user("old question 1"));
            conv.add(ConversationEntry::assistant("old answer 1"));
            conv.add(ConversationEntry::user("old question 2"));
            conv.add(ConversationEntry::assistant("old answer 2"));
            // Recent turns (will be kept)
            conv.add(ConversationEntry::user("recent question"));
            conv.add(ConversationEntry::assistant("recent answer"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary of old stuff".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let captured = provider.captured();
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 1)
                .await;

            assert_eq!(
                result.unwrap().map(|r| r.summary),
                Some("summary of old stuff".to_string())
            );
            // system + summary + recent user + recent assistant
            assert_eq!(conv.entries().len(), 4);
            assert!(conv.entries()[0].is_system());
            assert!(conv.entries()[1].is_summary());
            assert!(conv.entries()[2].is_user());
            assert_eq!(conv.entries()[2].content(), "recent question");
            assert!(conv.entries()[3].is_assistant());
            assert_eq!(conv.entries()[3].content(), "recent answer");

            // The recent entries must NOT appear in the summarization request.
            let reqs = captured.lock().unwrap();
            for m in &reqs[0].messages {
                assert_ne!(m.content, "recent question");
                assert_ne!(m.content, "recent answer");
            }
        }

        #[tokio::test]
        async fn keeps_tool_calls_with_recent_turns() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("old msg"));
            conv.add(ConversationEntry::assistant("old reply"));
            // Recent turn with tool calls
            conv.add(ConversationEntry::user("search for X"));
            conv.add(ConversationEntry::assistant_with_tool_calls(
                "searching",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "web_search".into(),
                    arguments: serde_json::json!({"q": "X"}),
                }],
                None,
            ));
            conv.add(ConversationEntry::tool("c1", "found X"));
            conv.add(ConversationEntry::assistant("here are results"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 1)
                .await;

            assert!(result.unwrap().is_some());
            // system + summary + user + assistant(tool_calls) + tool + assistant
            assert_eq!(conv.entries().len(), 6);
            assert!(conv.entries()[2].is_user());
            assert_eq!(conv.entries()[2].content(), "search for X");
            assert!(conv.entries()[4].is_tool());
        }

        // -------------------------------------------------------------------
        // Token-budget keep-recent + Files Touched
        // -------------------------------------------------------------------

        #[tokio::test]
        async fn token_budget_keeps_recent_snapped_to_user_boundary() {
            let mut conv = Conversation::with_system("sys");
            // Two old turns (~10 chars each) then one large recent turn.
            conv.add(ConversationEntry::user("old q one!!"));
            conv.add(ConversationEntry::assistant("old a one!!"));
            conv.add(ConversationEntry::user("old q two!!"));
            conv.add(ConversationEntry::assistant("old a two!!"));
            conv.add(ConversationEntry::user("recent question"));
            conv.add(ConversationEntry::assistant("recent answer"));

            // Budget of 10 tokens ≈ 20 chars (CHARS_PER_TOKEN=2). The newest
            // user+assistant pair (~28 chars) already exceeds it, so only the
            // last turn is kept and it snaps to the "recent question" boundary.
            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let result = conv
                .compact_with_llm_keeping_tokens(&provider, &test_model(), 10)
                .await;

            assert!(result.unwrap().is_some());
            // system + summary + recent user + recent assistant
            assert_eq!(conv.entries().len(), 4);
            assert!(conv.entries()[1].is_summary());
            assert!(conv.entries()[2].is_user());
            assert_eq!(conv.entries()[2].content(), "recent question");
            assert_eq!(conv.entries()[3].content(), "recent answer");
        }

        #[tokio::test]
        async fn token_budget_keeps_everything_when_under_budget() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("q"));
            conv.add(ConversationEntry::assistant("a"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            // Huge budget → nothing to summarize.
            let result = conv
                .compact_with_llm_keeping_tokens(&provider, &test_model(), 1_000_000)
                .await;

            assert!(result.unwrap().is_none());
            assert_eq!(conv.entries().len(), 3);
        }

        fn summary_provider() -> MockProvider {
            MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ])
        }

        /// A turn stopped after a tool round: the model has more to do.
        fn mid_task(conv: &mut Conversation, call_id: &str, output: &str) {
            conv.add(ConversationEntry::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: call_id.into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ));
            conv.add(ConversationEntry::tool(call_id, output));
        }

        #[tokio::test]
        async fn mid_task_compaction_asks_the_model_to_continue_the_request() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("fix the failing test"));
            mid_task(&mut conv, "c1", "test output");

            let result = conv
                .compact_with_llm_keeping_tokens(&summary_provider(), &test_model(), 0)
                .await;

            assert!(result.unwrap().is_some());
            let entries = conv.entries();
            assert_eq!(entries.len(), 3);
            assert!(entries[1].is_summary());
            assert!(entries[2].is_reminder());
            assert!(entries[2].content().starts_with(CONTINUATION_NOTE));
            assert!(entries[2].content().ends_with("fix the failing test"));
        }

        #[tokio::test]
        async fn second_mid_task_compaction_still_repeats_the_request() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("fix the failing test"));
            mid_task(&mut conv, "c1", "test output");
            conv.compact_with_llm_keeping_tokens(&summary_provider(), &test_model(), 0)
                .await
                .unwrap();
            mid_task(&mut conv, "c2", "more output");

            conv.compact_with_llm_keeping_tokens(&summary_provider(), &test_model(), 0)
                .await
                .unwrap();

            let last = conv.entries().last().unwrap();
            assert!(last.is_reminder());
            assert!(last.content().ends_with("fix the failing test"));
            assert_eq!(conv.entries().iter().filter(|e| e.is_reminder()).count(), 1);
        }

        #[tokio::test]
        async fn mid_task_turn_over_the_recent_budget_is_summarized() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("old question"));
            conv.add(ConversationEntry::assistant("old answer"));
            conv.add(ConversationEntry::user("long task"));
            mid_task(&mut conv, "c1", &"x".repeat(200));

            // 10 tokens is about 20 chars; the current turn alone is far over.
            conv.compact_with_llm_keeping_tokens(&summary_provider(), &test_model(), 10)
                .await
                .unwrap();

            let entries = conv.entries();
            assert_eq!(entries.len(), 3);
            assert!(entries[1].is_summary());
            assert!(entries[2].content().ends_with("long task"));
        }

        #[tokio::test]
        async fn finished_turn_over_the_recent_budget_is_kept() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("old question"));
            conv.add(ConversationEntry::assistant("old answer"));
            conv.add(ConversationEntry::user("long task"));
            mid_task(&mut conv, "c1", &"x".repeat(200));
            conv.add(ConversationEntry::assistant("done"));

            conv.compact_with_llm_keeping_tokens(&summary_provider(), &test_model(), 10)
                .await
                .unwrap();

            let entries = conv.entries();
            assert_eq!(entries[2].content(), "long task");
            assert!(!entries.iter().any(ConversationEntry::is_reminder));
        }

        #[tokio::test]
        async fn summary_appends_files_touched() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("edit the config"));
            conv.add(ConversationEntry::assistant_with_tool_calls(
                "reading then writing",
                vec![
                    ToolCall {
                        id: "c1".into(),
                        name: "file_read".into(),
                        arguments: serde_json::json!({"path": "src/lib.rs"}),
                    },
                    ToolCall {
                        id: "c2".into(),
                        name: "str_replace".into(),
                        arguments: serde_json::json!({"file_path": "src/main.rs"}),
                    },
                ],
                None,
            ));
            conv.add(ConversationEntry::tool("c1", "contents"));
            conv.add(ConversationEntry::tool("c2", "ok"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("did the edit".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let summary = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await
                .unwrap()
                .unwrap()
                .summary;

            assert!(summary.contains("## Files Touched"), "got: {summary}");
            assert!(
                summary.contains("### Modified\n- src/main.rs"),
                "got: {summary}"
            );
            assert!(summary.contains("### Read\n- src/lib.rs"), "got: {summary}");
        }

        #[test]
        fn files_touched_block_classifies_and_dedupes() {
            let read_then_modified = ConversationEntry::assistant_with_tool_calls(
                "",
                vec![
                    ToolCall {
                        id: "a".into(),
                        name: "grep".into(),
                        arguments: serde_json::json!({"path": "a.rs"}),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "file_read".into(),
                        arguments: serde_json::json!({"path": "b.rs"}),
                    },
                    // b.rs is later modified → should only show under Modified.
                    ToolCall {
                        id: "c".into(),
                        name: "file_write".into(),
                        arguments: serde_json::json!({"path": "b.rs"}),
                    },
                ],
                None,
            );
            let entries = [&read_then_modified];
            let block = files_touched_block(&entries).expect("some files");
            assert!(block.contains("### Modified\n- b.rs"), "got: {block}");
            assert!(block.contains("### Read\n- a.rs"), "got: {block}");
            // b.rs must not also appear under Read.
            assert_eq!(block.matches("b.rs").count(), 1, "got: {block}");

            // No file ops → None.
            let plain = ConversationEntry::assistant("just text");
            assert!(files_touched_block(&[&plain]).is_none());
        }

        #[tokio::test]
        async fn all_recent_returns_none() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("q1"));
            conv.add(ConversationEntry::assistant("a1"));

            let provider = MockProvider::new(vec![]);
            // keep_recent_turns=3 but only 1 turn exists — nothing to summarize
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 3)
                .await;

            assert!(matches!(result, Ok(None)));
            // Conversation must be untouched
            assert_eq!(conv.entries().len(), 3);
        }

        #[tokio::test]
        async fn iterative_summary_uses_iterative_prompt() {
            let mut conv = Conversation::with_system("sys");
            // Previous summary from earlier compaction
            conv.add(ConversationEntry::summary("old summary content"));
            conv.add(ConversationEntry::user("new question"));
            conv.add(ConversationEntry::assistant("new answer"));
            conv.add(ConversationEntry::user("another question"));
            conv.add(ConversationEntry::assistant("another answer"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("updated summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let captured = provider.captured();
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 1)
                .await;

            assert_eq!(
                result.unwrap().map(|r| r.summary),
                Some("updated summary".to_string())
            );
            // The system prompt sent to the compaction LLM should be the iterative variant
            let reqs = captured.lock().unwrap();
            assert!(
                reqs[0].messages[0].content.contains("Update the existing"),
                "should use iterative compaction prompt when prior summary exists"
            );
        }

        #[tokio::test]
        async fn llm_error_leaves_conversation_untouched() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::memory("keep me", "m1", 0.9));
            conv.add(ConversationEntry::user("hi"));
            conv.add(ConversationEntry::assistant("hello"));

            let provider = MockProvider::new(vec![Err("boom".into())]);
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            assert!(result.is_err());
            // Memories and reminders must survive a failed compaction.
            assert!(conv.entries().iter().any(|e| e.is_memory()));
            assert!(conv.entries().iter().any(|e| e.is_user()));
            assert!(conv.entries().iter().any(|e| e.is_assistant()));
        }

        #[tokio::test]
        async fn empty_summary_returns_ok_none_and_doesnt_mutate() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::memory("keep me", "m1", 0.9));
            conv.add(ConversationEntry::user("hi"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("   ".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let result = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            assert!(matches!(result, Ok(None)));
            assert!(conv.entries().iter().any(|e| e.is_memory()));
        }

        #[tokio::test]
        async fn assistant_tool_calls_are_stripped_from_request() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("run something"));
            conv.add(ConversationEntry::assistant_with_tool_calls(
                "ok working",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({"cmd": "ls"}),
                }],
                None,
            ));
            conv.add(ConversationEntry::tool("c1", "file1\nfile2"));
            conv.add(ConversationEntry::assistant("done"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let captured = provider.captured();
            let _ = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            let reqs = captured.lock().unwrap();
            assert_eq!(reqs.len(), 1);
            for m in &reqs[0].messages {
                assert!(
                    m.tool_calls.is_none(),
                    "compaction request must not include tool_calls; declared tools is empty"
                );
                assert_ne!(
                    m.role,
                    Role::Tool,
                    "tool-role messages get rejected without a paired tool_call; should be rewritten"
                );
            }
            // The assistant message keeps its content + a tool-call annotation.
            assert!(reqs[0].messages.iter().any(|m| m.role == Role::Assistant
                && m.content.contains("ok working")
                && m.content.contains("[tool calls: exec]")));
            // The tool result becomes a developer message.
            assert!(
                reqs[0]
                    .messages
                    .iter()
                    .any(|m| m.role == Role::Developer && m.content.contains("[tool result]"))
            );
        }

        #[tokio::test]
        async fn trailing_user_does_not_produce_consecutive_user_messages() {
            // Merge the instruction into the final user message so it stays
            // a user turn without adding adjacent user messages.
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("hi"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            let captured = provider.captured();
            let _ = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            let reqs = captured.lock().unwrap();
            let msgs = &reqs[0].messages;
            assert_eq!(msgs.last().unwrap().role, Role::User);
            assert!(msgs.last().unwrap().content.starts_with("hi\n\n"));
            // No two adjacent user messages.
            for pair in msgs.windows(2) {
                assert!(
                    !(pair[0].role == Role::User && pair[1].role == Role::User),
                    "adjacent user messages would be rejected by Anthropic"
                );
            }
        }

        #[tokio::test]
        async fn summary_request_ends_with_a_user_turn_after_assistant_output() {
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("question"));
            conv.add(ConversationEntry::assistant("answer"));
            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]);
            conv.compact_with_llm_keeping(&provider, &test_model(), 0)
                .await
                .unwrap();
            let requests = provider.captured();
            let requests = requests.lock().unwrap();
            let messages = &requests[0].messages;
            assert_eq!(messages.last().unwrap().role, Role::User);
            assert!(
                messages
                    .last()
                    .unwrap()
                    .content
                    .contains("Now produce the summary")
            );
            assert!(
                messages
                    .iter()
                    .any(|message| message.role == Role::Assistant && message.content == "answer")
            );
        }

        #[tokio::test]
        async fn budget_cut_drops_orphan_tool_results() {
            // Context window of 4_000 floors max_input_chars at MIN_INPUT_CHARS
            // (4_096). The reverse walk should cut between the assistant
            // (large enough to overflow) and the tool result that follows it.
            // The orphan-drop pass must then remove the now-unpaired tool.
            let big_assistant = "y".repeat(5_000);
            let mut conv = Conversation::with_system("sys");
            conv.add(ConversationEntry::user("ancient")); // would be cut
            conv.add(ConversationEntry::assistant_with_tool_calls(
                big_assistant,
                vec![ToolCall {
                    id: "c1".into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({}),
                }],
                None,
            )); // overflow point — gets cut
            conv.add(ConversationEntry::tool("c1", "result")); // orphan after cut
            conv.add(ConversationEntry::user("recent prompt"));
            conv.add(ConversationEntry::assistant("recent reply"));

            let provider = MockProvider::new(vec![
                Ok(StreamEvent::ContentDelta("summary".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ])
            .with_context_window(4_000);
            let captured = provider.captured();
            let _ = conv
                .compact_with_llm_keeping(&provider, &test_model(), 0)
                .await;

            let reqs = captured.lock().unwrap();
            let msgs = &reqs[0].messages;
            // We rewrite Tool→Developer in the request, so a leftover orphan
            // would show up as a Developer message with "[tool result]". The
            // orphan-drop pass must remove it.
            let orphans: Vec<_> = msgs
                .iter()
                .filter(|m| m.content.starts_with("[tool result]"))
                .collect();
            assert!(
                orphans.is_empty(),
                "orphan tool results must be dropped after the budget cut, got: {orphans:?}"
            );
        }
    }
}
