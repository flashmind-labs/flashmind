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

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use futures::StreamExt;
use rust_decimal::dec;
use serde::{Deserialize, Serialize};

use flashmind_types::{
    CompletionRequest, ContentPart, LlmProvider, Message, Model, ReasoningLevel, SamplingParams,
    StreamEvent, ToolCall,
};

use crate::compaction::COMPACTION_PROMPT;

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
    /// subagent progress, and compaction summaries. Differentiated by `tag`.
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
    },
    /// Result returned from a tool execution.
    Tool { call_id: String, output: String },
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
            },
            timestamp: Utc::now(),
        }
    }

    /// Create an assistant response that includes tool call requests.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            kind: EntryKind::Assistant {
                content: content.into(),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            timestamp: Utc::now(),
        }
    }

    /// Create a tool execution result, linked to its originating call by `call_id`.
    pub fn tool(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Tool {
                call_id: call_id.into(),
                output: output.into(),
            },
            timestamp: Utc::now(),
        }
    }

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

    pub fn subagent_progress(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            kind: EntryKind::Developer {
                content: content.into(),
                tag: Some("subagent_progress".into()),
                metadata: Some(serde_json::json!({"id": id.into()})),
            },
            timestamp: Utc::now(),
        }
    }

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

    pub fn role(&self) -> &'static str {
        match &self.kind {
            EntryKind::SystemPrompt(_) => "system",
            EntryKind::Developer { .. } => "developer",
            EntryKind::User { .. } => "user",
            EntryKind::Assistant { .. } => "assistant",
            EntryKind::Tool { .. } => "tool",
        }
    }

    pub fn to_message(&self) -> Message {
        match &self.kind {
            EntryKind::SystemPrompt(text) => Message::system(text),
            EntryKind::Developer { content, tag, metadata } => {
                if tag.as_deref() == Some("subagent_progress") {
                    let id = metadata
                        .as_ref()
                        .and_then(|m| m["id"].as_str())
                        .unwrap_or("?");
                    Message::developer(format!("[Subagent {id} progress]\n{content}"))
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
            } => {
                if let Some(tc) = tool_calls {
                    Message::assistant_with_tool_calls(content, tc.clone())
                } else {
                    Message::assistant(content)
                }
            }
            EntryKind::Tool { call_id, output } => Message::tool_result(call_id, output),
        }
    }

    pub fn content(&self) -> &str {
        match &self.kind {
            EntryKind::SystemPrompt(s)
            | EntryKind::Developer { content: s, .. }
            | EntryKind::User { content: s, .. }
            | EntryKind::Assistant { content: s, .. } => s,
            EntryKind::Tool { output, .. } => output,
        }
    }

    pub fn set_content(&mut self, new: String) {
        match &mut self.kind {
            EntryKind::SystemPrompt(s)
            | EntryKind::Developer { content: s, .. }
            | EntryKind::User { content: s, .. }
            | EntryKind::Assistant { content: s, .. } => *s = new,
            EntryKind::Tool { output, .. } => *output = new,
        }
    }

    pub fn is_system(&self) -> bool {
        matches!(self.kind, EntryKind::SystemPrompt(_))
    }

    pub fn is_developer(&self) -> bool {
        matches!(self.kind, EntryKind::Developer { .. })
    }

    fn has_tag(&self, t: &str) -> bool {
        matches!(&self.kind, EntryKind::Developer { tag: Some(tag), .. } if tag == t)
    }

    pub fn is_system_message(&self) -> bool {
        matches!(&self.kind, EntryKind::Developer { tag: None, .. })
    }

    pub fn is_user(&self) -> bool {
        matches!(self.kind, EntryKind::User { .. })
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self.kind, EntryKind::Assistant { .. })
    }

    pub fn is_tool(&self) -> bool {
        matches!(self.kind, EntryKind::Tool { .. })
    }

    pub fn is_memory(&self) -> bool {
        self.has_tag("memory")
    }

    pub fn is_reminder(&self) -> bool {
        self.has_tag("reminder")
    }

    pub fn is_summary(&self) -> bool {
        self.has_tag("summary")
    }

    pub fn is_subagent_progress(&self) -> bool {
        self.has_tag("subagent_progress")
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
/// | `Developer` | `developer` | System messages, reminders, memories, summaries, subagent progress (differentiated by `tag`) |
/// | `User` | `user` | Plain text or multimodal parts |
/// | `Assistant` | `assistant` | May include tool calls |
/// | `Tool` | `tool` | Linked to assistant by `call_id` |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    entries: Vec<ConversationEntry>,
}

impl Conversation {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
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

    /// Insert an entry at position 0.
    pub fn prepend(&mut self, entry: ConversationEntry) {
        self.entries.insert(0, entry);
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
        let mut call_ids: HashSet<String> = HashSet::new();
        let mut result_ids: HashSet<String> = HashSet::new();

        for entry in &self.entries {
            match &entry.kind {
                EntryKind::Assistant {
                    tool_calls: Some(calls),
                    ..
                } => {
                    for tc in calls {
                        call_ids.insert(tc.id.clone());
                    }
                }
                EntryKind::Tool { call_id, .. } => {
                    result_ids.insert(call_id.clone());
                }
                _ => {}
            }
        }

        let paired: HashSet<&String> = call_ids.intersection(&result_ids).collect();

        // Strip unpaired tool_calls from assistant entries
        for entry in &mut self.entries {
            if let EntryKind::Assistant {
                content,
                tool_calls: Some(calls),
            } = &mut entry.kind
            {
                calls.retain(|tc| paired.contains(&tc.id));
                if calls.is_empty() {
                    let content = std::mem::take(content);
                    entry.kind = EntryKind::Assistant {
                        content,
                        tool_calls: None,
                    };
                }
            }
        }

        // Remove orphan tool-result entries
        self.entries.retain(
            |e| !matches!(&e.kind, EntryKind::Tool { call_id, .. } if !paired.contains(call_id)),
        );

        // Enforce adjacency: tool results must immediately follow their
        // assistant entry. Pull all Tool entries out, then re-insert each
        // group right after the assistant that owns them.
        let tool_entries: Vec<ConversationEntry> = self
            .entries
            .iter()
            .filter(|e| matches!(&e.kind, EntryKind::Tool { .. }))
            .cloned()
            .collect();

        if tool_entries.is_empty() {
            return;
        }

        // Remove all tool entries from the list
        self.entries
            .retain(|e| !matches!(&e.kind, EntryKind::Tool { .. }));

        // Re-insert tool results right after their assistant
        let mut insert_at: Vec<(usize, Vec<ConversationEntry>)> = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            if let EntryKind::Assistant {
                tool_calls: Some(calls),
                ..
            } = &entry.kind
            {
                let ids: HashSet<&str> = calls.iter().map(|tc| tc.id.as_str()).collect();
                let matching: Vec<ConversationEntry> = tool_entries
                    .iter()
                    .filter(|e| {
                        matches!(&e.kind, EntryKind::Tool { call_id, .. } if ids.contains(call_id.as_str()))
                    })
                    .cloned()
                    .collect();

                if !matching.is_empty() {
                    insert_at.push((i + 1, matching));
                }
            }
        }

        // Insert in reverse order so indices stay valid
        for (pos, entries) in insert_at.into_iter().rev() {
            for (j, entry) in entries.into_iter().enumerate() {
                self.entries.insert(pos + j, entry);
            }
        }
    }

    /// Remove all Memory entries.
    pub fn strip_memories(&mut self) {
        self.entries.retain(|e| !e.is_memory());
    }

    /// Remove all Reminder entries.
    pub fn strip_reminders(&mut self) {
        self.entries.retain(|e| !e.is_reminder());
    }

    /// Remove all SubagentProgress entries.
    pub fn strip_subagent_progress(&mut self) {
        self.entries.retain(|e| !e.is_subagent_progress());
    }

    /// Strip existing reminders, then push a new Reminder entry.
    pub fn replace_reminder(&mut self, content: impl Into<String>) {
        self.strip_reminders();
        self.add(ConversationEntry::reminder(content));
    }

    pub fn replace_subagent_progress(&mut self, id: impl Into<String>, content: impl Into<String>) {
        let id = id.into();
        self.entries.retain(|e| {
            !matches!(
                &e.kind,
                EntryKind::Developer { tag: Some(t), metadata: Some(m), .. }
                if t == "subagent_progress" && m["id"].as_str() == Some(&id)
            )
        });
        self.add(ConversationEntry::subagent_progress(id, content));
    }

    pub fn memory_ids(&self) -> HashSet<&str> {
        let mut set = HashSet::new();
        for entry in &self.entries {
            if let EntryKind::Developer { tag: Some(t), metadata: Some(m), .. } = &entry.kind
                && t == "memory"
                && let Some(id) = m["id"].as_str()
                && !id.is_empty()
            {
                set.insert(id);
            }
        }
        set
    }

    pub fn strip_low_score_memories(&mut self, keep: usize) {
        let mut memory_entries: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let EntryKind::Developer { tag: Some(t), metadata: Some(m), .. } = &e.kind
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
    /// Each [`ConversationEntry`] is mapped 1:1 according to its [`EntryKind`]:
    /// system prompts become `Role::System`, user turns become `Role::User`,
    /// summaries and injected context become `Role::Developer`, etc.
    ///
    /// Call [`sanitize`](Self::sanitize) before this if tool call/result pairing
    /// might be inconsistent (e.g. after session load with partial writes).
    pub fn to_messages(&self) -> Vec<Message> {
        let mut msgs = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            msgs.push(entry.to_message());
        }

        msgs
    }

    // -- Compaction --------------------------------------------------------

    /// Compact the conversation by summarizing entries via an LLM call.
    ///
    /// # Process
    ///
    /// 1. Strips all Memory and Reminder entries (they are ephemeral)
    /// 2. Keeps the system prompt intact (if present)
    /// 3. Sends remaining entries to the LLM with the [`COMPACTION_PROMPT`]
    /// 4. Replaces the body of the conversation with a single summary `Developer` entry
    ///
    /// The content sent to the compaction model is capped at ~100K characters,
    /// dropping the oldest entries first if necessary.
    ///
    /// Returns the summary text if compaction succeeded, `None` if skipped
    /// (nothing to summarize) or failed (LLM returned empty or errored).
    pub async fn compact_with_llm(
        &mut self,
        provider: &dyn LlmProvider,
        model: &Model,
    ) -> Option<String> {
        // Strip memories and reminders before summarizing
        self.strip_memories();
        self.strip_reminders();

        let has_system = self.entries.first().is_some_and(|e| e.is_system());
        let prefix = if has_system { 1 } else { 0 };
        let total = self.entries.len();

        if total <= prefix {
            return None;
        }

        let to_summarize = &self.entries[prefix..total];
        if to_summarize.is_empty() {
            return None;
        }

        let conv_entries: Vec<&ConversationEntry> = to_summarize
            .iter()
            .filter(|e| {
                !matches!(
                    e.kind,
                    EntryKind::SystemPrompt(_) | EntryKind::Developer { .. }
                ) && !e.content().is_empty()
            })
            .collect();

        if conv_entries.is_empty() {
            return None;
        }

        // Estimate budget: ~4 chars/token, reserve 4K tokens for system + response.
        // Cap the conversation content sent to the compaction model.
        const MAX_CHARS: usize = 100_000;
        let mut total_chars = 0usize;
        let mut start_idx = 0;

        for (i, entry) in conv_entries.iter().enumerate().rev() {
            total_chars += entry.content().len();
            if total_chars > MAX_CHARS {
                start_idx = i + 1;
                break;
            }
        }

        let entries_for_llm = &conv_entries[start_idx..];
        if entries_for_llm.is_empty() {
            tracing::warn!("All entries exceed compaction budget, skipping LLM summary");
            return None;
        }

        if start_idx > 0 {
            tracing::info!(
                "Compaction: trimmed {} oldest entries to fit compaction model context",
                start_idx
            );
        }

        // System prompt, the conversation window, then a user prompt to trigger summarization.
        let mut messages = vec![Message::system(COMPACTION_PROMPT)];
        for entry in entries_for_llm {
            messages.push(entry.to_message());
        }
        messages.push(Message::user(
            "Summarize the conversation above for context continuity.",
        ));

        let request = CompletionRequest {
            model: model.clone(),
            messages,
            tools: vec![],
            max_tokens: Some(12_000),
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.3)),
                ..Default::default()
            },
            modalities: vec![],
            audio_config: None,
            image_config: None,
        };

        tracing::info!(
            model = %model,
            entries = entries_for_llm.len(),
            messages = request.messages.len(),
            "Compaction: sending LLM request"
        );

        let mut stream = provider.complete(request);
        let mut summary = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(StreamEvent::ContentDelta(delta)) => summary.push_str(&delta),
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("Compaction LLM stream error: {e}");
                }
            }
        }

        if summary.is_empty() {
            tracing::warn!("Compaction LLM call returned empty summary, skipping");
            return None;
        }

        tracing::info!(
            "Compacted {} entries into summary ({} chars)",
            to_summarize.len(),
            summary.len()
        );

        // Rebuild: system (if any) + summary entry
        let mut new_entries = Vec::new();
        if has_system {
            new_entries.push(self.entries[0].clone());
        }
        new_entries.push(ConversationEntry::summary(summary.clone()));

        self.entries = new_entries;
        Some(summary)
    }

    /// Prune tool result outputs, replacing them with `[output pruned]`.
    /// Preserves the last `keep_recent` tool results intact.
    /// Returns the number of entries pruned.
    pub fn prune_tool_outputs(&mut self, keep_recent: usize) -> usize {
        let tool_indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_tool())
            .map(|(i, _)| i)
            .collect();

        if tool_indices.len() <= keep_recent {
            return 0;
        }

        let to_prune = &tool_indices[..tool_indices.len() - keep_recent];
        let mut pruned = 0;

        for &i in to_prune {
            if self.entries[i].content() != "[output pruned]" {
                self.entries[i].set_content("[output pruned]".to_string());
                pruned += 1;
            }
        }

        pruned
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
            if let EntryKind::User {
                parts: Some(parts), ..
            } = &mut entry.kind
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
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Remove all tool-related entries: Tool entries and tool_calls from
    /// Assistant entries. Returns the number of entries removed.
    pub fn strip_tool_messages(&mut self) -> usize {
        let before = self.entries.len();

        self.entries.retain(|e| !e.is_tool());

        for entry in &mut self.entries {
            if let EntryKind::Assistant { tool_calls, .. } = &mut entry.kind {
                *tool_calls = None;
            }
        }

        before - self.entries.len()
    }

    /// Keep only the system prompt and the last user + assistant entries.
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
        }
        if let Some(asst) = last_assistant {
            self.entries.push(asst);
        }
    }

    /// Hard-reset the conversation to just the system prompt (if any) and the
    /// last user entry.
    pub fn truncate_to_latest(&mut self) {
        let system = self.entries.iter().find(|e| e.is_system()).cloned();
        let last_user = self.entries.iter().rev().find(|e| e.is_user()).cloned();

        self.entries.clear();

        if let Some(sys) = system {
            self.entries.push(sys);
        }
        if let Some(user) = last_user {
            self.entries.push(user);
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
    fn test_prune_tool_outputs() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::system("sys"));
        conv.add(ConversationEntry::user("do stuff"));
        conv.add(ConversationEntry::assistant("calling tools"));
        conv.add(ConversationEntry::tool("c1", "long output 1"));
        conv.add(ConversationEntry::tool("c2", "long output 2"));
        conv.add(ConversationEntry::tool("c3", "long output 3"));
        conv.add(ConversationEntry::tool("c4", "long output 4"));
        conv.add(ConversationEntry::tool("c5", "long output 5"));

        let pruned = conv.prune_tool_outputs(2);
        assert_eq!(pruned, 3);

        // First 3 tool results pruned, last 2 kept
        assert_eq!(conv.entries()[3].content(), "[output pruned]");
        assert_eq!(conv.entries()[4].content(), "[output pruned]");
        assert_eq!(conv.entries()[5].content(), "[output pruned]");
        assert_eq!(conv.entries()[6].content(), "long output 4");
        assert_eq!(conv.entries()[7].content(), "long output 5");
    }

    #[test]
    fn test_prune_tool_outputs_fewer_than_keep() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::tool("c1", "output 1"));
        conv.add(ConversationEntry::tool("c2", "output 2"));

        let pruned = conv.prune_tool_outputs(4);
        assert_eq!(pruned, 0);
        assert_eq!(conv.entries()[0].content(), "output 1");
    }

    #[test]
    fn test_prune_tool_outputs_idempotent() {
        let mut conv = Conversation::new();
        conv.add(ConversationEntry::tool("c1", "output"));
        conv.add(ConversationEntry::tool("c2", "keep"));

        let pruned1 = conv.prune_tool_outputs(1);
        assert_eq!(pruned1, 1);

        let pruned2 = conv.prune_tool_outputs(1);
        assert_eq!(pruned2, 0);
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
}
