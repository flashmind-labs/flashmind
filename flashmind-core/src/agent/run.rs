//! Shared turn policy with pause points for storage and interactive callers.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use flashmind_types::{
    AgentEvent, AgentInput, AgentStream, CompactionReason, Outcome, ToolCall, TurnResult,
    TurnStatus, TurnUsage,
};
use tokio::time::Instant as AsyncInstant;

use super::{Agent, MAX_CONTINUATIONS, check_iteration_limits};
use crate::{CancellationToken, Conversation, ConversationEntry, RecoveryAttempts};

/// Conversation changes a caller can persist before advancing the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checkpoint {
    /// A model response has been recorded, before its tools can execute.
    Assistant,
    /// A tool round has finished, before compaction or another model request.
    Tools,
    /// Proactive compaction has finished.
    Compaction,
    /// Error recovery has finished, before retrying the model request.
    Recovery,
}

/// The runtime pauses without borrowing the agent or conversation.
pub enum RunBoundary {
    /// Update application context before the next model request.
    BeforeModel,
    /// Persist the conversation before calling `advance` again.
    Checkpoint(Checkpoint),
    /// Execute these calls and record their results before advancing.
    /// Interactive callers may handle approvals here.
    Tools(Vec<ToolCall>),
    Done(String),
}

enum Phase {
    Ready,
    Model,
    Response(TurnResult),
    AfterTools(ToolRound),
    Compact {
        reason: CompactionReason,
        finish: bool,
    },
    Done,
}

struct ToolRound {
    usage: TurnUsage,
    entries_before: usize,
    working_dir: Option<PathBuf>,
}

/// Per-run scheduling, reminders, recovery, and compaction shared by all callers.
///
/// Drive each `advance` stream to exhaustion, then handle its boundary. A
/// checkpoint releases the conversation so storage can finish before the next
/// model request or tool execution. Input queues remain the caller's concern.
pub struct AgentRun {
    phase: Phase,
    recovery: RecoveryAttempts,
    max_iterations: Option<usize>,
    iteration: usize,
    consecutive_compactions: u8,
    continuations: u8,
    tool_rounds: usize,
    empty_response: bool,
    content: String,
    last_update: AsyncInstant,
    started: Instant,
    finished: bool,
    failed: bool,
}

impl AgentRun {
    pub(super) fn content(&self) -> &str {
        &self.content
    }

    /// Start a run, adding its input to the caller's conversation.
    pub fn new(
        conversation: &mut Conversation,
        input: AgentInput,
        max_iterations: Option<usize>,
    ) -> Self {
        if let AgentInput::User {
            content,
            context,
            parts,
        } = input
        {
            if let Some(context) = context {
                conversation.add(ConversationEntry::system_message(context));
            }
            match parts {
                Some(parts) if !parts.is_empty() => {
                    conversation.add(ConversationEntry::user_with_parts(content, parts));
                }
                _ => conversation.add(ConversationEntry::user(content)),
            }
        }
        conversation.mark_turn_start();
        metrics::counter!("agent.turns_started").increment(1);
        Self {
            phase: Phase::Ready,
            recovery: RecoveryAttempts::default(),
            max_iterations,
            iteration: 0,
            consecutive_compactions: 0,
            continuations: 0,
            tool_rounds: 0,
            empty_response: false,
            content: String::new(),
            last_update: AsyncInstant::now(),
            started: Instant::now(),
            finished: false,
            failed: false,
        }
    }

    /// Stream work until the next boundary. The caller executes returned tools
    /// and records their results before advancing again.
    pub fn advance<'a>(
        &'a mut self,
        agent: &'a mut Agent,
        conversation: &'a mut Conversation,
        cancel: &'a CancellationToken,
    ) -> AgentStream<'a, AgentEvent, anyhow::Result<RunBoundary>> {
        AgentStream::new(async_stream::stream! {
            loop {
                if cancel.is_cancelled() {
                    self.failed = true;
                    yield Outcome::Done(Err(anyhow::anyhow!("cancelled")));
                    return;
                }
                match std::mem::replace(&mut self.phase, Phase::Done) {
                    Phase::Ready => {
                        self.phase = Phase::Model;
                        yield Outcome::Done(Ok(RunBoundary::BeforeModel));
                        return;
                    }
                    Phase::Model => {
                        self.iteration += 1;
                        metrics::counter!("agent.iterations").increment(1);
                        let warn = self.max_iterations.map(|max| (max as f64 * 0.9).ceil() as usize);
                        if let Err(error) = check_iteration_limits(conversation, self.max_iterations, warn, self.iteration) {
                            self.failed = true;
                            yield Outcome::Done(Err(error));
                            return;
                        }
                        self.refresh_reminders(agent, conversation);
                        let mut partial_content = String::new();
                        let mut partial_reasoning = String::new();
                        let result = {
                            let mut turn = agent.run_turn(conversation, cancel);
                            while let Some(event) = turn.next().await {
                                match &event {
                                    AgentEvent::TextDelta(text) => {
                                        partial_content.push_str(text);
                                        if !text.trim().is_empty() {
                                            self.last_update = AsyncInstant::now();
                                        }
                                    }
                                    AgentEvent::ReasoningDelta(text) => partial_reasoning.push_str(text),
                                    _ => {}
                                }
                                yield Outcome::Item(event);
                            }
                            turn.take_result().unwrap_or_else(|| Err(anyhow::anyhow!("Model stream ended without a result")))
                        };
                        if result.is_ok() {
                            self.recovery.record_success();
                        } else if cancel.is_cancelled() && (!partial_content.is_empty() || !partial_reasoning.is_empty()) {
                            conversation.add(ConversationEntry::assistant_with_reasoning(
                                partial_content,
                                (!partial_reasoning.is_empty()).then_some(partial_reasoning),
                            ));
                        }
                        self.phase = Phase::Response(result);
                        yield Outcome::Done(Ok(RunBoundary::Checkpoint(Checkpoint::Assistant)));
                        return;
                    }
                    Phase::Response(Err(error)) => {
                        let recovered = {
                            let mut recovery = agent.recover(conversation, cancel, &error, &mut self.recovery);
                            while let Some(event) = recovery.next().await {
                                yield Outcome::Item(event);
                            }
                            recovery.take_result().unwrap_or(false)
                        };
                        if !recovered {
                            self.failed = true;
                            yield Outcome::Done(Err(error));
                            return;
                        }
                        self.phase = Phase::Ready;
                        yield Outcome::Done(Ok(RunBoundary::Checkpoint(Checkpoint::Recovery)));
                        return;
                    }
                    Phase::Response(Ok(status)) => {
                        let (content, _) = status.content_and_usage();
                        if !content.trim().is_empty() {
                            self.content = content.to_owned();
                        }
                        match status {
                            TurnStatus::ToolCalls { content, tool_calls, usage } => {
                                if !content.trim().is_empty() {
                                    self.empty_response = false;
                                }
                                self.content = content;
                                self.consecutive_compactions = 0;
                                self.phase = Phase::AfterTools(ToolRound {
                                    usage,
                                    entries_before: conversation.entries().len(),
                                    working_dir: agent.working_dir.clone(),
                                });
                                yield Outcome::Done(Ok(RunBoundary::Tools(tool_calls)));
                                return;
                            }
                            TurnStatus::Continue { .. } if self.continuations < MAX_CONTINUATIONS => {
                                self.continuations += 1;
                                self.consecutive_compactions = 0;
                                self.phase = Phase::Ready;
                            }
                            TurnStatus::Done { content, .. } if content.trim().is_empty() && !self.empty_response => {
                                self.empty_response = true;
                                self.phase = Phase::Ready;
                            }
                            TurnStatus::CompactionNeeded { reason, .. } => {
                                self.consecutive_compactions += 1;
                                if self.consecutive_compactions > 2 {
                                    self.failed = true;
                                    yield Outcome::Done(Err(anyhow::anyhow!("Context window is too small for this task: repeated compaction without progress")));
                                    return;
                                }
                                if agent.auto_compact {
                                    self.phase = Phase::Compact {
                                        reason,
                                        finish: matches!(reason, CompactionReason::ContextThreshold(_)),
                                    };
                                } else {
                                    yield Outcome::Item(AgentEvent::CompactionNeeded { reason });
                                }
                            }
                            _ => {}
                        }
                    }
                    Phase::AfterTools(round) => {
                        if agent.working_dir != round.working_dir
                            && let Some(path) = agent.working_dir()
                        {
                            conversation.add(ConversationEntry::system_message(format!(
                                "The working directory is now {}. Subsequent commands and file tools use this directory.", path.display(),
                            )));
                        }
                        self.tool_rounds += 1;
                        let added = conversation.entries().get(round.entries_before..).unwrap_or_default();
                        self.phase = match agent.compaction_due_after_tools(&round.usage, added) {
                            Some(tokens) => Phase::Compact {
                                reason: CompactionReason::ContextThreshold(tokens),
                                finish: false,
                            },
                            None => Phase::Ready,
                        };
                        yield Outcome::Done(Ok(RunBoundary::Checkpoint(Checkpoint::Tools)));
                        return;
                    }
                    Phase::Compact { reason, finish } => {
                        let (model, provider) = agent.compaction_model_and_provider();
                        let tokens = match reason {
                            CompactionReason::ContextThreshold(tokens) => tokens,
                            CompactionReason::OutputLength => agent.context_window,
                        };
                        let stream = crate::compaction::try_compact(
                            conversation, tokens, agent.context_window,
                            agent.keep_recent_tokens, &*provider, &model,
                        );
                        futures_util::pin_mut!(stream);
                        use futures_util::StreamExt;
                        while let Some(event) = stream.next().await {
                            yield Outcome::Item(event);
                        }
                        self.phase = if finish { Phase::Done } else { Phase::Ready };
                        yield Outcome::Done(Ok(RunBoundary::Checkpoint(Checkpoint::Compaction)));
                        return;
                    }
                    Phase::Done => {
                        self.finish(conversation);
                        yield Outcome::Done(Ok(RunBoundary::Done(self.content.clone())));
                        return;
                    }
                }
            }
        })
    }

    /// Clean transient context after completion, interruption, or an error.
    pub fn finish(&mut self, conversation: &mut Conversation) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.phase = Phase::Done;
        conversation.strip_agent_progress();
        conversation.strip_memories();
        conversation.strip_reminders();
        metrics::gauge!("agent.conversation.entries").set(conversation.entries().len() as f64);
        metrics::histogram!("agent.turn.duration_seconds")
            .record(self.started.elapsed().as_secs_f64());
        if self.failed {
            metrics::counter!("agent.turns_errors").increment(1);
        } else {
            metrics::counter!("agent.turns_completed").increment(1);
        }
    }

    fn refresh_reminders(&mut self, agent: &Agent, conversation: &mut Conversation) {
        if self.tool_rounds >= 10
            && let Some(reminder) = &agent.periodic_reminder
        {
            conversation.refresh_periodic_reminder(reminder);
            self.tool_rounds = 0;
        }
        if self.last_update.elapsed() >= Duration::from_secs(300)
            && let Some(reminder) = &agent.user_update_reminder
        {
            conversation.refresh_periodic_reminder(reminder);
            self.last_update = AsyncInstant::now();
        }
    }
}
