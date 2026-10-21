//! Agent runtime: the main loop that drives LLM calls, tool execution, and conversation management.
//!
//! The [`Agent`] struct is the core of the framework. It orchestrates:
//! - Streaming LLM responses via [`Agent::start`]
//! - Tool call execution and result injection
//! - Context window management through automatic compaction
//! - Subagent spawning and message routing
//!
//! # Building an Agent
//!
//! ```rust,ignore
//! let mut agent = Agent::builder(provider)
//!     .tools(tools)
//!     .build()
//!     .await;
//! ```
//!
//! # Agent Loop Flow
//!
//! 1. Stream LLM response → emit `AgentEvent`s
//! 2. Execute any tool calls from the response
//! 3. Compact conversation if context pressure detected
//! 4. Repeat until done or max iterations reached

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use rust_decimal::Decimal;
use tokio::sync::{Mutex as AsyncMutex, mpsc::UnboundedReceiver};
use tokio_util::sync::CancellationToken;

use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, AgentStream, AliasedModel, CompactionReason,
    FinishReason, LlmProvider, Model, ModelCapabilities, Outcome, ReasoningLevel, SamplingParams,
    ToolRegistry, TurnResult, TurnStatus, TurnUsage,
};

use crate::RecoveryAttempts;
use crate::conversation::{Conversation, ConversationEntry};
use crate::streaming::stream_llm_response;

/// Default context window assumed when the provider doesn't report one.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

/// Default tokens kept free for the model's reply when deciding to compact.
/// Compaction triggers once prompt tokens exceed `context_window - reserve_tokens`.
pub const DEFAULT_RESERVE_TOKENS: u32 = 16_384;

/// Default recent-history budget (in tokens) preserved verbatim through compaction.
pub const DEFAULT_KEEP_RECENT_TOKENS: u32 = 20_000;

/// Guard that cancels a [`CancellationToken`] when dropped.
///
/// Held inside the stream returned by [`Agent::start`] so that dropping the
/// stream automatically cancels in-flight LLM requests and tool executions.
struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Builder for constructing an [`Agent`] with sensible defaults.
///
/// Only a provider is required. Everything else has defaults:
/// - **tools**: empty registry
/// - **llm**: provider's default model, temperature 0.7, no reasoning
pub struct AgentBuilder {
    provider: Arc<dyn LlmProvider>,
    tools: Option<ToolRegistry>,
    llm: Option<AgentLlmConfig>,
    auto_compact: bool,
    compact_threshold: f64,
    reserve_tokens: u32,
    compact_at_tokens: Option<u32>,
    keep_recent_tokens: u32,
    working_dir: Option<PathBuf>,
    allow_parallel_tool_calls: bool,
    background_events: Option<UnboundedReceiver<AgentEvent>>,
}

impl AgentBuilder {
    /// Create an [`AgentBuilder`] with the given provider.
    fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            tools: None,
            llm: None,
            auto_compact: true,
            compact_threshold: 0.8,
            reserve_tokens: DEFAULT_RESERVE_TOKENS,
            compact_at_tokens: None,
            keep_recent_tokens: DEFAULT_KEEP_RECENT_TOKENS,
            working_dir: None,
            allow_parallel_tool_calls: false,
            background_events: None,
        }
    }

    /// Set the tool registry for this agent.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Merge background agent events into model and tool streams.
    pub fn background_events(mut self, events: UnboundedReceiver<AgentEvent>) -> Self {
        self.background_events = Some(events);
        self
    }

    /// Set the LLM configuration (model, temperature, reasoning, sampling).
    pub fn llm(mut self, llm: AgentLlmConfig) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Disable automatic compaction. When off, the agent yields
    /// `AgentEvent::CompactionNeeded` and ends the stream, letting the caller
    /// compact at a time of its choosing and resume with `AgentInput::Resume`.
    pub fn auto_compact(mut self, enabled: bool) -> Self {
        self.auto_compact = enabled;
        self
    }

    /// Set the compaction threshold as a fraction of the context window (0.0–1.0).
    /// When prompt tokens exceed `context_window * threshold`, compaction triggers.
    /// Default: `0.8` (80%).
    pub fn compact_threshold(mut self, threshold: f64) -> Self {
        if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
            self.compact_threshold = threshold;
        } else {
            tracing::warn!(
                threshold,
                "ignored invalid compaction threshold; expected finite value from 0.0 to 1.0"
            );
        }
        self
    }

    /// Tokens to keep free for the model's reply. Compaction triggers once
    /// prompt tokens exceed `context_window - reserve_tokens`. This is the
    /// primary control; [`compact_threshold`](Self::compact_threshold) acts as a
    /// floor so small-context models still compact. Default:
    /// [`DEFAULT_RESERVE_TOKENS`] (16,384).
    pub fn reserve_tokens(mut self, tokens: u32) -> Self {
        self.reserve_tokens = tokens;
        self
    }

    /// Compact once prompt tokens exceed this many, if that is earlier than
    /// the reserve-based trigger. `None` (the default) uses only the
    /// reserve-based trigger.
    pub fn compact_at_tokens(mut self, tokens: Option<u32>) -> Self {
        self.compact_at_tokens = tokens;
        self
    }

    /// Recent-history budget (in tokens) preserved verbatim through compaction.
    /// The summarizer keeps the newest entries up to this budget, snapped to a
    /// user-turn boundary. Default: [`DEFAULT_KEEP_RECENT_TOKENS`] (20,000).
    pub fn keep_recent_tokens(mut self, tokens: u32) -> Self {
        self.keep_recent_tokens = tokens;
        self
    }

    /// Set the workspace root passed to every tool invocation.
    ///
    /// File and process tools use this directory to resolve relative paths.
    /// Set it explicitly whenever an agent is allowed to access a workspace.
    pub fn working_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }

    /// Run consecutive concurrent tool calls from one model reply together.
    /// Tools whose [`Tool::concurrent`](flashmind_types::Tool::concurrent) is
    /// `false` run alone, preserving their position between batches. When a call in a batch
    /// interrupts, the rest of the batch still finishes and each interrupted
    /// call yields its own [`AgentEvent::Interrupted`], in call order.
    /// Default: `false`, every call runs in order.
    pub fn allow_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.allow_parallel_tool_calls = enabled;
        self
    }

    /// Build the [`Agent`], fetching context window and capabilities from the provider.
    pub async fn build(self) -> Agent {
        let mut agent = self.build_sync();
        agent.refresh_features().await;
        agent
    }

    /// Build the [`Agent`] without fetching context window from the provider.
    ///
    /// Prefer [`build`](Self::build). This one uses [`DEFAULT_CONTEXT_WINDOW`]
    /// (128k) and default capabilities until [`Agent::refresh_features`] is
    /// called, so a model with a larger window compacts far too early.
    pub fn build_sync(self) -> Agent {
        let provider_variant = self.provider.provider();

        let llm = self.llm.unwrap_or_else(|| AgentLlmConfig {
            model: Model {
                provider: provider_variant.clone(),
                model: AliasedModel {
                    name: provider_variant.default_model().into(),
                    real_name: None,
                },
            },
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(Decimal::new(7, 1)),
                ..Default::default()
            },
            user: None,
            provider_preferences: None,
            session_id: None,
        });

        let mut agent = Agent::new(self.provider, self.tools.unwrap_or_default(), llm);
        agent.auto_compact = self.auto_compact;
        agent.compact_threshold = self.compact_threshold;
        agent.reserve_tokens = self.reserve_tokens;
        agent.compact_at_tokens = self.compact_at_tokens;
        agent.keep_recent_tokens = self.keep_recent_tokens;
        agent.working_dir = self.working_dir;
        agent.allow_parallel_tool_calls = self.allow_parallel_tool_calls;
        agent.background_events = self.background_events.map(AsyncMutex::new);
        agent
    }
}

/// The main agent — owns provider, tool registry, and LLM config. Conversation
/// is passed in by the caller, keeping the agent stateless between turns.
///
/// Constructed via [`Agent::builder`] or [`Agent::new`], then configured with
/// builder-style helpers before calling [`start`](Self::start).
///
/// # Architecture
///
/// Each call to [`start`](Self::start) creates a stream of [`AgentEvent`] values
/// that drive the full loop: LLM completion → compaction check → next iteration.
/// The caller passes a [`CancellationToken`] for external abort.
///
/// Application concerns (system prompts, working directories, scopes, usernames)
/// live outside the agent. The caller prepends system prompts to the conversation
/// and provides context to tools directly.
///
/// # Memory management
///
/// The agent automatically handles context window pressure through progressive compaction:
/// 1. Truncate long tool outputs
/// 2. Run LLM summarization on conversation history
/// 3. Prune all tool outputs
/// 4. Strip tool message wrappers
/// 5. Last resort: truncate to only the final user exchange
///
/// # Examples
///
/// ```rust,ignore
/// let agent = Agent::builder(provider)
///     .tools(tools)
///     .build();
/// ```
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    llm: AgentLlmConfig,
    capabilities: ModelCapabilities,
    context_window: u32,
    auto_compact: bool,
    compact_threshold: f64,
    reserve_tokens: u32,
    compact_at_tokens: Option<u32>,
    keep_recent_tokens: u32,
    pub(crate) working_dir: Option<PathBuf>,
    pub(crate) allow_parallel_tool_calls: bool,
    background_events: Option<AsyncMutex<UnboundedReceiver<AgentEvent>>>,
}

impl Agent {
    /// Create an [`AgentBuilder`] with the given provider.
    ///
    /// This is the recommended way to construct an agent. Only a provider is
    /// required; everything else has sensible defaults.
    pub fn builder(provider: Arc<dyn LlmProvider>) -> AgentBuilder {
        AgentBuilder::new(provider)
    }

    /// Create a new agent with explicit parameters.
    ///
    /// Prefer [`Agent::builder`] for a more ergonomic API. This constructor
    /// is kept for backwards compatibility and for cases where all parameters
    /// are already available.
    ///
    /// # Parameters
    ///
    /// - **`provider`** — primary [`LlmProvider`] for completions.
    /// - **`tools`** — [`ToolRegistry`] of tools the agent can invoke.
    /// - **`llm`** — [`AgentLlmConfig`] controlling model, temperature, etc.
    pub fn new(provider: Arc<dyn LlmProvider>, tools: ToolRegistry, llm: AgentLlmConfig) -> Self {
        Self {
            provider,
            tools,
            llm,
            capabilities: ModelCapabilities::default(),
            context_window: DEFAULT_CONTEXT_WINDOW,
            auto_compact: true,
            compact_threshold: 0.8,
            reserve_tokens: DEFAULT_RESERVE_TOKENS,
            compact_at_tokens: None,
            keep_recent_tokens: DEFAULT_KEEP_RECENT_TOKENS,
            working_dir: None,
            allow_parallel_tool_calls: false,
            background_events: None,
        }
    }

    pub(crate) async fn next_background_event(&self) -> Option<AgentEvent> {
        match &self.background_events {
            Some(events) => events.lock().await.recv().await,
            None => futures::future::pending().await,
        }
    }

    /// Take queued child events, including while the parent is idle.
    pub fn drain_background_events(&self) -> Vec<AgentEvent> {
        let mut ready = Vec::new();
        if let Some(events) = &self.background_events
            && let Ok(mut events) = events.try_lock()
        {
            while let Ok(event) = events.try_recv() {
                ready.push(event);
            }
        }
        ready
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// The active LLM configuration (model, temperature, reasoning level).
    pub fn llm(&self) -> &AgentLlmConfig {
        &self.llm
    }

    /// The active LLM configuration, mutable.
    pub fn llm_mut(&mut self) -> &mut AgentLlmConfig {
        &mut self.llm
    }

    /// The registered tool registry.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// The registered tool registry, mutable.
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// Active LLM provider.
    pub fn provider(&self) -> &dyn LlmProvider {
        &*self.provider
    }

    /// Active LLM provider as a cloneable handle.
    pub fn provider_arc(&self) -> &Arc<dyn LlmProvider> {
        &self.provider
    }

    /// Swap the active LLM provider (e.g. when switching models).
    pub fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.provider = provider;
    }

    /// Cached model capabilities fetched from the provider on startup / model switch.
    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Context window size in tokens (fetched once and cached per model).
    pub fn context_window(&self) -> u32 {
        self.context_window
    }

    /// Current compaction threshold (0.0–1.0), used as a floor for the
    /// reserve-based trigger so small-context models still compact.
    pub fn compact_threshold(&self) -> f64 {
        self.compact_threshold
    }

    /// Tokens kept free for the model's reply; the primary compaction trigger
    /// fires once prompt tokens exceed `context_window - reserve_tokens`.
    pub fn reserve_tokens(&self) -> u32 {
        self.reserve_tokens
    }

    /// Recent-history budget (in tokens) preserved verbatim through compaction.
    pub fn keep_recent_tokens(&self) -> u32 {
        self.keep_recent_tokens
    }

    /// Workspace root supplied to tools for relative path resolution.
    pub fn working_dir(&self) -> Option<&Path> {
        self.working_dir.as_deref()
    }

    /// Set this agent's workspace without changing the process directory.
    pub fn set_working_dir(&mut self, path: PathBuf) {
        self.working_dir = Some(path);
    }

    /// Whether the agent compacts by itself when the trigger fires. When off,
    /// it yields `AgentEvent::CompactionNeeded` and leaves compaction to the
    /// caller.
    pub fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    /// Turn automatic compaction on or off at runtime.
    pub fn set_auto_compact(&mut self, enabled: bool) {
        self.auto_compact = enabled;
    }

    /// Token limit set with [`set_compact_at_tokens`](Self::set_compact_at_tokens).
    pub fn compact_at_tokens(&self) -> Option<u32> {
        self.compact_at_tokens
    }

    /// Compact once prompt tokens exceed `tokens`, if that is earlier than the
    /// reserve-based trigger. `None` restores the default trigger.
    pub fn set_compact_at_tokens(&mut self, tokens: Option<u32>) {
        self.compact_at_tokens = tokens;
    }

    /// Prompt-token count above which a turn requests compaction.
    ///
    /// Leaves `reserve_tokens` free for the reply, with `compact_threshold` of
    /// the window as a floor so small-context models (where the reserve
    /// saturates to 0) still compact on context pressure instead of on every
    /// turn. A `compact_at_tokens` limit can only move the trigger earlier.
    pub fn compaction_trigger_tokens(&self) -> u32 {
        let reserve_based = self.context_window.saturating_sub(self.reserve_tokens);
        let floor = (self.context_window as f64 * self.compact_threshold) as u32;
        let default = reserve_based.max(floor);
        self.compact_at_tokens
            .map_or(default, |limit| limit.min(default))
    }

    /// Update the compaction threshold at runtime (0.0–1.0).
    pub fn set_compact_threshold(&mut self, threshold: f64) {
        if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
            self.compact_threshold = threshold;
        } else {
            tracing::warn!(
                threshold,
                "ignored invalid compaction threshold; expected finite value from 0.0 to 1.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Features & model management
    // -----------------------------------------------------------------------

    /// Refresh `capabilities` and `context_window` from the current provider.
    ///
    /// Call this after switching models or at startup. Disables reasoning
    /// automatically if the new model doesn't support it.
    pub async fn refresh_features(&mut self) {
        self.capabilities = self.provider.capabilities(&self.llm.model).await;
        self.context_window = match self.provider.context_window(&self.llm.model).await {
            Some(size) => size,
            None => {
                let ctx = if self.context_window == 0 {
                    DEFAULT_CONTEXT_WINDOW
                } else {
                    self.context_window
                };
                tracing::warn!(model = %self.llm.model, ctx, "Failed to fetch context window");
                ctx
            }
        };

        if self.llm.reasoning.is_on() && !self.capabilities.reasoning {
            tracing::warn!("Disabled reasoning for {}. Not supported", self.llm.model);
            self.llm.reasoning = ReasoningLevel::Off;
        }

        tracing::debug!(
            "Refreshed features ctx={} - {:?}",
            self.context_window,
            self.capabilities,
        );
    }

    /// Returns the model and provider used for compaction (currently the active ones).
    fn compaction_model_and_provider(&self) -> (Model, Arc<dyn LlmProvider>) {
        (self.llm.model.clone(), Arc::clone(&self.provider))
    }

    // -----------------------------------------------------------------------
    // Core loop
    // -----------------------------------------------------------------------

    /// Run the full agent loop for one user input and return a stream of [`AgentEvent`]s.
    ///
    /// The caller owns the conversation and passes it in. The agent mutates it
    /// during the loop (adding entries, compacting) and returns it when the
    /// stream completes.
    ///
    /// The caller also owns the [`CancellationToken`]. To inject a new prompt
    /// mid-turn, cancel the token, wait for the stream to end, then call
    /// `start()` again with the same conversation — it already contains
    /// everything from the previous run.
    ///
    /// The returned stream yields events incrementally:
    /// - `ReasoningDelta` / `TextDelta` — live LLM output as it arrives
    /// - `ToolStart` / `ToolResult` — per-tool lifecycle
    /// - `Status` — compaction progress messages
    /// - `Done` or `Error` — terminal event
    ///
    /// After emitting the terminal event, the conversation is cleaned up
    /// (agent progress, memories, reminders stripped).
    pub fn start<'a>(
        &'a mut self,
        conversation: &'a mut Conversation,
        cancel_token: CancellationToken,
        input: AgentInput,
        max_iterations: Option<usize>,
    ) -> impl Stream<Item = AgentEvent> + 'a {
        let warn_iterations = max_iterations.map(|max| (max as f64 * 0.9).ceil() as usize);

        let cancel_token = cancel_token.child_token();
        async_stream::stream! {
            let _guard = CancelOnDrop(cancel_token.clone());

            let turn_start = Instant::now();
            metrics::counter!("agent.turns_started").increment(1);

            if let AgentInput::User { content, context, parts } = input {
                if let Some(ctx) = context {
                    conversation.add(ConversationEntry::system_message(ctx));
                }
                match parts {
                    Some(p) if !p.is_empty() => {
                        conversation.add(ConversationEntry::user_with_parts(&content, p));
                    }
                    _ => {
                        conversation.add(ConversationEntry::user(&content));
                    }
                }
            }

            conversation.mark_turn_start();

            let mut recovery = RecoveryAttempts::default();
            let mut consecutive_compactions: u8 = 0;
            let mut final_content = String::new();
            let mut empty_response = false;
            let mut continuations = 0u8;
            let mut iteration = 0usize;

            let final_result: TurnResult = loop {
                iteration += 1;
                    metrics::counter!("agent.iterations").increment(1);

                if let Err(e) = check_iteration_limits(
                    conversation,
                    max_iterations,
                    warn_iterations,
                    iteration,
                ) {
                    break Err(e);
                }

                let result = {
                    let mut turn = self.run_turn(
                        conversation,
                        &cancel_token,
                    );
                    while let Some(ev) = turn.next().await {
                        yield ev;
                    }
                    turn.take_result().unwrap_or_else(|| {
                        Err(anyhow::anyhow!("run_turn stream ended without result"))
                    })
                };

                if result.is_ok() {
                    recovery.record_success();
                }

                match result {
                    Ok(TurnStatus::CompactionNeeded { content, usage, reason }) => {
                        consecutive_compactions += 1;

                        if consecutive_compactions > 2 {
                            tracing::warn!("Agent compacted {consecutive_compactions} times consecutively — context is thrashing");
                            break Err(anyhow::anyhow!(
                                "Context window is too small for this task — compaction happened \
                                 {consecutive_compactions} times in a row without making progress. \
                                 Try breaking the task into smaller steps."
                            ));
                        }

                        if !self.auto_compact {
                            if !content.trim().is_empty() {
                                final_content = content;
                            }
                            yield AgentEvent::CompactionNeeded { reason };
                            break Ok(TurnStatus::Done { content: final_content.clone(), usage });
                        }

                        let (compact_model, compact_provider) = self.compaction_model_and_provider();
                        match reason {
                            CompactionReason::OutputLength => {
                                if let Err(e) = conversation
                                    .compact_with_llm(&*compact_provider, &compact_model)
                                    .await
                                {
                                    tracing::warn!("Output-length compaction failed: {e}");
                                }
                                if !content.trim().is_empty() {
                                    empty_response = false;
                                }
                                continue;
                            }
                            CompactionReason::ContextThreshold(prompt_tokens) => {
                                use futures::StreamExt as _;
                                let compact_stream = crate::compaction::try_compact(
                                    conversation,
                                    prompt_tokens,
                                    self.context_window,
                                    self.keep_recent_tokens,
                                    &*compact_provider,
                                    &compact_model,
                                );
                                tokio::pin!(compact_stream);
                                while let Some(ev) = compact_stream.next().await {
                                    yield ev;
                                }
                                if !content.trim().is_empty() {
                                    final_content = content;
                                }
                                break Ok(TurnStatus::Done { content: final_content.clone(), usage });
                            }
                        }
                    }

                    Err(error) if !cancel_token.is_cancelled() => {
                        let recovered = {
                            let mut attempt = self.recover(conversation, &cancel_token, &error, &mut recovery);
                            while let Some(event) = attempt.next().await {
                                yield event;
                            }
                            attempt.take_result().unwrap_or(false)
                        };
                        if recovered {
                            continue;
                        }
                        break Err(error);
                    }

                    Ok(TurnStatus::ToolCalls { ref content, ref tool_calls, usage }) if !cancel_token.is_cancelled() => {
                        if !content.trim().is_empty() {
                            empty_response = false;
                        }
                        final_content = content.clone();
                        consecutive_compactions = 0;

                        let mut interrupted = false;
                        let previous_dir = self.working_dir.clone();
                        for batch in self.tool_batches(tool_calls) {
                            let completed = {
                                let mut execution = self.execute_tool_batch(&batch, &cancel_token);
                                while let Some(event) = execution.next().await {
                                    yield event;
                                }
                                execution.take_result().unwrap_or_default()
                            };
                            for done in completed {
                                interrupted |= done.result.is_interrupt();
                                for event in done.finish(conversation) {
                                    yield event;
                                }
                            }
                            if interrupted || cancel_token.is_cancelled() {
                                break;
                            }
                        }

                        if !interrupted && !cancel_token.is_cancelled() && self.working_dir != previous_dir
                            && let Some(path) = self.working_dir()
                        {
                            conversation.add(ConversationEntry::system_message(format!(
                                "The working directory is now {}. Subsequent commands and file tools use this directory.", path.display(),
                            )));
                        }
                        if interrupted {
                            break Ok(TurnStatus::Done { content: final_content.clone(), usage });
                        }
                        continue;
                    }

                    Ok(TurnStatus::Continue { ref content, .. })
                        if continuations < MAX_CONTINUATIONS && !cancel_token.is_cancelled() =>
                    {
                        continuations += 1;
                        if !content.trim().is_empty() {
                            final_content = content.clone();
                        }
                        tracing::info!(continuations, "Reply hit the output limit, continuing");
                        continue;
                    }

                    Ok(TurnStatus::Done { ref content, .. }) if content.trim().is_empty() && !empty_response => {
                        empty_response = true;
                        continue;
                    }

                    other => break other,
                }
            };

            match &final_result {
                Ok(status) => {
                    let (c, _) = status.content_and_usage();
                    if !c.is_empty() {
                        final_content = c.to_string();
                    }

                    metrics::gauge!("agent.conversation.entries").set(conversation.entries().len() as f64);
                    metrics::counter!("agent.turns_completed").increment(1);
                    metrics::histogram!("agent.turn.duration_seconds").record(turn_start.elapsed().as_secs_f64());

                    if cancel_token.is_cancelled() {
                        yield AgentEvent::Error("cancelled".into());
                    } else {
                        yield AgentEvent::Done(final_content.clone());
                    }
                }
                Err(err) => {
                    metrics::counter!("agent.turns_errors").increment(1);
                    metrics::histogram!("agent.turn.duration_seconds").record(turn_start.elapsed().as_secs_f64());
                    yield AgentEvent::Error(err.to_string());
                }
            }

            if cancel_token.is_cancelled() {
                conversation.add(ConversationEntry::assistant("User interrupted the task"));
            }

            conversation.strip_agent_progress();
            conversation.strip_memories();
            conversation.strip_reminders();
        }
    }

    // -----------------------------------------------------------------------
    // Single turn
    // -----------------------------------------------------------------------

    /// Execute one LLM round-trip: prompt → stream → result.
    ///
    /// Returns an [`AgentStream`] that yields [`AgentEvent`]s during processing
    /// and terminates with a [`TurnResult`]. The caller is responsible for:
    /// - Checking iteration limits before calling this
    /// - Executing any tool calls returned in [`TurnStatus::ToolCalls`]
    /// - Handling errors and compaction (use [`handle_llm_error`] and [`try_compact`](crate::compaction::try_compact))
    pub fn run_turn<'a>(
        &'a mut self,
        conversation: &'a mut Conversation,
        cancel_token: &'a CancellationToken,
    ) -> AgentStream<'a, AgentEvent, TurnResult> {
        AgentStream::new(async_stream::stream! {
            let tool_definitions = self.tools.definitions();

            tracing::info!(
                model = %self.llm.model,
                sampling = %self.llm.sampling,
                reasoning = ?self.llm.reasoning,
                "Running turn"
            );

            let mut llm = stream_llm_response(
                &*self.provider,
                &self.llm,
                cancel_token,
                conversation,
                &tool_definitions,
            );
            loop {
                tokio::select! {
                    Some(event) = self.next_background_event() => yield Outcome::Item(event),
                    event = llm.next() => match event {
                        Some(event) => yield Outcome::Item(event),
                        None => break,
                    },
                }
            }
            for event in self.drain_background_events() {
                yield Outcome::Item(event);
            }

            let resp = match llm.take_result() {
                Some(Ok(r)) => r,
                Some(Err(e)) => {
                    yield Outcome::Done(Err(e));
                    return;
                }
                None => {
                    yield Outcome::Done(Err(anyhow::anyhow!("LLM stream ended without result")));
                    return;
                }
            };

            // Ghost response: provider returned HTTP 200 but no actual content.
            // prompt_tokens=0 is the tell — a real empty response still reports prompt size.
            if resp.prompt_tokens == 0
                && resp.completion_tokens == 0
                && resp.content.is_empty()
                && resp.tool_calls.is_empty()
            {
                yield Outcome::Done(Err(flashmind_types::LlmError::new(
                    flashmind_types::LlmErrorKind::StreamError,
                    "Ghost response from provider (0 prompt tokens, 0 completion tokens, no content)",
                ).into()));
                return;
            }

            let usage = TurnUsage {
                prompt_tokens: resp.prompt_tokens,
                completion_tokens: resp.completion_tokens,
                cache_read_tokens: resp.cache_read_tokens,
                cache_creation_tokens: resp.cache_creation_tokens,
            };

            if !resp.tool_calls.is_empty() {
                conversation.add(ConversationEntry::assistant_with_tool_calls(
                    &resp.content,
                    resp.tool_calls.clone(),
                    resp.reasoning.clone(),
                ));
                yield Outcome::Done(Ok(TurnStatus::ToolCalls {
                    content: resp.content,
                    tool_calls: resp.tool_calls,
                    usage,
                }));
                return;
            }

            // A continuation of a cut-off reply joins the reply it carries on,
            // so the conversation keeps one assistant message.
            let content = match continued_reply(conversation) {
                Some(entry) => {
                    let joined = format!("{}{}", entry.content(), resp.content);
                    entry.set_content(joined.clone());
                    joined
                }
                None => {
                    conversation.add(ConversationEntry::assistant_with_reasoning(
                        &resp.content,
                        resp.reasoning,
                    ));
                    resp.content
                }
            };

            // The model stopped at the output limit, its own or `max_tokens`:
            // ask it to carry the reply on rather than end the turn on half of it.
            if resp.finish_reason == FinishReason::Length {
                conversation.add(ConversationEntry::reminder(CONTINUE_NOTE));
                yield Outcome::Done(Ok(TurnStatus::Continue { content, usage }));
                return;
            }

            // Compact proactively while there's still room for the model's reply.
            if resp.prompt_tokens > self.compaction_trigger_tokens() {
                yield Outcome::Done(Ok(TurnStatus::CompactionNeeded {
                    content: content.clone(),
                    usage,
                    reason: CompactionReason::ContextThreshold(resp.prompt_tokens),
                }));
                return;
            }

            yield Outcome::Done(Ok(TurnStatus::Done {
                content: content.clone(),
                usage,
            }));
        })
    }
}

/// How many times one turn carries on a reply that hit the output limit.
pub const MAX_CONTINUATIONS: u8 = 2;

/// Sent after a reply that stopped at the output limit.
const CONTINUE_NOTE: &str = "Your last reply was cut off at the output limit. Continue it \
    exactly where it stopped, mid-word or mid-row if need be, without repeating anything \
    or adding a preamble. If nothing was written yet, write the reply now. Keep thinking \
    short.";

/// The cut-off reply a continuation joins: the assistant entry just before
/// the continue note, when that note is the last entry. Drops the note.
fn continued_reply(conversation: &mut Conversation) -> Option<&mut ConversationEntry> {
    let entries = conversation.entries_mut();
    let n = entries.len();
    let noted = n >= 2
        && entries[n - 1].is_reminder()
        && entries[n - 1].content() == CONTINUE_NOTE
        && entries[n - 2].is_assistant();
    if !noted {
        return None;
    }
    entries.pop();
    entries.last_mut()
}

// ---------------------------------------------------------------------------
// Free functions — shared turn helpers
// ---------------------------------------------------------------------------

/// Enforce `max_iterations` (abort) and `warn_iterations` (inject warning).
///
/// Call this *before* [`Agent::run_turn`] in the outer loop.
pub fn check_iteration_limits(
    conversation: &mut Conversation,
    max_iterations: Option<usize>,
    warn_iterations: Option<usize>,
    iteration: usize,
) -> anyhow::Result<()> {
    if let Some(max_iters) = max_iterations
        && iteration > max_iters
    {
        tracing::warn!("Agent hit hard iteration cap: {iteration}/{max_iters}");
        anyhow::bail!("Reached maximum iteration limit ({max_iters}). Stopping.");
    }

    if let Some(warn_iters) = warn_iterations
        && iteration == warn_iters
    {
        tracing::warn!("Agent approaching warn threshold: {iteration}/{warn_iters}");
        conversation.add(ConversationEntry::user(format!(
            "Warning: you have used {iteration}/{warn_iters} tool iterations. \
                 Consider wrapping up or asking the user for guidance."
        )));
    }

    Ok(())
}

/// Perform one compaction attempt for error recovery.
/// `attempt` is the current attempt number (1 = first, 2 = truncate-only).
pub fn handle_llm_error<'a>(
    conversation: &'a mut Conversation,
    compact_provider: &'a dyn LlmProvider,
    compact_model: &'a Model,
    _cancel_token: &'a CancellationToken,
    attempt: u8,
) -> AgentStream<'a, AgentEvent, ()> {
    AgentStream::new(async_stream::stream! {
        if attempt == 1 {
            tracing::warn!("LLM error (context overflow) — compacting and retrying");
            let binary_stripped = conversation.strip_binary_parts();
            if binary_stripped > 0 {
                tracing::info!("Stripped {binary_stripped} binary parts during error recovery");
            }

            // Reduce size BEFORE the LLM compaction call — the conversation
            // may already be too large for the compaction model's context.
            let truncated = conversation.truncate_long_tool_outputs(200);
            if truncated > 0 {
                tracing::info!("Truncated {truncated} long tool outputs during error recovery");
            }

            let compacted = match conversation
                .compact_with_llm(compact_provider, compact_model)
                .await
            {
                Ok(opt) => opt,
                Err(e) => {
                    tracing::warn!("Compaction LLM call errored during error recovery: {e}");
                    None
                }
            };

            if let Some(result) = compacted {
                yield Outcome::Item(AgentEvent::Compacted(result.summary));
                yield Outcome::Done(());
                return;
            }

            // LLM compaction failed — strip tool messages and retry once more
            tracing::warn!("LLM compaction failed on error recovery, escalating");
            let stripped = conversation.strip_tool_messages();
            if stripped > 0 {
                tracing::info!("Stripped {stripped} tool messages during error recovery");
                match conversation
                    .compact_with_llm(compact_provider, compact_model)
                    .await
                {
                    Ok(Some(result)) => {
                        tracing::info!("Compaction succeeded after stripping tool messages");
                        yield Outcome::Item(AgentEvent::Compacted(result.summary));
                        yield Outcome::Done(());
                        return;
                    }
                    Ok(None) => tracing::warn!("LLM compaction returned empty after strip"),
                    Err(e) => tracing::warn!("LLM compaction errored after strip: {e}"),
                }
            }

            if binary_stripped == 0 && truncated == 0 && stripped == 0 {
                tracing::warn!("No compaction possible — truncating to last exchange");
                conversation.truncate_to_last_exchange();
                yield Outcome::Item(AgentEvent::Compacted(
                    "Only the last exchange was kept; no summary was available.".into(),
                ));
            } else {
                yield Outcome::Item(AgentEvent::Compacted(
                    "Context reduced without a summary.".into(),
                ));
            }
        } else {
            tracing::warn!("Context still overflowing after compaction — truncating to last exchange");
            yield Outcome::Item(AgentEvent::Status(
                "Still too large — truncating to last exchange...".into(),
            ));
            conversation.truncate_to_last_exchange();
            yield Outcome::Item(AgentEvent::Compacted(
                "Only the last exchange was kept; no summary was available.".into(),
            ));
        }

        yield Outcome::Done(());
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use async_trait::async_trait;
    use flashmind_types::{
        AliasedModel, CompletionRequest, CompletionStream, FinishReason, Model, Provider,
        ReasoningLevel, SamplingParams, SessionId, StreamEvent, TokenUsage,
    };
    use futures::StreamExt;
    use rust_decimal_macros::dec;
    use std::any::Any;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::conversation::EntryKind;
    use flashmind_types::Tool;
    use flashmind_types::tool::{InterruptPayload, ToolContext, ToolResult};

    struct MockProvider {
        responses: std::sync::Mutex<Vec<Vec<StreamEvent>>>,
        sessions: Mutex<Vec<Option<SessionId>>>,
    }

    impl MockProvider {
        fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                sessions: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.sessions.lock().unwrap().push(request.session_id);
            let mut responses = self.responses.lock().unwrap();
            let events = if responses.is_empty() {
                vec![
                    StreamEvent::ContentDelta("No more responses".into()),
                    StreamEvent::Finished(FinishReason::Stop),
                ]
            } else {
                responses.remove(0)
            };
            Box::pin(stream! {
                for event in events {
                    yield Ok(event);
                }
            })
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }
    }

    fn test_agent(provider: Arc<dyn LlmProvider>) -> Agent {
        let model = Model {
            provider: Provider::Ollama,
            model: AliasedModel {
                name: "test".into(),
                real_name: None,
            },
        };
        let llm = AgentLlmConfig {
            model,
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.7)),
                ..Default::default()
            },
            user: None,
            provider_preferences: None,
            session_id: None,
        };
        Agent::new(provider, ToolRegistry::new(), llm)
    }

    /// Counts how many tools run at once.
    #[derive(Default)]
    struct Overlap {
        active: AtomicUsize,
        most: AtomicUsize,
        ran: Mutex<Vec<String>>,
    }

    #[derive(Debug)]
    struct Ask;

    impl InterruptPayload for Ask {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn display_output(&self) -> String {
            "asked".into()
        }
    }

    struct Slow {
        name: &'static str,
        concurrent: bool,
        asks: bool,
        overlap: Arc<Overlap>,
    }

    #[async_trait]
    impl Tool for Slow {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
            if self.asks {
                return Ok(ToolResult::interrupt(ctx.tool_call_id, Arc::new(Ask)));
            }
            let now = self.overlap.active.fetch_add(1, Ordering::SeqCst) + 1;
            if !self.concurrent {
                assert_eq!(now, 1, "A serial tool must run alone.");
            }
            self.overlap.most.fetch_max(now, Ordering::SeqCst);
            ctx.progress(self.name);
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.overlap.active.fetch_sub(1, Ordering::SeqCst);
            self.overlap.ran.lock().unwrap().push(self.name.into());
            Ok(ToolResult::success(ctx.tool_call_id, self.name))
        }

        fn concurrent(&self) -> bool {
            self.concurrent
        }
    }

    /// Runs one turn whose reply calls `tools` in order, then says "done".
    async fn run_batch(tools: Vec<Slow>, parallel: bool) -> (Vec<AgentEvent>, Conversation) {
        let mut reply: Vec<StreamEvent> = Vec::new();
        for (index, tool) in tools.iter().enumerate() {
            reply.push(StreamEvent::ToolCallStart {
                index,
                id: tool.name.into(),
                name: tool.name.into(),
            });
            reply.push(StreamEvent::ToolCallDelta {
                index,
                arguments: "{}".into(),
            });
        }
        reply.push(StreamEvent::Finished(FinishReason::ToolCalls));
        let provider = Arc::new(MockProvider::new(vec![
            reply,
            vec![
                StreamEvent::ContentDelta("done".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ],
        ]));
        let mut agent = test_agent(provider);
        agent.allow_parallel_tool_calls = parallel;
        for tool in tools {
            agent.tools_mut().register(Arc::new(tool));
        }
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::User {
                    content: "go".into(),
                    context: None,
                    parts: None,
                },
                None,
            )
            .collect::<Vec<_>>()
            .await;
        (events, conversation)
    }

    fn slow(name: &'static str, concurrent: bool, overlap: &Arc<Overlap>) -> Slow {
        Slow {
            name,
            concurrent,
            asks: false,
            overlap: overlap.clone(),
        }
    }

    fn answered(conversation: &Conversation) -> Vec<String> {
        conversation
            .entries()
            .iter()
            .filter_map(|entry| match &entry.kind {
                EntryKind::Tool { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn parallel_calls_overlap_and_serial_ones_run_alone() {
        let overlap = Arc::new(Overlap::default());
        let tools = vec![
            slow("a", true, &overlap),
            slow("b", true, &overlap),
            slow("phone", false, &overlap),
            slow("c", true, &overlap),
            slow("d", true, &overlap),
        ];
        let (events, conversation) = run_batch(tools, true).await;

        assert_eq!(overlap.most.load(Ordering::SeqCst), 2);
        assert_eq!(overlap.ran.lock().unwrap()[2], "phone");
        assert_eq!(answered(&conversation), ["a", "b", "phone", "c", "d"]);
        let progress: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolProgress { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(progress.len(), 5);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Done(text) if text == "done"))
        );
    }

    #[tokio::test]
    async fn calls_run_in_order_without_the_setting() {
        let overlap = Arc::new(Overlap::default());
        let tools = vec![slow("a", true, &overlap), slow("b", true, &overlap)];
        let (_, conversation) = run_batch(tools, false).await;

        assert_eq!(overlap.most.load(Ordering::SeqCst), 1);
        assert_eq!(answered(&conversation), ["a", "b"]);
    }

    #[tokio::test]
    async fn an_interrupt_lets_the_batch_finish_and_skips_serial_calls() {
        let overlap = Arc::new(Overlap::default());
        let ask = |name| Slow {
            name,
            concurrent: true,
            asks: true,
            overlap: overlap.clone(),
        };
        let tools = vec![
            ask("send"),
            slow("a", true, &overlap),
            ask("delete"),
            slow("phone", false, &overlap),
        ];
        let (events, conversation) = run_batch(tools, true).await;

        assert_eq!(answered(&conversation), ["a"]);
        assert_eq!(*overlap.ran.lock().unwrap(), ["a"]);
        let interrupted: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::Interrupted { tool_name, .. } => Some(tool_name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(interrupted, ["send", "delete"]);
    }

    /// Runs one turn over `replies`.
    async fn run_capped(replies: Vec<Vec<StreamEvent>>) -> (Vec<AgentEvent>, Conversation) {
        let provider = Arc::new(MockProvider::new(replies));
        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::User {
                    content: "go".into(),
                    context: None,
                    parts: None,
                },
                None,
            )
            .collect::<Vec<_>>()
            .await;
        (events, conversation)
    }

    fn assistant_texts(conversation: &Conversation) -> Vec<&str> {
        conversation
            .entries()
            .iter()
            .filter(|entry| entry.is_assistant())
            .map(|entry| entry.content())
            .collect()
    }

    fn cut_usage() -> StreamEvent {
        StreamEvent::Usage(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 100,
            total_tokens: 110,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        })
    }

    #[tokio::test]
    async fn a_reply_cut_at_the_cap_is_continued_into_one_message() {
        let (events, conversation) = run_capped(vec![
            vec![
                StreamEvent::ContentDelta("| a | 1 |\n| b | ".into()),
                StreamEvent::Finished(FinishReason::Length),
                cut_usage(),
            ],
            vec![
                StreamEvent::ContentDelta("2 |".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ],
        ])
        .await;

        assert_eq!(assistant_texts(&conversation), ["| a | 1 |\n| b | 2 |"]);
        assert!(
            !conversation
                .entries()
                .iter()
                .any(|entry| entry.is_reminder())
        );
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::Done(text) if text == "| a | 1 |\n| b | 2 |")
        ));
    }

    #[tokio::test]
    async fn a_reply_the_cap_left_empty_is_written_on_the_next_call() {
        let (events, conversation) = run_capped(vec![
            vec![StreamEvent::Finished(FinishReason::Length), cut_usage()],
            vec![
                StreamEvent::ContentDelta("answer".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ],
        ])
        .await;

        assert_eq!(assistant_texts(&conversation), ["answer"]);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Done(text) if text == "answer"))
        );
    }

    #[tokio::test]
    async fn continuing_stops_after_the_limit() {
        let cut = |text: &str| {
            vec![
                StreamEvent::ContentDelta(text.into()),
                StreamEvent::Finished(FinishReason::Length),
                cut_usage(),
            ]
        };
        let (_, conversation) = run_capped(vec![cut("a"), cut("b"), cut("c"), cut("d")]).await;

        assert_eq!(assistant_texts(&conversation), ["abc"]);
    }

    #[tokio::test]
    async fn session_id_follows_turns_and_session_changes() {
        let provider = Arc::new(MockProvider::new(Vec::new()));
        let mut agent = test_agent(provider.clone());
        let mut conversation = Conversation::new();
        for session_id in ["example-session", "example-session", "another-session"] {
            agent.llm_mut().session_id = Some(session_id.parse().unwrap());
            agent
                .start(
                    &mut conversation,
                    CancellationToken::new(),
                    AgentInput::User {
                        content: "hello".into(),
                        context: None,
                        parts: None,
                    },
                    None,
                )
                .collect::<Vec<_>>()
                .await;
        }
        assert_eq!(
            *provider.sessions.lock().unwrap(),
            [
                Some("example-session".parse().unwrap()),
                Some("example-session".parse().unwrap()),
                Some("another-session".parse().unwrap())
            ],
        );
    }

    #[tokio::test]
    async fn test_simple_response() {
        use futures::StreamExt;

        let provider = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Hello ".into()),
            StreamEvent::ContentDelta("world!".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();
        conversation.set_system("You are helpful");

        let mut deltas = String::new();
        let mut done_text = String::new();

        let s = agent.start(
            &mut conversation,
            CancellationToken::new(),
            AgentInput::User {
                content: "Hi".into(),
                context: None,
                parts: None,
            },
            None,
        );
        tokio::pin!(s);
        while let Some(ev) = s.next().await {
            match ev {
                AgentEvent::TextDelta(d) => deltas.push_str(&d),
                AgentEvent::Done(t) => done_text = t,
                _ => {}
            }
        }

        assert_eq!(deltas, "Hello world!");
        assert_eq!(done_text, "Hello world!");
    }

    #[tokio::test]
    async fn model_turn_forwards_background_usage_and_keeps_parent_context_separate() {
        use flashmind_types::TokenUsage;
        use tokio::sync::mpsc;

        let (sender, receiver) = mpsc::unbounded_channel();
        sender
            .send(AgentEvent::SpawnedEvent {
                id: "worker".into(),
                task: "Example task".into(),
                model: None,
                event: Box::new(AgentEvent::Usage(TokenUsage {
                    prompt_tokens: 500,
                    completion_tokens: 10,
                    total_tokens: 510,
                    cache_read_tokens: 400,
                    ..Default::default()
                })),
            })
            .unwrap();
        drop(sender);
        let provider = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Done.".into()),
            StreamEvent::Usage(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 1,
                total_tokens: 11,
                ..Default::default()
            }),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));
        let mut agent = Agent::builder(provider)
            .background_events(receiver)
            .build_sync();
        let mut conversation = Conversation::new();
        let cancel = CancellationToken::new();
        let mut turn = agent.run_turn(&mut conversation, &cancel);
        let mut background_count = 0;
        while let Some(event) = turn.next().await {
            if let AgentEvent::SpawnedEvent { event, .. } = event {
                assert!(
                    matches!(*event, AgentEvent::Usage(usage) if usage.cache_read_tokens == 400)
                );
                background_count += 1;
            }
        }
        let result = turn.take_result().unwrap().unwrap();
        assert_eq!(background_count, 1);
        assert_eq!(result.content_and_usage().1.prompt_tokens, 10);
    }

    #[tokio::test]
    async fn start_emits_usage_once_per_request_including_empty_reply_retry() {
        use flashmind_types::TokenUsage;

        let response = |text: &str| {
            vec![
                StreamEvent::ContentDelta(text.into()),
                StreamEvent::Usage(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 1,
                    total_tokens: 11,
                    ..Default::default()
                }),
                StreamEvent::Finished(FinishReason::Stop),
            ]
        };
        let provider = Arc::new(MockProvider::new(vec![response(""), response("Done.")]));
        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::user("Example"),
                None,
            )
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::Usage(_)))
                .count(),
            2
        );
        assert!(matches!(events.last(), Some(AgentEvent::Done(text)) if text == "Done."));
    }

    #[tokio::test(start_paused = true)]
    async fn repeated_midstream_disconnects_preserve_completed_tools() {
        use std::io::{Error as IoError, ErrorKind};

        use flashmind_types::{Message, Role};

        #[derive(Default)]
        struct DisconnectingProvider {
            requests: Mutex<Vec<Vec<Message>>>,
        }

        #[async_trait]
        impl LlmProvider for DisconnectingProvider {
            fn name(&self) -> &str {
                "mock"
            }

            fn provider(&self) -> Provider {
                Provider::Ollama
            }

            fn complete(&self, request: CompletionRequest) -> CompletionStream {
                let mut requests = self.requests.lock().unwrap();
                let attempt = requests.len();
                requests.push(request.messages);
                let tool_call = || {
                    vec![
                        Ok(StreamEvent::ToolCallStart {
                            index: 0,
                            id: "call-example".into(),
                            name: "record".into(),
                        }),
                        Ok(StreamEvent::ToolCallDelta {
                            index: 0,
                            arguments: "{}".into(),
                        }),
                    ]
                };
                let events = match attempt {
                    0 => {
                        let mut events = tool_call();
                        events.push(Ok(StreamEvent::Finished(FinishReason::ToolCalls)));
                        events
                    }
                    1 => {
                        let mut events = vec![
                            Ok(StreamEvent::ReasoningDelta("unfinished thought".into())),
                            Ok(StreamEvent::ContentDelta("unfinished reply".into())),
                        ];
                        events.extend(tool_call());
                        events.push(Err(IoError::new(
                            ErrorKind::ConnectionReset,
                            "connection reset",
                        )
                        .into()));
                        events
                    }
                    2 => vec![Ok(StreamEvent::ContentDelta(
                        "another partial reply".into(),
                    ))],
                    3 => vec![
                        Ok(StreamEvent::ContentDelta(
                            "partial reply before timeout".into(),
                        )),
                        Err(IoError::new(ErrorKind::TimedOut, "connection timed out").into()),
                    ],
                    4 => vec![
                        Ok(StreamEvent::ContentDelta("Recovered.".into())),
                        Ok(StreamEvent::Finished(FinishReason::Stop)),
                    ],
                    _ => panic!("Unexpected request after recovery"),
                };
                Box::pin(futures::stream::iter(events))
            }
        }

        let provider = Arc::new(DisconnectingProvider::default());
        let mut agent = test_agent(provider.clone());
        let overlap = Arc::new(Overlap::default());
        agent.tools_mut().register(Arc::new(Slow {
            name: "record",
            concurrent: false,
            asks: false,
            overlap: overlap.clone(),
        }));
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::user("Example"),
                None,
            )
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(events.last(), Some(AgentEvent::Done(text)) if text == "Recovered."));
        assert_eq!(overlap.ran.lock().unwrap().as_slice(), ["record"]);
        assert_eq!(assistant_texts(&conversation), ["", "Recovered."]);
        let statuses: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::Status(status) => Some(status),
                _ => None,
            })
            .collect();
        assert_eq!(statuses.len(), 3);
        for (status, delay) in statuses.iter().zip([2, 4, 8]) {
            assert!(status.contains(&format!("retrying in {delay}s")));
        }
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 5);
        let retries: Vec<_> = requests[1..]
            .iter()
            .map(|messages| {
                messages
                    .iter()
                    .map(|message| {
                        let mut message = serde_json::to_value(message).unwrap();
                        message.as_object_mut().unwrap().remove("timestamp");
                        message
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(retries.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(requests[1].iter().any(|message| message.role == Role::Tool));
    }

    #[tokio::test(start_paused = true)]
    async fn successful_response_resets_stream_retries() {
        let mut responses = vec![vec![]; 8];
        responses.push(vec![
            StreamEvent::ContentDelta("First part. ".into()),
            StreamEvent::Finished(FinishReason::Length),
        ]);
        responses.extend(vec![vec![]; 8]);
        responses.push(vec![
            StreamEvent::ContentDelta("Done.".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]);
        let provider = Arc::new(MockProvider::new(responses));
        let mut agent = test_agent(provider.clone());
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::user("Example"),
                None,
            )
            .collect::<Vec<_>>()
            .await;

        assert!(
            matches!(events.last(), Some(AgentEvent::Done(text)) if text == "First part. Done.")
        );
        assert_eq!(provider.sessions.lock().unwrap().len(), 18);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::Status(_)))
                .count(),
            16
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::Error(_)))
        );
    }

    /// Drive one turn with a mock provider reporting `prompt_tokens` and return
    /// whether the proactive (reserve-based) trigger fired `CompactionNeeded`.
    async fn ran_compaction_at(
        context_window: u32,
        reserve_tokens: u32,
        prompt_tokens: u32,
    ) -> bool {
        ran_compaction_with_limit(context_window, reserve_tokens, None, prompt_tokens).await
    }

    async fn ran_compaction_with_limit(
        context_window: u32,
        reserve_tokens: u32,
        compact_at_tokens: Option<u32>,
        prompt_tokens: u32,
    ) -> bool {
        use flashmind_types::TokenUsage;
        use futures::StreamExt;

        let provider = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("ok".into()),
            StreamEvent::Usage(TokenUsage {
                prompt_tokens,
                completion_tokens: 1,
                total_tokens: prompt_tokens + 1,
                ..Default::default()
            }),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = test_agent(provider);
        agent.auto_compact = false; // yield CompactionNeeded instead of compacting
        agent.context_window = context_window;
        agent.reserve_tokens = reserve_tokens;
        agent.compact_at_tokens = compact_at_tokens;

        let mut conversation = Conversation::new();
        conversation.set_system("sys");

        let s = agent.start(
            &mut conversation,
            CancellationToken::new(),
            AgentInput::User {
                content: "hi".into(),
                context: None,
                parts: None,
            },
            None,
        );
        tokio::pin!(s);
        let mut compaction_needed = false;
        while let Some(ev) = s.next().await {
            if matches!(ev, AgentEvent::CompactionNeeded { .. }) {
                compaction_needed = true;
            }
        }
        compaction_needed
    }

    #[tokio::test]
    async fn reserve_tokens_is_primary_trigger() {
        // window 128k, reserve 16k → threshold = max(111_616, floor 102_400) = 111_616.
        assert!(ran_compaction_at(128_000, 16_384, 112_000).await);
        // Between the floor and the reserve-based threshold: no trigger.
        assert!(!ran_compaction_at(128_000, 16_384, 111_000).await);
    }

    #[tokio::test]
    async fn compact_threshold_is_floor_for_small_windows() {
        // window 8k, reserve 16k → reserve_based saturates to 0; floor 0.8*8k=6_400 wins.
        assert!(ran_compaction_at(8_000, 16_384, 6_500).await);
        assert!(!ran_compaction_at(8_000, 16_384, 6_000).await);
    }

    #[tokio::test]
    async fn token_limit_moves_the_trigger_earlier_but_never_later() {
        assert!(ran_compaction_with_limit(128_000, 16_384, Some(50_000), 60_000).await);
        assert!(!ran_compaction_with_limit(128_000, 16_384, Some(50_000), 40_000).await);
        // A limit past the reserve-based trigger cannot overflow the window.
        assert!(ran_compaction_with_limit(128_000, 16_384, Some(500_000), 112_000).await);
    }

    #[tokio::test]
    async fn test_cancel_on_drop() {
        use futures::StreamExt;

        let provider = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Hello".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();

        let cancel_token = CancellationToken::new();
        {
            let s = agent.start(
                &mut conversation,
                cancel_token.clone(),
                AgentInput::User {
                    content: "Hi".into(),
                    context: None,
                    parts: None,
                },
                None,
            );
            tokio::pin!(s);
            // Poll once to enter the stream body (creates CancelOnDrop guard),
            // then drop without consuming.
            let _ = s.next().await;
        }
        // Dropping the stream cancels the agent's child token, not the parent.
        // The parent should remain alive for reuse.
        assert!(!cancel_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_conversation_persists_across_turns() {
        use futures::StreamExt;

        let provider = Arc::new(MockProvider::new(vec![
            vec![
                StreamEvent::ContentDelta("First".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ],
            vec![
                StreamEvent::ContentDelta("Second".into()),
                StreamEvent::Finished(FinishReason::Stop),
            ],
        ]));

        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();

        {
            let s = agent.start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::User {
                    content: "Turn 1".into(),
                    context: None,
                    parts: None,
                },
                None,
            );
            tokio::pin!(s);
            while s.next().await.is_some() {}
        }

        let entries_after_first = conversation.entries().len();
        assert!(entries_after_first >= 2);

        {
            let s = agent.start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::User {
                    content: "Turn 2".into(),
                    context: None,
                    parts: None,
                },
                None,
            );
            tokio::pin!(s);
            while s.next().await.is_some() {}
        }

        assert!(conversation.entries().len() > entries_after_first);
    }

    #[test]
    fn builder_minimal() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let _agent = Agent::builder(provider).build_sync();
    }

    #[test]
    fn builder_with_tools() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let tools = ToolRegistry::new();
        let agent = Agent::builder(provider).tools(tools).build_sync();
        assert!(agent.tools().list().is_empty());
    }

    #[test]
    fn builder_with_custom_llm() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let llm = AgentLlmConfig {
            model: "anthropic:claude-sonnet-4-20250514".parse().unwrap(),
            max_tokens: Some(4096),
            reasoning: ReasoningLevel::Medium,
            sampling: SamplingParams {
                temperature: Some(dec!(0.3)),
                ..Default::default()
            },
            user: None,
            provider_preferences: None,
            session_id: None,
        };
        let agent = Agent::builder(provider).llm(llm).build_sync();
        assert_eq!(agent.llm().sampling.temperature, Some(dec!(0.3)));
        assert_eq!(agent.llm().max_tokens, Some(4096));
    }

    #[test]
    fn builder_default_llm_uses_provider() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider).build_sync();
        assert_eq!(agent.llm().model.provider, Provider::Ollama);
    }

    #[test]
    fn builder_keeps_workspace_root() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider)
            .working_dir("workspace")
            .build_sync();
        assert_eq!(agent.working_dir(), Some(Path::new("workspace")));
    }

    #[tokio::test]
    async fn builder_agent_runs() {
        use futures::StreamExt;

        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Built!".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = Agent::builder(provider).build_sync();
        let mut conversation = Conversation::new();

        let mut done_text = String::new();
        let s = agent.start(
            &mut conversation,
            CancellationToken::new(),
            AgentInput::user("Hi"),
            None,
        );
        tokio::pin!(s);
        while let Some(ev) = s.next().await {
            if let AgentEvent::Done(t) = ev {
                done_text = t;
            }
        }
        assert_eq!(done_text, "Built!");
    }

    #[test]
    fn check_iteration_limits_allows_normal() {
        let mut conversation = Conversation::new();
        let result = check_iteration_limits(&mut conversation, Some(100), Some(90), 1);
        assert!(result.is_ok());
    }

    #[test]
    fn check_iteration_limits_warns_at_threshold() {
        let mut conversation = Conversation::new();
        let result = check_iteration_limits(&mut conversation, Some(100), Some(90), 90);
        assert!(result.is_ok());
        let has_warning = conversation.entries().iter().any(|e| e.is_user());
        assert!(has_warning);
    }

    #[test]
    fn check_iteration_limits_aborts_at_max() {
        let mut conversation = Conversation::new();
        let result = check_iteration_limits(&mut conversation, Some(100), Some(90), 101);
        assert!(result.is_err());
    }

    #[test]
    fn check_iteration_limits_no_limit() {
        let mut conversation = Conversation::new();
        let result = check_iteration_limits(&mut conversation, None, None, 9999);
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn workspace_switch_applies_before_the_next_call_without_changing_other_agents() {
        struct WorkspaceTool;
        #[async_trait]
        impl Tool for WorkspaceTool {
            fn name(&self) -> &str {
                "workspace"
            }
            fn description(&self) -> &str {
                "Test workspace switching"
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type":"object"})
            }
            fn concurrent(&self) -> bool {
                false
            }
            async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
                let result = ToolResult::success(
                    ctx.tool_call_id,
                    ctx.working_dir.unwrap().display().to_string(),
                );
                Ok(match ctx.args["path"].as_str() {
                    Some(path) => result.with_working_dir(PathBuf::from(path)),
                    None => result,
                })
            }
        }
        let root = PathBuf::from("/srv/project");
        let target = root.join(".worktrees/task");
        let mut reply = Vec::new();
        for (index, (id, args)) in [
            ("switch", serde_json::json!({"path":target})),
            ("read", serde_json::json!({})),
        ]
        .into_iter()
        .enumerate()
        {
            reply.push(StreamEvent::ToolCallStart {
                index,
                id: id.into(),
                name: "workspace".into(),
            });
            reply.push(StreamEvent::ToolCallDelta {
                index,
                arguments: args.to_string(),
            });
        }
        reply.push(StreamEvent::Finished(FinishReason::ToolCalls));
        let provider = Arc::new(MockProvider::new(vec![reply]));
        let mut agent = test_agent(provider.clone());
        agent.set_working_dir(root.clone());
        agent.allow_parallel_tool_calls = true;
        agent.tools_mut().register(Arc::new(WorkspaceTool));
        let mut other = test_agent(provider);
        other.set_working_dir(root.clone());
        let mut conversation = Conversation::new();
        let events = agent
            .start(
                &mut conversation,
                CancellationToken::new(),
                AgentInput::user("Switch and read"),
                None,
            )
            .collect::<Vec<_>>()
            .await;
        assert_eq!(agent.working_dir(), Some(target.as_path()));
        assert_eq!(other.working_dir(), Some(root.as_path()));
        let outputs: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolResult { name, output, .. } if name == "workspace" => {
                    Some(output.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(outputs, [root.to_str().unwrap(), target.to_str().unwrap()]);
        let workspace_note = conversation
            .entries()
            .iter()
            .position(|entry| {
                entry.is_developer() && entry.content().contains(target.to_str().unwrap())
            })
            .unwrap();
        let last_tool = conversation
            .entries()
            .iter()
            .rposition(|entry| matches!(entry.kind, EntryKind::Tool { .. }))
            .unwrap();
        assert!(
            workspace_note > last_tool,
            "Workspace context must follow all tool responses"
        );
    }
}
