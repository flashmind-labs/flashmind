use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tokio_util::sync::CancellationToken;

use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, AgentStream, AliasedModel, FinishReason, InjectEvent,
    InjectQueue, LlmProvider, Model, ModelCapabilities, Outcome, ReasoningLevel, SamplingParams,
    TokenUsage, ToolRegistry, TurnResult, TurnStatus, TurnUsage,
};

use crate::conversation::{Conversation, ConversationEntry};
use crate::streaming::{LlmResponse, stream_llm_response};

/// Default context window assumed when the provider doesn't report one.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

/// Result of a compaction check — either retry the turn or reset context to start.
#[derive(Debug, Clone, Copy)]
pub enum CompactOutcome {
    /// The model hit max output length. Retry after compacting (keeps current content).
    RetryAfterLength,
    /// Proactive threshold exceeded. Compaction succeeded; continue from here.
    ResetStart,
}

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

/// The main agent — owns provider, tool registry, and LLM config. Conversation
/// is passed in by the caller, keeping the agent stateless between turns.
///
/// Constructed via [`Agent::builder`] or [`Agent::new`], then configured with
/// builder-style helpers before calling [`start`](Self::start).
///
/// # Architecture
///
/// Each call to [`start`](Self::start) creates a stream of [`AgentEvent`] values
/// that drive the full loop: LLM completion → tool execution → compaction check → next iteration.
/// The returned stream includes a [`CancellationToken`] (accessible via the
/// [`AgentEvent::Started`] variant) for external abort.
///
/// # Examples
///
/// ```rust,ignore
/// let agent = Agent::builder(provider)
///     .scope("repl")
///     .tools(tools)
///     .max_iterations(100)
///     .build();
/// ```
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    llm: AgentLlmConfig,
    capabilities: ModelCapabilities,
    context_window: u32,
    scope: String,
    warn_iterations: Option<usize>,
    max_iterations: Option<usize>,
    working_dir: Option<PathBuf>,
    downloads_dir: Option<PathBuf>,
    system_prompt: Option<String>,
}

/// Builder for constructing an [`Agent`] with sensible defaults.
///
/// Only a provider is required. Everything else has defaults:
/// - **scope**: `"default"`
/// - **tools**: empty registry
/// - **llm**: provider's default model, temperature 0.7, no reasoning
/// - **max_iterations**: unlimited (with warnings at reasonable thresholds)
/// - **working_dir**: none (no path resolution)
/// - **downloads_dir**: none (file attachments discarded)
/// - **system_prompt**: none (set via `Conversation::prepend` instead)
pub struct AgentBuilder {
    provider: Arc<dyn LlmProvider>,
    tools: Option<ToolRegistry>,
    llm: Option<AgentLlmConfig>,
    scope: Option<String>,
    max_iterations: Option<usize>,
    working_dir: Option<PathBuf>,
    downloads_dir: Option<PathBuf>,
    system_prompt: Option<String>,
}

impl AgentBuilder {
    fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            tools: None,
            llm: None,
            scope: None,
            max_iterations: None,
            working_dir: None,
            downloads_dir: None,
            system_prompt: None,
        }
    }

    /// Set the session scope identifier (e.g. `"repl"`, `"telegram:123"`).
    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Set the tool registry for this agent.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the LLM configuration (model, temperature, reasoning, sampling).
    pub fn llm(mut self, llm: AgentLlmConfig) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the maximum number of tool-call iterations before aborting.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Set the working directory for relative path resolution.
    pub fn working_dir(mut self, dir: PathBuf) -> Self {
        self.working_dir = Some(dir);
        self
    }

    /// Set the directory where file attachments are saved.
    pub fn downloads_dir(mut self, dir: PathBuf) -> Self {
        self.downloads_dir = Some(dir);
        self
    }

    /// Set a system prompt that will be prepended to the conversation.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Build the [`Agent`]. Consumes this builder.
    pub fn build(self) -> Agent {
        use rust_decimal_macros::dec;

        let provider_variant = self.provider.provider();

        let llm = self.llm.unwrap_or_else(|| AgentLlmConfig {
            model: Model {
                provider: provider_variant,
                model: AliasedModel {
                    name: provider_variant.default_model().into(),
                    real_name: None,
                },
            },
            temperature: dec!(0.7),
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams::default(),
        });

        let mut agent = Agent::new(
            self.scope.unwrap_or_else(|| "default".into()),
            self.provider,
            self.tools.unwrap_or_default(),
            llm,
        );

        if let Some(max) = self.max_iterations {
            agent = agent.with_max_iterations(max);
        }

        if let Some(dir) = self.working_dir {
            agent = agent.with_working_dir(dir);
        }

        if let Some(dir) = self.downloads_dir {
            agent = agent.with_downloads_dir(dir);
        }

        if let Some(prompt) = self.system_prompt {
            agent.system_prompt = Some(prompt);
        }

        agent
    }
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
    /// - **`scope`** — session label (e.g. `"telegram:123"`, `"repl"`).
    /// - **`provider`** — primary [`LlmProvider`] for completions.
    /// - **`tools`** — [`ToolRegistry`] of tools the agent can invoke.
    /// - **`llm`** — [`AgentLlmConfig`] controlling model, temperature, etc.
    pub fn new(
        scope: impl Into<String>,
        provider: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        llm: AgentLlmConfig,
    ) -> Self {
        Self {
            provider,
            tools,
            llm,
            capabilities: ModelCapabilities::default(),
            context_window: DEFAULT_CONTEXT_WINDOW,
            warn_iterations: None,
            max_iterations: None,
            scope: scope.into(),
            working_dir: None,
            downloads_dir: None,
            system_prompt: None,
        }
    }

    /// Set the working directory used to resolve relative paths in tool contexts.
    pub fn with_working_dir(self, working_dir: PathBuf) -> Self {
        let canonical = std::fs::canonicalize(&working_dir).unwrap_or(working_dir);
        Self {
            working_dir: Some(canonical),
            ..self
        }
    }

    /// Set the directory where file attachments from the LLM stream are saved.
    pub fn with_downloads_dir(self, dir: PathBuf) -> Self {
        Self {
            downloads_dir: Some(dir),
            ..self
        }
    }

    /// Enforce iteration limits — emit a warning at 90% of `max`, abort at `max`.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self.warn_iterations = Some((max as f64 * 0.9).ceil() as usize);
        self
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// The scope identifier for this agent session (e.g. `"telegram:123"`, `"slack:C012"`).
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Working directory used for relative-path resolution in tool contexts.
    pub fn working_dir(&self) -> Option<&std::path::Path> {
        self.working_dir.as_deref()
    }

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
    /// The returned stream yields events incrementally:
    /// - `Started` — cancel token, injection sender, config snapshot
    /// - `ReasoningDelta` / `TextDelta` — live LLM output as it arrives
    /// - `ToolStart` / `ToolResult` — per-tool lifecycle
    /// - `Status` — compaction progress messages
    /// - `Done` or `Error` — terminal event
    ///
    /// After emitting the terminal event, the conversation is cleaned up
    /// (subagent progress, memories, reminders stripped).
    pub fn start<'a>(
        &'a mut self,
        conversation: &'a mut Conversation,
        input: AgentInput,
    ) -> impl Stream<Item = AgentEvent> + 'a {
        let cancel_token = CancellationToken::new();

        let inject_queue = InjectQueue::new();

        let started_cancel = cancel_token.clone();
        let started_queue = inject_queue.clone();

        async_stream::stream! {
            let _guard = CancelOnDrop(cancel_token.clone());

            let turn_start = Instant::now();
            metrics::counter!("agent.turns_started").increment(1);

            yield AgentEvent::Started {
                cancel_token: started_cancel,
                inject_queue: started_queue,
                sampling: self.llm.clone(),
                profile: None,
                role: None,
            };

            if let Some(sp) = &self.system_prompt
                && !conversation.entries().iter().any(|e| e.is_system())
            {
                conversation.prepend(ConversationEntry::system(sp));
            }

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

            let mut compacted_on_error: u8 = 0;
            let mut final_content = String::new();
            let mut empty_response = false;
            let mut iteration = 0usize;

            let final_result: TurnResult = loop {
                iteration += 1;
                    metrics::counter!("agent.iterations").increment(1);

                for event in inject_queue.drain() {
                    match event {
                        InjectEvent::UserMessage { text, parts } => {
                            match parts {
                                Some(p) if !p.is_empty() => {
                                    conversation.add(ConversationEntry::user_with_parts(&text, p));
                                }
                                _ => {
                                    conversation.add(ConversationEntry::user(&text));
                                }
                            }
                        }
                        InjectEvent::SubagentProgress { id, content, .. } => {
                            conversation.add(ConversationEntry::subagent_progress(&id, &content));
                        }
                        InjectEvent::SubagentError { id, error } => {
                            conversation.add(ConversationEntry::system_message(
                                format!("Subagent {id} failed: {error}")
                            ));
                        }
                    }
                }

                if let Err(e) = check_iteration_limits(
                    conversation,
                    self.max_iterations,
                    self.warn_iterations,
                    iteration,
                ) {
                    break Err(e);
                }

                let (compact_model, compact_provider) = self.compaction_model_and_provider();
                let result = {
                    let mut turn = self.run_turn(
                        conversation,
                        &cancel_token,
                        &*compact_provider,
                        &compact_model,
                        &mut compacted_on_error,
                        &mut empty_response,
                    );
                    while let Some(ev) = turn.next().await {
                        yield ev;
                    }
                    turn.take_result().unwrap_or_else(|| {
                        Err(anyhow::anyhow!("run_turn stream ended without result"))
                    })
                };

                match &result {
                    Ok(TurnStatus::Continue { content, usage }) if !cancel_token.is_cancelled() => {
                        if !content.trim().is_empty() {
                            empty_response = false;
                        }

                        yield AgentEvent::Usage((*usage).into());

                        continue;
                    }
                    _ => break result,
                }
            };

            match &final_result {
                Ok(status) => {
                    let (c, usage) = status.content_and_usage();
                    if !c.is_empty() {
                        final_content = c.to_string();
                    }

                    metrics::gauge!("agent.conversation.entries").set(conversation.entries().len() as f64);
                    metrics::counter!("agent.turns_completed").increment(1);
                    metrics::histogram!("agent.turn.duration_seconds").record(turn_start.elapsed().as_secs_f64());

                    yield AgentEvent::Usage(TokenUsage {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.prompt_tokens + usage.completion_tokens,
                    });

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
                conversation.add(ConversationEntry::assistant("Cancelled by user"));
            }

            conversation.strip_subagent_progress();
            conversation.strip_memories();
            conversation.strip_reminders();
        }
    }

    // -----------------------------------------------------------------------
    // Single turn
    // -----------------------------------------------------------------------

    /// Execute one LLM round-trip: prompt → error recovery → compaction → result.
    ///
    /// Returns an [`AgentStream`] that yields [`AgentEvent`]s during processing
    /// and terminates with a [`TurnResult`]. The caller is responsible for:
    /// - Checking iteration limits before calling this
    /// - Executing any tool calls returned in [`TurnStatus::ToolCalls`]
    ///
    /// `compact_provider` and `compact_model` specify which model to use for
    /// compaction and error recovery (may differ from the active model).
    pub fn run_turn<'a>(
        &'a mut self,
        conversation: &'a mut Conversation,
        cancel_token: &'a CancellationToken,
        compact_provider: &'a dyn LlmProvider,
        compact_model: &'a Model,
        compacted_on_error: &'a mut u8,
        empty_response: &'a mut bool,
    ) -> AgentStream<'a, AgentEvent, TurnResult> {
        AgentStream::new(async_stream::stream! {
            let tool_definitions = self.tools.definitions();

            tracing::info!(
                model = %self.llm.model,
                temperature = %self.llm.temperature,
                reasoning = ?self.llm.reasoning,
                "Running turn"
            );

            let mut llm = stream_llm_response(
                &*self.provider,
                &self.llm,
                cancel_token,
                conversation,
                &tool_definitions,
                self.downloads_dir.as_deref(),
            );
            while let Some(ev) = llm.next().await {
                yield Outcome::Item(ev);
            }

            let resp = match llm.take_result() {
                Some(Ok(r)) => r,
                Some(Err(e)) => {
                    let mut err = handle_llm_error(
                        conversation, compact_provider, compact_model,
                        cancel_token, e, compacted_on_error,
                    );
                    while let Some(ev) = err.next().await {
                        yield Outcome::Item(ev);
                    }
                    yield Outcome::Done(err.take_result().unwrap_or_else(|| {
                        Err(anyhow::anyhow!("handle_llm_error ended without result"))
                    }));
                    return;
                }
                None => {
                    let mut err = handle_llm_error(
                        conversation, compact_provider, compact_model,
                        cancel_token,
                        anyhow::anyhow!("LLM stream ended without result"),
                        compacted_on_error,
                    );
                    while let Some(ev) = err.next().await {
                        yield Outcome::Item(ev);
                    }
                    yield Outcome::Done(err.take_result().unwrap_or_else(|| {
                        Err(anyhow::anyhow!("handle_llm_error ended without result"))
                    }));
                    return;
                }
            };

            let usage = TurnUsage {
                prompt_tokens: resp.prompt_tokens,
                completion_tokens: resp.completion_tokens,
            };

            if !resp.tool_calls.is_empty() {
                conversation.add(ConversationEntry::assistant_with_tool_calls(
                    &resp.content,
                    resp.tool_calls.clone(),
                ));
            } else {
                if resp.content.trim().is_empty() && !*empty_response {
                    *empty_response = true;
                    yield Outcome::Done(Ok(TurnStatus::Continue {
                        content: resp.content,
                        usage,
                    }));
                    return;
                }
                conversation.add(ConversationEntry::assistant(&resp.content));
            }

            // Compaction
            let compact_outcome = {
                if let Some(mut stream) = try_compact_if_needed(
                    conversation, compact_provider, compact_model,
                    self.context_window, self.llm.max_tokens, &resp,
                ) {
                    while let Some(ev) = stream.next().await {
                        yield Outcome::Item(ev);
                    }
                    Some(stream.take_result().unwrap_or(Ok(CompactOutcome::ResetStart)))
                } else {
                    None
                }
            };
            match compact_outcome {
                Some(Ok(CompactOutcome::RetryAfterLength)) => {
                    yield Outcome::Done(Ok(TurnStatus::Continue {
                        content: resp.content,
                        usage,
                    }));
                    return;
                }
                Some(Ok(CompactOutcome::ResetStart)) => {}
                Some(Err(e)) => {
                    yield Outcome::Done(Err(e));
                    return;
                }
                None => {}
            }

            // Return tool calls for the caller to execute
            if !resp.tool_calls.is_empty() {
                yield Outcome::Done(Ok(TurnStatus::ToolCalls {
                    content: resp.content,
                    tool_calls: resp.tool_calls,
                    usage,
                }));
                return;
            }

            yield Outcome::Done(Ok(TurnStatus::Done {
                content: resp.content,
                usage,
            }));
        })
    }

    /// Manually trigger full compaction of the conversation.
    ///
    /// Strips binary parts, truncates long tool outputs, prunes all tool outputs,
    /// strips tool message wrappers, and runs an LLM summarization pass.
    /// Returns the summary text if compaction succeeded, or `None` if the
    /// model returned empty/invalid output.
    ///
    /// Performs a single LLM summarization pass to reduce conversation size.
    /// Useful when the caller knows the context is large but wants to retain
    /// more history than the automatic 80% threshold would allow.
    pub async fn compact_conversation(&self, conversation: &mut Conversation) -> Option<String> {
        let (model, provider) = self.compaction_model_and_provider();
        conversation.truncate_long_tool_outputs(2000);
        conversation.prune_tool_outputs(0);
        conversation.strip_tool_messages();
        conversation.compact_with_llm(&*provider, &model).await
    }
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
        && iteration >= max_iters
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

/// Attempt error recovery for LLM failures with escalating strategy:
/// compact → truncate → strip binary → fail.
pub fn handle_llm_error<'a>(
    conversation: &'a mut Conversation,
    compact_provider: &'a dyn LlmProvider,
    compact_model: &'a Model,
    cancel_token: &'a CancellationToken,
    error: anyhow::Error,
    compacted_on_error: &'a mut u8,
) -> AgentStream<'a, AgentEvent, TurnResult> {
    AgentStream::new(async_stream::stream! {
        let err_msg = error.to_string();

        let is_recoverable = err_msg.contains("maximum context length")
            || err_msg.contains("context_length_exceeded")
            || err_msg.contains("too many tokens")
            || err_msg.contains("exceeds the model's context")
            || err_msg.contains("reduce the length of the input")
            || err_msg.contains("maximum input length")
            || err_msg.contains("failed to parse JSON");

        if is_recoverable && *compacted_on_error < 2 {
            *compacted_on_error += 1;
            let attempt = *compacted_on_error;

            if attempt == 1 {
                tracing::warn!("LLM error (context overflow) — compacting and retrying");
                yield Outcome::Item(AgentEvent::Status(
                    "Request too large — compacting conversation and retrying...".into(),
                ));

                let binary_stripped = conversation.strip_binary_parts();
                if binary_stripped > 0 {
                    tracing::info!("Stripped {binary_stripped} binary parts during error recovery");
                }

                let compacted = conversation
                    .compact_with_llm(compact_provider, compact_model)
                    .await;

                if compacted.is_none() {
                    tracing::warn!("LLM compaction failed on error recovery, escalating");
                }

                let pruned = conversation.prune_tool_outputs(0);
                if pruned > 0 {
                    tracing::info!("Pruned {pruned} tool outputs during error recovery");
                }

                let stripped = conversation.strip_tool_messages();
                if stripped > 0 {
                    tracing::info!("Stripped {stripped} tool messages during error recovery");
                }

                if compacted.is_none() && pruned == 0 && stripped == 0 && binary_stripped == 0 {
                    tracing::warn!("No compaction possible — truncating to last exchange");
                    conversation.truncate_to_last_exchange();
                }
            } else {
                tracing::warn!("Context still overflowing after compaction — truncating to last exchange");
                yield Outcome::Item(AgentEvent::Status(
                    "Still too large — truncating to last exchange...".into(),
                ));
                conversation.truncate_to_last_exchange();
            }

            yield Outcome::Done(Ok(TurnStatus::Continue {
                content: String::new(),
                usage: TurnUsage::default(),
            }));
            return;
        }

        if cancel_token.is_cancelled() {
            yield Outcome::Done(Ok(TurnStatus::Done {
                content: String::new(),
                usage: TurnUsage::default(),
            }));
            return;
        }

        let binary_stripped = conversation.strip_binary_parts();
        if binary_stripped > 0 {
            tracing::warn!("LLM error with binary content — stripped {binary_stripped} parts and retrying");
            yield Outcome::Item(AgentEvent::Status(
                "Model rejected request — stripped images/documents and retrying...".into(),
            ));
            yield Outcome::Done(Ok(TurnStatus::Continue {
                content: String::new(),
                usage: TurnUsage::default(),
            }));
            return;
        }

        yield Outcome::Item(AgentEvent::Error(err_msg.clone()));
        yield Outcome::Done(Err(anyhow::anyhow!(err_msg)));
    })
}

/// Check whether a response warrants compaction and return a stream that
/// performs it if so.
///
/// Compaction is triggered in two cases:
/// - **Length**: the model returned `finish_reason=Length` without an explicit
///   `max_tokens` cap, indicating truncation. The response is retried after
///   compacting.
/// - **Proactive**: prompt tokens exceeded 90% of the context window. A
///   summarization pass is run before returning control to the caller.
///
/// Returns `None` when no compaction is needed (prompt below threshold, or
/// `max_tokens` is set so length-based truncation is expected).
pub fn try_compact_if_needed<'a>(
    conversation: &'a mut Conversation,
    compact_provider: &'a dyn LlmProvider,
    compact_model: &'a Model,
    context_window: u32,
    max_tokens: Option<u32>,
    resp: &'a LlmResponse,
) -> Option<AgentStream<'a, AgentEvent, anyhow::Result<CompactOutcome>>> {
    let length_case = resp.finish_reason == FinishReason::Length && max_tokens.is_none();
    let threshold = (context_window as f64 * 0.9) as u32;
    let proactive_case = resp.prompt_tokens > threshold;

    if !length_case && !proactive_case {
        return None;
    }

    Some(AgentStream::new(async_stream::stream! {
        if length_case {
            tracing::warn!("finish_reason=Length with no max_tokens set — compacting and retrying");
            yield Outcome::Item(AgentEvent::Status(
                "Response truncated — compacting conversation...".into(),
            ));
            conversation
                .compact_with_llm(compact_provider, compact_model)
                .await;
            yield Outcome::Done(Ok(CompactOutcome::RetryAfterLength));
            return;
        }

        let mut failed = false;
        {
            use futures::StreamExt;
            let compact_stream = crate::compaction::try_compact(
                conversation,
                resp.prompt_tokens,
                context_window,
                compact_provider,
                compact_model,
            );
            tokio::pin!(compact_stream);
            while let Some(ev) = compact_stream.next().await {
                if matches!(ev, AgentEvent::Error(_)) {
                    failed = true;
                }
                yield Outcome::Item(ev);
            }
        }

        let outcome = if failed {
            Err(anyhow::anyhow!("compaction: LLM summarization failed"))
        } else {
            Ok(CompactOutcome::ResetStart)
        };
        yield Outcome::Done(outcome);
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use async_trait::async_trait;
    use flashmind_types::{
        AliasedModel, CompletionRequest, CompletionStream, FinishReason, Model, Provider,
        ReasoningLevel, SamplingParams, StreamEvent,
    };
    use rust_decimal_macros::dec;

    struct MockProvider {
        responses: std::sync::Mutex<Vec<Vec<StreamEvent>>>,
    }

    impl MockProvider {
        fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn complete(&self, _request: CompletionRequest) -> CompletionStream {
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
            temperature: dec!(0.7),
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams::default(),
        };
        Agent::new("test", provider, ToolRegistry::new(), llm)
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
        conversation.prepend(ConversationEntry::system("You are helpful"));

        let mut deltas = String::new();
        let mut done_text = String::new();

        let s = agent.start(
            &mut conversation,
            AgentInput::User {
                content: "Hi".into(),
                context: None,
                parts: None,
            },
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
    async fn test_cancel_on_drop() {
        use futures::StreamExt;

        let provider = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Hello".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = test_agent(provider);
        let mut conversation = Conversation::new();

        let mut cancel_token = None;
        {
            let s = agent.start(
                &mut conversation,
                AgentInput::User {
                    content: "Hi".into(),
                    context: None,
                    parts: None,
                },
            );
            tokio::pin!(s);
            if let Some(AgentEvent::Started {
                cancel_token: ct, ..
            }) = s.next().await
            {
                cancel_token = Some(ct);
            }
        }
        assert!(cancel_token.unwrap().is_cancelled());
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
                AgentInput::User {
                    content: "Turn 1".into(),
                    context: None,
                    parts: None,
                },
            );
            tokio::pin!(s);
            while s.next().await.is_some() {}
        }

        let entries_after_first = conversation.entries().len();
        assert!(entries_after_first >= 2);

        {
            let s = agent.start(
                &mut conversation,
                AgentInput::User {
                    content: "Turn 2".into(),
                    context: None,
                    parts: None,
                },
            );
            tokio::pin!(s);
            while s.next().await.is_some() {}
        }

        assert!(conversation.entries().len() > entries_after_first);
    }

    #[test]
    fn builder_minimal() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider).build();
        assert_eq!(agent.scope(), "default");
        assert!(agent.working_dir().is_none());
        assert!(agent.max_iterations.is_none());
    }

    #[test]
    fn builder_with_scope() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider).scope("telegram:42").build();
        assert_eq!(agent.scope(), "telegram:42");
    }

    #[test]
    fn builder_with_max_iterations() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider).max_iterations(50).build();
        assert_eq!(agent.max_iterations, Some(50));
        assert_eq!(agent.warn_iterations, Some(45));
    }

    #[test]
    fn builder_with_tools() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let tools = ToolRegistry::new();
        let agent = Agent::builder(provider).tools(tools).build();
        assert!(agent.tools().list().is_empty());
    }

    #[test]
    fn builder_with_custom_llm() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let llm = AgentLlmConfig {
            model: "anthropic:claude-sonnet-4-20250514".parse().unwrap(),
            temperature: dec!(0.3),
            max_tokens: Some(4096),
            reasoning: ReasoningLevel::On,
            sampling: SamplingParams::default(),
        };
        let agent = Agent::builder(provider).llm(llm).build();
        assert_eq!(agent.llm().temperature, dec!(0.3));
        assert_eq!(agent.llm().max_tokens, Some(4096));
    }

    #[test]
    fn builder_default_llm_uses_provider() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![]));
        let agent = Agent::builder(provider).build();
        assert_eq!(agent.llm().model.provider, Provider::Ollama);
    }

    #[tokio::test]
    async fn builder_agent_runs() {
        use futures::StreamExt;

        let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![vec![
            StreamEvent::ContentDelta("Built!".into()),
            StreamEvent::Finished(FinishReason::Stop),
        ]]));

        let mut agent = Agent::builder(provider).scope("test").build();
        let mut conversation = Conversation::new();

        let mut done_text = String::new();
        let s = agent.start(&mut conversation, AgentInput::user("Hi"));
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
        let result = check_iteration_limits(&mut conversation, Some(100), Some(90), 100);
        assert!(result.is_err());
    }

    #[test]
    fn check_iteration_limits_no_limit() {
        let mut conversation = Conversation::new();
        let result = check_iteration_limits(&mut conversation, None, None, 9999);
        assert!(result.is_ok());
    }
}
