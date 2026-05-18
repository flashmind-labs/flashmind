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
//!     .scope("my-app")
//!     .tools(tools)
//!     .build();
//! ```
//!
//! # Agent Loop Flow
//!
//! 1. Stream LLM response → emit `AgentEvent`s
//! 2. Execute any tool calls from the response
//! 3. Compact conversation if context pressure detected
//! 4. Repeat until done or max iterations reached

use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tokio_util::sync::CancellationToken;

use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, AgentStream, AliasedModel, CompactionReason,
    FinishReason, LlmProvider, Model, ModelCapabilities, Outcome, ReasoningLevel, SamplingParams,
    TokenUsage, ToolRegistry, TurnResult, TurnStatus, TurnUsage,
};

use crate::conversation::{Conversation, ConversationEntry};
use crate::streaming::stream_llm_response;

/// Default context window assumed when the provider doesn't report one.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

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
}

impl AgentBuilder {
    /// Create an [`AgentBuilder`] with the given provider.
    fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            tools: None,
            llm: None,
        }
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
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.7)),
                ..Default::default()
            },
        });

        Agent::new(self.provider, self.tools.unwrap_or_default(), llm)
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
        }
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

            let mut compacted_on_error: u8 = 0;
            let mut final_content = String::new();
            let mut empty_response = false;
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

                match result {
                    Ok(TurnStatus::Continue { ref content, usage }) if !cancel_token.is_cancelled() => {
                        if !content.trim().is_empty() {
                            empty_response = false;
                        }
                        yield AgentEvent::Usage(usage.into());
                        continue;
                    }

                    Ok(TurnStatus::CompactionNeeded { content, usage, reason }) => {
                        yield AgentEvent::Usage(usage.into());
                        let (compact_model, compact_provider) = self.compaction_model_and_provider();
                        match reason {
                            CompactionReason::OutputLength => {
                                yield AgentEvent::Status("Response truncated — compacting conversation...".into());
                                conversation
                                    .compact_with_llm(&*compact_provider, &compact_model)
                                    .await;
                            }
                            CompactionReason::ContextThreshold(prompt_tokens) => {
                                use futures::StreamExt as _;
                                let compact_stream = crate::compaction::try_compact(
                                    conversation,
                                    prompt_tokens,
                                    self.context_window,
                                    &*compact_provider,
                                    &compact_model,
                                );
                                tokio::pin!(compact_stream);
                                while let Some(ev) = compact_stream.next().await {
                                    yield ev;
                                }
                            }
                        }
                        if !content.trim().is_empty() {
                            empty_response = false;
                        }
                        continue;
                    }

                    Err(ref e) if !cancel_token.is_cancelled() => {
                        let err_msg = e.to_string();
                        let is_recoverable = err_msg.contains("maximum context length")
                            || err_msg.contains("context_length_exceeded")
                            || err_msg.contains("too many tokens")
                            || err_msg.contains("exceeds the model's context")
                            || err_msg.contains("reduce the length of the input")
                            || err_msg.contains("maximum input length")
                            || err_msg.contains("failed to parse JSON");

                        if is_recoverable && compacted_on_error < 2 {
                            compacted_on_error += 1;
                            let (compact_model, compact_provider) = self.compaction_model_and_provider();
                            let mut err = handle_llm_error(
                                conversation, &*compact_provider, &compact_model,
                                &cancel_token, compacted_on_error,
                            );
                            while let Some(ev) = err.next().await {
                                yield ev;
                            }
                            continue;
                        }

                        let binary_stripped = conversation.strip_binary_parts();
                        if binary_stripped > 0 {
                            yield AgentEvent::Status(
                                "Model rejected request — stripped images/documents and retrying...".into(),
                            );
                            continue;
                        }

                        break Err(anyhow::anyhow!(err_msg));
                    }

                    Ok(TurnStatus::ToolCalls { ref content, ref tool_calls, usage }) if !cancel_token.is_cancelled() => {
                        if !content.trim().is_empty() {
                            empty_response = false;
                        }
                        yield AgentEvent::Usage(usage.into());

                        for tc in tool_calls {
                            let humanized = self.tools.humanize(tc);
                            yield AgentEvent::ToolStart {
                                name: tc.name.clone(),
                                id: tc.id.clone(),
                                humanized,
                            };

                            let tool_start = Instant::now();
                            let result = self.tools.execute(tc, None, &cancel_token).await;
                            let elapsed_ms = tool_start.elapsed().as_millis() as u64;

                            conversation.add(ConversationEntry::tool(&tc.id, result.output()));

                            for diff in result.diffs() {
                                yield AgentEvent::FileDiff {
                                    path: diff.path.clone(),
                                    diff: diff.diff.clone(),
                                };
                            }

                            if result.is_interrupt() {
                                yield AgentEvent::Interrupted {
                                    tool_call_id: tc.id.clone(),
                                    tool_name: tc.name.clone(),
                                    output: result.output(),
                                    payload: result.payload().cloned(),
                                };
                                break;
                            }

                            yield AgentEvent::ToolResult {
                                name: tc.name.clone(),
                                id: tc.id.clone(),
                                output: result.output().to_string(),
                                success: result.is_success(),
                                elapsed_ms,
                                sources: result.sources().to_vec(),
                            };
                        }

                        continue;
                    }

                    Ok(TurnStatus::Done { ref content, usage }) if content.trim().is_empty() && !empty_response => {
                        empty_response = true;
                        yield AgentEvent::Usage(usage.into());
                        continue;
                    }

                    other => break other,
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
            while let Some(ev) = llm.next().await {
                yield Outcome::Item(ev);
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

            let usage = TurnUsage {
                prompt_tokens: resp.prompt_tokens,
                completion_tokens: resp.completion_tokens,
            };

            if !resp.tool_calls.is_empty() {
                conversation.add(ConversationEntry::assistant_with_tool_calls(
                    &resp.content,
                    resp.tool_calls.clone(),
                ));
                yield Outcome::Done(Ok(TurnStatus::ToolCalls {
                    content: resp.content,
                    tool_calls: resp.tool_calls,
                    usage,
                }));
                return;
            }

            conversation.add(ConversationEntry::assistant(&resp.content));

            let threshold = (self.context_window as f64 * 0.9) as u32;
            if resp.finish_reason == FinishReason::Length && self.llm.max_tokens.is_none() {
                yield Outcome::Done(Ok(TurnStatus::CompactionNeeded {
                    content: resp.content,
                    usage,
                    reason: CompactionReason::OutputLength,
                }));
                return;
            }
            if resp.prompt_tokens > threshold {
                yield Outcome::Done(Ok(TurnStatus::CompactionNeeded {
                    content: resp.content,
                    usage,
                    reason: CompactionReason::ContextThreshold(resp.prompt_tokens),
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
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.7)),
                ..Default::default()
            },
        };
        Agent::new(provider, ToolRegistry::new(), llm)
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
        assert!(cancel_token.is_cancelled());
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
        let _agent = Agent::builder(provider).build();
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
            max_tokens: Some(4096),
            reasoning: ReasoningLevel::On,
            sampling: SamplingParams {
                temperature: Some(dec!(0.3)),
                ..Default::default()
            },
        };
        let agent = Agent::builder(provider).llm(llm).build();
        assert_eq!(agent.llm().sampling.temperature, Some(dec!(0.3)));
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

        let mut agent = Agent::builder(provider).build();
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
