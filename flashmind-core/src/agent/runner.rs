//! Per-run overrides for the agent loop.

use std::sync::Arc;

use flashmind_types::{AgentEvent, AgentInput, LlmProvider, Model};
use futures_util::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::{Agent, DEFAULT_CONTEXT_WINDOW};
use crate::Conversation;

/// Configures one run, leaving the agent's defaults unchanged.
///
/// The stream owns a separate agent configuration. Dropping it cancels its
/// work without changing the original agent. Registered tools and background
/// event sources remain shared with the original agent. The runner exclusively
/// borrows the agent until its stream is dropped, so two runs cannot consume
/// the same background events at once. The registry is a snapshot: tools added
/// or removed by the host are picked up when the next run starts.
///
/// ```no_run
/// use flashmind_core::{Agent, CancellationToken, Conversation};
/// use flashmind_types::{AgentEvent, AgentInput, Model};
/// use futures_util::StreamExt;
///
/// async fn review(agent: &mut Agent, conversation: &mut Conversation) -> anyhow::Result<()> {
///     let model: Model = "ollama:example".parse()?;
///     let stream = agent.run(conversation, AgentInput::user("Review this change"))
///         .model(model)
///         .max_iterations(20)
///         .start(CancellationToken::new());
///     tokio::pin!(stream);
///     while let Some(event) = stream.next().await {
///         if let AgentEvent::TextDelta(text) = event {
///             print!("{text}");
///         }
///     }
///     Ok(())
/// }
/// ```
pub struct AgentRunner<'a> {
    agent: &'a mut Agent,
    conversation: &'a mut Conversation,
    input: AgentInput,
    model: Option<Model>,
    provider: Option<Arc<dyn LlmProvider>>,
    max_iterations: Option<usize>,
}

impl<'a> AgentRunner<'a> {
    pub(super) fn new(
        agent: &'a mut Agent,
        conversation: &'a mut Conversation,
        input: AgentInput,
    ) -> Self {
        Self {
            agent,
            conversation,
            input,
            model: None,
            provider: None,
            max_iterations: None,
        }
    }

    /// Select a model for this run. A model from another provider requires
    /// an explicit provider through [`provider`](Self::provider).
    pub fn model(mut self, model: Model) -> Self {
        self.model = Some(model);
        self
    }

    /// Select the provider for this run; it must match the selected model.
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Limit model calls in this run.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Stream the run's events. An invalid provider/model pair yields an error
    /// event before the conversation or any tools are changed.
    pub fn start(self, cancel: CancellationToken) -> impl Stream<Item = AgentEvent> + 'a {
        let mut agent = self.agent.for_run();
        async_stream::stream! {
            let model = self.model.unwrap_or_else(|| agent.llm.model.clone());
            if self.provider.is_none() && model.provider != agent.llm.model.provider {
                yield AgentEvent::Error(format!("Model {} requires an explicit provider", model));
                return;
            }
            if let Some(provider) = self.provider.as_ref()
                && provider.provider() != model.provider
            {
                yield AgentEvent::Error(format!("Provider {} does not match model {}", provider.provider(), model));
                return;
            }
            let provider_changed = self.provider.is_some();
            let model_changed = model != agent.llm.model;
            if model_changed || provider_changed {
                let provider = self.provider.unwrap_or_else(|| agent.provider.clone());
                // Unknown model windows must not inherit the previous model's window.
                agent.context_window = DEFAULT_CONTEXT_WINDOW;
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        yield AgentEvent::Error("cancelled".into());
                        return;
                    }
                    _ = agent.switch_model(provider, model) => {}
                }
            }
            let stream = agent.start(self.conversation, cancel, self.input, self.max_iterations);
            tokio::pin!(stream);
            while let Some(event) = stream.next().await {
                yield event;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, Model, ModelCapabilities, Provider,
        StreamEvent,
    };
    use futures_util::StreamExt;

    use super::*;

    struct RecordingProvider {
        provider: Provider,
        models: Mutex<Vec<Model>>,
        features: Mutex<Vec<Model>>,
    }

    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            self.provider.clone()
        }
        async fn capabilities(&self, model: &Model) -> ModelCapabilities {
            self.features
                .lock()
                .expect("test mutex is not poisoned")
                .push(model.clone());
            ModelCapabilities::default()
        }
        async fn context_window(&self, _: &Model) -> Option<u32> {
            Some(64_000)
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.models
                .lock()
                .expect("test mutex is not poisoned")
                .push(request.model);
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamEvent::ContentDelta("done".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    fn provider(provider: Provider) -> Arc<RecordingProvider> {
        Arc::new(RecordingProvider {
            provider,
            models: Mutex::new(Vec::new()),
            features: Mutex::new(Vec::new()),
        })
    }

    #[tokio::test]
    async fn model_override_is_local_to_run_and_default_is_used_afterwards() {
        let provider = provider(Provider::Ollama);
        let mut agent = Agent::builder(provider.clone()).build_sync();
        let original = agent.llm().model.clone();
        let selected: Model = "ollama:alternate".parse().expect("valid test model");
        let mut conversation = Conversation::new();
        let events: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("first"))
            .model(selected.clone())
            .max_iterations(3)
            .start(CancellationToken::new())
            .collect()
            .await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Done(_)))
        );
        assert_eq!(agent.llm().model, original);
        let _: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("second"))
            .start(CancellationToken::new())
            .collect()
            .await;
        assert_eq!(
            *provider.models.lock().expect("test mutex is not poisoned"),
            vec![selected, original]
        );
    }

    #[tokio::test]
    async fn cross_provider_requires_matching_explicit_provider() {
        let first = provider(Provider::Ollama);
        let second = provider(Provider::Anthropic);
        let mut agent = Agent::builder(first.clone()).build_sync();
        let model: Model = "anthropic:example".parse().expect("valid test model");
        let mut conversation = Conversation::new();
        let events: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("first"))
            .model(model.clone())
            .start(CancellationToken::new())
            .collect()
            .await;
        assert!(matches!(&events[..], [AgentEvent::Error(_)]));
        assert!(
            first
                .models
                .lock()
                .expect("test mutex is not poisoned")
                .is_empty()
        );
        let events: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("second"))
            .model(model.clone())
            .provider(second.clone())
            .start(CancellationToken::new())
            .collect()
            .await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Done(_)))
        );
        assert_eq!(
            *second.models.lock().expect("test mutex is not poisoned"),
            vec![model.clone()]
        );
        assert_eq!(
            *second.features.lock().expect("test mutex is not poisoned"),
            vec![model]
        );
        assert_eq!(agent.llm().model.provider, Provider::Ollama);
    }

    #[tokio::test]
    async fn iteration_limit_is_applied_only_to_this_run() {
        let provider = provider(Provider::Ollama);
        let mut agent = Agent::builder(provider.clone()).build_sync();
        let mut conversation = Conversation::new();
        let events: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("first"))
            .max_iterations(0)
            .start(CancellationToken::new())
            .collect()
            .await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Error(_)))
        );
        assert!(
            provider
                .models
                .lock()
                .expect("test mutex is not poisoned")
                .is_empty()
        );
        let events: Vec<_> = agent
            .run(&mut conversation, AgentInput::user("second"))
            .start(CancellationToken::new())
            .collect()
            .await;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::Done(_)))
        );
    }

    #[tokio::test]
    async fn dropping_stream_does_not_change_agent_defaults() {
        let mut agent = Agent::builder(provider(Provider::Ollama)).build_sync();
        let original = agent.llm().model.clone();
        let mut conversation = Conversation::new();
        {
            let stream = agent
                .run(&mut conversation, AgentInput::user("first"))
                .model("ollama:alternate".parse().expect("valid test model"))
                .start(CancellationToken::new());
            tokio::pin!(stream);
            assert!(stream.next().await.is_some());
        }
        assert_eq!(agent.llm().model, original);
    }
}
