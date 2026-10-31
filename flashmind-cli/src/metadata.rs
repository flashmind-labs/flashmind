//! Model metadata fetched independently of terminal input and rendering.

use std::sync::Arc;
use std::time::Duration;

use flashmind_core::Agent;
use flashmind_types::{LlmProvider, Model, ModelCapabilities, ModelPricing};
use tokio::task::{JoinError, JoinSet};

use crate::provider::fetch_pricing;

const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

struct ModelFeatures {
    capabilities: ModelCapabilities,
    context_window: Option<u32>,
}

pub(crate) struct ModelMetadata {
    model: Model,
    provider: Arc<dyn LlmProvider>,
    features: Option<ModelFeatures>,
    pub pricing: ModelPricing,
}

impl ModelMetadata {
    pub fn context_window(&self) -> Option<u32> {
        self.features
            .as_ref()
            .and_then(|features| features.context_window)
    }

    /// Ignore a result if the session has switched model or provider.
    pub fn apply(&self, agent: &mut Agent) -> bool {
        if agent.llm().model != self.model || !Arc::ptr_eq(agent.provider_arc(), &self.provider) {
            return false;
        }
        if let Some(features) = &self.features {
            agent.apply_features(features.capabilities, features.context_window);
        }
        true
    }
}

/// The session owns the lookup task; dropping it cancels unfinished requests.
pub(crate) struct MetadataLoader {
    model: Model,
    provider: Arc<dyn LlmProvider>,
    tasks: JoinSet<ModelMetadata>,
}

impl MetadataLoader {
    pub fn start(agent: &Agent) -> Self {
        let provider = agent.provider_arc().clone();
        let model = agent.llm().model.clone();
        let mut loader = Self {
            model: model.clone(),
            provider: provider.clone(),
            tasks: JoinSet::new(),
        };
        loader.tasks.spawn(async move {
            let features = async {
                let capabilities = provider.capabilities(&model).await;
                let context_window = provider.context_window(&model).await;
                ModelFeatures {
                    capabilities,
                    context_window,
                }
            };
            let (features, pricing) = tokio::join!(
                tokio::time::timeout(FETCH_TIMEOUT, features),
                tokio::time::timeout(FETCH_TIMEOUT, fetch_pricing(&provider, &model)),
            );
            let features = match features {
                Ok(features) => Some(features),
                Err(error) => {
                    tracing::warn!(%error, "model features timed out");
                    None
                }
            };
            let pricing = pricing.unwrap_or_else(|error| {
                tracing::warn!(%error, "model pricing timed out");
                ModelPricing::default()
            });
            ModelMetadata {
                model,
                provider,
                features,
                pricing,
            }
        });
        loader
    }

    pub fn discard_if_changed(&mut self, agent: &Agent) {
        if agent.llm().model != self.model || !Arc::ptr_eq(agent.provider_arc(), &self.provider) {
            self.tasks = JoinSet::new();
        }
    }

    pub fn is_pending(&self) -> bool {
        !self.tasks.is_empty()
    }

    pub fn try_ready(&mut self) -> Option<ModelMetadata> {
        self.tasks.try_join_next().and_then(Self::result)
    }

    pub async fn ready(&mut self) -> Option<ModelMetadata> {
        self.tasks.join_next().await.and_then(Self::result)
    }

    fn result(result: Result<ModelMetadata, JoinError>) -> Option<ModelMetadata> {
        match result {
            Ok(metadata) => Some(metadata),
            Err(error) => {
                tracing::warn!(%error, "model metadata task failed");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use flashmind_types::{
        CompletionRequest, CompletionStream, ModelInfo, Provider, ReasoningLevel,
    };
    use rust_decimal::Decimal;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;

    #[derive(Default)]
    struct PausedProvider {
        started: Notify,
        release: Notify,
        cancelled: CancellationToken,
    }

    #[async_trait]
    impl LlmProvider for PausedProvider {
        fn name(&self) -> &str {
            "paused"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::pending())
        }
        async fn capabilities(&self, _: &Model) -> ModelCapabilities {
            let _guard = self.cancelled.clone().drop_guard();
            self.started.notify_one();
            self.release.notified().await;
            ModelCapabilities {
                images: true,
                reasoning: true,
                ..Default::default()
            }
        }
        async fn context_window(&self, _: &Model) -> Option<u32> {
            Some(64_000)
        }
        async fn list_models(&self) -> Option<Vec<ModelInfo>> {
            Some(vec![ModelInfo {
                image: Default::default(),
                id: "example".into(),
                name: None,
                created_at: None,
                context_length: Some(64_000),
                max_completion_tokens: None,
                capabilities: ModelCapabilities::default(),
                categories: vec![],
                pricing: ModelPricing {
                    prompt: Some(Decimal::new(1, 7)),
                    ..Default::default()
                },
            }])
        }
    }

    fn agent(provider: Arc<dyn LlmProvider>) -> Agent {
        let mut agent = Agent::builder(provider).build_sync();
        agent.llm_mut().model = "ollama:example".parse().unwrap();
        agent.llm_mut().reasoning = ReasoningLevel::High;
        agent
    }

    #[tokio::test]
    async fn lookup_returns_immediately_and_publishes_features_and_prices_when_ready() {
        let provider = Arc::new(PausedProvider::default());
        let mut agent = agent(provider.clone());
        let mut loader = MetadataLoader::start(&agent);
        provider.started.notified().await;
        assert!(loader.try_ready().is_none());
        assert!(!agent.capabilities().images);
        provider.release.notify_one();
        let result = loader.ready().await.unwrap();
        assert!(result.apply(&mut agent));
        assert!(agent.capabilities().images);
        assert_eq!(agent.context_window(), 64_000);
        assert_eq!(agent.llm().reasoning, ReasoningLevel::High);
        assert_eq!(result.pricing.prompt, Some(Decimal::new(1, 7)));
        assert!(!loader.is_pending());
    }

    #[tokio::test]
    async fn late_metadata_does_not_overwrite_a_switched_model_or_provider() {
        let provider = Arc::new(PausedProvider::default());
        let mut agent = agent(provider.clone());
        let mut loader = MetadataLoader::start(&agent);
        provider.release.notify_one();
        let result = loader.ready().await.unwrap();
        agent.llm_mut().model = "ollama:replacement".parse().unwrap();
        assert!(!result.apply(&mut agent));
        agent.llm_mut().model = "ollama:example".parse().unwrap();
        agent.set_provider(Arc::new(PausedProvider::default()));
        assert!(!result.apply(&mut agent));
        assert!(!agent.capabilities().images);
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_features_keep_existing_features_and_preserve_available_prices() {
        let provider = Arc::new(PausedProvider::default());
        let mut agent = agent(provider.clone());
        let mut loader = MetadataLoader::start(&agent);
        provider.started.notified().await;
        tokio::time::advance(FETCH_TIMEOUT).await;
        let result = loader.ready().await.unwrap();
        assert!(result.apply(&mut agent));
        assert!(!agent.capabilities().images);
        assert_eq!(agent.llm().reasoning, ReasoningLevel::High);
        assert_eq!(result.pricing.prompt, Some(Decimal::new(1, 7)));
    }

    #[tokio::test]
    async fn dropping_the_loader_cancels_its_task() {
        let provider = Arc::new(PausedProvider::default());
        let agent = agent(provider.clone());
        let loader = MetadataLoader::start(&agent);
        provider.started.notified().await;
        drop(loader);
        provider.cancelled.cancelled().await;
    }
}
