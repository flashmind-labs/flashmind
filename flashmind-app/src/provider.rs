//! Provider factory methods for instantiating LLM providers.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_types::LlmProvider;
use flashmind_types::model::Provider;

use crate::config::{AppConfig, ProviderConfig};

// ---------------------------------------------------------------------------
// Provider construction
// ---------------------------------------------------------------------------

impl AppConfig {
    /// Build a provider by its [`Provider`] enum variant.
    pub fn build_provider_for(&self, provider: &Provider) -> Result<Arc<dyn LlmProvider>> {
        let pc = self
            .provider_config_for(provider)
            .with_context(|| {
                format!(
                    "no config for provider {provider} — add [[llm.providers]] with name = \"{provider}\""
                )
            })?;
        make_provider(pc)
    }

    /// Build the first (active) provider from the config.
    pub fn build_active_provider(&self) -> Result<Arc<dyn LlmProvider>> {
        let pc = self.active_provider_config()?;
        make_provider(pc)
    }
}

/// Construct an [`LlmProvider`] from a [`ProviderConfig`].
pub fn make_provider(pc: &ProviderConfig) -> Result<Arc<dyn LlmProvider>> {
    let provider: Arc<dyn LlmProvider> = match pc.name {
        Provider::Ollama => Arc::new(OllamaProvider::new(pc.url.clone(), pc.num_ctx)),
        Provider::OpenRouter => {
            let key = pc.api_key.clone().context("openrouter requires api_key")?;
            Arc::new(OpenRouterProvider::new(key))
        }
        Provider::Anthropic => {
            let key = pc.api_key.clone().context("anthropic requires api_key")?;
            let limiter = Arc::new(
                ratelimit::Ratelimiter::builder(50)
                    .max_tokens(50)
                    .initial_available(50)
                    .build()
                    .expect("rate limiter"),
            );
            Arc::new(AnthropicProvider::new(key, limiter))
        }
        Provider::OpenAi => {
            let key = pc.api_key.clone();
            Arc::new(OpenAiProvider::new(
                pc.url.clone(),
                key,
                Default::default(),
                false,
                None,
            ))
        }
        Provider::Connect => bail!("connect provider not supported"),
    };
    Ok(provider)
}
