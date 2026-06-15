use std::sync::Arc;

use anyhow::{Context, Result, bail};

use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_types::{LlmProvider, Model, ModelPricing, Provider};

use crate::Cli;
use crate::config::Config;

pub fn resolve_model(cli: &Cli, config: &Config) -> Result<Model> {
    let raw = cli
        .model
        .as_deref()
        .or(config.model.as_deref())
        .unwrap_or("ollama:llama3.2");
    raw.parse()
        .with_context(|| format!("invalid model string: {raw}"))
}

pub fn build_provider(model: &Model, config: &Config) -> Result<Arc<dyn LlmProvider>> {
    match model.provider {
        Provider::Ollama => {
            let p = OllamaProvider::new(config.ollama_url.clone(), None)?;
            Ok(Arc::new(p))
        }
        Provider::OpenRouter => {
            let key = config
                .openrouter_api_key
                .as_ref()
                .context("openrouter_api_key required for OpenRouter models")?;
            Ok(Arc::new(OpenRouterProvider::new(key.clone())))
        }
        Provider::Anthropic => {
            let key = config
                .anthropic_api_key
                .as_ref()
                .context("anthropic_api_key required for Anthropic models")?;
            Ok(Arc::new(AnthropicProvider::new(key.clone())))
        }
        Provider::OpenAi => {
            let key = config
                .openai_api_key
                .as_ref()
                .context("openai_api_key required for OpenAI models")?;
            let base = config
                .openai_base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1");
            let p = OpenAiProvider::builder(base).api_key(key).build()?;
            Ok(Arc::new(p))
        }
        other => bail!("unsupported provider: {other:?}"),
    }
}

pub async fn fetch_pricing(provider: &Arc<dyn LlmProvider>, model: &Model) -> ModelPricing {
    let name = model.capability_name();
    let models = provider.list_models().await.unwrap_or_default();
    models
        .iter()
        .find(|m| m.id == name)
        .map(|m| m.pricing.clone())
        .unwrap_or_default()
}
