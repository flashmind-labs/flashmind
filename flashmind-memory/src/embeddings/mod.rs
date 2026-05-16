//! Embedding providers for vector memory.
//!
//! Converts text strings into fixed-dimensional float vectors for semantic search.
//! Three backends are supported: OpenAI, OpenRouter, and Ollama.
//!
//! # Creating an embedder
//!
//! ```rust,ignore
//! use flashmind_memory::embeddings::{create_embedding_provider, EmbeddingProviderConfig};
//!
//! // From a config struct (e.g., loaded from TOML)
//! let provider = create_embedding_provider(&config).await?;
//!
//! // Direct construction
//! let provider = OllamaEmbedding::new(None); // defaults to localhost:11434
//! ```

mod ollama;
mod openai;
mod openrouter;

pub use ollama::OllamaEmbedding;
pub use openai::OpenAIEmbedding;
pub use openrouter::OpenRouterEmbedding;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{FlashmemError, Result};

/// Trait for embedding providers that convert text to fixed-dimensional float vectors.
///
/// Implementations handle the HTTP communication with the embedding API and
/// return `Vec<f32>` vectors suitable for storage in sqlite-vec.
///
/// # Thread safety
///
/// All implementations are `Send + Sync` so they can be shared across async tasks
/// via `Arc<dyn EmbeddingProvider>`.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch).
    /// Default implementation calls `embed()` sequentially for each text.
    /// Providers may override this to use native batch endpoints when available.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Returns the dimension of the embedding vectors.
    fn dimensions(&self) -> usize;

    /// Returns the provider name for logging.
    fn name(&self) -> &str;
}

/// Embedding provider configuration - supports OpenAI, OpenRouter, or Ollama embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum EmbeddingProviderConfig {
    /// OpenAI embeddings API or OpenAI-compatible endpoint
    OpenAI {
        #[serde(default)]
        api_key: Option<String>,
        model: String,
        #[serde(default)]
        base_url: Option<String>,
    },
    /// OpenRouter embeddings (uses same API key as LLM if not specified)
    OpenRouter {
        api_key: Option<String>,
        model: String,
    },
    /// Ollama local embeddings (no API key needed)
    Ollama {
        #[serde(default)]
        url: Option<String>,
        model: String,
    },
}

/// Create an embedding provider from configuration.
///
/// Resolves the provider type (Ollama, OpenAI, OpenRouter) from the config enum.
///
/// # Arguments
/// * `config` — The embedding provider configuration.
/// * `fallback_api_key` — Optional API key to use if the config doesn't specify one.
pub fn create_embedding_provider(
    config: &EmbeddingProviderConfig,
    fallback_api_key: Option<&str>,
) -> Result<Arc<dyn EmbeddingProvider>> {
    match config {
        EmbeddingProviderConfig::OpenAI {
            api_key,
            model,
            base_url,
        } => Ok(Arc::new(OpenAIEmbedding::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
        ))),
        EmbeddingProviderConfig::OpenRouter { api_key, model } => {
            let key = api_key
                .clone()
                .or_else(|| fallback_api_key.map(String::from))
                .ok_or_else(|| {
                    FlashmemError::Config("OpenRouter embedding requires an API key".into())
                })?;
            Ok(Arc::new(OpenRouterEmbedding::new(key, model.clone())))
        }
        EmbeddingProviderConfig::Ollama { url, model } => {
            let parsed_url = url
                .as_ref()
                .map(|s| s.parse())
                .transpose()
                .map_err(|e| FlashmemError::Config(format!("Invalid URL: {}", e)))?;
            Ok(Arc::new(OllamaEmbedding::new(parsed_url, model.clone())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_provider() {
        let config = EmbeddingProviderConfig::OpenAI {
            api_key: Some("test-key".into()),
            model: "text-embedding-3-small".into(),
            base_url: None,
        };
        let provider = create_embedding_provider(&config, None).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.dimensions(), 1536);
    }

    #[test]
    fn test_create_openrouter_provider_with_fallback() {
        let config = EmbeddingProviderConfig::OpenRouter {
            api_key: None,
            model: "openai/text-embedding-3-small".into(),
        };
        let provider = create_embedding_provider(&config, Some("fallback-key")).unwrap();
        assert_eq!(provider.name(), "openrouter");
    }

    #[test]
    fn test_create_openrouter_provider_no_key_fails() {
        let config = EmbeddingProviderConfig::OpenRouter {
            api_key: None,
            model: "openai/text-embedding-3-small".into(),
        };
        let result = create_embedding_provider(&config, None);
        assert!(result.is_err());
    }
}
