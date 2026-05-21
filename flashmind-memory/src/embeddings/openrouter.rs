//! OpenRouter embedding provider.
//! Uses OpenRouter's embeddings endpoint which proxies to various providers.
//!
//! <https://openrouter.ai/docs/api-reference/embeddings>

use async_trait::async_trait;

use crate::error::Result;

use super::EmbeddingProvider;
use super::http_client::{HttpEmbeddingClient, HttpEmbeddingConfig};

const OPENROUTER_EMBEDDINGS_URL: &str = "https://openrouter.ai/api/v1/embeddings";

/// Embedding provider that uses the OpenRouter API.
///
/// Routes to various embedding models through OpenRouter's unified endpoint.
pub struct OpenRouterEmbedding {
    inner: HttpEmbeddingClient,
}

impl OpenRouterEmbedding {
    /// Create a new OpenRouter embedding provider.
    pub fn new(api_key: String, model: String) -> Self {
        let dimensions = model_dimensions(&model);
        Self {
            inner: HttpEmbeddingClient::new(HttpEmbeddingConfig {
                url: OPENROUTER_EMBEDDINGS_URL.to_string(),
                api_key: Some(api_key),
                model,
                dimensions,
                provider_name: "openrouter",
            }),
        }
    }
}

fn model_dimensions(model: &str) -> usize {
    match model {
        "openai/text-embedding-3-small" => 1536,
        "openai/text-embedding-3-large" => 3072,
        "openai/text-embedding-ada-002" => 1536,
        "cohere/embed-english-v3.0" => 1024,
        "cohere/embed-multilingual-v3.0" => 1024,
        "cohere/embed-english-light-v3.0" => 384,
        "voyage/voyage-large-2" => 1536,
        "voyage/voyage-2" => 1024,
        _ => 1536,
    }
}

#[async_trait]
impl EmbeddingProvider for OpenRouterEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed(text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch(texts).await
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn name(&self) -> &str {
        self.inner.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(model_dimensions("openai/text-embedding-3-small"), 1536);
        assert_eq!(model_dimensions("cohere/embed-english-v3.0"), 1024);
        assert_eq!(model_dimensions("cohere/embed-english-light-v3.0"), 384);
    }

    #[test]
    fn test_openrouter_embedding_new() {
        let provider =
            OpenRouterEmbedding::new("key".into(), "openai/text-embedding-3-small".into());
        assert_eq!(provider.name(), "openrouter");
        assert_eq!(provider.dimensions(), 1536);
    }
}
