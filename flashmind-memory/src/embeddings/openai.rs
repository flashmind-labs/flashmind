//! OpenAI embedding provider using the embeddings API directly.
//!
//! <https://platform.openai.com/docs/api-reference/embeddings>

use async_trait::async_trait;

use crate::error::Result;

use super::EmbeddingProvider;
use super::http_client::{HttpEmbeddingClient, HttpEmbeddingConfig};

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

/// Embedding provider that uses the OpenAI API.
///
/// Supports `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002`.
pub struct OpenAIEmbedding {
    inner: HttpEmbeddingClient,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding provider.
    pub fn new(api_key: Option<String>, model: String, base_url: Option<String>) -> Self {
        let dimensions = model_dimensions(&model);
        let url = base_url.unwrap_or_else(|| OPENAI_EMBEDDINGS_URL.to_string());
        Self {
            inner: HttpEmbeddingClient::new(HttpEmbeddingConfig {
                url,
                api_key,
                model,
                dimensions,
                provider_name: "openai",
            }),
        }
    }
}

fn model_dimensions(model: &str) -> usize {
    match model {
        "text-embedding-3-small" => 1536,
        "text-embedding-3-large" => 3072,
        "text-embedding-ada-002" => 1536,
        _ => 1536,
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbedding {
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
        assert_eq!(model_dimensions("text-embedding-3-small"), 1536);
        assert_eq!(model_dimensions("text-embedding-3-large"), 3072);
        assert_eq!(model_dimensions("text-embedding-ada-002"), 1536);
        assert_eq!(model_dimensions("unknown-model"), 1536);
    }

    #[test]
    fn test_openai_embedding_new() {
        let provider = OpenAIEmbedding::new(
            Some("key".to_string()),
            "text-embedding-3-small".to_string(),
            None,
        );
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.dimensions(), 1536);
    }
}
