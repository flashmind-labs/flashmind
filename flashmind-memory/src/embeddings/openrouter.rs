//! OpenRouter embedding provider.
//! Uses OpenRouter's embeddings endpoint which proxies to various providers.
//!
//! <https://openrouter.ai/docs/api-reference/embeddings>

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{FlashmemError, Result};
use crate::http::{http_client_builder, send_with_retry};

use super::EmbeddingProvider;

const OPENROUTER_EMBEDDINGS_URL: &str = "https://openrouter.ai/api/v1/embeddings";

/// OpenRouter embedding provider.
pub struct OpenRouterEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimensions: usize,
}

impl OpenRouterEmbedding {
    pub fn new(api_key: String, model: String) -> Self {
        let dimensions = model_dimensions(&model);
        Self {
            client: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to build HTTP client"),
            api_key,
            model,
            dimensions,
        }
    }
}

/// Get dimensions for known embedding models available through OpenRouter.
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
        _ => 1536, // Default fallback
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenRouterEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FlashmemError::Memory("No embedding returned".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        debug!(
            model = %self.model,
            dimensions = self.dimensions,
            batch_size = texts.len(),
            "openrouter embedding request"
        );

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let response = send_with_retry(|| {
            self.client
                .post(OPENROUTER_EMBEDDINGS_URL)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
        })
        .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            warn!(status = %status, "openrouter embedding API error: {}", text);
            return Err(FlashmemError::Memory(format!(
                "OpenRouter API error {}: {}",
                status, text
            )));
        }

        let resp: EmbeddingResponse = response.json().await.map_err(|e| {
            warn!("failed to parse openrouter embedding response: {}", e);
            FlashmemError::Memory(format!("Failed to parse OpenRouter response: {}", e))
        })?;

        debug!(count = resp.data.len(), "openrouter embeddings received");

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "openrouter"
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
