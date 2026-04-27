//! OpenAI embedding provider using the embeddings API directly.
//!
//! <https://platform.openai.com/docs/api-reference/embeddings>

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{FlashmemError, Result};
use crate::http::{http_client_builder, send_with_retry};

use super::EmbeddingProvider;

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

/// OpenAI embedding provider.
pub struct OpenAIEmbedding {
    client: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    model: String,
    dimensions: usize,
}

impl OpenAIEmbedding {
    pub fn new(api_key: Option<String>, model: String, base_url: Option<String>) -> Self {
        let dimensions = model_dimensions(&model);
        let url = base_url.unwrap_or_else(|| OPENAI_EMBEDDINGS_URL.to_string());
        Self {
            client: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to build HTTP client"),
            api_key,
            base_url: url,
            model,
            dimensions,
        }
    }
}

/// Get dimensions for known OpenAI embedding models.
fn model_dimensions(model: &str) -> usize {
    match model {
        "text-embedding-3-small" => 1536,
        "text-embedding-3-large" => 3072,
        "text-embedding-ada-002" => 1536,
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
impl EmbeddingProvider for OpenAIEmbedding {
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
            "openai embedding request"
        );

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let mut req = self.client.post(&self.base_url).json(&body);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = send_with_retry(|| req.try_clone().expect("clone request")).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            warn!(status = %status, "openai embedding API error: {}", text);
            return Err(FlashmemError::Memory(format!(
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        let resp: EmbeddingResponse = response.json().await.map_err(|e| {
            warn!("failed to parse openai embedding response: {}", e);
            FlashmemError::Memory(format!("Failed to parse OpenAI response: {}", e))
        })?;

        debug!(count = resp.data.len(), "openai embeddings received");

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "openai"
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
