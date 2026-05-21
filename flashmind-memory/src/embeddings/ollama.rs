//! Ollama embedding provider using the local /api/embed endpoint.
//!
//! <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings>

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use std::time::Duration;
use url::Url;

use crate::error::{FlashmemError, Result};
use crate::http::http_client_builder;

use super::EmbeddingProvider;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Ollama embedding provider for local models.
///
/// Uses the `/api/embed` endpoint. No API key required. Dimensions are auto-detected
/// on first call via a short probe embedding.
///
/// # Example
///
/// ```rust,ignore
/// let embedder = OllamaEmbedding::new(None, "nomic-embed-text".into());
/// let vec = embedder.embed("hello world").await?;
/// assert_eq!(vec.len(), embedder.dimensions()); // 768
/// ```
pub struct OllamaEmbedding {
    client: reqwest::Client,
    base_url: Url,
    model: String,
    dimensions: RwLock<Option<usize>>,
}

impl OllamaEmbedding {
    /// Create a new Ollama embedding provider.
    ///
    /// # Arguments
    /// * `base_url` — Ollama API base URL. Defaults to `http://localhost:11434`.
    /// * `model` — Embedding model name. Defaults to `nomic-embed-text`.
    pub fn new(base_url: Option<Url>, model: String) -> Self {
        let base_url = base_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.parse().unwrap());
        let client = http_client_builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url,
            model,
            dimensions: RwLock::new(None),
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct OllamaError {
    error: String,
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FlashmemError::Memory("No embedding returned".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        tracing::debug!(
            model = %self.model,
            base_url = %self.base_url,
            batch_size = texts.len(),
            "ollama embedding request"
        );

        let mut url = self.base_url.clone();
        url.set_path("/api/embed");

        let body = EmbedRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!("ollama embedding request failed: {}", e);
                FlashmemError::Memory(format!("Ollama request failed: {}", e))
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            if let Ok(err) = serde_json::from_str::<OllamaError>(&text) {
                tracing::warn!(status = %status, "ollama embedding error: {}", err.error);
                return Err(FlashmemError::Memory(format!(
                    "Ollama error: {}",
                    err.error
                )));
            }
            tracing::warn!(status = %status, "ollama embedding API error: {}", text);
            return Err(FlashmemError::Memory(format!(
                "Ollama API error {}: {}",
                status, text
            )));
        }

        let resp: EmbedResponse = response.json().await.map_err(|e| {
            tracing::warn!("failed to parse ollama embedding response: {}", e);
            FlashmemError::Memory(format!("Failed to parse Ollama response: {}", e))
        })?;

        // Cache dimensions from first response
        if let Some(first) = resp.embeddings.first() {
            let mut dims = self.dimensions.write().unwrap();
            if dims.is_none() {
                tracing::debug!(
                    dimensions = first.len(),
                    "cached ollama embedding dimensions"
                );
                *dims = Some(first.len());
            }
        }

        tracing::debug!(count = resp.embeddings.len(), "ollama embeddings received");

        Ok(resp.embeddings)
    }

    fn dimensions(&self) -> usize {
        match *self.dimensions.read().unwrap() {
            Some(d) => d,
            None => {
                tracing::warn!(
                    model = %self.model,
                    fallback = 384,
                    "embedding dimensions not yet detected; call embed() first to auto-detect"
                );
                384
            }
        }
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_embedding_new() {
        let provider = OllamaEmbedding::new(None, "nomic-embed-text".into());
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.base_url, DEFAULT_OLLAMA_URL.parse().unwrap());
    }

    #[test]
    fn test_ollama_embedding_custom_url() {
        let provider = OllamaEmbedding::new(
            Some("http://custom:11434".parse().unwrap()),
            "mxbai-embed-large".into(),
        );
        assert_eq!(provider.base_url, "http://custom:11434".parse().unwrap());
    }
}
