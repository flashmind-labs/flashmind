//! Shared HTTP embedding client for OpenAI-compatible embedding APIs.
//!
//! Both OpenAI and OpenRouter use the same request/response format for embeddings.
//! This module encapsulates the common HTTP logic: request construction, response
//! parsing, error handling, and retry.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{FlashmemError, Result};
use crate::http::{http_client_builder, send_with_retry};

/// Configuration for constructing an [`HttpEmbeddingClient`].
pub struct HttpEmbeddingConfig {
    /// The embeddings endpoint URL.
    pub url: String,
    /// Optional Bearer token for authentication.
    pub api_key: Option<String>,
    /// Embedding model identifier sent in the request body.
    pub model: String,
    /// Known vector dimensionality for this model.
    pub dimensions: usize,
    /// Provider name used in log messages and [`EmbeddingProvider::name()`].
    pub provider_name: &'static str,
}

/// Reusable HTTP client for OpenAI-compatible embedding endpoints.
///
/// Handles request construction, authentication, response parsing, and error
/// reporting. Both [`super::OpenAIEmbedding`] and [`super::OpenRouterEmbedding`]
/// delegate to this client.
pub struct HttpEmbeddingClient {
    client: reqwest::Client,
    api_key: Option<String>,
    url: String,
    model: String,
    dimensions: usize,
    provider_name: &'static str,
}

impl HttpEmbeddingClient {
    /// Build a new client from the given configuration.
    pub fn new(config: HttpEmbeddingConfig) -> Self {
        Self {
            client: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: config.api_key,
            url: config.url,
            model: config.model,
            dimensions: config.dimensions,
            provider_name: config.provider_name,
        }
    }

    /// The provider name (e.g. `"openai"`, `"openrouter"`).
    pub fn provider_name(&self) -> &str {
        self.provider_name
    }

    /// The vector dimensionality for the configured model.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Embed a single text string.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FlashmemError::Memory("No embedding returned".into()))
    }

    /// Embed a batch of texts in a single API call.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        tracing::debug!(
            model = %self.model,
            dimensions = self.dimensions,
            batch_size = texts.len(),
            "{} embedding request", self.provider_name
        );

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let api_key = self.api_key.clone();
        let url = self.url.clone();
        let response = send_with_retry(|| {
            let mut req = self.client.post(&url).json(&body);
            if let Some(ref key) = api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            req
        })
        .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            tracing::warn!(
                status = %status,
                "{} embedding API error: {}", self.provider_name, text
            );
            return Err(FlashmemError::Memory(format!(
                "{} API error {}: {}",
                self.provider_name, status, text
            )));
        }

        let resp: EmbeddingResponse = response.json().await.map_err(|e| {
            tracing::warn!(
                "failed to parse {} embedding response: {}",
                self.provider_name,
                e
            );
            FlashmemError::Memory(format!(
                "Failed to parse {} response: {}",
                self.provider_name, e
            ))
        })?;

        tracing::debug!(
            count = resp.data.len(),
            "{} embeddings received",
            self.provider_name
        );

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_client_construction() {
        let client = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            url: "https://example.com/v1/embeddings".into(),
            api_key: Some("test-key".into()),
            model: "test-model".into(),
            dimensions: 768,
            provider_name: "test",
        });
        assert_eq!(client.provider_name(), "test");
        assert_eq!(client.dimensions(), 768);
    }

    #[test]
    fn test_http_client_no_api_key() {
        let client = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            url: "http://localhost:8080/embed".into(),
            api_key: None,
            model: "local-model".into(),
            dimensions: 384,
            provider_name: "local",
        });
        assert_eq!(client.dimensions(), 384);
    }
}
