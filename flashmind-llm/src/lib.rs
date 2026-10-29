//! LLM provider implementations for the Flashmind AI agent framework.
//!
//! This crate provides streaming [`LlmProvider`] implementations for five backends:
//!
//! | Provider | Struct | Features |
//! |----------|--------|----------|
//! | OpenRouter | [`OpenRouterProvider`] | Model routing, SSE streaming, reasoning, audio and media generation |
//! | Anthropic | [`AnthropicProvider`] | Direct Claude API, SSE streaming, extended thinking, multimodal |
//! | OpenAI | [`OpenAiProvider`] | OpenAI-compatible endpoints (vLLM, LiteLLM), gzip compression, routing table |
//! | Ollama | [`OllamaProvider`] | Local models, NDJSON streaming, thinking mode, num_ctx override |
//! | Mistral | [`MistralProvider`] | Direct Mistral API, SSE streaming, reasoning, batch and realtime transcription |
//!
//! All providers implement [`LlmProvider`] from `flashmind-types` and return a
//! [`CompletionStream`] of [`StreamEvent`] values. The agent runtime in
//! `flashmind-core` consumes these streams generically.
//!
//! # Creating a provider
//!
//! ```no_run
//! use std::env;
//!
//! use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
//!
//! # fn example() -> anyhow::Result<()> {
//! let provider = OllamaProvider::new(None, None)?;
//! let provider = OpenRouterProvider::new(env::var("OPENROUTER_API_KEY")?);
//! let provider = AnthropicProvider::new(env::var("ANTHROPIC_API_KEY")?);
//! let provider = OpenAiProvider::builder("https://api.openai.com/v1/")
//!     .api_key(env::var("OPENAI_API_KEY")?)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! Constructors accept credentials explicitly; they do not read environment
//! variables. [`create_provider`] is a convenience factory that returns an
//! `Arc<dyn LlmProvider>` for use with an agent.
//!
//! # Shared utilities
//!
//! - [`ContextWindowCache`]: TTL cache for model context window sizes
//! - [`sse`]: SSE stream parsing shared by HTTP-based providers
//! - [`wire_types`]: shared OpenAI-compatible request and response structures
//! - [`oss_capabilities`]: capability detection for open-source models
//! - [`http`]: retries, rate limiting, and HTTP client configuration

pub mod anthropic;
pub mod http;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod oss_capabilities;
pub mod request_builder;
pub mod sse;
pub mod wire_types;

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

pub use flashmind_types::llm::{
    AudioFormat, CompletionRequest, CompletionResponse, CompletionStream, FinishReason,
    LlmProvider, ModelCapabilities, ModelInfo, StreamEvent, TokenUsage, ToolDefinition, TtsRequest,
    Voice,
};

pub use anthropic::AnthropicProvider;
pub use mistral::MistralProvider;
pub use ollama::OllamaProvider;
pub use openai::{OpenAiProvider, OpenAiProviderBuilder, RoutingTable};
pub use openrouter::{
    JEV_LATEST_MODEL, JevQuestion, JevRequest, JevResponse, JevUsage, OpenRouterProvider,
};

use flashmind_types::model::Provider;
use std::sync::Arc;

/// Create a provider from a [`Provider`] enum variant and an optional API key.
///
/// This is the simplest way to get a provider instance when you don't need
/// custom configuration. Each provider is created with sensible defaults
/// (default rate limits, base URLs, etc.).
///
/// # Example
///
/// ```no_run
/// use std::env;
///
/// use flashmind_llm::create_provider;
/// use flashmind_types::model::Provider;
///
/// # fn example() -> anyhow::Result<()> {
/// let api_key = env::var("ANTHROPIC_API_KEY")?;
/// let provider = create_provider(Provider::Anthropic, Some(&api_key))?;
/// let provider = create_provider(Provider::Ollama, None)?;
/// # Ok(())
/// # }
/// ```
pub fn create_provider(
    provider: Provider,
    api_key: Option<&str>,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    match provider {
        Provider::Ollama => Ok(Arc::new(OllamaProvider::new(None, None)?)),
        Provider::OpenRouter => {
            let key = api_key.ok_or_else(|| anyhow::anyhow!("OpenRouter requires an API key"))?;
            Ok(Arc::new(OpenRouterProvider::new(key.to_owned())))
        }
        Provider::Anthropic => {
            let key = api_key.ok_or_else(|| anyhow::anyhow!("Anthropic requires an API key"))?;
            Ok(Arc::new(AnthropicProvider::new(key.to_owned())))
        }
        Provider::Mistral => {
            let key = api_key.ok_or_else(|| anyhow::anyhow!("Mistral requires an API key"))?;
            Ok(Arc::new(MistralProvider::new(key)))
        }
        Provider::OpenAi => {
            let provider = OpenAiProvider::new(
                None,
                api_key.map(|s| s.to_owned()),
                RoutingTable::default(),
                false,
                None,
            )?;
            Ok(Arc::new(provider))
        }
        Provider::Connect => {
            anyhow::bail!(
                "Connect provider requires a remote server configuration — use OpenAiProvider::builder() directly"
            )
        }
        Provider::Custom(name) => {
            anyhow::bail!("Custom provider '{name}' must be constructed by its implementing crate")
        }
    }
}

/// Shared TTL cache for context window sizes, keyed by model name.
///
/// Providers query this before making API calls to discover a model's context limits.
/// Entries expire after 1 hour. Thread-safe via internal `RwLock`.
///
/// # Usage
///
/// Create once and share across providers via `Arc`. Check the cache before
/// querying a provider for its context window size.
///
/// ```
/// use flashmind_llm::ContextWindowCache;
///
/// let cache = ContextWindowCache::new();
/// cache.set("example/model", 128_000);
/// assert_eq!(cache.get("example/model"), Some(128_000));
/// ```
pub struct ContextWindowCache {
    entries: RwLock<HashMap<String, (u32, Instant)>>,
    ttl: Duration,
}

impl Default for ContextWindowCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextWindowCache {
    /// Create a new cache with a 1-hour TTL.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(60 * 60),
        }
    }

    /// Look up the cached context window size for `model`.
    ///
    /// Returns `None` if the model is not in the cache or the entry has expired.
    pub fn get(&self, model: &str) -> Option<u32> {
        let entries = self.entries.read().unwrap();
        let result = entries.get(model).and_then(|(size, ts)| {
            if ts.elapsed() < self.ttl {
                Some(*size)
            } else {
                None
            }
        });

        tracing::debug!(model, hit = result.is_some(), "context window cache lookup");
        result
    }

    /// Store a context window size for `model`. Overwrites any existing entry.
    pub fn set(&self, model: &str, size: u32) {
        tracing::debug!(model, size, "caching context window size");
        let mut entries = self.entries.write().unwrap();
        entries.insert(model.to_string(), (size, Instant::now()));
    }
}

#[cfg(test)]
#[ctor::ctor]
fn init_crypto_for_tests() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::utils::{truncate_utf8, truncate_utf8_line};

    #[test]
    fn context_window_cache_miss_on_empty() {
        let cache = ContextWindowCache::new();
        assert!(cache.get("some-model").is_none());
    }

    #[test]
    fn context_window_cache_hit_after_set() {
        let cache = ContextWindowCache::new();
        cache.set("gpt-4", 128_000);
        assert_eq!(cache.get("gpt-4"), Some(128_000));
    }

    #[test]
    fn context_window_cache_different_models() {
        let cache = ContextWindowCache::new();
        cache.set("model-a", 8192);
        cache.set("model-b", 32768);
        assert_eq!(cache.get("model-a"), Some(8192));
        assert_eq!(cache.get("model-b"), Some(32768));
        assert!(cache.get("model-c").is_none());
    }

    #[test]
    fn context_window_cache_overwrite() {
        let cache = ContextWindowCache::new();
        cache.set("model", 8192);
        cache.set("model", 16384);
        assert_eq!(cache.get("model"), Some(16384));
    }

    #[test]
    fn truncate_utf8_within_bounds() {
        assert_eq!(truncate_utf8("hello", 10), "hello");
    }

    #[test]
    fn truncate_utf8_at_boundary() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
    }

    #[test]
    fn truncate_utf8_multibyte() {
        let s = "café";
        let result = truncate_utf8(s, 4);
        assert_eq!(result, "caf");
    }

    #[test]
    fn truncate_utf8_emoji() {
        let s = "hi 👋 there";
        let result = truncate_utf8(s, 4);
        assert_eq!(result, "hi ");
    }

    #[test]
    fn truncate_utf8_line_stops_at_newline() {
        assert_eq!(truncate_utf8_line("first\nsecond", 100), "first");
    }

    #[test]
    fn truncate_utf8_line_stops_at_cr() {
        assert_eq!(truncate_utf8_line("first\r\nsecond", 100), "first");
    }

    #[test]
    fn truncate_utf8_line_max_before_newline() {
        assert_eq!(truncate_utf8_line("hello world\nmore", 5), "hello");
    }

    #[test]
    fn truncate_utf8_line_no_newline() {
        assert_eq!(truncate_utf8_line("no newline", 100), "no newline");
    }
}
