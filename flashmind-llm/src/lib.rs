//! LLM provider implementations for the Flashmind AI agent framework.
//!
//! This crate provides streaming [`LlmProvider`] implementations for four backends:
//!
//! | Provider | Struct | Features |
//! |----------|--------|----------|
//! | OpenRouter | [`OpenRouterProvider`] | 100+ models, SSE streaming, reasoning, TTS, auto capability detection |
//! | Anthropic | [`AnthropicProvider`] | Direct Claude API, SSE streaming, extended thinking, multimodal |
//! | OpenAI | [`OpenAiProvider`] | OpenAI-compatible endpoints (vLLM, LiteLLM), gzip compression, routing table |
//! | Ollama | [`OllamaProvider`] | Local models, NDJSON streaming, thinking mode, num_ctx override |
//!
//! All providers implement [`LlmProvider`] from `flashmind-types` and return a
//! [`CompletionStream`] of [`StreamEvent`] values. The agent runtime in
//! `flashmind-core` consumes these streams generically.
//!
//! # Creating a provider
//!
//! ```rust,ignore
//! // OpenRouter — requires OPENROUTER_API_KEY env var
//! let provider = OpenRouterProvider::new(api_key);
//!
//! // Ollama — connects to localhost:11434 by default
//! let provider = OllamaProvider::new(None, None);
//!
//! // Anthropic — requires ANTHROPIC_API_KEY env var
//! let provider = AnthropicProvider::new(api_key, rate_limiter);
//!
//! // OpenAI / compatible — requires OPENAI_API_KEY env var
//! let provider = OpenAiProvider::new(base_url, api_key, routing, compression, rate_limiter);
//! ```
//!
//! # Shared utilities
//!
//! - [`ContextWindowCache`] — shared TTL cache for model context window sizes
//! - [`sse`] — SSE stream parsing shared by HTTP-based providers (OpenRouter, OpenAI)
//! - [`wire_types`] — shared request/response structures for the OpenAI-compatible protocol
//! - [`oss_capabilities`] — capability detection for open-source models (e.g., Qwen thinking mode)
//! - [`http`] — retry logic, rate limiting, and shared HTTP client builder

pub mod anthropic;
pub mod http;
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
pub use ollama::OllamaProvider;
pub use openai::{OpenAiProvider, RoutingTable};
pub use openrouter::OpenRouterProvider;

/// Shared TTL cache for context window sizes, keyed by model name.
///
/// Providers query this before making API calls to discover a model's context limits.
/// Entries expire after 1 hour. Thread-safe via internal `RwLock`.
///
/// # Usage
///
/// Create once and share across providers via `Arc`. Before each completion
/// request, check the cache first — only hit the provider API on a miss.
///
/// ```rust,ignore
/// let cache = Arc::new(ContextWindowCache::new());
///
/// if let Some(ctx) = cache.get(&model_name) {
///     // Use cached value
/// } else {
///     let ctx = provider.context_window(&model).await.unwrap_or(DEFAULT);
///     cache.set(&model_name, ctx);
/// }
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

/// Truncate a string to at most `max` bytes, landing on a valid UTF-8 char boundary.
fn truncate_utf8(s: &str, max: usize) -> &str {
    &s[..s.floor_char_boundary(max)]
}

/// Like [`truncate_utf8`], but also breaks at the first newline if it comes before `max`.
/// Useful for log previews where multi-line output should show only the first line.
fn truncate_utf8_line(s: &str, max: usize) -> &str {
    let max = s
        .bytes()
        .position(|b| b == b'\r' || b == b'\n')
        .map_or(max, |nl| nl.min(max));

    truncate_utf8(s, max)
}

#[cfg(test)]
#[ctor::ctor]
fn init_crypto_for_tests() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(test)]
mod tests {
    use super::*;

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
