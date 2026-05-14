# flashmind-llm

LLM provider implementations. Each provider implements the `LlmProvider` trait from `flashmind-types`.

## Providers

| Provider | File | Constructor |
|----------|------|-------------|
| Ollama | `ollama.rs` | `OllamaProvider::new(base_url, num_ctx)` |
| OpenRouter | `openrouter.rs` | `OpenRouterProvider::new(api_key)` or `::with_rate_limit(api_key, rpm)` |
| Anthropic | `anthropic.rs` | `AnthropicProvider::new(api_key, rate_limiter)` |
| OpenAI | `openai.rs` | `OpenAiProvider::new(base_url, api_key, routing, compression, rate_limiter)` |

## Shared Infrastructure

- `sse.rs` — SSE stream parser shared across providers
- `http.rs` — shared HTTP client setup (rustls, compression)
- `wire_types.rs` — provider-specific JSON schemas
- `oss_capabilities.rs` — capability detection for open-source models

## Adding a Provider

1. Create `new_provider.rs` implementing `LlmProvider`
2. Only `complete()` is required — it returns a `CompletionStream`
3. Use `sse.rs` for SSE parsing, `http.rs` for client setup
4. Export from `lib.rs`
