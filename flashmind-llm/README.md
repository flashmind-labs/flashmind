# flashmind-llm

LLM provider implementations for [Flashmind](https://github.com/flashmind-labs/flashmind).

Streaming completions via SSE/NDJSON, rate limiting, capability detection, and context window caching.

## Providers

| Provider | Constructor | Features |
|----------|-------------|----------|
| OpenRouter | `OpenRouterProvider::new(api_key)` | 100+ models, auto capability detection, TTS |
| Anthropic | `AnthropicProvider::new(api_key)` | Claude models, extended thinking, multimodal |
| OpenAI | `OpenAiProvider::builder(url).build()` | OpenAI + vLLM/LiteLLM compatible |
| Ollama | `OllamaProvider::new(None, None)?` | Local models, no API key needed |

## Quick Start

```rust
use flashmind_llm::{create_provider, OpenRouterProvider, AnthropicProvider};
use flashmind_types::model::Provider;

// From enum (simplest)
let provider = create_provider(Provider::Anthropic, Some("sk-ant-..."))?;

// Direct construction (more control)
let provider = OpenRouterProvider::with_rate_limit(api_key, 120);
let provider = AnthropicProvider::new(api_key);
```

## License

MIT
