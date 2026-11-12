# flashmind-llm

LLM providers, each implementing `LlmProvider` from `flashmind-types`.

- Provider-specific request and response handling lives here, nowhere else.
- Reuse the shared SSE parser and HTTP client setup instead of writing new
  ones.
