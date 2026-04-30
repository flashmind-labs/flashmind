# Flashmind

AI agent framework in Rust — build, compose, and run LLM-powered agents with streaming, tool calling, memory, and conversation management.

## Features

- **Provider-agnostic** — works with OpenRouter, Anthropic, OpenAI, Ollama, and any custom [`LlmProvider`](flashmind-types/src/llm.rs) implementation
- **Streaming** — full SSE-based token streaming with incremental rendering
- **Tool calling** — 30+ built-in tools (file ops, bash, grep, HTTP, web scraping, MCP, audio/TTS, etc.) plus extensible trait
- **Long-term memory** — vector store with hybrid search (cosine similarity + BM25), tags, TTL, and cosine-similarity deduplication
- **Conversation compaction** — automatic context window management with multi-stage escalation ladder (truncate → summarize → prune → strip → last-exchange fallback)
- **Session persistence** — SQLite-backed session storage with JSON serialization
- **Subagents** — parallel task delegation with progress injection
- **On-demand tools** — load tools only when needed to reduce function-calling overhead
- **Tool gating** — restrict tool execution per mode (e.g., read-only plan mode with doc-path exceptions)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
flashmind = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
futures = "0.3"
```

For building from source:

```bash
cargo build --workspace
```

## Quick Start

```rust
use std::sync::Arc;
use flashmind::core::{Agent, Conversation, ConversationEntry};
use flashmind::types::{AgentEvent, AgentInput, LlmProvider};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    // Create a provider (e.g. Ollama, OpenRouter, Anthropic)
    let provider: Arc<dyn LlmProvider> = Arc::new(flashmind::llm::OllamaProvider::new(None, None));

    // Build the agent
    let mut agent = Agent::builder(provider)
        .scope("my-app")
        .max_iterations(50)
        .build();

    // Set up conversation with system prompt
    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system("You are helpful."));

    // Stream events as the agent processes the turn
    let stream = agent.start(&mut conversation, AgentInput::user("Hello!"));
    tokio::pin!(stream);
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Done(response) => println!("\n{response}"),
            _ => {}
        }
    }
}
```

### Available Providers

| Provider | Constructor | Requirements |
|----------|-------------|-------------|
| Ollama | `OllamaProvider::new(url, num_ctx)` | Local Ollama instance (default: `localhost:11434`) |
| OpenRouter | `OpenRouterProvider::new(api_key)` | `OPENROUTER_API_KEY` env var |
| Anthropic | `AnthropicProvider::new(api_key)` | `ANTHROPIC_API_KEY` env var |
| OpenAI | `OpenAiProvider::new(api_key, base_url)` | `OPENAI_API_KEY` env var; custom `base_url` for compatible endpoints |

### Examples

See [`flashmind/examples/`](flashmind/examples/) for working demos:
- **`ollama.rs`** — Chat with a local Ollama model
- **`streaming.rs`** — Inspect every `AgentEvent` from the agent loop
- **`custom_tool.rs`** — Implement and register a custom `Tool`

## Architecture

Flashmind is a workspace of six crates:

| Crate | Role |
|-------|------|
| [`flashmind-types`](flashmind-types) | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events |
| [`flashmind-core`](flashmind-core) | `Agent`, `AgentBuilder`, `Conversation`, streaming, compaction |
| [`flashmind-llm`](flashmind-llm) | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama) |
| [`flashmind-tools`](flashmind-tools) | 30+ built-in tool implementations |
| [`flashmind-memory`](flashmind-memory) | Vector memory (SQLite + sqlite-vec + FTS5) |
| [`flashmind`](flashmind) | Facade crate that re-exports everything under one namespace |

### High-level flow

```
┌─────────────────────────────────────────────┐
│                   Agent                      │
│                                              │
│  ┌──────────┐    ┌──────────────┐           │
│  │Conversation│──►│stream_llm_response│      │
│  └──────────┘    └───────┬────────┘          │
│                          │                    │
│              ┌───────────▼──────────┐        │
│              │  execute_tool_calls   │        │
│              └───────────┬──────────┘        │
│                          │                    │
│              ┌───────────▼──────────┐        │
│              │    try_compact       │         │
│              └──────────────────────┘        │
└─────────────────────────────────────────────┘
```

## License

MIT
