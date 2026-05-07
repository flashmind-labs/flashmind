# Flashmind

AI agent framework in Rust — build, compose, and run LLM-powered agents with streaming, tool calling, long-term vector memory, conversation management, image/video generation, and audio (TTS + STT).

Flashmind is designed as a library-first framework. The core runtime (`flashmind-core`) depends only on traits from `flashmind-types`, making it provider-agnostic and enabling fast incremental builds when only provider code changes. All LLM communication is streaming by default via `CompletionStream`.

## At a Glance

```rust
let mut agent = Agent::builder(provider)
    .scope("my-app")
    .tools(tools)
    .build();

let stream = agent.start(&mut conversation, AgentInput::user("Hello!"));
tokio::pin!(stream);
while let Some(event) = stream.next().await {
    // handle TextDelta, ToolStart, Done, etc.
}
```

## Features

- **Provider-agnostic** — works with OpenRouter, Anthropic, OpenAI, Ollama, and any custom [`LlmProvider`](flashmind-types/src/llm.rs) implementation. Add new backends without touching the agent core.
- **Streaming** — full SSE-based token streaming with incremental rendering
- **Tool calling** — 30+ built-in tools (file ops, bash, grep, HTTP, web scraping, search, SQLite, MCP, audio/TTS/STT, image/video generation, etc.) plus extensible trait
- **Long-term memory** — vector store with hybrid search (cosine similarity + BM25 via FTS5), tags, TTL, and cosine-similarity deduplication
- **Conversation compaction** — automatic context window management with multi-stage escalation ladder (truncate → summarize → prune → strip → last-exchange fallback)
- **Session persistence** — SQLite-backed session storage with JSON serialization
- **Agent delegation** — parallel task delegation with progress injection
- **On-demand tools** — load tools only when needed to reduce function-calling overhead
- **Tool gating** — restrict tool execution per mode (e.g., read-only plan mode with doc-path exceptions)
- **Image & video generation** — generate images (DALL·E, GPT Image) and videos via dedicated APIs
- **Audio** — text-to-speech (TTS), speech-to-text (transcription), and voice listing
- **SQLite queries** — direct read-only SQL access to any SQLite database file

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

| Provider   | Constructor                                                                      | Requirements                                                         |
| ---------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Ollama     | `OllamaProvider::new(base_url, num_ctx)`                                         | Local Ollama instance (default: `localhost:11434`)                   |
| OpenRouter | `OpenRouterProvider::new(api_key, rate_limiter)`                                 | API key; shared `Ratelimiter` instance                               |
| Anthropic  | `AnthropicProvider::new(api_key, rate_limiter)`                                  | API key; shared `Ratelimiter` instance                               |
| OpenAI     | `OpenAiProvider::new(base_url, api_key, routing, compression, rate_limiter)`     | API key; routing table (`Arc<RwLock<HashMap>>`); compatible endpoints supported |

### Built-in Tools

| Category          | Tools                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| File & filesystem | `file_read`, `file_write`, `file_delete`, `file_list`, `read_lines`, `glob`, `grep`, `str_replace`, `str_replace_regex`, `str_diff` |
| Shell & process   | `exec` (alias: `bash_exec`), `process`                                                             |
| HTTP & web        | `http_request`, `web_fetch`, `web_scrape`, `web_crawl`, `web_map`                                  |
| Search            | `brave_search`, `firecrawl_search`, `web_search_read`                                              |
| Database          | `sqlite_query`                                                                                     |
| Memory            | `memory_store`, `memory_recall`, `memory_forget`, `memory_list`                                    |
| Audio             | `tts`, `transcribe`, `list_voices`                                                                 |
| Image & video     | `image_gen`, `image_edit`, `video_gen`, `image_read`                                               |
| Model discovery   | `list_models`                                                                                      |
| MCP               | `mcp_add`, `mcp_list`, `mcp_auth`, `mcp_remove`                                                    |
| Utility           | `get_time`, `json_query`                                                                           |

### Examples

See [`flashmind/examples/`](flashmind/examples/) for working demos:

- **`ollama.rs`** — Chat with a local Ollama model
- **`streaming.rs`** — Inspect every `AgentEvent` from the agent loop
- **`custom_tool.rs`** — Implement and register a custom `Tool`
- **`image_gen.rs`** — Generate images via DALL·E or GPT Image
- **`image_edit.rs`** — Edit images using generative models
- **`video_gen.rs`** — Generate short videos from text prompts
- **`list_models.rs`** — Discover available models across providers

## Architecture

Flashmind is a workspace of six crates:

| Crate                                  | Role                                                                        |
| -------------------------------------- | --------------------------------------------------------------------------- |
| [`flashmind-types`](flashmind-types)   | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events, model info |
| [`flashmind-core`](flashmind-core)     | `Agent`, `AgentBuilder`, `Conversation`, streaming, compaction              |
| [`flashmind-llm`](flashmind-llm)       | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama) + SSE parsing + shared HTTP/rate-limiting |
| [`flashmind-tools`](flashmind-tools)   | 30+ built-in tool implementations + composable `ToolBuilder`                |
| [`flashmind-memory`](flashmind-memory) | Vector memory (SQLite + sqlite-vec + FTS5), sessions, users, sharing       |
| [`flashmind`](flashmind)               | Facade crate that re-exports everything under one namespace                 |

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

---

## Development

### Prerequisites

- Rust 2024 edition (latest stable `rustc`)
- [Just](https://just.systems/) or `cargo` for running commands

### Building and Testing

```bash
# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace

# Run tests for a single crate
cargo test -p flashmind-core

# Generate documentation
cargo doc --workspace --no-deps --open

# Lint with Clippy
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --workspace
```

### Running Examples

```bash
# Chat with a local Ollama model
ollama pull llama3.2
cargo run -p flashmind --example ollama

# Inspect AgentEvent stream with a mock provider
cargo run -p flashmind --example streaming

# Custom tool implementation demo (no LLM backend needed)
cargo run -p flashmind --example custom_tool

# Generate an image
cargo run -p flashmind --example image_gen -- \
  --api-key YOUR_KEY \
  --prompt "A cat wearing a top hat"

# List available models
cargo run -p flashmind --example list_models
```

### Adding a New Tool

1. Create a new module in `flashmind-tools/src/` (e.g., `my_tool.rs`)
2. Implement the [`Tool`](flashmind-types/src/tool.rs) trait:

```rust
use async_trait::async_trait;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful" }
    fn parameters(&self) -> serde_json::Value { /* JSON Schema */ }
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> { /* ... */ }
    fn humanize(&self, args: &serde_json::Value) -> String { /* display summary */ }
}
```

3. Export it from `flashmind-tools/src/lib.rs`:
   ```rust
   pub mod my_tool;
   ```
4. Register it in `flashmind-tools/src/builder.rs` in the appropriate `.with_*()` method
5. Add tests in a `#[cfg(test)] mod tests` block at the bottom of the file

### Adding a New Provider

Implement [`LlmProvider`](flashmind-types/src/llm.rs) in `flashmind-llm/`:

```rust
#[async_trait]
impl LlmProvider for MyProvider {
    fn name(&self) -> &str { "my-provider" }
    fn provider(&self) -> Provider { /* ... */ }
    fn complete(&self, request: CompletionRequest) -> CompletionStream { /* stream StreamEvents */ }
}
```

The only required method is `complete()` — all others have sensible defaults. Use existing providers (especially `openai.rs` and `anthropic.rs`) as templates. Share SSE parsing via the `sse` module.

### Using Vector Memory

Flashmind includes a built-in vector memory store with hybrid search (cosine similarity + BM25 keyword matching via Reciprocal Rank Fusion):

```rust,ignore
use std::sync::Arc;
use flashmind_memory::{DbStore, OllamaEmbedding, VectorMemory};
use flashmind_types::memory::{MemoryEntry, MemoryMetadata, MemoryProvider};

// 1. Create an embedding provider
let embedder = Arc::new(OllamaEmbedding::new(None));

// 2. Open or create a database
let db = DbStore::open("memory.db", embedder.dimensions()).await?;

// 3. Wrap into a MemoryProvider
let memory = VectorMemory::new(db, embedder);

// 4. Store, search, forget
let id = memory.store("user prefers dark mode", MemoryMetadata::default()).await?;
let results = memory.search("preferences", 10).await?;
for entry in results {
    println!("{} (score: {:.2})", entry.content, entry.score);
}
memory.forget(&id).await?;
```

Embedding backends: `OllamaEmbedding`, `OpenAIEmbedding`, `OpenRouterEmbedding`.

### Code Style

- **Edition**: Rust 2024
- **Errors**: Use `anyhow::Result` for internal errors; specific error types at public boundaries
- **Logging**: Use `tracing` macros (`info!`, `debug!`, `warn!`, `error!`) — never `println!` in library code
- **Metrics**: Use the `metrics` crate for counters, gauges, and histograms
- **Rustdoc**: All public items must have documentation comments; module-level docs explain purpose and list key types
- **Section separators**: Use `// ---------------------------------------------------------------------------\n// Section Name` between logical blocks within files

## License

MIT
