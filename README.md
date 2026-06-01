# Flashmind

[![Crates.io](https://img.shields.io/crates/v/flashmind.svg)](https://crates.io/crates/flashmind)
[![Documentation](https://docs.rs/flashmind/badge.svg)](https://docs.rs/flashmind)
[![License: MPL-2.0](https://img.shields.io/crates/l/flashmind.svg)](LICENSE)

Build AI-powered apps in Rust. Flashmind gives you everything you need to go from idea to working agent in minutes — streaming LLM providers, 70+ tools, vector memory, multi-agent orchestration, and MCP support, all behind a clean async API.

```rust
use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::llm::create_provider;
use flashmind::types::{AgentEvent, AgentInput, Provider};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = create_provider(Provider::Ollama, None)?;

    let mut agent = Agent::builder(provider).build_sync();
    let mut conversation = Conversation::new();
    conversation.set_system("You are helpful.");

    let cancel = CancellationToken::new();
    let stream = agent.start(
        &mut conversation,
        cancel,
        AgentInput::user("Hello!"),
        None,
    );
    tokio::pin!(stream);
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(text) = event {
            print!("{text}");
        }
    }
    Ok(())
}
```

That's a complete, working AI chat app. Swap `Ollama` for `Anthropic` or `OpenRouter` and you're talking to Claude or any of 300+ models.

## Why Flashmind

- **Ship fast** — a working agent is ~15 lines of code. Add tools, memory, or child agents as you need them
- **4 providers** — Ollama, OpenRouter, Anthropic, OpenAI (+ vLLM/LiteLLM compatible)
- **70+ built-in tools** — file ops, shell, web search, databases, Docker, K8s, SSH, and more
- **Vector memory** — hybrid search (cosine similarity + BM25) backed by SQLite
- **Multi-agent** — spawn child agents, delegate tasks, bidirectional communication
- **MCP support** — connect to any Model Context Protocol server
- **OAuth integrations** — Gmail, Google Calendar, Outlook, GitHub, Slack, Cloudflare
- **Automatic compaction** — 5-stage context window management when conversations get long
- **Fully streaming** — every LLM call streams tokens with cancellation support

## Getting Started

```toml
[dependencies]
flashmind = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
futures = "0.3"
```

### Pick a Provider

```rust
use flashmind::llm::*;
use flashmind::types::Provider;

// Quickest — auto-detect provider from enum
let provider = create_provider(Provider::Ollama, None)?;
let provider = create_provider(Provider::OpenRouter, Some("sk-or-..."))?;
let provider = create_provider(Provider::Anthropic, Some("sk-ant-..."))?;

// Or construct directly for more control
let provider = OllamaProvider::new(None, None)?;                       // localhost:11434
let provider = OpenRouterProvider::new("sk-or-...".into());             // default 60 RPM
let provider = OpenRouterProvider::with_rate_limit("sk-or-...".into(), 120);
let provider = AnthropicProvider::new("sk-ant-...".into());
let provider = OpenAiProvider::builder("http://localhost:8000/")
    .api_key("sk-...")
    .name("vllm")
    .build()?;
```

### Stream Events

The agent emits events as it thinks and acts:

```rust
while let Some(event) = stream.next().await {
    match event {
        AgentEvent::TextDelta(text) => print!("{text}"),
        AgentEvent::ToolStart { name, .. } => println!("\n> Running {name}..."),
        AgentEvent::ToolResult { output, .. } => println!("  {output}"),
        AgentEvent::Done(response) => println!("\n{response}"),
        _ => {}
    }
}
```

### Add a Custom Tool

```rust
use async_trait::async_trait;
use flashmind::types::tool::{Tool, ToolContext, ToolResult};
use serde_json::json;

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get current weather for a city" }
    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let city = ctx.args["city"].as_str().unwrap_or("unknown");
        Ok(ToolResult::success(&ctx.tool_call_id, format!("72°F and sunny in {city}")))
    }
}
```

Register it and the LLM can call it automatically:

```rust
use std::sync::Arc;
use flashmind::types::ToolRegistry;

let mut tools = ToolRegistry::new();
tools.register(Arc::new(WeatherTool));

let mut agent = Agent::builder(provider)
    .tools(tools)
    .build_sync();
```

### Implement a Custom Provider

```rust
use async_trait::async_trait;
use flashmind::types::{CompletionRequest, CompletionStream, LlmProvider, Provider};

struct MyProvider;

#[async_trait]
impl LlmProvider for MyProvider {
    fn name(&self) -> &str { "my-provider" }
    fn provider(&self) -> Provider { Provider::OpenAi }
    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        todo!()
    }
}
```

Only `complete()` is required — TTS, transcription, image gen, and model listing have sensible defaults.

### Vector Memory

```rust
use std::path::Path;
use std::sync::Arc;
use flashmind::memory::{MemoryStore, OllamaEmbedding};

let embedder = Arc::new(OllamaEmbedding::new(None, "nomic-embed-text".into()));
let store = MemoryStore::connect(Path::new("memory.db"), embedder).await?;

// Store with metadata
store.store("user prefers dark mode").meta("scope", "preferences").await?;

// Hybrid search (vector + keyword)
let results = store.search("preferences").filter("scope", "preferences").limit(5).await?;
```

### Feature Flags

Enable integrations via feature flags:

```toml
flashmind = { version = "0.1", features = ["mcp", "gmail", "github"] }
```

| Feature | What it enables |
|---------|----------------|
| `mcp` | Model Context Protocol server management |
| `gmail`, `google-calendar`, `google-contacts` | Google OAuth integrations |
| `outlook` | Outlook Mail/Calendar/Contacts |
| `github` | GitHub repos, issues, PRs |
| `slack` | Slack channels, messages, search |
| `caldav` | CalDAV (Fastmail, Nextcloud, iCloud) |
| `composio` | 250+ app integrations via Composio.dev |
| `tui` | Terminal UI primitives |
| `session` | Conversation session persistence |
| `full` | Everything |

## Architecture

Eleven crates, one facade:

```
flashmind              re-exports everything
├── flashmind-types        traits (LlmProvider, Tool, MemoryProvider) + wire types
├── flashmind-core         Agent runtime, conversation IR, compaction, subagents
├── flashmind-llm          Providers: OpenRouter, Anthropic, OpenAI, Ollama
├── flashmind-tools        70+ tool implementations + ToolBuilder
├── flashmind-memory       SQLite vector store with hybrid search
├── flashmind-prompts      Reusable prompt fragments
├── flashmind-cron         Cron scheduling
├── flashmind-skills       Skill discovery and execution
├── flashmind-tui          Terminal UI primitives
└── flashmind-tailscale    Tailscale local API + Funnel
```

The agent runtime (`flashmind-core`) depends only on traits from `flashmind-types`, making it fully provider-agnostic. Use the `flashmind` facade for convenience, or depend on individual crates for finer control.

## Examples

```bash
# Chat with local Ollama
cargo run -p flashmind --example ollama

# Inspect every AgentEvent
cargo run -p flashmind --example streaming

# Custom tool implementation
cargo run -p flashmind --example custom_tool

# Multi-agent debate
cargo run -p flashmind --example subagents

# Terminal REPL
MODEL=qwen3:8b cargo run -p flashmind --example tui_repl

# Image generation
cargo run -p flashmind --example image_gen

# MCP server integration
cargo run -p flashmind --example mcp --features mcp
```

## Development

```bash
cargo build --workspace          # build all
cargo test --workspace           # test all
cargo clippy --workspace -- -D warnings
cargo fmt --all
cargo doc --workspace --no-deps --open
```

## License

MPL-2.0
