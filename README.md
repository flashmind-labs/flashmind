# Flashmind

AI agent framework in Rust. Stream LLM responses, call tools, manage memory, and compose multi-agent systems — all with a clean async API.

```rust
use flashmind::{core::Agent, llm::create_provider, types::*};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = create_provider(Provider::Ollama, None)?;

    let mut agent = Agent::builder(provider).build_sync();
    let mut conversation = Conversation::new();
    conversation.set_system("You are helpful.");

    let stream = agent.start(&mut conversation, CancellationToken::new(), AgentInput::user("Hello!"), None);
    tokio::pin!(stream);
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(text) = event {
            print!("{text}");
        }
    }
    Ok(())
}
```

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
let provider = flashmind::llm::create_provider(Provider::Ollama, None)?;
let provider = flashmind::llm::create_provider(Provider::OpenRouter, Some("sk-or-..."))?;
let provider = flashmind::llm::create_provider(Provider::Anthropic, Some("sk-ant-..."))?;

// Or construct directly for more control
let provider = OllamaProvider::new(None, None)?;                    // localhost:11434
let provider = OpenRouterProvider::new(api_key);                     // default 60 RPM
let provider = OpenRouterProvider::with_rate_limit(api_key, 120);    // custom RPM
let provider = AnthropicProvider::new(api_key);                      // default 60 RPM
let provider = OpenAiProvider::builder("http://localhost:8000/")
    .api_key("sk-...")
    .name("vllm")
    .build()?;
```

### Add Tools

```rust
use flashmind::tools::ToolBuilder;

let (tools, sync) = ToolBuilder::new()
    .file_ops(None, &protected)
    .bash(secrets, &protected, forbidden, allowlist)
    .search(brave_key, firecrawl_key)
    .time()
    .http()
    .build_with_sync().await;

let mut agent = Agent::builder(provider)
    .tools(tools)
    .build_sync();
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

## Features

| What | How |
|------|-----|
| **Streaming** | All LLM calls are streaming — get tokens as they arrive |
| **Tool calling** | 70+ built-in tools, or implement the `Tool` trait for your own |
| **Memory** | Vector store with hybrid search (cosine + BM25), TTL, metadata |
| **Compaction** | Automatic context window management when conversations get long |
| **Multi-agent** | Spawn child agents, delegate tasks, communicate between them |
| **MCP** | Connect to any MCP server and use its tools |
| **OAuth integrations** | Google (Gmail, Calendar, Contacts), Outlook, GitHub, Slack |
| **Database tools** | SQLite, Postgres, MySQL, ClickHouse, Redis |
| **Infrastructure** | Docker, Kubernetes, SSH, Cloudflare |
| **Audio/Visual** | TTS, transcription, image gen, video gen |
| **Cron** | Schedule recurring agent tasks |
| **TUI** | Terminal REPL with streaming output and rich widgets |

Enable integrations via feature flags:

```toml
flashmind = { version = "0.1", features = ["mcp", "gmail", "github"] }
```

## Architecture

Eleven crates, one facade:

```
flashmind          ← facade, re-exports everything
├── flashmind-types     ��� traits (LlmProvider, Tool, MemoryProvider) + wire types
├── flashmind-core      ← Agent runtime, conversation IR, compaction, subagents
├── flashmind-llm       ← Providers: OpenRouter, Anthropic, OpenAI, Ollama
├── flashmind-tools     ← 70+ tool implementations + ToolBuilder
├── flashmind-memory    ← SQLite vector store with hybrid search
├── flashmind-prompts   ← Reusable prompt fragments
├── flashmind-cron      ← Cron scheduling
├── flashmind-skills    ← Skill discovery and execution
├── flashmind-tui       ← Terminal UI primitives
└── flashmind-tailscale ← Tailscale local API + Funnel
```

The agent runtime (`flashmind-core`) depends only on traits from `flashmind-types`, making it fully provider-agnostic.

### Agent Loop

```
Agent::start(conversation, cancel, input, max_iterations)
  └─ loop:
     ├─ stream_llm_response() → TextDelta, ToolCallStart, Usage events
     ├─ execute_tool_calls() → ToolStart, ToolResult events
     ├─ try_compact() if context pressure detected
     └─ break on Done or max iterations
```

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

## Implementing a Custom Tool

```rust
use async_trait::async_trait;
use flashmind::types::tool::{Tool, ToolContext, ToolResult};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get current weather for a city" }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let city: String = ctx.parse_args::<serde_json::Value>()?["city"]
            .as_str().unwrap_or("unknown").to_owned();
        Ok(ToolResult::success(format!("72°F and sunny in {city}")))
    }
}
```

## Implementing a Custom Provider

```rust
use async_trait::async_trait;
use flashmind::types::*;

struct MyProvider;

#[async_trait]
impl LlmProvider for MyProvider {
    fn name(&self) -> &str { "my-provider" }
    fn provider(&self) -> Provider { Provider::OpenAi }
    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        // Return a stream of StreamEvent values
        todo!()
    }
}
```

Only `complete()` is required — TTS, transcription, image gen, and model listing have sensible defaults.

## Vector Memory

```rust
use flashmind::memory::{MemoryStore, OllamaEmbedding};

let embedder = Arc::new(OllamaEmbedding::new(None, "nomic-embed-text".into()));
let store = MemoryStore::connect("memory.db", embedder).await?;

// Store with metadata
store.store("user prefers dark mode").meta("scope", "preferences").await?;

// Hybrid search (vector + keyword)
let results = store.search("preferences").filter("scope", "preferences").limit(5).await?;
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

MIT
