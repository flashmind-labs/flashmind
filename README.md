# Flashmind

[![Crates.io](https://img.shields.io/crates/v/flashmind.svg)](https://crates.io/crates/flashmind)
[![Documentation](https://docs.rs/flashmind/badge.svg)](https://docs.rs/flashmind)
[![License: MPL-2.0](https://img.shields.io/crates/l/flashmind.svg)](LICENSE)

Build AI-powered apps in Rust. Flashmind gives you everything you need to go from idea to working agent in minutes  -  streaming LLM providers, 70+ tools, vector memory, multi-agent orchestration, and MCP support, all behind a clean async API.

```rust
use flashmind::core::{Agent, CancellationToken, Conversation};
use flashmind::llm::create_provider;
use flashmind::types::{AgentEvent, AgentInput, Provider};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = create_provider(Provider::Ollama, None)?;

    let mut agent = Agent::builder(provider).build().await;
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

- **Ship fast**  -  a working agent is ~15 lines of code. Add tools, memory, or child agents as you need them
- **4 providers**  -  Ollama, OpenRouter, Anthropic, OpenAI (+ vLLM/LiteLLM compatible)
- **70+ built-in tools**  -  file ops, shell, web search, databases, Docker, K8s, SSH, and more
- **Vector memory**  -  hybrid search (cosine similarity + BM25) backed by SQLite
- **Multi-agent**  -  spawn child agents, delegate tasks, bidirectional communication
- **MCP support**  -  connect to any Model Context Protocol server
- **OAuth integrations**  -  Gmail, Google Calendar, Outlook, GitHub, Slack, Cloudflare
- **Automatic compaction**  -  5-stage context window management when conversations get long
- **Fully streaming**  -  every LLM call streams tokens with cancellation support

## Getting Started

```toml
[dependencies]
flashmind = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
futures = "0.3"
```

### Jev CLI

Install the `jev` command:

```sh
cargo install flashmind-jev-cli
```

Put `openrouter_key = "YOUR_OPENROUTER_KEY"` in `~/.jev-cli`, then pass a JSON
request on stdin or as a file. See [the Jev CLI guide](src/bin/jev-cli/README.md) for
an example and [its agent skill](src/bin/jev-cli/SKILL.md) for the request format.

### flsh CLI

See the [flsh guide](flashmind-cli/GUIDE.md) for setup and command examples.

Start the terminal UI or send a single prompt:

```sh
flsh
flsh -p "Explain this project"
flsh text -m 'openrouter:<model>' "Write a short poem"
```

For Moonshot or Z.ai, set `MOONSHOT_API_KEY` or `ZAI_API_KEY` and select a
model with `-m moonshot:<model>` or `-m zai:<model>`. Z.ai image generation is
available with `flsh image gen -m zai:glm-image`.

Replace `<model>` and other placeholders in the examples with values from your
provider. Run `flsh models` to browse and search available models.

### Shell completion

Run `flsh completions install` to enable completion in your login shell. Use
`--shell zsh`, `--shell bash`, or `--shell fish` to select a different shell.
Start a new shell or source its startup file to activate it. To generate a
script without installing, run `flsh completions <shell>`.

### Model generation

`flsh` can generate text, images, video and speech, and transcribe audio:

```sh
flsh models openrouter
flsh image gen -m 'openrouter:<model>' --aspect-ratio 3:4 --size 1K -o image.png "A fox"
flsh video -m 'openrouter:<model>' -o video.mp4 "Waves at dusk"
flsh speech -m 'openrouter:<model>' --voice '<voice>' -o speech.mp3 "Hello"
flsh transcribe -m 'openrouter:<model>' audio.mp3
flsh ocr document.pdf
```

Run `flsh <command> --help` for options. Model browsing supports price filters
such as `in<=1 out<=5`, in dollars per million tokens. OpenRouter commands use
the `flsh` provider configuration or `OPENROUTER_API_KEY`.
Use `flsh models mistral` to browse Mistral models. Press Tab to move between
columns. Enter sorts by name, context, release date or price; press it again
to reverse the order. F2 shows model details and provider-reported image options.
Enter maximum prices in dollars
per million tokens; use Left/Right and Space to select features.
You can also type filters such as `cap:vision cap:tools` in the search field.
Other capabilities include `chat`, `reasoning`, `documents`, `audio-input`,
`video-input`, `image`, `video`, `speech`, and `transcribe`.
OCR uses Mistral and requires `MISTRAL_API_KEY`; pass `--json` for the full
response or `-e 'table_format="html"'` for provider-specific options.

List Mistral voices and generate speech:

```sh
flsh voices mistral
flsh speech -m mistral:voxtral-mini-tts-3-0 --voice voice_478 -o speech.mp3 "Hello"
```

Use `-e` to add provider-specific JSON fields to generation requests:

```sh
flsh image -m 'venice:<model>' -e 'lora_strength=50' -e 'negative_prompt="clouds,rain"' "A landscape"
```

Each value must be JSON. Repeat `-e` for additional fields. Use `--reference`
for an image reference and `flsh video --video-reference` for a reference video.
Reference support depends on the provider and model.

Edit an existing image with a Venice edit model. Repeat `--image` to combine
multiple inputs, with the base image first:

```sh
flsh image edit -m venice:qwen-image-2-pro-edit --image reference.webp -o edited.png "Replace the sky with a sunset"
```

Use `--format png`, `--format jpeg`, or `--format webp` when the provider
supports it. Existing output files are protected; pass `--overwrite` to
replace them. OpenAI image edits also accept `--mask mask.png`: transparent
parts mark the area to edit, and the mask must match the source dimensions.

Image editing also supports OpenRouter, OpenAI, Google Gemini and fal:

```sh
flsh image edit -m openrouter:openai/gpt-image-2 --image reference.webp -o edited.png "Replace the sky with a sunset"
flsh image edit -m openai:gpt-image-2 --image reference.webp -o edited.png "Replace the sky with a sunset"
flsh image edit -m google:gemini-nano-banana-2.1 --image reference.webp -o edited.png "Replace the sky with a sunset"
flsh image edit -m fal:fal-ai/flux-2/edit --image reference.webp -o edited.png "Replace the sky with a sunset"
```

Google and fal use `GEMINI_API_KEY` and `FAL_KEY`, respectively. For fal FLUX.2,
use `-e 'image_size="landscape_4_3"'` to select an output size.

For ElevenLabs, set `ELEVENLABS_API_KEY` or `[providers.elevenlabs].api_key`,
then list voices and generate speech:

```sh
flsh models elevenlabs
flsh voices elevenlabs
flsh speech -m 'elevenlabs:<model>' --voice '<voice-id>' -o speech.mp3 "Hello"
```

Generate GLB models from text or an image with Meshy or Tripo:

```sh
flsh 3d -m meshy:latest -o model.glb "A wooden chair"
flsh 3d -m tripo:v3.1-20260211 --image reference.png --texture -o model.glb
```

Set `MESHY_API_KEY` or `TRIPO_API_KEY`, or configure the corresponding provider
key in `config.toml`. Repeat `--image` for up to four views; Tripo expects front,
left, back, then right. Without `--texture`, the output is untextured. For Meshy
text generation, `--texture` creates a separate refine task. Use `-e` for
provider-specific fields in the initial generation request.

Seedance models are available through OpenRouter. For example:

```sh
flsh video -m openrouter:bytedance/seedance-2.5 --duration 5 --resolution 720p -o video.mp4 "Waves at dusk"
```

## Credentials

Keep provider keys and OAuth tokens out of the repository. Load them from
environment variables or a secret manager, and keep local values in an ignored
`.env` file. Never put a real key in source, examples, tests, or committed
configuration. If a credential is committed or published, revoke it at the
provider and replace it.

### Pick a Provider

```rust
use flashmind::llm::*;
use flashmind::types::Provider;

// Quickest  -  auto-detect provider from enum
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

### Session Routing

Reuse the same ID across turns and resume. Use a new ID for a new or forked
conversation. Existing UUIDs can be passed directly to `with_session_id`.

```rust
use flashmind::types::{AgentLlmConfig, CompletionRequest, Message, SessionId};

let model = "openrouter:example/model".parse()?;
let llm = AgentLlmConfig::new(model).with_session_id(SessionId::new());
let request = CompletionRequest::from_config(&llm, vec![Message::user("Hello")]);
```

Pass `llm` to the agent builder's `.llm(llm)`. OpenRouter uses the ID for
[sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching#provider-sticky-routing);
an explicit provider `order` overrides it.

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
    .build()
    .await;
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

Only `complete()` is required  -  TTS, transcription, image gen, and model listing have sensible defaults.

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

# Embedding arithmetic (requires `ollama pull nomic-embed-text`)
cargo run -p flashmind --example embedding_calculator --features memory -- '(king + man) - (queen + woman)'

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
