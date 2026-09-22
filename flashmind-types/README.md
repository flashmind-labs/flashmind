# flashmind-types

Shared traits and types for the [Flashmind](https://github.com/flashmind-labs/flashmind) AI agent framework.

This is the foundation crate  -  all other flashmind crates depend on it.

## Core Traits

- **`LlmProvider`**  -  streaming completions interface (only `complete()` is required)
- **`Tool`**  -  tool definition + execution for agents
- **`MemoryProvider`**  -  vector memory store/search/forget

## Key Types

- `AgentEvent`  -  streaming events: `TextDelta`, `ToolStart`, `ToolResult`, `Done`, etc.
- `Message` / `ContentPart`  -  wire format for LLM messages (text, images, audio, video)
- `Model` / `Provider`  -  model identification (`openrouter:anthropic/claude-sonnet-4`)
- `CompletionRequest` / `StreamEvent`  -  LLM request and response types
- `ToolRegistry`  -  O(1) tool lookup with alias support

## Usage

```rust
use flashmind_types::{LlmProvider, CompletionRequest, StreamEvent};
```

Most users should depend on the [`flashmind`](https://crates.io/crates/flashmind) facade crate instead of this one directly.

## License

MPL-2.0
