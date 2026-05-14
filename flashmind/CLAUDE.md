# flashmind (facade)

Re-exports all workspace crates under a single namespace. This is what users add to their `Cargo.toml`.

## Re-export Map

```rust
pub use flashmind_core as core;
pub use flashmind_cron as cron;
pub use flashmind_llm as llm;
pub use flashmind_memory as memory;
pub use flashmind_prompts as prompts;
pub use flashmind_skills as skills;
pub use flashmind_tailscale as tailscale;
pub use flashmind_tools as tools;
pub use flashmind_types as types;

#[cfg(feature = "tui")]
pub use flashmind_tui as tui;
```

`flashmind-tui` is available via the optional `tui` feature (`flashmind = { features = ["tui"] }`). It's also a dev-dependency for examples.

## Examples

All in `examples/`:

| Example | What it demonstrates |
|---------|---------------------|
| `ollama.rs` | Chat with a local Ollama model |
| `streaming.rs` | Inspect every `AgentEvent` from the agent loop |
| `custom_tool.rs` | Implement and register a custom `Tool` |
| `tui_repl.rs` | Interactive TUI REPL using `flashmind-tui` |
| `widgets.rs` | Interactive demo of all flashmind-tui widgets |
| `mcp.rs` | MCP server management with tool execution loop |
| `image_gen.rs` | Generate images via DALL-E / GPT Image |
| `image_edit.rs` | Edit images using generative models |
| `video_gen.rs` | Generate short videos from text prompts |
| `list_models.rs` | Discover available models across providers |
