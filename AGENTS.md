# AGENTS.md

Instructions for AI coding agents working in the Flashmind repository.

## Project Overview

Flashmind is an AI agent framework written in Rust. It provides streaming LLM completions, tool calling, long-term vector memory, conversation compaction, and session persistence. The codebase is a Cargo workspace of six crates plus a top-level `README.md`.

## Workspace Structure

| Crate | Role |
|-------|------|
| `flashmind-types` | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events |
| `flashmind-core` | `Agent`, `Conversation`, streaming loop, compaction, session store |
| `flashmind-llm` | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama) |
| `flashmind-memory` | Vector memory (SQLite + sqlite-vec + FTS5 hybrid search) |
| `flashmind-tools` | 30+ built-in tool implementations (file ops, bash, HTTP, web, MCP, etc.) |
| `flashmind` | Facade crate re-exporting everything under one namespace |

## Build Commands

```bash
# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint with Clippy
cargo clippy --workspace -- -D warnings

# Generate docs
cargo doc --workspace --no-deps --open

# Format
cargo fmt --workspace
```

## Code Style Rules

- **Edition**: Rust 2024
- **Errors**: Use `anyhow::Result` for internal errors; specific error types at public boundaries
- **Logging**: Use `tracing` macros (`info!`, `debug!`, `warn!`, `error!`) — never `println!` in library code
- **Metrics**: Use the `metrics` crate for counters, gauges, and histograms
- **Rustdoc**: All public items must have documentation comments
- **Module-level docs**: Each module starts with `//!` docs explaining purpose and listing key types
- **Section separators**: Use `// ---------------------------------------------------------------------------\n// Section Name` between logical blocks within files
- **No subagents**: Do not spawn sub-agents for tasks — work directly
- **Testing**: Every new public function or struct should have corresponding unit tests in `#[cfg(test)] mod tests`

## Key Design Decisions

1. **Provider-agnostic**: The agent runtime (`flashmind-core`) depends only on the `LlmProvider` trait from `flashmind-types`. Provider implementations live in `flashmind-llm`. This enables fast incremental builds when only provider code changes.

2. **Streaming-first**: All LLM communication is streaming via `CompletionStream` (`Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>>>>`). Non-streaming wrappers exist but are not the primary path.

3. **Conversation as IR**: `Conversation` in `flashmind-core` is a richer intermediate representation than raw `Message` arrays. Entry kinds (`SystemPrompt`, `Reminder`, `Memory`, `Summary`, etc.) control how entries map to wire format. Conversion happens in `to_messages()`.

4. **Compaction ladder**: Context pressure is handled via escalating strategies: truncate long outputs → LLM summarization → prune tool outputs → strip tool messages → last-exchange fallback.

5. **Tool registry with gating**: Tools can be gated per mode (e.g., read-only plan mode) with path-based exemptions. On-demand loading reduces function-calling overhead by exposing only essential tools initially.

6. **Session persistence**: SQLite-backed via `flashmind-memory`. Each session is identified by a string scope. `SessionStore` maps domain types to storage rows.

## Adding Tools

1. Create a new module in `flashmind-tools/src/` (e.g., `my_tool.rs`)
2. Implement the `Tool` trait (see `flashmind-types/src/tool.rs`)
3. Export it from `flashmind-tools/src/lib.rs`

## Adding Providers

Implement `LlmProvider` in `flashmind-llm/`. Only `complete()` is required. Use existing providers (especially `openai.rs` and `anthropic.rs`) as templates. Share SSE parsing via the `sse` module.
