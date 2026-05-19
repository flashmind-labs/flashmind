# Flashmind

AI agent framework in Rust. Workspace of 11 crates.

## Build & Test

```bash
cargo build --workspace -q
cargo test --workspace -q
cargo clippy --workspace -q -- -D warnings
cargo fmt --all
```

## Workspace Layout

| Crate                 | Purpose                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| `flashmind-types`     | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events, model info                |
| `flashmind-prompts`   | System prompt constants and workspace-aware prompt builder                                             |
| `flashmind-core`      | `Agent`, `Conversation`, streaming loop, compaction, subagent management                               |
| `flashmind-llm`       | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama)                                       |
| `flashmind-memory`    | Vector memory (SQLite + sqlite-vec + FTS5 hybrid search); session persistence (`session` feature)      |
| `flashmind-tools`     | 30+ built-in tool implementations + subagent tools + `ToolBuilder`                                     |
| `flashmind-cron`      | Cron job scheduling with pluggable storage backends                                                    |
| `flashmind-skills`    | Skill discovery, loading, and execution from `SKILL.md` packages                                       |
| `flashmind-tailscale` | Tailscale local API client and Funnel route management                                                 |
| `flashmind-tui`       | TUI primitives for building interactive agent CLIs                                                     |
| `flashmind`           | Facade crate re-exporting everything under one namespace                                               |

## Dependency Graph

```
flashmind-types          (leaf — no internal deps)
flashmind-prompts        (leaf — no deps at all)
flashmind-tailscale      (leaf — no internal deps)
flashmind-llm            → flashmind-types
flashmind-memory         → flashmind-types
flashmind-core           → flashmind-types
flashmind-cron           → flashmind-types
flashmind-skills         → flashmind-types
flashmind-tools          → flashmind-types, flashmind-core
flashmind-tui            → flashmind-types
flashmind                → all of the above
```

## Subagent Communication

Agents can spawn child agents and communicate with them:

- **Spawning**: `delegate` tool creates a child agent as a tokio task, returns an ID immediately
- **Messaging**: `communicate` sends a message to a child's `InjectQueue`; the child sees it on its next iteration
- **Status**: `agent_status` shows `running (turn N)`, `awaiting message`, `completed`, `failed`, or `cancelled`
- **Waiting**: `agent_wait` blocks the parent until a child finishes; `await_message` (auto-registered in `flashmind-core`) blocks a child until it receives a message — enables back-and-forth conversations without the child finishing prematurely
- **Termination**: `agent_terminate` cancels a child via `CancellationToken`

The `InjectQueue` (in `flashmind-types`) is the core primitive: a thread-safe async queue with push/drain/notified/cancel. Parent→child messages flow through it, and child progress events flow back to the parent's queue.

## Key Design Decisions

1. **Provider-agnostic**: `flashmind-core` depends only on the `LlmProvider` trait from `flashmind-types`. Provider implementations live in `flashmind-llm`.
2. **Streaming-first**: All LLM communication uses `CompletionStream` (`Pin<Box<dyn Stream<Item = Result<StreamEvent>>>>`).
3. **Conversation as IR**: `Conversation` is richer than raw `Message` arrays. Entry kinds (`SystemPrompt`, `Reminder`, `Memory`, `Summary`, etc.) control wire format mapping via `to_messages()`.
4. **Compaction ladder**: Escalating strategies for context pressure: truncate long outputs → LLM summarization → prune tool outputs → strip tool messages → last-exchange fallback.
5. **Tool registry with gating**: Tools can be gated per mode (e.g., read-only plan mode) with path-based exemptions. On-demand loading reduces function-calling overhead.

## Code Conventions

- **Edition**: Rust 2024
- **Errors**: `anyhow::Result` internally; specific error types at public boundaries
- **Logging**: `tracing` macros only — never `println!` in library code
- **Metrics**: `metrics` crate for counters/gauges/histograms
- **Docs**: All public items get rustdoc. Modules start with `//!` explaining purpose.
- **Testing**: Every new public function or struct should have unit tests in `#[cfg(test)] mod tests`
- **Section separators**: `// ---------------------------------------------------------------------------` between logical blocks
- **No thin wrappers**: Prefer adding deps directly to existing crates over creating thin wrapper crates

## Adding a Tool

1. Create `my_tool.rs` in `flashmind-tools/src/`
2. Implement the `Tool` trait: `name()`, `description()`, `parameters()` (JSON Schema), `execute()`, `humanize()`
3. Export from `flashmind-tools/src/lib.rs`
4. Register in `builder.rs` in the appropriate `.with_*()` method

## Adding a Provider

1. Create `new_provider.rs` in `flashmind-llm/src/` implementing `LlmProvider`
2. Only `complete()` is required — it returns a `CompletionStream`
3. Use `sse.rs` for SSE parsing, `http.rs` for client setup
4. Export from `flashmind-llm/src/lib.rs`

## Running Examples

```bash
cargo run -p flashmind --example ollama
cargo run -p flashmind --example streaming
cargo run -p flashmind --example custom_tool
cargo run -p flashmind --example widgets
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl
cargo run -p flashmind --example mcp
```
