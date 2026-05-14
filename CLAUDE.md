# Flashmind

AI agent framework in Rust. Workspace of 12 crates.

## Build & Test

```bash
cargo build --workspace -q
cargo test --workspace -q
cargo clippy --workspace -q -- -D warnings
cargo fmt --all
```

## Workspace Layout

| Crate               | Purpose                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------- |
| `flashmind-types`    | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`), wire types, events, model info |
| `flashmind-prompts`  | System prompt constants and workspace-aware prompt builder                              |
| `flashmind-core`     | `Agent`, `Conversation`, streaming loop, compaction, subagent management                |
| `flashmind-llm`      | Provider implementations (OpenRouter, Anthropic, OpenAI, Ollama)                        |
| `flashmind-memory`   | Vector memory (SQLite + sqlite-vec + FTS5 hybrid search); session persistence (`session` feature) |
| `flashmind-tools`    | 30+ built-in tool implementations + subagent tools + `ToolBuilder`                      |
| `flashmind-app`      | Shared application logic for CLI + desktop: config, sessions, display, memory tools, background agents |
| `flashmind-cron`     | Cron job scheduling with pluggable storage backends                                     |
| `flashmind-skills`   | Skill discovery, loading, and execution from `SKILL.md` packages                        |
| `flashmind-tailscale`| Tailscale local API client and Funnel route management                                  |
| `flashmind-tui`      | TUI primitives for building interactive agent CLIs                                      |
| `flashmind`          | Facade crate re-exporting everything under one namespace                                |

Both `flashmind-cli` (this repo) and `flashmind-desktop` (sibling repo at `../flashmind-desktop`) depend on `flashmind-app` as their shared foundation. Each adds its own UI layer and app-specific tools on top.

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
flashmind-app            → flashmind-types, flashmind-core, flashmind-llm, flashmind-memory, flashmind-tools, flashmind-prompts, flashmind-skills
flashmind                → all of the above
```

## flashmind-app

Shared application layer used by both CLI and desktop. Owns:

- **Config** (`config.rs`): `AppConfig` loaded from `~/.flashmind/config.toml`, path helpers, `CaptureConfig`
- **Provider/LLM** (`provider.rs`, `llm.rs`): `build_active_provider()`, `build_provider_for()`, `build_llm_config()`
- **Tools** (`tools.rs`): `ToolSet` with `build_tools()`, `build_embedder()`, `build_memory_components()`
- **Memory** (`memory.rs`): 5 memory tools (store, recall, forget, edit, list) using SQLite + sqlite-vec + FTS5
- **Sessions** (`session.rs`): `Sessions` wrapper over `SessionStore`, `ConversationEntry` ↔ `SessionEntry` mapping, `LocalSession` metadata
- **Display** (`display.rs`): `DisplayLog`, `DisplayEvent`, `ServerMessage`, JSONL save/load
- **Prompt** (`prompt.rs`): system prompt resolution (config → SOUL.md → default), project instructions, git context
- **Agents** (`agents/`): Background agents spawned after each prompt:
  - `enrichment.rs` — generates kebab-case session titles
  - `capture.rs` — extracts durable facts into memory via a mini `flashmind_core::Agent` with store/recall/forget tools
  - `post_turn.rs` — `spawn_post_turn()` orchestrates both, returns `mpsc::Receiver<PostTurnEvent>`
  - `types.rs` — `PostTurnEvent` enum: `TitleSet`, `MemoryStored`, `MemoryForgotten`, `CaptureComplete`

### Memory Capture

Enabled via config:
```toml
[memory.capture]
enable = true
model = "ollama:llama3.2"   # optional, defaults to main model
debug = false
```

The capture agent runs after each prompt with only the last exchange (user + assistant + truncated tool outputs). It follows a strict workflow: recall existing memories, store at most 3 new durable facts, consolidate related facts, and forget outdated ones. Results are reported via `PostTurnEvent` so callers can display feedback.

## Subagent Communication

Agents can spawn child agents and communicate with them:

- **Spawning**: `delegate` tool creates a child agent as a tokio task, returns an ID immediately
- **Messaging**: `communicate` sends a message to a child's `InjectQueue`; the child sees it on its next iteration
- **Status**: `agent_status` shows `running (turn N)`, `awaiting message`, `completed`, `failed`, or `cancelled`
- **Waiting**: `agent_wait` blocks the parent until a child finishes; `await_message` (auto-registered in `flashmind-core`) blocks a child until it receives a message — enables back-and-forth conversations without the child finishing prematurely
- **Termination**: `agent_terminate` cancels a child via `CancellationToken`

The `InjectQueue` (in `flashmind-types`) is the core primitive: a thread-safe async queue with push/drain/notified/cancel. Parent→child messages flow through it, and child progress events flow back to the parent's queue.

## Code Conventions

- **Edition**: Rust 2024
- **Errors**: `anyhow::Result` internally; specific error types at public boundaries
- **Logging**: `tracing` macros only — never `println!` in library code
- **Metrics**: `metrics` crate for counters/gauges/histograms
- **Docs**: All public items get rustdoc. Modules start with `//!` explaining purpose.
- **Section separators**: `// ---------------------------------------------------------------------------` between logical blocks
- **Streaming-first**: All LLM communication uses `CompletionStream` (`Pin<Box<dyn Stream<Item = Result<StreamEvent>>>>`)
- **No thin wrappers**: Prefer adding deps directly to existing crates over creating thin wrapper crates

## Running Examples

```bash
cargo run -p flashmind --example ollama
cargo run -p flashmind --example streaming
cargo run -p flashmind --example custom_tool
cargo run -p flashmind --example widgets
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl
cargo run -p flashmind --example mcp
```
