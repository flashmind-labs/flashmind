# Flashmind

AI agent framework in Rust. Workspace of 11 crates, Rust 2024 edition, resolver v3.

## Build & Test

```bash
cargo build --workspace -q
cargo test --workspace -q
cargo clippy --workspace -q -- -D warnings
cargo fmt --all
```

Feature-gated examples:

```bash
cargo run -p flashmind --example mcp --features mcp
cargo run -p flashmind --example composio --features composio
```

## Workspace Layout

All crates live under the repo root (not in a `crates/` subdirectory).

| Crate | Purpose |
| --- | --- |
| `flashmind-types` | Shared traits (`LlmProvider`, `Tool`, `MemoryProvider`, `CommandAllowList`), wire types, events, model info, `ToolRegistry`, `InjectQueue` |
| `flashmind-prompts` | System prompt constants (`CODING_AGENT`, `CONVERSATIONAL`, `CODE_REVIEW`, `SUMMARIZER`) and composable fragments (`TOOL_USE_INSTRUCTIONS`, `SAFETY_GUARDRAILS`, `MEMORY_INSTRUCTIONS`, `CRON_INSTRUCTIONS`). Zero dependencies. |
| `flashmind-core` | `Agent` + `AgentBuilder`, `Conversation` IR, streaming loop (`stream_llm_response`), compaction ladder, subagent `AgentManager` + `SpawnBuilder` |
| `flashmind-llm` | Provider implementations: OpenRouter, Anthropic, OpenAI (+ vLLM/LiteLLM compatible), Ollama. Shared SSE parser, HTTP retry, rate limiting, 2500+ model capability registry |
| `flashmind-memory` | Vector memory (SQLite + sqlite-vec + FTS5 hybrid search), embedding providers (OpenAI/Ollama/OpenRouter), session persistence (`session` feature) |
| `flashmind-tools` | 70+ built-in tools + subagent tools + MCP + Composio + OAuth integrations (Google, Outlook, GitHub, Slack, CalDAV, Cloudflare) + config integrations (Postgres, MySQL, ClickHouse, Redis, Docker, SSH, K8s, Twilio/SMTP) + `ToolBuilder` |
| `flashmind-cron` | Cron scheduling with `CronStore` trait, TOML file backend, nom-based POSIX cron parser, `CronRunner` + `CronHandler` |
| `flashmind-skills` | Skill discovery from `SKILL.md` files, `SkillProvider` trait, `DiskSkillProvider`, `SkillRunner` with env isolation and secret redaction |
| `flashmind-tailscale` | Tailscale local API (Unix socket) + CLI fallback, `FunnelManager` for route exposure |
| `flashmind-tui` | Append-mode TUI: `Repl`, `TextArea`, `EventRenderer`, `PlanPicker`, `ChoicePicker`, `Dropdown`, `Tree`, `StatusBar`, `Spinner`. Optional `markdown` feature for rich rendering |
| `flashmind` | Facade re-exporting all crates under `flashmind::{types, core, llm, tools, memory, prompts, cron, skills, tailscale, tui}` |

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

## Feature Flags

The `flashmind` facade has a `full` feature that enables everything. Individual features:

| Feature | Crate(s) affected | What it gates |
| --- | --- | --- |
| `tui` | flashmind → flashmind-tui | TUI re-export (optional dep) |
| `markdown` | flashmind-tui | Rich markdown rendering (nom, comfy-table, ansi-to-tui) |
| `session` | flashmind-memory | Conversation session persistence tables |
| `subagent` | flashmind-core, flashmind-tools | Child agent spawning and management (default in tools) |
| `mcp` | flashmind-tools | MCP server management via `rmcp` |
| `gmail` | flashmind-tools | Gmail OAuth + 13 tools |
| `google-calendar` | flashmind-tools | Google Calendar OAuth + 6 tools |
| `google-contacts` | flashmind-tools | Google Contacts OAuth + 6 tools |
| `outlook` | flashmind-tools | Outlook Mail/Calendar/Contacts OAuth |
| `caldav` | flashmind-tools | CalDAV (Fastmail, Nextcloud, iCloud) |
| `github` | flashmind-tools | GitHub OAuth + 8 tools |
| `slack` | flashmind-tools | Slack OAuth + 6 tools |
| `cloudflare` | flashmind-tools | Cloudflare API + 8 tools |
| `composio` | flashmind-tools | Composio.dev unified app integration |
| `redis` | flashmind-tools | Redis (dep: `redis`) |
| `postgres` | flashmind-tools | PostgreSQL (dep: `tokio-postgres`) |
| `mysql` | flashmind-tools | MySQL (dep: `mysql_async`) |
| `clickhouse` | flashmind-tools | ClickHouse (HTTP-based, no extra dep) |
| `docker` | flashmind-tools | Docker API (dep: `bollard`) |
| `ssh` | flashmind-tools | SSH sessions (dep: `russh`) |
| `kubernetes` | flashmind-tools | K8s API (dep: `kube`, `k8s-openapi`) |
| `messaging` | flashmind-tools | SMTP email (dep: `lettre`) |

## Core Architecture

### Agent Lifecycle

```
Agent::builder(provider).tools(registry).llm(config).auto_compact(true).build()

Agent::start(conversation, cancel_token, input, max_iterations) → impl Stream<Item = AgentEvent>
  ├─ Input injection (user message + optional context/multimodal parts)
  └─ Iteration loop:
     ├─ run_turn() → stream_llm_response() → AgentStream<AgentEvent, Result<LlmResponse>>
     │   ├─ Sends CompletionRequest to provider
     │   ├─ Yields TextDelta, ReasoningDelta, Usage events
     │   ├─ Accumulates tool calls via ToolCallTracker
     │   └─ Returns LlmResponse with assembled content + finalized tool calls
     ├─ TurnStatus::Done → break, yield Done event
     ├─ TurnStatus::ToolCalls → execute each tool, add results, continue
     │   ├─ Yield ToolStart → execute → yield ToolResult (+ FileDiff events)
     │   └─ On Interrupt → yield Interrupted, break tool loop
     ├─ TurnStatus::CompactionNeeded → auto-compact or yield CompactionNeeded
     └─ Error → recoverable check → escalating recovery (strip binary → compact → truncate)
```

The `Agent` struct owns `provider: Arc<dyn LlmProvider>`, `tools: ToolRegistry`, `llm: AgentLlmConfig`, `capabilities: ModelCapabilities`, `context_window: u32`, `auto_compact: bool`. Conversation is caller-owned (passed by `&mut`), enabling reuse across turns.

### Conversation IR

`Conversation` is richer than raw `Message` arrays. Each entry has an `EntryKind`:

| EntryKind | Wire mapping | Purpose |
| --- | --- | --- |
| `SystemPrompt(String)` | `Message::system()` | Always position 0 |
| `Developer { content, tag, metadata }` | `Message::developer()` | Tags: `"reminder"`, `"memory"` (has id+score metadata), `"agent_progress"` (has agent id), `"summary"`, or `null` |
| `User { content, parts }` | `Message::user()` / `::user_with_parts()` | Plain text or multimodal |
| `Assistant { content, tool_calls }` | `Message::assistant()` / `::assistant_with_tool_calls()` | May include tool calls |
| `Tool { call_id, output }` | `Message::tool_result()` | Linked to assistant's tool call by ID |

`sanitize()` enforces tool call/result pairing and adjacency before `to_messages()` — required by provider APIs that reject interleaved entries between tool calls and results.

### Compaction Ladder

Triggered when `finish_reason == Length` (without explicit max_tokens) or when prompt tokens exceed `context_window - reserve_tokens` (default reserve `16_384`; the `compact_threshold` fraction, default `0.8`, acts as a floor for small-context models).

| Stage | Strategy | Effect |
| --- | --- | --- |
| 1 | Truncate long tool outputs | Cap at 200 bytes + `[truncated]` |
| 2 | LLM summarization | Send conversation to compaction model (temp 0.3, 120s timeout), replace older entries with a summary entry. Keeps the last `keep_recent_tokens` (default `20_000`) of history verbatim, snapped to a user-turn boundary, and appends a computed `## Files Touched` section |
| 3 | Strip tool messages | Remove all Tool entries, clear tool_calls from Assistant entries |
| 4 | Last exchange fallback | Keep only system prompt + last user + last assistant |

Error recovery follows the same escalation: strip binary parts → truncate tool outputs → compact → strip tool messages → truncate to last exchange. Up to 2 recovery attempts before giving up.

### Streaming Types

- `CompletionStream` = `Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>`
- `AgentStream<I, O>` = stream yielding `Outcome::Item(I)` events with a terminal `Outcome::Done(O)` stored internally, accessible via `take_result()`
- `StreamEvent` variants: `ContentDelta`, `ReasoningDelta`, `ToolCallStart`, `ToolCallDelta`, `Usage`, `AudioDelta`, `FileAttachment`, `Finished`
- `AgentEvent` variants: `TextDelta`, `ReasoningDelta`, `ToolStart`, `ToolResult`, `AudioChunk`, `FileDiff`, `Status`, `Compacted`, `Usage`, `Done`, `Error`, `SpawnedEvent`, `Interrupted`, `CompactionNeeded`

### Cancellation

Every stream gets a child `CancellationToken` from the caller's token. A `CancelOnDrop` guard cancels on drop. `tokio::select! { biased; _ = cancel_token.cancelled() => break, event = stream.next() => event }` checks cancellation each iteration.

## Subagent Communication

`AgentManager` enforces `max_concurrent` and `max_depth` limits. Each spawned agent runs in an independent tokio task with its own `Conversation`.

| Tool | Purpose | Key args |
| --- | --- | --- |
| `delegate` | Spawn child agent | `task` (required), `name`, `system_prompt` |
| `agent_status` | Check status | `agent` (name or UUID prefix) |
| `agent_wait` | Block until done | `agent`, `timeout` |
| `communicate` | Send message | `agent`, `message` |
| `agent_terminate` | Cancel child | `agent` |

`SpawnBuilder` supports `.name()`, `.system_prompt()`, `.model()`, `.tools()`, `.strip_prefixes()` (remove parent tools by prefix), `.max_iterations()`. If `.tools()` is set, `.strip_prefixes()` is ignored.

Status enum: `Running { turn }`, `Completed { result }`, `Failed { error }`, `Cancelled`.

Resolution order: exact name match → UUID prefix (hex, up to 8 chars) → full UUID parse.

## LLM Providers

### Provider Differences

| | OpenRouter | Anthropic | OpenAI | Ollama |
| --- | --- | --- | --- | --- |
| Protocol | SSE (OpenAI-compat) | SSE (native events) | SSE (OpenAI) | NDJSON |
| Auth | Bearer token | x-api-key header | Bearer token (optional) | None |
| Context window | API discovery (`/api/v1/models`) | Fixed 200k | API discovery (`/v1/models`) | `/api/show` metadata |
| Reasoning | `chat_template_kwargs.enable_thinking` | `thinking` block config | N/A | `think` param |
| Capabilities | `input_modalities`/`output_modalities` + `supported_parameters` → fallback to oss_capabilities | All Claude → `ModelCapabilities::all()` | oss_capabilities registry | `/api/show` capabilities array |
| TTS/STT | Yes (dedicated endpoints) | No | Yes | No |
| Special handling | Kimi: skip presence_penalty; Gemma4: `skip_special_tokens=false` when thinking | System messages → top-level `system` field; developer role → `<system>` tags in user messages; tool results → user messages with ToolResult blocks | vLLM compat: keeps `skip_special_tokens`; OpenAI.com: strips `top_k`, `min_p`, `repetition_penalty` | Coerces tool args to objects; warns on >100KB payloads |

### Shared Infrastructure (flashmind-llm)

- **`http.rs`**: Shared `LazyLock<Client>` (rustls + webpki-roots), `send_with_retry()` with 120s budget / 30 max retries, handles 429 (respects Retry-After) and 5xx
- **`sse.rs`**: OpenAI-compatible SSE parser with `ToolCallTracker` (accumulates tool call deltas, emits `ToolCallStart` once both id and name arrive), `process_chunk()` handles all delta types
- **`wire_types.rs`**: OpenAI-compatible request/response schemas (`ApiMessage`, `ApiContent`, `ApiToolCall`, `StreamChunk`). `ApiMessage` content serializes as string (text only) or array (multimodal)
- **`oss_capabilities.rs`**: 2500+ model entries mapping IDs to `ModelCapabilities`. Lookup: exact match → partial name → contains. Default fallback: `tool_calling=true`, all media `false`
- **`ContextWindowCache`**: Thread-safe cache with 1-hour TTL per model

### Rate Limiting

Token-bucket via `ratelimit` crate. Per-provider, configurable RPM. `wait_for_rate_limit()` blocks until token available.

## Tool System

### Tool Trait

```rust
#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "..." }
    fn parameters(&self) -> Value { json!({...}) }  // JSON Schema
    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> { ... }
    fn humanize(&self, args: &Value) -> String { ... }  // optional
    fn timeout_secs(&self) -> Option<u64> { None }      // optional
    fn max_output_bytes(&self) -> usize { 256 * 1024 }  // optional, default 256 KiB
    fn max_output_lines(&self) -> usize { 10_000 }      // optional
}
```

`ToolContext` provides: `tool_call_id`, `args: Value`, `working_dir: Option<&PathBuf>`, `cancel_token()`, `child_token()`, `parse_args<T: DeserializeOwned>()`, `check_absolute_path()`.

`ToolResult` variants: `Success { output, sources, diffs }`, `Failure { output }`, `Interrupt { payload }`. Factory methods: `success()`, `failure()`, `success_with_diffs()`, `interrupt()`. Chain: `.with_sources(vec)`.

### ToolRegistry

`register()`, `remove()`, `alias()` (not exposed in definitions), `get()` (resolves aliases), `retain()`, `strip_prefixes()`, `list()`, `definitions()` (sorted, stable), `execute()` (races timeout + cancellation).

### ToolBuilder

Fluent API for composing tool registrations:

```rust
ToolBuilder::new()
    .with_providers(providers)
    .with_offline(false)
    .file_ops(ocr_model, &protected)
    .bash(secrets, &protected, forbidden_cmds, allowlist)
    .search(brave_key, firecrawl_key)
    .time().sqlite().http().json().audio(...).models()
    .generate(image_model, video_model, output_dir)
    .subagents(manager, provider, llm)
    .mcp(provider, auth_handler)
    .google(&config) / .google_readonly(&config)
    .outlook(config) / .outlook_readonly(config)
    .github(config) / .github_readonly(config)
    .slack(config) / .slack_readonly(config)
    .caldav(config) / .caldav_readonly(config)
    .cloudflare(config) / .cloudflare_readonly(config)
    .docker(config) / .docker_readonly(config)
    .kubernetes(config) / .kubernetes_readonly(config)
    .ssh(config) / .ssh_readonly(config)
    .messaging(config) / .messaging_readonly(config)
    .postgres(config) / .postgres_readonly(config)
    .mysql(config) / .mysql_readonly(config)
    .clickhouse(config) / .clickhouse_readonly(config)
    .redis(config) / .redis_readonly(config)
    .composio(config)
    .build_with_sync().await  // → (ToolRegistry, ToolSync)
```

### Tool Catalog by Category

**File & Filesystem** (12): `file_read`, `file_write`, `file_delete`, `file_list`, `read_lines`, `glob`, `grep`, `str_replace`, `str_replace_regex`, `image_read`, `str_diff`

**Shell & Process** (2): `exec` (alias: `bash_exec`), `process`

**Web Search** (6): `brave_search`, `firecrawl_search`, `web_search_read`, `web_crawl`, `web_scrape`, `web_map`

**HTTP** (1): `http_request`

**Database** (6): `sqlite_query`, `postgres_query`, `mysql_query`, `clickhouse_query`, `redis_query`, `redis_info`

**Audio** (3): `tts`, `transcribe`, `list_voices`

**Image/Video** (3): `image_gen`, `image_edit`, `video_gen`

**Utilities** (3): `get_time`, `json_query`, `list_models`

**Subagent** (5): `delegate`, `agent_status`, `agent_wait`, `communicate`, `agent_terminate`

**Google** (25): Gmail (13), Calendar (6), Contacts (6) — all via OAuth

**Outlook** (18): Mail (10), Calendar (6), Contacts (2+) — OAuth

**GitHub** (8): repos, issues, PRs, notifications, comments, merge — OAuth

**Slack** (6): channels, messages, search, reactions — OAuth

**CalDAV** (9): calendars, events — any CalDAV server

**Cloudflare** (8): zones, DNS, workers, cache — API key

**Docker** (9): containers, images, exec, logs — config

**Kubernetes** (11): pods, deployments, services, apply, exec, scale, logs — config

**SSH** (5): open, close, exec, upload, download — config

**Messaging** (6): Twilio SMS/WhatsApp (4), SMTP email (1) — config

**MCP** (4): `mcp_add`, `mcp_remove`, `mcp_list`, `mcp_auth` — dynamic server management

**Composio**: dynamic tools via Composio.dev (250+ unified apps)

### Dynamic Tool Registration

OAuth tools use a two-phase flow:

1. If cached token exists → register service tools immediately
2. If no token → register only `{provider}_auth` tool
3. At runtime: auth tool returns interrupt with URL → user authenticates → tool exchanges code for token → pushes service tools into `PendingTools` queue
4. `ToolSync::sync()` drains `PendingTools` each agent loop iteration

MCP tools similarly: `mcp_add` starts server, fetches tools, queues `McpToolOp::Register`. Background reconnection on startup via `load_saved()`.

### Gating

- **Readonly mode**: Every integration has `.readonly()` variant. DB tools detect write keywords. Bash has forbidden command blocklist + `CommandAllowList`
- **Offline mode**: `.with_offline(true)` skips all network tools
- **Path protection**: `ProtectedPaths` enforces allowed/forbidden paths in file ops and bash

### Shared Tool Infrastructure

| Module | Purpose |
| --- | --- |
| `oauth.rs` | `CachedToken`, `load_token()`, `save_token()`, token persistence in `~/.flashmind/cache/tokens/` |
| `db_common.rs` | `is_write_query()`, `format_table()`, `clamp_limit()` |
| `file_cache.rs` | File content caching for file_read, str_replace |
| `protected.rs` | Path allowlist/blocklist enforcement |
| `search_cache.rs` | Dedup for web search results |
| `tool_sync.rs` | `ToolSync`: drains pending MCP + OAuth tools each agent loop |

## Memory System

### Hybrid Search

Combines vector similarity (sqlite-vec) and full-text search (FTS5 BM25):

1. Vector phase: embed query → query sqlite-vec → top 3x candidates, score = `1.0 - (distance / 2.0)`
2. FTS5 phase: clean query → BM25 ranking → normalize scores to [0, 1]
3. Fusion: `(vec_sim * 1.0 + fts * 0.3).min(1.0)`
4. Metadata filtering: key-value AND logic post-fusion
5. Expiration: both phases filter `expires_at IS NULL OR expires_at > now`

### Builder API

```rust
// Store
store.store("content").meta("key", "value").meta_opt("key", opt).expires_at(ts).await?

// Search
store.search("query").filter("key", "value").limit(10).await?
```

Both use `IntoFuture` for deferred execution.

### Schema

- **`memories`**: id (TEXT PK), content, created_at, expires_at
- **`memories_vec`**: sqlite-vec virtual table, float[N] embeddings
- **`memories_fts`**: FTS5 external content table synced via triggers (AFTER INSERT/UPDATE/DELETE)
- **`memory_meta`**: key-value pairs with `ON DELETE CASCADE` from memories

Pragmas: WAL journal mode, `synchronous=NORMAL`, `busy_timeout=5000`, `foreign_keys=ON`.

### Embedding Providers

| Provider | Default dims | Batch support | Auto-detect dims |
| --- | --- | --- | --- |
| OpenAI | 1536 | Yes (native) | No |
| Ollama | 384 | Sequential | Yes (lazy via RwLock) |
| OpenRouter | 1536 | Sequential | No |

Factory: `create_embedding_provider(config, fallback_api_key)`.

### Session Persistence (feature: `session`)

`SessionStore` wraps the same `tokio_rusqlite::Connection`. Schema: `sessions` table with chat_key, entry_kind (JSON), content, tool_calls (JSON), tool_call_id, tool_name, metadata (JSON), turn_index, created_at.

API: `save_entry()`, `save_entries()` (batch in transaction), `load(chat_key)`, `delete_session()`, `rewrite()` (atomic delete + replace), `branch(from, to)`, `prune(max_age_days)`, `list_sessions()`.

## Cron System

- **`CronSchedule`**: nom-parsed POSIX 5-field expressions. Supports `*`, `*/N`, `N-M`, `N-M/S`, comma lists, named months/days. POSIX OR semantics for day_of_month + day_of_week.
- **`JobSchedule`** enum: `Cron(String)`, `Once(DateTime)`, `OnWake { from_hour, min_gap_secs, max_gap_secs }`
- **`CronStore`** trait: async CRUD. `TomlCronStore` is the file-backed implementation.
- **`CronRunner`**: spawns per-job tokio tasks, watches for mutations via `Arc<Notify>`, respawns affected jobs.
- **`CronLog`**: JSONL-backed execution history with `record()`, `recent(n)`, `for_job(id)`.
- **Tools** (6): `cron_create`, `schedule_once`, `cron_list`, `cron_edit`, `cron_delete`, `cron_history`

## Skills System

- **`SKILL.md`** format: optional YAML frontmatter (`name`, `description`, `usage`) + markdown body. No frontmatter → name from directory name.
- **`DiskSkillProvider`**: scans directories for SKILL.md, deduplicates by name (first wins). `refresh()` re-scans.
- **`SkillRunner`**: executes commands in skill directory, loads `.env`, prepends skill dir to PATH, applies timeout, redacts secret values (>=4 chars from .env) from output.
- **Tools** (4): `skill_list`, `skill_load`, `skill_run`, `skill_install`

## TUI System

Append-mode terminal (preserves scrollback, no full-screen viewport). Uses ratatui for styling primitives but renders directly to stdout via crossterm escape sequences.

- **`Repl`**: `read_input()` → `ReplEvent`, `stream_response(impl Stream<Item = AgentEvent>)` → renders events incrementally. Activity indicator (`set_activity`/`clear_activity`) shows spinner + label in the input bar title during agent turns. Reverse-i-search (Ctrl+R) for history.
- **`EventRenderer`**: AgentEvent → styled `Line`s. Buffers text deltas, flushes at newlines. Tool results do in-place replacement of running-tool lines. Spawned events prefixed with `[task_name]`.
- **Markdown rendering** (`markdown` feature): `markdown.rs` (nom-based block+inline parser) → `markdown_render.rs` (ANSI renderer) → `ansi-to-tui` (ratatui Lines). Block types: Paragraph, Heading, CodeBlock, Table, BulletList, OrderedList, Blockquote, HorizontalRule. Incremental flush via `render_text_incremental` renders complete blocks during streaming while holding incomplete ones (unclosed code fences, partial tables). Paragraphs join single `\n` into spaces (soft breaks). Lone ordered items (e.g. `1. heading`) stay as paragraphs with prefix preserved.
- **`TextArea`**: multi-line input with word-aware nav (Ctrl+Left/Right), Emacs bindings (Ctrl+A/E/K/U), auto-scroll
- **Interactive widgets**: `PlanPicker` (approval with checkboxes), `ChoicePicker` (single-select), `Dropdown`, `Tree` (hierarchical with box-drawing), `StatusBar` (composable with spinner + toasts)
- **`Tui`**: `draw_lines()` → `DrawnArea`, `erase(DrawnArea)`, `raw_mode()` → RAII guard

Styles: `S_DIM`, `S_TOOL_RUN` (yellow), `S_TOOL_OK` (green), `S_TOOL_FAIL` (red), `S_ERROR` (bold red), `S_USER` (bold green), `S_AGENT` (bold cyan), `S_SPAWNED` (magenta), `S_DIFF_ADD` (green), `S_DIFF_DEL` (red).

## Key Types Reference

### flashmind-types

| Type | Purpose |
| --- | --- |
| `LlmProvider` trait | `complete()` (required) + optional `text_to_speech()`, `transcribe()`, `generate_image()`, `generate_video()`, `list_models()`, `context_window()`, `capabilities()` |
| `Tool` trait | `name()`, `description()`, `parameters()`, `execute()` required; `humanize()`, `timeout_secs()`, `max_output_bytes()`, `max_output_lines()` optional |
| `MemoryProvider` trait | `store()`, `search()`, `forget()` |
| `CommandAllowList` trait | `is_allowed()`, `add_session_pattern()` |
| `Message` | Wire format: role, content, tool_calls, tool_call_id, parts (multimodal), timestamp |
| `ContentPart` | `Text`, `Image`, `ImageUrl`, `Document`, `Video`, `VideoUrl`, `Audio` — tagged union |
| `Model` | `provider:name` format (e.g., `openrouter:anthropic/claude-sonnet-4`). `AliasedModel` supports `alias,real_name` |
| `Provider` | `OpenRouter`, `Ollama` (default), `Anthropic`, `OpenAi`, `Connect` |
| `AgentLlmConfig` | model + max_tokens + reasoning (`Off`/`On`) + `SamplingParams` (temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty — all `Option<Decimal>`) |
| `ModelCapabilities` | `tool_calling`, `images`, `documents`, `video`, `audio`, `reasoning`, `audio_output`, `image_generation`, `video_generation` |
| `CompletionRequest` | messages + tools + model + max_tokens + reasoning + sampling + modalities + audio/image config |
| `TokenUsage` | prompt_tokens, completion_tokens, total_tokens |
| `ToolRegistry` | O(1) lookup, alias resolution, sorted definitions output |
| `ProviderRegistry` | `Arc<HashMap<Provider, Arc<dyn LlmProvider>>>` |

### Dual ToolResult Types

- **`tool::ToolResult`** (rich): `Success { output, sources, diffs }` / `Failure { output }` / `Interrupt { payload }` — returned by `Tool::execute()`
- **`message::ToolResult`** (wire): `{ tool_call_id, output, success }` — sent to LLM

## Code Conventions

- **Edition**: Rust 2024, resolver v3
- **Errors**: `anyhow::Result` internally; `thiserror` enums at public boundaries (crates: types, memory, cron, skills, tailscale)
- **Logging**: `tracing` macros only — never `println!` in library code. Use `%` for Display (`path = %p.display()`), bare for Debug. Levels: `info!` for startup/major ops, `debug!` for details, `warn!` for recoverable errors
- **Metrics**: `metrics` crate. Naming: `{subsystem}.{operation}.{unit}` (e.g., `memory.store.duration_seconds`, `llm.prompt_tokens`, `agent.compactions.triggered`). Counters for events, histograms for durations/sizes, gauges for state
- **Docs**: All public items get rustdoc. Modules start with `//!` explaining purpose
- **Testing**: `#[cfg(test)] mod tests` with `#[tokio::test]`. Mock trait impls (e.g., `ConstantEmbedder`, `EchoProvider`). `tempfile` for temporary directories. Tests exercise full public API flows
- **Section separators**: `// ---------------------------------------------------------------------------` between logical blocks
- **No thin wrappers**: Prefer adding deps directly to existing crates
- **Async patterns**: `async-trait` for trait definitions, `async_stream::stream!` for generator-style streams, `IntoFuture` for builder-to-future conversion, `tokio_rusqlite::Connection::call()` for blocking SQLite in async context
- **Builder pattern**: fluent methods returning `Self`, `build()` finalizes. Simple types use `pub fn new()`
- **Serde conventions**: `#[serde(tag = "type")]` or `#[serde(tag = "type", content = "data")]` for enums, `rename_all = "snake_case"` / `"lowercase"`, `skip_serializing_if = "Option::is_none"` for optional fields. Complex values stored as JSON strings in SQLite
- **TLS**: rustls (no openssl) with webpki-roots. Explicit `rustls-no-provider` + `ring` feature combo in llm and tools crates
- **Decimal precision**: `rust_decimal` for sampling params (avoids f64 quantization)
- **Client sharing**: Integration tools hold `Arc<{Provider}Client>` for connection pool and token state reuse

## Adding a Tool

1. Create `my_tool.rs` in `flashmind-tools/src/`
2. Implement the `Tool` trait: `name()`, `description()`, `parameters()` (JSON Schema), `execute()`, optionally `humanize()`
3. Export from `flashmind-tools/src/lib.rs`
4. Register in `builder.rs` — either add to an existing `.with_*()` method or create a new builder method
5. Add unit tests in `#[cfg(test)] mod tests`

For integration tools with OAuth: follow the pattern in `google/`, `outlook/`, etc. — create `auth.rs` (credentials + auth_url + exchange_code), `auth_tool.rs` (two-phase interrupt flow), client module, and service tool modules.

## Adding a Provider

1. Create `new_provider.rs` in `flashmind-llm/src/` implementing `LlmProvider`
2. Only `complete()` is required — it returns a `CompletionStream`
3. Use `sse.rs` + `wire_types.rs` for OpenAI-compatible providers, or implement custom parsing (see `anthropic.rs` for native SSE, `ollama/` for NDJSON)
4. Use `http.rs` for `send_with_retry()` and shared client setup
5. Add to `oss_capabilities.rs` if the provider serves open-source models
6. Export from `flashmind-llm/src/lib.rs`

## Running Examples

```bash
cargo run -p flashmind --example ollama              # Chat loop with local Ollama
cargo run -p flashmind --example streaming            # Event stream inspection with mock provider
cargo run -p flashmind --example custom_tool          # Custom Tool trait implementation
cargo run -p flashmind --example command_approval      # Interrupt-based tool approval
cargo run -p flashmind --example widgets              # TUI widget showcase
MODEL=qwen3.5:2b cargo run -p flashmind --example tui_repl  # Interactive REPL
cargo run -p flashmind --example image_gen            # Image generation (DALL-E)
cargo run -p flashmind --example image_edit           # Image gen via multimodal chat
cargo run -p flashmind --example video_gen            # Video generation (Veo)
cargo run -p flashmind --example list_models          # Model discovery
cargo run -p flashmind --example subagents            # Multi-agent debate
cargo run -p flashmind --example mcp --features mcp   # MCP server management
cargo run -p flashmind --example composio --features composio  # Composio integration
```
