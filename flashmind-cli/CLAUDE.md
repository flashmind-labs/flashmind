# flashmind-cli

Binary crate producing the `flashmind` executable. Single file: `src/main.rs`.

## Build & Run

```bash
cargo build -p flashmind-cli -q
cargo run -p flashmind-cli             # interactive REPL
cargo run -p flashmind-cli -- -p "hi"  # one-shot mode
cargo run -p flashmind-cli -- setup    # guided setup wizard
cargo run -p flashmind-cli -- resume   # restore a previous session
```

## Modes

| Mode | Trigger | Behavior |
| --- | --- | --- |
| Interactive | no flags | REPL with streaming, status bar (model, cost, context usage), session auto-save |
| One-shot | `-p "prompt"` | Print response to stdout and exit |
| Setup | `flashmind setup` | Wizard: provider → API key → model picker → reasoning level → memory → search → writes config |
| Resume | `flashmind resume` | Pick from saved sessions, restore conversation, continue interactively |

## CLI Flags

- `-m, --model` — model string (e.g. `ollama:llama3.2`, `openrouter:anthropic/claude-sonnet-4`)
- `-p, --prompt` — one-shot prompt
- `--system` — system prompt override

## Config

TOML at `~/.flashmind/config.toml`. All API key fields have env var overrides (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_URL`, `BRAVE_API_KEY`, `FIRECRAWL_API_KEY`).

Fields: `model`, `system_prompt`, `reasoning` (off/low/medium/high), provider API keys, `memory_provider` (openrouter/openai), `memory_model`, search keys.

## Registered Tools

File ops, bash/exec, web search (Brave/Firecrawl), time. Memory tools (`memory_store`, `memory_recall`, `memory_forget`) added when an embedding provider is configured. MCP management tools (`mcp_add`, `mcp_remove`, `mcp_list`, `mcp_auth`) are always registered. MCP wrapper tools for connected servers are registered dynamically via `ToolSync` after each agent turn.

## MCP

MCP servers are managed via CLI subcommands (`flashmind mcp add/remove/list`) or agent tools during chat. Configs persist as JSON files in `~/.flashmind/mcp/` via `McpDiskConfig`. On launch, saved servers auto-reconnect and their tools appear in the registry. Dynamic tool registration uses `ToolSync::sync()` after each agent turn — the `mcp_add` tool triggers connection + tool discovery, and `ToolSync` drains the pending ops into the live `ToolRegistry`.

## Memory

Optional. Requires `memory_provider` + corresponding API key in config. Uses `flashmind-memory` with SQLite + sqlite-vec hybrid search. Embedding via OpenRouter or OpenAI. DB at `~/.flashmind/memory.db`. Three custom `Tool` impls defined inline in main.rs.

## Session Persistence

Conversations saved to `~/.flashmind/sessions.db` via `flashmind-memory`'s `session` feature. Rewrites full conversation each turn. Single `"default"` chat key for interactive mode; resume mode shows all saved sessions.

## Setup Wizard

Interactive TUI flow using `ChoicePicker` and custom `ModelPicker`:

1. Provider (Ollama/OpenRouter/Anthropic/OpenAI)
2. API key (prompted if not in config/env)
3. Model — searchable table with pricing, context window, capabilities columns
4. Reasoning level (if model supports it)
5. Memory embedding provider (OpenRouter/OpenAI/None)
6. Web search (Brave/Firecrawl/None)
7. Writes `~/.flashmind/config.toml`

## Key Dependencies

Uses `flashmind-core` (Agent), `flashmind-llm` (all 4 providers), `flashmind-tools` (ToolBuilder), `flashmind-memory` (MemoryStore + session), `flashmind-prompts` (CODING_AGENT, MEMORY_INSTRUCTIONS), `flashmind-tui` (Tui, Repl, ChoicePicker, StatusInfo, styles).
