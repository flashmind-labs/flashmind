# flashmind

Facade crate for the [Flashmind](https://github.com/flashmind-labs/flashmind) AI agent framework.

Re-exports the Flashmind libraries under one namespace. See the
[API documentation](https://docs.rs/flashmind) for a checked streaming example,
provider setup, and feature requirements.

## Usage

```toml
[dependencies]
flashmind = "0.3"
```

## Modules and features

| Module | Feature |
|--------|---------|
| `flashmind::types` | Always available |
| `flashmind::core` | `core` (default) |
| `flashmind::llm` | `llm` (default) |
| `flashmind::tools` | `tools` |
| `flashmind::memory` | `memory`, `session`, `session-sqlite`, or `memory-postgres` |
| `flashmind::prompts` | `prompts` |
| `flashmind::cron` | `cron` |
| `flashmind::skills` | `skills` |
| `flashmind::tailscale` | `tailscale` |
| `flashmind::tui` | `tui` |

The default `sqlite` feature adds SQLite support when `memory`, `session`, or
`tools` is enabled. Enable `memory` for SQLite vector storage with HTTP embeddings:

```toml
flashmind = { version = "0.3", features = ["memory"] }
```

`mcp`, `gmail`, `google-calendar`, `google-contacts`, `github`, `slack`,
`outlook`, `caldav`, `composio`, and `lsp` enable their integration and `tools`.
`full` enables all modules and tool integrations, SQLite memory, and sessions.
Add `memory-postgres` separately for the PostgreSQL memory backend.

Use `default-features = false` to select minimal components. `session` alone
provides JSONL history without SQLite or HTTP embedding clients.
`session-sqlite` alone provides `memory::SqliteSessionStore` without vector
storage, HTTP embedding clients, or JSONL hashing.
`mistral-realtime`, `http-compression`, `http2`, `request-compression`, `markdown`,
`clipboard`, `tool-http`, `tool-skills`, and `subagent` are separate default
features. They configure enabled modules without adding those modules themselves.
`tools` alone provides file and system tools without the agent runtime.

See the [main README](https://github.com/flashmind-labs/flashmind) for the full getting-started guide.

## License

MPL-2.0
