# flashmind

Facade crate for the [Flashmind](https://github.com/flashmind-labs/flashmind) AI agent framework.

Re-exports the Flashmind libraries under one namespace. See the
[API documentation](https://docs.rs/flashmind) for a checked streaming example,
provider setup, and feature requirements.

## Usage

```toml
[dependencies]
flashmind = "0.1"
```

## Modules and features

| Module | Feature |
|--------|---------|
| `flashmind::types` | Always available |
| `flashmind::core` | `core` (default) |
| `flashmind::llm` | `llm` (default) |
| `flashmind::tools` | `tools` |
| `flashmind::memory` | `memory`, `session`, or `memory-postgres` |
| `flashmind::prompts` | `prompts` |
| `flashmind::cron` | `cron` |
| `flashmind::skills` | `skills` |
| `flashmind::tailscale` | `tailscale` |
| `flashmind::tui` | `tui` |

The default `sqlite` feature adds SQLite support when `memory` or `tools` is
enabled. Enable `memory` for SQLite vector storage with HTTP embeddings:

```toml
flashmind = { version = "0.1", features = ["memory"] }
```

`mcp`, `gmail`, `google-calendar`, `google-contacts`, `github`, `slack`,
`outlook`, `caldav`, `composio`, and `lsp` enable their integration and `tools`.
`full` enables all modules and tool integrations, SQLite memory, and sessions.
Add `memory-postgres` separately for the PostgreSQL memory backend.

See the [main README](https://github.com/flashmind-labs/flashmind) for the full getting-started guide.

## License

MPL-2.0
