# flashmind-tools

70+ built-in tool implementations for [Flashmind](https://github.com/flashmind-labs/flashmind) agents.

## Tool Categories

| Category | Tools |
|----------|-------|
| File & filesystem | `file_read`, `file_write`, `file_delete`, `glob`, `grep`, `str_replace` |
| Shell | `exec`, `process` |
| Web | `http_request`, `brave_search`, `web_scrape`, `web_crawl` |
| Database | `sqlite_query`, `postgres_query`, `mysql_query`, `redis_query` |
| Audio/Visual | `tts`, `transcribe`, `image_gen`, `video_gen` |
| Multi-agent | `delegate`, `communicate`, `agent_status`, `agent_wait` |
| MCP | `mcp_add`, `mcp_list`, `mcp_remove` |

## Usage

```rust
use flashmind_tools::ToolBuilder;

let (tools, sync) = ToolBuilder::new()
    .file_ops(None, &protected)
    .bash(secrets, &protected, forbidden, allowlist)
    .time()
    .http()
    .build_with_sync().await;
```

## Feature Flags

Enable integrations:

```toml
flashmind-tools = { version = "0.1", features = ["gmail", "github", "postgres"] }
```

Available: `mcp`, `gmail`, `google-calendar`, `google-contacts`, `outlook`, `caldav`, `github`, `slack`, `redis`, `postgres`, `mysql`, `clickhouse`, `docker`, `ssh`, `kubernetes`, `cloudflare`, `messaging`, `composio`

## License

MIT
