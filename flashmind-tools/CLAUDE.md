# flashmind-tools

30+ built-in tool implementations plus the composable `ToolBuilder` for registering tools with an agent.

## Tool Categories

| Category | Files |
|----------|-------|
| File & filesystem | `file_ops.rs`, `glob.rs`, `grep.rs`, `text_replace.rs`, `text_replace_regex.rs` |
| Shell & process | `bash.rs`, `process.rs` |
| HTTP & web | `http.rs`, `brave.rs`, `firecrawl.rs`, `search_read.rs` |
| Database | `sqlite.rs` |
| Audio | `audio.rs` |
| Image & video | `image_gen.rs`, `image_edit.rs`, `image_read.rs`, `video_gen.rs` |
| Model discovery | `list_models.rs` |
| MCP | `mcp/` (conditional on `mcp` feature) |
| Google APIs | `google/` (Gmail, Calendar, Contacts) — features `gmail`, `google-calendar`, `google-contacts` |
| Outlook | `outlook/` (Mail, Calendar, Contacts) — feature `outlook` |
| GitHub | `github/` (repos, issues, PRs, notifications) — feature `github` |
| Slack | `slack/` (channels, messages, search, reactions) — feature `slack` |
| CalDAV | `caldav/` (any CalDAV server — Fastmail, Nextcloud, iCloud) — feature `caldav` |
| Subagent control | `subagent.rs` — `delegate`, `communicate`, `agent_status`, `agent_wait`, `agent_terminate` |
| Utility | `time.rs`, `json_query.rs`, `str_diff.rs` |

## Adding a Tool

1. Create `my_tool.rs` in this crate
2. Implement the `Tool` trait: `name()`, `description()`, `parameters()` (JSON Schema), `execute()`, `humanize()`
3. Export from `lib.rs`
4. Register in `builder.rs` in the appropriate `.with_*()` method

## Shared Infrastructure

- `utils.rs` — common helpers
- `file_cache.rs` — file content caching for tools that read files
- `protected.rs` — path protection / forbidden command checking
- `search_cache.rs` — search result deduplication
- `builder.rs` — `ToolBuilder` that composes tool sets with gating (read-only mode, path exemptions)

## OAuth Integrations

Google, Outlook, GitHub, Slack, and CalDAV follow a shared pattern:

- `{provider}/mod.rs` — `{Provider}Config` + `{Provider}Client` with token caching, 401 retry, and per-method HTTP helpers
- `{provider}/auth.rs` — credentials struct + `auth_url()`, `exchange_code()`, optional `refresh_token()`
- `{provider}/auth_tool.rs` — two-phase OAuth tool: no `code` returns interrupt with URL; `code` exchanges + saves token + registers service tools via `PendingTools`
- `{provider}/tools.rs` — individual `Tool` impls each holding `Arc<{Provider}Client>`
- `{provider}/types.rs` — API response types

CalDAV additionally has `caldav/xml.rs` for WebDAV XML builders/parsers and supports both `CalDavAuth::Basic` and `CalDavAuth::OAuth`.

Shared: `oauth.rs` (`CachedToken`, `load_token`, `save_token`, `TokenResponse`) — gated on any integration feature.

### Builder API

Every integration exposes a full and read-only variant:

```rust
.github(config)            // all tools
.github_readonly(config)   // read tools only
.slack(config) / .slack_readonly(config)
.caldav(config) / .caldav_readonly(config)
.outlook(config) / .outlook_readonly(config)
.google(&config) / .google_readonly(&config)
```

If a valid cached token exists on disk, service tools are registered immediately; otherwise only the `{provider}_auth` tool is registered, which adds service tools dynamically via `PendingTools` after a successful code exchange.

## Testing

Each tool file has `#[cfg(test)] mod tests`. Use `execute_tool()` helper from `lib.rs` for integration tests.
