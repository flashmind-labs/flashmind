# flashmind-tools

50+ built-in tool implementations plus the composable `ToolBuilder` for registering tools with an agent.

## Tool Categories

| Category | Files |
|----------|-------|
| File & filesystem | `file_ops.rs`, `glob.rs`, `grep.rs`, `text_replace.rs`, `text_replace_regex.rs` |
| Shell & process | `bash.rs`, `process.rs` |
| HTTP & web | `http.rs`, `brave.rs`, `firecrawl.rs`, `search_read.rs` |
| Database | `sqlite.rs`, `postgres.rs`, `mysql.rs`, `clickhouse.rs`, `redis_tools.rs`, `db_common.rs` |
| Audio | `audio.rs` |
| Image & video | `image_gen.rs`, `image_edit.rs`, `image_read.rs`, `video_gen.rs` |
| Model discovery | `list_models.rs` |
| MCP | `mcp/` (conditional on `mcp` feature) |
| Google APIs | `google/` (Gmail, Calendar, Contacts) — features `gmail`, `google-calendar`, `google-contacts` |
| Outlook | `outlook/` (Mail, Calendar, Contacts) — feature `outlook` |
| GitHub | `github/` (repos, issues, PRs, notifications) — feature `github` |
| Slack | `slack/` (channels, messages, search, reactions) — feature `slack` |
| CalDAV | `caldav/` (any CalDAV server — Fastmail, Nextcloud, iCloud) — feature `caldav` |
| Docker | `docker/` (containers, images, exec) — feature `docker` |
| Cloudflare | `cloudflare/` (zones, DNS, workers, cache) — feature `cloudflare` |
| Messaging | `messaging/` (Twilio SMS/WhatsApp, SMTP email) — feature `messaging` |
| SSH | `ssh/` (exec, upload, download) — feature `ssh` |
| Kubernetes | `kubernetes/` (pods, deployments, services, events, apply, exec) — feature `kubernetes` |
| Composio | `composio/` (250+ apps via composio.dev unified API) — feature `composio` |
| Subagent control | `subagent.rs` — `delegate`, `communicate`, `agent_status`, `agent_wait`, `agent_terminate` |
| Utility | `time.rs`, `json_query.rs`, `str_diff.rs` |

## Adding a Tool

1. Create `my_tool.rs` in this crate
2. Implement the `Tool` trait: `name()`, `description()`, `parameters()` (JSON Schema), `execute()`, `humanize()`
3. Export from `lib.rs`
4. Register in `builder.rs` in the appropriate `.with_*()` method

## Shared Infrastructure

- `utils.rs` — common helpers
- `db_common.rs` — shared DB helpers: `is_write_query()`, `format_table()`, `clamp_limit()` (used by sqlite, postgres, mysql, clickhouse)
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

## Config-Based Integrations

Simpler integrations that take credentials/config directly (no OAuth flow):

- **Postgres** (`postgres.rs`): `PostgresConfig { connection_string }`. Read-only enforced via keyword blocklist + `SET default_transaction_read_only = on`.
- **MySQL** (`mysql.rs`): `MysqlConfig { connection_string }`. Read-only via keyword blocklist + `SET SESSION TRANSACTION READ ONLY`.
- **ClickHouse** (`clickhouse.rs`): `ClickHouseConfig { url, user, password, database }`. HTTP API, no new deps. Read-only via keyword blocklist + `&readonly=1`.
- **Redis** (`redis_tools.rs`): `RedisConfig { url }`. Read-only via safe command allowlist (34 commands).
- **Docker** (`docker/`): `DockerConfig { endpoint }`. 9 tools via bollard crate. Read-only skips exec/create/stop/remove/pull.
- **Cloudflare** (`cloudflare/`): `CloudflareConfig { token_path }`. API token auth validated via `/user/tokens/verify`, saved as `CachedToken`. 8 tools (zones, DNS, workers, cache).
- **Messaging** (`messaging/`): `MessagingConfig { twilio, smtp }`. Twilio: 4 tools (SMS, WhatsApp, list/get messages). SMTP: send email via lettre. Read-only skips all write tools + SMTP entirely.
- **SSH** (`ssh/`): `SshConfig { profiles: Vec<SshProfile> }`. Auth via key, password, or agent. 3 tools (exec, upload, download). Read-only skips upload.
- **Kubernetes** (`kubernetes/`): `KubernetesConfig { kubeconfig_path, context, namespace }`. 11 tools via kube crate. Read-only skips apply/delete/scale/exec.

### Builder API

Every integration exposes a full and read-only variant:

```rust
// OAuth integrations
.github(config)            // all tools
.github_readonly(config)   // read tools only
.slack(config) / .slack_readonly(config)
.caldav(config) / .caldav_readonly(config)
.outlook(config) / .outlook_readonly(config)
.google(&config) / .google_readonly(&config)

// Config-based integrations
.postgres(config) / .postgres_readonly(config)
.mysql(config) / .mysql_readonly(config)
.clickhouse(config) / .clickhouse_readonly(config)
.redis(config) / .redis_readonly(config)
.docker(config) / .docker_readonly(config)
.cloudflare(config) / .cloudflare_readonly(config)
.messaging(config) / .messaging_readonly(config)
.ssh(config) / .ssh_readonly(config)
.kubernetes(config) / .kubernetes_readonly(config)

// Composio (250+ apps via unified API)
.composio(config)
```

For OAuth integrations, if a valid cached token exists on disk, service tools are registered immediately; otherwise only the `{provider}_auth` tool is registered, which adds service tools dynamically via `PendingTools` after a successful code exchange. Cloudflare follows the same pattern with API token validation instead of OAuth.

## Composio Integration

Composio (`composio/`) wraps 250+ app integrations from composio.dev as local `Tool` implementations via a unified REST API. Follows the MCP wrapper pattern but simpler (pure REST, no persistent connection).

- `composio/mod.rs` — `ComposioConfig { api_key, connected_account_id, toolkits, base_url }`
- `composio/client.rs` — `ComposioClient` with `list_tools()` and `execute_tool()` methods
- `composio/types.rs` — `ComposioToolDef`, `ToolsListResponse`, `ExecuteResponse`
- `composio/wrapper.rs` — `ComposioToolWrapper` (impl `Tool`), `make_composio_tool_wrappers()` factory

Tools are fetched eagerly in `build_with_sync()` and registered with lowercased slug names (e.g. `GITHUB_CREATE_ISSUE` → `github_create_issue`). The original-case slug is preserved internally for API calls.

## Testing

Each tool file has `#[cfg(test)] mod tests`. Use `execute_tool()` helper from `lib.rs` for integration tests.
