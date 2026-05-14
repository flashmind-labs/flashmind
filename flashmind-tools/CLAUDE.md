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

## Testing

Each tool file has `#[cfg(test)] mod tests`. Use `execute_tool()` helper from `lib.rs` for integration tests.
