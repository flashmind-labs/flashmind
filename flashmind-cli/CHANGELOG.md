# Changelog

## 0.3.7 - 2026-11-18

### Added

- `flsh mcp add`, `flsh mcp list` and `flsh mcp remove` manage MCP servers
  in `config.toml`.
- flsh connects to remote MCP servers over Streamable HTTP. `flsh mcp add
  <name> --url <url>` adds one and `flsh mcp login <name>` signs in with OAuth.

## 0.3.6 - 2026-11-17

### Added

- flsh can discover and call tools from stdio MCP servers. ACP clients can
  supply servers per session, and standalone sessions can use servers from
  the CLI configuration.

## 0.3.5 - 2026-11-17

### Fixed

- ACP clients receive delegated agent progress, tool calls, results and completion status.
- Delegated agent replies stay separate from the parent reply.
