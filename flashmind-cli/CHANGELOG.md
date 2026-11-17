# Changelog

## 0.3.6 - 2026-11-17

### Added

- flsh can discover and call tools from stdio MCP servers. ACP clients can
  supply servers per session, and standalone sessions can use servers from
  the CLI configuration.

## 0.3.5 - 2026-11-17

### Fixed

- ACP clients receive delegated agent progress, tool calls, results and completion status.
- Delegated agent replies stay separate from the parent reply.
