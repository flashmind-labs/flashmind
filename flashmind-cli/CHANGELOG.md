# Changelog

## 0.3.11 - 2026-11-20

### Added

- `--log-all-tools <FILE>` appends every tool call to a file as a JSON line
  before it runs.
- Musl builds for Alpine and other musl Linux systems. The install script and
  `flsh update` pick them automatically.

## 0.3.10 - 2026-11-19

### Added

- `VENICE_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY` and
  `DEEPSEEK_API_KEY` select those providers without a config entry.
- A Docker image at `ghcr.io/flashmind-labs/flsh`.

## 0.3.9 - 2026-11-18

### Changed

- Updated dependencies, including the Agent Client Protocol SDK 5.4, rmcp 3.11
  and jsonwebtoken 12.

## 0.3.8 - 2026-11-18

### Changed

- Provider API key environment variables, such as `OPENROUTER_API_KEY`, now
  take precedence over keys in `config.toml`. They are never written to it.

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
