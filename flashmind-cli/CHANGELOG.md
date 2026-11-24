# Changelog

## 0.3.19 - 2026-11-24

### Changed

- Raise default stream timeouts to 600s before output starts and 180s once
  output starts, and to 1200s and 600s for Gemini models. Reasoning chunks keep
  resetting the startup window until answer or tool output begins.

### Fixed

- Compaction resets its 600s (1200s for Gemini) timeout whenever reasoning or
  summary chunks arrive instead of failing after a fixed wall-clock limit, and
  no longer suppresses reasoning deltas on OpenRouter when reasoning is off.
- Apply stream liveness timeouts without a 120s total request cap across
  Anthropic, OpenAI, OpenAI Responses, Mistral, Ollama, and OpenRouter native
  endpoints.

## 0.3.18 - 2026-11-23

### Fixed

- ACP and CLI sessions save accepted input, completed model responses and tool
  results as they arrive. Restarting during a later request preserves that work.
- Session save errors are reported, and a second writer cannot overwrite an
  active session.
- Session usage records attribute billed costs to the model that handled each
  request.

## 0.3.17 - 2026-11-22

### Added

- ACP clients can start, load or resume a session with
  `_meta.flsh.subagents: false` to leave out flsh's subagent tools.

## 0.3.16 - 2026-11-22

### Added

- Override OpenRouter routing per model in
  `[settings."openrouter:example/model".routing]`. Omitted fields inherit the
  global settings; empty lists clear inherited filters.

### Changed

- Highlighted user input no longer includes a `You` label.

## 0.3.15 - 2026-11-21

### Changed

- The terminal keeps commands on one row and opens failed calls automatically.
  Edits show line counts. Press Ctrl+O for tool output.
- The header shows the session title and workspace. Queued follow-ups stay
  beside the input. Press Up to edit the newest one, or F1 for shortcuts.
- The animated pet is now opt-in.

### Fixed

- Starting or finishing a command no longer moves the transcript to make room
  for an activity indicator. Streaming updates share a redraw.

## 0.3.14 - 2026-11-21

### Added

- Each session file records what every turn spent, with its time and model,
  in its header's `usage_log`, so tools can tell when a resumed session's cost
  was spent.

## 0.3.13 - 2026-11-21

### Changed

- `flsh models` lists models newest first, as the interactive model picker
  does, instead of by id.

## 0.3.12 - 2026-11-20

### Added

- Sessions get a short title written by a model from the first prompt, in
  interactive, one-shot and ACP sessions. ACP clients receive it as a
  `session_info_update`.
- `--title-model <MODEL>` and `title_model` in the config choose the model that
  writes titles, or `off`. Without either, the session's model writes them.

### Fixed

- Resuming a session no longer clears its saved title on the first turn.

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
