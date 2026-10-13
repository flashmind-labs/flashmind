# flashmind-cli

Binary crate producing `flsh`, a coding agent with a terminal UI.

## Checks

```bash
cargo fmt --all
cargo clippy -p flashmind-cli -- -D warnings
cargo test -p flashmind-cli
```

## Modes

```bash
cargo run -p flashmind-cli              # interactive chat
cargo run -p flashmind-cli -- -p "hi"   # one-shot response
cargo run -p flashmind-cli -- setup     # provider, model, reasoning
cargo run -p flashmind-cli -- resume    # restore a saved session
```

`/autocompact <tokens>` (or `auto_compact_tokens` in config) compacts earlier
than the model default; `/autocompact off` waits for an overflow.

`-m` selects the model; `--system` overrides the system prompt. Configuration
lives at `{config}/config.toml` (see File Locations), with provider credentials overridable by
environment variables. Shell commands require approval unless covered by
`command_allowlist`; `--dangerously-skip-permissions` skips shell approvals.

## File Locations

Resolved in `src/paths.rs`. `$FLASHMIND_HOME` puts everything in one directory;
otherwise an existing `~/.flashmind` (legacy) is used as-is; otherwise XDG on
Unix/macOS and `%APPDATA%`/`%LOCALAPPDATA%` on Windows:

| Kind | Contents | Default (Unix/macOS) |
| --- | --- | --- |
| config | `config.toml`, `agents/`, `skills/` | `$XDG_CONFIG_HOME/flashmind` or `~/.config/flashmind` |
| data | `sessions/` | `$XDG_DATA_HOME/flashmind` or `~/.local/share/flashmind` |
| state | `logs/`, `display-*.jsonl`, `history` | `$XDG_STATE_HOME/flashmind` or `~/.local/state/flashmind` |

## Tools

Files: `read`, `write`, `edit`, `find`, `image_read`.
Shell: `exec`, `process`.
Subagents: `delegate`, `agent_status`, `agent_wait`, `agent_terminate`.

`read` supports line ranges. `edit` requires one exact match. `find` accepts
filename globs or content regexes. `--tools` filters exposed names;
`--tools all` keeps the complete coding toolkit. `ToolSync` shares the final
registry with subagents.

Tool calls show one line each. Ctrl+O, `/tools expand` and
`/tools collapse` switch every tool in the transcript to its full
description and output. Each turn ends with tokens and cost, with and
without the cache.

## Skills and agents

Skills are filesystem instructions, listed with descriptions and source paths
in the system prompt. Agents read their source files using `read` and execute
scripts using `exec`. `/{skill}` activates a skill; `/skills` lists them.
Discovery uses configured `skill_dirs`, the default skills directory, project
skills, and global/project `.agents/skills`, `.claude/skills`, and
`.claude/commands` directories. Subagents receive the same skills index.

Custom personas persist via `AgentStore`. `/agents` lists them, `/agents-create`
creates one, `/agents delete` removes one, and `/agents reset` restores defaults.
`/{agent-name}` switches prompt, model, and reasoning. Agent names take precedence
over skill names in slash dispatch.

## Sessions

`session.rs` uses `flashmind-memory`'s `session` feature for SQLite persistence.
Sessions and display logs are grouped by working directory. Resume restores
conversation, display, cost, and usage. `flsh import pi` imports pi sessions.
