# flashmind-cli TODO

Goal: keep `flashmind-cli` a simple, single-binary CLI agent: no daemon, no
listeners, no multi-user, no connect server. Make the REPL/TUI genuinely good
and add only the features that fit a CLI scope.

`../agent` (Flash) is the full runtime. Its daemon, listener, connect,
multi-user and sandbox features are deliberately not ported.

## Next

- [ ] **Cron**: `flashmind-cron` exists but is not wired in. Add a `cron`
      subcommand, `cron_create`/`schedule_once` tools and a `CronRunner`.
- [ ] **Profiles**: sampling-parameter TOML presets in `{config}/profiles/`,
      a `--profile` flag and a `switch_profile` tool.
- [ ] **Roles**: persona markdown files (SOUL.md, REMINDER.md) that replace the
      system prompt, a `--role` flag and `switch_role`/`role_list`/`role_load`
      tools.

## Later

- [ ] **Canvas**: web UI builder (subagent, SSE file watcher, HTTP server).
      Large.

## Declined

- Browser tool: heavy dependency, out of CLI scope.
- Automatic memory capture and RAG injection: the manual `memory_store`,
  `memory_recall` and `memory_forget` tools are enough.
- User-profile curation agent: depended on automatic memory capture.
- `/stop`: `Esc` and `Ctrl+C` already stop a running turn.

## Done

- TUI: compact banner, status bar with context and git branch, activity line
  with tool name, elapsed time, tokens and cost, mouse scroll, selection with
  clipboard copy, collapsible reasoning, diff blocks, collapsed tool output.
- Input: `@` file mentions, `!` shell escape, steering a running turn.
- Commands: `/status /context /model /thinking /reasoning /new /clear
  /compact /autocompact /undo /retry /resume /rename /system /export /fork
  /skills /tools /agents /help`.
- Subcommands: `completions`, `logs`, `clean`, `mcp add/remove/list`.
- Skills: discovery from config, cwd and `.claude`/`.agents` dirs, the skill
  index in the system prompt, and `always_skills` for skills loaded in full.
- MCP tools, manual memory tools, session persistence and resume,
  AGENTS.md/CLAUDE.md loading, `--json` one-shot output.
