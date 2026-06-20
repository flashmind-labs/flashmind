# flashmind-cli TODO

Goal: keep `flashmind-cli` a simple, single-binary CLI agent — no daemon, no
listeners, no multi-user, no connect server. Focus on making the REPL/TUI
genuinely good (inspired by pi.dev's append-mode feel) and adding the handful
of features that fit a CLI scope.

Reference comparison: `../agent` (Flash) is the full runtime; we deliberately
do **not** port its daemon/listener/connect/multi-user/sandbox features.

---

## Batch 1 — TUI polish + commands (in progress)

- [x] Compact, pi-style banner (model, version, key hints — drop verbose tool list)
- [x] `/status` slash command — model, thinking, context %, cost, session key, git branch
- [x] `/context` slash command — token breakdown (entries by kind + last tokens + window)
- [x] Update `/help` to list new commands
- [x] Status bar: populate `context` from real prompt tokens + show git branch
- [x] `flashmind completions <shell>` subcommand (`clap_complete`)
- [x] `flashmind logs [date] [--follow] [--session]` subcommand
- [x] `flashmind clean --days N` subcommand (prune logs/sessions/display-logs)

Notes:

- `/stop` is redundant with `Esc`/`Ctrl+C` mid-stream (already handled in
  `stream_events`); not added as a separate slash command.

## Batch 2 — TUI deeper polish

- [x] `@mention` file completion in input (cwd-scoped fuzzy picker; attach file as context part)
      — `CwdMentionProvider` walks cwd (ignores .git/target/node_modules/...), substring
        match with basename preference, Tab/Esc/Enter handling in `Repl`; `extract_mentions`
        parses `@path` tokens, `build_context_block` attaches contents as a developer context
        entry before the user message (interactive + one-shot via `AgentInput::User.context`)
- [x] `!` shell escape (lines starting with `!` run as bash, output to scrollback)
      — `run_shell_escape` uses `$SHELL -c`, streams stdout line-by-line, drains stderr
        in red, reports non-zero exit codes; `/help` documents it
- [ ] Mouse-wheel scroll of scrollback (without disturbing input area) — **deferred**: terminal emulator handles this in append mode
- [ ] Click-drag text selection with auto-copy to clipboard — **deferred**: same reason
- [ ] Reasoning collapse/expand block (dimmed `▸ thinking…`, expand on key/click)
- [ ] Live tool progress line (persistent spinner + elapsed + tool name)
- [x] Tool result blocks: green background on success for diff-producing tools
      (`str_replace`, `file_write` via `SILENT_TOOLS`) — whole `FileDiff` block on
      green bg (`S_DIFF_BLOCK_OK`), black fg, +/- prefix distinguishes add/remove
- [x] Diff rendering polish (path header + gutter colors; `FileDiff` already truncated at 100)

## Batch 3 — Skills

- [ ] Wire `flashmind-skills` via `ToolBuilder`: `skill_list`, `skill_load`, `skill_run`, `skill_install`
- [ ] Skills dir: `~/.flashmind/skills/`
- [ ] Add `SKILL_INSTRUCTIONS` prompt fragment (already referenced in main.rs)
- [ ] `/skills` slash command to list installed skills

## Batch 4 — Memory capture (highest-impact single feature)

- [ ] Auto-capture: post-turn background task extracts facts
      (LLM-based + keyword heuristic for "prefer/always/never"), stores with
      tags + optional TTL
- [ ] RAG injection: pre-turn multi-query expansion → inject relevant memories
      as developer messages
- [ ] Dedup at 92% cosine threshold
- [ ] Add capture prompt constant to `flashmind-prompts`

---

## Deferred (not in scope now)

- [ ] **Profiles** — sampling-param TOML presets (`~/.flashmind/profiles/*.toml`),
      `--profile` flag, `switch_profile` tool
- [ ] **Roles + SOUL.md / REMINDER.md** — persona markdown files replacing
      system prompt, `--role` flag, `switch_role`/`role_list`/`role_load` tools,
      live `prompt_watcher`
- [ ] **Canvas** — self-contained web UI builder (subagent + SSE file watcher +
      HTTP server); large, revisit later
- [ ] **Cron exposure** — `flashmind-cron` lib exists but not wired into CLI;
      `cron` subcommand + `cron_create`/`schedule_once` tools + `CronRunner`
- [ ] **Browser tool** — declined (heavy dep; out of CLI scope)
- [ ] **User-profile curation agent** — periodic consolidation of conversation
      patterns into profile memories (depends on memory capture)

---

## Completed

- [x] MCP support (`mcp add/remove/list` + `mcp_*` tools + ToolSync + persistence)
- [x] Memory store wiring (manual `memory_store`/`recall`/`forget` tools)
- [x] Session persistence + resume
- [x] Interactive model picker (`/model`)
- [x] Slash commands: `/model /thinking /new /clear /compact /undo /retry
      /sessions /rename /system /export /fork /mcp /memory /help`
