# flsh

Terminal chat and coding agent built with Flashmind.

```sh
cargo install flashmind-cli
flsh
```

Use `flsh --help` for commands and options. Load provider credentials through
the setup flow or environment variables, and keep them out of source control.

All available tools are enabled by default. Pass `--tools` with a
comma-separated list to choose a subset, or `--tools all` for the full set.

See the [flsh guide](GUIDE.md) for setup, commands, and examples.

`/undo` removes the last user message and its replies from the conversation.
`/retry` removes that exchange and sends the user message again, with optional
extra instructions. Both leave project files untouched.

Streamed text and reasoning target 400 characters per second. Set
`stream_chars_per_second` in `config.toml` to change the rate, or `0` for unlimited.
Completed commands keep their command and output tail visible until the next
tool, thinking block, or reply. `tool_display_interval_ms` sets the target minimum
display time (default 500 ms, `0` disables the delay).
Pacing is skipped when the display falls behind or a stream ends, so tools and
model requests never wait for display timers.

The `lsp` tool supports Rust, TypeScript/JavaScript, Python, Go, and C/C++ using
installed `rust-analyzer`, `typescript-language-server`, `pyright-langserver`,
`gopls`, and `clangd`. Writes and edits include diagnostics when a server is
available. Use `lsp` with `operation = "diagnostics"` to check a file explicitly.
Override a server in `config.toml`, or disable it:

```toml
[lsp.typescript]
command = ["typescript-language-server", "--stdio"]

[lsp.cpp]
disabled = true
```

Custom personas can be imported from Markdown files directly inside
`.claude/agents/` or `.agents/`, in your home directory or any project ancestor.
`CLAUDE_CONFIG_DIR` overrides the user Claude directory. For example,
`.agents/reviewer.md`:

```markdown
---
name: reviewer
description: Reviews code and explains concrete problems.
model: inherit
---
You are a careful code reviewer. Keep feedback brief and actionable.
```

Invoke `/reviewer` to switch the current session's persona, or `/reviewer review
this change` to switch and send a message. This imports the personality, not
Claude's subagent execution behavior. Project files override user files;
`.agents` wins within a scope; saved Flashmind JSON personas override imports.
Imported files are never changed by `/agents delete`.

`model` accepts `inherit`, a full Claude model ID, or `provider:model`.
Claude aliases such as `sonnet` keep the current model and display a notice.
Ask the agent to "spawn a reviewer to review this change" to use a persona in
an independent background worker. The `delegate` tool lists available personas
and accepts `persona: "reviewer"`; `name` remains the worker's optional label.
Workers use the persona's prompt and default model/reasoning. An explicit
`model` overrides the persona model; subagent models must use the current
provider. `persona` and `system_prompt` cannot be combined.

Other frontmatter settings, including tools, permissions, and hooks, are not
applied and are reported when switching personas or spawning workers.
