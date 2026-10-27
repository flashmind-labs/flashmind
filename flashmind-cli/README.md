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

See the [Flashmind guide](https://github.com/flashmind-labs/flashmind) for
providers, tools, and examples.

`/undo` restores the previous turn's conversation and project files; `/redo`
restores the undone turn. `/retry` undoes the turn before rerunning it. History
survives resume and forks. Files changed since a turn cause undo/redo to stop
without overwriting them. Snapshots cover tracked and nonignored project files,
including shell edits, but exclude ignored files and paths outside the project.
Only turns recorded with this version have file snapshots.

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
