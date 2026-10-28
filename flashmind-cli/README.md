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

`/undo` removes the last user message and its replies from the conversation.
`/retry` removes that exchange and sends the user message again, with optional
extra instructions. Both leave project files untouched.

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
