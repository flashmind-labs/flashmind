# flashmind-cli

The `flsh` binary: a coding agent with a terminal UI.

```bash
cargo run -p flashmind-cli              # interactive chat
cargo run -p flashmind-cli -- -p "hi"   # one-shot response
```

- Resolve config, data and state paths through `src/paths.rs`; never
  hard-code a home directory.
- Credentials come from config or environment variables.
- Shell commands run without approval prompts.
