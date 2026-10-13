# Flashmind

Rust workspace for AI agents. The public facade is `flashmind`; implementation
crates live beside it.

## Commands

Run checks only for crates you change:

```bash
cargo fmt --all
cargo clippy -p <crate> -- -D warnings
cargo test -p <crate>
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

## Layout

- `flashmind-types`: shared types and traits
- `flashmind-core`: agent runtime and conversations
- `flashmind-llm`: provider implementations
- `flashmind-tools`: built-in tools and integrations
- `flashmind-memory`: vector memory
- `flashmind-tui`: terminal UI
- `flashmind-cli`: the `flsh` binary
- `flashmind`: public facade

Other crates provide prompts, scheduling, skills, and local network helpers.

## Conventions

- Keep provider-specific code in `flashmind-llm` and tools in `flashmind-tools`.
- Add feature-gated integrations behind their existing feature flags.
- Return errors from library code instead of panicking.
- Keep credentials out of source and committed configuration.
- CLAUDE.md files hold lasting rules only. Do not list files, types or
  current behavior there; the code is the reference.
