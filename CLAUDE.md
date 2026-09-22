# Flashmind

Rust workspace for AI agents. The public facade is `flashmind`; implementation
crates live beside it.

## Commands

Run checks only for crates you change:

```bash
cargo fmt --all
cargo clippy -p <crate> -- -D warnings
cargo test -p <crate>
```

## Layout

- `flashmind-types`: shared types and traits
- `flashmind-core`: agent runtime and conversations
- `flashmind-llm`: provider implementations
- `flashmind-tools`: built-in tools and integrations
- `flashmind-memory`: vector memory
- `flashmind`: public facade

Other crates provide prompts, scheduling, skills, terminal UI, and local
network helpers.

## Conventions

- Keep provider-specific code in `flashmind-llm` and tools in `flashmind-tools`.
- Add feature-gated integrations behind their existing feature flags.
- Return errors from library code instead of panicking.
- Keep credentials out of source and committed configuration.
