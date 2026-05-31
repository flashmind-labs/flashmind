# flashmind

Facade crate for the [Flashmind](https://github.com/flashmind-labs/flashmind) AI agent framework.

Re-exports all workspace crates under a single namespace for convenience.

## Usage

```toml
[dependencies]
flashmind = "0.1"
```

```rust
use flashmind::{core::Agent, llm::create_provider, types::*};
```

## Namespace Map

| Module | Crate |
|--------|-------|
| `flashmind::types` | flashmind-types |
| `flashmind::core` | flashmind-core |
| `flashmind::llm` | flashmind-llm |
| `flashmind::tools` | flashmind-tools |
| `flashmind::memory` | flashmind-memory |
| `flashmind::prompts` | flashmind-prompts |
| `flashmind::cron` | flashmind-cron |
| `flashmind::skills` | flashmind-skills |
| `flashmind::tailscale` | flashmind-tailscale |
| `flashmind::tui` | flashmind-tui (feature: `tui`) |

## Feature Flags

- `full` — enable everything
- `tui` — terminal UI
- `mcp`, `gmail`, `github`, `slack`, `outlook`, etc. — individual integrations

See the [main README](https://github.com/flashmind-labs/flashmind) for the full getting-started guide.

## License

MPL-2.0
