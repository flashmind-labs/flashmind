# flashmind-skills

Skill discovery, loading, and execution for [Flashmind](https://github.com/flashmind-labs/flashmind).

Skills are self-contained directories with a `SKILL.md` definition file that extend agent capabilities.

## Usage

```rust
use flashmind_skills::{DiskSkillProvider, SkillProvider, SkillRunner};

let provider = DiskSkillProvider::discover(vec![skills_dir]).await?;

for skill in provider.list() {
    println!("{} — {}", skill.meta.name, skill.meta.description);
}

let output = SkillRunner::run(&skill, "make deploy").await?;
```

## SKILL.md Format

```markdown
---
name: deploy
description: Deploy the application
---

Instructions for the agent...
```

## Features

- Auto-discovery from directories
- `.env` loading with secret redaction in output
- Timeout enforcement
- Install/uninstall lifecycle

## License

MIT
