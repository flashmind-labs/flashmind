# flashmind-skills

Skill discovery, loading, and execution. Skills are self-contained directories with a `SKILL.md` definition file and optional `.env` for secrets.

Depends only on `flashmind-types`.

## Key Types

- `SkillMeta` / `Skill` (`skill.rs`) — parsed frontmatter (name, description, usage) + markdown body + directory path
- `SkillProvider` trait (`skill.rs`) — read-only access to skills: `list()`, `get(name)`. Decouples tools and prompt builder from storage backend.
- `DiskSkillProvider` (`registry.rs`) — filesystem-backed implementation. `discover(dirs)` scans for `SKILL.md`, `refresh()` re-scans. Deduplicates by name (first wins).
- `SkillRunner` (`runner.rs`) — executes commands in a skill's directory. Loads `.env`, prepends skill dir to `PATH`, applies timeout, redacts secrets from output.
- `SkillInstaller` (`installer.rs`) — creates skill directories with optional `.env` files.
- `SkillError` (`error.rs`) — `NotFound`, `InvalidFormat`, `ExecutionFailed`, `Timeout`, `Io`

## SKILL.md Format

```markdown
---
name: deploy
description: Deploy the application
usage: skill_run deploy "make deploy"
---
# Deploy Skill

Markdown body with full instructions.
```

If no frontmatter, skill name is derived from the directory name.

## Agent Tools

Four tools registered by `flashmind-app` in `tools.rs`:

| Tool | Purpose |
|------|---------|
| `skill_list` | List all skills (name + description) |
| `skill_load` | Load a skill's full markdown body |
| `skill_run` | Execute a command in a skill's directory with `.env` and secret redaction |
| `skill_install` | Create a new skill directory, refresh provider |

All tools use `Arc<RwLock<dyn SkillProvider>>` for thread-safe access.

## System Prompt Integration

`flashmind-app/src/prompt.rs` calls `build_skills_section()` which lists skill names and descriptions in the system prompt. The agent uses `skill_load` to read full instructions on demand.

## Testing

```bash
cargo test -p flashmind-skills
```
