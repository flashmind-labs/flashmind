//! SKILL.md frontmatter and body parser.
//!
//! Parses skill definition files that use YAML frontmatter delimited by `---`
//! fences. If no frontmatter is present, the skill name is derived from the
//! directory name.

use std::path::Path;

use crate::error::SkillError;
use crate::skill::{Skill, SkillMeta};

/// Parse a SKILL.md file's content into a [`Skill`].
///
/// Expects optional YAML frontmatter between `---` fences at the top of the
/// file. Everything after the closing `---` becomes the markdown body. If no
/// frontmatter is present, the skill name is derived from the parent directory.
pub fn parse_skill_md(content: &str, dir: &Path) -> Result<Skill, SkillError> {
    let trimmed = content.trim_start();

    if let Some(after_open) = trimmed.strip_prefix("---") {
        let close_pos = after_open.find("\n---").ok_or_else(|| {
            SkillError::InvalidFormat("missing closing --- for frontmatter".into())
        })?;

        let frontmatter = &after_open[..close_pos];
        let body_start = close_pos + 4; // skip \n---
        let body = after_open[body_start..]
            .trim_start_matches('\n')
            .to_string();

        let meta: SkillMeta = serde_yaml::from_str(frontmatter)
            .map_err(|e| SkillError::InvalidFormat(format!("YAML parse error: {e}")))?;

        Ok(Skill {
            meta,
            body,
            dir: dir.to_path_buf(),
        })
    } else {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Skill {
            meta: SkillMeta {
                name,
                description: None,
                usage: None,
            },
            body: content.to_string(),
            dir: dir.to_path_buf(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parse_with_frontmatter() {
        let content = r#"---
name: deploy
description: Deploy the application
usage: skill_run deploy "make deploy"
---
# Deploy Skill

Run deployment commands.
"#;
        let dir = PathBuf::from("/skills/deploy");
        let skill = parse_skill_md(content, &dir).unwrap();

        assert_eq!(skill.meta.name, "deploy");
        assert_eq!(
            skill.meta.description.as_deref(),
            Some("Deploy the application")
        );
        assert_eq!(
            skill.meta.usage.as_deref(),
            Some(r#"skill_run deploy "make deploy""#)
        );
        assert!(skill.body.starts_with("# Deploy Skill"));
        assert_eq!(skill.dir, dir);
    }

    #[test]
    fn parse_without_frontmatter() {
        let content = "# Just some markdown\n\nNo frontmatter here.";
        let dir = PathBuf::from("/skills/my-tool");
        let skill = parse_skill_md(content, &dir).unwrap();

        assert_eq!(skill.meta.name, "my-tool");
        assert!(skill.meta.description.is_none());
        assert_eq!(skill.body, content);
    }

    #[test]
    fn parse_empty_content() {
        let content = "";
        let dir = PathBuf::from("/skills/empty");
        let skill = parse_skill_md(content, &dir).unwrap();

        assert_eq!(skill.meta.name, "empty");
        assert_eq!(skill.body, "");
    }

    #[test]
    fn parse_missing_closing_fence() {
        let content = "---\nname: broken\n";
        let dir = PathBuf::from("/skills/broken");
        let err = parse_skill_md(content, &dir).unwrap_err();

        assert!(err.to_string().contains("missing closing ---"));
    }
}
