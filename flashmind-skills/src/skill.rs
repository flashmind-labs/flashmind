//! Core skill types and the [`SkillProvider`] trait.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Metadata parsed from the YAML frontmatter of a `SKILL.md` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMeta {
    /// Unique identifier used to reference this skill in tools and the prompt.
    pub name: String,
    /// One-line human-readable summary shown in the system prompt.
    #[serde(default)]
    pub description: Option<String>,
    /// Example invocation string (e.g. `skill_run deploy "make deploy"`).
    #[serde(default)]
    pub usage: Option<String>,
}

/// A discovered skill — a self-contained directory with a `SKILL.md` definition
/// file and optional `.env` for secrets.
///
/// The [`meta`](Self::meta) fields come from the YAML frontmatter while
/// [`body`](Self::body) holds the markdown content after the frontmatter.
/// [`dir`](Self::dir) is the filesystem directory the skill was loaded from;
/// commands executed via the skill runner run inside this directory.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Parsed frontmatter metadata.
    pub meta: SkillMeta,
    /// Markdown body (everything after the YAML frontmatter).
    pub body: String,
    /// Directory the skill was discovered in.
    pub dir: PathBuf,
}

/// Read-only access to a collection of skills.
///
/// Implemented by [`DiskSkillProvider`](crate::DiskSkillProvider) for
/// filesystem-backed discovery. Tools and the system prompt builder depend
/// on this trait so they stay decoupled from the storage backend.
pub trait SkillProvider: Send + Sync {
    /// List all skills, sorted by name.
    fn list(&self) -> Vec<&Skill>;

    /// Look up a skill by name.
    fn get(&self, name: &str) -> Option<&Skill>;

    /// Format a name + description index of all skills for injection into the
    /// system prompt. Returns an empty string when no skills are discovered.
    fn skill_index(&self) -> String {
        let skills = self.list();
        if skills.is_empty() {
            return String::new();
        }
        let mut buf = String::from("\n\nAvailable skills:\n");
        for s in &skills {
            let desc = s.meta.description.as_deref().unwrap_or("no description");
            buf.push_str(&format!("- **{}** — {}\n", s.meta.name, desc));
        }
        buf.push_str("\nUse `skill_load` to read a skill's full instructions before using it.");
        buf
    }
}
