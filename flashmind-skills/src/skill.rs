//! Core skill types representing a discovered skill and its metadata.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Metadata parsed from SKILL.md YAML frontmatter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMeta {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub usage: Option<String>,
}

/// A discovered skill with parsed metadata and content.
#[derive(Debug, Clone)]
pub struct Skill {
    pub meta: SkillMeta,
    pub body: String,
    pub dir: PathBuf,
}
