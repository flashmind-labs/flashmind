//! Skill discovery, loading, and execution for Flashmind.
//!
//! Skills are self-contained directories with a `SKILL.md` definition file and
//! optional `.env` for secrets. This crate provides:
//!
//! - **Parsing** — extract YAML frontmatter and markdown body from `SKILL.md`
//! - **Discovery** — scan directories for skill definitions
//! - **Execution** — run commands in a skill's environment with secret redaction
//! - **Installation** — create new skill directories
//! - **Tools** — agent-callable tools for listing, loading, running, and installing skills

pub mod error;
pub mod installer;
pub mod parser;
pub mod registry;
pub mod runner;
pub mod skill;
pub mod tools;

pub use error::SkillError;
pub use installer::SkillInstaller;
pub use parser::parse_skill_md;
pub use registry::DiskSkillProvider;
pub use runner::{SkillOutput, SkillRunner};
pub use skill::{Skill, SkillMeta, SkillProvider};
pub use tools::{SkillInstallTool, SkillListTool, SkillLoadTool, SkillRunTool};
