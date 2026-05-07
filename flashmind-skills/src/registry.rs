//! Skill registry for discovering and managing skills from the filesystem.
//!
//! Walks configured search directories looking for subdirectories containing
//! `SKILL.md` files, parses them, and makes them available by name.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::parser::parse_skill_md;
use crate::skill::Skill;

/// Registry of discovered skills, keyed by name.
pub struct SkillRegistry {
    search_dirs: Vec<PathBuf>,
    skills: HashMap<String, Skill>,
}

impl SkillRegistry {
    /// Create a new registry that will search the given directories for skills.
    pub fn new(search_dirs: Vec<PathBuf>) -> Self {
        Self {
            search_dirs,
            skills: HashMap::new(),
        }
    }

    /// Walk each search directory and discover skills.
    ///
    /// Each immediate subdirectory containing a `SKILL.md` file is parsed as a
    /// skill. Deduplicates by name (first match wins). Returns the number of
    /// skills discovered.
    pub async fn discover(&mut self) -> anyhow::Result<usize> {
        self.skills.clear();
        let mut count = 0;

        for search_dir in &self.search_dirs {
            let mut entries = match tokio::fs::read_dir(search_dir).await {
                Ok(entries) => entries,
                Err(e) => {
                    tracing::warn!(dir = %search_dir.display(), error = %e, "failed to read skill search directory");
                    continue;
                }
            };

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let skill_md = path.join("SKILL.md");
                if !skill_md.exists() {
                    continue;
                }

                let content = match tokio::fs::read_to_string(&skill_md).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(path = %skill_md.display(), error = %e, "failed to read SKILL.md");
                        continue;
                    }
                };

                match parse_skill_md(&content, &path) {
                    Ok(skill) => {
                        let name = skill.meta.name.clone();
                        if self.skills.contains_key(&name) {
                            tracing::debug!(name = %name, "skipping duplicate skill");
                            continue;
                        }
                        tracing::debug!(name = %name, dir = %path.display(), "discovered skill");
                        self.skills.insert(name, skill);
                        count += 1;
                    }
                    Err(e) => {
                        tracing::warn!(path = %skill_md.display(), error = %e, "failed to parse SKILL.md");
                    }
                }
            }
        }

        tracing::info!(count, "skill discovery complete");
        Ok(count)
    }

    /// List all discovered skills.
    pub fn list(&self) -> Vec<&Skill> {
        let mut skills: Vec<&Skill> = self.skills.values().collect();
        skills.sort_by_key(|s| &s.meta.name);
        skills
    }

    /// Look up a skill by name.
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discover_from_temp_dirs() {
        let tmp = tempfile::tempdir().unwrap();

        // Create two skill directories
        let skill_a = tmp.path().join("alpha");
        std::fs::create_dir(&skill_a).unwrap();
        std::fs::write(
            skill_a.join("SKILL.md"),
            "---\nname: alpha\ndescription: First skill\n---\nAlpha body.",
        )
        .unwrap();

        let skill_b = tmp.path().join("beta");
        std::fs::create_dir(&skill_b).unwrap();
        std::fs::write(skill_b.join("SKILL.md"), "---\nname: beta\n---\nBeta body.").unwrap();

        // Directory without SKILL.md should be ignored
        let no_skill = tmp.path().join("ignored");
        std::fs::create_dir(&no_skill).unwrap();

        let mut registry = SkillRegistry::new(vec![tmp.path().to_path_buf()]);
        let count = registry.discover().await.unwrap();

        assert_eq!(count, 2);
        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("alpha").is_some());
        assert!(registry.get("beta").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn discover_deduplicates_by_name() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();

        let skill1 = tmp1.path().join("dupe");
        std::fs::create_dir(&skill1).unwrap();
        std::fs::write(
            skill1.join("SKILL.md"),
            "---\nname: dupe\ndescription: first\n---\nFirst.",
        )
        .unwrap();

        let skill2 = tmp2.path().join("dupe");
        std::fs::create_dir(&skill2).unwrap();
        std::fs::write(
            skill2.join("SKILL.md"),
            "---\nname: dupe\ndescription: second\n---\nSecond.",
        )
        .unwrap();

        let mut registry =
            SkillRegistry::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]);
        let count = registry.discover().await.unwrap();

        assert_eq!(count, 1);
        let skill = registry.get("dupe").unwrap();
        assert_eq!(skill.meta.description.as_deref(), Some("first"));
    }
}
