//! Filesystem-backed skill provider.
//!
//! [`DiskSkillProvider`] scans directories for `SKILL.md` files and makes
//! discovered skills available via the [`SkillProvider`] trait.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::parser::parse_skill_md;
use crate::skill::{Skill, SkillProvider};

/// Skill provider that discovers skills from filesystem directories.
pub struct DiskSkillProvider {
    search_dirs: Vec<PathBuf>,
    skills: HashMap<String, Skill>,
}

impl DiskSkillProvider {
    /// Scan `search_dirs` for skill directories and return a populated provider.
    ///
    /// Each immediate subdirectory containing a `SKILL.md` is parsed as a skill.
    /// Deduplicates by name (first directory wins).
    pub async fn discover(search_dirs: Vec<PathBuf>) -> anyhow::Result<Self> {
        let mut provider = Self {
            search_dirs,
            skills: HashMap::new(),
        };
        provider.refresh().await?;
        Ok(provider)
    }

    /// Re-scan all search directories, replacing the current skill set.
    pub async fn refresh(&mut self) -> anyhow::Result<()> {
        self.skills.clear();
        let mut count: usize = 0;

        for dir in self.search_dirs.clone() {
            scan_dir(&dir, &mut self.skills, &mut count).await?;
        }

        tracing::info!(count, "skill discovery complete");
        Ok(())
    }

    /// The directories this provider scans.
    pub fn search_dirs(&self) -> &[PathBuf] {
        &self.search_dirs
    }
}

impl SkillProvider for DiskSkillProvider {
    fn list(&self) -> Vec<&Skill> {
        let mut skills: Vec<&Skill> = self.skills.values().collect();
        skills.sort_by_key(|s| &s.meta.name);
        skills
    }

    fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

async fn scan_dir(
    dir: &Path,
    skills: &mut HashMap<String, Skill>,
    count: &mut usize,
) -> anyhow::Result<()> {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "failed to read skill search directory");
            return Ok(());
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
                if skills.contains_key(&name) {
                    tracing::debug!(name = %name, "skipping duplicate skill");
                    continue;
                }
                tracing::debug!(name = %name, dir = %path.display(), "discovered skill");
                skills.insert(name, skill);
                *count += 1;
            }
            Err(e) => {
                tracing::warn!(path = %skill_md.display(), error = %e, "failed to parse SKILL.md");
            }
        }
    }

    // Second pass: flat .md files (Claude Code commands compatibility).
    let mut flat_entries = match tokio::fs::read_dir(dir).await {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    while let Some(entry) = flat_entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() || path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read skill file");
                continue;
            }
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        if skills.contains_key(&name) {
            continue;
        }
        let skill_dir = path.parent().unwrap_or(dir);
        match parse_skill_md(&content, skill_dir) {
            Ok(mut skill) => {
                if skill.meta.name != name {
                    skill.meta.name = name.clone();
                }
                tracing::debug!(name = %name, path = %path.display(), "discovered flat skill file");
                skills.insert(name, skill);
                *count += 1;
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to parse skill file");
            }
        }
    }

    Ok(())
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

        let no_skill = tmp.path().join("ignored");
        std::fs::create_dir(&no_skill).unwrap();

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 2);
        assert!(provider.get("alpha").is_some());
        assert!(provider.get("beta").is_some());
        assert!(provider.get("nonexistent").is_none());
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

        let provider =
            DiskSkillProvider::discover(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()])
                .await
                .unwrap();

        assert_eq!(provider.list().len(), 1);
        let skill = provider.get("dupe").unwrap();
        assert_eq!(skill.meta.description.as_deref(), Some("first"));
    }

    #[tokio::test]
    async fn discover_flat_md_files() {
        let tmp = tempfile::tempdir().unwrap();

        // Flat .md file (Claude Code commands style)
        std::fs::write(
            tmp.path().join("deploy.md"),
            "---\nname: deploy\ndescription: Deploy the app\n---\nRun deploy steps.",
        )
        .unwrap();

        // Another flat file without frontmatter
        std::fs::write(tmp.path().join("greet.md"), "Say hello to the user.").unwrap();

        // A subdirectory skill should also work alongside flat files
        let skill_dir = tmp.path().join("build");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: build\n---\nBuild the project.",
        )
        .unwrap();

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 3);

        let deploy = provider.get("deploy").unwrap();
        assert_eq!(deploy.meta.description.as_deref(), Some("Deploy the app"));
        assert!(deploy.body.contains("Run deploy steps"));

        let greet = provider.get("greet").unwrap();
        assert!(greet.body.contains("Say hello"));

        assert!(provider.get("build").is_some());
    }

    #[tokio::test]
    async fn flat_md_deduplicates_with_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();

        // Subdirectory skill takes priority (scanned first)
        let skill_dir = tmp.path().join("deploy");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: from subdir\n---\nSubdir body.",
        )
        .unwrap();

        // Flat file with same name should be skipped
        std::fs::write(
            tmp.path().join("deploy.md"),
            "---\nname: deploy\ndescription: from flat\n---\nFlat body.",
        )
        .unwrap();

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 1);
        let skill = provider.get("deploy").unwrap();
        assert_eq!(skill.meta.description.as_deref(), Some("from subdir"));
    }
}
