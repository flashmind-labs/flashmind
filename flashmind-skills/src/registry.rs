//! Filesystem-backed skill provider.
//!
//! [`DiskSkillProvider`] scans directories for `SKILL.md` files and makes
//! discovered skills available via the [`SkillProvider`] trait.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::ErrorKind;
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
    /// Searches collections up to eight directory levels deep, stopping at each
    /// `SKILL.md`. Flat Markdown commands load only from search roots.
    /// Deduplicates by name (first search directory, then shallowest skill wins).
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

        for dir in &self.search_dirs {
            scan_dir(dir, &mut self.skills).await?;
        }

        tracing::info!(count = self.skills.len(), "skill discovery complete");
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

const MAX_SKILL_DEPTH: usize = 8;

async fn scan_dir(dir: &Path, skills: &mut HashMap<String, Skill>) -> anyhow::Result<()> {
    let mut pending = VecDeque::from([(dir.to_path_buf(), 0)]);
    let mut visited = HashSet::new();

    while let Some((directory, depth)) = pending.pop_front() {
        let canonical = match tokio::fs::canonicalize(&directory).await {
            Ok(path) => path,
            Err(error) => {
                tracing::warn!(dir = %directory.display(), %error, "failed to resolve skill directory");
                continue;
            }
        };
        if !visited.insert(canonical) {
            continue;
        }

        let mut reader = match tokio::fs::read_dir(&directory).await {
            Ok(reader) => reader,
            Err(error) => {
                tracing::warn!(dir = %directory.display(), %error, "failed to read skill directory");
                continue;
            }
        };
        let mut entries = Vec::new();
        while let Some(entry) = reader.next_entry().await? {
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.file_name());

        for entry in entries {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') || name == "node_modules" || name == "target"
            {
                continue;
            }
            let path = entry.path();
            let metadata = match tokio::fs::metadata(&path).await {
                Ok(metadata) => metadata,
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "failed to inspect skill directory");
                    continue;
                }
            };
            if !metadata.is_dir() {
                continue;
            }

            let skill_md = path.join("SKILL.md");
            let content = match tokio::fs::read_to_string(&skill_md).await {
                Ok(content) => content,
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    if depth + 1 < MAX_SKILL_DEPTH {
                        pending.push_back((path, depth + 1));
                    }
                    continue;
                }
                Err(error) => {
                    tracing::warn!(path = %skill_md.display(), %error, "failed to read SKILL.md");
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
                }
                Err(error) => {
                    tracing::warn!(path = %skill_md.display(), %error, "failed to parse SKILL.md");
                }
            }
        }

        if depth == 0 {
            scan_commands(dir, skills).await?;
        }
    }

    Ok(())
}

async fn scan_commands(dir: &Path, skills: &mut HashMap<String, Skill>) -> anyhow::Result<()> {
    let mut flat_entries = match tokio::fs::read_dir(dir).await {
        Ok(e) => e,
        Err(error) => {
            tracing::warn!(dir = %dir.display(), %error, "failed to read command directory");
            return Ok(());
        }
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
                skill.source_path = path.clone();
                skill.meta.name = name.clone();
                tracing::debug!(name = %name, path = %path.display(), "discovered flat skill file");
                skills.insert(name, skill);
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

    fn write_skill(root: &Path, relative: &str, body: &str) -> PathBuf {
        let dir = root.join(relative);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), body).unwrap();
        dir
    }

    #[tokio::test]
    async fn discovers_nested_collections_without_loading_their_readmes() {
        let tmp = tempfile::tempdir().unwrap();
        let alpha = write_skill(tmp.path(), "collection/alpha", "Alpha instructions.");
        let beta = write_skill(tmp.path(), "collection/skills/beta", "Beta instructions.");
        std::fs::write(
            tmp.path().join("collection/README.md"),
            "Collection overview.",
        )
        .unwrap();

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 2);
        let skill = provider.get("alpha").unwrap();
        assert_eq!(skill.body, "Alpha instructions.");
        assert_eq!(skill.dir, alpha);
        assert_eq!(skill.source_path, alpha.join("SKILL.md"));
        assert_eq!(provider.get("beta").unwrap().dir, beta);
    }

    #[tokio::test]
    async fn skill_directories_are_leaves() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "collection/parent", "Parent instructions.");
        write_skill(
            tmp.path(),
            "collection/parent/examples/child",
            "Example instructions.",
        );
        std::fs::write(
            tmp.path().join("collection/parent/reference.md"),
            "Supporting material.",
        )
        .unwrap();

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 1);
        assert!(provider.get("parent").is_some());
    }

    #[tokio::test]
    async fn nested_discovery_preserves_search_order_and_prefers_shallower_skills() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write_skill(first.path(), "collection/shared", "Nested in first root.");
        write_skill(second.path(), "shared", "Direct in second root.");
        write_skill(
            first.path(),
            "collection/shallow",
            "---\nname: shallow\n---\nNested duplicate.",
        );
        let shallow = write_skill(first.path(), "shallow", "Direct skill.");
        std::fs::write(first.path().join("flat.md"), "Direct command.").unwrap();
        write_skill(first.path(), "collection/flat", "Nested duplicate command.");

        let provider = DiskSkillProvider::discover(vec![
            first.path().to_path_buf(),
            second.path().to_path_buf(),
        ])
        .await
        .unwrap();

        assert_eq!(
            provider.get("shared").unwrap().body,
            "Nested in first root."
        );
        assert_eq!(provider.get("shallow").unwrap().dir, shallow);
        assert_eq!(provider.get("flat").unwrap().body, "Direct command.");
    }

    #[tokio::test]
    async fn nested_duplicates_prefer_depth_then_alphabetical_order() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "z-collection/shared", "Later collection.");
        write_skill(tmp.path(), "a-collection/shared", "Earlier collection.");
        write_skill(tmp.path(), "a-collection/deeper/example", "Deeper skill.");
        write_skill(tmp.path(), "z-collection/example", "Shallower skill.");

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.get("shared").unwrap().body, "Earlier collection.");
        assert_eq!(provider.get("example").unwrap().body, "Shallower skill.");
    }

    #[tokio::test]
    async fn discovery_stops_after_eight_directory_levels() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "a/b/c/d/e/f/g/included", "Within limit.");
        write_skill(tmp.path(), "a/b/c/d/e/f/g/h/excluded", "Beyond limit.");

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 1);
        assert!(provider.get("included").is_some());
        assert!(provider.get("excluded").is_none());
    }

    #[tokio::test]
    async fn nested_discovery_skips_metadata_and_build_directories() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "collection/real", "Real instructions.");
        for directory in [".git", "node_modules", "target"] {
            write_skill(
                tmp.path(),
                &format!("collection/{directory}/ignored"),
                "Ignore.",
            );
        }

        let provider = DiskSkillProvider::discover(vec![tmp.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 1);
        assert!(provider.get("real").is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn follows_symlinked_collections_without_following_cycles() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let collection = tempfile::tempdir().unwrap();
        write_skill(
            collection.path(),
            "nested/example",
            "Symlinked instructions.",
        );
        symlink(collection.path(), root.path().join("collection")).unwrap();
        symlink(root.path(), collection.path().join("loop")).unwrap();

        let provider = DiskSkillProvider::discover(vec![root.path().to_path_buf()])
            .await
            .unwrap();

        assert_eq!(provider.list().len(), 1);
        let skill = provider.get("example").unwrap();
        assert_eq!(skill.body, "Symlinked instructions.");
        assert_eq!(skill.dir, root.path().join("collection/nested/example"));
    }

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
        assert_eq!(deploy.source_path, tmp.path().join("deploy.md"));

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
        assert_eq!(skill.source_path, skill_dir.join("SKILL.md"));
    }
}
