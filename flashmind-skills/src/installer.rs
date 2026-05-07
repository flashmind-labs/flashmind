//! Skill directory creation and `.env` file setup.

use std::path::{Path, PathBuf};

/// Creates skill directories and writes initial configuration files.
pub struct SkillInstaller;

impl SkillInstaller {
    /// Create a new skill directory with an optional `.env` file.
    ///
    /// Creates `{base_dir}/{name}/` and writes a `.env` file if `env_vars` is
    /// provided. Returns the path to the created skill directory.
    pub async fn install(
        base_dir: &Path,
        name: &str,
        env_vars: Option<Vec<(String, String)>>,
    ) -> anyhow::Result<PathBuf> {
        let skill_dir = base_dir.join(name);
        tokio::fs::create_dir_all(&skill_dir).await?;

        if let Some(vars) = env_vars {
            let content: String = vars
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("\n");
            tokio::fs::write(skill_dir.join(".env"), content).await?;
        }

        tracing::info!(name, dir = %skill_dir.display(), "skill installed");
        Ok(skill_dir)
    }
}
