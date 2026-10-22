//! Skill directories from enabled Claude plugins.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Settings {
    #[serde(default)]
    enabled_plugins: HashMap<String, bool>,
}

#[derive(Default, Deserialize)]
struct InstalledPlugins {
    plugins: BTreeMap<String, Vec<Installation>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Installation {
    scope: Scope,
    install_path: PathBuf,
    project_path: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Scope {
    User,
    Project,
    Local,
    #[serde(other)]
    Other,
}

fn read_json<T: DeserializeOwned + Default>(path: &Path) -> Result<T> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .with_context(|| format!("invalid plugin config: {}", path.display())),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(T::default()),
        Err(error) => Err(error).with_context(|| format!("could not read {}", path.display())),
    }
}

pub(crate) fn skill_dirs(claude_dir: &Path, cwd: &Path) -> Result<Vec<PathBuf>> {
    let installed: InstalledPlugins =
        read_json(&claude_dir.join("plugins/installed_plugins.json"))?;
    let mut enabled = read_json::<Settings>(&claude_dir.join("settings.json"))?.enabled_plugins;
    let mut projects: Vec<&Path> = cwd.ancestors().collect();
    projects.reverse();
    for project in projects {
        for file in ["settings.json", "settings.local.json"] {
            enabled.extend(
                read_json::<Settings>(&project.join(".claude").join(file))?.enabled_plugins,
            );
        }
    }

    let mut dirs = Vec::new();
    for (name, installations) in installed.plugins {
        if enabled.get(&name) != Some(&true) {
            continue;
        }
        for installation in installations {
            let applies = match installation.scope {
                Scope::User => true,
                Scope::Project | Scope::Local => installation
                    .project_path
                    .as_ref()
                    .is_some_and(|project| cwd.starts_with(project)),
                Scope::Other => false,
            };
            let dir = installation.install_path.join("skills");
            if applies && dir.is_dir() && !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    Ok(dirs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_skills::DiskSkillProvider;
    use serde_json::{Value, json};

    fn write_json(path: &Path, value: Value) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[tokio::test]
    async fn discovers_enabled_installations_and_respects_project_settings() {
        let temp = tempfile::tempdir().unwrap();
        let claude_dir = temp.path().join("config");
        let project = temp.path().join("project");
        let cwd = project.join("src");
        let other_project = temp.path().join("other-project");
        let user = temp.path().join("user-plugin");
        let local = temp.path().join("local-plugin");
        let other = temp.path().join("other-plugin");
        let disabled = temp.path().join("disabled-plugin");
        for install in [&user, &local, &other, &disabled] {
            fs::create_dir_all(install.join("skills")).unwrap();
        }
        write_json(
            &claude_dir.join("plugins/installed_plugins.json"),
            json!({"version": 2, "plugins": {
                "a@examples": [{"scope": "user", "installPath": user}],
                "b@examples": [{"scope": "project", "installPath": local, "projectPath": project}],
                "c@examples": [{"scope": "local", "installPath": other, "projectPath": other_project}],
                "d@examples": [{"scope": "user", "installPath": disabled}]
            }}),
        );
        write_json(
            &claude_dir.join("settings.json"),
            json!({"enabledPlugins": {"a@examples": true, "b@examples": false, "c@examples": true, "d@examples": true}}),
        );
        write_json(
            &project.join(".claude/settings.json"),
            json!({"enabledPlugins": {"b@examples": true}}),
        );
        write_json(
            &project.join(".claude/settings.local.json"),
            json!({"enabledPlugins": {"d@examples": false}}),
        );
        assert_eq!(
            skill_dirs(&claude_dir, &cwd).unwrap(),
            vec![user.join("skills"), local.join("skills")]
        );

        let skill = user.join("skills/terse");
        fs::create_dir_all(&skill).unwrap();
        fs::write(skill.join("SKILL.md"), "Answer in one sentence.").unwrap();
        fs::write(claude_dir.join(".terse-always"), "").unwrap();
        let provider = DiskSkillProvider::discover(skill_dirs(&claude_dir, &cwd).unwrap())
            .await
            .unwrap();
        let loaded = crate::tools::always_on_skills(&provider, &[], &[claude_dir]);
        assert_eq!(loaded.loaded, ["terse"]);
        assert!(loaded.missing.is_empty());
        assert!(loaded.prompt.contains("Answer in one sentence."));
    }

    #[test]
    fn missing_registry_is_empty_and_invalid_registry_returns_an_error() {
        let temp = tempfile::tempdir().unwrap();
        let claude_dir = temp.path().join("config");
        assert!(skill_dirs(&claude_dir, temp.path()).unwrap().is_empty());
        fs::create_dir_all(claude_dir.join("plugins")).unwrap();
        fs::write(claude_dir.join("plugins/installed_plugins.json"), "{").unwrap();
        assert!(skill_dirs(&claude_dir, temp.path()).is_err());
    }
}
