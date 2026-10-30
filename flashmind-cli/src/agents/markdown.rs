//! Import Claude-style agent prompts as Flashmind personas.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_yaml::Value;

use super::{AgentDef, validate_name};

#[derive(Deserialize)]
struct Metadata {
    name: String,
    description: String,
    model: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

fn parse(content: &str) -> Result<AgentDef> {
    let mut lines = content.trim_start_matches('\u{feff}').lines();
    if lines.next().map(str::trim) != Some("---") {
        bail!("expected YAML frontmatter");
    }
    let mut yaml = String::new();
    loop {
        match lines.next() {
            Some(line) if line.trim() == "---" => break,
            Some(line) => {
                yaml.push_str(line);
                yaml.push('\n');
            }
            None => bail!("missing closing frontmatter delimiter"),
        }
    }
    let meta: Metadata = serde_yaml::from_str(&yaml).context("invalid agent metadata")?;
    validate_name(&meta.name).map_err(anyhow::Error::msg)?;
    if meta.name.trim() != meta.name || meta.description.trim().is_empty() {
        bail!(
            "agent name and description must be nonempty and name must not have surrounding whitespace"
        );
    }
    let system_prompt = lines.collect::<Vec<_>>().join("\n").trim().to_owned();
    if system_prompt.is_empty() {
        bail!("agent prompt is empty");
    }
    let mut notices = Vec::new();
    let model = match meta.model.as_deref() {
        None | Some("inherit") => None,
        Some(model) if model.contains(':') => Some(model.to_owned()),
        Some(model) if model.starts_with("claude-") => Some(format!("anthropic:{model}")),
        Some(model) => {
            notices.push(format!(
                "Agent model alias '{model}' is unsupported; keeping the current model. Use provider:model or a full Claude model ID to select a model."
            ));
            None
        }
    };
    if !meta.extra.is_empty() {
        notices.push(format!(
            "Imported as a persona. Unsupported agent settings are not applied: {}.",
            meta.extra.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(AgentDef {
        name: meta.name,
        description: Some(meta.description),
        notices,
        system_prompt,
        model,
        reasoning: None,
    })
}

pub(super) fn load_dir(dir: &Path) -> Vec<AgentDef> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            tracing::warn!(path = %dir.display(), %error, "cannot read agent directory");
            return Vec::new();
        }
    };
    let mut paths = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) if entry.path().extension().is_some_and(|ext| ext == "md") => {
                paths.push(entry.path());
            }
            Ok(_) => {}
            Err(error) => tracing::warn!(%error, "cannot read agent directory entry"),
        }
    }
    paths.sort();
    paths.into_iter().filter_map(|path| {
        match fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display())).and_then(|text| parse(&text)) {
            Ok(def) => Some(def),
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "skipping invalid Markdown agent");
                None
            }
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::AgentStore;

    const PERSONA: &str = "---\nname: reviewer\ndescription: |\n  Review code carefully.\nmodel: inherit\n---\nYou are a meticulous reviewer.\n\nKeep feedback brief.\n";

    #[test]
    fn loads_personality_and_multiline_description_with_crlf() {
        let def = parse(&PERSONA.replace('\n', "\r\n")).unwrap();
        assert_eq!(def.name, "reviewer");
        assert_eq!(def.description.as_deref(), Some("Review code carefully.\n"));
        assert_eq!(
            def.system_prompt,
            "You are a meticulous reviewer.\n\nKeep feedback brief."
        );
        assert!(def.model.is_none());
        assert!(def.notices.is_empty());
    }

    #[test]
    fn handles_models_and_reports_unsupported_settings() {
        let def = parse(&PERSONA.replace(
            "model: inherit",
            "model: sonnet\ntools: [Read, Grep]\nhooks: {}",
        ))
        .unwrap();
        assert!(def.model.is_none());
        assert!(def.notices[0].contains("sonnet"));
        assert!(def.notices[1].contains("hooks, tools"));
        let def = parse(&PERSONA.replace("inherit", "claude-example")).unwrap();
        assert_eq!(def.model.as_deref(), Some("anthropic:claude-example"));
        let def = parse(&PERSONA.replace("inherit", "ollama:example")).unwrap();
        assert_eq!(def.model.as_deref(), Some("ollama:example"));
    }

    #[test]
    fn rejects_invalid_files_and_reserved_names() {
        for text in [
            "# Notes",
            "---\nname: reviewer",
            "---\nname: reviewer\n---\nPrompt",
            "---\nname: reviewer\ndescription: Review\n---\n",
            "---\nname: [bad]\ndescription: Review\n---\nPrompt",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
        assert!(parse(&PERSONA.replace("name: reviewer", "name: model")).is_err());
    }

    #[test]
    fn discovers_personas_overrides_by_scope_and_preserves_sources() {
        let temp = tempfile::tempdir().unwrap();
        let user = temp.path().join("user/.claude/agents");
        let project = temp.path().join("project/.agents");
        let saved = temp.path().join("saved");
        for dir in [&user, &project, &saved] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(user.join("reviewer.md"), PERSONA).unwrap();
        let project_text = PERSONA.replace("meticulous reviewer", "project reviewer");
        fs::write(project.join("reviewer.md"), &project_text).unwrap();
        fs::write(project.join("broken.md"), "---\nnot valid").unwrap();
        fs::create_dir_all(project.join("skills/example")).unwrap();
        fs::write(project.join("skills/example/SKILL.md"), PERSONA).unwrap();
        let store = AgentStore {
            dir: saved,
            markdown_dirs: vec![user, project.clone()],
        };
        assert_eq!(store.list().len(), 1);
        let instructions = store.instructions(true);
        assert!(instructions.contains("## Agent personas"));
        assert!(instructions.contains("- reviewer: Review code carefully."));
        assert!(instructions.contains("`delegate`"));
        assert!(!instructions.contains("project reviewer"));
        assert!(store.instructions(false).is_empty());
        let mut def = store.get("REVIEWER").unwrap();
        assert!(def.system_prompt.contains("project reviewer"));
        assert!(!store.delete("reviewer").unwrap());
        def.system_prompt = "Saved persona".into();
        store.save(&def).unwrap();
        assert_eq!(
            store.get("reviewer").unwrap().system_prompt,
            "Saved persona"
        );
        assert!(store.delete("reviewer").unwrap());
        assert!(
            store
                .get("reviewer")
                .unwrap()
                .system_prompt
                .contains("project reviewer")
        );
        assert_eq!(
            fs::read_to_string(project.join("reviewer.md")).unwrap(),
            project_text
        );
    }
}
