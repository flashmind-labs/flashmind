//! Claude-style Markdown personas and their discovery.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_yaml::Value;

use super::catalog::{PersonaSource, merge};
use super::definition::{Persona, PersonaError, validate_name};

#[derive(Deserialize)]
struct Metadata {
    name: String,
    description: String,
    model: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl Persona {
    /// Parse a Claude-style agent file: YAML frontmatter with `name`,
    /// `description` and optional `model`, followed by the system prompt.
    ///
    /// `model: inherit` keeps the current model and a bare `claude-*` ID maps
    /// to the Anthropic provider. Other model aliases and any other
    /// frontmatter keys are not applied and are reported in
    /// [`notices`](Self::notices).
    pub fn from_markdown<S: AsRef<str>>(
        content: &str,
        reserved: &[S],
    ) -> Result<Self, PersonaError> {
        let mut lines = content.trim_start_matches('\u{feff}').lines();
        if lines.next().map(str::trim) != Some("---") {
            return Err(PersonaError::MissingFrontmatter);
        }
        let mut yaml = String::new();
        loop {
            match lines.next() {
                Some(line) if line.trim() == "---" => break,
                Some(line) => {
                    yaml.push_str(line);
                    yaml.push('\n');
                }
                None => return Err(PersonaError::UnclosedFrontmatter),
            }
        }
        let meta: Metadata = serde_yaml::from_str(&yaml)?;
        validate_name(&meta.name, reserved)?;
        if meta.name.trim() != meta.name || meta.description.trim().is_empty() {
            return Err(PersonaError::Header);
        }
        let system_prompt = lines.collect::<Vec<_>>().join("\n").trim().to_owned();
        if system_prompt.is_empty() {
            return Err(PersonaError::EmptyPrompt);
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
        Ok(Self {
            name: meta.name,
            description: Some(meta.description),
            notices,
            system_prompt,
            model,
            reasoning: None,
        })
    }
}

/// Markdown personas read from a caller-given list of directories.
///
/// Each directory's `*.md` files are read in name order; subdirectories are
/// not searched. A persona in a later directory replaces one with the same
/// name (case-insensitive) from an earlier one, so list directories from the
/// widest scope to the narrowest. Missing directories are skipped, and
/// unreadable or invalid files are skipped with a warning.
///
/// The files are read again on every [`load`](Self::load), so edits apply
/// without a restart.
#[derive(Debug, Clone, Default)]
pub struct MarkdownPersonas {
    dirs: Vec<PathBuf>,
    reserved: Vec<String>,
}

impl MarkdownPersonas {
    /// Search `dirs`, lowest precedence first.
    pub fn new(dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            dirs: dirs.into_iter().collect(),
            reserved: Vec::new(),
        }
    }

    /// Reject personas whose name matches one of `names`, such as the host's
    /// own commands.
    pub fn with_reserved_names(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.reserved = names.into_iter().collect();
        self
    }

    /// The directories searched, lowest precedence first.
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// The reserved names given to [`with_reserved_names`](Self::with_reserved_names).
    pub fn reserved_names(&self) -> &[String] {
        &self.reserved
    }

    /// Read every directory and return the personas, sorted by name.
    pub fn load(&self) -> Vec<Persona> {
        merge(
            self.dirs
                .iter()
                .flat_map(|dir| load_dir(dir, &self.reserved)),
        )
    }
}

impl PersonaSource for MarkdownPersonas {
    fn personas(&self) -> Vec<Persona> {
        self.load()
    }
}

/// The persona directories of one scope, lowest precedence first:
/// `root/.claude/agents`, then `root/.agents`.
pub fn scope_dirs(root: &Path) -> [PathBuf; 2] {
    [root.join(".claude/agents"), root.join(".agents")]
}

/// The [`scope_dirs`] of `dir` and each of its ancestors, outermost first, so
/// the folder nearest `dir` wins.
pub fn project_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut roots: Vec<&Path> = dir.ancestors().collect();
    roots.reverse();
    roots.into_iter().flat_map(scope_dirs).collect()
}

fn load_dir(dir: &Path, reserved: &[String]) -> Vec<Persona> {
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
    paths
        .into_iter()
        .filter_map(|path| match load_file(&path, reserved) {
            Ok(persona) => Some(persona),
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "skipping invalid Markdown agent");
                None
            }
        })
        .collect()
}

fn load_file(path: &Path, reserved: &[String]) -> Result<Persona, PersonaError> {
    let text = fs::read_to_string(path).map_err(|source| PersonaError::Read {
        path: path.to_owned(),
        source,
    })?;
    Persona::from_markdown(&text, reserved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona::delegate_instructions;

    const PERSONA: &str = "---\nname: reviewer\ndescription: |\n  Review code carefully.\nmodel: inherit\n---\nYou are a meticulous reviewer.\n\nKeep feedback brief.\n";

    fn parse(text: &str) -> Result<Persona, PersonaError> {
        Persona::from_markdown(text, &["model"])
    }

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
        assert!(matches!(
            parse(&PERSONA.replace("name: reviewer", "name: model")),
            Err(PersonaError::Name(_))
        ));
    }

    #[test]
    fn discovers_personas_and_overrides_by_scope() {
        let temp = tempfile::tempdir().unwrap();
        let user = temp.path().join("user/.claude/agents");
        let project = temp.path().join("project/.agents");
        for dir in [&user, &project] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(user.join("reviewer.md"), PERSONA).unwrap();
        let project_text = PERSONA.replace("meticulous reviewer", "project reviewer");
        fs::write(project.join("reviewer.md"), &project_text).unwrap();
        fs::write(project.join("broken.md"), "---\nnot valid").unwrap();
        fs::create_dir_all(project.join("skills/example")).unwrap();
        fs::write(project.join("skills/example/SKILL.md"), PERSONA).unwrap();
        let personas = MarkdownPersonas::new([user, project.clone()]);
        let list = personas.load();
        assert_eq!(list.len(), 1);
        let instructions = delegate_instructions(&list);
        assert!(instructions.contains("## Agent personas"));
        assert!(instructions.contains("- reviewer: Review code carefully."));
        assert!(instructions.contains("`delegate`"));
        assert!(!instructions.contains("project reviewer"));
        let def = personas.get("REVIEWER").unwrap();
        assert!(def.system_prompt.contains("project reviewer"));
        assert_eq!(
            fs::read_to_string(project.join("reviewer.md")).unwrap(),
            project_text
        );
    }

    #[test]
    fn discovery_reads_only_the_given_dirs() {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("srv/project");
        let other = temp.path().join("srv/other");
        for root in [&project, &other] {
            for dir in scope_dirs(root) {
                fs::create_dir_all(dir).unwrap();
            }
        }
        fs::write(project.join(".agents/reviewer.md"), PERSONA).unwrap();
        fs::write(
            other.join(".agents/planner.md"),
            PERSONA.replace("name: reviewer", "name: planner"),
        )
        .unwrap();
        fs::write(
            project.join(".claude/agents/writer.md"),
            PERSONA.replace("name: reviewer", "name: writer"),
        )
        .unwrap();

        assert!(MarkdownPersonas::default().load().is_empty());
        let names: Vec<String> = MarkdownPersonas::new(scope_dirs(&project))
            .load()
            .into_iter()
            .map(|persona| persona.name)
            .collect();
        assert_eq!(names, ["reviewer", "writer"]);

        let reserved =
            MarkdownPersonas::new(scope_dirs(&project)).with_reserved_names(["writer".to_owned()]);
        assert_eq!(reserved.load().len(), 1);
    }

    #[test]
    fn scope_dirs_prefer_dot_agents_within_a_scope() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir_all(root.join(".claude/agents")).unwrap();
        fs::create_dir_all(root.join(".agents")).unwrap();
        fs::write(root.join(".claude/agents/reviewer.md"), PERSONA).unwrap();
        fs::write(
            root.join(".agents/reviewer.md"),
            PERSONA.replace("meticulous", "dot agents"),
        )
        .unwrap();
        let personas = MarkdownPersonas::new(scope_dirs(root));
        assert!(
            personas
                .get("reviewer")
                .unwrap()
                .system_prompt
                .contains("dot agents")
        );
    }

    #[test]
    fn project_dirs_list_ancestors_outermost_first() {
        let dirs = project_dirs(Path::new("/srv/project"));
        let expected: Vec<PathBuf> = [
            "/.claude/agents",
            "/.agents",
            "/srv/.claude/agents",
            "/srv/.agents",
            "/srv/project/.claude/agents",
            "/srv/project/.agents",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(dirs, expected);
    }
}
