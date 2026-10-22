//! Loads AGENTS.md and CLAUDE.md files into the system prompt.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use flashmind_types::{Model, Provider};

/// Instruction files found at one level: the global config, or one project
/// directory. CLAUDE.md files are skipped when the level has an AGENTS.md,
/// unless the model is a Claude model.
#[derive(Debug, Default)]
pub struct Level {
    pub agents: Vec<PathBuf>,
    pub claude: Vec<PathBuf>,
}

impl Level {
    fn dir(dir: &Path) -> Self {
        Self {
            agents: vec![dir.join("AGENTS.md")],
            claude: vec![dir.join("CLAUDE.md")],
        }
    }
}

/// Global instruction files, read before any project file.
pub fn global_level() -> Level {
    let mut level = Level::default();
    if let Ok(dir) = crate::paths::config_dir() {
        level = Level::dir(&dir);
    }
    if let Some(home) = dirs::home_dir() {
        level.claude.push(home.join(".claude/CLAUDE.md"));
    }
    level
}

/// Whether `model` is a Claude model, served by Anthropic or a router.
pub fn is_claude_model(model: &Model) -> bool {
    model.provider == Provider::Anthropic
        || model
            .model
            .capability_name()
            .to_ascii_lowercase()
            .contains("claude")
}

/// Instruction files from `global`, then from every directory between the
/// filesystem root and `cwd`, outermost first so closer files come last and
/// take precedence. Returns an empty string when none exist.
pub fn project_instructions(cwd: &Path, global: Level, claude_model: bool) -> String {
    let mut dirs: Vec<&Path> = cwd.ancestors().collect();
    dirs.reverse();
    let levels = std::iter::once(global).chain(dirs.into_iter().map(Level::dir));
    let candidates = levels.flat_map(|level| {
        let has_agents = level.agents.iter().any(|path| path.is_file());
        let claude = if claude_model || !has_agents {
            level.claude
        } else {
            Vec::new()
        };
        level.agents.into_iter().chain(claude)
    });

    let mut seen_paths = HashSet::new();
    let mut seen_contents = HashSet::new();
    let mut out = String::new();
    for path in candidates {
        // Canonicalizing drops symlinks such as CLAUDE.md -> AGENTS.md.
        let Ok(real) = fs::canonicalize(&path) else {
            continue;
        };
        if !real.is_file() || !seen_paths.insert(real) {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let content = content.trim();
        if content.is_empty() || !seen_contents.insert(content.to_string()) {
            continue;
        }
        out.push_str(&format!(
            "\n\n<instructions path=\"{}\">\n{content}\n</instructions>",
            path.display()
        ));
    }

    if out.is_empty() {
        return out;
    }
    format!(
        "\n\n## Instructions\n\nFollow these instruction files. When they conflict, later files win over earlier ones.{out}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global(path: PathBuf) -> Level {
        Level {
            agents: Vec::new(),
            claude: vec![path],
        }
    }

    #[test]
    fn empty_without_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(project_instructions(dir.path(), Level::default(), true), "");
    }

    #[test]
    fn loads_global_then_outer_then_inner() {
        let dir = tempfile::tempdir().expect("temp dir");
        let inner = dir.path().join("repo/crate");
        fs::create_dir_all(&inner).expect("create dirs");
        let global_file = dir.path().join("global.md");
        fs::write(&global_file, "global rules").expect("write");
        fs::write(dir.path().join("repo/AGENTS.md"), "repo agents").expect("write");
        fs::write(dir.path().join("repo/CLAUDE.md"), "repo claude").expect("write");
        fs::write(inner.join("CLAUDE.md"), "crate rules").expect("write");

        let got = project_instructions(&inner, global(global_file), true);
        let order: Vec<usize> = ["global rules", "repo agents", "repo claude", "crate rules"]
            .iter()
            .map(|s| got.find(s).expect("content included"))
            .collect();
        assert!(order.is_sorted(), "{got}");
    }

    #[test]
    fn agents_md_hides_claude_md_for_other_models() {
        let dir = tempfile::tempdir().expect("temp dir");
        let inner = dir.path().join("repo/crate");
        fs::create_dir_all(&inner).expect("create dirs");
        fs::write(dir.path().join("repo/AGENTS.md"), "repo agents").expect("write");
        fs::write(dir.path().join("repo/CLAUDE.md"), "repo claude").expect("write");
        fs::write(inner.join("CLAUDE.md"), "crate rules").expect("write");

        let got = project_instructions(&inner, Level::default(), false);
        assert!(got.contains("repo agents"), "{got}");
        assert!(!got.contains("repo claude"), "{got}");
        assert!(got.contains("crate rules"), "{got}");
    }

    #[test]
    fn global_agents_md_hides_global_claude_md() {
        let dir = tempfile::tempdir().expect("temp dir");
        let agents = dir.path().join("AGENTS.md");
        let claude = dir.path().join("home-claude.md");
        fs::write(&agents, "global agents").expect("write");
        fs::write(&claude, "global claude").expect("write");
        let cwd = dir.path().join("work");
        fs::create_dir_all(&cwd).expect("create dirs");
        let level = || Level {
            agents: vec![agents.clone()],
            claude: vec![claude.clone()],
        };

        let other = project_instructions(&cwd, level(), false);
        assert!(!other.contains("global claude"), "{other}");
        let claude_got = project_instructions(&cwd, level(), true);
        assert!(claude_got.contains("global claude"), "{claude_got}");
    }

    #[test]
    fn detects_claude_models() {
        let claude = |s: &str| is_claude_model(&s.parse().expect("model"));
        assert!(claude("anthropic:claude-sonnet-4-20250514"));
        assert!(claude("openrouter:anthropic/claude-sonnet-4"));
        assert!(claude("openrouter:sonnet,anthropic/claude-sonnet-4"));
        assert!(!claude("openrouter:openai/gpt-5"));
        assert!(!claude("ollama:llama3.2"));
    }

    #[test]
    fn skips_duplicates_and_empty_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("AGENTS.md"), "same").expect("write");
        fs::write(dir.path().join("CLAUDE.md"), "same\n").expect("write");
        let blank = dir.path().join("blank.md");
        fs::write(&blank, "  \n").expect("write");

        let got = project_instructions(dir.path(), global(blank), true);
        assert_eq!(got.matches("same").count(), 1, "{got}");
        assert!(!got.contains("blank.md"), "{got}");
    }

    #[cfg(unix)]
    #[test]
    fn follows_symlink_once() {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("AGENTS.md"), "linked").expect("write");
        std::os::unix::fs::symlink("AGENTS.md", dir.path().join("CLAUDE.md")).expect("symlink");

        let got = project_instructions(dir.path(), Level::default(), true);
        assert_eq!(got.matches("<instructions").count(), 1, "{got}");
    }
}
