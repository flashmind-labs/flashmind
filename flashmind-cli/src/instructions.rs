//! Loads AGENTS.md and CLAUDE.md files into the system prompt.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const FILE_NAMES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// Global instruction files, read before any project file.
pub fn global_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(dir) = crate::paths::config_dir() {
        files.extend(FILE_NAMES.iter().map(|name| dir.join(name)));
    }
    if let Some(home) = dirs::home_dir() {
        files.push(home.join(".claude/CLAUDE.md"));
    }
    files
}

/// Instruction files from `global`, then from every directory between the
/// filesystem root and `cwd`, outermost first so closer files come last and
/// take precedence. Returns an empty string when none exist.
pub fn project_instructions(cwd: &Path, global: &[PathBuf]) -> String {
    let mut dirs: Vec<&Path> = cwd.ancestors().collect();
    dirs.reverse();
    let candidates = global.iter().cloned().chain(
        dirs.into_iter()
            .flat_map(|dir| FILE_NAMES.iter().map(move |name| dir.join(name))),
    );

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

    #[test]
    fn empty_without_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(project_instructions(dir.path(), &[]), "");
    }

    #[test]
    fn loads_global_then_outer_then_inner() {
        let dir = tempfile::tempdir().expect("temp dir");
        let inner = dir.path().join("repo/crate");
        fs::create_dir_all(&inner).expect("create dirs");
        let global = dir.path().join("global.md");
        fs::write(&global, "global rules").expect("write");
        fs::write(dir.path().join("repo/AGENTS.md"), "repo agents").expect("write");
        fs::write(dir.path().join("repo/CLAUDE.md"), "repo claude").expect("write");
        fs::write(inner.join("CLAUDE.md"), "crate rules").expect("write");

        let got = project_instructions(&inner, &[global]);
        let order: Vec<usize> = ["global rules", "repo agents", "repo claude", "crate rules"]
            .iter()
            .map(|s| got.find(s).expect("content included"))
            .collect();
        assert!(order.is_sorted(), "{got}");
    }

    #[test]
    fn skips_duplicates_and_empty_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("AGENTS.md"), "same").expect("write");
        fs::write(dir.path().join("CLAUDE.md"), "same\n").expect("write");
        let blank = dir.path().join("blank.md");
        fs::write(&blank, "  \n").expect("write");

        let got = project_instructions(dir.path(), &[blank]);
        assert_eq!(got.matches("same").count(), 1, "{got}");
        assert!(!got.contains("blank.md"), "{got}");
    }

    #[cfg(unix)]
    #[test]
    fn follows_symlink_once() {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("AGENTS.md"), "linked").expect("write");
        std::os::unix::fs::symlink("AGENTS.md", dir.path().join("CLAUDE.md")).expect("symlink");

        let got = project_instructions(dir.path(), &[]);
        assert_eq!(got.matches("<instructions").count(), 1, "{got}");
    }
}
