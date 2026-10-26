//! Language server commands and project detection.

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A stdio language server. Commands must already be installed.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LanguageServerConfig {
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub root_markers: Vec<String>,
    #[serde(default)]
    pub initialization: Value,
    #[serde(default)]
    pub disabled: bool,
}

impl LanguageServerConfig {
    fn builtin(command: &[&str], extensions: &[&str], markers: &[&str]) -> Self {
        Self {
            command: command.iter().map(|value| (*value).to_owned()).collect(),
            extensions: extensions.iter().map(|value| (*value).to_owned()).collect(),
            root_markers: markers.iter().map(|value| (*value).to_owned()).collect(),
            initialization: Value::Null,
            disabled: false,
        }
    }

    pub(super) fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                self.extensions
                    .iter()
                    .any(|candidate| candidate.trim_start_matches('.') == extension)
            })
    }
}

pub(super) fn builtins() -> BTreeMap<String, LanguageServerConfig> {
    BTreeMap::from([
        (
            "rust".into(),
            LanguageServerConfig::builtin(&["rust-analyzer"], &["rs"], &["Cargo.toml"]),
        ),
        (
            "typescript".into(),
            LanguageServerConfig::builtin(
                &["typescript-language-server", "--stdio"],
                &["ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"],
                &["tsconfig.json", "jsconfig.json", "package.json"],
            ),
        ),
        (
            "python".into(),
            LanguageServerConfig::builtin(
                &["pyright-langserver", "--stdio"],
                &["py", "pyi"],
                &[
                    "pyproject.toml",
                    "pyrightconfig.json",
                    "setup.py",
                    "requirements.txt",
                ],
            ),
        ),
        (
            "go".into(),
            LanguageServerConfig::builtin(&["gopls"], &["go"], &["go.work", "go.mod"]),
        ),
        (
            "cpp".into(),
            LanguageServerConfig::builtin(
                &["clangd"],
                &["c", "h", "cc", "cpp", "cxx", "hpp", "hh"],
                &[
                    "compile_commands.json",
                    "compile_flags.txt",
                    "CMakeLists.txt",
                ],
            ),
        ),
    ])
}

pub(super) fn executable(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() {
        return path.is_file().then(|| path.to_path_buf());
    }
    env::split_paths(&env::var_os("PATH")?)
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}

pub(super) fn language_id(path: &Path) -> &str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
    {
        "rs" => "rust",
        "ts" | "mts" | "cts" => "typescript",
        "tsx" => "typescriptreact",
        "js" | "mjs" | "cjs" => "javascript",
        "jsx" => "javascriptreact",
        "py" | "pyi" => "python",
        "go" => "go",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hh" => "cpp",
        extension => extension,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_language_ids_and_extensions() {
        let servers = builtins();
        assert!(servers["typescript"].supports(Path::new("src/view.tsx")));
        assert!(servers["typescript"].supports(Path::new("src/app.js")));
        assert!(!servers["rust"].supports(Path::new("app.py")));
        assert_eq!(language_id(Path::new("view.tsx")), "typescriptreact");
        assert_eq!(language_id(Path::new("app.py")), "python");
    }
}
