//! Command allow-list for interactive permission prompts.

use std::sync::Mutex;

use glob::Pattern;
use tracing;

use flashmind_types::tool::CommandAllowList;

// ---------------------------------------------------------------------------
// GlobAllowList
// ---------------------------------------------------------------------------

/// Glob-based command allow-list with permanent (config) and session patterns.
///
/// Implements [`CommandAllowList`] — no channels, no async. The CLI creates
/// one at startup, passes `Arc<GlobAllowList>` to the tool builder, and holds
/// a clone for adding session patterns after approval prompts.
pub struct GlobAllowList {
    permanent: Vec<Pattern>,
    session: Mutex<Vec<Pattern>>,
}

impl GlobAllowList {
    /// Create a new allow-list from config-defined pattern strings.
    pub fn new(allowed_patterns: Vec<String>) -> Self {
        let permanent = allowed_patterns
            .iter()
            .filter_map(|p| match Pattern::new(p) {
                Ok(pat) => Some(pat),
                Err(e) => {
                    tracing::warn!(pattern = %p, error = %e, "invalid allowed command pattern");
                    None
                }
            })
            .collect();

        Self {
            permanent,
            session: Mutex::new(Vec::new()),
        }
    }
}

impl CommandAllowList for GlobAllowList {
    fn is_allowed(&self, command: &str) -> bool {
        if self.permanent.iter().any(|p| p.matches(command)) {
            return true;
        }
        let session = self.session.lock().unwrap();
        session.iter().any(|p| p.matches(command))
    }

    fn add_session_pattern(&self, pattern: &str) {
        match Pattern::new(pattern) {
            Ok(pat) => {
                tracing::info!(pattern = %pattern, "added session-allowed command pattern");
                self.session.lock().unwrap().push(pat);
            }
            Err(e) => {
                tracing::warn!(pattern = %pattern, error = %e, "invalid session pattern");
            }
        }
    }
}

/// Derive a glob pattern from a command for "allow always (session)" mode.
///
/// For multi-command tools (git, docker, npm, kubectl, cargo), uses the first
/// two tokens + `*`. Otherwise uses the first token + `*`.
pub fn derive_session_pattern(command: &str) -> String {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let multi_command = [
        "git", "docker", "kubectl", "npm", "cargo", "npx", "yarn", "pnpm",
    ];

    match tokens.as_slice() {
        [] => "*".to_string(),
        [prog] => format!("{prog} *"),
        [prog, sub, ..] if multi_command.contains(prog) => format!("{prog} {sub} *"),
        [prog, ..] => format!("{prog} *"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_allowed_permanent() {
        let allowlist = GlobAllowList::new(vec!["cargo *".to_string()]);
        assert!(allowlist.is_allowed("cargo build"));
        assert!(allowlist.is_allowed("cargo test --release"));
        assert!(!allowlist.is_allowed("rm -rf /"));
    }

    #[test]
    fn test_is_allowed_session() {
        let allowlist = GlobAllowList::new(vec![]);
        assert!(!allowlist.is_allowed("git push origin main"));
        allowlist.add_session_pattern("git push *");
        assert!(allowlist.is_allowed("git push origin main"));
        assert!(allowlist.is_allowed("git push origin dev"));
    }

    #[test]
    fn test_derive_session_pattern() {
        assert_eq!(derive_session_pattern("git push origin main"), "git push *");
        assert_eq!(derive_session_pattern("docker build ."), "docker build *");
        assert_eq!(derive_session_pattern("python script.py"), "python *");
        assert_eq!(derive_session_pattern("make"), "make *");
    }

    #[test]
    fn test_invalid_pattern_ignored() {
        let allowlist = GlobAllowList::new(vec!["[invalid".to_string()]);
        assert!(!allowlist.is_allowed("[invalid"));
    }
}
