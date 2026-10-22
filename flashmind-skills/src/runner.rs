//! Skill command execution with environment loading, timeout, and secret redaction.

use std::io::ErrorKind;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use flashmind_types::MIN_SECRET_REDACT_LEN;

use crate::error::SkillError;
use crate::skill::Skill;

/// Output from executing a skill command.
#[derive(Debug, Clone)]
pub struct SkillOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Executes commands in a skill's directory with environment and timeout handling.
pub struct SkillRunner {
    timeout: Duration,
}

impl SkillRunner {
    /// Create a runner with the given execution timeout.
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    /// Execute a shell command in the skill's directory.
    ///
    /// Loads `.env` from the skill directory, prepends the skill dir to `PATH`,
    /// applies a timeout, and redacts secret values from output.
    pub async fn run(&self, skill: &Skill, command: &str) -> Result<SkillOutput, SkillError> {
        let env_path = skill.dir.join(".env");
        let env_vars = if env_path.exists() {
            let content = tokio::fs::read_to_string(&env_path).await?;
            parse_env_file(&content)
        } else {
            Vec::new()
        };

        let secrets: Vec<&str> = env_vars
            .iter()
            .map(|(_, v)| v.as_str())
            .filter(|v| v.len() >= MIN_SECRET_REDACT_LEN)
            .collect();

        let current_path = std::env::var_os("PATH").unwrap_or_default();
        let new_path = std::env::join_paths(
            std::iter::once(skill.dir.clone()).chain(std::env::split_paths(&current_path)),
        )
        .map_err(|e| SkillError::ExecutionFailed(format!("invalid PATH entry: {e}")))?;

        let spawn = |shell| {
            let mut cmd = Command::new(shell);
            cmd.arg("-c")
                .arg(command)
                .current_dir(&skill.dir)
                .env("PATH", &new_path)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            for (key, value) in &env_vars {
                cmd.env(key, value);
            }
            cmd.spawn()
        };
        let child = spawn("bash")
            .or_else(|error| {
                if error.kind() == ErrorKind::NotFound {
                    spawn("sh")
                } else {
                    Err(error)
                }
            })
            .map_err(|error| SkillError::ExecutionFailed(error.to_string()))?;

        let result = tokio::time::timeout(self.timeout, child.wait_with_output()).await;

        match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();

                redact_secrets(&mut stdout, &secrets);
                redact_secrets(&mut stderr, &secrets);

                Ok(SkillOutput {
                    stdout,
                    stderr,
                    exit_code: output.status.code().unwrap_or(-1),
                })
            }
            Ok(Err(e)) => Err(SkillError::ExecutionFailed(e.to_string())),
            Err(_) => Err(SkillError::Timeout(self.timeout.as_secs())),
        }
    }
}

// ---------------------------------------------------------------------------
// .env parsing
// ---------------------------------------------------------------------------

fn parse_env_file(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                return None;
            }
            let (key, raw_value) = trimmed.split_once('=')?;
            let key = key.trim().to_string();
            let value = unquote(raw_value.trim());
            Some((key, value))
        })
        .collect()
}

fn unquote(s: &str) -> String {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn redact_secrets(text: &mut String, secrets: &[&str]) {
    for secret in secrets {
        if text.contains(secret) {
            *text = text.replace(secret, "[REDACTED]");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_basic() {
        let content = r#"
# A comment
FOO=bar
BAZ="quoted value"
SINGLE='single quoted'
EMPTY=

"#;
        let vars = parse_env_file(content);
        assert_eq!(vars.len(), 4);
        assert_eq!(vars[0], ("FOO".into(), "bar".into()));
        assert_eq!(vars[1], ("BAZ".into(), "quoted value".into()));
        assert_eq!(vars[2], ("SINGLE".into(), "single quoted".into()));
        assert_eq!(vars[3], ("EMPTY".into(), "".into()));
    }

    #[test]
    fn redact_secrets_from_output() {
        let mut text = "Token is sk-abc123xyz and key is mykey".to_string();
        redact_secrets(&mut text, &["sk-abc123xyz", "mykey"]);
        assert_eq!(text, "Token is [REDACTED] and key is [REDACTED]");
    }

    #[test]
    fn short_secrets_not_redacted() {
        let env = "X=ab\n";
        let vars = parse_env_file(env);
        let secrets: Vec<&str> = vars
            .iter()
            .map(|(_, v)| v.as_str())
            .filter(|v| v.len() >= MIN_SECRET_REDACT_LEN)
            .collect();
        assert!(secrets.is_empty());
    }

    #[tokio::test]
    async fn run_echo_command() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("SKILL.md"), "---\nname: test\n---\nTest.").unwrap();

        let skill =
            crate::parser::parse_skill_md("---\nname: test\n---\nTest.", tmp.path()).unwrap();

        let runner = SkillRunner::new(Duration::from_secs(10));
        let output = runner.run(&skill, "echo hello").await.unwrap();

        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stdout.trim(), "hello");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_without_bash_preserves_skill_environment_and_output() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        symlink("/bin/sh", tmp.path().join("sh")).unwrap();
        std::fs::write(
            tmp.path().join(".env"),
            format!("PATH={}\nTOKEN=example-secret\n", tmp.path().display()),
        )
        .unwrap();
        let skill =
            crate::parser::parse_skill_md("---\nname: test\n---\nTest.", tmp.path()).unwrap();
        let output = SkillRunner::new(Duration::from_secs(10))
            .run(
                &skill,
                "printf '%s' \"$TOKEN\"; printf error >&2; printf marker > marker; exit 7",
            )
            .await
            .unwrap();

        assert_eq!(output.exit_code, 7);
        assert_eq!(output.stdout, "[REDACTED]");
        assert_eq!(output.stderr, "error");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("marker")).unwrap(),
            "marker"
        );
    }
}
