use std::ffi::OsStr;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::builder::{PossibleValue, TypedValueParser};
use clap::error::ErrorKind;
use clap::{Arg, Command, CommandFactory, Error, ValueEnum};
use clap_complete::Shell;
use flashmind_types::{AudioFormat, Provider, ReasoningLevel};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum CompletionTarget {
    Install,
    Bash,
    Elvish,
    Fish,
    Powershell,
    Zsh,
}

impl CompletionTarget {
    pub(crate) fn shell(self) -> Option<Shell> {
        match self {
            Self::Install => None,
            Self::Bash => Some(Shell::Bash),
            Self::Elvish => Some(Shell::Elvish),
            Self::Fish => Some(Shell::Fish),
            Self::Powershell => Some(Shell::PowerShell),
            Self::Zsh => Some(Shell::Zsh),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum InstallShell {
    Bash,
    Fish,
    Zsh,
}

impl InstallShell {
    fn detect(value: &OsStr) -> Result<Self> {
        match Path::new(value).file_name().and_then(OsStr::to_str) {
            Some("bash") => Ok(Self::Bash),
            Some("fish") => Ok(Self::Fish),
            Some("zsh") => Ok(Self::Zsh),
            _ => bail!("cannot detect a supported shell; pass --shell bash, fish, or zsh"),
        }
    }

    fn snippet(self) -> &'static str {
        match self {
            Self::Bash => {
                "if command -v flsh >/dev/null 2>&1; then\n  eval \"$(flsh completions bash)\"\nfi\n"
            }
            Self::Zsh => {
                "if command -v flsh >/dev/null 2>&1; then\n  if (( ! $+functions[compdef] )); then\n    autoload -Uz compinit\n    compinit\n  fi\n  eval \"$(flsh completions zsh)\"\nfi\n"
            }
            Self::Fish => {
                "if status is-interactive; and type -q flsh\n  flsh completions fish | source\nend\n"
            }
        }
    }

    fn completion_command(self) -> &'static str {
        match self {
            Self::Bash => "flsh completions bash",
            Self::Fish => "flsh completions fish",
            Self::Zsh => "flsh completions zsh",
        }
    }
}

pub(crate) async fn install(shell: Option<InstallShell>) -> Result<()> {
    let shell = match shell {
        Some(shell) => shell,
        None => InstallShell::detect(
            &std::env::var_os("SHELL")
                .context("SHELL is not set; pass --shell bash, fish, or zsh")?,
        )?,
    };
    let path = crate::paths::shell_startup_file(shell)?;
    let installed = install_into(shell, &path).await?;
    if installed {
        println!(
            "Installed completion in {}. Start a new shell or source the file.",
            path.display()
        );
    } else {
        println!("Completion is already configured in {}.", path.display());
    }
    Ok(())
}

async fn install_into(shell: InstallShell, path: &Path) -> Result<bool> {
    let existing = match fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error).with_context(|| format!("cannot read {}", path.display())),
    };
    if existing.lines().any(|line| {
        !line.trim_start().starts_with('#') && line.contains(shell.completion_command())
    }) {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .with_context(|| format!("cannot open {}", path.display()))?;
    let separator = if !existing.is_empty() && !existing.ends_with('\n') {
        "\n\n"
    } else {
        "\n"
    };
    let snippet = format!("{separator}# flsh shell completion\n{}", shell.snippet());
    file.write_all(snippet.as_bytes())
        .await
        .with_context(|| format!("cannot update {}", path.display()))?;
    Ok(true)
}

#[derive(Clone)]
pub(crate) struct Suggestions<T> {
    values: &'static [&'static str],
    marker: PhantomData<T>,
}

impl<T> Suggestions<T> {
    const fn new(values: &'static [&'static str]) -> Self {
        Self {
            values,
            marker: PhantomData,
        }
    }
}

impl<T> TypedValueParser for Suggestions<T>
where
    T: FromStr + Clone + Send + Sync + 'static,
    T::Err: std::fmt::Display,
{
    type Value = T;

    fn parse_ref(&self, _command: &Command, _arg: Option<&Arg>, value: &OsStr) -> Result<T, Error> {
        let value = value
            .to_str()
            .ok_or_else(|| Error::raw(ErrorKind::InvalidUtf8, "argument must be valid UTF-8"))?;
        value
            .parse::<T>()
            .map_err(|error| Error::raw(ErrorKind::InvalidValue, error.to_string()))
    }

    fn possible_values(&self) -> Option<Box<dyn Iterator<Item = PossibleValue> + '_>> {
        Some(Box::new(
            self.values.iter().copied().map(PossibleValue::new),
        ))
    }
}

const PROVIDERS: &[&str] = &[
    "ollama",
    "openrouter",
    "anthropic",
    "openai",
    "mistral",
    "elevenlabs",
    "venice",
    "groq",
    "together",
    "fireworks",
    "deepseek",
    "moonshot",
    "zai",
];

pub(crate) fn providers() -> Suggestions<Provider> {
    Suggestions::new(PROVIDERS)
}

pub(crate) fn generate(shell: Shell, mut output: impl Write) -> io::Result<()> {
    clap_complete::generate(shell, &mut crate::Cli::command(), "flsh", &mut output);
    if shell == Shell::Fish {
        writeln!(
            output,
            "complete -c flsh -n '__fish_flsh_using_subcommand models voices' -f -a '{}'",
            PROVIDERS.join(" ")
        )?;
    }
    Ok(())
}

pub(crate) fn audio_formats() -> Suggestions<AudioFormat> {
    Suggestions::new(&["mp3", "wav", "opus", "aac", "flac"])
}

pub(crate) fn reasoning_levels() -> Suggestions<ReasoningLevel> {
    Suggestions::new(&["off", "low", "medium", "high"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_preserves_content_and_is_idempotent() {
        for shell in [InstallShell::Bash, InstallShell::Fish, InstallShell::Zsh] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("config").join("startup");
            fs::create_dir_all(path.parent().unwrap()).await.unwrap();
            fs::write(&path, "existing setting").await.unwrap();
            assert!(install_into(shell, &path).await.unwrap());
            let contents = fs::read_to_string(&path).await.unwrap();
            assert!(contents.starts_with("existing setting\n"));
            assert!(contents.contains(shell.completion_command()));
            assert!(!install_into(shell, &path).await.unwrap());
            assert_eq!(fs::read_to_string(&path).await.unwrap(), contents);
        }
    }

    #[tokio::test]
    async fn existing_completion_is_not_added_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("startup");
        let contents = "eval \"$(flsh completions zsh)\"\n";
        fs::write(&path, contents).await.unwrap();
        assert!(!install_into(InstallShell::Zsh, &path).await.unwrap());
        assert_eq!(fs::read_to_string(path).await.unwrap(), contents);
    }

    #[test]
    fn install_snippets_have_valid_shell_syntax() {
        use std::process::{Command as ProcessCommand, Stdio};

        for (binary, shell) in [
            ("bash", InstallShell::Bash),
            ("zsh", InstallShell::Zsh),
            ("fish", InstallShell::Fish),
        ] {
            let mut child = match ProcessCommand::new(binary)
                .arg("-n")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(child) => child,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => panic!("cannot run {binary}: {error}"),
            };
            std::io::Write::write_all(&mut child.stdin.take().unwrap(), shell.snippet().as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{binary}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn detects_supported_shell_names() {
        assert!(matches!(
            InstallShell::detect(OsStr::new("/bin/zsh")).unwrap(),
            InstallShell::Zsh
        ));
        assert!(matches!(
            InstallShell::detect(OsStr::new("/bin/bash")).unwrap(),
            InstallShell::Bash
        ));
        assert!(matches!(
            InstallShell::detect(OsStr::new("/bin/fish")).unwrap(),
            InstallShell::Fish
        ));
        assert!(InstallShell::detect(OsStr::new("/bin/sh")).is_err());
    }
}
