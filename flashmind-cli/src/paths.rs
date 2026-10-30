//! Where `flsh` keeps its files.
//!
//! Resolution order:
//!
//! 1. `$FLASHMIND_HOME`: everything lives in that one directory. It must be an
//!    absolute path; a relative one is an error rather than being ignored, so
//!    a typo can't silently fall through to the real config.
//! 2. `~/.flashmind`, if it existed when first checked: the legacy
//!    single-directory layout.
//! 3. Platform defaults, split by kind:
//!    - Unix (including macOS): `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME` and
//!      `$XDG_STATE_HOME`, falling back to `~/.config`, `~/.local/share` and
//!      `~/.local/state`.
//!    - Windows: config in `%APPDATA%`, everything else in `%LOCALAPPDATA%`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};

const APP: &str = "flashmind";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// User-edited settings: `config.toml`, `agents/`, `skills/`.
    Config,
    /// Data worth keeping: `sessions/`, including display logs.
    Data,
    /// Disposable history: `logs/`, REPL history.
    State,
}

/// Directory for user-edited settings.
pub fn config_dir() -> Result<PathBuf> {
    ensure(Kind::Config)
}

/// Directories containing `.<skill-name>-always` flags.
pub fn skill_flag_dirs() -> Result<Vec<PathBuf>> {
    let mut dirs = vec![config_dir()?];
    if let Some(dir) = claude_config_dir()
        && !dirs.contains(&dir)
    {
        dirs.push(dir);
    }
    Ok(dirs)
}

pub(crate) fn claude_config_dir() -> Option<PathBuf> {
    std::env::var_os("CLAUDE_CONFIG_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".claude")))
}

/// Import order: user definitions, then project ancestors, nearest last.
pub(crate) fn markdown_agent_dirs() -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    if let Some(claude) = claude_config_dir() {
        dirs.push(claude.join("agents"));
    }
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".agents"));
    }
    let cwd = std::env::current_dir().context("cannot determine working directory")?;
    for root in cwd.ancestors().collect::<Vec<_>>().into_iter().rev() {
        for relative in [".claude/agents", ".agents"] {
            let path = root.join(relative);
            if !dirs.contains(&path) {
                dirs.push(path);
            }
        }
    }
    Ok(dirs)
}

/// Directory for persistent data such as the memory database and sessions.
pub fn data_dir() -> Result<PathBuf> {
    ensure(Kind::Data)
}

/// Directory for logs and history.
pub fn state_dir() -> Result<PathBuf> {
    ensure(Kind::State)
}

pub fn display_logs_dir() -> Result<PathBuf> {
    let dir = data_dir()?.join("sessions").join("display");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory {}", dir.display()))?;
    Ok(dir)
}

pub fn migrate_display_logs(destination: &Path) -> Result<()> {
    move_display_logs(&state_dir()?, destination)
}

fn move_display_logs(source: &Path, destination: &Path) -> Result<()> {
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if !entry.file_type()?.is_file()
            || !name
                .to_str()
                .is_some_and(|name| name.starts_with("display-") && name.ends_with(".jsonl"))
        {
            continue;
        }
        let target = destination.join(&name);
        if target.exists() {
            tracing::warn!(
                "keeping legacy display log {}: destination exists",
                entry.path().display()
            );
            continue;
        }
        std::fs::rename(entry.path(), &target)
            .with_context(|| format!("cannot move display log to {}", target.display()))?;
    }
    Ok(())
}

/// Whether `~/.flashmind` exists, checked once per process so the layout
/// can't change mid-session if something creates that directory.
fn legacy_exists() -> bool {
    static LEGACY: OnceLock<bool> = OnceLock::new();
    *LEGACY.get_or_init(|| dirs::home_dir().is_some_and(|h| h.join(".flashmind").is_dir()))
}

fn ensure(kind: Kind) -> Result<PathBuf> {
    let home = dirs::home_dir();
    let dir = resolve(
        kind,
        |k| std::env::var_os(k),
        home.as_deref(),
        legacy_exists(),
    )?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory {}", dir.display()))?;
    Ok(dir)
}

fn resolve(
    kind: Kind,
    env: impl Fn(&str) -> Option<OsString>,
    home: Option<&Path>,
    legacy_exists: bool,
) -> Result<PathBuf> {
    if let Some(dir) = env("FLASHMIND_HOME").filter(|v| !v.is_empty()) {
        let dir = PathBuf::from(dir);
        if !dir.is_absolute() {
            bail!(
                "FLASHMIND_HOME must be an absolute path, got {}",
                dir.display()
            );
        }
        return Ok(dir);
    }
    let no_home = || anyhow!("cannot determine home directory");
    if legacy_exists {
        return home.map(|h| h.join(".flashmind")).ok_or_else(no_home);
    }

    if cfg!(windows) {
        let base = match kind {
            Kind::Config => dirs::config_dir().context("cannot determine %APPDATA%")?,
            Kind::Data | Kind::State => {
                dirs::data_local_dir().context("cannot determine %LOCALAPPDATA%")?
            }
        };
        let dir = base.join(APP);
        return Ok(match kind {
            Kind::Config | Kind::Data => dir,
            Kind::State => dir.join("state"),
        });
    }

    let (var, fallback) = match kind {
        Kind::Config => ("XDG_CONFIG_HOME", ".config"),
        Kind::Data => ("XDG_DATA_HOME", ".local/share"),
        Kind::State => ("XDG_STATE_HOME", ".local/state"),
    };
    // XDG says relative paths are invalid and must be ignored.
    let base = env(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| home.map(|h| h.join(fallback)))
        .ok_or_else(no_home)?;
    Ok(base.join(APP))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<OsString> {
        move |k| {
            pairs
                .iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| OsString::from(v))
        }
    }

    const HOME: &str = "/home/u";

    #[test]
    fn display_logs_move_without_changing_contents_or_other_files() {
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let name = "display-session_example.jsonl";
        let contents = "{\"type\":\"user\",\"text\":\"hello\"}\n";
        std::fs::write(source.path().join(name), contents).unwrap();
        std::fs::write(source.path().join("config.toml"), "").unwrap();

        move_display_logs(source.path(), destination.path()).unwrap();
        move_display_logs(source.path(), destination.path()).unwrap();

        assert!(!source.path().join(name).exists());
        assert_eq!(
            std::fs::read_to_string(destination.path().join(name)).unwrap(),
            contents
        );
        assert!(source.path().join("config.toml").exists());
    }

    #[test]
    fn display_log_migration_preserves_both_files_on_conflict() {
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let name = "display-session_example.jsonl";
        std::fs::write(source.path().join(name), "legacy").unwrap();
        std::fs::write(destination.path().join(name), "current").unwrap();

        move_display_logs(source.path(), destination.path()).unwrap();

        assert_eq!(
            std::fs::read_to_string(source.path().join(name)).unwrap(),
            "legacy"
        );
        assert_eq!(
            std::fs::read_to_string(destination.path().join(name)).unwrap(),
            "current"
        );
    }

    #[test]
    fn xdg_defaults_under_home() {
        let home = Some(Path::new(HOME));
        let env = env_of(&[]);
        assert_eq!(
            resolve(Kind::Config, &env, home, false).unwrap(),
            PathBuf::from("/home/u/.config/flashmind")
        );
        assert_eq!(
            resolve(Kind::Data, &env, home, false).unwrap(),
            PathBuf::from("/home/u/.local/share/flashmind")
        );
        assert_eq!(
            resolve(Kind::State, &env, home, false).unwrap(),
            PathBuf::from("/home/u/.local/state/flashmind")
        );
    }

    #[test]
    fn xdg_vars_respected_and_relative_ignored() {
        let env = env_of(&[
            ("XDG_CONFIG_HOME", "/xdg/conf"),
            ("XDG_DATA_HOME", "rel/data"),
        ]);
        let home = Some(Path::new(HOME));
        assert_eq!(
            resolve(Kind::Config, &env, home, false).unwrap(),
            PathBuf::from("/xdg/conf/flashmind")
        );
        assert_eq!(
            resolve(Kind::Data, &env, home, false).unwrap(),
            PathBuf::from("/home/u/.local/share/flashmind")
        );
    }

    #[test]
    fn legacy_dir_wins_over_xdg() {
        let env = env_of(&[("XDG_CONFIG_HOME", "/xdg/conf")]);
        let home = Some(Path::new(HOME));
        for kind in [Kind::Config, Kind::Data, Kind::State] {
            assert_eq!(
                resolve(kind, &env, home, true).unwrap(),
                PathBuf::from("/home/u/.flashmind")
            );
        }
    }

    #[test]
    fn flashmind_home_wins_over_everything() {
        let env = env_of(&[("FLASHMIND_HOME", "/fm"), ("XDG_CONFIG_HOME", "/xdg/conf")]);
        let home = Some(Path::new(HOME));
        for kind in [Kind::Config, Kind::Data, Kind::State] {
            assert_eq!(
                resolve(kind, &env, home, true).unwrap(),
                PathBuf::from("/fm")
            );
        }
    }

    #[test]
    fn relative_flashmind_home_is_an_error() {
        let env = env_of(&[("FLASHMIND_HOME", "flashmind-test")]);
        let err = resolve(Kind::Config, env, Some(Path::new(HOME)), false).unwrap_err();
        assert!(err.to_string().contains("absolute"), "{err}");
    }

    #[test]
    fn empty_flashmind_home_is_unset() {
        let env = env_of(&[("FLASHMIND_HOME", "")]);
        assert_eq!(
            resolve(Kind::Config, env, Some(Path::new(HOME)), false).unwrap(),
            PathBuf::from("/home/u/.config/flashmind")
        );
    }

    #[test]
    fn no_home_without_xdg_is_an_error() {
        assert!(resolve(Kind::Config, env_of(&[]), None, false).is_err());
        assert_eq!(
            resolve(
                Kind::Config,
                env_of(&[("XDG_CONFIG_HOME", "/x")]),
                None,
                false
            )
            .unwrap(),
            PathBuf::from("/x/flashmind")
        );
    }
}
