//! Where `flsh` keeps its files.
//!
//! Resolution order:
//!
//! 1. `$FLASHMIND_HOME`: everything lives in that one directory.
//! 2. `~/.flashmind`, if it already exists: the legacy single-directory layout.
//! 3. Platform defaults, split by kind:
//!    - Unix (including macOS): `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME` and
//!      `$XDG_STATE_HOME`, falling back to `~/.config`, `~/.local/share` and
//!      `~/.local/state`.
//!    - Windows: config in `%APPDATA%`, everything else in `%LOCALAPPDATA%`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const APP: &str = "flashmind";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// User-edited settings: `config.toml`, `mcp/`, `agents/`, `skills/`.
    Config,
    /// Data worth keeping: `memory.db`, `sessions/`.
    Data,
    /// Disposable history: `logs/`, display logs, REPL history.
    State,
}

/// Directory for user-edited settings.
pub fn config_dir() -> Result<PathBuf> {
    ensure(Kind::Config)
}

/// Directory for persistent data such as the memory database and sessions.
pub fn data_dir() -> Result<PathBuf> {
    ensure(Kind::Data)
}

/// Directory for logs and history.
pub fn state_dir() -> Result<PathBuf> {
    ensure(Kind::State)
}

fn ensure(kind: Kind) -> Result<PathBuf> {
    let home = dirs::home_dir();
    let legacy_exists = home
        .as_deref()
        .is_some_and(|h| h.join(".flashmind").is_dir());
    let dir = resolve(
        kind,
        |k| std::env::var_os(k),
        home.as_deref(),
        legacy_exists,
    )
    .context("cannot determine home directory")?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory {}", dir.display()))?;
    Ok(dir)
}

fn resolve(
    kind: Kind,
    env: impl Fn(&str) -> Option<OsString>,
    home: Option<&Path>,
    legacy_exists: bool,
) -> Option<PathBuf> {
    // XDG says relative paths are invalid and must be ignored; apply the same
    // rule to our own override.
    let abs_env = |key: &str| env(key).map(PathBuf::from).filter(|p| p.is_absolute());

    if let Some(dir) = abs_env("FLASHMIND_HOME") {
        return Some(dir);
    }
    if legacy_exists {
        return home.map(|h| h.join(".flashmind"));
    }

    if cfg!(windows) {
        let base = match kind {
            Kind::Config => dirs::config_dir()?,
            Kind::Data | Kind::State => dirs::data_local_dir()?,
        };
        let dir = base.join(APP);
        return Some(match kind {
            Kind::Config | Kind::Data => dir,
            Kind::State => dir.join("state"),
        });
    }

    let (var, fallback) = match kind {
        Kind::Config => ("XDG_CONFIG_HOME", ".config"),
        Kind::Data => ("XDG_DATA_HOME", ".local/share"),
        Kind::State => ("XDG_STATE_HOME", ".local/state"),
    };
    let base = abs_env(var).or_else(|| home.map(|h| h.join(fallback)))?;
    Some(base.join(APP))
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
    fn no_home_without_xdg_is_none() {
        assert!(resolve(Kind::Config, env_of(&[]), None, false).is_none());
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
