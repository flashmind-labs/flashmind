//! Keeps one language server per workspace root and stops idle ones.

use std::collections::{BTreeMap, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::AsyncMutex;
use super::client::LspClient;
use super::servers::{LanguageServerConfig, builtins, executable};
use anyhow::{Context, bail};
use tokio_util::sync::CancellationToken;

const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const REAP_INTERVAL: Duration = Duration::from_secs(60);
const MAX_SERVERS: usize = 2;

#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct ServerKey {
    root: PathBuf,
    name: String,
}

pub(super) struct Selection {
    key: ServerKey,
    config: LanguageServerConfig,
}

type Servers = AsyncMutex<HashMap<ServerKey, Arc<LspClient>>>;

/// Starts language servers on demand and shares them between tool calls.
///
/// At most two servers run at once; starting a third stops the one used
/// least recently. A server unused for ten minutes is stopped.
pub struct LspManager {
    configs: BTreeMap<String, LanguageServerConfig>,
    servers: Arc<Servers>,
}

impl LspManager {
    /// Create a manager for a particular rust-analyzer executable.
    pub fn new(command: PathBuf) -> Self {
        let mut configs = builtins();
        configs.retain(|name, _| name == "rust");
        if let Some(config) = configs.get_mut("rust") {
            config.command = vec![command.to_string_lossy().into_owned()];
        }
        Self {
            configs,
            servers: Arc::default(),
        }
    }

    /// Discover built-in servers and apply named configuration overrides.
    pub fn with_servers(overrides: BTreeMap<String, LanguageServerConfig>) -> Self {
        let mut configs = builtins();
        for (name, mut config) in overrides {
            if let Some(default) = configs.get(&name) {
                if config.command.is_empty() {
                    config.command = default.command.clone();
                }
                if config.extensions.is_empty() {
                    config.extensions = default.extensions.clone();
                }
                if config.root_markers.is_empty() {
                    config.root_markers = default.root_markers.clone();
                }
            }
            configs.insert(name, config);
        }
        Self {
            configs,
            servers: Arc::default(),
        }
    }

    pub(super) async fn select(
        &self,
        file: Option<&Path>,
        cwd: Option<&Path>,
    ) -> anyhow::Result<Selection> {
        let start = file
            .and_then(Path::parent)
            .or(cwd)
            .context("pass a file or working directory")?;
        let mut candidates = Vec::new();
        for (name, config) in &self.configs {
            if config.disabled || file.is_some_and(|file| !config.supports(file)) {
                continue;
            }
            let project = start.ancestors().enumerate().find(|(_, directory)| {
                config
                    .root_markers
                    .iter()
                    .any(|marker| directory.join(marker).exists())
            });
            if let Some((distance, root)) = project {
                candidates.push((
                    distance,
                    Selection {
                        key: ServerKey {
                            root: root.to_path_buf(),
                            name: name.clone(),
                        },
                        config: config.clone(),
                    },
                ));
            } else if file.is_some() {
                candidates.push((
                    usize::MAX,
                    Selection {
                        key: ServerKey {
                            root: cwd
                                .filter(|cwd| start.starts_with(cwd))
                                .unwrap_or(start)
                                .to_path_buf(),
                            name: name.clone(),
                        },
                        config: config.clone(),
                    },
                ));
            }
        }
        candidates.sort_by_key(|(distance, _)| *distance);
        let mut selection = candidates
            .into_iter()
            .next()
            .map(|(_, selection)| selection)
            .context(
                "No language server matches this project. Pass a file or configure lsp servers.",
            )?;
        if selection.key.name == "rust"
            && let Some(root) = cwd
                .and_then(find_cargo_root)
                .filter(|_| file.is_some_and(is_dependency_source))
                .or_else(|| find_cargo_root(start))
        {
            selection.key.root = root;
        }
        let command = selection
            .config
            .command
            .first()
            .context("language server command is empty")?;
        if executable(command).is_none() {
            bail!("Language server `{command}` is not installed or is not on PATH.");
        }
        Ok(selection)
    }

    /// Return diagnostics after an edit when a matching server is installed.
    pub async fn diagnostics_after_edit(
        &self,
        path: &Path,
        cwd: Option<&Path>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Option<String>> {
        // Missing optional servers do not make an otherwise successful edit fail.
        if !self.configs.values().any(|config| {
            !config.disabled
                && config.supports(path)
                && config
                    .command
                    .first()
                    .is_some_and(|command| executable(command).is_some())
        }) {
            return Ok(None);
        }
        let selection = self.select(Some(path), cwd).await?;
        let client = self.client(&selection).await?;
        let diagnostics = client.diagnostics(path, cancel).await?;
        client.touch();
        Ok(Some(super::tool::render_diagnostics(
            path,
            &diagnostics,
            20,
        )))
    }

    /// Return the running server for `root`, starting one if needed.
    pub(super) async fn client(&self, selection: &Selection) -> anyhow::Result<Arc<LspClient>> {
        let key = &selection.key;
        let root = &key.root;
        let mut servers = self.servers.lock().await;
        if let Some(client) = servers.get(key) {
            if client.is_alive() {
                client.touch();
                return Ok(client.clone());
            }
            servers.remove(key);
        }

        if servers.len() >= MAX_SERVERS
            && let Some(oldest) = servers
                .iter()
                .max_by_key(|(_, client)| client.idle_for())
                .map(|(root, _)| root.clone())
            && let Some(evicted) = servers.remove(&oldest)
        {
            tracing::info!(root = %oldest.root.display(), "stopping least recently used language server");
            tokio::spawn(async move { evicted.shutdown().await });
        }

        tracing::info!(root = %root.display(), server = %key.name, "starting language server");
        let client = LspClient::start(&selection.config, root, key.name == "rust").await?;
        servers.insert(key.clone(), client.clone());
        tokio::spawn(reap_when_idle(
            Arc::downgrade(&self.servers),
            Arc::downgrade(&client),
            key.clone(),
        ));
        Ok(client)
    }
}

async fn reap_when_idle(servers: Weak<Servers>, client: Weak<LspClient>, key: ServerKey) {
    loop {
        tokio::time::sleep(REAP_INTERVAL).await;
        let (Some(servers), Some(client)) = (servers.upgrade(), client.upgrade()) else {
            return;
        };
        if client.is_alive() && client.idle_for() < IDLE_TIMEOUT {
            continue;
        }
        let mut map = servers.lock().await;
        if map
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, &client))
        {
            map.remove(&key);
        }
        drop(map);
        tracing::info!(root = %client.root().display(), "stopping idle language server");
        client.shutdown().await;
        return;
    }
}

/// Find the Cargo workspace that contains `start`.
///
/// Prefers the outermost ancestor whose `Cargo.toml` declares `[workspace]`,
/// and falls back to the nearest `Cargo.toml`.
pub(crate) fn find_cargo_root(start: &Path) -> Option<PathBuf> {
    let mut nearest = None;
    let mut workspace = None;
    for dir in start.ancestors() {
        let manifest = dir.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        if nearest.is_none() {
            nearest = Some(dir.to_path_buf());
        }
        if text.lines().any(|line| line.trim() == "[workspace]") {
            workspace = Some(dir.to_path_buf());
        }
    }
    workspace.or(nearest)
}

/// Whether `path` is dependency or toolchain source rather than a project.
///
/// Such files are served by the project's server, not a server of their own.
pub(crate) fn is_dependency_source(path: &Path) -> bool {
    let names: Vec<_> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => name.to_str(),
            _ => None,
        })
        .collect();
    names.windows(2).any(|pair| {
        matches!(
            pair,
            [".cargo", "registry" | "git"] | [".rustup", "toolchains"]
        )
    }) || ["CARGO_HOME", "RUSTUP_HOME"]
        .iter()
        .filter_map(std::env::var_os)
        .any(|home| path.starts_with(home))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_root_prefers_outer_workspace() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = [\"a\"]\n")
            .expect("write manifest");
        std::fs::create_dir_all(root.join("a/src")).expect("create member");
        std::fs::write(root.join("a/Cargo.toml"), "[package]\nname = \"a\"\n")
            .expect("write manifest");

        assert_eq!(
            find_cargo_root(&root.join("a/src")),
            Some(root.to_path_buf())
        );
    }

    #[test]
    fn cargo_root_falls_back_to_nearest_package() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let package = dir.path().join("pkg");
        std::fs::create_dir_all(package.join("src")).expect("create package");
        std::fs::write(package.join("Cargo.toml"), "[package]\nname = \"pkg\"\n")
            .expect("write manifest");

        assert_eq!(find_cargo_root(&package.join("src")), Some(package));
    }

    #[test]
    fn registry_and_toolchain_paths_are_dependencies() {
        assert!(is_dependency_source(Path::new(
            "/home/u/.cargo/registry/src/index/serde-1.0.0/src/lib.rs"
        )));
        assert!(is_dependency_source(Path::new(
            "/home/u/.rustup/toolchains/stable/lib/rustlib/src/rust/library/core/src/lib.rs"
        )));
        assert!(!is_dependency_source(Path::new("/srv/project/src/lib.rs")));
    }
}
