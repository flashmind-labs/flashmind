//! Keeps one language server per workspace root and stops idle ones.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::AsyncMutex;
use super::client::LspClient;

const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const REAP_INTERVAL: Duration = Duration::from_secs(60);
const MAX_SERVERS: usize = 2;

type Servers = AsyncMutex<HashMap<PathBuf, Arc<LspClient>>>;

/// Starts language servers on demand and shares them between tool calls.
///
/// At most two servers run at once; starting a third stops the one used
/// least recently. A server unused for ten minutes is stopped.
pub struct LspManager {
    command: PathBuf,
    servers: Arc<Servers>,
}

impl LspManager {
    /// Create a manager that starts servers with `command` (a rust-analyzer binary).
    pub fn new(command: PathBuf) -> Self {
        Self {
            command,
            servers: Arc::default(),
        }
    }

    /// Return the running server for `root`, starting one if needed.
    pub(crate) async fn client(&self, root: &Path) -> anyhow::Result<Arc<LspClient>> {
        let mut servers = self.servers.lock().await;
        if let Some(client) = servers.get(root) {
            if client.is_alive() {
                client.touch();
                return Ok(client.clone());
            }
            servers.remove(root);
        }

        if servers.len() >= MAX_SERVERS
            && let Some(oldest) = servers
                .iter()
                .max_by_key(|(_, client)| client.idle_for())
                .map(|(root, _)| root.clone())
            && let Some(evicted) = servers.remove(&oldest)
        {
            tracing::info!(root = %oldest.display(), "stopping least recently used rust-analyzer");
            tokio::spawn(async move { evicted.shutdown().await });
        }

        tracing::info!(root = %root.display(), "starting rust-analyzer");
        let client = LspClient::start(&self.command, root).await?;
        servers.insert(root.to_path_buf(), client.clone());
        tokio::spawn(reap_when_idle(
            Arc::downgrade(&self.servers),
            Arc::downgrade(&client),
        ));
        Ok(client)
    }

    /// Root of the server used most recently, if any is running.
    pub(crate) async fn most_recent_root(&self) -> Option<PathBuf> {
        let servers = self.servers.lock().await;
        servers
            .values()
            .filter(|client| client.is_alive())
            .min_by_key(|client| client.idle_for())
            .map(|client| client.root().to_path_buf())
    }
}

async fn reap_when_idle(servers: Weak<Servers>, client: Weak<LspClient>) {
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
            .get(client.root())
            .is_some_and(|current| Arc::ptr_eq(current, &client))
        {
            map.remove(client.root());
        }
        drop(map);
        tracing::info!(root = %client.root().display(), "stopping idle rust-analyzer");
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
