//! Semantic code navigation through installed stdio language servers.
//!
//! [`LspTool`] answers definition, reference, implementation, hover and
//! symbol queries and diagnostics. [`LspManager`] starts servers on demand
//! for each language and project, and stops them when idle.

mod client;
mod manager;
mod protocol;
mod servers;
mod tool;

use std::env;
use std::path::PathBuf;

pub use manager::LspManager;
pub use servers::LanguageServerConfig;
pub use tool::LspTool;

type AsyncMutex<T> = tokio::sync::Mutex<T>;

/// Find a `rust-analyzer` executable on `PATH`.
pub fn find_rust_analyzer() -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join("rust-analyzer"))
        .find(|candidate| candidate.is_file())
}
