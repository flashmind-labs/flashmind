//! Rust code navigation through rust-analyzer.
//!
//! [`LspTool`] answers definition, reference, implementation, hover and
//! symbol queries. [`LspManager`] starts one rust-analyzer per Cargo
//! workspace on first use and stops it when idle.

mod client;
mod manager;
mod protocol;
mod tool;

use std::env;
use std::path::PathBuf;

pub use manager::LspManager;
pub use tool::LspTool;

type AsyncMutex<T> = tokio::sync::Mutex<T>;

/// Find a `rust-analyzer` executable on `PATH`.
pub fn find_rust_analyzer() -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join("rust-analyzer"))
        .find(|candidate| candidate.is_file())
}
