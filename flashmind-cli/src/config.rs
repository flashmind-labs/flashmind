//! CLI configuration — re-exports shared config from flashmind-app
//! and adds CLI-specific extensions.

pub use flashmind_app::config::*;

/// Alias for backward compatibility within the CLI.
pub type Config = AppConfig;
