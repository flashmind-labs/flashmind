//! CLI configuration — re-exports shared config from flashmind-app
//! and adds CLI-specific extensions.

pub use flashmind_app::config::*;
pub use flashmind_app::tools::ToolSet;

/// Alias for backward compatibility within the CLI.
pub type Config = AppConfig;

// ---------------------------------------------------------------------------
// CLI-specific extensions
// ---------------------------------------------------------------------------

use std::sync::Arc;

use anyhow::{Context, Result};

use flashmind_types::{AgentLlmConfig, LlmProvider};

/// Build tools with memory registration (CLI-specific).
///
/// Calls [`AppConfig::build_tools`] and then registers memory tools
/// if an embedding provider is configured.
pub async fn build_tools_with_memory(
    config: &AppConfig,
    provider: Arc<dyn LlmProvider>,
    llm: &AgentLlmConfig,
) -> Result<ToolSet> {
    let mut tool_set = config.build_tools(provider, llm).await?;

    if let Some(embedder) = config.build_embedder() {
        let dim = embedder.dimensions();
        let db = flashmind_memory::DbStore::connect(&AppConfig::db_path(), dim)
            .await
            .context("opening memory database")?;
        crate::memory::register_tools(&mut tool_set.tools, db, embedder);
    }

    Ok(tool_set)
}
