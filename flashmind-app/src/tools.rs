//! Tool building and registration.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use flashmind_memory::embeddings::create_embedding_provider;
use tokio::sync::RwLock;

use flashmind_core::AgentManager;
use flashmind_skills::{SkillRegistry, SkillRunner};
use flashmind_tools::ToolBuilder;
use flashmind_tools::mcp::McpDiskConfig;
use flashmind_tools::protected::ProtectedPaths;
use flashmind_tools::tool_sync::ToolSync;
use flashmind_types::tool::ToolRegistry;
use flashmind_types::{AgentLlmConfig, InjectQueue, LlmProvider};

use crate::config::AppConfig;

// ---------------------------------------------------------------------------
// ToolSet
// ---------------------------------------------------------------------------

/// Everything produced by [`AppConfig::build_tools`].
pub struct ToolSet {
    pub tools: ToolRegistry,
    pub tool_sync: ToolSync,
    pub memory: Option<(
        flashmind_memory::DbStore,
        Arc<dyn flashmind_memory::EmbeddingProvider>,
    )>,
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl AppConfig {
    /// Build the full tool registry from config.
    ///
    /// Memory tools are registered automatically when embedding is configured.
    pub async fn build_tools(
        &self,
        provider: Arc<dyn LlmProvider>,
        llm: &AgentLlmConfig,
    ) -> Result<ToolSet> {
        let protected = Arc::new(ProtectedPaths::new(&Self::base_dir()));
        let secrets = self.collect_secrets();

        // Subagent manager
        let inject_queue = InjectQueue::new();
        let manager = Arc::new(AgentManager::new(inject_queue.clone(), 8, 3));

        // MCP config
        let mcp_provider = McpDiskConfig::new(Self::mcp_dir());

        let builder = ToolBuilder::new()
            .file_ops(None, &protected)
            .bash(secrets, &protected)
            .search(self.tools.brave_api_key.clone(), self.tools.firecrawl_api_key.clone())
            .http()
            .time()
            .json()
            .models()
            .subagents(manager, provider.clone(), Some(llm.clone()))
            .mcp(mcp_provider, None);

        // Skills
        let skill_registry = Arc::new(RwLock::new(SkillRegistry::new(vec![Self::skills_dir()])));
        let skill_runner = Arc::new(SkillRunner::new(Duration::from_secs(300)));

        let (mut tools, tool_sync) = builder.build_with_sync().await;

        // Register skills tools
        tools.register(Arc::new(flashmind_skills::SkillListTool {
            registry: skill_registry.clone(),
        }));
        tools.register(Arc::new(flashmind_skills::SkillLoadTool {
            registry: skill_registry.clone(),
        }));
        tools.register(Arc::new(flashmind_skills::SkillRunTool {
            registry: skill_registry.clone(),
            runner: skill_runner,
        }));
        tools.register(Arc::new(flashmind_skills::SkillInstallTool {
            registry: skill_registry,
        }));

        // Memory tools
        let memory = self.build_memory_components().await;
        if let Some((ref db, ref embedder)) = memory {
            crate::memory::register_tools(&mut tools, db.clone(), embedder.clone());
        }

        Ok(ToolSet {
            tools,
            tool_sync,
            memory,
        })
    }

    /// Build an embedding provider from the memory config, if configured.
    pub fn build_embedder(&self) -> Option<Arc<dyn flashmind_memory::EmbeddingProvider>> {
        let mem = self.memory.as_ref()?;
        let emb_config = mem.embedding.as_ref()?;
        let fallback_key = self.llm.providers.iter().find_map(|p| p.api_key.as_deref());
        match create_embedding_provider(emb_config, fallback_key) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!("failed to create embedding provider: {e}");
                None
            }
        }
    }

    /// Build memory components (DbStore + EmbeddingProvider) if configured.
    ///
    /// Returns `None` if no embedding provider is configured or if the database
    /// cannot be opened. Callers can use these for memory tools and capture.
    pub async fn build_memory_components(
        &self,
    ) -> Option<(
        flashmind_memory::DbStore,
        Arc<dyn flashmind_memory::EmbeddingProvider>,
    )> {
        flashmind_memory::register_sqlite_vec();
        let embedder = self.build_embedder()?;
        let dim = embedder.dimensions();
        match flashmind_memory::DbStore::connect(&Self::db_path(), dim).await {
            Ok(db) => Some((db, embedder)),
            Err(e) => {
                tracing::warn!("memory db not available: {e}");
                None
            }
        }
    }
}
