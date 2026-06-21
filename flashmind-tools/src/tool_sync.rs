//! Simplified handle for syncing dynamic tool changes with a [`ToolRegistry`].

use std::sync::Arc;

use flashmind_types::{Tool, tool::ToolRegistry};
#[cfg(feature = "subagent")]
use tokio::sync::RwLock;

use crate::builder::PendingTools;

// ---------------------------------------------------------------------------

/// Handle for syncing dynamic tool changes with a [`ToolRegistry`].
///
/// Created by [`ToolBuilder::build_with_sync()`](crate::builder::ToolBuilder::build_with_sync).
/// Call [`sync()`](Self::sync) after each agent turn to apply any pending
/// tool registrations or removals (from MCP servers, OAuth flows, etc.).
///
/// # Example
///
/// ```rust,ignore
/// let (tools, sync) = ToolBuilder::new()
///     .mcp(config, None)
///     .google(&google_config, false)
///     .build_with_sync()
///     .await;
///
/// let mut agent = Agent::builder(provider).tools(tools).build();
///
/// loop {
///     let stream = agent.start(&mut conv, input, None);
///     repl.stream_response(stream).await?;
///     sync.sync(agent.tools_mut());
/// }
///
/// sync.shutdown().await;
/// ```
pub struct ToolSync {
    #[cfg(feature = "mcp")]
    mcp_registry: Option<crate::mcp::McpRegistry>,
    pending_tools: PendingTools,
    #[cfg(feature = "subagent")]
    delegate_tools: Option<Arc<RwLock<ToolRegistry>>>,
}

impl ToolSync {
    pub(crate) fn new(
        #[cfg(feature = "mcp")] mcp_registry: Option<crate::mcp::McpRegistry>,
        pending_tools: PendingTools,
        #[cfg(feature = "subagent")] delegate_tools: Option<Arc<RwLock<ToolRegistry>>>,
    ) -> Self {
        Self {
            #[cfg(feature = "mcp")]
            mcp_registry,
            pending_tools,
            #[cfg(feature = "subagent")]
            delegate_tools,
        }
    }

    /// Apply pending tool registrations and removals (MCP + OAuth).
    pub async fn sync(&self, tools: &mut ToolRegistry) {
        // Drain MCP ops — the proxy `mcp` tool reads from
        // `McpRegistry::current_mcp_tools()` dynamically, so we just
        // drain the queue to keep it from growing unbounded.
        #[cfg(feature = "mcp")]
        if let Some(registry) = &self.mcp_registry {
            let _ = registry.drain_pending_ops();
        }

        // Drain OAuth / auth tool registrations
        let pending: Vec<Arc<dyn Tool>> = std::mem::take(&mut *self.pending_tools.lock().unwrap());
        for tool in pending {
            tools.register(tool);
        }

        // Snapshot current registry for delegate tool's child agents.
        #[cfg(feature = "subagent")]
        if let Some(ref delegate_tools) = self.delegate_tools {
            let count = tools.list().len();
            *delegate_tools.write().await = tools.clone();
            tracing::debug!(tool_count = count, "snapshotted tools for delegate");
        }
    }

    /// Shut down all MCP server connections (no-op if MCP is not enabled).
    pub async fn shutdown(&self) {
        #[cfg(feature = "mcp")]
        if let Some(registry) = &self.mcp_registry {
            registry.shutdown_all().await;
        }
    }

    /// Access the underlying MCP registry, if MCP is enabled and was configured.
    #[cfg(feature = "mcp")]
    pub fn mcp_registry(&self) -> Option<&crate::mcp::McpRegistry> {
        self.mcp_registry.as_ref()
    }

    /// Access the pending tools queue (for auth tools to push into).
    pub fn pending_tools(&self) -> &PendingTools {
        &self.pending_tools
    }
}
