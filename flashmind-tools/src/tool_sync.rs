//! Simplified handle for syncing dynamic tool changes with a [`ToolRegistry`].

use std::sync::Arc;

use flashmind_types::{Tool, tool::ToolRegistry};

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
}

impl ToolSync {
    pub(crate) fn new(
        #[cfg(feature = "mcp")] mcp_registry: Option<crate::mcp::McpRegistry>,
        pending_tools: PendingTools,
    ) -> Self {
        Self {
            #[cfg(feature = "mcp")]
            mcp_registry,
            pending_tools,
        }
    }

    /// Apply pending tool registrations and removals (MCP + OAuth).
    pub fn sync(&self, tools: &mut ToolRegistry) {
        // Drain MCP ops
        #[cfg(feature = "mcp")]
        if let Some(registry) = &self.mcp_registry {
            use crate::mcp::tools::make_mcp_tool_wrappers;
            use crate::mcp::types::McpToolOp;

            for op in registry.drain_pending_ops() {
                match op {
                    McpToolOp::Register {
                        server_name,
                        tool_defs,
                    } => {
                        let wrappers = make_mcp_tool_wrappers(registry, &server_name, &tool_defs);
                        for w in wrappers {
                            tools.register(w);
                        }
                    }
                    McpToolOp::Unregister { server_name } => {
                        tools.strip_prefixes(&[&format!("{server_name}_")]);
                    }
                }
            }
        }

        // Drain OAuth / auth tool registrations
        let pending: Vec<Arc<dyn Tool>> = std::mem::take(&mut *self.pending_tools.lock().unwrap());
        for tool in pending {
            tools.register(tool);
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
