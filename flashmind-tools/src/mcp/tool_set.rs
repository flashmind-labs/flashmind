//! Simplified handle for syncing MCP tool changes with a [`ToolRegistry`].

use flashmind_types::ToolRegistry;

use super::registry::McpRegistry;
use super::tools::make_mcp_tool_wrappers;
use super::types::McpToolOp;

// ---------------------------------------------------------------------------

/// Handle for syncing MCP tool changes with a [`ToolRegistry`].
///
/// Created by [`ToolBuilder::build_with_mcp()`](crate::builder::ToolBuilder::build_with_mcp).
/// Call [`sync()`](Self::sync) after each agent turn to apply any pending
/// tool registrations or removals from MCP servers.
///
/// # Example
///
/// ```rust,ignore
/// let (tools, mcp) = ToolBuilder::new()
///     .mcp(config, None)
///     .build_with_mcp()
///     .await;
///
/// let mut agent = Agent::builder(provider).tools(tools).build();
///
/// loop {
///     let stream = agent.start(&mut conv, input, None);
///     repl.stream_response(stream).await?;
///     mcp.sync(agent.tools_mut());
/// }
///
/// mcp.shutdown().await;
/// ```
pub struct McpToolSet {
    registry: McpRegistry,
}

impl McpToolSet {
    pub(crate) fn new(registry: McpRegistry) -> Self {
        Self { registry }
    }

    /// Apply pending MCP tool registrations and removals.
    pub fn sync(&self, tools: &mut ToolRegistry) {
        for op in self.registry.drain_pending_ops() {
            match op {
                McpToolOp::Register {
                    server_name,
                    tool_defs,
                } => {
                    let wrappers = make_mcp_tool_wrappers(&self.registry, &server_name, &tool_defs);
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

    /// Shut down all MCP server connections.
    pub async fn shutdown(&self) {
        self.registry.shutdown_all().await;
    }

    /// Access the underlying [`McpRegistry`] for advanced use cases.
    pub fn registry(&self) -> &McpRegistry {
        &self.registry
    }
}
