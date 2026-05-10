//! Built-in tool implementations for the Flashmind AI agent framework.
//!
//! This crate contains self-contained [`Tool`] implementations that don't depend
//! on binary-specific components (canvas, agent manager, daemon services).
//!
//! # Tool categories
//!
//! | Category | Modules |
//! |----------|---------|
//! | File & System | [`bash`], [`file_ops`], [`glob`], [`grep`], [`text_replace`], [`process`] |
//! | HTTP | [`http`] |
//! | Web search | [`brave`], [`firecrawl`], [`search_read`] |
//! | Text editing | [`text_replace`], [`text_replace_regex`] |
//! | Audio | [`audio`] |
//! | Generation | [`image_gen`], [`image_edit`], [`video_gen`] |
//! | Image/media reading | [`image_read`] |
//! | Database | [`sqlite`] |
//! | Model discovery | [`list_models`] |
//! | MCP client | [`mcp`] |
//! | Utilities | [`json_query`], [`time`], [`str_diff`] |
//! | Infrastructure | [`file_cache`], [`protected`], [`search_cache`], [`utils`] |
//!
//! # Implementing a custom tool
//!
//! ```rust,ignore
//! use async_trait::async_trait;
//! use flashmind_tools::{Tool, ToolContext, ToolResult};
//! use serde_json::json;
//!
//! struct MyTool;
//!
//! #[async_trait]
//! impl Tool for MyTool {
//!     fn name(&self) -> &str { "my_tool" }
//!     fn description(&self) -> &str { "Does something useful" }
//!     fn parameters(&self) -> serde_json::Value {
//!         json!({"type": "object", "properties": {}, "required": []})
//!     }
//!     async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
//!         Ok(ToolResult::success(ctx.tool_call_id, "done"))
//!     }
//!     fn humanize(&self, _args: &serde_json::Value) -> String {
//!         "Running my tool".into()
//!     }
//! }
//! ```

pub mod utils;

// Infrastructure (no binary deps)
pub mod file_cache;
pub mod protected;
pub mod search_cache;

// Simple tools
pub mod json_query;
pub mod process;
pub mod str_diff;
pub mod tailscale;
pub mod time;

// File tools
pub mod bash;
pub mod file_ops;
pub mod glob;
pub mod grep;
pub mod text_replace;
pub mod text_replace_regex;

// HTTP tools
pub mod http;

// Web search/scraping tools
pub mod brave;
pub mod firecrawl;
pub mod search_read;

// Audio tools
pub mod audio;

// Generation tools
pub mod image_edit;
pub mod image_gen;
pub mod video_gen;

// Image/media reading
pub mod image_read;

// Database
pub mod sqlite;

// Model discovery
pub mod list_models;

// Agent delegation and communication
#[cfg(feature = "subagent")]
pub mod subagent;

// MCP (Model Context Protocol) client
#[cfg(feature = "mcp")]
pub mod mcp;

// Gmail API tools
#[cfg(feature = "gmail")]
pub mod gmail;

// Builder
pub mod builder;

// Re-export commonly used types from flashmind-types
pub use builder::ToolBuilder;
pub use flashmind_types::tool::{
    FileDiff, ForbiddenCmd, Tool, ToolContext, ToolRegistry, ToolResult, parse_args,
};

/// Test helpers for tool implementations.
#[cfg(test)]
pub mod tests {
    use serde_json::Value;
    use tokio_util::sync::CancellationToken;

    use flashmind_types::tool::{Tool, ToolContext, ToolResult};

    /// Execute a tool with a minimal context for testing.
    pub async fn execute_tool(
        tool: &dyn Tool,
        tool_call_id: &str,
        args: Value,
    ) -> anyhow::Result<ToolResult> {
        let cancel_token = CancellationToken::new();
        let ctx = ToolContext::new(tool_call_id, args, None, &cancel_token);
        tool.execute(ctx).await
    }

    /// Minimal mock provider for tests that need an `Arc<dyn LlmProvider>`.
    pub fn mock_provider() -> impl flashmind_types::LlmProvider {
        struct MockLlm;
        #[async_trait::async_trait]
        impl flashmind_types::LlmProvider for MockLlm {
            fn complete(
                &self,
                _req: flashmind_types::CompletionRequest,
            ) -> flashmind_types::CompletionStream {
                Box::pin(futures::stream::empty())
            }
            fn name(&self) -> &str {
                "mock"
            }
            fn provider(&self) -> flashmind_types::Provider {
                flashmind_types::Provider::Ollama
            }
        }
        MockLlm
    }
}

#[cfg(test)]
#[ctor::ctor]
fn init_crypto_for_tests() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
