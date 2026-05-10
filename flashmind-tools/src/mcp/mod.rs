//! Model Context Protocol (MCP) integration.
//!
//! Enables the agent to connect to remote MCP servers and use their tools.
//! Supports both stdio (local process) and HTTP/SSE (remote) transports.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`McpRegistry`] | Manages connected MCP servers and their tool definitions |
//! | [`McpConfigProvider`] | Trait for persisting MCP server configurations |
//! | [`McpAuthHandler`] | Trait for handling server authentication |
//! | [`McpServerConfig`] | Configuration for a single MCP server connection |
//! | [`McpDiskConfig`] | Filesystem-backed config provider (`{base_dir}/{name}.json`) |
//! | [`McpToolDef`] | Tool definition received from an MCP server |
//!
//! # Tools
//!
//! - `mcp_add` — Connect to a new MCP server
//! - `mcp_list` — List connected servers and their tools
//! - `mcp_auth` — Authenticate with an MCP server
//! - `mcp_remove` — Disconnect from an MCP server

pub mod auth;
pub mod config;
pub mod registry;
pub mod tools;
mod transport;
pub mod types;

/// Type alias for MCP server names used as map keys.
pub type Host = String;

pub use auth::{AuthOutcome, McpAuthHandler};
pub use config::{McpConfigProvider, McpDiskConfig, McpServerConfig};
pub use registry::McpRegistry;
pub use tools::{McpToolWrapper, make_mcp_tool_wrappers};
pub use types::{McpAuthRequired, McpContent, McpToolCallResult, McpToolDef, McpToolOp};
