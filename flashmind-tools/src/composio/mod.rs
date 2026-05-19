//! Composio integration — unified API for 250+ app integrations via composio.dev.

pub mod client;
pub mod types;
pub mod wrapper;

pub use client::ComposioClient;
pub use types::{ComposioToolDef, ComposioToolkit, ExecuteResponse};
pub use wrapper::{ComposioToolWrapper, make_composio_tool_wrappers};

/// Configuration for the Composio integration.
#[derive(Debug, Clone)]
pub struct ComposioConfig {
    /// Composio API key (from the composio.dev dashboard).
    pub api_key: String,
    /// Connected account ID for tools that require per-user auth.
    /// If `None`, tools that need auth will return an error directing
    /// the user to connect via the Composio dashboard.
    pub connected_account_id: Option<String>,
    /// Only load tools from these toolkits (e.g. `["github", "slack"]`).
    /// Empty means load all available tools.
    pub toolkits: Vec<String>,
    /// Custom base URL for self-hosted or enterprise deployments.
    /// Defaults to `https://backend.composio.dev/api/v3.1`.
    pub base_url: Option<String>,
}
