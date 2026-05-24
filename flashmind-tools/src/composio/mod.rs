//! Composio integration — unified API for 250+ app integrations via composio.dev.
//!
//! # Overview
//!
//! Composio provides OAuth-managed access to 250+ apps (GitHub, Slack, Gmail, etc.)
//! through a single REST API. This module wraps that API as native [`Tool`](flashmind_types::tool::Tool)
//! implementations so an agent can call any connected app.
//!
//! # Multi-user OAuth
//!
//! Each of your users gets their own isolated OAuth connections — they never need
//! a Composio account. Your platform holds one API key; per-user tokens are managed
//! by Composio.
//!
//! ```rust,no_run
//! # use flashmind_tools::composio::*;
//! # async fn example() -> anyhow::Result<()> {
//! let client = ComposioClient::new("api-key".into(), None, None);
//!
//! // 1. Create a session — returns OAuth URLs per toolkit
//! let session = client.create_session(
//!     "user_42",
//!     Some(SessionToolkits { enable: Some(vec!["github".into()]), disable: None }),
//!     Some("https://myapp.com/callback"),
//! ).await?;
//!
//! // 2. Redirect user to session.connection_urls["github"]
//! //    After they authorize, your callback receives the connected_account_id.
//!
//! // 3. Poll to check completion
//! let updated = client.get_session(&session.session_id).await?;
//! let account_id = &updated.connected_accounts["github"][0];
//!
//! // 3b. Or list/execute tools scoped to the session directly:
//! let tools = client.list_session_tools(&session.session_id).await?;
//! // let result = client.execute_tool_in_session(&session.session_id, "GITHUB_CREATE_AN_ISSUE", args).await?;
//!
//! // 4. Build an agent scoped to this user's connections
//! let config = ComposioConfig {
//!     api_key: "api-key".into(),
//!     connected_account_id: Some(account_id.clone()),
//!     toolkits: vec!["github".into()],
//!     base_url: None,
//! };
//! # Ok(())
//! # }
//! ```
//!
//! # Discovery
//!
//! Use [`ComposioClient::list_toolkits`] to browse available apps and
//! [`ComposioClient::list_tools`] to see what tools a toolkit provides.
//!
//! # Direct execution
//!
//! [`ComposioClient::execute_tool`] lets you call any tool without going through
//! the agent loop — useful for one-off API calls or testing.

pub mod auth_tool;
pub mod client;
pub mod types;
pub mod wrapper;

pub use auth_tool::ComposioAuthTool;
pub use client::ComposioClient;
pub use types::{
    ComposioPage, ComposioSession, ComposioToolDef, ComposioToolkit, ComposioTriggerInstance,
    ComposioTriggerType, ConnectedAccountInfo, ExecuteResponse, McpInfo, SessionConfig,
    SessionLinkResponse, SessionToolkits, TriggerLogEntry, TriggerLogsRequest, TriggerLogsResponse,
    TriggerUpsertResponse,
};
pub use wrapper::{ComposioReauthInterrupt, ComposioToolWrapper, make_composio_tool_wrappers};

/// Configuration for the Composio integration.
#[derive(Debug, Clone)]
pub struct ComposioConfig {
    /// Composio API key (from the composio.dev dashboard).
    pub api_key: String,
    /// Per-user connected account ID from [`ComposioClient::create_session`].
    /// Each user on your platform gets their own ID, isolating their OAuth tokens.
    pub connected_account_id: Option<String>,
    /// Toolkits to enable (e.g. `["github", "slack"]`).
    /// Empty means no Composio tools are registered.
    pub toolkits: Vec<String>,
    /// Custom base URL for self-hosted or enterprise deployments.
    /// Defaults to `https://backend.composio.dev/api/v3.1`.
    pub base_url: Option<String>,
}
