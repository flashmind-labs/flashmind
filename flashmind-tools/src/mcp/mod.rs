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
//! | [`McpConfigProvider`] | Trait for resolving MCP server configurations |
//! | [`McpServerConfig`] | Configuration for a single MCP server connection |
//! | [`McpToolDef`] | Tool definition received from an MCP server |
//!
//! # Tools
//!
//! - `mcp_add` — Connect to a new MCP server
//! - `mcp_list` — List connected servers and their tools
//! - `mcp_auth` — Authenticate with an MCP server
//! - `mcp_remove` — Disconnect from an MCP server

pub mod tools;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientCapabilities, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::auth::{
    AuthClient, AuthError, AuthorizationSession, CredentialStore, OAuthClientConfig, OAuthState,
    StoredCredentials,
};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

type McpService = RunningService<RoleClient, rmcp::model::ClientInfo>;

// ---------------------------------------------------------------------------
// Config provider trait
// ---------------------------------------------------------------------------

/// Trait for resolving MCP server configurations.
///
/// Implementations can load configs from files, environment variables,
/// or dynamic sources. Called when `mcp_add` needs to resolve a server name.
#[async_trait]
pub trait McpConfigProvider: Send + Sync {
    /// List all saved MCP server configurations.
    async fn list_configs(&self) -> Result<Vec<McpServerConfig>>;
    /// Save or update an MCP server configuration.
    async fn save_config(&self, config: &McpServerConfig) -> Result<()>;
    /// Delete an MCP server configuration by name.
    async fn delete_config(&self, name: &str) -> Result<()>;
    /// Save OAuth credentials for a server.
    async fn save_credentials(&self, name: &str, credentials_json: &str) -> Result<()>;
    /// Load stored OAuth credentials for a server.
    async fn load_credentials(&self, name: &str) -> Result<Option<String>>;
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique name identifying this server (e.g., "github", "fastmail").
    pub name: String,
    /// Command to spawn the MCP server process (stdio transport). Exclusive with `url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Arguments passed to the command.
    #[serde(default)]
    pub args: Vec<String>,
    /// HTTP/SSE endpoint URL (alternative to command+args for remote servers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Environment variables set for the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// OAuth client ID (for providers that require pre-registered credentials).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// OAuth client secret.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// OAuth scopes to request during authentication.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Serialized OAuth credentials from a previous auth session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_json: Option<String>,
    /// Reauthentication hint/timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reauth: Option<String>,
    /// Cached tool definitions from the last successful connection (used at startup).
    #[serde(default)]
    pub cached_tools: Vec<McpToolDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

impl PartialEq for McpToolDef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.description == other.description
            && self.input_schema == other.input_schema
    }
}

/// Result of calling a tool on an MCP server.
pub struct McpToolCallResult {
    /// Content items returned by the tool (text, images, etc.).
    pub content: Vec<McpContent>,
    /// Whether the MCP server flagged this result as an error.
    pub is_error: bool,
}

/// A content item from an MCP tool call result.
///
/// MCP servers can return multiple content blocks per tool call. Most tools
/// return plain text, but arbitrary content types are represented as `Other`
/// with a debug string.
pub enum McpContent {
    /// Plain text content.
    Text(String),
    /// Non-text content (debug representation of the original).
    Other(String),
}

impl McpContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text(s) | McpContent::Other(s) => Some(s),
        }
    }
}

/// Queued operation for the caller to apply to its `ToolRegistry`.
///
/// The MCP registry queues these operations when servers connect or disconnect.
/// The caller (agent loop) should drain them each turn via
/// [`McpRegistry::drain_pending_ops`] and register/unregister wrapper tools.
pub enum McpToolOp {
    /// A server connected and its tools should be registered.
    Register {
        server_name: String,
        tool_defs: Vec<McpToolDef>,
    },
    /// A server was removed and its tools should be unregistered.
    Unregister { server_name: String },
}

#[derive(Debug)]
pub struct McpAuthRequired {
    pub server: String,
}

impl std::fmt::Display for McpAuthRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCP server '{}' requires authentication", self.server)
    }
}

impl std::error::Error for McpAuthRequired {}

fn tool_def_from_rmcp(t: &rmcp::model::Tool) -> McpToolDef {
    McpToolDef {
        name: t.name.to_string(),
        description: t.description.as_ref().map(|d| d.to_string()),
        input_schema: serde_json::to_value(&*t.input_schema).unwrap_or_default(),
    }
}

fn call_result_from_rmcp(r: CallToolResult) -> McpToolCallResult {
    let content = r
        .content
        .into_iter()
        .map(|c| match c.as_text() {
            Some(text) => McpContent::Text(text.text.clone()),
            None => McpContent::Other(format!("{c:?}")),
        })
        .collect();
    McpToolCallResult {
        content,
        is_error: r.is_error.unwrap_or(false),
    }
}

// ---------------------------------------------------------------------------
// Credential store adapter
// ---------------------------------------------------------------------------

struct ProviderCredentialStore {
    provider: Arc<dyn McpConfigProvider>,
    server_name: String,
}

#[async_trait]
impl CredentialStore for ProviderCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        let json = self
            .provider
            .load_credentials(&self.server_name)
            .await
            .map_err(|e| AuthError::InternalError(e.to_string()))?;
        match json {
            Some(j) if !j.is_empty() => {
                let creds: StoredCredentials = serde_json::from_str(&j)
                    .map_err(|e| AuthError::InternalError(e.to_string()))?;
                Ok(Some(creds))
            }
            _ => Ok(None),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let json = serde_json::to_string(&credentials)
            .map_err(|e| AuthError::InternalError(e.to_string()))?;
        self.provider
            .save_credentials(&self.server_name, &json)
            .await
            .map_err(|e| AuthError::InternalError(e.to_string()))
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        self.provider
            .save_credentials(&self.server_name, "")
            .await
            .map_err(|e| AuthError::InternalError(e.to_string()))
    }
}

struct ArcCredentialStore(Arc<dyn CredentialStore>);

#[async_trait]
impl CredentialStore for ArcCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        self.0.load().await
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        self.0.save(credentials).await
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        self.0.clear().await
    }
}

// ---------------------------------------------------------------------------
// McpConnection
// ---------------------------------------------------------------------------

struct McpConnection {
    #[allow(dead_code)]
    config: McpServerConfig,
    service: McpService,
    tool_names: Vec<String>,
    #[allow(dead_code)]
    tool_defs: Vec<McpToolDef>,
}

// ---------------------------------------------------------------------------
// McpRegistry
// ---------------------------------------------------------------------------

/// Registry managing MCP server connections and their tool definitions.
///
/// Servers are added/removed at runtime via `mcp_add`/`mcp_remove` tools.
/// Each server's tools are wrapped in [`McpToolWrapper`](tools::McpToolWrapper)
/// and registered in the agent's tool registry.
#[derive(Clone)]
pub struct McpRegistry {
    provider: Arc<dyn McpConfigProvider>,
    configs: Arc<Mutex<HashMap<String, McpServerConfig>>>,
    connections: Arc<Mutex<HashMap<String, McpConnection>>>,
    pending_ops: Arc<std::sync::Mutex<Vec<McpToolOp>>>,
    current_tools: Arc<std::sync::Mutex<HashMap<String, Vec<McpToolDef>>>>,
    pending_auth: Arc<Mutex<HashMap<String, OAuthState>>>,
}

impl McpRegistry {
    pub fn new(provider: Arc<dyn McpConfigProvider>) -> Self {
        Self {
            provider,
            configs: Arc::new(Mutex::new(HashMap::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
            pending_ops: Arc::new(std::sync::Mutex::new(Vec::new())),
            current_tools: Arc::new(std::sync::Mutex::new(HashMap::new())),
            pending_auth: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn make_credential_store(&self, server_name: &str) -> Arc<dyn CredentialStore> {
        Arc::new(ProviderCredentialStore {
            provider: self.provider.clone(),
            server_name: server_name.to_string(),
        })
    }

    fn client_info() -> rmcp::model::ClientInfo {
        rmcp::model::ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("flashmind", env!("CARGO_PKG_VERSION")),
        )
    }

    // -- Startup & config management ------------------------------------------

    pub async fn load_saved(&self) {
        let saved = match self.provider.list_configs().await {
            Ok(configs) => configs,
            Err(e) => {
                tracing::warn!(error = %e, "failed to load MCP configs");
                return;
            }
        };

        for cfg in &saved {
            self.configs
                .lock()
                .await
                .insert(cfg.name.clone(), cfg.clone());

            if !cfg.cached_tools.is_empty() {
                tracing::info!(server = %cfg.name, tools = cfg.cached_tools.len(), "registering cached MCP tools");
                self.current_tools
                    .lock()
                    .unwrap()
                    .insert(cfg.name.clone(), cfg.cached_tools.clone());
                self.pending_ops.lock().unwrap().push(McpToolOp::Register {
                    server_name: cfg.name.clone(),
                    tool_defs: cfg.cached_tools.clone(),
                });
            }
        }

        for cfg in saved {
            let name = cfg.name.clone();
            let registry = self.clone();
            tokio::spawn(async move {
                match registry.ensure_connected(&name).await {
                    Ok(()) => {}
                    Err(e) => {
                        if cfg.cached_tools.is_empty() {
                            tracing::warn!(server = %name, error = %e, "startup MCP connect failed (no cached tools)");
                        } else {
                            tracing::debug!(server = %name, error = %e, "startup MCP connect failed (using cached tools)");
                        }
                    }
                }
            });
        }
    }

    pub async fn add(&self, config: McpServerConfig) -> Result<()> {
        self.provider.save_config(&config).await?;
        self.configs
            .lock()
            .await
            .insert(config.name.clone(), config);
        Ok(())
    }

    pub async fn remove(&self, name: &str) -> Result<()> {
        if let Some(conn) = self.connections.lock().await.remove(name) {
            conn.service.cancel().await.ok();
        }

        self.configs.lock().await.remove(name);
        self.provider.delete_config(name).await?;

        self.pending_ops
            .lock()
            .unwrap()
            .push(McpToolOp::Unregister {
                server_name: name.to_string(),
            });
        self.current_tools.lock().unwrap().remove(name);

        tracing::info!(server = %name, "MCP server removed");
        Ok(())
    }

    // -- Accessors ------------------------------------------------------------

    pub fn drain_pending_ops(&self) -> Vec<McpToolOp> {
        std::mem::take(&mut *self.pending_ops.lock().unwrap())
    }

    pub fn current_mcp_tools(&self) -> HashMap<String, Vec<McpToolDef>> {
        self.current_tools.lock().unwrap().clone()
    }

    // -- Connection management ------------------------------------------------

    pub async fn connect(&self, config: McpServerConfig) -> Result<Vec<McpToolDef>> {
        let service = self.open_transport(&config).await?;
        let tool_defs = self.list_server_tools(&service).await?;
        let tool_names: Vec<String> = tool_defs.iter().map(|t| t.name.clone()).collect();

        tracing::info!(
            server = %config.name, tool_count = tool_names.len(),
            "MCP server connected"
        );

        self.store_connection(&config, service, tool_names, &tool_defs)
            .await;

        Ok(tool_defs)
    }

    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolCallResult> {
        self.ensure_connected(server_name).await?;

        match self.execute_call(server_name, tool_name, &arguments).await {
            Ok(r) => Ok(r),
            Err(e) => {
                tracing::warn!(
                    server = %server_name, tool = %tool_name, error = %e,
                    "MCP call failed, reconnecting"
                );
                self.drop_connection(server_name).await;
                self.ensure_connected(server_name).await?;
                self.execute_call(server_name, tool_name, &arguments).await
            }
        }
    }

    pub async fn list_tools(&self, server_name: &str) -> Result<Vec<McpToolDef>> {
        self.ensure_connected(server_name).await?;

        let conns = self.connections.lock().await;
        let conn = conns
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' is not registered"))?;

        let tools = conn
            .service
            .list_all_tools()
            .await
            .context("failed to list tools")?;

        Ok(tools.iter().map(tool_def_from_rmcp).collect())
    }

    pub async fn list(&self) -> Vec<(String, Vec<String>, Option<String>)> {
        let to_connect: Vec<McpServerConfig> = {
            let configs = self.configs.lock().await;
            let conns = self.connections.lock().await;
            configs
                .values()
                .filter(|c| !conns.contains_key(&c.name))
                .cloned()
                .collect()
        };

        let mut errors: HashMap<String, String> = HashMap::new();
        for config in to_connect {
            let name = config.name.clone();
            if let Err(e) = self.connect(config).await {
                errors.insert(name, e.to_string());
            }
        }

        let mut result: Vec<(String, Vec<String>, Option<String>)> = Vec::new();
        {
            let conns = self.connections.lock().await;
            let configs = self.configs.lock().await;
            for name in configs.keys() {
                match conns.get(name) {
                    Some(conn) => result.push((name.clone(), conn.tool_names.clone(), None)),
                    None => result.push((name.clone(), vec![], errors.get(name).cloned())),
                }
            }
        }

        result
    }

    pub async fn shutdown_all(&self) {
        let mut conns = self.connections.lock().await;
        for (name, conn) in conns.drain() {
            conn.service.cancel().await.ok();
            tracing::info!(server = %name, "MCP server shut down");
        }
    }

    pub async fn refresh_all(&self) -> Vec<(String, Result<Vec<McpToolDef>>)> {
        self.shutdown_all().await;
        self.connections.lock().await.clear();
        self.current_tools.lock().unwrap().clear();
        self.pending_ops.lock().unwrap().clear();

        let configs: Vec<McpServerConfig> = self.configs.lock().await.values().cloned().collect();

        let mut results = Vec::new();
        for cfg in configs {
            let name = cfg.name.clone();
            match self.connect(cfg).await {
                Ok(tool_defs) => results.push((name, Ok(tool_defs))),
                Err(e) => results.push((name, Err(e))),
            }
        }

        results
    }

    pub async fn reconnect(&self, server_name: &str) -> Result<Vec<McpToolDef>> {
        self.drop_connection(server_name).await;

        let config = {
            let configs = self.configs.lock().await;
            configs
                .get(server_name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' not found"))?
        };

        self.connect(config).await
    }

    // -- OAuth ----------------------------------------------------------------

    pub async fn start_auth(&self, server_name: &str, redirect_uri: &str) -> Result<String> {
        let (url, client_id, client_secret, scopes) = {
            let configs = self.configs.lock().await;
            let cfg = configs
                .get(server_name)
                .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' not found"))?;
            let url = cfg
                .url
                .clone()
                .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' has no URL"))?;
            (
                url,
                cfg.client_id.clone(),
                cfg.client_secret.clone(),
                cfg.scopes.clone(),
            )
        };

        let mut oauth_state = OAuthState::new(&url, None)
            .await
            .context("OAuth metadata discovery failed")?;

        let store = self.make_credential_store(server_name);
        if let OAuthState::Unauthorized(ref mut mgr) = oauth_state {
            mgr.set_credential_store(ArcCredentialStore(store));
        }

        if let Some(cid) = client_id {
            if let OAuthState::Unauthorized(mut mgr) = oauth_state {
                let metadata = mgr
                    .discover_metadata()
                    .await
                    .context("OAuth metadata discovery failed")?;
                mgr.set_metadata(metadata);

                let mut config = OAuthClientConfig::new(&cid, redirect_uri);
                if let Some(ref secret) = client_secret {
                    config = config.with_client_secret(secret);
                }
                mgr.configure_client(config)
                    .map_err(|e| anyhow::anyhow!("configure OAuth client: {e}"))?;

                let effective_scopes = if scopes.is_empty() {
                    mgr.select_scopes(None, &[])
                } else {
                    scopes.clone()
                };
                let scope_refs: Vec<&str> = effective_scopes.iter().map(|s| s.as_str()).collect();
                let auth_url = mgr
                    .get_authorization_url(&scope_refs)
                    .await
                    .map_err(|e| anyhow::anyhow!("get authorization URL: {e}"))?;

                let session =
                    AuthorizationSession::for_scope_upgrade(mgr, auth_url.clone(), redirect_uri);
                oauth_state = OAuthState::Session(session);
            } else {
                bail!("unexpected OAuth state for server '{server_name}'");
            }
        } else {
            let scope_refs: Vec<&str> = scopes.iter().map(|s| s.as_str()).collect();
            oauth_state
                .start_authorization(&scope_refs, redirect_uri, Some("flashmind"))
                .await
                .context("OAuth authorization setup failed")?;
        }

        let auth_url = oauth_state
            .get_authorization_url()
            .await
            .context("failed to get authorization URL")?;

        tracing::info!(%auth_url, server = %server_name, "OAuth authorization started");

        self.pending_auth
            .lock()
            .await
            .insert(server_name.to_string(), oauth_state);

        Ok(auth_url.to_string())
    }

    pub async fn complete_auth(&self, server_name: &str, code: &str, state: &str) -> Result<()> {
        let mut oauth_state = self
            .pending_auth
            .lock()
            .await
            .remove(server_name)
            .ok_or_else(|| anyhow::anyhow!("no pending OAuth flow for server '{server_name}'"))?;

        oauth_state
            .handle_callback(code, state)
            .await
            .context("OAuth token exchange failed")?;

        let config = self.configs.lock().await.get(server_name).cloned();
        if let Some(cfg) = config {
            self.connect(cfg).await?;
        }

        tracing::info!(server = %server_name, "OAuth authentication completed");
        Ok(())
    }

    // -- Private helpers ------------------------------------------------------

    async fn cache_tools_to_config(&self, config: &McpServerConfig, tool_defs: &[McpToolDef]) {
        let mut updated = config.clone();
        updated.cached_tools = tool_defs.to_vec();
        if let Err(e) = self.provider.save_config(&updated).await {
            tracing::warn!(server = %config.name, error = %e, "failed to persist cached MCP tools");
        } else {
            tracing::debug!(server = %config.name, tools = tool_defs.len(), "cached MCP tools updated");
        }

        if let Ok(mut guard) = self.configs.try_lock() {
            if let Some(entry) = guard.get_mut(&config.name) {
                entry.cached_tools = tool_defs.to_vec();
            }
        }
    }

    async fn open_transport(&self, config: &McpServerConfig) -> Result<McpService> {
        if let Some(ref url) = config.url {
            self.connect_http(&config.name, url).await
        } else if let Some(ref command) = config.command {
            self.connect_stdio(command, &config.args, &config.env).await
        } else {
            bail!("MCP server '{}' has neither command nor url", config.name);
        }
    }

    async fn list_server_tools(&self, service: &McpService) -> Result<Vec<McpToolDef>> {
        let tools = service
            .list_all_tools()
            .await
            .context("failed to list tools")?;
        Ok(tools.iter().map(tool_def_from_rmcp).collect())
    }

    async fn store_connection(
        &self,
        config: &McpServerConfig,
        service: McpService,
        tool_names: Vec<String>,
        tool_defs: &[McpToolDef],
    ) {
        self.cache_tools_to_config(config, tool_defs).await;

        let conn = McpConnection {
            config: config.clone(),
            service,
            tool_names,
            tool_defs: tool_defs.to_vec(),
        };

        let mut conns = self.connections.lock().await;
        if let Some(old) = conns.remove(&config.name) {
            old.service.cancel().await.ok();
        }
        conns.insert(config.name.clone(), conn);

        self.current_tools
            .lock()
            .unwrap()
            .insert(config.name.clone(), tool_defs.to_vec());

        self.pending_ops.lock().unwrap().push(McpToolOp::Register {
            server_name: config.name.clone(),
            tool_defs: tool_defs.to_vec(),
        });
    }

    async fn ensure_connected(&self, server_name: &str) -> Result<()> {
        if self.connections.lock().await.contains_key(server_name) {
            return Ok(());
        }

        let config = self
            .configs
            .lock()
            .await
            .get(server_name)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("MCP server '{server_name}' is not registered. Use mcp_add first.")
            })?;

        tracing::info!(server = %server_name, "auto-connecting to MCP server");
        self.connect(config).await?;
        Ok(())
    }

    async fn execute_call(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<McpToolCallResult> {
        let args_map = arguments.as_object().cloned().unwrap_or_default();
        let params = CallToolRequestParams::new(tool_name.to_string()).with_arguments(args_map);

        let conns = self.connections.lock().await;
        let conn = conns
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' is not connected"))?;

        let r = conn
            .service
            .call_tool(params)
            .await
            .with_context(|| format!("tools/call failed for '{tool_name}'"))?;

        Ok(call_result_from_rmcp(r))
    }

    async fn drop_connection(&self, server_name: &str) {
        if let Some(dead) = self.connections.lock().await.remove(server_name) {
            dead.service.cancel().await.ok();
        }
    }

    async fn connect_http(&self, server_name: &str, url: &str) -> Result<McpService> {
        if let Some(service) = self.try_connect_with_credentials(server_name, url).await {
            return Ok(service);
        }

        if let Some(service) = self.try_connect_plain(url).await {
            return Ok(service);
        }

        bail!(McpAuthRequired {
            server: server_name.to_string(),
        })
    }

    async fn try_connect_with_credentials(
        &self,
        server_name: &str,
        url: &str,
    ) -> Option<McpService> {
        let store = self.make_credential_store(server_name);
        let creds = match store.load().await {
            Ok(Some(c)) => c,
            Ok(None) => {
                tracing::debug!(server = %server_name, "no stored credentials");
                return None;
            }
            Err(e) => {
                tracing::warn!(server = %server_name, error = %e, "failed to load credentials");
                return None;
            }
        };

        tracing::debug!(server = %server_name, "found stored OAuth credentials");

        let mut oauth_state = match OAuthState::new(url, None).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(server = %server_name, error = %e, "OAuth state init failed");
                return None;
            }
        };

        let Some(token_response) = creds.token_response else {
            tracing::warn!(server = %server_name, "stored credentials have no token_response");
            return None;
        };
        if let Err(e) = oauth_state
            .set_credentials(&creds.client_id, token_response)
            .await
        {
            tracing::warn!(server = %server_name, error = %e, "set_credentials failed");
            return None;
        }
        let Some(mut mgr) = oauth_state.into_authorization_manager() else {
            tracing::warn!(server = %server_name, "into_authorization_manager returned None");
            return None;
        };

        mgr.set_credential_store(ArcCredentialStore(store));
        let auth_client = AuthClient::new(reqwest::Client::default(), mgr);
        let config = StreamableHttpClientTransportConfig::with_uri(url);
        let transport = StreamableHttpClientTransport::with_client(auth_client, config);

        match Self::client_info().serve(transport).await {
            Ok(service) => Some(service),
            Err(e) => {
                tracing::warn!(server = %server_name, error = %e, "connect with stored credentials failed");
                None
            }
        }
    }

    async fn try_connect_plain(&self, url: &str) -> Option<McpService> {
        let config = StreamableHttpClientTransportConfig::with_uri(url);
        let transport = StreamableHttpClientTransport::from_config(config);

        match Self::client_info().serve(transport).await {
            Ok(service) => Some(service),
            Err(e) => {
                tracing::debug!(error = %e, "unauthenticated connect failed");
                None
            }
        }
    }

    async fn connect_stdio(
        &self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<McpService> {
        use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
        use tokio::process::Command;

        let resolved = if !command.contains('/') {
            resolve_command(command).unwrap_or_else(|| command.to_owned())
        } else {
            command.to_owned()
        };

        let args = args.to_vec();
        let env = env.clone();
        let (transport, _stderr) =
            TokioChildProcess::builder(Command::new(&resolved).configure(move |cmd| {
                for arg in &args {
                    cmd.arg(arg);
                }
                for (k, v) in &env {
                    cmd.env(k, v);
                }
            }))
            .stderr(std::process::Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn MCP server: {resolved}"))?;

        let service = Self::client_info()
            .serve(transport)
            .await
            .context("MCP initialize failed")?;

        Ok(service)
    }
}

// ---------------------------------------------------------------------------
// Stdio command resolution
// ---------------------------------------------------------------------------

fn resolve_command(command: &str) -> Option<String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned());

    let output = std::process::Command::new(&shell)
        .args(["-l", "-c", &format!("which {command}")])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?.trim().to_owned();

    if path.is_empty() || !path.starts_with('/') {
        return None;
    }

    tracing::debug!(command, resolved = %path, "resolved MCP command via login shell");
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_command_finds_common_binaries() {
        let resolved = resolve_command("ls");
        assert!(resolved.is_some());
        assert!(resolved.unwrap().starts_with('/'));
    }

    #[test]
    fn test_resolve_command_returns_none_for_nonexistent() {
        let resolved = resolve_command("definitely_not_a_real_binary_abc123");
        assert!(resolved.is_none());
    }
}
