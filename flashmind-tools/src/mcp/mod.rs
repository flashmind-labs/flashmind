//! Model Context Protocol (MCP) client support.
//!
//! Provides tools for registering, listing, calling, and removing MCP servers.
//! Supports both stdio and HTTP/SSE transport modes via the `rmcp` crate.
//!
//! ## Submodules
//!
//! - [`tools`] — MCP tool implementations (`mcp_add`, `mcp_list`, `mcp_remove`) and first-class wrappers

pub mod tools;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientCapabilities, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::auth::{
    AuthClient, AuthError, CredentialStore, OAuthState, StoredCredentials,
};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

// Re-export so the agent crate can implement CredentialStore without
// depending on rmcp directly.
pub use rmcp::transport::auth::{
    AuthError as McpAuthError, CredentialStore as McpCredentialStore,
    StoredCredentials as McpStoredCredentials,
};

/// Adapter: wraps `Arc<dyn CredentialStore>` into an owned `CredentialStore`
/// so it can be passed to rmcp's `set_credential_store(S)` which takes by value.
struct ArcCredentialStore(Arc<dyn CredentialStore>);

#[async_trait::async_trait]
impl CredentialStore for ArcCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        self.0.load().await
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        self.0.save(credentials).await
    }

    async fn clear(&self) -> Result<(), AuthError> {
        self.0.clear().await
    }
}

/// Persisted TOML configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_json: Option<String>,
}

impl McpServerConfig {
    pub fn load_all_from(dir: &Path) -> Vec<Self> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let mut configs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            match std::fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|s| toml::from_str::<McpServerConfig>(&s).map_err(anyhow::Error::from))
            {
                Ok(cfg) => configs.push(cfg),
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to load MCP server config")
                }
            }
        }
        configs
    }

    pub fn save_to(config: &McpServerConfig, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.toml", config.name));
        let content = toml::to_string_pretty(config)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn delete_from(name: &str, dir: &Path) -> Result<()> {
        let path = dir.join(format!("{name}.toml"));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tool definition — thin wrapper over rmcp::model::Tool for our API
// ---------------------------------------------------------------------------

/// MCP server-side tool definition extracted from a connected service.
/// Converted from rmcp's model for our internal API.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

impl From<&rmcp::model::Tool> for McpToolDef {
    fn from(t: &rmcp::model::Tool) -> Self {
        Self {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            input_schema: serde_json::to_value(&*t.input_schema).unwrap_or_default(),
        }
    }
}

/// Tool call result — thin wrapper for our API.
pub struct McpToolCallResult {
    pub content: Vec<McpContent>,
    pub is_error: bool,
}

/// Content returned from an MCP tool call.
pub enum McpContent {
    Text(String),
    Other(String),
}

impl McpContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text(s) => Some(s),
            McpContent::Other(s) => Some(s),
        }
    }
}

impl From<CallToolResult> for McpToolCallResult {
    fn from(r: CallToolResult) -> Self {
        let content = r
            .content
            .into_iter()
            .map(|c| {
                if let Some(text) = c.as_text() {
                    McpContent::Text(text.text.clone())
                } else {
                    McpContent::Other(format!("{c:?}"))
                }
            })
            .collect();
        Self {
            content,
            is_error: r.is_error.unwrap_or(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// Alias for a running MCP service handling stdio or HTTP transport.
type McpService = RunningService<RoleClient, rmcp::model::ClientInfo>;

/// An active connection to an MCP server, holding the runtime service and tool list.
pub struct McpConnection {
    pub config: McpServerConfig,
    pub service: McpService,
    pub tool_names: Vec<String>,
    pub tool_defs: Vec<McpToolDef>,
}

/// Pending tool registration/unregistration operations for the owning ToolRegistry.
pub enum McpToolOp {
    Register {
        server_name: String,
        tool_defs: Vec<McpToolDef>,
    },
    Unregister {
        server_name: String,
    },
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Factory that produces a `CredentialStore` for a given MCP server name.
/// Injected by the agent crate for connect mode; local mode uses file-based
/// storage by default.
pub type CredentialStoreFactory = Arc<dyn Fn(&str) -> Arc<dyn CredentialStore> + Send + Sync>;

/// File-based credential store — reads/writes credentials_json in the MCP
/// server's TOML config file.
struct FileCredentialStore {
    mcp_dir: PathBuf,
    server_name: String,
}

#[async_trait::async_trait]
impl CredentialStore for FileCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        let path = self.mcp_dir.join(format!("{}.toml", self.server_name));
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Ok(None),
        };
        let config: McpServerConfig = toml::from_str(&content)
            .map_err(|e| AuthError::InternalError(format!("parse config: {e}")))?;
        match config.credentials_json {
            Some(json) => {
                let creds = serde_json::from_str(&json)
                    .map_err(|e| AuthError::InternalError(format!("parse credentials: {e}")))?;
                Ok(Some(creds))
            }
            None => Ok(None),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let path = self.mcp_dir.join(format!("{}.toml", self.server_name));
        let content = std::fs::read_to_string(&path)
            .map_err(|e| AuthError::InternalError(format!("read config: {e}")))?;
        let mut config: McpServerConfig = toml::from_str(&content)
            .map_err(|e| AuthError::InternalError(format!("parse config: {e}")))?;
        config.credentials_json = Some(
            serde_json::to_string(&credentials)
                .map_err(|e| AuthError::InternalError(format!("serialize credentials: {e}")))?,
        );
        let toml_str = toml::to_string_pretty(&config)
            .map_err(|e| AuthError::InternalError(format!("serialize config: {e}")))?;
        std::fs::write(&path, toml_str)
            .map_err(|e| AuthError::InternalError(format!("write config: {e}")))?;
        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        let path = self.mcp_dir.join(format!("{}.toml", self.server_name));
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };
        let mut config: McpServerConfig = toml::from_str(&content)
            .map_err(|e| AuthError::InternalError(format!("parse config: {e}")))?;
        config.credentials_json = None;
        let toml_str = toml::to_string_pretty(&config)
            .map_err(|e| AuthError::InternalError(format!("serialize config: {e}")))?;
        std::fs::write(&path, toml_str)
            .map_err(|e| AuthError::InternalError(format!("write config: {e}")))?;
        Ok(())
    }
}

/// Error indicating an MCP server requires OAuth authentication.
#[derive(Debug)]
pub struct McpAuthRequired {
    pub server: String,
}

impl std::fmt::Display for McpAuthRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCP server '{}' requires authentication — run mcp_auth",
            self.server
        )
    }
}

impl std::error::Error for McpAuthRequired {}

/// Registry of MCP server configurations and active connections.
///
/// Manages saving configs to disk, connecting/disconnecting servers, and
/// resolving tool calls through connected services. Clones share state via Arcs.
#[derive(Clone)]
pub struct McpRegistry {
    mcp_dir: PathBuf,
    configs: Arc<Mutex<HashMap<String, McpServerConfig>>>,
    connections: Arc<Mutex<HashMap<String, McpConnection>>>,
    credential_store_factory: Option<CredentialStoreFactory>,
    pending_ops: Arc<std::sync::Mutex<Vec<McpToolOp>>>,
    current_tools: Arc<std::sync::Mutex<HashMap<String, Vec<McpToolDef>>>>,
}

impl McpRegistry {
    pub fn new(mcp_dir: PathBuf) -> Self {
        Self {
            mcp_dir,
            configs: Arc::new(Mutex::new(HashMap::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
            credential_store_factory: None,
            pending_ops: Arc::new(std::sync::Mutex::new(Vec::new())),
            current_tools: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub fn with_credential_store_factory(mut self, factory: CredentialStoreFactory) -> Self {
        self.credential_store_factory = Some(factory);
        self
    }

    fn make_credential_store(&self, server_name: &str) -> Arc<dyn CredentialStore> {
        if let Some(ref factory) = self.credential_store_factory {
            factory(server_name)
        } else {
            Arc::new(FileCredentialStore {
                mcp_dir: self.mcp_dir.clone(),
                server_name: server_name.to_string(),
            })
        }
    }

    pub async fn load_saved(&self) {
        let saved = McpServerConfig::load_all_from(&self.mcp_dir);
        for cfg in saved {
            let name = cfg.name.clone();
            if let Err(e) = self.connect(cfg).await {
                tracing::warn!(server = %name, error = %e, "failed to connect saved MCP server");
            }
        }
    }

    pub fn load_for_user(&self, username: &str) -> Vec<McpServerConfig> {
        let mut configs: HashMap<String, McpServerConfig> = HashMap::new();

        for cfg in McpServerConfig::load_all_from(&self.mcp_dir) {
            configs.insert(cfg.name.clone(), cfg);
        }

        let user_dir = self.mcp_dir.join("users").join(username);
        for cfg in McpServerConfig::load_all_from(&user_dir) {
            configs.insert(cfg.name.clone(), cfg);
        }

        configs.into_values().collect()
    }

    pub fn save_for_user(&self, username: &str, config: &McpServerConfig) -> Result<()> {
        let user_dir = self.mcp_dir.join("users").join(username);
        McpServerConfig::save_to(config, &user_dir)
    }

    pub fn delete_for_user(&self, username: &str, server_name: &str) -> Result<()> {
        let user_dir = self.mcp_dir.join("users").join(username);
        McpServerConfig::delete_from(server_name, &user_dir)
    }

    pub fn mcp_dir(&self) -> &Path {
        &self.mcp_dir
    }
}

impl McpRegistry {
    /// Register and immediately connect. For HTTP servers requiring OAuth,
    /// returns `Err` with the auth URL — the caller should show it to the user.
    pub async fn add(&self, config: McpServerConfig) -> Result<()> {
        McpServerConfig::save_to(&config, &self.mcp_dir)?;
        self.configs
            .lock()
            .await
            .insert(config.name.clone(), config);
        Ok(())
    }

    /// Connect to an MCP server, initialize it, and return its tool list.
    ///
    /// For HTTP servers requiring OAuth, returns `Err` wrapping an `AuthRequired`
    /// with the URL the user must open. A background task waits for the callback;
    /// retry `connect()` after the user authorizes.
    pub async fn connect(&self, config: McpServerConfig) -> Result<Vec<McpToolDef>> {
        let client_info = rmcp::model::ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("flash", env!("CARGO_PKG_VERSION")),
        );

        let service: McpService = if let Some(ref url) = config.url {
            self.connect_http(&config.name, url, client_info).await?
        } else if let Some(ref command) = config.command {
            self.connect_stdio(command, &config.args, &config.env, client_info)
                .await?
        } else {
            bail!("MCP server '{}' has neither command nor url", config.name);
        };

        let tools = service
            .list_all_tools()
            .await
            .context("failed to list tools")?;

        let tool_defs: Vec<McpToolDef> = tools.iter().map(McpToolDef::from).collect();
        let tool_names: Vec<String> = tool_defs.iter().map(|t| t.name.clone()).collect();

        tracing::info!(
            server = %config.name,
            tool_count = tool_names.len(),
            "MCP server connected"
        );

        McpServerConfig::save_to(&config, &self.mcp_dir)?;
        self.configs
            .lock()
            .await
            .insert(config.name.clone(), config.clone());

        let mut conns = self.connections.lock().await;
        if let Some(old) = conns.remove(&config.name) {
            tracing::warn!(server = %config.name, "replacing existing MCP connection");
            old.service.cancel().await.ok();
        }

        conns.insert(
            config.name.clone(),
            McpConnection {
                config: config.clone(),
                service,
                tool_names,
                tool_defs: tool_defs.clone(),
            },
        );

        {
            let mut ops = self.pending_ops.lock().unwrap();
            ops.push(McpToolOp::Register {
                server_name: config.name.clone(),
                tool_defs: tool_defs.clone(),
            });
        }
        {
            let mut ct = self.current_tools.lock().unwrap();
            ct.insert(config.name, tool_defs.clone());
        }

        Ok(tool_defs)
    }

    /// Connect to an HTTP/SSE MCP server. Handles OAuth transparently via rmcp.
    ///
    /// If a credential store factory is configured, tokens are persisted
    /// externally. On reconnect, stored tokens are loaded automatically so
    /// the user doesn't need to re-authorize.
    async fn connect_http(
        &self,
        server_name: &str,
        url: &str,
        client_info: rmcp::model::ClientInfo,
    ) -> Result<McpService> {
        // If we have stored credentials, try connecting with them first
        let store = self.make_credential_store(server_name);
        if let Ok(Some(creds)) = store.load().await {
            tracing::debug!(server = %server_name, "found stored OAuth credentials");

            let mut oauth_state = OAuthState::new(url, None)
                .await
                .context("OAuth metadata discovery failed")?;

            if let Some(token_response) = creds.token_response
                && oauth_state
                    .set_credentials(&creds.client_id, token_response)
                    .await
                    .is_ok()
                && let Some(mut mgr) = oauth_state.into_authorization_manager()
            {
                mgr.set_credential_store(ArcCredentialStore(store));
                let auth_client = AuthClient::new(reqwest::Client::default(), mgr);
                let config = StreamableHttpClientTransportConfig::with_uri(url);
                let transport = StreamableHttpClientTransport::with_client(auth_client, config);

                match client_info.clone().serve(transport).await {
                    Ok(service) => return Ok(service),
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "connect with stored credentials failed, will re-auth"
                        );
                    }
                }
            }
        }

        // Try plain connect (no auth)
        let config = StreamableHttpClientTransportConfig::with_uri(url);
        let transport = StreamableHttpClientTransport::from_config(config);

        match client_info.clone().serve(transport).await {
            Ok(service) => return Ok(service),
            Err(e) => {
                tracing::debug!(error = %e, "initial connect failed, trying OAuth");
            }
        }

        bail!(McpAuthRequired {
            server: server_name.to_string(),
        })
    }

    /// Connect to a stdio MCP server (child process).
    async fn connect_stdio(
        &self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        client_info: rmcp::model::ClientInfo,
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
        let transport = TokioChildProcess::new(Command::new(&resolved).configure(move |cmd| {
            for arg in &args {
                cmd.arg(arg);
            }
            for (k, v) in &env {
                cmd.env(k, v);
            }
        }))
        .with_context(|| format!("failed to spawn MCP server: {resolved}"))?;

        let service = client_info
            .serve(transport)
            .await
            .context("MCP initialize failed")?;

        Ok(service)
    }

    pub async fn remove(&self, name: &str) -> Result<()> {
        {
            let mut conns = self.connections.lock().await;
            if let Some(conn) = conns.remove(name) {
                conn.service.cancel().await.ok();
            }
        }

        self.configs.lock().await.remove(name);
        McpServerConfig::delete_from(name, &self.mcp_dir)?;

        {
            let mut ops = self.pending_ops.lock().unwrap();
            ops.push(McpToolOp::Unregister {
                server_name: name.to_string(),
            });
        }
        {
            let mut ct = self.current_tools.lock().unwrap();
            ct.remove(name);
        }

        tracing::info!(server = %name, "MCP server removed");
        Ok(())
    }

    pub fn drain_pending_ops(&self) -> Vec<McpToolOp> {
        let mut ops = self.pending_ops.lock().unwrap();
        std::mem::take(&mut *ops)
    }

    pub fn current_mcp_tools(&self) -> HashMap<String, Vec<McpToolDef>> {
        self.current_tools.lock().unwrap().clone()
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

        Ok(tools.iter().map(McpToolDef::from).collect())
    }

    async fn ensure_connected(&self, server_name: &str) -> Result<()> {
        {
            let conns = self.connections.lock().await;
            if conns.contains_key(server_name) {
                return Ok(());
            }
        }

        let config = {
            let configs = self.configs.lock().await;
            configs.get(server_name).cloned()
        };

        if let Some(config) = config {
            tracing::info!(server = %server_name, "auto-connecting to MCP server");
            self.connect(config).await?;
            Ok(())
        } else {
            bail!("MCP server '{server_name}' is not registered. Use mcp_add first.")
        }
    }

    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolCallResult> {
        self.ensure_connected(server_name).await?;

        let conns = self.connections.lock().await;
        let conn = conns
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' is not connected"))?;

        let args_map = arguments.as_object().cloned().unwrap_or_default();

        let params = CallToolRequestParams::new(tool_name.to_string()).with_arguments(args_map);

        let result = conn
            .service
            .call_tool(params)
            .await
            .with_context(|| format!("tools/call failed for '{tool_name}'"))?;

        Ok(McpToolCallResult::from(result))
    }

    /// Run the OAuth flow for an MCP server: bind a localhost callback listener,
    /// return the authorization URL, wait for the callback, and store credentials.
    /// After success, call `connect()` to establish the MCP connection.
    pub async fn authenticate(&self, server_name: &str) -> Result<String> {
        let url = {
            let configs = self.configs.lock().await;
            configs
                .get(server_name)
                .and_then(|c| c.url.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("MCP server '{server_name}' not found or has no URL")
                })?
        };

        let mut oauth_state = OAuthState::new(&url, None)
            .await
            .context("OAuth metadata discovery failed")?;

        let store = self.make_credential_store(server_name);
        if let OAuthState::Unauthorized(ref mut mgr) = oauth_state {
            mgr.set_credential_store(ArcCredentialStore(store));
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("failed to bind OAuth callback listener")?;
        let port = listener.local_addr()?.port();
        let redirect_uri = format!("http://localhost:{port}/callback");

        oauth_state
            .start_authorization(&[], &redirect_uri, Some("flash"))
            .await
            .context("OAuth authorization setup failed")?;

        let auth_url = oauth_state
            .get_authorization_url()
            .await
            .context("failed to get authorization URL")?;

        tracing::info!(%auth_url, server = %server_name, "OAuth authorization required");

        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open")
            .arg(auth_url.as_str())
            .spawn();
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open")
            .arg(auth_url.as_str())
            .spawn();

        let (code, csrf_state) = accept_oauth_callback(listener)
            .await
            .context("OAuth callback failed")?;

        oauth_state
            .handle_callback(&code, &csrf_state)
            .await
            .context("OAuth token exchange failed")?;

        // Reconnect now that we have credentials
        let config = {
            let configs = self.configs.lock().await;
            configs.get(server_name).cloned()
        };
        if let Some(cfg) = config {
            self.connect(cfg).await?;
        }

        Ok(auth_url.to_string())
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

        let conns = self.connections.lock().await;
        let configs = self.configs.lock().await;
        let mut result = Vec::new();

        for name in configs.keys() {
            if let Some(conn) = conns.get(name) {
                result.push((name.clone(), conn.tool_names.clone(), None));
            } else {
                let err = errors.get(name).cloned();
                result.push((name.clone(), vec![], err));
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
}

// ---------------------------------------------------------------------------
// OAuth callback listener
// ---------------------------------------------------------------------------

/// Accept a single HTTP request on the OAuth callback listener, parse the
/// authorization code and CSRF state from the query string, and respond
/// with a success or failure HTML page.
async fn accept_oauth_callback(listener: tokio::net::TcpListener) -> Result<(String, String)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (mut stream, _) = listener.accept().await.context("accept failed")?;

    let mut buf = vec![0u8; 4096];
    let n = stream.read(&mut buf).await.context("read failed")?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let params: HashMap<String, String> = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|path| path.split('?').nth(1))
        .map(|query| {
            url::form_urlencoded::parse(query.as_bytes())
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let has_code = params.contains_key("code");
    let html = if has_code {
        "<html><body><h2>Authentication successful</h2><p>You can close this tab.</p></body></html>"
    } else {
        "<html><body><h2>Authentication failed</h2><p>No authorization code received.</p></body></html>"
    };

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        html.len(),
        html
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;

    if let Some(error) = params.get("error") {
        bail!("OAuth error: {error}");
    }

    let code = params
        .get("code")
        .cloned()
        .context("no authorization code in callback")?;
    let state = params.get("state").cloned().unwrap_or_default();

    Ok((code, state))
}

// ---------------------------------------------------------------------------
// Stdio command resolution
// ---------------------------------------------------------------------------

/// Resolve a bare command name (e.g., `"npx"`) to an absolute path by running `which` in a login shell.
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
    use tempfile::tempdir;

    #[test]
    fn test_server_config_roundtrip() {
        let config = McpServerConfig {
            name: "test-server".into(),
            command: Some("npx".into()),
            args: vec!["-y".into(), "@mcp/server".into()],
            url: None,
            env: HashMap::from([("API_KEY".into(), "secret".into())]),
            credentials_json: None,
        };
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: McpServerConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.name, "test-server");
        assert_eq!(parsed.args.len(), 2);
        assert_eq!(parsed.env.len(), 1);
    }

    #[test]
    fn test_load_configs_from_dir() {
        let dir = tempdir().unwrap();
        let config = McpServerConfig {
            name: "github".into(),
            command: Some("npx".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        let path = dir.path().join("github.toml");
        std::fs::write(&path, toml::to_string_pretty(&config).unwrap()).unwrap();
        let configs = McpServerConfig::load_all_from(dir.path());
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "github");
    }

    #[test]
    fn test_save_config() {
        let dir = tempdir().unwrap();
        let config = McpServerConfig {
            name: "postgres".into(),
            command: Some("mcp-postgres".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&config, dir.path()).unwrap();
        let path = dir.path().join("postgres.toml");
        assert!(path.exists());
    }

    #[test]
    fn test_delete_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(&path, "name = \"test\"").unwrap();
        McpServerConfig::delete_from("test", dir.path()).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_load_for_user_merges_org_and_personal() {
        let dir = tempdir().unwrap();
        let mcp_dir = dir.path().join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();

        let org_config = McpServerConfig {
            name: "org-server".into(),
            command: Some("org-cmd".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&org_config, &mcp_dir).unwrap();

        let user_dir = mcp_dir.join("users").join("alice");
        let user_config = McpServerConfig {
            name: "alice-server".into(),
            command: Some("alice-cmd".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&user_config, &user_dir).unwrap();

        let registry = McpRegistry::new(mcp_dir);
        let configs = registry.load_for_user("alice");
        assert_eq!(configs.len(), 2);
        let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"org-server"));
        assert!(names.contains(&"alice-server"));
    }

    #[test]
    fn test_user_config_overrides_org() {
        let dir = tempdir().unwrap();
        let mcp_dir = dir.path().join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();

        let org_config = McpServerConfig {
            name: "foo".into(),
            command: Some("org-cmd".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&org_config, &mcp_dir).unwrap();

        let user_dir = mcp_dir.join("users").join("bob");
        let user_config = McpServerConfig {
            name: "foo".into(),
            command: Some("bob-cmd".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&user_config, &user_dir).unwrap();

        let registry = McpRegistry::new(mcp_dir);
        let configs = registry.load_for_user("bob");
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].command.as_deref(), Some("bob-cmd"));
    }

    #[test]
    fn test_save_and_delete_for_user() {
        let dir = tempdir().unwrap();
        let mcp_dir = dir.path().join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();

        let registry = McpRegistry::new(mcp_dir.clone());

        let config = McpServerConfig {
            name: "my-server".into(),
            command: Some("cmd".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        registry.save_for_user("carol", &config).unwrap();

        let path = mcp_dir.join("users/carol/my-server.toml");
        assert!(path.exists());

        registry.delete_for_user("carol", "my-server").unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_user_isolation() {
        let dir = tempdir().unwrap();
        let mcp_dir = dir.path().join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();

        let registry = McpRegistry::new(mcp_dir.clone());

        let config_a = McpServerConfig {
            name: "server-a".into(),
            command: Some("cmd-a".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        registry.save_for_user("alice", &config_a).unwrap();

        let config_b = McpServerConfig {
            name: "server-b".into(),
            command: Some("cmd-b".into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
            credentials_json: None,
        };
        registry.save_for_user("bob", &config_b).unwrap();

        let alice_configs = registry.load_for_user("alice");
        let bob_configs = registry.load_for_user("bob");

        let alice_names: Vec<&str> = alice_configs.iter().map(|c| c.name.as_str()).collect();
        let bob_names: Vec<&str> = bob_configs.iter().map(|c| c.name.as_str()).collect();

        assert!(alice_names.contains(&"server-a"));
        assert!(!alice_names.contains(&"server-b"));
        assert!(bob_names.contains(&"server-b"));
        assert!(!bob_names.contains(&"server-a"));
    }

    #[test]
    fn test_credentials_json_roundtrip() {
        let dir = tempdir().unwrap();
        let config = McpServerConfig {
            name: "gmail".into(),
            command: None,
            args: vec![],
            url: Some("https://mcp.gmail.com".into()),
            env: HashMap::new(),
            credentials_json: Some(r#"{"client_id":"abc","token_response":null}"#.into()),
        };
        McpServerConfig::save_to(&config, dir.path()).unwrap();
        let configs = McpServerConfig::load_all_from(dir.path());
        assert_eq!(configs.len(), 1);
        assert_eq!(
            configs[0].credentials_json.as_deref(),
            Some(r#"{"client_id":"abc","token_response":null}"#)
        );
    }

    #[tokio::test]
    async fn test_file_credential_store_save_load_clear() {
        let dir = tempdir().unwrap();
        let config = McpServerConfig {
            name: "test-server".into(),
            command: None,
            args: vec![],
            url: Some("https://mcp.example.com".into()),
            env: HashMap::new(),
            credentials_json: None,
        };
        McpServerConfig::save_to(&config, dir.path()).unwrap();

        let store = FileCredentialStore {
            mcp_dir: dir.path().to_path_buf(),
            server_name: "test-server".into(),
        };

        assert!(store.load().await.unwrap().is_none());

        let creds_json = r#"{"client_id":"my-client","token_response":null,"granted_scopes":[]}"#;
        let creds: StoredCredentials = serde_json::from_str(creds_json).unwrap();
        store.save(creds).await.unwrap();

        let loaded = store.load().await.unwrap().unwrap();
        assert_eq!(loaded.client_id, "my-client");

        store.clear().await.unwrap();
        assert!(store.load().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_file_credential_store_user_isolation() {
        let dir = tempdir().unwrap();
        let mcp_dir = dir.path().join("mcp");

        let registry = McpRegistry::new(mcp_dir.clone());

        let config = McpServerConfig {
            name: "gmail".into(),
            command: None,
            args: vec![],
            url: Some("https://mcp.gmail.com".into()),
            env: HashMap::new(),
            credentials_json: Some(r#"{"client_id":"alice-token"}"#.into()),
        };
        registry.save_for_user("alice", &config).unwrap();

        let config_bob = McpServerConfig {
            name: "gmail".into(),
            command: None,
            args: vec![],
            url: Some("https://mcp.gmail.com".into()),
            env: HashMap::new(),
            credentials_json: None,
        };
        registry.save_for_user("bob", &config_bob).unwrap();

        let alice_store = FileCredentialStore {
            mcp_dir: mcp_dir.join("users/alice"),
            server_name: "gmail".into(),
        };
        let bob_store = FileCredentialStore {
            mcp_dir: mcp_dir.join("users/bob"),
            server_name: "gmail".into(),
        };

        let alice_creds = alice_store.load().await.unwrap();
        assert!(alice_creds.is_some());

        let bob_creds = bob_store.load().await.unwrap();
        assert!(bob_creds.is_none());
    }

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
