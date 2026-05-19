use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use rmcp::model::CallToolRequestParams;
use rmcp::transport::auth::{
    AuthorizationSession, OAuthClientConfig, OAuthState, StoredCredentials,
};
use tokio::sync::Mutex as AsyncMutex;

use super::Host;
use super::auth::{ArcCredentialStore, AuthOutcome, McpAuthHandler};
use super::config::{McpConfigProvider, McpServerConfig};
use super::transport::{self, McpService, make_credential_store};
use super::types::{
    McpToolCallResult, McpToolDef, McpToolOp, call_result_from_rmcp, tool_def_from_rmcp,
};

struct McpConnection {
    #[allow(dead_code)]
    config: McpServerConfig,
    service: McpService,
    tool_names: Vec<String>,
    #[allow(dead_code)]
    tool_defs: Vec<McpToolDef>,
}

/// Registry managing MCP server connections and their tool definitions.
///
/// Servers are added/removed at runtime via `mcp_add`/`mcp_remove` tools.
/// Each server's tools are wrapped in
/// [`McpToolWrapper`](super::tools::McpToolWrapper) and registered in the
/// agent's tool registry.
#[derive(Clone)]
pub struct McpRegistry {
    provider: Arc<dyn McpConfigProvider>,
    auth_handler: Option<Arc<dyn McpAuthHandler>>,
    configs: Arc<AsyncMutex<HashMap<Host, McpServerConfig>>>,
    connections: Arc<AsyncMutex<HashMap<Host, McpConnection>>>,
    pending_ops: Arc<Mutex<Vec<McpToolOp>>>,
    current_tools: Arc<Mutex<HashMap<Host, Vec<McpToolDef>>>>,
    pending_auth: Arc<AsyncMutex<HashMap<Host, OAuthState>>>,
}

impl McpRegistry {
    /// Create a new registry backed by the given config provider.
    ///
    /// The optional `auth_handler` is called when a tool invocation discovers
    /// the server needs authentication (e.g. expired OAuth token). Pass `None`
    /// to fall back to [`AuthOutcome::InteractionRequired`], which lets the
    /// caller (REPL, CLI) drive the flow.
    ///
    /// ```ignore
    /// let provider = McpDiskConfig::new("~/.flashmind/mcp");
    /// let registry = McpRegistry::new(Arc::new(provider), None);
    /// registry.load_saved().await;
    /// ```
    pub fn new(
        provider: Arc<dyn McpConfigProvider>,
        auth_handler: Option<Arc<dyn McpAuthHandler>>,
    ) -> Self {
        Self {
            provider,
            auth_handler,
            configs: Arc::new(AsyncMutex::new(HashMap::new())),
            connections: Arc::new(AsyncMutex::new(HashMap::new())),
            pending_ops: Arc::new(Mutex::new(Vec::new())),
            current_tools: Arc::new(Mutex::new(HashMap::new())),
            pending_auth: Arc::new(AsyncMutex::new(HashMap::new())),
        }
    }

    // -- Startup & config management ------------------------------------------

    /// Load all server configs from the provider and start connecting.
    ///
    /// Servers with cached tool definitions are registered immediately so the
    /// agent can see them while the real connections happen in the background.
    /// Each server is connected in a spawned task — failures are logged but
    /// do not prevent other servers from connecting.
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

    /// Persist a server config and register it in the in-memory map.
    ///
    /// This does **not** connect — call [`reconnect`](Self::reconnect) or
    /// [`connect`](Self::connect) afterwards if you want to establish the
    /// connection immediately.
    ///
    /// ```ignore
    /// registry.add(config).await?;
    /// let tools = registry.reconnect("my-server").await?;
    /// println!("{} tools available", tools.len());
    /// ```
    pub async fn add(&self, config: McpServerConfig) -> Result<()> {
        self.provider.save_config(&config).await?;
        self.configs
            .lock()
            .await
            .insert(config.name.clone(), config);
        Ok(())
    }

    /// Remove a server: disconnect, delete config from disk, and queue an
    /// [`McpToolOp::Unregister`] so the agent drops its tool wrappers.
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

    /// Take all pending tool registration/unregistration ops.
    ///
    /// Called by [`ToolSync`](crate::tool_sync::ToolSync) each agent turn to
    /// apply MCP tool changes to the agent's [`ToolRegistry`](flashmind_types::tool::ToolRegistry).
    pub fn drain_pending_ops(&self) -> Vec<McpToolOp> {
        std::mem::take(&mut *self.pending_ops.lock().unwrap())
    }

    /// Snapshot of all currently known tools, keyed by server name.
    ///
    /// Includes both live tools from connected servers and cached tools from
    /// servers that haven't connected yet.
    pub fn current_mcp_tools(&self) -> HashMap<Host, Vec<McpToolDef>> {
        self.current_tools.lock().unwrap().clone()
    }

    /// Look up a server's config by name. Returns `None` if not registered.
    pub async fn get_config(&self, server_name: &str) -> Option<McpServerConfig> {
        self.configs.lock().await.get(server_name).cloned()
    }

    // -- Connection management ------------------------------------------------

    /// Open a transport to the server, list its tools, and store the connection.
    ///
    /// For HTTP servers with stored OAuth credentials, the transport is created
    /// with an authenticated client. For stdio servers, the command is spawned
    /// as a child process.
    ///
    /// On success, queues an [`McpToolOp::Register`] and caches the tool
    /// definitions to disk so they're available on next startup.
    pub async fn connect(&self, config: McpServerConfig) -> Result<Vec<McpToolDef>> {
        let service = self.open_transport(&config).await?;
        let tool_defs = Self::list_server_tools(&service).await?;
        let tool_names: Vec<String> = tool_defs.iter().map(|t| t.name.clone()).collect();

        tracing::info!(
            server = %config.name, tool_count = tool_names.len(),
            "MCP server connected"
        );

        self.store_connection(&config, service, tool_names, &tool_defs)
            .await;

        Ok(tool_defs)
    }

    /// Invoke a tool on a server, auto-connecting if needed.
    ///
    /// If the first call fails (e.g. stale connection), drops the connection
    /// and retries once with a fresh transport.
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

    /// List tools from a single server, auto-connecting if needed.
    ///
    /// Unlike [`current_mcp_tools`](Self::current_mcp_tools), this always
    /// queries the live server rather than returning cached definitions.
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

    /// List all registered servers with their tool names and connection errors.
    ///
    /// Attempts to connect any servers that aren't connected yet. Returns
    /// `(server_name, tool_names, optional_error)` for each server.
    pub async fn list(&self) -> Vec<(Host, Vec<String>, Option<String>)> {
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

        let mut result: Vec<(Host, Vec<String>, Option<String>)> = Vec::new();
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

    /// Gracefully cancel all active server connections.
    pub async fn shutdown_all(&self) {
        let mut conns = self.connections.lock().await;
        for (name, conn) in conns.drain() {
            conn.service.cancel().await.ok();
            tracing::info!(server = %name, "MCP server shut down");
        }
    }

    /// Disconnect all servers and reconnect from scratch.
    ///
    /// Useful after config changes or when the agent wants a clean slate.
    /// Returns per-server results so the caller can report failures.
    pub async fn refresh_all(&self) -> Vec<(Host, Result<Vec<McpToolDef>>)> {
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

    /// Drop the current connection (if any) and reconnect a single server.
    ///
    /// ```ignore
    /// // After a reauth command succeeds:
    /// let tools = registry.reconnect("gmail").await?;
    /// println!("reconnected with {} tools", tools.len());
    /// ```
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

    // -- Auth -----------------------------------------------------------------

    /// Try to authenticate with a server via the auth handler.
    ///
    /// If an [`McpAuthHandler`] was provided, delegates to it. Otherwise
    /// falls back to [`AuthOutcome::InteractionRequired`].
    /// On [`AuthOutcome::Completed`], automatically reconnects the server.
    pub async fn authenticate(&self, server_name: &str) -> Result<AuthOutcome> {
        let outcome = match &self.auth_handler {
            Some(handler) => handler.authenticate(server_name).await?,
            None => AuthOutcome::InteractionRequired {
                message: serde_json::json!({"server": server_name}).to_string(),
            },
        };

        if matches!(outcome, AuthOutcome::Completed) {
            self.drop_connection(server_name).await;
            let config = self
                .configs
                .lock()
                .await
                .get(server_name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' not found"))?;
            self.connect(config).await?;
            tracing::info!(server = %server_name, "re-connected after authentication");
        }

        Ok(outcome)
    }

    /// Begin an OAuth authorization flow for an HTTP server.
    ///
    /// Discovers the server's OAuth metadata, registers a dynamic client (or
    /// uses a pre-configured `client_id`), and returns the authorization URL
    /// to open in a browser. The caller should redirect the user there, then
    /// call [`complete_auth`](Self::complete_auth) with the callback params.
    ///
    /// ```ignore
    /// let url = registry.start_auth("fastmail", "http://localhost:19836/callback").await?;
    /// open_browser(&url);
    /// // ... wait for callback with code + state ...
    /// registry.complete_auth("fastmail", &code, &state).await?;
    /// ```
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

        let store = make_credential_store(&self.provider, server_name);
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

    /// Complete an OAuth flow started by [`start_auth`](Self::start_auth).
    ///
    /// Exchanges the authorization `code` for tokens, persists the credentials
    /// to the config provider, and connects the server. After this succeeds,
    /// the server's tools are available via [`call_tool`](Self::call_tool).
    pub async fn complete_auth(&self, server_name: &str, code: &str, state: &str) -> Result<()> {
        let mut oauth_state = self
            .pending_auth
            .lock()
            .await
            .remove(server_name)
            .ok_or_else(|| anyhow::anyhow!("no pending OAuth flow for server '{server_name}'"))?;

        tracing::debug!(server = %server_name, code_len = code.len(), state_len = state.len(), "exchanging OAuth code for token");
        oauth_state
            .handle_callback(code, state)
            .await
            .map_err(|e| anyhow::anyhow!("OAuth token exchange for '{server_name}': {e}"))?;

        // Persist credentials from the now-authorized state.
        // We must also update the in-memory config so that `connect` →
        // `cache_tools_to_config` doesn't overwrite the file without them.
        match oauth_state.get_credentials().await {
            Ok((client_id, token_response)) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let stored = StoredCredentials::new(client_id, token_response, vec![], Some(now));
                let value = serde_json::to_value(&stored).context("serialize OAuth credentials")?;
                self.provider
                    .save_credentials(server_name, &value)
                    .await
                    .unwrap_or_else(|e| {
                        tracing::warn!(server = %server_name, error = %e, "failed to persist OAuth credentials");
                    });
                if let Some(entry) = self.configs.lock().await.get_mut(server_name) {
                    entry.credentials = Some(value);
                }
            }
            Err(e) => {
                tracing::error!(server = %server_name, error = %e, "failed to read credentials from OAuth state after successful token exchange");
            }
        }

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

        if let Ok(mut guard) = self.configs.try_lock()
            && let Some(entry) = guard.get_mut(&config.name)
        {
            entry.cached_tools = tool_defs.to_vec();
        }
    }

    async fn open_transport(&self, config: &McpServerConfig) -> Result<McpService> {
        if let Some(ref url) = config.url {
            transport::connect_http(
                &self.provider,
                &config.name,
                url,
                config.client_secret.as_deref(),
                !config.scopes.is_empty(),
            )
            .await
        } else if let Some(ref command) = config.command {
            transport::connect_stdio(command, &config.args, &config.env).await
        } else {
            bail!("MCP server '{}' has neither command nor url", config.name);
        }
    }

    async fn list_server_tools(service: &McpService) -> Result<Vec<McpToolDef>> {
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
}
