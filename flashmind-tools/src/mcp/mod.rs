pub mod client;
pub mod tools;
pub mod wire;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use self::client::McpClient;
use self::wire::{McpToolDef, ToolCallResult};

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
}

impl McpServerConfig {
    /// Read all `.toml` files from `dir` and parse each as `McpServerConfig`.
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
                    warn!(path = %path.display(), error = %e, "failed to load MCP server config")
                }
            }
        }
        configs
    }

    /// Write `config` as `{name}.toml` into `dir`, creating the directory if needed.
    pub fn save_to(config: &McpServerConfig, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.toml", config.name));
        let content = toml::to_string_pretty(config)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Remove `{name}.toml` from `dir` if it exists.
    pub fn delete_from(name: &str, dir: &Path) -> Result<()> {
        let path = dir.join(format!("{name}.toml"));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }
}

/// An active connection to a single MCP server.
pub struct McpConnection {
    pub config: McpServerConfig,
    pub client: McpClient,
    pub tool_names: Vec<String>,
}

/// Registry storing both registered MCP server configs and active connections.
#[derive(Clone)]
pub struct McpRegistry {
    /// Directory where MCP server configs are persisted.
    mcp_dir: PathBuf,
    /// Registered server configs (persisted on disk).
    configs: Arc<Mutex<HashMap<String, McpServerConfig>>>,
    /// Active connections (in-memory only).
    connections: Arc<Mutex<HashMap<String, McpConnection>>>,
}

impl McpRegistry {
    pub fn new(mcp_dir: PathBuf) -> Self {
        Self {
            mcp_dir,
            configs: Arc::new(Mutex::new(HashMap::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load all saved configs from disk.
    pub async fn load_saved(&self) {
        let saved = McpServerConfig::load_all_from(&self.mcp_dir);
        let mut configs = self.configs.lock().await;
        for cfg in saved {
            configs.insert(cfg.name.clone(), cfg);
        }
    }
}

impl McpRegistry {
    /// Add/register an MCP server config without connecting.
    /// This saves the config to disk - connection happens on first mcp_run.
    pub async fn add(&self, config: McpServerConfig) -> Result<()> {
        let mut configs = self.configs.lock().await;
        McpServerConfig::save_to(&config, &self.mcp_dir)?;
        configs.insert(config.name.clone(), config);
        Ok(())
    }

    /// Connect to an MCP server, initialize it, and persist its config.
    ///
    /// Returns the list of tools advertised by the server.
    pub async fn connect(&self, config: McpServerConfig) -> Result<Vec<McpToolDef>> {
        // Spawn/connect outside of any lock — this can be slow
        let mut client = if let Some(ref url) = config.url {
            McpClient::connect_sse(url).await?
        } else if let Some(ref command) = config.command {
            let args: Vec<&str> = config.args.iter().map(String::as_str).collect();
            let env: Vec<(&str, &str)> = config
                .env
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            McpClient::spawn_stdio(command, &args, &env).await?
        } else {
            bail!("MCP server '{}' has neither command nor url", config.name);
        };

        client.initialize().await?;

        let tools = client.tools.clone();
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();

        info!(
            server = %config.name,
            tool_count = tool_names.len(),
            "MCP server connected"
        );

        // Persist config
        McpServerConfig::save_to(&config, &self.mcp_dir)?;
        self.configs
            .lock()
            .await
            .insert(config.name.clone(), config.clone());

        // Register connection — if a duplicate raced us, shut down the old one
        let mut conns = self.connections.lock().await;
        if let Some(mut old) = conns.remove(&config.name) {
            warn!(server = %config.name, "Replacing existing MCP connection (concurrent connect)");
            old.client.shutdown().await;
        }

        conns.insert(
            config.name.clone(),
            McpConnection {
                config,
                client,
                tool_names,
            },
        );

        Ok(tools)
    }

    /// Remove an MCP server — disconnect if active and delete the saved config.
    pub async fn remove(&self, name: &str) -> Result<()> {
        // Disconnect if active
        {
            let mut conns = self.connections.lock().await;
            if let Some(mut conn) = conns.remove(name) {
                conn.client.shutdown().await;
            }
        }

        // Remove from in-memory configs
        self.configs.lock().await.remove(name);

        // Delete from disk
        McpServerConfig::delete_from(name, &self.mcp_dir)?;

        info!(server = %name, "MCP server removed");
        Ok(())
    }

    /// List tools available on a server. Auto-connects if needed.
    pub async fn list_tools(&self, server_name: &str) -> Result<Vec<McpToolDef>> {
        self.ensure_connected(server_name).await?;

        let conns = self.connections.lock().await;
        let conn = conns
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' is not registered"))?;
        Ok(conn.client.tools.clone())
    }

    /// Ensure a server is connected, auto-connecting from saved config if needed.
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
            info!(server = %server_name, "Auto-connecting to MCP server");
            self.connect(config).await?;
            Ok(())
        } else {
            bail!("MCP server '{server_name}' is not registered. Use mcp_add first.")
        }
    }

    /// Invoke a tool on the named server. Auto-connects if not already connected.
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        self.ensure_connected(server_name).await?;

        let mut conns = self.connections.lock().await;
        let conn = conns
            .get_mut(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' is not connected"))?;
        conn.client.call_tool(tool_name, arguments).await
    }

    /// List all registered servers with their tools.
    ///
    /// Auto-connects disconnected servers so tool lists are always available.
    /// Servers that fail to connect are listed with an error note.
    pub async fn list(&self) -> Vec<(String, Vec<String>, Option<String>)> {
        // Collect configs that need connecting
        let to_connect: Vec<McpServerConfig> = {
            let configs = self.configs.lock().await;
            let conns = self.connections.lock().await;
            configs
                .values()
                .filter(|c| !conns.contains_key(&c.name))
                .cloned()
                .collect()
        };

        // Auto-connect each (errors are captured, not propagated)
        let mut errors: HashMap<String, String> = HashMap::new();
        for config in to_connect {
            let name = config.name.clone();
            if let Err(e) = self.connect(config).await {
                errors.insert(name, e.to_string());
            }
        }

        // Build result from connections + any errors
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

    /// Shut down all active connections.
    pub async fn shutdown_all(&self) {
        let mut conns = self.connections.lock().await;
        for (name, mut conn) in conns.drain() {
            conn.client.shutdown().await;
            info!(server = %name, "MCP server shut down");
        }
    }
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

    #[tokio::test]
    async fn test_mcp_connect_call_disconnect() {
        let dir = tempdir().unwrap();

        // Write mock MCP server script
        let script = dir.path().join("mock_mcp.sh");
        std::fs::write(
            &script,
            r#"#!/bin/bash
while IFS= read -r line; do
    read method id <<< $(echo "$line" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d.get('method',''), d.get('id',''))
except: pass
" 2>/dev/null)
    [ -z "$id" ] && continue
    case "$method" in
        initialize)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"serverInfo\":{\"name\":\"mock\",\"version\":\"1.0\"}}}"
            ;;
        tools/list)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo input\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}]}}"
            ;;
        tools/call)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"echoed\"}]}}"
            ;;
    esac
done
"#,
        )
        .unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let registry = McpRegistry::new(dir.path().join("mcp"));

        // Connect
        let config = McpServerConfig {
            name: "mock".into(),
            command: Some(script.to_str().unwrap().into()),
            args: vec![],
            url: None,
            env: HashMap::new(),
        };
        let tools = registry.connect(config).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        // Call tool
        let result = registry
            .call_tool("mock", "echo", serde_json::json!({"text": "hi"}))
            .await
            .unwrap();
        assert_eq!(result.content.len(), 1);
        assert!(!result.is_error);

        // Remove (disconnect + delete config)
        registry.remove("mock").await.unwrap();
    }

    /// Load MCP configs from a directory and connect to each one, verifying
    /// the full config→spawn→initialize→tool-discovery pipeline.
    #[tokio::test]
    async fn test_load_configs_and_connect() {
        let dir = tempdir().unwrap();

        // Write a mock MCP server script that speaks JSON-RPC.
        let script = dir.path().join("server.sh");
        std::fs::write(
            &script,
            r#"#!/bin/bash
while IFS= read -r line; do
    read method id <<< $(echo "$line" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d.get('method',''), d.get('id',''))
except: pass
" 2>/dev/null)
    [ -z "$id" ] && continue
    case "$method" in
        initialize)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"serverInfo\":{\"name\":\"test\",\"version\":\"1.0\"}}}"
            ;;
        tools/list)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"ping\",\"description\":\"Ping\",\"inputSchema\":{\"type\":\"object\",\"properties\":{}}}]}}"
            ;;
        tools/call)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"pong\"}]}}"
            ;;
    esac
done
"#,
        )
        .unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        // Write a TOML config pointing at the script.
        let config_dir = dir.path().join("mcp");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::write(
            config_dir.join("test-server.toml"),
            format!(
                "name = \"test-server\"\ncommand = \"{}\"\nargs = []\n",
                script.display()
            ),
        )
        .unwrap();

        // Load configs from disk, connect each, and call a tool.
        let configs = McpServerConfig::load_all_from(&config_dir);
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "test-server");

        let registry = McpRegistry::new(config_dir);

        for cfg in configs {
            let tools = registry.connect(cfg).await.unwrap();
            assert!(
                !tools.is_empty(),
                "server should advertise at least one tool"
            );
        }

        // Verify tool invocation works.
        let result = registry
            .call_tool("test-server", "ping", serde_json::json!({}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content[0].as_text().unwrap(), "pong");

        // Clean up.
        registry.remove("test-server").await.unwrap();
    }
}
