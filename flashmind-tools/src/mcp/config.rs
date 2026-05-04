use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::types::McpToolDef;

/// Trait for persisting MCP server configurations and credentials.
///
/// Implementations can store data on disk, in a database, in the cloud, etc.
/// See [`McpDiskConfig`] for a ready-to-use filesystem-backed implementation.
#[async_trait]
pub trait McpConfigProvider: Send + Sync {
    /// List all saved MCP server configurations.
    async fn list_configs(&self) -> Result<Vec<McpServerConfig>>;
    /// Save or update an MCP server configuration.
    async fn save_config(&self, config: &McpServerConfig) -> Result<()>;
    /// Delete an MCP server configuration by name.
    async fn delete_config(&self, name: &str) -> Result<()>;
    /// Save OAuth credentials for a server.
    async fn save_credentials(&self, name: &str, credentials: &Value) -> Result<()>;
    /// Load stored OAuth credentials for a server.
    async fn load_credentials(&self, name: &str) -> Result<Option<Value>>;
}

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
    /// Stored OAuth credentials from a previous auth session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<Value>,
    /// Reauthentication hint/timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reauth: Option<String>,
    /// Cached tool definitions from the last successful connection (used at startup).
    #[serde(default)]
    pub cached_tools: Vec<McpToolDef>,
}

/// Filesystem-backed [`McpConfigProvider`].
///
/// Stores one JSON file per server at `{base_dir}/{name}.json`.
/// Each file contains the full [`McpServerConfig`] including credentials
/// and cached tools.
pub struct McpDiskConfig {
    base_dir: PathBuf,
}

impl McpDiskConfig {
    pub fn new(base_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self { base_dir })
    }

    fn path_for(&self, name: &str) -> PathBuf {
        self.base_dir.join(format!("{name}.json"))
    }

    async fn load_config(&self, name: &str) -> Result<McpServerConfig> {
        let data = tokio::fs::read_to_string(self.path_for(name)).await?;
        Ok(serde_json::from_str(&data)?)
    }
}

#[async_trait]
impl McpConfigProvider for McpDiskConfig {
    async fn list_configs(&self) -> Result<Vec<McpServerConfig>> {
        let mut configs = Vec::new();

        let mut entries = match tokio::fs::read_dir(&self.base_dir).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(configs),
            Err(e) => return Err(e.into()),
        };

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                match tokio::fs::read_to_string(&path).await {
                    Ok(data) => match serde_json::from_str(&data) {
                        Ok(config) => configs.push(config),
                        Err(e) => {
                            tracing::warn!(path = %path.display(), error = %e, "skipping malformed MCP config");
                        }
                    },
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to read MCP config");
                    }
                }
            }
        }

        Ok(configs)
    }

    async fn save_config(&self, config: &McpServerConfig) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        let data = serde_json::to_string_pretty(config)?;
        tokio::fs::write(self.path_for(&config.name), data).await?;
        Ok(())
    }

    async fn delete_config(&self, name: &str) -> Result<()> {
        match tokio::fs::remove_file(self.path_for(name)).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    async fn save_credentials(&self, name: &str, credentials: &Value) -> Result<()> {
        let mut config = self.load_config(name).await?;
        config.credentials = Some(credentials.clone());
        self.save_config(&config).await
    }

    async fn load_credentials(&self, name: &str) -> Result<Option<Value>> {
        match self.load_config(name).await {
            Ok(config) => Ok(config.credentials),
            Err(e) => {
                tracing::debug!(server = %name, error = %e, "no config file for credentials lookup");
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_disk_config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let provider = McpDiskConfig::new(dir.path().to_path_buf());

        let config = McpServerConfig {
            name: "test-server".into(),
            command: Some("echo".into()),
            args: vec!["hello".into()],
            url: None,
            env: HashMap::new(),
            client_id: None,
            client_secret: None,
            scopes: vec![],
            credentials: None,
            reauth: None,
            cached_tools: vec![],
        };

        provider.save_config(&config).await.unwrap();

        let configs = provider.list_configs().await.unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "test-server");
        assert_eq!(configs[0].command.as_deref(), Some("echo"));

        let creds = serde_json::json!({"token": "secret"});
        provider
            .save_credentials("test-server", &creds)
            .await
            .unwrap();

        let loaded = provider.load_credentials("test-server").await.unwrap();
        assert_eq!(loaded, Some(creds));

        provider.delete_config("test-server").await.unwrap();
        let configs = provider.list_configs().await.unwrap();
        assert!(configs.is_empty());
    }

    #[tokio::test]
    async fn test_disk_config_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let provider = McpDiskConfig::new(dir.path().join("nonexistent"));
        let configs = provider.list_configs().await.unwrap();
        assert!(configs.is_empty());
    }
}
