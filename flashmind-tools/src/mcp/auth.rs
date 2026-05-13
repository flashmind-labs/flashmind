use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde_json::Value;

use super::config::McpConfigProvider;

/// Outcome of an authentication attempt by the auth handler.
///
/// Returned by [`McpAuthHandler::authenticate`] to indicate whether
/// authentication was handled automatically or requires user interaction.
pub enum AuthOutcome {
    /// Authentication completed successfully. The server will be reconnected
    /// and its tools re-registered.
    Completed,
    /// User interaction is required (e.g. OAuth browser flow). The message
    /// is passed through as `ToolResult::interrupt` so the caller (REPL,
    /// web UI, etc.) can present it to the user.
    InteractionRequired { message: String },
}

/// Trait for handling MCP server authentication.
///
/// Implementations can try automatic methods (running a reauth command,
/// refreshing a token) and return [`AuthOutcome::Completed`] on success,
/// or [`AuthOutcome::InteractionRequired`] to pause the agent loop for
/// user-driven OAuth.
#[async_trait]
pub trait McpAuthHandler: Send + Sync {
    async fn authenticate(&self, server_name: &str) -> Result<AuthOutcome>;
}

/// Adapts [`McpConfigProvider`] to rmcp's [`CredentialStore`] trait.
pub(crate) struct ProviderCredentialStore {
    pub provider: Arc<dyn McpConfigProvider>,
    pub server_name: String,
}

#[async_trait]
impl CredentialStore for ProviderCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        let value = self
            .provider
            .load_credentials(&self.server_name)
            .await
            .map_err(|e| AuthError::InternalError(e.to_string()))?;
        match value {
            Some(v) if !v.is_null() => {
                let creds: StoredCredentials = serde_json::from_value(v)
                    .map_err(|e| AuthError::InternalError(e.to_string()))?;
                Ok(Some(creds))
            }
            _ => Ok(None),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        tracing::info!(server = %self.server_name, "persisting OAuth credentials to disk");
        let value = serde_json::to_value(&credentials)
            .map_err(|e| AuthError::InternalError(e.to_string()))?;
        self.provider
            .save_credentials(&self.server_name, &value)
            .await
            .map_err(|e| {
                tracing::error!(server = %self.server_name, error = %e, "failed to persist credentials");
                AuthError::InternalError(e.to_string())
            })
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        self.provider
            .save_credentials(&self.server_name, &Value::Null)
            .await
            .map_err(|e| AuthError::InternalError(e.to_string()))
    }
}

/// Newtype wrapper for passing `Arc<dyn CredentialStore>` where rmcp expects
/// an owned `impl CredentialStore`.
pub(crate) struct ArcCredentialStore(pub Arc<dyn CredentialStore>);

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
