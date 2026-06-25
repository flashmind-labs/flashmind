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

/// Credential store that delegates `load` but drops `save`/`clear`.
///
/// Used for the connection-time `AuthClient`. rmcp may proactively persist the
/// in-memory credentials it was seeded with during a connect attempt. When a
/// process is carrying a stale token (e.g. a long-running chat session whose
/// token was invalidated, or one that re-auth happened in *another* process),
/// that proactive save does an unconditional whole-file overwrite and clobbers
/// fresh credentials another process just wrote — a last-writer-wins race.
///
/// Wrapping the connect-time store in this read-only adapter makes the clobber
/// impossible: a doomed/background connection can read credentials but never
/// write them. Persistence happens only on the two explicit paths that produce
/// a genuinely new token — OAuth completion (`complete_auth`) and the
/// successful pre-connect refresh in `try_connect_with_credentials`.
pub(crate) struct ReadOnlyCredentialStore(pub Arc<dyn CredentialStore>);

#[async_trait]
impl CredentialStore for ReadOnlyCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        self.0.load().await
    }

    async fn save(&self, _credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        tracing::debug!("ignoring credential save on read-only connection store");
        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        tracing::debug!("ignoring credential clear on read-only connection store");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Counting credential store: records every save/clear so tests can assert
    /// whether a write reached the underlying provider.
    #[derive(Default)]
    struct CountingStore {
        saves: AtomicUsize,
        clears: AtomicUsize,
        creds: Option<StoredCredentials>,
    }

    #[async_trait]
    impl CredentialStore for CountingStore {
        async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
            Ok(self.creds.clone())
        }

        async fn save(&self, _: StoredCredentials) -> std::result::Result<(), AuthError> {
            self.saves.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn clear(&self) -> std::result::Result<(), AuthError> {
            self.clears.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn read_only_store_does_not_write_through() {
        let inner = Arc::new(CountingStore {
            creds: Some(StoredCredentials::new(
                "client".into(),
                None,
                vec![],
                Some(42),
            )),
            ..Default::default()
        });
        let store = ReadOnlyCredentialStore(inner.clone());

        // load delegates to the inner store…
        let loaded = store.load().await.unwrap();
        assert!(loaded.is_some(), "load should delegate to the inner store");

        // …but save and clear are dropped so a stale process can't clobber.
        store
            .save(StoredCredentials::new(
                "client".into(),
                None,
                vec![],
                Some(1),
            ))
            .await
            .unwrap();
        store.clear().await.unwrap();

        assert_eq!(
            inner.saves.load(Ordering::SeqCst),
            0,
            "save must not reach provider"
        );
        assert_eq!(
            inner.clears.load(Ordering::SeqCst),
            0,
            "clear must not reach provider"
        );
    }
}
