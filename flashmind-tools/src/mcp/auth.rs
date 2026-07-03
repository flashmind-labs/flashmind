use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use oauth2::TokenResponse;
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

/// Credential store that persists only genuinely-new tokens on the live
/// connection, dropping proactive re-saves of the seed token.
///
/// The live `AuthClient` refreshes the access token mid-session. For providers
/// that rotate refresh tokens (e.g. Fastmail's ratchet), that refresh returns a
/// new refresh token which MUST be persisted, or the on-disk credential falls
/// behind the server and is rejected on the next launch. rmcp persists via an
/// unconditional whole-file overwrite, so a stale/background process must not be
/// allowed to write — that would clobber fresher credentials another process
/// wrote (last-writer-wins race).
///
/// This store reconciles both: it remembers the access token it was seeded with
/// and writes a `save` through only when the incoming token differs from the
/// seed. A differing token can only come from a successful server refresh, which
/// (for a ratcheting provider) succeeds only from the server's current
/// generation — so it is provably the freshest and safe to persist. A stale
/// process only ever holds its seed, so its proactive save equals the seed and
/// is dropped. `clear` stays a no-op so a stale process can't wipe credentials.
pub(crate) struct RefreshCapturingStore {
    inner: Arc<dyn CredentialStore>,
    seed_access_token: Option<String>,
}

impl RefreshCapturingStore {
    pub(crate) fn new(inner: Arc<dyn CredentialStore>, seed_access_token: Option<String>) -> Self {
        Self {
            inner,
            seed_access_token,
        }
    }
}

#[async_trait]
impl CredentialStore for RefreshCapturingStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        self.inner.load().await
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let incoming = credentials
            .token_response
            .as_ref()
            .map(|t| t.access_token().secret().to_string());

        if incoming.is_some() && incoming == self.seed_access_token {
            tracing::debug!("ignoring proactive re-save of seed token on connection store");
            return Ok(());
        }

        tracing::info!("persisting rotated OAuth token from live-session refresh");
        self.inner.save(credentials).await
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        tracing::debug!("ignoring credential clear on connection store");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Counting credential store: records save/clear counts and the last
    /// credentials saved, so tests can assert what reached the provider.
    #[derive(Default)]
    struct CountingStore {
        saves: AtomicUsize,
        clears: AtomicUsize,
        creds: Option<StoredCredentials>,
        last_saved: Mutex<Option<StoredCredentials>>,
    }

    #[async_trait]
    impl CredentialStore for CountingStore {
        async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
            Ok(self.creds.clone())
        }

        async fn save(&self, creds: StoredCredentials) -> std::result::Result<(), AuthError> {
            self.saves.fetch_add(1, Ordering::SeqCst);
            *self.last_saved.lock().unwrap() = Some(creds);
            Ok(())
        }

        async fn clear(&self) -> std::result::Result<(), AuthError> {
            self.clears.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Build a `StoredCredentials` carrying a token with the given access token,
    /// via the same JSON shape `ProviderCredentialStore::load` deserializes.
    fn creds_with_access_token(access_token: &str) -> StoredCredentials {
        let value = serde_json::json!({
            "client_id": "client",
            "granted_scopes": [],
            "token_received_at": 0,
            "token_response": {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": format!("refresh-for-{access_token}"),
            }
        });
        serde_json::from_value(value).expect("valid stored credentials")
    }

    #[tokio::test]
    async fn drops_save_equal_to_seed() {
        let inner = Arc::new(CountingStore::default());
        let store = RefreshCapturingStore::new(inner.clone(), Some("seed-token".into()));

        // rmcp proactively re-saves the seed token: no new info, must be dropped.
        store
            .save(creds_with_access_token("seed-token"))
            .await
            .unwrap();

        assert_eq!(
            inner.saves.load(Ordering::SeqCst),
            0,
            "a save equal to the seed must not reach the provider"
        );
    }

    #[tokio::test]
    async fn persists_save_differing_from_seed() {
        let inner = Arc::new(CountingStore::default());
        let store = RefreshCapturingStore::new(inner.clone(), Some("seed-token".into()));

        // rmcp refreshed mid-session: token differs from the seed, must persist.
        store
            .save(creds_with_access_token("rotated-token"))
            .await
            .unwrap();

        assert_eq!(
            inner.saves.load(Ordering::SeqCst),
            1,
            "a rotated token must be written through"
        );
        let saved = inner.last_saved.lock().unwrap().clone().unwrap();
        let saved_access = saved
            .token_response
            .as_ref()
            .map(|t| t.access_token().secret().to_string());
        assert_eq!(saved_access.as_deref(), Some("rotated-token"));
    }

    #[tokio::test]
    async fn persists_when_seed_is_none() {
        let inner = Arc::new(CountingStore::default());
        let store = RefreshCapturingStore::new(inner.clone(), None);

        // No seed to compare against: any real token is written through.
        store
            .save(creds_with_access_token("some-token"))
            .await
            .unwrap();

        assert_eq!(inner.saves.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn clear_never_reaches_provider() {
        let inner = Arc::new(CountingStore::default());
        let store = RefreshCapturingStore::new(inner.clone(), Some("seed-token".into()));

        store.clear().await.unwrap();

        assert_eq!(
            inner.clears.load(Ordering::SeqCst),
            0,
            "clear must stay a no-op so a stale process can't wipe creds"
        );
    }
}
