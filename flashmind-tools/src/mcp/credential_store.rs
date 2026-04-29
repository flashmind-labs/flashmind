//! Remote credential store that persists OAuth tokens on the daemon via HTTP.
//!
//! Implements rmcp's `CredentialStore` trait so token save/load/clear
//! go through `GET/PUT/DELETE /api/v2/oauth/tokens/{server}`,
//! scoped to the authenticated user's identity.

use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};

pub struct DaemonCredentialStore {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    mcp_server: String,
}

impl DaemonCredentialStore {
    pub fn new(base_url: String, api_key: String, mcp_server: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            mcp_server,
        }
    }

    fn url(&self) -> String {
        format!(
            "{}/api/v2/oauth/tokens/{}",
            self.base_url, self.mcp_server
        )
    }
}

#[async_trait::async_trait]
impl CredentialStore for DaemonCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let resp = self
            .client
            .get(self.url())
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| AuthError::InternalError(format!("credential load failed: {e}")))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !resp.status().is_success() {
            return Err(AuthError::InternalError(format!(
                "credential load: HTTP {}",
                resp.status()
            )));
        }

        #[derive(serde::Deserialize)]
        struct Resp {
            credentials: String,
        }

        let body: Resp = resp
            .json()
            .await
            .map_err(|e| AuthError::InternalError(format!("credential load parse: {e}")))?;

        let creds: StoredCredentials = serde_json::from_str(&body.credentials)
            .map_err(|e| AuthError::InternalError(format!("credential deserialize: {e}")))?;

        Ok(Some(creds))
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let json = serde_json::to_string(&credentials)
            .map_err(|e| AuthError::InternalError(format!("credential serialize: {e}")))?;

        let resp = self
            .client
            .put(self.url())
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({ "credentials": json }))
            .send()
            .await
            .map_err(|e| AuthError::InternalError(format!("credential save failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AuthError::InternalError(format!(
                "credential save: HTTP {}",
                resp.status()
            )));
        }

        Ok(())
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let resp = self
            .client
            .delete(self.url())
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| AuthError::InternalError(format!("credential clear failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AuthError::InternalError(format!(
                "credential clear: HTTP {}",
                resp.status()
            )));
        }

        Ok(())
    }
}
