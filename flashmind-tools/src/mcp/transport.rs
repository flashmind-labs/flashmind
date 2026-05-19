use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rmcp::ServiceExt;
use rmcp::model::{ClientCapabilities, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::auth::{AuthClient, CredentialStore, OAuthClientConfig, OAuthState};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use tokio::time::timeout;

use super::auth::{ArcCredentialStore, ProviderCredentialStore};
use super::config::McpConfigProvider;
use super::types::McpAuthRequired;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) type McpService = RunningService<RoleClient, rmcp::model::ClientInfo>;

pub(crate) fn client_info() -> rmcp::model::ClientInfo {
    rmcp::model::ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("flashmind", env!("CARGO_PKG_VERSION")),
    )
}

pub(crate) fn make_credential_store(
    provider: &Arc<dyn McpConfigProvider>,
    server_name: &str,
) -> Arc<dyn CredentialStore> {
    Arc::new(ProviderCredentialStore {
        provider: provider.clone(),
        server_name: server_name.to_string(),
    })
}

pub(crate) async fn connect_http(
    provider: &Arc<dyn McpConfigProvider>,
    server_name: &str,
    url: &str,
    client_secret: Option<&str>,
) -> Result<McpService> {
    let result = timeout(
        CONNECTION_TIMEOUT,
        try_connect_with_credentials(provider, server_name, url, client_secret),
    )
    .await;
    match result {
        Ok(Some(service)) => return Ok(service),
        Ok(None) => {}
        Err(_) => {
            tracing::warn!(server = %server_name, "MCP connection timed out (credentials)");
        }
    }

    let result = timeout(CONNECTION_TIMEOUT, try_connect_plain(url)).await;
    match result {
        Ok(Some(service)) => return Ok(service),
        Ok(None) => {}
        Err(_) => {
            tracing::warn!(server = %server_name, "MCP connection timed out (plain)");
        }
    }

    bail!(McpAuthRequired {
        server: server_name.to_string(),
    })
}

fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn try_connect_with_credentials(
    provider: &Arc<dyn McpConfigProvider>,
    server_name: &str,
    url: &str,
    client_secret: Option<&str>,
) -> Option<McpService> {
    let store = make_credential_store(provider, server_name);
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

    let Some(token_response) = creds.token_response else {
        tracing::warn!(server = %server_name, "stored credentials have no token_response");
        return None;
    };

    // `set_credentials` resets `token_received_at` to now, masking real
    // expiry. Check *before* calling it so we don't send a stale token.
    if let Some(received_at) = creds.token_received_at {
        let expires_in = serde_json::to_value(&token_response)
            .ok()
            .and_then(|v| v.get("expires_in")?.as_u64());
        if let Some(ttl) = expires_in {
            let elapsed = now_epoch_secs().saturating_sub(received_at);
            if elapsed >= ttl {
                let has_refresh = serde_json::to_value(&token_response)
                    .ok()
                    .and_then(|v| v.get("refresh_token")?.as_str().map(|_| ()))
                    .is_some();
                if !has_refresh {
                    tracing::info!(server = %server_name, "stored token expired with no refresh token");
                    let _ = store.clear().await;
                    return None;
                }
                tracing::debug!(server = %server_name, "stored token expired, will attempt refresh");
            }
        }
    }

    let mut oauth_state = match OAuthState::new(url, None).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "OAuth state init failed");
            return None;
        }
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

    // Reconfigure the OAuth client with the client_secret so token
    // refresh requests include proper client authentication.
    if let Some(secret) = client_secret {
        let client_config =
            OAuthClientConfig::new(&creds.client_id, url).with_client_secret(secret);
        if let Err(e) = mgr.configure_client(client_config) {
            tracing::warn!(server = %server_name, error = %e, "failed to reconfigure client with secret");
        }
    }

    mgr.set_credential_store(ArcCredentialStore(store));
    let auth_client = AuthClient::new(reqwest::Client::default(), mgr);
    let config = StreamableHttpClientTransportConfig::with_uri(url);
    let transport = StreamableHttpClientTransport::with_client(auth_client, config);

    match client_info().serve(transport).await {
        Ok(service) => Some(service),
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "connect with stored credentials failed");
            None
        }
    }
}

async fn try_connect_plain(url: &str) -> Option<McpService> {
    let config = StreamableHttpClientTransportConfig::with_uri(url);
    let transport = StreamableHttpClientTransport::from_config(config);

    match client_info().serve(transport).await {
        Ok(service) => Some(service),
        Err(e) => {
            tracing::debug!(error = %e, "unauthenticated connect failed");
            None
        }
    }
}

pub(crate) async fn connect_stdio(
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

    let service = client_info()
        .serve(transport)
        .await
        .context("MCP initialize failed")?;

    Ok(service)
}

pub(crate) fn resolve_command(command: &str) -> Option<String> {
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
