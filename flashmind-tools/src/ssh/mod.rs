//! SSH remote execution tools.
//!
//! Provides tools for executing commands, uploading, and downloading files on
//! remote hosts via SSH using the [`russh`] crate.  Connections are configured
//! through named [`SshProfile`]s stored in an [`SshConfig`].

pub mod tools;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, bail};

// ---------------------------------------------------------------------------
// Auth / Profile / Config
// ---------------------------------------------------------------------------

/// Authentication method for SSH connections.
#[derive(Clone)]
pub enum SshAuth {
    /// Authenticate with a private key file.
    Key {
        /// Path to the private key.
        path: PathBuf,
        /// Optional passphrase protecting the key.
        passphrase: Option<String>,
    },
    /// Authenticate with a password.
    Password {
        /// The password.
        password: String,
    },
    /// Use the system SSH agent (reads `SSH_AUTH_SOCK`).
    Agent,
}

/// A named SSH connection profile.
#[derive(Clone)]
pub struct SshProfile {
    /// Human-readable profile name used to select the target host.
    pub name: String,
    /// Hostname or IP address.
    pub host: String,
    /// TCP port (usually 22).
    pub port: u16,
    /// Remote username.
    pub user: String,
    /// Authentication method.
    pub auth: SshAuth,
}

/// Configuration for SSH tools.
pub struct SshConfig {
    /// Available connection profiles.
    pub profiles: Vec<SshProfile>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Minimal [`russh::client::Handler`] that accepts all host keys.
pub(crate) struct SshHandler;

#[async_trait::async_trait]
impl russh::client::Handler for SshHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // Accept all keys (equivalent to `StrictHostKeyChecking=no`).
        // In production this should verify against known_hosts.
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// connect()
// ---------------------------------------------------------------------------

/// Open an authenticated SSH session to `profile`.
pub(crate) async fn connect(
    profile: &SshProfile,
) -> anyhow::Result<russh::client::Handle<SshHandler>> {
    let config = Arc::new(russh::client::Config::default());

    let handler = SshHandler;
    let mut session =
        russh::client::connect(config, (profile.host.as_str(), profile.port), handler)
            .await
            .with_context(|| format!("SSH connect to {}:{}", profile.host, profile.port))?;

    let authenticated = match &profile.auth {
        SshAuth::Key { path, passphrase } => {
            let key = russh_keys::load_secret_key(path, passphrase.as_deref())
                .with_context(|| format!("Load SSH key from {}", path.display()))?;
            session
                .authenticate_publickey(&profile.user, Arc::new(key))
                .await
                .context("Public-key authentication")?
        }
        SshAuth::Password { password } => session
            .authenticate_password(&profile.user, password)
            .await
            .context("Password authentication")?,
        SshAuth::Agent => {
            let mut agent = russh_keys::agent::client::AgentClient::connect_env()
                .await
                .context("Connect to SSH agent (SSH_AUTH_SOCK)")?;
            let identities = agent
                .request_identities()
                .await
                .context("List agent identities")?;

            let mut ok = false;
            for key in identities {
                let (returned_agent, result) =
                    session.authenticate_future(&profile.user, key, agent).await;
                agent = returned_agent;
                match result {
                    Ok(true) => {
                        ok = true;
                        break;
                    }
                    _ => continue,
                }
            }
            ok
        }
    };

    if !authenticated {
        bail!(
            "SSH authentication failed for {}@{}:{}",
            profile.user,
            profile.host,
            profile.port
        );
    }

    Ok(session)
}
