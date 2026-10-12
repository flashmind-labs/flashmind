//! SSH remote execution tools with persistent session support.
//!
//! Provides tools for executing commands, uploading, and downloading files on
//! remote hosts via SSH using the [`russh`] crate.  Connections are configured
//! through named [`SshProfile`]s stored in an [`SshConfig`].
//!
//! Sessions can be opened with [`SshOpenTool`](tools::SshOpenTool) and reused
//! across multiple commands via an opaque session ID, avoiding repeated
//! handshakes and preserving shell state (cwd, env vars).  The
//! [`SshSessionManager`] holds live sessions and can be cloned by the consumer
//! for custom cleanup.

pub mod tools;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, bail};
use dashmap::DashMap;
use russh::client::AuthResult;
use russh::keys::agent::AgentIdentity;
use russh::keys::agent::client::AgentClient;
use russh::keys::{PrivateKeyWithHashAlg, PublicKeyOrCertificate, load_secret_key};

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
// Session manager
// ---------------------------------------------------------------------------

/// Shared store for persistent SSH sessions.
///
/// Clone the `SshSessionManager` to inspect or reap sessions from outside the
/// tool layer.  When the last `SshSessionManager` is dropped, all connections
/// are closed.
#[derive(Clone, Default)]
pub struct SshSessionManager {
    sessions: Arc<DashMap<String, Arc<russh::client::Handle<SshHandler>>>>,
}

impl SshSessionManager {
    /// Create an empty session manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a handle under `id`.
    pub fn insert(&self, id: String, handle: russh::client::Handle<SshHandler>) {
        self.sessions.insert(id, Arc::new(handle));
    }

    /// Retrieve a session handle.
    pub fn get(&self, id: &str) -> Option<Arc<russh::client::Handle<SshHandler>>> {
        self.sessions.get(id).map(|r| Arc::clone(&*r))
    }

    /// Remove and close a session. Returns `true` if it existed.
    pub fn remove(&self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    /// Number of open sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether any sessions are open.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Minimal [`russh::client::Handler`] that accepts all host keys.
pub struct SshHandler;

impl russh::client::Handler for SshHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &PublicKeyOrCertificate,
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
            let key = load_secret_key(path, passphrase.as_deref())
                .with_context(|| format!("Load SSH key from {}", path.display()))?;
            let hash_alg = session.best_supported_rsa_hash().await?.flatten();
            session
                .authenticate_publickey(
                    &profile.user,
                    PrivateKeyWithHashAlg::new(Arc::new(key), hash_alg),
                )
                .await
                .context("Public-key authentication")?
                .success()
        }
        SshAuth::Password { password } => session
            .authenticate_password(&profile.user, password)
            .await
            .context("Password authentication")?
            .success(),
        SshAuth::Agent => {
            let mut agent = AgentClient::connect_env()
                .await
                .context("Connect to SSH agent (SSH_AUTH_SOCK)")?;
            let identities = agent
                .request_identities()
                .await
                .context("List agent identities")?;

            let hash_alg = session.best_supported_rsa_hash().await?.flatten();
            let mut ok = false;
            for identity in identities {
                let result = match identity {
                    AgentIdentity::PublicKey { key, .. } => {
                        session
                            .authenticate_publickey_with(&profile.user, key, hash_alg, &mut agent)
                            .await
                    }
                    AgentIdentity::Certificate { certificate, .. } => {
                        session
                            .authenticate_certificate_with(
                                &profile.user,
                                certificate,
                                hash_alg,
                                &mut agent,
                            )
                            .await
                    }
                };
                match result {
                    Ok(AuthResult::Success) => {
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
