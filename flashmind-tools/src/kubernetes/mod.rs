//! Kubernetes cluster management tools.
//!
//! Provides tools for inspecting and managing Kubernetes resources via the
//! [`kube`] client library. All tools are gated behind the `kubernetes`
//! Cargo feature.

pub mod tools;
pub mod types;

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Kubernetes tools.
///
/// Controls which kubeconfig, context, and default namespace the tools use.
pub struct KubernetesConfig {
    /// Path to kubeconfig file. `None` uses the default `~/.kube/config`.
    pub kubeconfig_path: Option<PathBuf>,
    /// Kubernetes context to use. `None` uses the current context.
    pub context: Option<String>,
    /// Default namespace. Falls back to `"default"`.
    pub namespace: Option<String>,
}

// ---------------------------------------------------------------------------
// Client helper
// ---------------------------------------------------------------------------

/// Build a [`kube::Client`] from the given configuration.
pub(crate) async fn make_client(config: &KubernetesConfig) -> anyhow::Result<kube::Client> {
    let kube_config = if let Some(path) = &config.kubeconfig_path {
        kube::config::Kubeconfig::read_from(path)?
    } else {
        kube::config::Kubeconfig::read()?
    };

    let options = kube::config::KubeConfigOptions {
        context: config.context.clone(),
        ..Default::default()
    };

    let client_config = kube::Config::from_custom_kubeconfig(kube_config, &options).await?;
    Ok(kube::Client::try_from(client_config)?)
}
