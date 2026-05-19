//! Docker Engine API tools.
//!
//! Provides tools for managing containers and images via the Docker daemon,
//! using the [`bollard`] crate for communication over the local socket or TCP.

pub mod tools;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Docker integration.
///
/// When `endpoint` is `None` the client connects to the platform default
/// (e.g. `/var/run/docker.sock` on Linux, named pipe on Windows).
pub struct DockerConfig {
    /// Docker endpoint.  `None` for the default local socket, or a URL such as
    /// `tcp://host:2376`.
    pub endpoint: Option<String>,
}
