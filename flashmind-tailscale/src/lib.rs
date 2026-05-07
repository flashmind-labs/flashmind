//! Tailscale integration for Flashmind.
//!
//! Provides a dual-mode client for the Tailscale daemon — preferring the local
//! API (Unix domain socket) for speed and richness, with automatic fallback to
//! the `tailscale` CLI. Also includes helpers for managing Tailscale Funnel
//! routes to expose local services over HTTPS.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use flashmind_tailscale::TailscaleClient;
//!
//! let client = TailscaleClient::connect();
//! let status = client.status().await?;
//! println!("Device: {}", status.self_device.device_name);
//! println!("Peers: {}", status.peers.len());
//! ```

mod client;
mod error;
mod funnel;
pub mod local_api;
mod status;
pub mod types;

pub use client::TailscaleClient;
pub use error::TailscaleError;
pub use funnel::FunnelManager;
pub use local_api::LocalApi;
pub use status::{DeviceStatus, PeerStatus, TailscaleStatus};
pub use types::{PeerNode, PingResult, UserProfile, WhoisResult};
