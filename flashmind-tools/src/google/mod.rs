//! Native Google API tool implementations.
//!
//! Provides authenticated access to Gmail, Calendar, and Contacts APIs
//! without requiring external MCP servers.

pub mod auth;
pub mod client;

#[cfg(feature = "gmail")]
pub mod gmail;
#[cfg(feature = "google-calendar")]
pub mod calendar;
#[cfg(feature = "google-contacts")]
pub mod contacts;

pub use client::GoogleConfig;
