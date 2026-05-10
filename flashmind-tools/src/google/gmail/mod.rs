//! Native Gmail API tools.
//!
//! Provides 11 tools for reading, searching, drafting, and labelling Gmail
//! messages — equivalent to the MCP Gmail server but without process overhead.

pub mod tools;
pub mod types;

use anyhow::Result;

use super::client::{GoogleClient, GoogleConfig};

pub const BASE_URL: &str = "https://gmail.googleapis.com/gmail/v1/users/me";
pub const SCOPE: &str = "https://mail.google.com/";

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Type alias for a Google client configured for the Gmail API.
pub type GmailClient = GoogleClient;

/// Create a new authenticated Gmail API client.
pub fn new_client(config: &GoogleConfig) -> Result<GmailClient> {
    GoogleClient::new(config.clone(), BASE_URL, SCOPE)
}
