//! Native Google Contacts (People API) tools.
//!
//! Provides 6 tools for listing, searching, creating, updating, and deleting
//! contacts via the Google People API v1.

pub mod tools;
pub mod types;

use anyhow::Result;

use super::client::{GoogleClient, GoogleConfig};

pub const BASE_URL: &str = "https://people.googleapis.com/v1";
pub const SCOPE: &str = "https://www.googleapis.com/auth/contacts";

/// Type alias for a Google client configured for the People API.
pub type ContactsClient = GoogleClient;

/// Create a new authenticated Google Contacts (People API) client.
pub fn new_client(config: &GoogleConfig) -> Result<ContactsClient> {
    GoogleClient::new(config.clone(), BASE_URL, SCOPE)
}
