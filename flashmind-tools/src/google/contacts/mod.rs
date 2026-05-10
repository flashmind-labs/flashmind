//! Native Google Contacts (People API) tools.
//!
//! Provides 6 tools for listing, searching, creating, updating, and deleting
//! contacts via the Google People API v1.

pub mod tools;
pub mod types;

use super::client::GoogleClient;

pub const BASE_URL: &str = "https://people.googleapis.com/v1";
pub const SCOPE: &str = "https://www.googleapis.com/auth/contacts";

pub type ContactsClient = GoogleClient;
