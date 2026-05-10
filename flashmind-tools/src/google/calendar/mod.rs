//! Native Google Calendar API tools.
//!
//! Provides 6 tools for listing calendars, listing/creating/updating/deleting
//! events via the Google Calendar v3 REST API.

pub mod tools;
pub mod types;

use anyhow::Result;

use super::client::{GoogleClient, GoogleConfig};

pub const BASE_URL: &str = "https://www.googleapis.com/calendar/v3";
pub const SCOPE: &str = "https://www.googleapis.com/auth/calendar";

/// Type alias for a Google client configured for the Calendar API.
pub type CalendarClient = GoogleClient;

/// Create a new authenticated Google Calendar API client.
pub fn new_client(config: &GoogleConfig) -> Result<CalendarClient> {
    GoogleClient::new(config.clone(), BASE_URL, SCOPE)
}
