//! Native Google Calendar API tools.
//!
//! Provides 6 tools for listing calendars, listing/creating/updating/deleting
//! events via the Google Calendar v3 REST API.

pub mod tools;
pub mod types;

use super::client::GoogleClient;

pub const BASE_URL: &str = "https://www.googleapis.com/calendar/v3";
pub const SCOPE: &str = "https://www.googleapis.com/auth/calendar";

pub type CalendarClient = GoogleClient;
