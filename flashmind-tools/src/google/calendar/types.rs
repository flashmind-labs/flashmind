//! Google Calendar API response models.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Calendar list
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalendarListResponse {
    #[serde(default)]
    pub items: Vec<CalendarEntry>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalendarEntry {
    pub id: String,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub primary: Option<bool>,
    pub time_zone: Option<String>,
    pub access_role: Option<String>,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventListResponse {
    #[serde(default)]
    pub items: Vec<Event>,
    pub next_page_token: Option<String>,
    pub summary: Option<String>,
    pub time_zone: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    pub id: Option<String>,
    pub status: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start: Option<EventDateTime>,
    pub end: Option<EventDateTime>,
    pub created: Option<String>,
    pub updated: Option<String>,
    pub html_link: Option<String>,
    #[serde(default)]
    pub attendees: Vec<Attendee>,
    pub organizer: Option<Organizer>,
    pub recurrence: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventDateTime {
    pub date: Option<String>,
    pub date_time: Option<String>,
    pub time_zone: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Attendee {
    pub email: Option<String>,
    pub display_name: Option<String>,
    pub response_status: Option<String>,
    pub organizer: Option<bool>,
    #[serde(rename = "self")]
    pub is_self: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Organizer {
    pub email: Option<String>,
    pub display_name: Option<String>,
    #[serde(rename = "self")]
    pub is_self: Option<bool>,
}
