//! Microsoft Graph API calendar response types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Paginated list of Outlook calendar events.
pub struct EventListResponse {
    pub value: Vec<Event>,
    #[serde(rename = "@odata.nextLink")]
    pub next_link: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// An Outlook calendar event with attendees and body.
pub struct Event {
    pub id: String,
    pub subject: Option<String>,
    pub start: Option<DateTimeTimeZone>,
    pub end: Option<DateTimeTimeZone>,
    pub location: Option<Location>,
    pub organizer: Option<Recipient>,
    pub attendees: Option<Vec<Attendee>>,
    pub body: Option<ItemBody>,
    pub is_all_day: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Date/time with timezone (response format).
pub struct DateTimeTimeZone {
    pub date_time: Option<String>,
    pub time_zone: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Event location (response format).
pub struct Location {
    pub display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A calendar event organizer or recipient.
pub struct Recipient {
    pub email_address: Option<EmailAddress>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Email address with optional display name (response format).
pub struct EmailAddress {
    pub address: Option<String>,
    pub name: Option<String>,
}

/// A calendar event attendee with RSVP status.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attendee {
    pub email_address: Option<EmailAddress>,
    pub status: Option<ResponseStatus>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// RSVP response status of an attendee.
pub struct ResponseStatus {
    pub response: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Event body content with type (text or HTML).
pub struct ItemBody {
    pub content_type: Option<String>,
    pub content: Option<String>,
}

/// Response from creating/updating an event.
#[derive(Debug, Deserialize)]
pub struct EventResponse {
    pub id: String,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Request body for creating a new calendar event.
pub struct CreateEventRequest {
    pub subject: String,
    pub start: DateTimeInput,
    pub end: DateTimeInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<LocationInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<BodyInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attendees: Option<Vec<AttendeeInput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_all_day: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Request body for updating an existing calendar event (partial).
pub struct UpdateEventRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<DateTimeInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<DateTimeInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<LocationInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<BodyInput>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Date/time with timezone for outgoing event requests.
pub struct DateTimeInput {
    pub date_time: String,
    pub time_zone: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Location for outgoing event requests.
pub struct LocationInput {
    pub display_name: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Body content for outgoing event requests.
pub struct BodyInput {
    pub content_type: &'static str,
    pub content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Attendee for outgoing event requests.
pub struct AttendeeInput {
    pub email_address: EmailAddressInput,
    #[serde(rename = "type")]
    pub attendee_type: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Email address for outgoing event requests.
pub struct EmailAddressInput {
    pub address: String,
}
