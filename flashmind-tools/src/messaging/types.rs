//! Twilio API response types.
//!
//! These types map to the JSON structures returned by the Twilio REST API
//! for message resources.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A single Twilio message resource.
#[derive(Debug, Deserialize)]
pub struct TwilioMessage {
    /// Unique message identifier (e.g. `SM1234567890abcdef`).
    pub sid: String,
    /// Sender phone number or channel address.
    pub from: Option<String>,
    /// Recipient phone number or channel address.
    pub to: Option<String>,
    /// Message body text.
    pub body: Option<String>,
    /// Delivery status (e.g. `queued`, `sent`, `delivered`, `failed`).
    pub status: String,
    /// Timestamp when the message was sent (RFC 2822 format).
    pub date_sent: Option<String>,
    /// Direction of the message (e.g. `outbound-api`, `inbound`).
    pub direction: Option<String>,
}

// ---------------------------------------------------------------------------
// Message list
// ---------------------------------------------------------------------------

/// Paginated list of Twilio messages.
#[derive(Debug, Deserialize)]
pub struct TwilioMessageList {
    /// The messages on this page.
    pub messages: Vec<TwilioMessage>,
}
