//! Messaging tools (Twilio SMS/WhatsApp, SMTP email).
//!
//! Gated behind the `messaging` feature flag. Twilio tools use the REST API
//! via `reqwest`; SMTP tools use the `lettre` crate for email delivery.

pub mod smtp;
pub mod twilio;
pub mod types;

// ---------------------------------------------------------------------------
// Twilio config
// ---------------------------------------------------------------------------

/// Configuration for the Twilio REST API.
pub struct TwilioConfig {
    /// Twilio Account SID.
    pub account_sid: String,
    /// Twilio Auth Token.
    pub auth_token: String,
    /// Default sender phone number in E.164 format (e.g. `+15551234567`).
    pub from_number: String,
}

// ---------------------------------------------------------------------------
// SMTP config
// ---------------------------------------------------------------------------

/// Configuration for SMTP email delivery.
#[derive(Clone)]
pub struct SmtpConfig {
    /// SMTP server hostname (e.g. `smtp.gmail.com`).
    pub host: String,
    /// SMTP server port (typically 587 for STARTTLS, 465 for implicit TLS).
    pub port: u16,
    /// SMTP username.
    pub username: String,
    /// SMTP password or app password.
    pub password: String,
    /// Default sender email address (e.g. `user@example.com`).
    pub from_address: String,
}

// ---------------------------------------------------------------------------
// Combined config
// ---------------------------------------------------------------------------

/// Combined messaging configuration.
///
/// Either or both of `twilio` and `smtp` may be provided depending on which
/// channels the agent should have access to.
pub struct MessagingConfig {
    /// Twilio configuration for SMS and WhatsApp.
    pub twilio: Option<TwilioConfig>,
    /// SMTP configuration for email.
    pub smtp: Option<SmtpConfig>,
}
