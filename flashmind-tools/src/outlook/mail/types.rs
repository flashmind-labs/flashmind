//! Microsoft Graph API mail response types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Paginated list of Outlook mail messages.
pub struct MessageListResponse {
    pub value: Vec<Message>,
    #[serde(rename = "@odata.nextLink")]
    pub next_link: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// An Outlook email message with headers and body.
pub struct Message {
    pub id: String,
    pub subject: Option<String>,
    pub from: Option<Recipient>,
    pub received_date_time: Option<String>,
    pub is_read: Option<bool>,
    pub body_preview: Option<String>,
    pub body: Option<ItemBody>,
    pub to_recipients: Option<Vec<Recipient>>,
    pub cc_recipients: Option<Vec<Recipient>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A mail recipient (from/to/cc) in response format.
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Message body content with type (text or HTML).
pub struct ItemBody {
    pub content_type: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// List of mail folders in the user's mailbox.
pub struct FolderListResponse {
    pub value: Vec<MailFolder>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// An Outlook mail folder (inbox, sent, drafts, etc.).
pub struct MailFolder {
    pub id: String,
    pub display_name: Option<String>,
    pub unread_item_count: Option<u64>,
    pub total_item_count: Option<u64>,
}

/// Response from POST /sendMail (empty on success).
#[derive(Debug, Deserialize)]
pub struct SendMailResponse {}

/// Response from POST /messages (create draft).
#[derive(Debug, Deserialize)]
pub struct CreateDraftResponse {
    pub id: String,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Request body for POST `/sendMail`.
pub struct SendMailRequest {
    pub message: NewMessage,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// New message payload used in send and draft requests.
pub struct NewMessage {
    pub subject: String,
    pub body: MessageBody,
    pub to_recipients: Vec<RecipientInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cc_recipients: Option<Vec<RecipientInput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bcc_recipients: Option<Vec<RecipientInput>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Message body for outgoing mail (content type + content).
pub struct MessageBody {
    pub content_type: &'static str,
    pub content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// A recipient in request format (for sending/drafting).
pub struct RecipientInput {
    pub email_address: EmailAddressInput,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Email address for outgoing mail requests.
pub struct EmailAddressInput {
    pub address: String,
}
