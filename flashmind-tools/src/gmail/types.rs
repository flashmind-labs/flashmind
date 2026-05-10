//! Gmail API response models.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListResponse {
    #[serde(default)]
    pub threads: Vec<ThreadSummary>,
    pub next_page_token: Option<String>,
    pub result_size_estimate: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSummary {
    pub id: String,
    pub snippet: Option<String>,
    pub history_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub id: String,
    #[serde(default)]
    pub messages: Vec<Message>,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    pub thread_id: Option<String>,
    #[serde(default)]
    pub label_ids: Vec<String>,
    pub snippet: Option<String>,
    pub payload: Option<MessagePayload>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessagePayload {
    pub mime_type: Option<String>,
    #[serde(default)]
    pub headers: Vec<Header>,
    pub body: Option<MessageBody>,
    #[serde(default)]
    pub parts: Vec<MessagePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessagePart {
    pub mime_type: Option<String>,
    pub body: Option<MessageBody>,
    #[serde(default)]
    pub parts: Vec<MessagePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageBody {
    pub data: Option<String>,
    pub size: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Header {
    pub name: String,
    pub value: String,
}

// ---------------------------------------------------------------------------
// Drafts
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftListResponse {
    #[serde(default)]
    pub drafts: Vec<DraftSummary>,
    pub next_page_token: Option<String>,
    pub result_size_estimate: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftSummary {
    pub id: String,
    pub message: Option<DraftMessageSummary>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftMessageSummary {
    pub id: String,
    pub thread_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelListResponse {
    #[serde(default)]
    pub labels: Vec<Label>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Label {
    pub id: String,
    pub name: String,
    pub r#type: Option<String>,
    pub message_list_visibility: Option<String>,
    pub label_list_visibility: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Decode a base64url-encoded Gmail message body to UTF-8.
pub fn decode_body(data: &str) -> anyhow::Result<String> {
    let bytes = URL_SAFE_NO_PAD.decode(data)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Extract the plain-text body from a message payload, recursively walking
/// multipart MIME parts.
pub fn extract_text(payload: &MessagePayload) -> Option<String> {
    if payload.mime_type.as_deref() == Some("text/plain")
        && let Some(data) = payload.body.as_ref().and_then(|b| b.data.as_deref())
    {
        return decode_body(data).ok();
    }
    payload.parts.iter().find_map(extract_text_from_part)
}

fn extract_text_from_part(part: &MessagePart) -> Option<String> {
    if part.mime_type.as_deref() == Some("text/plain")
        && let Some(data) = part.body.as_ref().and_then(|b| b.data.as_deref())
    {
        return decode_body(data).ok();
    }
    part.parts.iter().find_map(extract_text_from_part)
}
