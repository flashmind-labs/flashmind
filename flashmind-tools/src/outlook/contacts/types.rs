//! Microsoft Graph API contacts response types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Paginated list of Outlook contacts.
pub struct ContactListResponse {
    pub value: Vec<Contact>,
    #[serde(rename = "@odata.nextLink")]
    pub next_link: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// An Outlook contact with name, email, phone, and company info.
pub struct Contact {
    pub id: String,
    pub display_name: Option<String>,
    pub given_name: Option<String>,
    pub surname: Option<String>,
    pub email_addresses: Option<Vec<EmailAddress>>,
    pub mobile_phone: Option<String>,
    pub business_phones: Option<Vec<String>>,
    pub company_name: Option<String>,
    pub job_title: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Email address with optional display name (response format).
pub struct EmailAddress {
    pub address: Option<String>,
    pub name: Option<String>,
}

/// Response from POST /contacts.
#[derive(Debug, Deserialize)]
pub struct CreateContactResponse {
    pub id: String,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Request body for creating a new contact.
pub struct CreateContactRequest {
    pub given_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surname: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email_addresses: Option<Vec<EmailAddressInput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mobile_phone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub company_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_title: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
/// Email address for outgoing contact requests.
pub struct EmailAddressInput {
    pub address: String,
}
