//! Google People API response models.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Connections list
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionsListResponse {
    #[serde(default)]
    pub connections: Vec<Person>,
    pub next_page_token: Option<String>,
    pub total_people: Option<u32>,
    pub total_items: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    #[serde(default)]
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub person: Option<Person>,
}

// ---------------------------------------------------------------------------
// Person
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Person {
    pub resource_name: Option<String>,
    pub etag: Option<String>,
    #[serde(default)]
    pub names: Vec<Name>,
    #[serde(default)]
    pub email_addresses: Vec<EmailAddress>,
    #[serde(default)]
    pub phone_numbers: Vec<PhoneNumber>,
    #[serde(default)]
    pub addresses: Vec<Address>,
    #[serde(default)]
    pub organizations: Vec<Organization>,
    #[serde(default)]
    pub biographies: Vec<Biography>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Name {
    pub display_name: Option<String>,
    pub given_name: Option<String>,
    pub family_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmailAddress {
    pub value: Option<String>,
    pub r#type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneNumber {
    pub value: Option<String>,
    pub r#type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Address {
    pub formatted_value: Option<String>,
    pub r#type: Option<String>,
    pub street_address: Option<String>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Organization {
    pub name: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Biography {
    pub value: Option<String>,
}
