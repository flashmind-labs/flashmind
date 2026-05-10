//! Google Contacts `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::ContactsClient;
use super::types::{ConnectionsListResponse, Person, SearchResponse};

const PERSON_FIELDS: &str = "names,emailAddresses,phoneNumbers,addresses,organizations,biographies";

fn format_person(p: &Person) -> String {
    let mut out = String::new();
    let name = p
        .names
        .first()
        .and_then(|n| n.display_name.as_deref())
        .unwrap_or("(unnamed)");
    let rn = p.resource_name.as_deref().unwrap_or("?");
    out.push_str(&format!("{name} [{rn}]\n"));

    for email in &p.email_addresses {
        if let Some(v) = &email.value {
            let t = email.r#type.as_deref().unwrap_or("other");
            out.push_str(&format!("  Email ({t}): {v}\n"));
        }
    }
    for phone in &p.phone_numbers {
        if let Some(v) = &phone.value {
            let t = phone.r#type.as_deref().unwrap_or("other");
            out.push_str(&format!("  Phone ({t}): {v}\n"));
        }
    }
    for org in &p.organizations {
        let name = org.name.as_deref().unwrap_or("");
        let title = org.title.as_deref().unwrap_or("");
        if !name.is_empty() || !title.is_empty() {
            out.push_str(&format!("  Org: {name} — {title}\n"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// gcontacts_list
// ---------------------------------------------------------------------------

pub struct GcontactsListTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct ListContactsArgs {
    max_results: Option<u32>,
    page_token: Option<String>,
}

#[async_trait]
impl Tool for GcontactsListTool {
    fn name(&self) -> &str {
        "gcontacts_list"
    }

    fn description(&self) -> &str {
        "List Google Contacts for the authenticated user."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Max contacts to return (default 25, max 100)"
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token from a previous response"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListContactsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let max = args.max_results.unwrap_or(25).min(100);

        let mut path = format!(
            "people/me/connections?pageSize={max}&personFields={PERSON_FIELDS}"
        );
        if let Some(token) = &args.page_token {
            path.push_str(&format!("&pageToken={token}"));
        }

        let resp: ConnectionsListResponse =
            serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        for p in &resp.connections {
            out.push_str(&format_person(p));
            out.push('\n');
        }
        if let Some(next) = &resp.next_page_token {
            out.push_str(&format!("Next page token: {next}"));
        }
        if resp.connections.is_empty() {
            out.push_str("No contacts found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Google Contacts".to_string()
    }
}

// ---------------------------------------------------------------------------
// gcontacts_search
// ---------------------------------------------------------------------------

pub struct GcontactsSearchTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct SearchContactsArgs {
    query: String,
    max_results: Option<u32>,
}

#[async_trait]
impl Tool for GcontactsSearchTool {
    fn name(&self) -> &str {
        "gcontacts_search"
    }

    fn description(&self) -> &str {
        "Search Google Contacts by name, email, phone number, or other fields."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 10, max 30)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchContactsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let max = args.max_results.unwrap_or(10).min(30);

        let path = format!(
            "people:searchContacts?query={}&pageSize={max}&readMask={PERSON_FIELDS}",
            urlencoding::encode(&args.query)
        );

        let resp: SearchResponse = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        for result in &resp.results {
            if let Some(person) = &result.person {
                out.push_str(&format_person(person));
                out.push('\n');
            }
        }
        if resp.results.is_empty() {
            out.push_str("No contacts found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let q = args["query"].as_str().unwrap_or("...");
        format!("Searching contacts for '{q}'")
    }
}

// ---------------------------------------------------------------------------
// gcontacts_get
// ---------------------------------------------------------------------------

pub struct GcontactsGetTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct GetContactArgs {
    resource_name: String,
}

#[async_trait]
impl Tool for GcontactsGetTool {
    fn name(&self) -> &str {
        "gcontacts_get"
    }

    fn description(&self) -> &str {
        "Get a Google Contact by resource name (e.g. 'people/c12345')."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Contact resource name (e.g. 'people/c12345')"
                }
            },
            "required": ["resource_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("{}?personFields={PERSON_FIELDS}", args.resource_name);

        let person: Person = serde_json::from_value(self.client.get(&path).await?)?;
        let out = format_person(&person);

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let rn = args["resource_name"].as_str().unwrap_or("...");
        format!("Getting contact {rn}")
    }
}

// ---------------------------------------------------------------------------
// gcontacts_create
// ---------------------------------------------------------------------------

pub struct GcontactsCreateTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct CreateContactArgs {
    given_name: String,
    family_name: Option<String>,
    email: Option<String>,
    phone: Option<String>,
    organization: Option<String>,
    title: Option<String>,
}

#[async_trait]
impl Tool for GcontactsCreateTool {
    fn name(&self) -> &str {
        "gcontacts_create"
    }

    fn description(&self) -> &str {
        "Create a new Google Contact."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "given_name": {
                    "type": "string",
                    "description": "First name"
                },
                "family_name": {
                    "type": "string",
                    "description": "Last name"
                },
                "email": {
                    "type": "string",
                    "description": "Email address"
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number"
                },
                "organization": {
                    "type": "string",
                    "description": "Company/organization name"
                },
                "title": {
                    "type": "string",
                    "description": "Job title"
                }
            },
            "required": ["given_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        let mut body = json!({
            "names": [{
                "givenName": args.given_name,
                "familyName": args.family_name.as_deref().unwrap_or("")
            }]
        });

        if let Some(email) = &args.email {
            body["emailAddresses"] = json!([{ "value": email }]);
        }
        if let Some(phone) = &args.phone {
            body["phoneNumbers"] = json!([{ "value": phone }]);
        }
        if args.organization.is_some() || args.title.is_some() {
            body["organizations"] = json!([{
                "name": args.organization.as_deref().unwrap_or(""),
                "title": args.title.as_deref().unwrap_or("")
            }]);
        }

        let resp = self.client.post("people:createContact", body).await?;
        let rn = resp["resourceName"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Contact created: {rn}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args["given_name"].as_str().unwrap_or("...");
        format!("Creating contact '{name}'")
    }
}

// ---------------------------------------------------------------------------
// gcontacts_update
// ---------------------------------------------------------------------------

pub struct GcontactsUpdateTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct UpdateContactArgs {
    resource_name: String,
    given_name: Option<String>,
    family_name: Option<String>,
    email: Option<String>,
    phone: Option<String>,
    organization: Option<String>,
    title: Option<String>,
}

#[async_trait]
impl Tool for GcontactsUpdateTool {
    fn name(&self) -> &str {
        "gcontacts_update"
    }

    fn description(&self) -> &str {
        "Update an existing Google Contact. Only provided fields are changed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Contact resource name (e.g. 'people/c12345')"
                },
                "given_name": {
                    "type": "string",
                    "description": "New first name"
                },
                "family_name": {
                    "type": "string",
                    "description": "New last name"
                },
                "email": {
                    "type": "string",
                    "description": "New email address"
                },
                "phone": {
                    "type": "string",
                    "description": "New phone number"
                },
                "organization": {
                    "type": "string",
                    "description": "New company/organization"
                },
                "title": {
                    "type": "string",
                    "description": "New job title"
                }
            },
            "required": ["resource_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: UpdateContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        // Fetch current state to get etag
        let current_path = format!("{}?personFields={PERSON_FIELDS}", args.resource_name);
        let current: Person = serde_json::from_value(self.client.get(&current_path).await?)?;

        let mut body = json!({
            "etag": current.etag,
        });
        let mut update_fields = Vec::new();

        if args.given_name.is_some() || args.family_name.is_some() {
            body["names"] = json!([{
                "givenName": args.given_name.as_deref().unwrap_or(
                    current.names.first().and_then(|n| n.given_name.as_deref()).unwrap_or("")
                ),
                "familyName": args.family_name.as_deref().unwrap_or(
                    current.names.first().and_then(|n| n.family_name.as_deref()).unwrap_or("")
                )
            }]);
            update_fields.push("names");
        }
        if let Some(email) = &args.email {
            body["emailAddresses"] = json!([{ "value": email }]);
            update_fields.push("emailAddresses");
        }
        if let Some(phone) = &args.phone {
            body["phoneNumbers"] = json!([{ "value": phone }]);
            update_fields.push("phoneNumbers");
        }
        if args.organization.is_some() || args.title.is_some() {
            body["organizations"] = json!([{
                "name": args.organization.as_deref().unwrap_or(""),
                "title": args.title.as_deref().unwrap_or("")
            }]);
            update_fields.push("organizations");
        }

        let path = format!(
            "{}:updateContact?updatePersonFields={}",
            args.resource_name,
            update_fields.join(",")
        );
        self.client.patch(&path, body).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Contact {} updated", args.resource_name),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let rn = args["resource_name"].as_str().unwrap_or("...");
        format!("Updating contact {rn}")
    }
}

// ---------------------------------------------------------------------------
// gcontacts_delete
// ---------------------------------------------------------------------------

pub struct GcontactsDeleteTool {
    pub client: Arc<ContactsClient>,
}

#[derive(Deserialize)]
struct DeleteContactArgs {
    resource_name: String,
}

#[async_trait]
impl Tool for GcontactsDeleteTool {
    fn name(&self) -> &str {
        "gcontacts_delete"
    }

    fn description(&self) -> &str {
        "Delete a Google Contact by resource name."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Contact resource name (e.g. 'people/c12345')"
                }
            },
            "required": ["resource_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: DeleteContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("{}:deleteContact", args.resource_name);
        self.client.delete(&path).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Contact {} deleted", args.resource_name),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let rn = args["resource_name"].as_str().unwrap_or("...");
        format!("Deleting contact {rn}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::google::client::GoogleClient;

    #[test]
    fn tool_names_are_correct() {
        let client = Arc::new(GoogleClient::new_for_test(super::super::BASE_URL, super::super::SCOPE));

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(GcontactsListTool { client: client.clone() }),
            Box::new(GcontactsSearchTool { client: client.clone() }),
            Box::new(GcontactsGetTool { client: client.clone() }),
            Box::new(GcontactsCreateTool { client: client.clone() }),
            Box::new(GcontactsUpdateTool { client: client.clone() }),
            Box::new(GcontactsDeleteTool { client: client.clone() }),
        ];

        let expected = [
            "gcontacts_list",
            "gcontacts_search",
            "gcontacts_get",
            "gcontacts_create",
            "gcontacts_update",
            "gcontacts_delete",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
