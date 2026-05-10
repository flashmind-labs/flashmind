//! Outlook Contacts `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::types::*;
use crate::outlook::OutlookClient;

// ---------------------------------------------------------------------------
// outlook_list_contacts
// ---------------------------------------------------------------------------

/// List Outlook contacts with optional search.
pub struct OutlookListContactsTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct ListContactsArgs {
    top: Option<u32>,
    search: Option<String>,
}

#[async_trait]
impl Tool for OutlookListContactsTool {
    fn name(&self) -> &str {
        "outlook_list_contacts"
    }

    fn description(&self) -> &str {
        "List Outlook contacts with optional search."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "top": {
                    "type": "integer",
                    "description": "Max contacts to return (default 25, max 50)"
                },
                "search": {
                    "type": "string",
                    "description": "Search query (searches name and email)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListContactsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let top = args.top.unwrap_or(25).min(50);

        let mut path = format!(
            "contacts?$top={top}&$select=id,displayName,emailAddresses,mobilePhone,businessPhones,companyName,jobTitle"
        );
        if let Some(search) = &args.search {
            path.push_str(&format!("&$search=\"{}\"", urlencoding::encode(search)));
        }

        let resp: ContactListResponse = self.client.get(&path).await?;

        let mut out = String::new();
        if resp.value.is_empty() {
            out.push_str("No contacts found.");
        } else {
            for contact in &resp.value {
                let name = contact.display_name.as_deref().unwrap_or("(unnamed)");
                let email = contact
                    .email_addresses
                    .as_ref()
                    .and_then(|a| a.first())
                    .and_then(|e| e.address.as_deref())
                    .unwrap_or("");
                let company = contact.company_name.as_deref().unwrap_or("");

                out.push_str(&format!("- {name}"));
                if !email.is_empty() {
                    out.push_str(&format!(" <{email}>"));
                }
                if !company.is_empty() {
                    out.push_str(&format!(" @ {company}"));
                }
                out.push_str(&format!(" [{}]\n", contact.id));
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Outlook contacts".to_string()
    }
}

// ---------------------------------------------------------------------------
// outlook_get_contact
// ---------------------------------------------------------------------------

/// Get full details of a specific Outlook contact.
pub struct OutlookGetContactTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct GetContactArgs {
    contact_id: String,
}

#[async_trait]
impl Tool for OutlookGetContactTool {
    fn name(&self) -> &str {
        "outlook_get_contact"
    }

    fn description(&self) -> &str {
        "Get details of a specific Outlook contact."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "contact_id": {
                    "type": "string",
                    "description": "The contact ID"
                }
            },
            "required": ["contact_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("contacts/{}", args.contact_id);

        let contact: Contact = self.client.get(&path).await?;

        let name = contact.display_name.as_deref().unwrap_or("(unnamed)");
        let mut out = format!("{name}\n");

        if let Some(emails) = &contact.email_addresses {
            for e in emails {
                let addr = e.address.as_deref().unwrap_or("?");
                let name = e.name.as_deref().unwrap_or("");
                out.push_str(&format!("  Email: {addr} ({name})\n"));
            }
        }
        if let Some(phone) = &contact.mobile_phone {
            out.push_str(&format!("  Mobile: {phone}\n"));
        }
        if let Some(phones) = &contact.business_phones {
            for phone in phones {
                out.push_str(&format!("  Business: {phone}\n"));
            }
        }
        if let Some(company) = &contact.company_name {
            out.push_str(&format!("  Company: {company}\n"));
        }
        if let Some(title) = &contact.job_title {
            out.push_str(&format!("  Title: {title}\n"));
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["contact_id"].as_str().unwrap_or("...");
        format!("Getting Outlook contact {id}")
    }
}

// ---------------------------------------------------------------------------
// outlook_create_contact
// ---------------------------------------------------------------------------

/// Create a new Outlook contact.
pub struct OutlookCreateContactTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct CreateContactArgs {
    given_name: String,
    surname: Option<String>,
    email: Option<String>,
    mobile_phone: Option<String>,
    company_name: Option<String>,
    job_title: Option<String>,
}

#[async_trait]
impl Tool for OutlookCreateContactTool {
    fn name(&self) -> &str {
        "outlook_create_contact"
    }

    fn description(&self) -> &str {
        "Create a new Outlook contact."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "given_name": {
                    "type": "string",
                    "description": "First name"
                },
                "surname": {
                    "type": "string",
                    "description": "Last name"
                },
                "email": {
                    "type": "string",
                    "description": "Email address"
                },
                "mobile_phone": {
                    "type": "string",
                    "description": "Mobile phone number"
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name"
                },
                "job_title": {
                    "type": "string",
                    "description": "Job title"
                }
            },
            "required": ["given_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateContactArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        let body = CreateContactRequest {
            given_name: args.given_name,
            surname: args.surname,
            email_addresses: args
                .email
                .map(|addr| vec![EmailAddressInput { address: addr }]),
            mobile_phone: args.mobile_phone,
            company_name: args.company_name,
            job_title: args.job_title,
        };

        let resp: CreateContactResponse = self.client.post("contacts", &body).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Contact created: id={}", resp.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args["given_name"].as_str().unwrap_or("...");
        format!("Creating Outlook contact '{name}'")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_correct() {
        let client = Arc::new(OutlookClient::new_for_test());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(OutlookListContactsTool {
                client: client.clone(),
            }),
            Box::new(OutlookGetContactTool {
                client: client.clone(),
            }),
            Box::new(OutlookCreateContactTool {
                client: client.clone(),
            }),
        ];

        let expected = [
            "outlook_list_contacts",
            "outlook_get_contact",
            "outlook_create_contact",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
