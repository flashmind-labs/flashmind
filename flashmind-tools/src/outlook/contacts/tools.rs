//! Outlook Contacts `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::outlook::OutlookClient;

// ---------------------------------------------------------------------------
// outlook_list_contacts
// ---------------------------------------------------------------------------

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

        let mut path = format!("contacts?$top={top}&$select=id,displayName,emailAddresses,mobilePhone,businessPhones,companyName,jobTitle");
        if let Some(search) = &args.search {
            path.push_str(&format!("&$search=\"{}\"", urlencoding::encode(search)));
        }

        let resp = self.client.get(&path).await?;
        let contacts = resp["value"].as_array();

        let mut out = String::new();
        if let Some(items) = contacts {
            for c in items {
                let name = c["displayName"].as_str().unwrap_or("(unnamed)");
                let id = c["id"].as_str().unwrap_or("?");
                let email = c["emailAddresses"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|e| e["address"].as_str())
                    .unwrap_or("");
                let company = c["companyName"].as_str().unwrap_or("");
                out.push_str(&format!("- {name}"));
                if !email.is_empty() {
                    out.push_str(&format!(" <{email}>"));
                }
                if !company.is_empty() {
                    out.push_str(&format!(" @ {company}"));
                }
                out.push_str(&format!(" [{id}]\n"));
            }
            if items.is_empty() {
                out.push_str("No contacts found.");
            }
        } else {
            out.push_str("No contacts found.");
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

        let c = self.client.get(&path).await?;

        let name = c["displayName"].as_str().unwrap_or("(unnamed)");
        let mut out = format!("{name}\n");

        if let Some(emails) = c["emailAddresses"].as_array() {
            for e in emails {
                let addr = e["address"].as_str().unwrap_or("?");
                let name = e["name"].as_str().unwrap_or("");
                out.push_str(&format!("  Email: {addr} ({name})\n"));
            }
        }
        if let Some(phone) = c["mobilePhone"].as_str() {
            out.push_str(&format!("  Mobile: {phone}\n"));
        }
        if let Some(phones) = c["businessPhones"].as_array() {
            for p in phones {
                if let Some(phone) = p.as_str() {
                    out.push_str(&format!("  Business: {phone}\n"));
                }
            }
        }
        if let Some(company) = c["companyName"].as_str() {
            out.push_str(&format!("  Company: {company}\n"));
        }
        if let Some(title) = c["jobTitle"].as_str() {
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

        let mut body = json!({
            "givenName": args.given_name
        });
        if let Some(surname) = &args.surname {
            body["surname"] = json!(surname);
        }
        if let Some(email) = &args.email {
            body["emailAddresses"] = json!([{"address": email}]);
        }
        if let Some(phone) = &args.mobile_phone {
            body["mobilePhone"] = json!(phone);
        }
        if let Some(company) = &args.company_name {
            body["companyName"] = json!(company);
        }
        if let Some(title) = &args.job_title {
            body["jobTitle"] = json!(title);
        }

        let resp = self.client.post("contacts", body).await?;
        let id = resp["id"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Contact created: id={id}"),
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
            Box::new(OutlookListContactsTool { client: client.clone() }),
            Box::new(OutlookGetContactTool { client: client.clone() }),
            Box::new(OutlookCreateContactTool { client: client.clone() }),
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
