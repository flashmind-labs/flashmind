//! Outlook Mail `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::types::{
    CreateDraftResponse, EmailAddressInput, FolderListResponse, MessageBody, MessageListResponse,
    NewMessage, RecipientInput, SendMailRequest, SendMailResponse,
};
use crate::outlook::OutlookClient;

// ---------------------------------------------------------------------------
// outlook_list_messages
// ---------------------------------------------------------------------------

/// List Outlook email messages with filtering and search.
pub struct OutlookListMessagesTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct ListMessagesArgs {
    folder: Option<String>,
    filter: Option<String>,
    top: Option<u32>,
    search: Option<String>,
}

#[async_trait]
impl Tool for OutlookListMessagesTool {
    fn name(&self) -> &str {
        "outlook_list_messages"
    }

    fn description(&self) -> &str {
        "List email messages from Outlook. Supports filtering and search."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Folder ID or well-known name (inbox, sentitems, drafts). Default: inbox"
                },
                "filter": {
                    "type": "string",
                    "description": "OData $filter expression (e.g. \"isRead eq false\")"
                },
                "top": {
                    "type": "integer",
                    "description": "Number of messages to return (default 10, max 50)"
                },
                "search": {
                    "type": "string",
                    "description": "Search query (searches subject, body, sender)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListMessagesArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let top = args.top.unwrap_or(10).min(50);

        let base = if let Some(folder) = &args.folder {
            format!("mailFolders/{folder}/messages")
        } else {
            "messages".to_string()
        };

        let mut path = format!(
            "{base}?$top={top}&$select=id,subject,from,receivedDateTime,isRead,bodyPreview"
        );
        if let Some(filter) = &args.filter {
            path.push_str(&format!("&$filter={}", urlencoding::encode(filter)));
        }
        if let Some(search) = &args.search {
            path.push_str(&format!("&$search=\"{}\"", urlencoding::encode(search)));
        }

        let resp: MessageListResponse = self.client.get(&path).await?;

        let mut out = String::new();
        if resp.value.is_empty() {
            out.push_str("No messages found.");
        } else {
            for msg in &resp.value {
                let id = &msg.id;
                let subject = msg.subject.as_deref().unwrap_or("(no subject)");
                let from = msg
                    .from
                    .as_ref()
                    .and_then(|r| r.email_address.as_ref())
                    .and_then(|e| e.address.as_deref())
                    .unwrap_or("?");
                let date = msg.received_date_time.as_deref().unwrap_or("?");
                let read = if msg.is_read.unwrap_or(false) {
                    ""
                } else {
                    " [UNREAD]"
                };
                out.push_str(&format!(
                    "- {subject}{read}\n  From: {from} | {date}\n  [{id}]\n\n"
                ));
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Outlook messages".to_string()
    }
}

// ---------------------------------------------------------------------------
// outlook_get_message
// ---------------------------------------------------------------------------

/// Get a specific Outlook message with full body.
pub struct OutlookGetMessageTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct GetMessageArgs {
    message_id: String,
}

#[async_trait]
impl Tool for OutlookGetMessageTool {
    fn name(&self) -> &str {
        "outlook_get_message"
    }

    fn description(&self) -> &str {
        "Get a specific Outlook email message with full body."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The message ID"
                }
            },
            "required": ["message_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetMessageArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("messages/{}", args.message_id);

        let msg: super::types::Message = self.client.get(&path).await?;

        let subject = msg.subject.as_deref().unwrap_or("(no subject)");
        let from = msg
            .from
            .as_ref()
            .and_then(|r| r.email_address.as_ref())
            .and_then(|e| e.address.as_deref())
            .unwrap_or("?");
        let date = msg.received_date_time.as_deref().unwrap_or("?");
        let body = msg
            .body
            .as_ref()
            .and_then(|b| b.content.as_deref())
            .unwrap_or("");
        let body_type = msg
            .body
            .as_ref()
            .and_then(|b| b.content_type.as_deref())
            .unwrap_or("text");

        let mut out = String::new();
        out.push_str(&format!("Subject: {subject}\n"));
        out.push_str(&format!("From: {from}\n"));
        out.push_str(&format!("Date: {date}\n"));
        out.push_str(&format!("Content-Type: {body_type}\n\n"));
        out.push_str(body);

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["message_id"].as_str().unwrap_or("...");
        format!("Reading Outlook message {id}")
    }
}

// ---------------------------------------------------------------------------
// outlook_list_folders
// ---------------------------------------------------------------------------

/// List all mail folders in the Outlook mailbox.
pub struct OutlookListFoldersTool {
    pub client: Arc<OutlookClient>,
}

#[async_trait]
impl Tool for OutlookListFoldersTool {
    fn name(&self) -> &str {
        "outlook_list_folders"
    }

    fn description(&self) -> &str {
        "List all mail folders in the Outlook mailbox."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let resp: FolderListResponse = self.client.get("mailFolders?$top=50").await?;

        let mut out = String::new();
        for f in &resp.value {
            let name = f.display_name.as_deref().unwrap_or("?");
            let id = &f.id;
            let unread = f.unread_item_count.unwrap_or(0);
            let total = f.total_item_count.unwrap_or(0);
            out.push_str(&format!(
                "- {name} ({unread} unread / {total} total) [{id}]\n"
            ));
        }
        if out.is_empty() {
            out.push_str("No folders found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Outlook mail folders".to_string()
    }
}

// ---------------------------------------------------------------------------
// outlook_send_mail
// ---------------------------------------------------------------------------

/// Send an email via Outlook.
pub struct OutlookSendMailTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct SendMailArgs {
    to: Vec<String>,
    subject: String,
    body: String,
    cc: Option<Vec<String>>,
    bcc: Option<Vec<String>>,
}

fn make_recipients(addresses: &[String]) -> Vec<RecipientInput> {
    addresses
        .iter()
        .map(|addr| RecipientInput {
            email_address: EmailAddressInput {
                address: addr.clone(),
            },
        })
        .collect()
}

#[async_trait]
impl Tool for OutlookSendMailTool {
    fn name(&self) -> &str {
        "outlook_send_mail"
    }

    fn description(&self) -> &str {
        "Send an email via Outlook."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Recipient email addresses"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text)"
                },
                "cc": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "CC recipients"
                },
                "bcc": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "BCC recipients"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SendMailArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        let request = SendMailRequest {
            message: NewMessage {
                subject: args.subject,
                body: MessageBody {
                    content_type: "Text",
                    content: args.body,
                },
                to_recipients: make_recipients(&args.to),
                cc_recipients: args.cc.as_deref().map(make_recipients),
                bcc_recipients: args.bcc.as_deref().map(make_recipients),
            },
        };

        let _: SendMailResponse = self.client.post("sendMail", &request).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Email sent to {}", args.to.join(", ")),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args["to"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        format!("Sending Outlook email to '{to}'")
    }
}

// ---------------------------------------------------------------------------
// outlook_create_draft
// ---------------------------------------------------------------------------

/// Create a draft email in Outlook.
pub struct OutlookCreateDraftTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct CreateDraftArgs {
    to: Vec<String>,
    subject: String,
    body: String,
    cc: Option<Vec<String>>,
}

#[async_trait]
impl Tool for OutlookCreateDraftTool {
    fn name(&self) -> &str {
        "outlook_create_draft"
    }

    fn description(&self) -> &str {
        "Create a draft email in Outlook."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Recipient email addresses"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text)"
                },
                "cc": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "CC recipients"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateDraftArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        let draft = NewMessage {
            subject: args.subject,
            body: MessageBody {
                content_type: "Text",
                content: args.body,
            },
            to_recipients: make_recipients(&args.to),
            cc_recipients: args.cc.as_deref().map(make_recipients),
            bcc_recipients: None,
        };

        let resp: CreateDraftResponse = self.client.post("messages", &draft).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Draft created: id={}", resp.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args["to"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        format!("Creating Outlook draft to '{to}'")
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
            Box::new(OutlookListMessagesTool {
                client: client.clone(),
            }),
            Box::new(OutlookGetMessageTool {
                client: client.clone(),
            }),
            Box::new(OutlookListFoldersTool {
                client: client.clone(),
            }),
            Box::new(OutlookSendMailTool {
                client: client.clone(),
            }),
            Box::new(OutlookCreateDraftTool {
                client: client.clone(),
            }),
        ];

        let expected = [
            "outlook_list_messages",
            "outlook_get_message",
            "outlook_list_folders",
            "outlook_send_mail",
            "outlook_create_draft",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
