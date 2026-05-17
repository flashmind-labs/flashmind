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
// outlook_batch_update
// ---------------------------------------------------------------------------

/// Update properties on multiple Outlook messages at once.
pub struct OutlookBatchUpdateTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct BatchUpdateArgs {
    message_ids: Vec<String>,
    is_read: Option<bool>,
    categories: Option<Vec<String>>,
}

#[async_trait]
impl Tool for OutlookBatchUpdateTool {
    fn name(&self) -> &str {
        "outlook_batch_update"
    }

    fn description(&self) -> &str {
        "Update properties on multiple Outlook messages at once. \
         Useful for bulk mark-read/unread, set categories, etc."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Message IDs to update"
                },
                "is_read": {
                    "type": "boolean",
                    "description": "Set read/unread status"
                },
                "categories": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Categories to set on messages"
                }
            },
            "required": ["message_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: BatchUpdateArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        if args.message_ids.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No message IDs provided.".to_string(),
            ));
        }

        let mut patch_body = serde_json::Map::new();
        if let Some(is_read) = args.is_read {
            patch_body.insert("isRead".into(), json!(is_read));
        }
        if let Some(categories) = &args.categories {
            patch_body.insert("categories".into(), json!(categories));
        }

        if patch_body.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No properties to update.".to_string(),
            ));
        }

        let body_val = Value::Object(patch_body);
        let mut headers = serde_json::Map::new();
        headers.insert("Content-Type".into(), json!("application/json"));

        let requests: Vec<_> = args
            .message_ids
            .iter()
            .enumerate()
            .map(|(i, id)| crate::outlook::BatchRequest {
                id: i.to_string(),
                method: "PATCH".into(),
                url: format!("/me/messages/{id}"),
                body: Some(body_val.clone()),
                headers: Some(headers.clone()),
            })
            .collect();

        let responses = self.client.batch(requests).await?;
        let failed: Vec<_> = responses.iter().filter(|r| r.status >= 400).collect();

        let count = args.message_ids.len();
        if failed.is_empty() {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Updated {count} messages"),
            ))
        } else {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Updated {} of {count} messages ({} failed)",
                    count - failed.len(),
                    failed.len()
                ),
            ))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let n = args["message_ids"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        format!("Batch-updating {n} messages")
    }
}

// ---------------------------------------------------------------------------
// outlook_batch_move
// ---------------------------------------------------------------------------

/// Move multiple Outlook messages to a folder at once.
pub struct OutlookBatchMoveTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct BatchMoveArgs {
    message_ids: Vec<String>,
    destination_folder: String,
}

#[async_trait]
impl Tool for OutlookBatchMoveTool {
    fn name(&self) -> &str {
        "outlook_batch_move"
    }

    fn description(&self) -> &str {
        "Move multiple Outlook messages to a folder at once. \
         Useful for bulk archive, move to trash, etc."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Message IDs to move"
                },
                "destination_folder": {
                    "type": "string",
                    "description": "Destination folder ID or well-known name (inbox, archive, deleteditems, junkemail)"
                }
            },
            "required": ["message_ids", "destination_folder"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: BatchMoveArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        if args.message_ids.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No message IDs provided.".to_string(),
            ));
        }

        let body_val = json!({ "destinationId": args.destination_folder });
        let mut headers = serde_json::Map::new();
        headers.insert("Content-Type".into(), json!("application/json"));

        let requests: Vec<_> = args
            .message_ids
            .iter()
            .enumerate()
            .map(|(i, id)| crate::outlook::BatchRequest {
                id: i.to_string(),
                method: "POST".into(),
                url: format!("/me/messages/{id}/move"),
                body: Some(body_val.clone()),
                headers: Some(headers.clone()),
            })
            .collect();

        let responses = self.client.batch(requests).await?;
        let failed: Vec<_> = responses.iter().filter(|r| r.status >= 400).collect();

        let count = args.message_ids.len();
        let dest = &args.destination_folder;
        if failed.is_empty() {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Moved {count} messages to '{dest}'"),
            ))
        } else {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Moved {} of {count} messages to '{dest}' ({} failed)",
                    count - failed.len(),
                    failed.len()
                ),
            ))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let n = args["message_ids"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        let dest = args["destination_folder"].as_str().unwrap_or("...");
        format!("Batch-moving {n} messages to '{dest}'")
    }
}

// ---------------------------------------------------------------------------
// outlook_batch_delete
// ---------------------------------------------------------------------------

/// Delete multiple Outlook messages at once (moves to Deleted Items).
pub struct OutlookBatchDeleteTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct BatchDeleteArgs {
    message_ids: Vec<String>,
}

#[async_trait]
impl Tool for OutlookBatchDeleteTool {
    fn name(&self) -> &str {
        "outlook_batch_delete"
    }

    fn description(&self) -> &str {
        "Delete multiple Outlook messages at once. \
         Messages are moved to the Deleted Items folder."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Message IDs to delete"
                }
            },
            "required": ["message_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: BatchDeleteArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        if args.message_ids.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No message IDs provided.".to_string(),
            ));
        }

        let requests: Vec<_> = args
            .message_ids
            .iter()
            .enumerate()
            .map(|(i, id)| crate::outlook::BatchRequest {
                id: i.to_string(),
                method: "DELETE".into(),
                url: format!("/me/messages/{id}"),
                body: None,
                headers: None,
            })
            .collect();

        let responses = self.client.batch(requests).await?;
        let failed: Vec<_> = responses.iter().filter(|r| r.status >= 400).collect();

        let count = args.message_ids.len();
        if failed.is_empty() {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Deleted {count} messages"),
            ))
        } else {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Deleted {} of {count} messages ({} failed)",
                    count - failed.len(),
                    failed.len()
                ),
            ))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let n = args["message_ids"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        format!("Batch-deleting {n} messages")
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
            Box::new(OutlookBatchUpdateTool {
                client: client.clone(),
            }),
            Box::new(OutlookBatchMoveTool {
                client: client.clone(),
            }),
            Box::new(OutlookBatchDeleteTool {
                client: client.clone(),
            }),
        ];

        let expected = [
            "outlook_list_messages",
            "outlook_get_message",
            "outlook_list_folders",
            "outlook_send_mail",
            "outlook_create_draft",
            "outlook_batch_update",
            "outlook_batch_move",
            "outlook_batch_delete",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
