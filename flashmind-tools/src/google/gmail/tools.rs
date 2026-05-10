//! Gmail `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::GmailClient;
use super::types::{self, DraftListResponse, LabelListResponse, Thread, ThreadListResponse};

// ---------------------------------------------------------------------------
// gmail_search_threads
// ---------------------------------------------------------------------------

pub struct GmailSearchThreadsTool {
    pub client: Arc<GmailClient>,
}

#[derive(Deserialize)]
struct SearchThreadsArgs {
    query: String,
    max_results: Option<u32>,
    page_token: Option<String>,
}

#[async_trait]
impl Tool for GmailSearchThreadsTool {
    fn name(&self) -> &str {
        "gmail_search_threads"
    }

    fn description(&self) -> &str {
        "Search Gmail threads using Gmail search syntax (same as the Gmail search bar). \
         Returns thread IDs and snippets."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query (e.g. 'from:alice subject:meeting is:unread')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max threads to return (default 10, max 50)"
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token from a previous response"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SearchThreadsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let max = args.max_results.unwrap_or(10).min(50);

        let mut path = format!(
            "threads?q={}&maxResults={max}",
            urlencoding::encode(&args.query)
        );
        if let Some(token) = &args.page_token {
            path.push_str(&format!("&pageToken={token}"));
        }

        let resp: ThreadListResponse = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        for t in &resp.threads {
            out.push_str(&format!(
                "- {} {}\n",
                t.id,
                t.snippet.as_deref().unwrap_or("")
            ));
        }
        if let Some(next) = &resp.next_page_token {
            out.push_str(&format!("\nNext page token: {next}"));
        }
        if resp.threads.is_empty() {
            out.push_str("No threads found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let q = args["query"].as_str().unwrap_or("...");
        format!("Searching Gmail for '{q}'")
    }
}

// ---------------------------------------------------------------------------
// gmail_get_thread
// ---------------------------------------------------------------------------

pub struct GmailGetThreadTool {
    pub client: Arc<GmailClient>,
}

#[derive(Deserialize)]
struct GetThreadArgs {
    thread_id: String,
    format: Option<String>,
}

#[async_trait]
impl Tool for GmailGetThreadTool {
    fn name(&self) -> &str {
        "gmail_get_thread"
    }

    fn description(&self) -> &str {
        "Get a Gmail thread by ID, including all messages with decoded bodies."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The thread ID"
                },
                "format": {
                    "type": "string",
                    "enum": ["minimal", "full", "metadata"],
                    "description": "Response format (default: full)"
                }
            },
            "required": ["thread_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetThreadArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let fmt = args.format.as_deref().unwrap_or("full");
        let path = format!("threads/{}?format={fmt}", args.thread_id);

        let resp: Thread = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = format!("Thread {} ({} messages)\n\n", resp.id, resp.messages.len());
        for msg in &resp.messages {
            let from = msg
                .payload
                .as_ref()
                .and_then(|p| {
                    p.headers
                        .iter()
                        .find(|h| h.name.eq_ignore_ascii_case("from"))
                })
                .map(|h| h.value.as_str())
                .unwrap_or("unknown");
            let subject = msg
                .payload
                .as_ref()
                .and_then(|p| {
                    p.headers
                        .iter()
                        .find(|h| h.name.eq_ignore_ascii_case("subject"))
                })
                .map(|h| h.value.as_str())
                .unwrap_or("");
            let date = msg
                .payload
                .as_ref()
                .and_then(|p| {
                    p.headers
                        .iter()
                        .find(|h| h.name.eq_ignore_ascii_case("date"))
                })
                .map(|h| h.value.as_str())
                .unwrap_or("");

            out.push_str(&format!("--- Message {} ---\n", msg.id));
            out.push_str(&format!("From: {from}\nSubject: {subject}\nDate: {date}\n"));
            out.push_str(&format!("Labels: {}\n", msg.label_ids.join(", ")));

            if let Some(payload) = &msg.payload
                && let Some(body) = types::extract_text(payload)
            {
                out.push_str(&format!("\n{body}\n"));
            }
            out.push('\n');
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["thread_id"].as_str().unwrap_or("...");
        format!("Reading Gmail thread {id}")
    }
}

// ---------------------------------------------------------------------------
// gmail_create_draft
// ---------------------------------------------------------------------------

pub struct GmailCreateDraftTool {
    pub client: Arc<GmailClient>,
}

#[derive(Deserialize)]
struct CreateDraftArgs {
    to: String,
    subject: String,
    body: String,
    cc: Option<String>,
    bcc: Option<String>,
    in_reply_to: Option<String>,
    thread_id: Option<String>,
}

fn build_rfc2822(args: &CreateDraftArgs) -> String {
    let mut msg = String::new();
    msg.push_str(&format!("To: {}\r\n", args.to));
    msg.push_str(&format!("Subject: {}\r\n", args.subject));
    if let Some(cc) = &args.cc {
        msg.push_str(&format!("Cc: {cc}\r\n"));
    }
    if let Some(bcc) = &args.bcc {
        msg.push_str(&format!("Bcc: {bcc}\r\n"));
    }
    if let Some(reply_to) = &args.in_reply_to {
        msg.push_str(&format!("In-Reply-To: {reply_to}\r\n"));
        msg.push_str(&format!("References: {reply_to}\r\n"));
    }
    msg.push_str("Content-Type: text/plain; charset=utf-8\r\n");
    msg.push_str("\r\n");
    msg.push_str(&args.body);
    msg
}

#[async_trait]
impl Tool for GmailCreateDraftTool {
    fn name(&self) -> &str {
        "gmail_create_draft"
    }

    fn description(&self) -> &str {
        "Create a new Gmail draft email."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email(s), comma-separated"
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
                    "type": "string",
                    "description": "CC recipients, comma-separated"
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC recipients, comma-separated"
                },
                "in_reply_to": {
                    "type": "string",
                    "description": "Message-ID to reply to"
                },
                "thread_id": {
                    "type": "string",
                    "description": "Thread ID to attach this draft to"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateDraftArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let raw = URL_SAFE_NO_PAD.encode(build_rfc2822(&args));

        let mut body = json!({
            "message": {
                "raw": raw
            }
        });
        if let Some(tid) = &args.thread_id {
            body["message"]["threadId"] = json!(tid);
        }

        let resp = self.client.post("drafts", body).await?;
        let draft_id = resp["id"].as_str().unwrap_or("unknown");
        let msg_id = resp["message"]["id"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Draft created: id={draft_id}, message_id={msg_id}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args["to"].as_str().unwrap_or("...");
        format!("Creating Gmail draft to '{to}'")
    }
}

// ---------------------------------------------------------------------------
// gmail_send
// ---------------------------------------------------------------------------

pub struct GmailSendTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailSendTool {
    fn name(&self) -> &str {
        "gmail_send"
    }

    fn description(&self) -> &str {
        "Send an email immediately via Gmail."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email(s), comma-separated"
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
                    "type": "string",
                    "description": "CC recipients, comma-separated"
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC recipients, comma-separated"
                },
                "in_reply_to": {
                    "type": "string",
                    "description": "Message-ID to reply to"
                },
                "thread_id": {
                    "type": "string",
                    "description": "Thread ID to attach this message to"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateDraftArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let raw = URL_SAFE_NO_PAD.encode(build_rfc2822(&args));

        let mut body = json!({
            "raw": raw
        });
        if let Some(tid) = &args.thread_id {
            body["threadId"] = json!(tid);
        }

        let resp = self.client.post("messages/send", body).await?;
        let msg_id = resp["id"].as_str().unwrap_or("unknown");
        let thread_id = resp["threadId"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Email sent: message_id={msg_id}, thread_id={thread_id}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args["to"].as_str().unwrap_or("...");
        format!("Sending email to '{to}'")
    }
}

// ---------------------------------------------------------------------------
// gmail_list_drafts
// ---------------------------------------------------------------------------

pub struct GmailListDraftsTool {
    pub client: Arc<GmailClient>,
}

#[derive(Deserialize)]
struct ListDraftsArgs {
    max_results: Option<u32>,
    page_token: Option<String>,
}

#[async_trait]
impl Tool for GmailListDraftsTool {
    fn name(&self) -> &str {
        "gmail_list_drafts"
    }

    fn description(&self) -> &str {
        "List Gmail drafts with optional pagination."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Max drafts to return (default 10)"
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token from a previous response"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListDraftsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let max = args.max_results.unwrap_or(10).min(50);

        let mut path = format!("drafts?maxResults={max}");
        if let Some(token) = &args.page_token {
            path.push_str(&format!("&pageToken={token}"));
        }

        let resp: DraftListResponse = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        for d in &resp.drafts {
            let tid = d
                .message
                .as_ref()
                .and_then(|m| m.thread_id.as_deref())
                .unwrap_or("?");
            out.push_str(&format!("- {} (thread: {tid})\n", d.id));
        }
        if let Some(next) = &resp.next_page_token {
            out.push_str(&format!("\nNext page token: {next}"));
        }
        if resp.drafts.is_empty() {
            out.push_str("No drafts found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Gmail drafts".to_string()
    }
}

// ---------------------------------------------------------------------------
// gmail_list_labels
// ---------------------------------------------------------------------------

pub struct GmailListLabelsTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailListLabelsTool {
    fn name(&self) -> &str {
        "gmail_list_labels"
    }

    fn description(&self) -> &str {
        "List all Gmail labels (system and user-created)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let resp: LabelListResponse = serde_json::from_value(self.client.get("labels").await?)?;

        let mut out = String::new();
        for label in &resp.labels {
            let kind = label.r#type.as_deref().unwrap_or("user");
            out.push_str(&format!("- {} ({kind}) [{}]\n", label.name, label.id));
        }
        if resp.labels.is_empty() {
            out.push_str("No labels found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Gmail labels".to_string()
    }
}

// ---------------------------------------------------------------------------
// gmail_create_label
// ---------------------------------------------------------------------------

pub struct GmailCreateLabelTool {
    pub client: Arc<GmailClient>,
}

#[derive(Deserialize)]
struct CreateLabelArgs {
    name: String,
}

#[async_trait]
impl Tool for GmailCreateLabelTool {
    fn name(&self) -> &str {
        "gmail_create_label"
    }

    fn description(&self) -> &str {
        "Create a new Gmail label."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Label name"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateLabelArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;

        let body = json!({
            "name": args.name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show"
        });

        let resp = self.client.post("labels", body).await?;
        let id = resp["id"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Label created: name='{}', id={id}", args.name),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args["name"].as_str().unwrap_or("...");
        format!("Creating Gmail label '{name}'")
    }
}

// ---------------------------------------------------------------------------
// Label modification tools (shared helper)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ModifyLabelsArgs {
    #[serde(alias = "message_id", alias = "thread_id")]
    id: String,
    label_ids: Vec<String>,
}

async fn modify_labels(
    client: &GmailClient,
    entity: &str,
    id: &str,
    add: &[String],
    remove: &[String],
) -> anyhow::Result<serde_json::Value> {
    let path = format!("{entity}/{id}/modify");
    let body = json!({
        "addLabelIds": add,
        "removeLabelIds": remove,
    });
    client.post(&path, body).await
}

// ---------------------------------------------------------------------------
// gmail_label_message
// ---------------------------------------------------------------------------

pub struct GmailLabelMessageTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailLabelMessageTool {
    fn name(&self) -> &str {
        "gmail_label_message"
    }

    fn description(&self) -> &str {
        "Add labels to a Gmail message."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The message ID"
                },
                "label_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Label IDs to add"
                }
            },
            "required": ["message_id", "label_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ModifyLabelsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        modify_labels(&self.client, "messages", &args.id, &args.label_ids, &[]).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Added labels {:?} to message {}", args.label_ids, args.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["message_id"].as_str().unwrap_or("...");
        format!("Adding labels to message {id}")
    }
}

// ---------------------------------------------------------------------------
// gmail_label_thread
// ---------------------------------------------------------------------------

pub struct GmailLabelThreadTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailLabelThreadTool {
    fn name(&self) -> &str {
        "gmail_label_thread"
    }

    fn description(&self) -> &str {
        "Add labels to all messages in a Gmail thread."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The thread ID"
                },
                "label_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Label IDs to add"
                }
            },
            "required": ["thread_id", "label_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ModifyLabelsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        modify_labels(&self.client, "threads", &args.id, &args.label_ids, &[]).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Added labels {:?} to thread {}", args.label_ids, args.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["thread_id"].as_str().unwrap_or("...");
        format!("Adding labels to thread {id}")
    }
}

// ---------------------------------------------------------------------------
// gmail_unlabel_message
// ---------------------------------------------------------------------------

pub struct GmailUnlabelMessageTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailUnlabelMessageTool {
    fn name(&self) -> &str {
        "gmail_unlabel_message"
    }

    fn description(&self) -> &str {
        "Remove labels from a Gmail message."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The message ID"
                },
                "label_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Label IDs to remove"
                }
            },
            "required": ["message_id", "label_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ModifyLabelsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        modify_labels(&self.client, "messages", &args.id, &[], &args.label_ids).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Removed labels {:?} from message {}",
                args.label_ids, args.id
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["message_id"].as_str().unwrap_or("...");
        format!("Removing labels from message {id}")
    }
}

// ---------------------------------------------------------------------------
// gmail_unlabel_thread
// ---------------------------------------------------------------------------

pub struct GmailUnlabelThreadTool {
    pub client: Arc<GmailClient>,
}

#[async_trait]
impl Tool for GmailUnlabelThreadTool {
    fn name(&self) -> &str {
        "gmail_unlabel_thread"
    }

    fn description(&self) -> &str {
        "Remove labels from all messages in a Gmail thread."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The thread ID"
                },
                "label_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Label IDs to remove"
                }
            },
            "required": ["thread_id", "label_ids"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ModifyLabelsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        modify_labels(&self.client, "threads", &args.id, &[], &args.label_ids).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Removed labels {:?} from thread {}",
                args.label_ids, args.id
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["thread_id"].as_str().unwrap_or("...");
        format!("Removing labels from thread {id}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::google::auth;
    use crate::google::client::GoogleConfig;

    #[test]
    fn tool_names_are_correct() {
        let client = Arc::new(dummy_client());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(GmailSearchThreadsTool { client: client.clone() }),
            Box::new(GmailGetThreadTool { client: client.clone() }),
            Box::new(GmailSendTool { client: client.clone() }),
            Box::new(GmailCreateDraftTool { client: client.clone() }),
            Box::new(GmailListDraftsTool { client: client.clone() }),
            Box::new(GmailListLabelsTool { client: client.clone() }),
            Box::new(GmailCreateLabelTool { client: client.clone() }),
            Box::new(GmailLabelMessageTool { client: client.clone() }),
            Box::new(GmailLabelThreadTool { client: client.clone() }),
            Box::new(GmailUnlabelMessageTool { client: client.clone() }),
            Box::new(GmailUnlabelThreadTool { client: client.clone() }),
        ];

        let expected = [
            "gmail_search_threads",
            "gmail_get_thread",
            "gmail_send",
            "gmail_create_draft",
            "gmail_list_drafts",
            "gmail_list_labels",
            "gmail_create_label",
            "gmail_label_message",
            "gmail_label_thread",
            "gmail_unlabel_message",
            "gmail_unlabel_thread",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            let params = tool.parameters();
            assert_eq!(params["type"], "object");
        }
    }

    #[test]
    fn rfc2822_basic() {
        let args = CreateDraftArgs {
            to: "alice@example.com".into(),
            subject: "Hello".into(),
            body: "Hi Alice".into(),
            cc: None,
            bcc: None,
            in_reply_to: None,
            thread_id: None,
        };
        let msg = build_rfc2822(&args);
        assert!(msg.contains("To: alice@example.com\r\n"));
        assert!(msg.contains("Subject: Hello\r\n"));
        assert!(msg.contains("Content-Type: text/plain; charset=utf-8\r\n"));
        assert!(msg.contains("\r\n\r\nHi Alice"));
    }

    #[test]
    fn rfc2822_with_reply() {
        let args = CreateDraftArgs {
            to: "bob@example.com".into(),
            subject: "Re: Test".into(),
            body: "reply body".into(),
            cc: Some("cc@example.com".into()),
            bcc: Some("bcc@example.com".into()),
            in_reply_to: Some("<msg-id@example.com>".into()),
            thread_id: None,
        };
        let msg = build_rfc2822(&args);
        assert!(msg.contains("Cc: cc@example.com\r\n"));
        assert!(msg.contains("Bcc: bcc@example.com\r\n"));
        assert!(msg.contains("In-Reply-To: <msg-id@example.com>\r\n"));
        assert!(msg.contains("References: <msg-id@example.com>\r\n"));
    }

    #[test]
    fn humanize_outputs() {
        let client = Arc::new(dummy_client());

        let search = GmailSearchThreadsTool { client: client.clone() };
        assert_eq!(
            search.humanize(&json!({"query": "is:unread"})),
            "Searching Gmail for 'is:unread'"
        );

        let get = GmailGetThreadTool { client: client.clone() };
        assert_eq!(
            get.humanize(&json!({"thread_id": "abc123"})),
            "Reading Gmail thread abc123"
        );

        let send = GmailSendTool { client: client.clone() };
        assert_eq!(
            send.humanize(&json!({"to": "x@y.com"})),
            "Sending email to 'x@y.com'"
        );

        let draft = GmailCreateDraftTool { client: client.clone() };
        assert_eq!(
            draft.humanize(&json!({"to": "x@y.com"})),
            "Creating Gmail draft to 'x@y.com'"
        );
    }

    fn dummy_client() -> GmailClient {
        // Can't go through the normal constructor without real credentials,
        // so we construct directly for testing.
        use crate::google::client::GoogleClient;
        use tokio::sync::RwLock;

        // SAFETY: This is test-only; we never call access_token() on this client.
        unsafe {
            // We use a raw approach to construct the client for tests
        }

        // Instead, use a helper that bypasses credential loading
        GoogleClient::new_for_test(
            super::super::BASE_URL,
            super::super::SCOPE,
        )
    }
}
