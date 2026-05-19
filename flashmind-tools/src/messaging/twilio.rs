//! Twilio SMS and WhatsApp tools via the Twilio REST API.
//!
//! All tools authenticate using HTTP Basic auth (`account_sid:auth_token`)
//! and communicate with the `https://api.twilio.com/2010-04-01/Accounts/` endpoint.

use std::fmt::Write;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use crate::utils::http_client;

use super::types::{TwilioMessage, TwilioMessageList};

/// Base URL for the Twilio REST API.
const BASE_URL: &str = "https://api.twilio.com/2010-04-01/Accounts";

// ---------------------------------------------------------------------------
// twilio_list_messages
// ---------------------------------------------------------------------------

/// List recent Twilio messages for the account.
pub struct TwilioListMessagesTool {
    /// Twilio Account SID.
    pub account_sid: String,
    /// Twilio Auth Token.
    pub auth_token: String,
}

#[derive(Deserialize)]
struct ListMessagesArgs {
    limit: Option<u32>,
    to: Option<String>,
    from: Option<String>,
}

#[async_trait]
impl Tool for TwilioListMessagesTool {
    fn name(&self) -> &str {
        "twilio_list_messages"
    }

    fn description(&self) -> &str {
        "List recent SMS/MMS messages from the Twilio account. \
         Returns message SID, sender, recipient, status, date, and body."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default 20)"
                },
                "to": {
                    "type": "string",
                    "description": "Filter by recipient phone number (E.164 format)"
                },
                "from": {
                    "type": "string",
                    "description": "Filter by sender phone number (E.164 format)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: ListMessagesArgs = parse_args(self.name(), ctx.args)?;
        let limit = args.limit.unwrap_or(20).min(100);

        let client = http_client();
        let url = format!("{BASE_URL}/{}/Messages.json", self.account_sid);

        let mut params = vec![("PageSize".to_string(), limit.to_string())];
        if let Some(to) = &args.to {
            params.push(("To".to_string(), to.clone()));
        }
        if let Some(from) = &args.from {
            params.push(("From".to_string(), from.clone()));
        }

        debug!("Twilio GET Messages.json (limit={limit})");

        let resp = client
            .get(&url)
            .basic_auth(&self.account_sid, Some(&self.auth_token))
            .query(&params)
            .send()
            .await
            .context("Twilio API request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Twilio response body")?;
        if !status.is_success() {
            anyhow::bail!("Twilio API error (HTTP {status}): {text}");
        }

        let list: TwilioMessageList =
            serde_json::from_str(&text).context("parsing Twilio message list")?;

        let mut out = String::new();
        if list.messages.is_empty() {
            out.push_str("No messages found.");
        } else {
            for msg in &list.messages {
                let sid_short = if msg.sid.len() > 8 {
                    &msg.sid[msg.sid.len() - 8..]
                } else {
                    &msg.sid
                };
                let from = msg.from.as_deref().unwrap_or("—");
                let to = msg.to.as_deref().unwrap_or("—");
                let date = msg.date_sent.as_deref().unwrap_or("—");
                let body = msg.body.as_deref().unwrap_or("");
                let body_truncated = if body.len() > 80 {
                    let end = body.floor_char_boundary(77);
                    format!("{}...", &body[..end])
                } else {
                    body.to_string()
                };
                let _ = writeln!(
                    out,
                    "- ...{sid_short} | {from} -> {to} | {} | {date} | {body_truncated}",
                    msg.status,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Twilio messages".into()
    }
}

// ---------------------------------------------------------------------------
// twilio_get_message
// ---------------------------------------------------------------------------

/// Retrieve details of a single Twilio message by SID.
pub struct TwilioGetMessageTool {
    /// Twilio Account SID.
    pub account_sid: String,
    /// Twilio Auth Token.
    pub auth_token: String,
}

#[derive(Deserialize)]
struct GetMessageArgs {
    sid: String,
}

#[async_trait]
impl Tool for TwilioGetMessageTool {
    fn name(&self) -> &str {
        "twilio_get_message"
    }

    fn description(&self) -> &str {
        "Get details of a single Twilio message by its SID."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "sid": {
                    "type": "string",
                    "description": "The message SID (e.g. SM1234567890abcdef)"
                }
            },
            "required": ["sid"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: GetMessageArgs = parse_args(self.name(), ctx.args)?;
        let client = http_client();
        let url = format!("{BASE_URL}/{}/Messages/{}.json", self.account_sid, args.sid);

        debug!("Twilio GET Messages/{}.json", args.sid);

        let resp = client
            .get(&url)
            .basic_auth(&self.account_sid, Some(&self.auth_token))
            .send()
            .await
            .context("Twilio API request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Twilio response body")?;
        if !status.is_success() {
            anyhow::bail!("Twilio API error (HTTP {status}): {text}");
        }

        let msg: TwilioMessage = serde_json::from_str(&text).context("parsing Twilio message")?;

        let mut out = String::new();
        let _ = writeln!(out, "SID: {}", msg.sid);
        let _ = writeln!(out, "From: {}", msg.from.as_deref().unwrap_or("—"));
        let _ = writeln!(out, "To: {}", msg.to.as_deref().unwrap_or("—"));
        let _ = writeln!(out, "Status: {}", msg.status);
        let _ = writeln!(
            out,
            "Direction: {}",
            msg.direction.as_deref().unwrap_or("—")
        );
        let _ = writeln!(
            out,
            "Date sent: {}",
            msg.date_sent.as_deref().unwrap_or("—")
        );
        let _ = writeln!(out, "Body: {}", msg.body.as_deref().unwrap_or(""));

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let sid = args
            .get("sid")
            .and_then(|v| v.as_str())
            .unwrap_or("message");
        format!("Getting Twilio message {sid}")
    }
}

// ---------------------------------------------------------------------------
// twilio_send_sms
// ---------------------------------------------------------------------------

/// Send an SMS via the Twilio REST API.
pub struct TwilioSendSmsTool {
    /// Twilio Account SID.
    pub account_sid: String,
    /// Twilio Auth Token.
    pub auth_token: String,
    /// Default sender phone number in E.164 format.
    pub from_number: String,
}

#[derive(Deserialize)]
struct SendSmsArgs {
    to: String,
    body: String,
}

#[async_trait]
impl Tool for TwilioSendSmsTool {
    fn name(&self) -> &str {
        "twilio_send_sms"
    }

    fn description(&self) -> &str {
        "Send an SMS message via Twilio. Requires recipient phone number (E.164) and body text."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient phone number in E.164 format (e.g. +15551234567)"
                },
                "body": {
                    "type": "string",
                    "description": "Message body text"
                }
            },
            "required": ["to", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: SendSmsArgs = parse_args(self.name(), ctx.args)?;
        let client = http_client();
        let url = format!("{BASE_URL}/{}/Messages.json", self.account_sid);

        debug!("Twilio POST Messages.json (SMS to {})", args.to);

        let resp = client
            .post(&url)
            .basic_auth(&self.account_sid, Some(&self.auth_token))
            .form(&[
                ("From", self.from_number.as_str()),
                ("To", args.to.as_str()),
                ("Body", args.body.as_str()),
            ])
            .send()
            .await
            .context("Twilio API request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Twilio response body")?;
        if !status.is_success() {
            anyhow::bail!("Twilio API error (HTTP {status}): {text}");
        }

        let msg: TwilioMessage = serde_json::from_str(&text).context("parsing Twilio response")?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("SMS sent. SID: {} | Status: {}", msg.sid, msg.status),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .unwrap_or("recipient");
        format!("Sending SMS to {to}")
    }
}

// ---------------------------------------------------------------------------
// twilio_send_whatsapp
// ---------------------------------------------------------------------------

/// Send a WhatsApp message via the Twilio REST API.
pub struct TwilioSendWhatsappTool {
    /// Twilio Account SID.
    pub account_sid: String,
    /// Twilio Auth Token.
    pub auth_token: String,
    /// Default sender phone number in E.164 format (without `whatsapp:` prefix).
    pub from_number: String,
}

#[derive(Deserialize)]
struct SendWhatsappArgs {
    to: String,
    body: String,
}

#[async_trait]
impl Tool for TwilioSendWhatsappTool {
    fn name(&self) -> &str {
        "twilio_send_whatsapp"
    }

    fn description(&self) -> &str {
        "Send a WhatsApp message via Twilio. Requires recipient phone number (E.164) and body text."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient phone number in E.164 format (e.g. +15551234567)"
                },
                "body": {
                    "type": "string",
                    "description": "Message body text"
                }
            },
            "required": ["to", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: SendWhatsappArgs = parse_args(self.name(), ctx.args)?;
        let client = http_client();
        let url = format!("{BASE_URL}/{}/Messages.json", self.account_sid);

        let from_wa = format!("whatsapp:{}", self.from_number);
        let to_wa = format!("whatsapp:{}", args.to);

        debug!("Twilio POST Messages.json (WhatsApp to {})", args.to);

        let resp = client
            .post(&url)
            .basic_auth(&self.account_sid, Some(&self.auth_token))
            .form(&[
                ("From", from_wa.as_str()),
                ("To", to_wa.as_str()),
                ("Body", args.body.as_str()),
            ])
            .send()
            .await
            .context("Twilio API request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Twilio response body")?;
        if !status.is_success() {
            anyhow::bail!("Twilio API error (HTTP {status}): {text}");
        }

        let msg: TwilioMessage = serde_json::from_str(&text).context("parsing Twilio response")?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "WhatsApp message sent. SID: {} | Status: {}",
                msg.sid, msg.status
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .unwrap_or("recipient");
        format!("Sending WhatsApp message to {to}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_unique() {
        let list = TwilioListMessagesTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
        };
        let get = TwilioGetMessageTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
        };
        let sms = TwilioSendSmsTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
            from_number: "+15550001111".into(),
        };
        let wa = TwilioSendWhatsappTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
            from_number: "+15550001111".into(),
        };

        let names: Vec<&str> = vec![list.name(), get.name(), sms.name(), wa.name()];
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "tool names must be unique");
    }

    #[test]
    fn parameters_are_valid_json_schema() {
        let tool = TwilioSendSmsTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
            from_number: "+15550001111".into(),
        };
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["to"].is_object());
        assert!(params["properties"]["body"].is_object());
    }

    #[test]
    fn humanize_includes_recipient() {
        let tool = TwilioSendSmsTool {
            account_sid: "AC_test".into(),
            auth_token: "tok".into(),
            from_number: "+15550001111".into(),
        };
        let h = tool.humanize(&json!({"to": "+15559998888", "body": "hi"}));
        assert!(h.contains("+15559998888"));
    }
}
