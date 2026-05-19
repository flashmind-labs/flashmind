//! SMTP email tool using the [`lettre`] crate.
//!
//! Sends email via an authenticated SMTP connection with TLS. Supports both
//! plain-text and HTML message bodies.

use anyhow::{Context, Result};
use async_trait::async_trait;
use lettre::message::{MultiPart, SinglePart, header::ContentType};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::SmtpConfig;

// ---------------------------------------------------------------------------
// smtp_send_email
// ---------------------------------------------------------------------------

/// Send an email via SMTP.
pub struct SmtpSendEmailTool {
    /// SMTP configuration (cloned from [`SmtpConfig`]).
    pub config: SmtpConfig,
}

#[derive(Deserialize)]
struct SendEmailArgs {
    to: ToField,
    subject: String,
    body: String,
    html: Option<bool>,
}

/// The `to` field accepts either a single string or an array of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum ToField {
    Single(String),
    Multiple(Vec<String>),
}

impl ToField {
    fn into_vec(self) -> Vec<String> {
        match self {
            ToField::Single(s) => vec![s],
            ToField::Multiple(v) => v,
        }
    }
}

#[async_trait]
impl Tool for SmtpSendEmailTool {
    fn name(&self) -> &str {
        "smtp_send_email"
    }

    fn description(&self) -> &str {
        "Send an email via SMTP. Supports plain-text and HTML bodies, \
         single or multiple recipients."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "description": "Recipient email address(es) — a single string or array of strings",
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ]
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text, or HTML if html=true)"
                },
                "html": {
                    "type": "boolean",
                    "description": "If true, the body is treated as HTML (default false)"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: SendEmailArgs = parse_args(self.name(), ctx.args)?;
        let recipients = args.to.into_vec();
        let is_html = args.html.unwrap_or(false);

        if recipients.is_empty() {
            anyhow::bail!("At least one recipient is required");
        }

        // Build the message
        let from_mailbox = self
            .config
            .from_address
            .parse()
            .context("invalid from_address")?;

        let mut builder = Message::builder().from(from_mailbox).subject(&args.subject);

        for addr in &recipients {
            builder = builder.to(addr.parse().context("invalid recipient address")?);
        }

        let email = if is_html {
            builder.multipart(
                MultiPart::alternative()
                    .singlepart(SinglePart::plain(args.body.clone()))
                    .singlepart(
                        SinglePart::builder()
                            .header(ContentType::TEXT_HTML)
                            .body(args.body),
                    ),
            )?
        } else {
            builder.body(args.body)?
        };

        // Build the transport
        let creds = Credentials::new(self.config.username.clone(), self.config.password.clone());

        let transport = AsyncSmtpTransport::<Tokio1Executor>::relay(&self.config.host)
            .context("failed to create SMTP transport")?
            .port(self.config.port)
            .credentials(creds)
            .build();

        debug!(
            "SMTP sending email to {} via {}:{}",
            recipients.join(", "),
            self.config.host,
            self.config.port
        );

        transport
            .send(email)
            .await
            .context("failed to send email via SMTP")?;

        let recipient_list = recipients.join(", ");
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Email sent to {recipient_list}."),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let to = match args.get("to") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => {
                let addrs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                addrs.join(", ")
            }
            _ => "recipient".into(),
        };
        let subject = args
            .get("subject")
            .and_then(|v| v.as_str())
            .unwrap_or("(no subject)");
        format!("Sending email to {to}: {subject}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SmtpConfig {
        SmtpConfig {
            host: "smtp.example.com".into(),
            port: 587,
            username: "user@example.com".into(),
            password: "password".into(),
            from_address: "user@example.com".into(),
        }
    }

    #[test]
    fn tool_name() {
        let tool = SmtpSendEmailTool {
            config: test_config(),
        };
        assert_eq!(tool.name(), "smtp_send_email");
    }

    #[test]
    fn parameters_are_valid_json_schema() {
        let tool = SmtpSendEmailTool {
            config: test_config(),
        };
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["to"].is_object());
        assert!(params["properties"]["subject"].is_object());
        assert!(params["properties"]["body"].is_object());
    }

    #[test]
    fn humanize_single_recipient() {
        let tool = SmtpSendEmailTool {
            config: test_config(),
        };
        let h = tool.humanize(&json!({
            "to": "bob@example.com",
            "subject": "Hello",
            "body": "Hi Bob"
        }));
        assert!(h.contains("bob@example.com"));
        assert!(h.contains("Hello"));
    }

    #[test]
    fn humanize_multiple_recipients() {
        let tool = SmtpSendEmailTool {
            config: test_config(),
        };
        let h = tool.humanize(&json!({
            "to": ["a@example.com", "b@example.com"],
            "subject": "Test",
            "body": "body"
        }));
        assert!(h.contains("a@example.com"));
        assert!(h.contains("b@example.com"));
    }

    #[test]
    fn to_field_single() {
        let v: ToField = serde_json::from_value(json!("a@b.com")).unwrap();
        assert_eq!(v.into_vec(), vec!["a@b.com"]);
    }

    #[test]
    fn to_field_multiple() {
        let v: ToField = serde_json::from_value(json!(["a@b.com", "c@d.com"])).unwrap();
        assert_eq!(v.into_vec(), vec!["a@b.com", "c@d.com"]);
    }
}
