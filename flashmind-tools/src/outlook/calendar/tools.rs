//! Outlook Calendar `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::outlook::OutlookClient;

// ---------------------------------------------------------------------------
// outlook_list_events
// ---------------------------------------------------------------------------

pub struct OutlookListEventsTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct ListEventsArgs {
    top: Option<u32>,
    filter: Option<String>,
}

#[async_trait]
impl Tool for OutlookListEventsTool {
    fn name(&self) -> &str {
        "outlook_list_events"
    }

    fn description(&self) -> &str {
        "List upcoming calendar events from Outlook."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "top": {
                    "type": "integer",
                    "description": "Max events to return (default 25, max 50)"
                },
                "filter": {
                    "type": "string",
                    "description": "OData $filter expression (e.g. \"start/dateTime ge '2024-01-01T00:00'\")"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListEventsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let top = args.top.unwrap_or(25).min(50);

        let mut path = format!("events?$top={top}&$orderby=start/dateTime&$select=id,subject,start,end,location,organizer,isAllDay");
        if let Some(filter) = &args.filter {
            path.push_str(&format!("&$filter={}", urlencoding::encode(filter)));
        }

        let resp = self.client.get(&path).await?;
        let events = resp["value"].as_array();

        let mut out = String::new();
        if let Some(items) = events {
            for event in items {
                let subject = event["subject"].as_str().unwrap_or("(no title)");
                let start = event["start"]["dateTime"].as_str().unwrap_or("?");
                let end = event["end"]["dateTime"].as_str().unwrap_or("?");
                let id = event["id"].as_str().unwrap_or("?");
                let location = event["location"]["displayName"].as_str().unwrap_or("");
                let loc_str = if location.is_empty() { String::new() } else { format!(" @ {location}") };
                out.push_str(&format!("- {subject}{loc_str}\n  {start} → {end}\n  [{id}]\n\n"));
            }
            if items.is_empty() {
                out.push_str("No events found.");
            }
        } else {
            out.push_str("No events found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Outlook calendar events".to_string()
    }
}

// ---------------------------------------------------------------------------
// outlook_get_event
// ---------------------------------------------------------------------------

pub struct OutlookGetEventTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct GetEventArgs {
    event_id: String,
}

#[async_trait]
impl Tool for OutlookGetEventTool {
    fn name(&self) -> &str {
        "outlook_get_event"
    }

    fn description(&self) -> &str {
        "Get details of a specific Outlook calendar event."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The event ID"
                }
            },
            "required": ["event_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GetEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("events/{}", args.event_id);

        let event = self.client.get(&path).await?;

        let subject = event["subject"].as_str().unwrap_or("(no title)");
        let start = event["start"]["dateTime"].as_str().unwrap_or("?");
        let end = event["end"]["dateTime"].as_str().unwrap_or("?");
        let location = event["location"]["displayName"].as_str().unwrap_or("");
        let body = event["body"]["content"].as_str().unwrap_or("");

        let mut out = String::new();
        out.push_str(&format!("Subject: {subject}\n"));
        out.push_str(&format!("Start: {start}\nEnd: {end}\n"));
        if !location.is_empty() {
            out.push_str(&format!("Location: {location}\n"));
        }
        if let Some(attendees) = event["attendees"].as_array() {
            out.push_str("Attendees:\n");
            for a in attendees {
                let email = a["emailAddress"]["address"].as_str().unwrap_or("?");
                let status = a["status"]["response"].as_str().unwrap_or("?");
                out.push_str(&format!("  - {email} ({status})\n"));
            }
        }
        if !body.is_empty() {
            out.push_str(&format!("\n{body}\n"));
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Getting Outlook event {id}")
    }
}

// ---------------------------------------------------------------------------
// outlook_create_event
// ---------------------------------------------------------------------------

pub struct OutlookCreateEventTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct CreateEventArgs {
    subject: String,
    start: String,
    end: String,
    time_zone: Option<String>,
    location: Option<String>,
    body: Option<String>,
    attendees: Option<Vec<String>>,
    is_all_day: Option<bool>,
}

#[async_trait]
impl Tool for OutlookCreateEventTool {
    fn name(&self) -> &str {
        "outlook_create_event"
    }

    fn description(&self) -> &str {
        "Create a new Outlook calendar event."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Event title"
                },
                "start": {
                    "type": "string",
                    "description": "Start time (ISO 8601, e.g. '2024-01-15T09:00:00')"
                },
                "end": {
                    "type": "string",
                    "description": "End time (ISO 8601)"
                },
                "time_zone": {
                    "type": "string",
                    "description": "IANA time zone (default: UTC)"
                },
                "location": {
                    "type": "string",
                    "description": "Event location"
                },
                "body": {
                    "type": "string",
                    "description": "Event description/notes"
                },
                "attendees": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Attendee email addresses"
                },
                "is_all_day": {
                    "type": "boolean",
                    "description": "Whether this is an all-day event"
                }
            },
            "required": ["subject", "start", "end"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let tz = args.time_zone.as_deref().unwrap_or("UTC");

        let mut event = json!({
            "subject": args.subject,
            "start": {
                "dateTime": args.start,
                "timeZone": tz
            },
            "end": {
                "dateTime": args.end,
                "timeZone": tz
            }
        });

        if let Some(loc) = &args.location {
            event["location"] = json!({"displayName": loc});
        }
        if let Some(body) = &args.body {
            event["body"] = json!({"contentType": "Text", "content": body});
        }
        if let Some(attendees) = &args.attendees {
            let list: Vec<Value> = attendees.iter()
                .map(|e| json!({"emailAddress": {"address": e}, "type": "required"}))
                .collect();
            event["attendees"] = json!(list);
        }
        if let Some(true) = args.is_all_day {
            event["isAllDay"] = json!(true);
        }

        let resp = self.client.post("events", event).await?;
        let event_id = resp["id"].as_str().unwrap_or("unknown");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event created: id={event_id}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let subject = args["subject"].as_str().unwrap_or("...");
        format!("Creating Outlook event '{subject}'")
    }
}

// ---------------------------------------------------------------------------
// outlook_update_event
// ---------------------------------------------------------------------------

pub struct OutlookUpdateEventTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct UpdateEventArgs {
    event_id: String,
    subject: Option<String>,
    start: Option<String>,
    end: Option<String>,
    time_zone: Option<String>,
    location: Option<String>,
    body: Option<String>,
}

#[async_trait]
impl Tool for OutlookUpdateEventTool {
    fn name(&self) -> &str {
        "outlook_update_event"
    }

    fn description(&self) -> &str {
        "Update an existing Outlook calendar event. Only provided fields are changed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The event ID to update"
                },
                "subject": {
                    "type": "string",
                    "description": "New event title"
                },
                "start": {
                    "type": "string",
                    "description": "New start time (ISO 8601)"
                },
                "end": {
                    "type": "string",
                    "description": "New end time (ISO 8601)"
                },
                "time_zone": {
                    "type": "string",
                    "description": "IANA time zone"
                },
                "location": {
                    "type": "string",
                    "description": "New location"
                },
                "body": {
                    "type": "string",
                    "description": "New description"
                }
            },
            "required": ["event_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: UpdateEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let tz = args.time_zone.as_deref().unwrap_or("UTC");

        let mut patch = json!({});
        if let Some(s) = &args.subject {
            patch["subject"] = json!(s);
        }
        if let Some(start) = &args.start {
            patch["start"] = json!({"dateTime": start, "timeZone": tz});
        }
        if let Some(end) = &args.end {
            patch["end"] = json!({"dateTime": end, "timeZone": tz});
        }
        if let Some(loc) = &args.location {
            patch["location"] = json!({"displayName": loc});
        }
        if let Some(body) = &args.body {
            patch["body"] = json!({"contentType": "Text", "content": body});
        }

        let path = format!("events/{}", args.event_id);
        self.client.patch(&path, patch).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event {} updated", args.event_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Updating Outlook event {id}")
    }
}

// ---------------------------------------------------------------------------
// outlook_delete_event
// ---------------------------------------------------------------------------

pub struct OutlookDeleteEventTool {
    pub client: Arc<OutlookClient>,
}

#[derive(Deserialize)]
struct DeleteEventArgs {
    event_id: String,
}

#[async_trait]
impl Tool for OutlookDeleteEventTool {
    fn name(&self) -> &str {
        "outlook_delete_event"
    }

    fn description(&self) -> &str {
        "Delete an Outlook calendar event."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The event ID to delete"
                }
            },
            "required": ["event_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: DeleteEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let path = format!("events/{}", args.event_id);
        self.client.delete(&path).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event {} deleted", args.event_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Deleting Outlook event {id}")
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
            Box::new(OutlookListEventsTool { client: client.clone() }),
            Box::new(OutlookGetEventTool { client: client.clone() }),
            Box::new(OutlookCreateEventTool { client: client.clone() }),
            Box::new(OutlookUpdateEventTool { client: client.clone() }),
            Box::new(OutlookDeleteEventTool { client: client.clone() }),
        ];

        let expected = [
            "outlook_list_events",
            "outlook_get_event",
            "outlook_create_event",
            "outlook_update_event",
            "outlook_delete_event",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
