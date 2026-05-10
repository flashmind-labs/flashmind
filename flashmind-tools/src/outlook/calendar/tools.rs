//! Outlook Calendar `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::types::{
    AttendeeInput, BodyInput, CreateEventRequest, DateTimeInput, EmailAddressInput, Event,
    EventListResponse, EventResponse, LocationInput, UpdateEventRequest,
};
use crate::outlook::OutlookClient;

// ---------------------------------------------------------------------------
// outlook_list_events
// ---------------------------------------------------------------------------

/// List upcoming Outlook calendar events with optional filtering.
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

        let mut path = format!(
            "events?$top={top}&$orderby=start/dateTime&$select=id,subject,start,end,location,organizer,isAllDay"
        );
        if let Some(filter) = &args.filter {
            path.push_str(&format!("&$filter={}", urlencoding::encode(filter)));
        }

        let resp: EventListResponse = self.client.get(&path).await?;

        let mut out = String::new();
        if resp.value.is_empty() {
            out.push_str("No events found.");
        } else {
            for event in &resp.value {
                let subject = event.subject.as_deref().unwrap_or("(no title)");
                let start = event
                    .start
                    .as_ref()
                    .and_then(|d| d.date_time.as_deref())
                    .unwrap_or("?");
                let end = event
                    .end
                    .as_ref()
                    .and_then(|d| d.date_time.as_deref())
                    .unwrap_or("?");
                let id = &event.id;
                let location = event
                    .location
                    .as_ref()
                    .and_then(|l| l.display_name.as_deref())
                    .unwrap_or("");
                let loc_str = if location.is_empty() {
                    String::new()
                } else {
                    format!(" @ {location}")
                };
                out.push_str(&format!(
                    "- {subject}{loc_str}\n  {start} → {end}\n  [{id}]\n\n"
                ));
            }
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

/// Get full details of a specific Outlook calendar event.
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

        let event: Event = self.client.get(&path).await?;

        let subject = event.subject.as_deref().unwrap_or("(no title)");
        let start = event
            .start
            .as_ref()
            .and_then(|d| d.date_time.as_deref())
            .unwrap_or("?");
        let end = event
            .end
            .as_ref()
            .and_then(|d| d.date_time.as_deref())
            .unwrap_or("?");
        let location = event
            .location
            .as_ref()
            .and_then(|l| l.display_name.as_deref())
            .unwrap_or("");
        let body = event
            .body
            .as_ref()
            .and_then(|b| b.content.as_deref())
            .unwrap_or("");

        let mut out = String::new();
        out.push_str(&format!("Subject: {subject}\n"));
        out.push_str(&format!("Start: {start}\nEnd: {end}\n"));
        if !location.is_empty() {
            out.push_str(&format!("Location: {location}\n"));
        }
        if let Some(attendees) = &event.attendees {
            out.push_str("Attendees:\n");
            for a in attendees {
                let email = a
                    .email_address
                    .as_ref()
                    .and_then(|e| e.address.as_deref())
                    .unwrap_or("?");
                let status = a
                    .status
                    .as_ref()
                    .and_then(|s| s.response.as_deref())
                    .unwrap_or("?");
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

/// Create a new Outlook calendar event.
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

        let request = CreateEventRequest {
            subject: args.subject,
            start: DateTimeInput {
                date_time: args.start,
                time_zone: tz.to_owned(),
            },
            end: DateTimeInput {
                date_time: args.end,
                time_zone: tz.to_owned(),
            },
            location: args.location.map(|l| LocationInput { display_name: l }),
            body: args.body.map(|b| BodyInput {
                content_type: "Text",
                content: b,
            }),
            attendees: args.attendees.map(|list| {
                list.into_iter()
                    .map(|email| AttendeeInput {
                        email_address: EmailAddressInput { address: email },
                        attendee_type: "required",
                    })
                    .collect()
            }),
            is_all_day: args.is_all_day,
        };

        let resp: EventResponse = self.client.post("events", &request).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event created: id={}", resp.id),
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

/// Update an existing Outlook calendar event (partial update via PATCH).
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

        let request = UpdateEventRequest {
            subject: args.subject,
            start: args.start.map(|s| DateTimeInput {
                date_time: s,
                time_zone: tz.to_owned(),
            }),
            end: args.end.map(|e| DateTimeInput {
                date_time: e,
                time_zone: tz.to_owned(),
            }),
            location: args.location.map(|l| LocationInput { display_name: l }),
            body: args.body.map(|b| BodyInput {
                content_type: "Text",
                content: b,
            }),
        };

        let path = format!("events/{}", args.event_id);
        let _resp: EventResponse = self.client.patch(&path, &request).await?;

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

/// Delete an Outlook calendar event by ID.
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
            Box::new(OutlookListEventsTool {
                client: client.clone(),
            }),
            Box::new(OutlookGetEventTool {
                client: client.clone(),
            }),
            Box::new(OutlookCreateEventTool {
                client: client.clone(),
            }),
            Box::new(OutlookUpdateEventTool {
                client: client.clone(),
            }),
            Box::new(OutlookDeleteEventTool {
                client: client.clone(),
            }),
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
