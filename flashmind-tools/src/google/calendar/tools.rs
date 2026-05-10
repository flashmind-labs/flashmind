//! Google Calendar `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use super::CalendarClient;
use super::types::{CalendarListResponse, Event, EventListResponse};

// ---------------------------------------------------------------------------
// gcal_list_calendars
// ---------------------------------------------------------------------------

/// List all calendars accessible by the authenticated user.
pub struct GcalListCalendarsTool {
    pub client: Arc<CalendarClient>,
}

#[async_trait]
impl Tool for GcalListCalendarsTool {
    fn name(&self) -> &str {
        "gcal_list_calendars"
    }

    fn description(&self) -> &str {
        "List all Google Calendar calendars accessible by the authenticated user."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let resp: CalendarListResponse =
            serde_json::from_value(self.client.get("users/me/calendarList").await?)?;

        let mut out = String::new();
        for cal in &resp.items {
            let primary = if cal.primary.unwrap_or(false) {
                " (primary)"
            } else {
                ""
            };
            let summary = cal.summary.as_deref().unwrap_or("(untitled)");
            out.push_str(&format!("- {}{primary} [{}]\n", summary, cal.id));
        }
        if resp.items.is_empty() {
            out.push_str("No calendars found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Google Calendars".to_string()
    }
}

// ---------------------------------------------------------------------------
// gcal_list_events
// ---------------------------------------------------------------------------

/// List events from a Google Calendar with filtering and pagination.
pub struct GcalListEventsTool {
    pub client: Arc<CalendarClient>,
}

#[derive(Deserialize)]
struct ListEventsArgs {
    calendar_id: Option<String>,
    time_min: Option<String>,
    time_max: Option<String>,
    max_results: Option<u32>,
    q: Option<String>,
    page_token: Option<String>,
}

#[async_trait]
impl Tool for GcalListEventsTool {
    fn name(&self) -> &str {
        "gcal_list_events"
    }

    fn description(&self) -> &str {
        "List events from a Google Calendar. Defaults to the primary calendar."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')"
                },
                "time_min": {
                    "type": "string",
                    "description": "Lower bound (RFC3339) for event start time, e.g. '2024-01-01T00:00:00Z'"
                },
                "time_max": {
                    "type": "string",
                    "description": "Upper bound (RFC3339) for event end time"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max events to return (default 25, max 250)"
                },
                "q": {
                    "type": "string",
                    "description": "Free-text search terms"
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token from a previous response"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListEventsArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let cal_id = args.calendar_id.as_deref().unwrap_or("primary");
        let max = args.max_results.unwrap_or(25).min(250);

        let mut path = format!(
            "calendars/{}/events?maxResults={max}&singleEvents=true&orderBy=startTime",
            urlencoding::encode(cal_id)
        );
        if let Some(t) = &args.time_min {
            path.push_str(&format!("&timeMin={}", urlencoding::encode(t)));
        }
        if let Some(t) = &args.time_max {
            path.push_str(&format!("&timeMax={}", urlencoding::encode(t)));
        }
        if let Some(q) = &args.q {
            path.push_str(&format!("&q={}", urlencoding::encode(q)));
        }
        if let Some(token) = &args.page_token {
            path.push_str(&format!("&pageToken={token}"));
        }

        let resp: EventListResponse = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        for event in &resp.items {
            let summary = event.summary.as_deref().unwrap_or("(no title)");
            let start = event
                .start
                .as_ref()
                .and_then(|s| s.date_time.as_deref().or(s.date.as_deref()))
                .unwrap_or("?");
            let end = event
                .end
                .as_ref()
                .and_then(|e| e.date_time.as_deref().or(e.date.as_deref()))
                .unwrap_or("?");
            let id = event.id.as_deref().unwrap_or("?");
            out.push_str(&format!("- {summary} | {start} → {end} [{id}]\n"));
        }
        if let Some(next) = &resp.next_page_token {
            out.push_str(&format!("\nNext page token: {next}"));
        }
        if resp.items.is_empty() {
            out.push_str("No events found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let cal = args["calendar_id"].as_str().unwrap_or("primary");
        format!("Listing events from '{cal}'")
    }
}

// ---------------------------------------------------------------------------
// gcal_get_event
// ---------------------------------------------------------------------------

/// Get full details of a specific Google Calendar event.
pub struct GcalGetEventTool {
    pub client: Arc<CalendarClient>,
}

#[derive(Deserialize)]
struct GetEventArgs {
    calendar_id: Option<String>,
    event_id: String,
}

#[async_trait]
impl Tool for GcalGetEventTool {
    fn name(&self) -> &str {
        "gcal_get_event"
    }

    fn description(&self) -> &str {
        "Get details of a specific Google Calendar event by ID."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')"
                },
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
        let cal_id = args.calendar_id.as_deref().unwrap_or("primary");
        let path = format!(
            "calendars/{}/events/{}",
            urlencoding::encode(cal_id),
            urlencoding::encode(&args.event_id)
        );

        let event: Event = serde_json::from_value(self.client.get(&path).await?)?;

        let mut out = String::new();
        out.push_str(&format!(
            "Event: {}\n",
            event.summary.as_deref().unwrap_or("(no title)")
        ));
        if let Some(desc) = &event.description {
            out.push_str(&format!("Description: {desc}\n"));
        }
        if let Some(loc) = &event.location {
            out.push_str(&format!("Location: {loc}\n"));
        }
        if let Some(start) = &event.start {
            let t = start
                .date_time
                .as_deref()
                .or(start.date.as_deref())
                .unwrap_or("?");
            out.push_str(&format!("Start: {t}\n"));
        }
        if let Some(end) = &event.end {
            let t = end
                .date_time
                .as_deref()
                .or(end.date.as_deref())
                .unwrap_or("?");
            out.push_str(&format!("End: {t}\n"));
        }
        if let Some(status) = &event.status {
            out.push_str(&format!("Status: {status}\n"));
        }
        if !event.attendees.is_empty() {
            out.push_str("Attendees:\n");
            for a in &event.attendees {
                let email = a.email.as_deref().unwrap_or("?");
                let status = a.response_status.as_deref().unwrap_or("?");
                out.push_str(&format!("  - {email} ({status})\n"));
            }
        }
        if let Some(link) = &event.html_link {
            out.push_str(&format!("Link: {link}\n"));
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Getting calendar event {id}")
    }
}

// ---------------------------------------------------------------------------
// gcal_create_event
// ---------------------------------------------------------------------------

/// Create a new event in Google Calendar.
pub struct GcalCreateEventTool {
    pub client: Arc<CalendarClient>,
}

#[derive(Deserialize)]
struct CreateEventArgs {
    calendar_id: Option<String>,
    summary: String,
    description: Option<String>,
    location: Option<String>,
    start: String,
    end: String,
    start_time_zone: Option<String>,
    end_time_zone: Option<String>,
    attendees: Option<Vec<String>>,
    all_day: Option<bool>,
}

#[async_trait]
impl Tool for GcalCreateEventTool {
    fn name(&self) -> &str {
        "gcal_create_event"
    }

    fn description(&self) -> &str {
        "Create a new event in Google Calendar."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')"
                },
                "summary": {
                    "type": "string",
                    "description": "Event title"
                },
                "description": {
                    "type": "string",
                    "description": "Event description"
                },
                "location": {
                    "type": "string",
                    "description": "Event location"
                },
                "start": {
                    "type": "string",
                    "description": "Start time (RFC3339 for timed, YYYY-MM-DD for all-day)"
                },
                "end": {
                    "type": "string",
                    "description": "End time (RFC3339 for timed, YYYY-MM-DD for all-day)"
                },
                "start_time_zone": {
                    "type": "string",
                    "description": "IANA time zone for start (e.g. 'America/New_York')"
                },
                "end_time_zone": {
                    "type": "string",
                    "description": "IANA time zone for end"
                },
                "attendees": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Email addresses of attendees"
                },
                "all_day": {
                    "type": "boolean",
                    "description": "If true, use date instead of dateTime fields"
                }
            },
            "required": ["summary", "start", "end"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: CreateEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let cal_id = args.calendar_id.as_deref().unwrap_or("primary");
        let all_day = args.all_day.unwrap_or(false);

        let start = if all_day {
            json!({ "date": args.start })
        } else {
            let mut s = json!({ "dateTime": args.start });
            if let Some(tz) = &args.start_time_zone {
                s["timeZone"] = json!(tz);
            }
            s
        };
        let end = if all_day {
            json!({ "date": args.end })
        } else {
            let mut e = json!({ "dateTime": args.end });
            if let Some(tz) = &args.end_time_zone {
                e["timeZone"] = json!(tz);
            }
            e
        };

        let mut body = json!({
            "summary": args.summary,
            "start": start,
            "end": end,
        });
        if let Some(desc) = &args.description {
            body["description"] = json!(desc);
        }
        if let Some(loc) = &args.location {
            body["location"] = json!(loc);
        }
        if let Some(attendees) = &args.attendees {
            body["attendees"] = json!(
                attendees
                    .iter()
                    .map(|e| json!({"email": e}))
                    .collect::<Vec<_>>()
            );
        }

        let path = format!("calendars/{}/events", urlencoding::encode(cal_id));
        let resp: Event = self.client.post(&path, &body).await?;
        let event_id = resp.id.as_deref().unwrap_or("unknown");
        let link = resp.html_link.as_deref().unwrap_or("");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event created: id={event_id}\n{link}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let summary = args["summary"].as_str().unwrap_or("...");
        format!("Creating calendar event '{summary}'")
    }
}

// ---------------------------------------------------------------------------
// gcal_update_event
// ---------------------------------------------------------------------------

/// Update an existing Google Calendar event (partial update via PATCH).
pub struct GcalUpdateEventTool {
    pub client: Arc<CalendarClient>,
}

#[derive(Deserialize)]
struct UpdateEventArgs {
    calendar_id: Option<String>,
    event_id: String,
    summary: Option<String>,
    description: Option<String>,
    location: Option<String>,
    start: Option<String>,
    end: Option<String>,
    start_time_zone: Option<String>,
    end_time_zone: Option<String>,
}

#[async_trait]
impl Tool for GcalUpdateEventTool {
    fn name(&self) -> &str {
        "gcal_update_event"
    }

    fn description(&self) -> &str {
        "Update an existing Google Calendar event. Only provided fields are changed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')"
                },
                "event_id": {
                    "type": "string",
                    "description": "The event ID to update"
                },
                "summary": {
                    "type": "string",
                    "description": "New event title"
                },
                "description": {
                    "type": "string",
                    "description": "New event description"
                },
                "location": {
                    "type": "string",
                    "description": "New event location"
                },
                "start": {
                    "type": "string",
                    "description": "New start time (RFC3339)"
                },
                "end": {
                    "type": "string",
                    "description": "New end time (RFC3339)"
                },
                "start_time_zone": {
                    "type": "string",
                    "description": "IANA time zone for start"
                },
                "end_time_zone": {
                    "type": "string",
                    "description": "IANA time zone for end"
                }
            },
            "required": ["event_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: UpdateEventArgs = flashmind_types::tool::parse_args(self.name(), ctx.args)?;
        let cal_id = args.calendar_id.as_deref().unwrap_or("primary");

        let mut body = json!({});
        if let Some(s) = &args.summary {
            body["summary"] = json!(s);
        }
        if let Some(d) = &args.description {
            body["description"] = json!(d);
        }
        if let Some(l) = &args.location {
            body["location"] = json!(l);
        }
        if let Some(start) = &args.start {
            let mut s = json!({ "dateTime": start });
            if let Some(tz) = &args.start_time_zone {
                s["timeZone"] = json!(tz);
            }
            body["start"] = s;
        }
        if let Some(end) = &args.end {
            let mut e = json!({ "dateTime": end });
            if let Some(tz) = &args.end_time_zone {
                e["timeZone"] = json!(tz);
            }
            body["end"] = e;
        }

        let path = format!(
            "calendars/{}/events/{}",
            urlencoding::encode(cal_id),
            urlencoding::encode(&args.event_id)
        );
        let _: Event = self.client.patch(&path, &body).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event {} updated", args.event_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Updating calendar event {id}")
    }
}

// ---------------------------------------------------------------------------
// gcal_delete_event
// ---------------------------------------------------------------------------

/// Delete a Google Calendar event by ID.
pub struct GcalDeleteEventTool {
    pub client: Arc<CalendarClient>,
}

#[derive(Deserialize)]
struct DeleteEventArgs {
    calendar_id: Option<String>,
    event_id: String,
}

#[async_trait]
impl Tool for GcalDeleteEventTool {
    fn name(&self) -> &str {
        "gcal_delete_event"
    }

    fn description(&self) -> &str {
        "Delete a Google Calendar event by ID."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')"
                },
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
        let cal_id = args.calendar_id.as_deref().unwrap_or("primary");
        let path = format!(
            "calendars/{}/events/{}",
            urlencoding::encode(cal_id),
            urlencoding::encode(&args.event_id)
        );

        self.client.delete(&path).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event {} deleted", args.event_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args["event_id"].as_str().unwrap_or("...");
        format!("Deleting calendar event {id}")
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
        let client = Arc::new(GoogleClient::new_for_test(
            super::super::BASE_URL,
            super::super::SCOPE,
        ));

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(GcalListCalendarsTool {
                client: client.clone(),
            }),
            Box::new(GcalListEventsTool {
                client: client.clone(),
            }),
            Box::new(GcalGetEventTool {
                client: client.clone(),
            }),
            Box::new(GcalCreateEventTool {
                client: client.clone(),
            }),
            Box::new(GcalUpdateEventTool {
                client: client.clone(),
            }),
            Box::new(GcalDeleteEventTool {
                client: client.clone(),
            }),
        ];

        let expected = [
            "gcal_list_calendars",
            "gcal_list_events",
            "gcal_get_event",
            "gcal_create_event",
            "gcal_update_event",
            "gcal_delete_event",
        ];

        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
            assert!(!tool.description().is_empty());
            assert_eq!(tool.parameters()["type"], "object");
        }
    }
}
