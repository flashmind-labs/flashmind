//! CalDAV `Tool` trait implementations.
//!
//! Provides 10 tools for interacting with any CalDAV-compliant server:
//! list/create/delete/rename calendars, and list/get/create/update/delete/search events.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::CalDavClient;
use super::types::{self, CalendarInfo, EventInfo};
use super::xml;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an ISO 8601 datetime string to iCalendar UTC format.
///
/// Handles formats like `2024-01-15T10:00:00Z` → `20240115T100000Z`
/// and `2024-01-15` → `20240115`.
fn iso_to_ical(iso: &str) -> String {
    iso.replace(['-', ':'], "")
}

/// Format a list of events into a human-readable string.
fn format_events(events: &[EventInfo]) -> String {
    if events.is_empty() {
        return "No events found.".to_string();
    }

    let mut out = String::new();
    for ev in events {
        let end_str = ev.dtend.as_deref().unwrap_or("?");
        let loc_str = ev
            .location
            .as_ref()
            .map(|l| format!(" @ {l}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "- {}{loc_str}\n  {} -> {end_str}\n  [{}]\n\n",
            ev.summary, ev.dtstart, ev.href,
        ));
    }
    out
}

/// Parse events from multistatus XML response, attaching hrefs.
fn parse_events_from_multistatus(xml_body: &str) -> Result<Vec<EventInfo>> {
    let dav_responses = xml::parse_multistatus(xml_body)?;
    let mut events = Vec::new();
    for resp in &dav_responses {
        if let Some(ical_data) = &resp.calendar_data {
            let mut parsed = types::parse_events(ical_data);
            for ev in &mut parsed {
                ev.href = resp.href.clone();
            }
            events.extend(parsed);
        }
    }
    Ok(events)
}

// ---------------------------------------------------------------------------
// caldav_list_calendars
// ---------------------------------------------------------------------------

/// List all calendars on the CalDAV server.
pub struct CalDavListCalendarsTool {
    pub client: Arc<CalDavClient>,
}

#[async_trait]
impl Tool for CalDavListCalendarsTool {
    fn name(&self) -> &str {
        "caldav_list_calendars"
    }

    fn description(&self) -> &str {
        "List all calendars on the CalDAV server."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let body = xml::propfind_calendars();
        let xml_resp = self.client.propfind("/", 1, &body).await?;
        let dav_responses = xml::parse_multistatus(&xml_resp)?;

        let calendars: Vec<CalendarInfo> = dav_responses
            .iter()
            .filter(|r| {
                r.properties
                    .get("resourcetype")
                    .is_some_and(|rt| rt.contains("calendar"))
            })
            .map(|r| CalendarInfo {
                href: r.href.clone(),
                display_name: r
                    .properties
                    .get("displayname")
                    .cloned()
                    .unwrap_or_else(|| "(unnamed)".into()),
                color: r.properties.get("calendar-color").cloned(),
                description: r.properties.get("calendar-description").cloned(),
            })
            .collect();

        if calendars.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No calendars found.".to_string(),
            ));
        }

        let mut out = String::new();
        for cal in &calendars {
            let color_str = cal
                .color
                .as_ref()
                .map(|c| format!(" ({c})"))
                .unwrap_or_default();
            let desc_str = cal
                .description
                .as_ref()
                .map(|d| format!("\n  {d}"))
                .unwrap_or_default();
            out.push_str(&format!(
                "- {}{color_str}{desc_str}\n  [{}]\n\n",
                cal.display_name, cal.href,
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing CalDAV calendars".into()
    }
}

// ---------------------------------------------------------------------------
// caldav_list_events
// ---------------------------------------------------------------------------

/// List events in a calendar within a time range.
pub struct CalDavListEventsTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct ListEventsArgs {
    calendar_href: String,
    start: String,
    end: String,
}

#[async_trait]
impl Tool for CalDavListEventsTool {
    fn name(&self) -> &str {
        "caldav_list_events"
    }

    fn description(&self) -> &str {
        "List events in a CalDAV calendar within a time range."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_href": {
                    "type": "string",
                    "description": "Calendar path (from caldav_list_calendars)"
                },
                "start": {
                    "type": "string",
                    "description": "Start of time range in ISO 8601 format (e.g. 2024-01-15T00:00:00Z)"
                },
                "end": {
                    "type": "string",
                    "description": "End of time range in ISO 8601 format (e.g. 2024-02-15T00:00:00Z)"
                }
            },
            "required": ["calendar_href", "start", "end"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: ListEventsArgs = parse_args(self.name(), ctx.args)?;
        let ical_start = iso_to_ical(&args.start);
        let ical_end = iso_to_ical(&args.end);

        let body = xml::calendar_query(&ical_start, &ical_end);
        let xml_resp = self.client.report(&args.calendar_href, &body).await?;
        let events = parse_events_from_multistatus(&xml_resp)?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format_events(&events),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let cal = args
            .get("calendar_href")
            .and_then(|v| v.as_str())
            .unwrap_or("calendar");
        format!("Listing events in {cal}")
    }
}

// ---------------------------------------------------------------------------
// caldav_get_event
// ---------------------------------------------------------------------------

/// Get full details of a single CalDAV event.
pub struct CalDavGetEventTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct GetEventArgs {
    event_href: String,
}

#[async_trait]
impl Tool for CalDavGetEventTool {
    fn name(&self) -> &str {
        "caldav_get_event"
    }

    fn description(&self) -> &str {
        "Get full details of a CalDAV event by its href path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_href": {
                    "type": "string",
                    "description": "Event resource path (from caldav_list_events)"
                }
            },
            "required": ["event_href"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: GetEventArgs = parse_args(self.name(), ctx.args)?;
        let ical_data = self.client.get_raw(&args.event_href).await?;
        let mut events = types::parse_events(&ical_data);

        if events.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "No VEVENT found in the resource.".to_string(),
            ));
        }

        let ev = &mut events[0];
        ev.href = args.event_href;

        let output = serde_json::to_string_pretty(ev)?;
        Ok(ToolResult::success(ctx.tool_call_id, output))
    }

    fn humanize(&self, args: &Value) -> String {
        let href = args
            .get("event_href")
            .and_then(|v| v.as_str())
            .unwrap_or("event");
        format!("Getting event {href}")
    }
}

// ---------------------------------------------------------------------------
// caldav_create_event
// ---------------------------------------------------------------------------

/// Create a new event on a CalDAV calendar.
pub struct CalDavCreateEventTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct CreateEventArgs {
    calendar_href: String,
    summary: String,
    dtstart: String,
    dtend: String,
    location: Option<String>,
    description: Option<String>,
    attendees: Option<Vec<String>>,
    rrule: Option<String>,
}

#[async_trait]
impl Tool for CalDavCreateEventTool {
    fn name(&self) -> &str {
        "caldav_create_event"
    }

    fn description(&self) -> &str {
        "Create a new event on a CalDAV calendar."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_href": {
                    "type": "string",
                    "description": "Calendar path to create the event in"
                },
                "summary": {
                    "type": "string",
                    "description": "Event title"
                },
                "dtstart": {
                    "type": "string",
                    "description": "Start date/time in iCalendar format (e.g. 20240115T100000Z)"
                },
                "dtend": {
                    "type": "string",
                    "description": "End date/time in iCalendar format (e.g. 20240115T110000Z)"
                },
                "location": {
                    "type": "string",
                    "description": "Event location"
                },
                "description": {
                    "type": "string",
                    "description": "Event description"
                },
                "attendees": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Email addresses of attendees"
                },
                "rrule": {
                    "type": "string",
                    "description": "Recurrence rule (e.g. FREQ=WEEKLY;BYDAY=MO)"
                }
            },
            "required": ["calendar_href", "summary", "dtstart", "dtend"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: CreateEventArgs = parse_args(self.name(), ctx.args)?;
        let uid = uuid::Uuid::new_v4().to_string();

        let ical = types::build_vevent(
            &uid,
            &args.summary,
            &args.dtstart,
            &args.dtend,
            args.location.as_deref(),
            args.description.as_deref(),
            args.attendees.as_deref(),
            args.rrule.as_deref(),
        );

        let cal_href = args.calendar_href.trim_end_matches('/');
        let event_href = format!("{cal_href}/{uid}.ics");

        self.client.put_ical(&event_href, &ical).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event created: {event_href}"),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let summary = args
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("event");
        format!("Creating CalDAV event: {summary}")
    }
}

// ---------------------------------------------------------------------------
// caldav_update_event
// ---------------------------------------------------------------------------

/// Update an existing CalDAV event.
pub struct CalDavUpdateEventTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct UpdateEventArgs {
    event_href: String,
    summary: Option<String>,
    dtstart: Option<String>,
    dtend: Option<String>,
    location: Option<String>,
    description: Option<String>,
}

#[async_trait]
impl Tool for CalDavUpdateEventTool {
    fn name(&self) -> &str {
        "caldav_update_event"
    }

    fn description(&self) -> &str {
        "Update an existing CalDAV event. Fetches the current event, applies changes, \
         and saves it back. Only specified fields are modified."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_href": {
                    "type": "string",
                    "description": "Event resource path to update"
                },
                "summary": {
                    "type": "string",
                    "description": "New event title"
                },
                "dtstart": {
                    "type": "string",
                    "description": "New start date/time in iCalendar format"
                },
                "dtend": {
                    "type": "string",
                    "description": "New end date/time in iCalendar format"
                },
                "location": {
                    "type": "string",
                    "description": "New location"
                },
                "description": {
                    "type": "string",
                    "description": "New description"
                }
            },
            "required": ["event_href"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: UpdateEventArgs = parse_args(self.name(), ctx.args)?;

        // Fetch existing event
        let existing_ical = self.client.get_raw(&args.event_href).await?;
        let existing_events = types::parse_events(&existing_ical);
        let existing = existing_events
            .first()
            .ok_or_else(|| anyhow::anyhow!("No VEVENT found at {}", args.event_href))?;

        // Merge fields
        let summary = args.summary.as_deref().unwrap_or(&existing.summary);
        let dtstart = args.dtstart.as_deref().unwrap_or(&existing.dtstart);
        let dtend = args
            .dtend
            .as_deref()
            .or(existing.dtend.as_deref())
            .unwrap_or(dtstart);
        let location = args.location.as_deref().or(existing.location.as_deref());
        let description = args
            .description
            .as_deref()
            .or(existing.description.as_deref());

        let ical = types::build_vevent(
            &existing.uid,
            summary,
            dtstart,
            dtend,
            location,
            description,
            None,
            existing.rrule.as_deref(),
        );

        self.client.put_ical(&args.event_href, &ical).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event updated: {}", args.event_href),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let href = args
            .get("event_href")
            .and_then(|v| v.as_str())
            .unwrap_or("event");
        format!("Updating CalDAV event: {href}")
    }
}

// ---------------------------------------------------------------------------
// caldav_delete_event
// ---------------------------------------------------------------------------

/// Delete a CalDAV event.
pub struct CalDavDeleteEventTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct DeleteEventArgs {
    event_href: String,
}

#[async_trait]
impl Tool for CalDavDeleteEventTool {
    fn name(&self) -> &str {
        "caldav_delete_event"
    }

    fn description(&self) -> &str {
        "Delete a CalDAV event by its href path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "event_href": {
                    "type": "string",
                    "description": "Event resource path to delete"
                }
            },
            "required": ["event_href"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: DeleteEventArgs = parse_args(self.name(), ctx.args)?;
        self.client.delete(&args.event_href).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Event deleted: {}", args.event_href),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let href = args
            .get("event_href")
            .and_then(|v| v.as_str())
            .unwrap_or("event");
        format!("Deleting CalDAV event: {href}")
    }
}

// ---------------------------------------------------------------------------
// caldav_search_events
// ---------------------------------------------------------------------------

/// Search for events by text in their summary.
pub struct CalDavSearchEventsTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct SearchEventsArgs {
    calendar_href: String,
    query: String,
}

#[async_trait]
impl Tool for CalDavSearchEventsTool {
    fn name(&self) -> &str {
        "caldav_search_events"
    }

    fn description(&self) -> &str {
        "Search for CalDAV events by text match on the summary field."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_href": {
                    "type": "string",
                    "description": "Calendar path to search in"
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for in event summaries"
                }
            },
            "required": ["calendar_href", "query"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: SearchEventsArgs = parse_args(self.name(), ctx.args)?;

        let body = xml::text_search(&args.query);
        let xml_resp = self.client.report(&args.calendar_href, &body).await?;
        let events = parse_events_from_multistatus(&xml_resp)?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format_events(&events),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
        format!("Searching CalDAV events for \"{query}\"")
    }
}

// ---------------------------------------------------------------------------
// caldav_create_calendar
// ---------------------------------------------------------------------------

/// Create a new calendar collection.
pub struct CalDavCreateCalendarTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct CreateCalendarArgs {
    path: String,
    name: String,
    color: Option<String>,
    description: Option<String>,
}

#[async_trait]
impl Tool for CalDavCreateCalendarTool {
    fn name(&self) -> &str {
        "caldav_create_calendar"
    }

    fn description(&self) -> &str {
        "Create a new calendar on the CalDAV server."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path for the new calendar (e.g. /calendars/user/new-cal/)"
                },
                "name": {
                    "type": "string",
                    "description": "Display name for the calendar"
                },
                "color": {
                    "type": "string",
                    "description": "Calendar color as hex (e.g. #FF0000)"
                },
                "description": {
                    "type": "string",
                    "description": "Calendar description"
                }
            },
            "required": ["path", "name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: CreateCalendarArgs = parse_args(self.name(), ctx.args)?;

        let body = xml::mkcalendar_body(
            &args.name,
            args.color.as_deref(),
            args.description.as_deref(),
        );
        self.client.mkcalendar(&args.path, &body).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Calendar created: {}", args.path),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("calendar");
        format!("Creating CalDAV calendar: {name}")
    }
}

// ---------------------------------------------------------------------------
// caldav_delete_calendar
// ---------------------------------------------------------------------------

/// Delete a calendar collection.
pub struct CalDavDeleteCalendarTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct DeleteCalendarArgs {
    calendar_href: String,
}

#[async_trait]
impl Tool for CalDavDeleteCalendarTool {
    fn name(&self) -> &str {
        "caldav_delete_calendar"
    }

    fn description(&self) -> &str {
        "Delete a calendar from the CalDAV server."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_href": {
                    "type": "string",
                    "description": "Calendar path to delete"
                }
            },
            "required": ["calendar_href"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: DeleteCalendarArgs = parse_args(self.name(), ctx.args)?;
        self.client.delete(&args.calendar_href).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Calendar deleted: {}", args.calendar_href),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let href = args
            .get("calendar_href")
            .and_then(|v| v.as_str())
            .unwrap_or("calendar");
        format!("Deleting CalDAV calendar: {href}")
    }
}

// ---------------------------------------------------------------------------
// caldav_rename_calendar
// ---------------------------------------------------------------------------

/// Rename a calendar by updating its display name.
pub struct CalDavRenameCalendarTool {
    pub client: Arc<CalDavClient>,
}

#[derive(Deserialize)]
struct RenameCalendarArgs {
    calendar_href: String,
    new_name: String,
}

#[async_trait]
impl Tool for CalDavRenameCalendarTool {
    fn name(&self) -> &str {
        "caldav_rename_calendar"
    }

    fn description(&self) -> &str {
        "Rename a CalDAV calendar by updating its display name."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "calendar_href": {
                    "type": "string",
                    "description": "Calendar path to rename"
                },
                "new_name": {
                    "type": "string",
                    "description": "New display name for the calendar"
                }
            },
            "required": ["calendar_href", "new_name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: RenameCalendarArgs = parse_args(self.name(), ctx.args)?;

        let body = xml::proppatch_displayname(&args.new_name);
        self.client.proppatch(&args.calendar_href, &body).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Calendar renamed to: {}", args.new_name),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("new_name")
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        format!("Renaming CalDAV calendar to \"{name}\"")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names() {
        let client = Arc::new(CalDavClient::new_for_test());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(CalDavListCalendarsTool {
                client: client.clone(),
            }),
            Box::new(CalDavListEventsTool {
                client: client.clone(),
            }),
            Box::new(CalDavGetEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavCreateEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavUpdateEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavDeleteEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavSearchEventsTool {
                client: client.clone(),
            }),
            Box::new(CalDavCreateCalendarTool {
                client: client.clone(),
            }),
            Box::new(CalDavDeleteCalendarTool {
                client: client.clone(),
            }),
            Box::new(CalDavRenameCalendarTool {
                client: client.clone(),
            }),
        ];

        let expected = [
            "caldav_list_calendars",
            "caldav_list_events",
            "caldav_get_event",
            "caldav_create_event",
            "caldav_update_event",
            "caldav_delete_event",
            "caldav_search_events",
            "caldav_create_calendar",
            "caldav_delete_calendar",
            "caldav_rename_calendar",
        ];

        assert_eq!(tools.len(), expected.len());
        for (tool, name) in tools.iter().zip(expected.iter()) {
            assert_eq!(tool.name(), *name);
        }
    }

    #[test]
    fn tool_schemas_are_valid() {
        let client = Arc::new(CalDavClient::new_for_test());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(CalDavListCalendarsTool {
                client: client.clone(),
            }),
            Box::new(CalDavListEventsTool {
                client: client.clone(),
            }),
            Box::new(CalDavGetEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavCreateEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavUpdateEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavDeleteEventTool {
                client: client.clone(),
            }),
            Box::new(CalDavSearchEventsTool {
                client: client.clone(),
            }),
            Box::new(CalDavCreateCalendarTool {
                client: client.clone(),
            }),
            Box::new(CalDavDeleteCalendarTool {
                client: client.clone(),
            }),
            Box::new(CalDavRenameCalendarTool {
                client: client.clone(),
            }),
        ];

        for tool in &tools {
            let schema = tool.parameters();
            assert_eq!(
                schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "tool {} schema must be an object",
                tool.name()
            );
            assert!(
                schema.get("properties").is_some(),
                "tool {} schema must have properties",
                tool.name()
            );
        }
    }

    #[test]
    fn humanize_output() {
        let client = Arc::new(CalDavClient::new_for_test());

        let list_tool = CalDavListCalendarsTool {
            client: client.clone(),
        };
        assert_eq!(list_tool.humanize(&json!({})), "Listing CalDAV calendars");

        let search_tool = CalDavSearchEventsTool {
            client: client.clone(),
        };
        assert_eq!(
            search_tool.humanize(&json!({"query": "standup"})),
            "Searching CalDAV events for \"standup\""
        );

        let create_tool = CalDavCreateEventTool {
            client: client.clone(),
        };
        assert_eq!(
            create_tool.humanize(&json!({"summary": "Team Lunch"})),
            "Creating CalDAV event: Team Lunch"
        );

        let rename_tool = CalDavRenameCalendarTool {
            client: client.clone(),
        };
        assert_eq!(
            rename_tool.humanize(&json!({"new_name": "Work"})),
            "Renaming CalDAV calendar to \"Work\""
        );
    }

    #[test]
    fn iso_to_ical_conversion() {
        assert_eq!(iso_to_ical("2024-01-15T10:00:00Z"), "20240115T100000Z");
        assert_eq!(iso_to_ical("2024-01-15"), "20240115");
        assert_eq!(iso_to_ical("2024-06-30T23:59:59Z"), "20240630T235959Z");
    }

    #[test]
    fn format_events_empty() {
        assert_eq!(format_events(&[]), "No events found.");
    }

    #[test]
    fn format_events_with_data() {
        let events = vec![EventInfo {
            href: "/cal/event1.ics".into(),
            uid: "uid1".into(),
            summary: "Meeting".into(),
            dtstart: "20240115T100000Z".into(),
            dtend: Some("20240115T110000Z".into()),
            location: Some("Room A".into()),
            description: None,
            rrule: None,
        }];
        let output = format_events(&events);
        assert!(output.contains("Meeting"));
        assert!(output.contains("Room A"));
        assert!(output.contains("/cal/event1.ics"));
    }
}
