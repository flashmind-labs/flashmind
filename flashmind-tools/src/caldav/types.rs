//! CalDAV data types and iCalendar helpers.
//!
//! Provides structured types for calendars and events, plus simple iCalendar
//! parsing and generation without external iCalendar crate dependencies.

use chrono::Utc;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Calendar info
// ---------------------------------------------------------------------------

/// Metadata for a CalDAV calendar collection.
#[derive(Debug, Clone, Serialize)]
pub struct CalendarInfo {
    /// The CalDAV href path for this calendar.
    pub href: String,
    /// Display name of the calendar.
    pub display_name: String,
    /// Calendar color (hex string), if set.
    pub color: Option<String>,
    /// Calendar description, if set.
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// Event info
// ---------------------------------------------------------------------------

/// Parsed VEVENT data from iCalendar format.
#[derive(Debug, Clone, Serialize)]
pub struct EventInfo {
    /// The CalDAV href path for this event resource.
    pub href: String,
    /// Unique identifier (UID property).
    pub uid: String,
    /// Event summary/title.
    pub summary: String,
    /// Start date/time as it appears in the iCalendar data.
    pub dtstart: String,
    /// End date/time, if present.
    pub dtend: Option<String>,
    /// Location, if present.
    pub location: Option<String>,
    /// Description, if present.
    pub description: Option<String>,
    /// Recurrence rule, if present.
    pub rrule: Option<String>,
}

// ---------------------------------------------------------------------------
// iCalendar parsing
// ---------------------------------------------------------------------------

/// Parse VEVENT blocks from raw iCalendar data.
///
/// Handles line unfolding (continuation lines starting with space/tab) and
/// extracts standard VEVENT properties. The `href` field on returned events
/// will be empty and should be set by the caller.
pub fn parse_events(ical_data: &str) -> Vec<EventInfo> {
    let unfolded = unfold_lines(ical_data);
    let mut events = Vec::new();
    let mut in_vevent = false;
    let mut uid = String::new();
    let mut summary = String::new();
    let mut dtstart = String::new();
    let mut dtend: Option<String> = None;
    let mut location: Option<String> = None;
    let mut description: Option<String> = None;
    let mut rrule: Option<String> = None;

    for line in unfolded.lines() {
        let trimmed = line.trim();
        if trimmed == "BEGIN:VEVENT" {
            in_vevent = true;
            uid.clear();
            summary.clear();
            dtstart.clear();
            dtend = None;
            location = None;
            description = None;
            rrule = None;
            continue;
        }
        if trimmed == "END:VEVENT" {
            if in_vevent && !uid.is_empty() {
                events.push(EventInfo {
                    href: String::new(),
                    uid: uid.clone(),
                    summary: summary.clone(),
                    dtstart: dtstart.clone(),
                    dtend: dtend.clone(),
                    location: location.clone(),
                    description: description.clone(),
                    rrule: rrule.clone(),
                });
            }
            in_vevent = false;
            continue;
        }
        if !in_vevent {
            continue;
        }

        if let Some(val) = extract_property(trimmed, "UID") {
            uid = val;
        } else if let Some(val) = extract_property(trimmed, "SUMMARY") {
            summary = val;
        } else if let Some(val) = extract_property(trimmed, "DTSTART") {
            dtstart = val;
        } else if let Some(val) = extract_property(trimmed, "DTEND") {
            dtend = Some(val);
        } else if let Some(val) = extract_property(trimmed, "LOCATION") {
            location = Some(val);
        } else if let Some(val) = extract_property(trimmed, "DESCRIPTION") {
            description = Some(val);
        } else if let Some(val) = extract_property(trimmed, "RRULE") {
            rrule = Some(val);
        }
    }

    events
}

/// Extract the value of an iCalendar property, handling both simple (`NAME:value`)
/// and parameterized (`NAME;PARAM=x:value`) forms.
fn extract_property(line: &str, name: &str) -> Option<String> {
    // Check for exact match: "NAME:" or "NAME;...:"
    if let Some(rest) = line.strip_prefix(name) {
        if let Some(stripped) = rest.strip_prefix(':') {
            return Some(stripped.to_string());
        }
        if rest.starts_with(';') {
            // Skip parameters to find the colon
            if let Some(colon_pos) = rest.find(':') {
                return Some(rest[colon_pos + 1..].to_string());
            }
        }
    }
    None
}

/// Unfold iCalendar continuation lines.
///
/// Per RFC 5545, lines starting with a space or tab are continuations of the
/// previous line.
fn unfold_lines(data: &str) -> String {
    let mut result = String::with_capacity(data.len());
    for line in data.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            // Continuation: append without the leading whitespace
            result.push_str(&line[1..]);
        } else {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(line);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// iCalendar generation
// ---------------------------------------------------------------------------

/// Build a complete VCALENDAR document containing a single VEVENT.
///
/// `dtstart` and `dtend` should be in iCalendar format (e.g. `20240115T100000Z`
/// or `20240115` for all-day events).
#[allow(clippy::too_many_arguments)]
pub fn build_vevent(
    uid: &str,
    summary: &str,
    dtstart: &str,
    dtend: &str,
    location: Option<&str>,
    description: Option<&str>,
    attendees: Option<&[String]>,
    rrule: Option<&str>,
) -> String {
    let dtstamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    let mut lines = vec![
        "BEGIN:VCALENDAR".to_string(),
        "VERSION:2.0".to_string(),
        "PRODID:-//Flashmind//CalDAV//EN".to_string(),
        "BEGIN:VEVENT".to_string(),
        format!("UID:{uid}"),
        format!("DTSTAMP:{dtstamp}"),
        format!("DTSTART:{dtstart}"),
        format!("DTEND:{dtend}"),
        format!("SUMMARY:{summary}"),
    ];

    if let Some(loc) = location {
        lines.push(format!("LOCATION:{loc}"));
    }
    if let Some(desc) = description {
        lines.push(format!("DESCRIPTION:{desc}"));
    }
    if let Some(attendee_list) = attendees {
        for attendee in attendee_list {
            lines.push(format!("ATTENDEE;CN={attendee}:mailto:{attendee}"));
        }
    }
    if let Some(rule) = rrule {
        lines.push(format!("RRULE:{rule}"));
    }

    lines.push("END:VEVENT".to_string());
    lines.push("END:VCALENDAR".to_string());

    // iCalendar uses CRLF line endings
    lines.join("\r\n") + "\r\n"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_vevent() {
        let ical = "\
BEGIN:VCALENDAR\r\n\
VERSION:2.0\r\n\
BEGIN:VEVENT\r\n\
UID:test-uid-123@example.com\r\n\
SUMMARY:Team Standup\r\n\
DTSTART:20240115T100000Z\r\n\
DTEND:20240115T101500Z\r\n\
LOCATION:Conference Room A\r\n\
DESCRIPTION:Daily standup meeting\r\n\
END:VEVENT\r\n\
END:VCALENDAR\r\n";

        let events = parse_events(ical);
        assert_eq!(events.len(), 1);

        let ev = &events[0];
        assert_eq!(ev.uid, "test-uid-123@example.com");
        assert_eq!(ev.summary, "Team Standup");
        assert_eq!(ev.dtstart, "20240115T100000Z");
        assert_eq!(ev.dtend.as_deref(), Some("20240115T101500Z"));
        assert_eq!(ev.location.as_deref(), Some("Conference Room A"));
        assert_eq!(ev.description.as_deref(), Some("Daily standup meeting"));
        assert!(ev.rrule.is_none());
    }

    #[test]
    fn parse_parameterized_properties() {
        let ical = "\
BEGIN:VCALENDAR\r\n\
BEGIN:VEVENT\r\n\
UID:param-test@example.com\r\n\
SUMMARY;LANGUAGE=en:Board Meeting\r\n\
DTSTART;TZID=America/New_York:20240115T100000\r\n\
DTEND;VALUE=DATE:20240116\r\n\
END:VEVENT\r\n\
END:VCALENDAR\r\n";

        let events = parse_events(ical);
        assert_eq!(events.len(), 1);

        let ev = &events[0];
        assert_eq!(ev.summary, "Board Meeting");
        assert_eq!(ev.dtstart, "20240115T100000");
        assert_eq!(ev.dtend.as_deref(), Some("20240116"));
    }

    #[test]
    fn parse_with_rrule() {
        let ical = "\
BEGIN:VCALENDAR\r\n\
BEGIN:VEVENT\r\n\
UID:recurring@example.com\r\n\
SUMMARY:Weekly Sync\r\n\
DTSTART:20240115T140000Z\r\n\
DTEND:20240115T150000Z\r\n\
RRULE:FREQ=WEEKLY;BYDAY=MO\r\n\
END:VEVENT\r\n\
END:VCALENDAR\r\n";

        let events = parse_events(ical);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].rrule.as_deref(), Some("FREQ=WEEKLY;BYDAY=MO"));
    }

    #[test]
    fn parse_unfolded_lines() {
        // RFC 5545: folding inserts CRLF + single whitespace. The whitespace is
        // a fold marker and is stripped during unfolding, so the original content
        // must include any desired space before the fold point.
        let ical = "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:fold-test@example.com\r\nSUMMARY:A very long summary that has been \r\n folded across multiple lines\r\nDTSTART:20240115T100000Z\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n";

        let events = parse_events(ical);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].summary,
            "A very long summary that has been folded across multiple lines"
        );
    }

    #[test]
    fn parse_multiple_events() {
        let ical = "\
BEGIN:VCALENDAR\r\n\
BEGIN:VEVENT\r\n\
UID:first@example.com\r\n\
SUMMARY:First\r\n\
DTSTART:20240115T100000Z\r\n\
END:VEVENT\r\n\
BEGIN:VEVENT\r\n\
UID:second@example.com\r\n\
SUMMARY:Second\r\n\
DTSTART:20240116T100000Z\r\n\
END:VEVENT\r\n\
END:VCALENDAR\r\n";

        let events = parse_events(ical);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].uid, "first@example.com");
        assert_eq!(events[1].uid, "second@example.com");
    }

    #[test]
    fn build_vevent_minimal() {
        let ical = build_vevent(
            "test-uid",
            "Test Event",
            "20240115T100000Z",
            "20240115T110000Z",
            None,
            None,
            None,
            None,
        );

        assert!(ical.contains("BEGIN:VCALENDAR"));
        assert!(ical.contains("END:VCALENDAR"));
        assert!(ical.contains("BEGIN:VEVENT"));
        assert!(ical.contains("END:VEVENT"));
        assert!(ical.contains("UID:test-uid"));
        assert!(ical.contains("SUMMARY:Test Event"));
        assert!(ical.contains("DTSTART:20240115T100000Z"));
        assert!(ical.contains("DTEND:20240115T110000Z"));
        assert!(ical.contains("PRODID:-//Flashmind//CalDAV//EN"));
        assert!(!ical.contains("LOCATION:"));
        assert!(!ical.contains("DESCRIPTION:"));
    }

    #[test]
    fn build_vevent_full() {
        let attendees = vec![
            "alice@example.com".to_string(),
            "bob@example.com".to_string(),
        ];
        let ical = build_vevent(
            "full-uid",
            "Team Lunch",
            "20240115T120000Z",
            "20240115T130000Z",
            Some("Cafeteria"),
            Some("Monthly team lunch"),
            Some(&attendees),
            Some("FREQ=MONTHLY;BYDAY=3TU"),
        );

        assert!(ical.contains("LOCATION:Cafeteria"));
        assert!(ical.contains("DESCRIPTION:Monthly team lunch"));
        assert!(ical.contains("ATTENDEE;CN=alice@example.com:mailto:alice@example.com"));
        assert!(ical.contains("ATTENDEE;CN=bob@example.com:mailto:bob@example.com"));
        assert!(ical.contains("RRULE:FREQ=MONTHLY;BYDAY=3TU"));
    }

    #[test]
    fn build_vevent_roundtrip() {
        let ical = build_vevent(
            "roundtrip@example.com",
            "Roundtrip Test",
            "20240115T100000Z",
            "20240115T110000Z",
            Some("Room 101"),
            None,
            None,
            None,
        );

        let events = parse_events(&ical);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].uid, "roundtrip@example.com");
        assert_eq!(events[0].summary, "Roundtrip Test");
        assert_eq!(events[0].location.as_deref(), Some("Room 101"));
    }
}
