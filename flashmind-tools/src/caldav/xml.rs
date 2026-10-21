//! WebDAV XML request builders and response parsers.
//!
//! Builds CalDAV-specific XML request bodies and parses multistatus responses
//! using `quick_xml`.

use std::collections::HashMap;

use anyhow::{Context, Result};
use quick_xml::Reader;
use quick_xml::events::Event;

// ---------------------------------------------------------------------------
// Request builders
// ---------------------------------------------------------------------------

/// Build a PROPFIND body to discover calendars and their properties.
pub fn propfind_calendars() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<D:propfind xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav" xmlns:A="http://apple.com/ns/ical/">
  <D:prop>
    <D:resourcetype/>
    <D:displayname/>
    <A:calendar-color/>
    <C:calendar-description/>
  </D:prop>
</D:propfind>"#
        .to_string()
}

/// Build a REPORT body for a calendar-query with time-range filter.
///
/// `start` and `end` must be in iCalendar UTC format (e.g. `20240115T100000Z`).
pub fn calendar_query(start: &str, end: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/>
    <C:calendar-data/>
  </D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        <C:time-range start="{start}" end="{end}"/>
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>"#
    )
}

/// Build a REPORT body for text-match search on event summaries.
pub fn text_search(query: &str) -> String {
    // Escape XML special characters in the query
    let escaped = quick_xml::escape::escape(query);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/>
    <C:calendar-data/>
  </D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        <C:prop-filter name="SUMMARY">
          <C:text-match collation="i;unicode-casemap" match-type="contains">{escaped}</C:text-match>
        </C:prop-filter>
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>"#
    )
}

/// Build a MKCALENDAR body with optional display name, color, and description.
pub fn mkcalendar_body(name: &str, color: Option<&str>, description: Option<&str>) -> String {
    let escaped_name = quick_xml::escape::escape(name);
    let color_prop = color
        .map(|c| {
            let escaped = quick_xml::escape::escape(c);
            format!("      <A:calendar-color>{escaped}</A:calendar-color>\n")
        })
        .unwrap_or_default();
    let desc_prop = description
        .map(|d| {
            let escaped = quick_xml::escape::escape(d);
            format!("      <C:calendar-description>{escaped}</C:calendar-description>\n")
        })
        .unwrap_or_default();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<C:mkcalendar xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav" xmlns:A="http://apple.com/ns/ical/">
  <D:set>
    <D:prop>
      <D:displayname>{escaped_name}</D:displayname>
{color_prop}{desc_prop}    </D:prop>
  </D:set>
</C:mkcalendar>"#
    )
}

/// Build a PROPPATCH body to update a resource's display name.
pub fn proppatch_displayname(new_name: &str) -> String {
    let escaped = quick_xml::escape::escape(new_name);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<D:propertyupdate xmlns:D="DAV:">
  <D:set>
    <D:prop>
      <D:displayname>{escaped}</D:displayname>
    </D:prop>
  </D:set>
</D:propertyupdate>"#
    )
}

// ---------------------------------------------------------------------------
// Response parser
// ---------------------------------------------------------------------------

/// A single response entry from a WebDAV multistatus response.
#[derive(Debug, Clone)]
pub struct DavResponse {
    /// The resource path (from `<D:href>`).
    pub href: String,
    /// The HTTP status line (from `<D:status>`), if present.
    pub status: Option<String>,
    /// Parsed properties keyed by local name (e.g. `"displayname"`, `"calendar-color"`).
    pub properties: HashMap<String, String>,
    /// Raw iCalendar data from `<C:calendar-data>`, if present.
    pub calendar_data: Option<String>,
}

/// Parse a WebDAV multistatus XML response into individual response entries.
pub fn parse_multistatus(xml: &str) -> Result<Vec<DavResponse>> {
    let mut reader = Reader::from_str(xml);

    let mut responses = Vec::new();
    let mut current: Option<DavResponse> = None;
    let mut text = String::new();
    let mut in_resourcetype = false;
    let mut resourcetype_values: Vec<String> = Vec::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local = e.local_name().as_ref().to_owned();
                text.clear();
                match local.as_str() {
                    "response" => {
                        current = Some(DavResponse {
                            href: String::new(),
                            status: None,
                            properties: HashMap::new(),
                            calendar_data: None,
                        });
                        resourcetype_values.clear();
                    }
                    "resourcetype" => {
                        in_resourcetype = true;
                        resourcetype_values.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let local = e.local_name().as_ref().to_owned();
                if in_resourcetype {
                    resourcetype_values.push(local);
                }
            }
            Ok(Event::Text(e)) if current.is_some() => {
                text.push_str(e.as_ref());
            }
            Ok(Event::CData(e)) if current.is_some() => {
                text.push_str(e.as_ref());
            }
            Ok(Event::GeneralRef(e)) if current.is_some() => {
                let reference = format!("&{};", e.as_ref());
                text.push_str(
                    &quick_xml::escape::unescape(&reference).context("unescaping XML reference")?,
                );
            }
            Ok(Event::End(ref e)) => {
                let local = e.local_name();
                if let Some(ref mut resp) = current {
                    match local.as_ref() {
                        "href" => resp.href = text.trim().to_owned(),
                        "status" => resp.status = Some(text.trim().to_owned()),
                        "displayname" | "calendar-color" | "calendar-description" | "getetag" => {
                            resp.properties
                                .insert(local.as_ref().to_owned(), text.trim().to_owned());
                        }
                        "calendar-data" => resp.calendar_data = Some(text.trim().to_owned()),
                        _ => {}
                    }
                }
                match local.as_ref() {
                    "response" => {
                        if let Some(mut resp) = current.take() {
                            if !resourcetype_values.is_empty() {
                                resp.properties
                                    .insert("resourcetype".into(), resourcetype_values.join(","));
                            }
                            responses.push(resp);
                        }
                        resourcetype_values.clear();
                    }
                    "resourcetype" => {
                        in_resourcetype = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "XML parse error at position {}: {e}",
                    reader.error_position()
                ));
            }
            _ => {}
        }
    }

    Ok(responses)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propfind_calendars_is_valid_xml() {
        let xml = propfind_calendars();
        assert!(xml.contains("<D:propfind"));
        assert!(xml.contains("<D:displayname/>"));
        assert!(xml.contains("<A:calendar-color/>"));
        // Verify it parses without error
        let mut reader = Reader::from_str(&xml);
        loop {
            match reader.read_event() {
                Ok(Event::Eof) => break,
                Err(e) => panic!("invalid XML: {e}"),
                _ => {}
            }
        }
    }

    #[test]
    fn calendar_query_includes_time_range() {
        let xml = calendar_query("20240101T000000Z", "20240201T000000Z");
        assert!(xml.contains("20240101T000000Z"));
        assert!(xml.contains("20240201T000000Z"));
        assert!(xml.contains("<C:time-range"));
    }

    #[test]
    fn text_search_escapes_xml() {
        let xml = text_search("meeting & lunch");
        assert!(xml.contains("meeting &amp; lunch"));
        assert!(xml.contains("<C:text-match"));
    }

    #[test]
    fn mkcalendar_body_with_all_fields() {
        let xml = mkcalendar_body("Work", Some("#FF0000"), Some("Work events"));
        assert!(xml.contains("<D:displayname>Work</D:displayname>"));
        assert!(xml.contains("<A:calendar-color>#FF0000</A:calendar-color>"));
        assert!(xml.contains("<C:calendar-description>Work events</C:calendar-description>"));
    }

    #[test]
    fn mkcalendar_body_minimal() {
        let xml = mkcalendar_body("Personal", None, None);
        assert!(xml.contains("<D:displayname>Personal</D:displayname>"));
        assert!(!xml.contains("calendar-color"));
        assert!(!xml.contains("calendar-description"));
    }

    #[test]
    fn proppatch_displayname_escapes() {
        let xml = proppatch_displayname("My <Calendar>");
        assert!(xml.contains("My &lt;Calendar&gt;"));
    }

    #[test]
    fn parse_multistatus_calendars() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<D:multistatus xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav" xmlns:A="http://apple.com/ns/ical/">
  <D:response>
    <D:href>/dav/calendars/user/</D:href>
    <D:propstat>
      <D:prop>
        <D:resourcetype><D:collection/></D:resourcetype>
        <D:displayname>Calendars</D:displayname>
      </D:prop>
    </D:propstat>
  </D:response>
  <D:response>
    <D:href>/dav/calendars/user/personal/</D:href>
    <D:propstat>
      <D:prop>
        <D:resourcetype><D:collection/><C:calendar/></D:resourcetype>
        <D:displayname>Personal</D:displayname>
        <A:calendar-color>#0000FF</A:calendar-color>
        <C:calendar-description>My personal calendar</C:calendar-description>
      </D:prop>
    </D:propstat>
  </D:response>
</D:multistatus>"#;

        let responses = parse_multistatus(xml).unwrap();
        assert_eq!(responses.len(), 2);

        let root = &responses[0];
        assert_eq!(root.href, "/dav/calendars/user/");
        assert_eq!(root.properties.get("resourcetype").unwrap(), "collection");

        let cal = &responses[1];
        assert_eq!(cal.href, "/dav/calendars/user/personal/");
        assert_eq!(cal.properties.get("displayname").unwrap(), "Personal");
        assert_eq!(cal.properties.get("calendar-color").unwrap(), "#0000FF");
        assert_eq!(
            cal.properties.get("calendar-description").unwrap(),
            "My personal calendar"
        );
        assert!(
            cal.properties
                .get("resourcetype")
                .unwrap()
                .contains("calendar")
        );
    }

    #[test]
    fn parse_multistatus_events() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<D:multistatus xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:response>
    <D:href>/dav/calendars/user/personal/event1.ics</D:href>
    <D:propstat>
      <D:prop>
        <D:getetag>"abc123"</D:getetag>
        <C:calendar-data>BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event1@example.com
SUMMARY:Team Meeting
DTSTART:20240115T100000Z
DTEND:20240115T110000Z
END:VEVENT
END:VCALENDAR</C:calendar-data>
      </D:prop>
    </D:propstat>
  </D:response>
</D:multistatus>"#;

        let responses = parse_multistatus(xml).unwrap();
        assert_eq!(responses.len(), 1);

        let resp = &responses[0];
        assert_eq!(resp.href, "/dav/calendars/user/personal/event1.ics");
        assert_eq!(resp.properties.get("getetag").unwrap(), "\"abc123\"");
        assert!(resp.calendar_data.is_some());
        assert!(
            resp.calendar_data
                .as_ref()
                .unwrap()
                .contains("Team Meeting")
        );
    }

    #[test]
    fn parse_empty_multistatus() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<D:multistatus xmlns:D="DAV:">
</D:multistatus>"#;
        let responses = parse_multistatus(xml).unwrap();
        assert!(responses.is_empty());
    }

    #[test]
    fn parse_multistatus_preserves_text_across_references_and_cdata() {
        let xml = r#"<multistatus xmlns="DAV:">
  <response>
    <href>/calendars/work/?a=1&amp;b=2</href>
    <propstat><prop>
      <displayname> Work &amp; Personal &#38; &#x1F4C5; </displayname>
      <getetag>&quot;abc123&quot;</getetag>
      <calendar-description>Text <![CDATA[with <tags> & literal &amp;]]> after</calendar-description>
      <calendar-data>BEGIN:VCALENDAR
SUMMARY:Work &amp; Personal
END:VCALENDAR</calendar-data>
    </prop></propstat>
  </response>
</multistatus>"#;
        let responses = parse_multistatus(xml).unwrap();
        let response = &responses[0];
        assert_eq!(response.href, "/calendars/work/?a=1&b=2");
        assert_eq!(
            response.properties["displayname"],
            "Work & Personal & \u{1f4c5}"
        );
        assert_eq!(response.properties["getetag"], "\"abc123\"");
        assert_eq!(
            response.properties["calendar-description"],
            "Text with <tags> & literal &amp; after"
        );
        assert_eq!(
            response.calendar_data.as_deref(),
            Some("BEGIN:VCALENDAR\nSUMMARY:Work & Personal\nEND:VCALENDAR")
        );
    }

    #[test]
    fn parse_multistatus_rejects_unknown_entity() {
        let xml =
            "<multistatus><response><displayname>&unknown;</displayname></response></multistatus>";
        assert!(parse_multistatus(xml).is_err());
    }
}
