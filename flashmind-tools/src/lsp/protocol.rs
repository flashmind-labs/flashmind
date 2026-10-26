//! JSON-RPC framing and the few LSP response shapes the tool reads.

use std::fmt;
use std::io;

use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// LSP error code for a request invalidated by a concurrent change.
pub(crate) const CONTENT_MODIFIED: i64 = -32801;

/// Write one message with its `Content-Length` header.
pub(crate) async fn write_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    message: &Value,
) -> io::Result<()> {
    let body = serde_json::to_vec(message)?;
    writer
        .write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
        .await?;
    writer.write_all(&body).await?;
    writer.flush().await
}

/// Read one message. Returns `None` at end of stream.
pub(crate) async fn read_message<R: AsyncBufRead + Unpin>(
    reader: &mut R,
) -> io::Result<Option<Value>> {
    let mut length = None;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(None);
        }
        let header = line.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some(value) = header.strip_prefix("Content-Length:") {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            length = Some(parsed);
        }
    }
    let length = length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "message without Content-Length")
    })?;
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await?;
    Ok(Some(serde_json::from_slice(&body)?))
}

/// Diagnostics published for one document revision.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct PublishedDiagnostics {
    pub uri: String,
    pub version: Option<i32>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct Diagnostic {
    pub range: Range,
    pub severity: Option<u32>,
    pub message: String,
}

/// An error returned by the server for one request.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RpcError {
    pub code: i64,
    pub message: String,
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) struct Position {
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct Range {
    pub start: Position,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Location {
    pub uri: String,
    pub range: Range,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocationLink {
    target_uri: String,
    target_selection_range: Range,
}

/// Location of a workspace symbol. The range is optional in `WorkspaceSymbol`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SymbolLocation {
    pub uri: String,
    pub range: Option<Range>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WorkspaceSymbol {
    pub name: String,
    pub kind: u32,
    pub location: SymbolLocation,
    pub container_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DocumentSymbol {
    pub name: String,
    pub kind: u32,
    pub detail: Option<String>,
    pub selection_range: Range,
    #[serde(default)]
    pub children: Vec<DocumentSymbol>,
}

/// Normalize a `Location | Location[] | LocationLink[] | null` response.
pub(crate) fn parse_locations(value: Value) -> Vec<Location> {
    let items = match value {
        Value::Null => Vec::new(),
        Value::Array(items) => items,
        other => vec![other],
    };
    items
        .into_iter()
        .filter_map(|item| {
            if let Ok(link) = LocationLink::deserialize(&item) {
                return Some(Location {
                    uri: link.target_uri,
                    range: link.target_selection_range,
                });
            }
            Location::deserialize(&item).ok()
        })
        .collect()
}

/// Pull the text out of a hover response (`MarkupContent`, `MarkedString` or an array).
pub(crate) fn hover_text(value: &Value) -> String {
    fn collect(contents: &Value, out: &mut Vec<String>) {
        match contents {
            Value::String(text) => out.push(text.clone()),
            Value::Array(items) => items.iter().for_each(|item| collect(item, out)),
            Value::Object(map) => {
                if let Some(Value::String(text)) = map.get("value") {
                    out.push(text.clone());
                }
            }
            _ => {}
        }
    }
    let mut parts = Vec::new();
    if let Some(contents) = value.get("contents") {
        collect(contents, &mut parts);
    }
    parts.join("\n\n").trim().to_string()
}

/// Display name for an LSP `SymbolKind` number.
pub(crate) fn symbol_kind_name(kind: u32) -> &'static str {
    const NAMES: [&str; 26] = [
        "file",
        "module",
        "namespace",
        "package",
        "class",
        "method",
        "property",
        "field",
        "constructor",
        "enum",
        "interface",
        "function",
        "variable",
        "constant",
        "string",
        "number",
        "boolean",
        "array",
        "object",
        "key",
        "null",
        "variant",
        "struct",
        "event",
        "operator",
        "type_param",
    ];
    kind.checked_sub(1)
        .and_then(|index| NAMES.get(index as usize))
        .copied()
        .unwrap_or("symbol")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::BufReader;

    #[tokio::test]
    async fn framing_round_trips() {
        let message = json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": "é"}});
        let mut buffer = Vec::new();
        write_message(&mut buffer, &message)
            .await
            .expect("write to Vec");
        write_message(&mut buffer, &message)
            .await
            .expect("write to Vec");

        let mut reader = BufReader::new(buffer.as_slice());
        for _ in 0..2 {
            let read = read_message(&mut reader).await.expect("valid frame");
            assert_eq!(read, Some(message.clone()));
        }
        assert!(
            read_message(&mut reader)
                .await
                .expect("clean EOF")
                .is_none()
        );
    }

    #[test]
    fn locations_accept_links_and_single_location() {
        let link = json!([{
            "targetUri": "file:///a.rs",
            "targetRange": {"start": {"line": 0, "character": 0}, "end": {"line": 9, "character": 0}},
            "targetSelectionRange": {"start": {"line": 4, "character": 2}, "end": {"line": 4, "character": 5}}
        }]);
        let parsed = parse_locations(link);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].range.start.line, 4);

        let single = json!({
            "uri": "file:///b.rs",
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 3}}
        });
        assert_eq!(parse_locations(single)[0].uri, "file:///b.rs");
        assert!(parse_locations(Value::Null).is_empty());
    }

    #[test]
    fn hover_text_handles_markup_and_arrays() {
        let markup = json!({"contents": {"kind": "markdown", "value": "pub struct Foo"}});
        assert_eq!(hover_text(&markup), "pub struct Foo");
        let marked = json!({"contents": [{"language": "rust", "value": "fn a()"}, "docs"]});
        assert_eq!(hover_text(&marked), "fn a()\n\ndocs");
    }

    #[test]
    fn symbol_kinds_map_to_names() {
        assert_eq!(symbol_kind_name(23), "struct");
        assert_eq!(symbol_kind_name(12), "function");
        assert_eq!(symbol_kind_name(0), "symbol");
        assert_eq!(symbol_kind_name(99), "symbol");
    }
}
