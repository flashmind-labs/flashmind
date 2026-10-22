use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, Utc};
use flashmind_memory::session::{SessionEntry, SessionEntryKind, SessionStore};
use flashmind_types::{ContentPart, SessionId, ToolCall};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Record {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "parentId")]
    parent_id: Option<String>,
    timestamp: DateTime<Utc>,
    #[serde(flatten)]
    kind: Kind,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Kind {
    Session {
        #[serde(default = "legacy_version")]
        version: u32,
        cwd: PathBuf,
    },
    Message {
        message: PiMessage,
    },
    Compaction {
        summary: String,
        #[serde(rename = "firstKeptEntryId")]
        first_kept: String,
    },
    BranchSummary {
        summary: String,
    },
    CustomMessage {
        content: Value,
    },
    ContextEdit {
        #[serde(rename = "targetId")]
        target: String,
        replacement: Value,
    },
    SessionInfo {
        name: Option<String>,
    },
    #[serde(other)]
    Other,
}

fn legacy_version() -> u32 {
    1
}

#[derive(Deserialize)]
struct PiMessage {
    role: PiRole,
    #[serde(default)]
    content: Value,
    #[serde(rename = "toolCallId")]
    tool_call_id: Option<String>,
    #[serde(rename = "toolName")]
    tool_name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
enum PiRole {
    User,
    Assistant,
    ToolResult,
    System,
    Custom,
    HookMessage,
    CompactionSummary,
    BranchSummary,
}

struct ImportedSession {
    key: SessionId,
    cwd: PathBuf,
    title: Option<String>,
    entries: Vec<SessionEntry>,
}

pub async fn run(path: Option<PathBuf>) -> Result<()> {
    let root = match path {
        Some(path) => path,
        None => std::env::var_os("PI_CODING_AGENT_DIR")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".pi/agent")))
            .context("cannot locate pi's agent directory")?
            .join("sessions"),
    };
    let mut files = Vec::new();
    collect_files(&root, &mut files)?;
    files.sort();
    let mut imported = 0;
    let mut skipped = 0;
    let mut failed = 0;
    for path in files {
        let result = async {
            let input = fs::read_to_string(&path)?;
            let session = parse(&input)?;
            let store = crate::session::open_session_store_for(&session.cwd).await?;
            persist(&store, &session).await
        }
        .await;
        match result {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(error) => {
                eprintln!("{}: {error:#}", path.display());
                failed += 1;
            }
        }
    }
    println!(
        "Imported {imported} sessions, skipped {skipped}, failed {failed}. Resume with `flsh resume`."
    );
    ensure!(failed == 0, "some pi sessions could not be imported");
    Ok(())
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("cannot read {}", path.display()))? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_files(&entry.path(), files)?;
        } else if kind.is_file() && entry.path().extension().is_some_and(|ext| ext == "jsonl") {
            files.push(entry.path());
        }
    }
    Ok(())
}

async fn persist(store: &SessionStore, session: &ImportedSession) -> Result<bool> {
    if !store.load(session.key.as_str()).await?.is_empty() || session.entries.is_empty() {
        return Ok(false);
    }
    store
        .save_meta(
            session.key.as_str(),
            session.title.as_deref(),
            None,
            session.cwd.to_str(),
        )
        .await?;
    store.save_entries(&session.entries).await?;
    Ok(true)
}

fn parse(input: &str) -> Result<ImportedSession> {
    let mut records = Vec::new();
    for (index, line) in input.lines().enumerate() {
        if !line.trim().is_empty() {
            records.push(
                serde_json::from_str::<Record>(line)
                    .with_context(|| format!("invalid pi record on line {}", index + 1))?,
            );
        }
    }
    let header = records.first().context("empty pi session")?;
    let Kind::Session { version, cwd } = &header.kind else {
        bail!("missing pi session header");
    };
    ensure!(
        (1..=3).contains(version),
        "unsupported pi session version {version}"
    );
    ensure!(cwd.is_absolute(), "pi working directory must be absolute");
    let key = format!(
        "pi_{}",
        header.id.as_deref().context("missing pi session ID")?
    )
    .parse::<SessionId>()?;
    let mut path = Vec::new();
    if *version == 1 {
        path.extend(records.iter().skip(1));
    } else {
        let mut index = HashMap::new();
        for record in records.iter().skip(1) {
            let id = record.id.as_deref().context("missing pi entry ID")?;
            ensure!(
                index.insert(id, record).is_none(),
                "duplicate pi entry ID {id}"
            );
        }
        let mut current = records.last().filter(|_| records.len() > 1);
        let mut visited = HashSet::new();
        while let Some(record) = current {
            ensure!(
                visited.insert(record.id.as_deref()),
                "cycle in pi session tree"
            );
            path.push(record);
            current = match record.parent_id.as_deref() {
                Some(parent) => Some(*index.get(parent).context("missing pi parent entry")?),
                None => None,
            };
        }
        path.reverse();
    }
    let title = records
        .iter()
        .rev()
        .find_map(|record| match &record.kind {
            Kind::SessionInfo { name } => Some(name.clone()),
            _ => None,
        })
        .flatten();
    let edits: HashMap<&str, &Value> = path
        .iter()
        .filter_map(|record| match &record.kind {
            Kind::ContextEdit {
                target,
                replacement,
            } => Some((target.as_str(), replacement)),
            _ => None,
        })
        .collect();
    if let Some(position) = path
        .iter()
        .rposition(|record| matches!(record.kind, Kind::Compaction { .. }))
    {
        let compaction = path[position];
        if let Kind::Compaction { first_kept, .. } = &compaction.kind {
            let kept = if compaction.id.as_deref() == Some(first_kept) {
                position
            } else {
                path[..position]
                    .iter()
                    .position(|record| record.id.as_deref() == Some(first_kept))
                    .context("missing first kept entry for pi compaction")?
            };
            path = std::iter::once(compaction)
                .chain(path[kept..position].iter().copied())
                .chain(path[position + 1..].iter().copied())
                .collect();
        }
    }
    let mut entries = Vec::new();
    for record in path {
        let replacement = record.id.as_deref().and_then(|id| edits.get(id).copied());
        if replacement.is_some_and(Value::is_null) {
            continue;
        }
        if let Some(mut entry) = convert(record, replacement, key.as_str())? {
            entry.turn_index = entries.len() as i64;
            entries.push(entry);
        }
    }
    Ok(ImportedSession {
        key,
        cwd: cwd.clone(),
        title,
        entries,
    })
}

fn text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| match part.get("type")?.as_str()? {
                "text" => part.get("text")?.as_str().map(String::from),
                "image" => Some("[Image]".to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn convert(
    record: &Record,
    replacement: Option<&Value>,
    key: &str,
) -> Result<Option<SessionEntry>> {
    let mut entry = SessionEntry {
        id: 0,
        chat_key: key.to_string(),
        entry_kind: SessionEntryKind::User,
        content: String::new(),
        tool_calls: None,
        tool_call_id: None,
        tool_name: None,
        metadata: None,
        turn_index: 0,
        created_at: record.timestamp.timestamp(),
    };
    match &record.kind {
        Kind::Message { message } => {
            let content = replacement.unwrap_or(&message.content);
            entry.content = text(content);
            match message.role {
                PiRole::System => return Ok(None),
                PiRole::User => {
                    let parts: Vec<ContentPart> = content
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|part| part["type"] == "image")
                        .map(|part| {
                            Ok(ContentPart::Image {
                                media_type: part["mimeType"]
                                    .as_str()
                                    .context("missing image MIME type")?
                                    .to_string(),
                                data: part["data"]
                                    .as_str()
                                    .context("missing image data")?
                                    .to_string(),
                            })
                        })
                        .collect::<Result<_>>()?;
                    if !parts.is_empty() {
                        entry.metadata = Some(json!({ "parts": parts }));
                    }
                }
                PiRole::Assistant => {
                    entry.entry_kind = SessionEntryKind::Assistant;
                    let mut calls = Vec::new();
                    let mut thinking = Vec::new();
                    for part in content.as_array().into_iter().flatten() {
                        match part["type"].as_str() {
                            Some("toolCall") => calls.push(ToolCall {
                                id: part["id"]
                                    .as_str()
                                    .context("missing tool call ID")?
                                    .to_string(),
                                name: part["name"]
                                    .as_str()
                                    .context("missing tool name")?
                                    .to_string(),
                                arguments: part
                                    .get("arguments")
                                    .context("missing tool arguments")?
                                    .clone(),
                            }),
                            Some("thinking") => {
                                if let Some(value) = part["thinking"].as_str() {
                                    thinking.push(value);
                                }
                            }
                            _ => {}
                        }
                    }
                    if !calls.is_empty() {
                        entry.tool_calls = Some(serde_json::to_value(calls)?);
                    }
                    if !thinking.is_empty() {
                        entry.metadata = Some(json!({ "reasoning": thinking.join("\n") }));
                    }
                }
                PiRole::ToolResult => {
                    entry.entry_kind = SessionEntryKind::Tool;
                    entry.tool_call_id = Some(
                        message
                            .tool_call_id
                            .clone()
                            .context("missing tool result call ID")?,
                    );
                    entry.tool_name = message.tool_name.clone();
                }
                PiRole::Custom
                | PiRole::HookMessage
                | PiRole::CompactionSummary
                | PiRole::BranchSummary => {}
            }
        }
        Kind::Compaction { summary, .. } | Kind::BranchSummary { summary } => {
            entry.content = format!("[Previous session summary]\n{summary}");
        }
        Kind::CustomMessage { content } => entry.content = text(replacement.unwrap_or(content)),
        _ => return Ok(None),
    }
    Ok(Some(entry))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(mut records: Vec<Value>) -> String {
        records.insert(
            0,
            json!({"type":"session", "version":3, "id":"example", "cwd":"/path/to/work"}),
        );
        records
            .into_iter()
            .map(|mut record| {
                record["timestamp"] = json!("2026-01-01T00:00:00Z");
                record.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn selects_active_branch_and_keeps_tool_calls() {
        let input = session(vec![
            json!({"type":"message", "id":"a", "parentId":null, "message":{"role":"user","content":"hello"}}),
            json!({"type":"message", "id":"discard", "parentId":"a", "message":{"role":"assistant","content":[{"type":"text","text":"abandoned"}]}}),
            json!({"type":"message", "id":"b", "parentId":"a", "message":{"role":"assistant","content":[{"type":"thinking","thinking":"plan"},{"type":"toolCall","id":"call-1","name":"read","arguments":{"path":"README.md"}}]}}),
            json!({"type":"message", "id":"c", "parentId":"b", "message":{"role":"toolResult","toolCallId":"call-1","toolName":"read","content":[{"type":"text","text":"file content"}]}}),
            json!({"type":"session_info","id":"d","parentId":"c","name":"Example"}),
        ]);
        let parsed = parse(&input).unwrap();
        assert_eq!(parsed.entries.len(), 3);
        assert_eq!(parsed.title.as_deref(), Some("Example"));
        assert_eq!(
            parsed.entries[1].tool_calls.as_ref().unwrap()[0]["id"],
            "call-1"
        );
        assert_eq!(
            parsed.entries[1].metadata.as_ref().unwrap()["reasoning"],
            "plan"
        );
        assert_eq!(parsed.entries[2].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(parsed.entries[2].created_at, 1767225600);
    }

    #[test]
    fn compaction_and_edits_preserve_current_context() {
        let input = session(vec![
            json!({"type":"message","id":"a","parentId":null,"message":{"role":"user","content":"old"}}),
            json!({"type":"message","id":"b","parentId":"a","message":{"role":"user","content":"keep"}}),
            json!({"type":"compaction","id":"c","parentId":"b","summary":"summary","firstKeptEntryId":"b"}),
            json!({"type":"context_edit","id":"d","parentId":"c","targetId":"b","replacement":"edited"}),
            json!({"type":"custom_message","id":"e","parentId":"d","content":"custom context"}),
            json!({"type":"context_edit","id":"f","parentId":"e","targetId":"e","replacement":null}),
        ]);
        let parsed = parse(&input).unwrap();
        assert_eq!(parsed.entries.len(), 2);
        assert!(parsed.entries[0].content.contains("summary"));
        assert_eq!(parsed.entries[1].content, "edited");
    }

    #[tokio::test]
    async fn import_round_trip_and_repeat_does_not_overwrite() {
        let parsed = parse(&session(vec![
            json!({"type":"message","id":"a","parentId":null,"message":{"role":"user","content":[{"type":"text","text":"photo"},{"type":"image","mimeType":"image/png","data":"example"}]}}),
        ])).unwrap();
        let conn = tokio_rusqlite::Connection::open_in_memory().await.unwrap();
        conn.call(|conn| {
            flashmind_memory::session::schema::init_session_schema(conn)?;
            Ok::<_, flashmind_memory::rusqlite::Error>(())
        })
        .await
        .unwrap();
        let store = SessionStore::new(conn);
        assert!(persist(&store, &parsed).await.unwrap());
        store
            .update_title(parsed.key.as_str(), "Changed in flsh")
            .await
            .unwrap();
        assert!(!persist(&store, &parsed).await.unwrap());
        assert_eq!(
            store.list_sessions().await.unwrap()[0].title.as_deref(),
            Some("Changed in flsh")
        );
        let saved = store.load(parsed.key.as_str()).await.unwrap();
        let conv = crate::session::session_to_conv(&saved[0]);
        let again = crate::session::conv_to_session(&conv, parsed.key.as_str(), 0);
        assert_eq!(saved[0].metadata, again.metadata);
    }

    #[test]
    fn rejects_broken_trees_and_json() {
        assert!(parse("not JSON").is_err());
        assert!(parse(&session(vec![json!({"type":"message","id":"a","parentId":"missing","message":{"role":"user","content":"hello"}})])).is_err());
        assert!(parse(&session(vec![json!({"type":"message","id":"a","parentId":"a","message":{"role":"user","content":"hello"}})])).is_err());
    }

    #[test]
    fn imports_legacy_linear_sessions() {
        let input = session(vec![
            json!({"type":"message","message":{"role":"user","content":"legacy"}}),
        ])
        .replace("\"version\":3", "\"version\":1");
        assert_eq!(parse(&input).unwrap().entries[0].content, "legacy");
    }
}
