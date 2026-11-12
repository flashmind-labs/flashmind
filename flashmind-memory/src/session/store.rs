//! JSONL conversation storage.
use super::types::{SessionEntry, SessionEntryKind};
use anyhow::{Context, Result, ensure};
use flashmind_types::{Model, ReasoningLevel, SessionUsage, TokenUsage};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub use super::types::SessionSummary;

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
struct Metadata {
    title: Option<String>,
    model: Option<String>,
    working_dir: Option<String>,
    total_cost: Option<String>,
    last_usage: Option<String>,
    total_usage: Option<SessionUsage>,
    reasoning: Option<ReasoningLevel>,
}
#[derive(Clone, Default, PartialEq)]
struct Session {
    entries: Vec<SessionEntry>,
    meta: Metadata,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Record {
    Header {
        version: u32,
        chat_key: String,
        metadata: Metadata,
    },
    Entry {
        entry: SessionEntry,
    },
}

// The envelope follows the message-oriented structure used by Claude Code.
// Message fields remain provider independent and preserve the session IR.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageRecord {
    #[serde(rename = "type")]
    kind: String,
    uuid: uuid::Uuid,
    parent_uuid: Option<uuid::Uuid>,
    session_id: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    message: StoredMessage,
    entry_id: i64,
    turn_index: i64,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionRecord {
    #[serde(rename = "type")]
    kind: String,
    subtype: String,
    version: u32,
    session_id: String,
    metadata: Metadata,
}
impl MessageRecord {
    fn from_entry(entry: &SessionEntry, parent_uuid: Option<uuid::Uuid>) -> Result<Self> {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(entry)?);
        if let Some(parent) = parent_uuid {
            hash.update(parent.as_bytes());
        }
        let digest = hash.finalize();
        let mut bytes = [0; 16];
        bytes.copy_from_slice(&digest[..16]);
        // Set RFC 9562 version 8 and variant bits for a content-derived UUID.
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let (role, tag) = match &entry.entry_kind {
            SessionEntryKind::SystemPrompt => ("system", None),
            SessionEntryKind::Developer { tag } => ("developer", tag.clone()),
            SessionEntryKind::User => ("user", None),
            SessionEntryKind::Assistant => ("assistant", None),
            SessionEntryKind::Tool => ("tool", None),
        };
        Ok(Self {
            kind: role.into(),
            uuid: uuid::Uuid::from_bytes(bytes),
            parent_uuid,
            session_id: entry.chat_key.clone(),
            timestamp: chrono::DateTime::from_timestamp(entry.created_at, 0)
                .context("invalid entry timestamp")?,
            entry_id: entry.id,
            turn_index: entry.turn_index,
            message: StoredMessage {
                role: role.into(),
                content: entry.content.clone(),
                tag,
                tool_calls: entry.tool_calls.clone(),
                tool_call_id: entry.tool_call_id.clone(),
                tool_name: entry.tool_name.clone(),
                metadata: entry.metadata.clone(),
            },
        })
    }
    fn into_entry(self) -> Result<SessionEntry> {
        ensure!(
            self.kind == self.message.role,
            "message role does not match record type"
        );
        let entry_kind = match self.kind.as_str() {
            "system" => SessionEntryKind::SystemPrompt,
            "developer" => SessionEntryKind::Developer {
                tag: self.message.tag,
            },
            "user" => SessionEntryKind::User,
            "assistant" => SessionEntryKind::Assistant,
            "tool" => SessionEntryKind::Tool,
            other => anyhow::bail!("unsupported message type {other}"),
        };
        Ok(SessionEntry {
            id: self.entry_id,
            chat_key: self.session_id,
            entry_kind,
            content: self.message.content,
            tool_calls: self.message.tool_calls,
            tool_call_id: self.message.tool_call_id,
            tool_name: self.message.tool_name,
            metadata: self.message.metadata,
            turn_index: self.turn_index,
            created_at: self.timestamp.timestamp(),
        })
    }
}

#[derive(Clone)]
enum Backend {
    Files(PathBuf),
    Memory(Arc<Mutex<BTreeMap<String, Session>>>),
}
/// Session entries and metadata persisted as one JSONL file per conversation.
#[derive(Clone)]
pub struct SessionStore {
    backend: Backend,
}

impl SessionStore {
    /// Import a legacy database once, keeping the source database unchanged.
    #[cfg(feature = "session-sqlite-import")]
    pub async fn import_sqlite(&self, path: &Path) -> Result<()> {
        let Backend::Files(dir) = &self.backend else {
            anyhow::bail!("SQLite import requires a persistent store");
        };
        let dir = dir.clone();
        let lock = tokio::task::spawn_blocking(move || -> Result<File> {
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(dir.join(".import.lock"))?;
            file.lock()?;
            Ok(file)
        })
        .await??;
        let Backend::Files(dir) = &self.backend else {
            unreachable!()
        };
        let marker = dir.join(".sqlite-imported");
        if marker.exists() || !path.exists() {
            return Ok(());
        }
        let source = path.to_owned();
        let connection = tokio_rusqlite::Connection::open_in_memory().await?;
        connection
            .call(move |target| {
                let source = rusqlite::Connection::open_with_flags(
                    source,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
                )?;
                {
                    let backup = rusqlite::backup::Backup::new(&source, target)?;
                    backup.run_to_completion(100, std::time::Duration::from_millis(1), None)?;
                }
                super::schema::init_session_schema(target)?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;
        let legacy = super::sqlite::SqliteSessionStore::new(connection);
        let mut imported = BTreeMap::new();
        for summary in legacy.list_sessions().await? {
            let key = summary.chat_key;
            let reasoning = legacy.reasoning(&key).await?;
            let entries = legacy.load(&key).await?;
            imported.insert(
                key,
                Session {
                    entries,
                    meta: Metadata {
                        title: summary.title,
                        model: summary.model,
                        working_dir: summary.working_dir,
                        total_cost: summary.total_cost,
                        last_usage: summary.last_usage,
                        total_usage: summary.total_usage,
                        reasoning,
                    },
                },
            );
        }
        self.access(true, move |data| {
            for (key, session) in imported {
                data.entry(key).or_insert(session);
            }
            Ok(())
        })
        .await?;
        tokio::task::spawn_blocking(move || -> Result<()> {
            File::create(marker)?.sync_all()?;
            drop(lock);
            Ok(())
        })
        .await??;
        Ok(())
    }
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(directory.as_ref())?;
        Ok(Self {
            backend: Backend::Files(directory.as_ref().to_owned()),
        })
    }
    pub fn in_memory() -> Self {
        Self {
            backend: Backend::Memory(Arc::default()),
        }
    }
    async fn access<T: Send + 'static>(
        &self,
        write: bool,
        action: impl FnOnce(&mut BTreeMap<String, Session>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let backend = self.backend.clone();
        tokio::task::spawn_blocking(move || match backend {
            Backend::Memory(data) => action(
                &mut *data
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session lock poisoned"))?,
            ),
            Backend::Files(dir) => {
                let lock = OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join(".lock"))?;
                lock.lock()?;
                let mut sessions = read_sessions(&dir)?;
                let before = if write { Some(sessions.clone()) } else { None };
                let result = action(&mut sessions)?;
                if let Some(before) = before {
                    for (key, session) in &sessions {
                        if before.get(key) != Some(session) {
                            write_session(&dir, key, session)?;
                        }
                    }
                    for key in before.keys().filter(|key| !sessions.contains_key(*key)) {
                        fs::remove_file(session_path(&dir, key))?;
                    }
                    #[cfg(unix)]
                    File::open(&dir)?.sync_all()?;
                }
                Ok(result)
            }
        })
        .await?
    }
    pub async fn save_entry(&self, entry: &SessionEntry) -> Result<i64> {
        let entry = entry.clone();
        self.access(true, move |data| {
            let session = data.entry(entry.chat_key.clone()).or_default();
            let mut entry = entry;
            entry.id = session
                .entries
                .iter()
                .map(|e| e.id)
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .context("entry ID overflow")?;
            let id = entry.id;
            session.entries.push(entry);
            Ok(id)
        })
        .await
    }
    pub async fn save_entries(&self, entries: &[SessionEntry]) -> Result<()> {
        let entries = entries.to_vec();
        self.access(true, move |data| {
            for mut entry in entries {
                let session = data.entry(entry.chat_key.clone()).or_default();
                entry.id = session
                    .entries
                    .iter()
                    .map(|e| e.id)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .context("entry ID overflow")?;
                session.entries.push(entry);
            }
            Ok(())
        })
        .await
    }
    pub async fn load(&self, key: &str) -> Result<Vec<SessionEntry>> {
        let key = key.to_owned();
        self.access(false, move |data| {
            let mut entries = data
                .get(&key)
                .map(|s| s.entries.clone())
                .unwrap_or_default();
            entries.sort_by_key(|e| e.turn_index);
            Ok(entries)
        })
        .await
    }
    pub async fn rewrite(&self, key: &str, entries: &[SessionEntry]) -> Result<()> {
        ensure!(
            entries.iter().all(|e| e.chat_key == key),
            "entry belongs to another session"
        );
        let key = key.to_owned();
        let entries = entries.to_vec();
        self.access(true, move |data| {
            data.entry(key).or_default().entries = entries;
            Ok(())
        })
        .await
    }
    pub async fn delete_session(&self, key: &str) -> Result<u64> {
        let key = key.to_owned();
        self.access(true, move |data| {
            let s = data.entry(key).or_default();
            let count = s.entries.len() as u64;
            s.entries.clear();
            Ok(count)
        })
        .await
    }
    pub async fn delete_sessions_by_prefix(&self, prefix: &str) -> Result<u64> {
        let prefix = prefix.to_owned();
        self.access(true, move |data| {
            let mut count = 0;
            for (key, s) in data {
                if key.starts_with(&prefix) {
                    count += s.entries.len() as u64;
                    s.entries.clear();
                }
            }
            Ok(count)
        })
        .await
    }
    pub async fn branch(&self, from: &str, to: &str) -> Result<u64> {
        let from = from.to_owned();
        let to = to.to_owned();
        self.access(true, move |data| {
            let mut entries = data
                .get(&from)
                .map(|s| s.entries.clone())
                .unwrap_or_default();
            entries.sort_by_key(|e| e.turn_index);
            for (i, e) in entries.iter_mut().enumerate() {
                e.chat_key = to.clone();
                e.turn_index = i64::try_from(i)?;
            }
            let count = entries.len() as u64;
            data.entry(to).or_default().entries.extend(entries);
            Ok(count)
        })
        .await
    }
    pub async fn prune(&self, days: u32) -> Result<u64> {
        let cutoff = chrono::Utc::now().timestamp() - i64::from(days) * 86400;
        self.access(true, move |data| {
            let mut count = 0;
            for s in data.values_mut() {
                if s.entries
                    .iter()
                    .map(|e| e.created_at)
                    .max()
                    .is_some_and(|ts| ts < cutoff)
                {
                    count += s.entries.len() as u64;
                    s.entries.clear();
                }
            }
            Ok(count)
        })
        .await
    }
    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        self.access(false, |data| {
            let mut result = Vec::new();
            for (key, s) in data.iter() {
                if s.entries.is_empty() {
                    continue;
                }
                let mut entries = s.entries.iter().collect::<Vec<_>>();
                entries.sort_by_key(|e| e.turn_index);
                result.push(SessionSummary {
                    chat_key: key.clone(),
                    entry_count: i64::try_from(entries.len())?,
                    last_updated: entries.iter().map(|e| e.created_at).max().unwrap_or(0),
                    title: s.meta.title.clone(),
                    model: s.meta.model.clone(),
                    working_dir: s.meta.working_dir.clone(),
                    total_cost: s.meta.total_cost.clone(),
                    last_usage: s.meta.last_usage.clone(),
                    total_usage: s.meta.total_usage.clone(),
                    first_message: entries
                        .iter()
                        .find(|e| matches!(e.entry_kind, SessionEntryKind::User))
                        .map(|e| e.content.clone()),
                    last_message: entries
                        .iter()
                        .rev()
                        .find(|e| {
                            matches!(
                                e.entry_kind,
                                SessionEntryKind::User | SessionEntryKind::Assistant
                            ) && !e.content.is_empty()
                        })
                        .map(|e| e.content.clone()),
                });
            }
            result.sort_by(|a, b| {
                b.last_updated
                    .cmp(&a.last_updated)
                    .then(a.chat_key.cmp(&b.chat_key))
            });
            Ok(result)
        })
        .await
    }
    async fn meta(
        &self,
        key: &str,
        action: impl FnOnce(&mut Metadata) + Send + 'static,
    ) -> Result<()> {
        let key = key.to_owned();
        self.access(true, move |data| {
            action(&mut data.entry(key).or_default().meta);
            Ok(())
        })
        .await
    }
    pub async fn save_meta(
        &self,
        key: &str,
        title: Option<&str>,
        model: Option<&str>,
        working_dir: Option<&str>,
    ) -> Result<()> {
        let (title, model, working_dir) = (
            title.map(str::to_owned),
            model.map(str::to_owned),
            working_dir.map(str::to_owned),
        );
        self.meta(key, move |m| {
            m.title = title;
            m.model = model;
            m.working_dir = working_dir;
        })
        .await
    }
    pub async fn update_reasoning(&self, key: &str, level: ReasoningLevel) -> Result<()> {
        self.meta(key, move |m| m.reasoning = Some(level)).await
    }
    pub async fn reasoning(&self, key: &str) -> Result<Option<ReasoningLevel>> {
        let key = key.to_owned();
        self.access(false, move |data| {
            Ok(data.get(&key).and_then(|s| s.meta.reasoning))
        })
        .await
    }
    pub async fn update_model(&self, key: &str, model: &Model) -> Result<()> {
        let model = model.to_string();
        self.meta(key, move |m| m.model = Some(model)).await
    }
    pub async fn update_working_dir(&self, key: &str, path: &Path) -> Result<()> {
        let path = path
            .to_str()
            .context("session directory is not UTF-8")?
            .to_owned();
        self.meta(key, move |m| m.working_dir = Some(path)).await
    }
    pub async fn update_title(&self, key: &str, title: &str) -> Result<()> {
        let title = title.to_owned();
        self.meta(key, move |m| m.title = Some(title)).await
    }
    pub async fn update_cost_usage(&self, key: &str, cost: &str, usage: &str) -> Result<()> {
        let (cost, usage) = (cost.to_owned(), usage.to_owned());
        self.meta(key, move |m| {
            m.total_cost = Some(cost);
            m.last_usage = Some(usage);
        })
        .await
    }
    pub async fn update_session_usage(
        &self,
        key: &str,
        total: &SessionUsage,
        last: Option<&TokenUsage>,
    ) -> Result<()> {
        let total = total.clone();
        let last = last.map(serde_json::to_string).transpose()?;
        self.meta(key, move |m| {
            m.total_cost = Some(total.total_cost.to_string());
            m.total_usage = Some(total);
            m.last_usage = last;
        })
        .await
    }
    pub async fn delete_meta(&self, key: &str) -> Result<()> {
        self.meta(key, |m| *m = Metadata::default()).await
    }
}
fn session_path(dir: &Path, key: &str) -> PathBuf {
    use sha2::{Digest, Sha256};
    let name: String = Sha256::digest(key.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    dir.join(format!("{name}.jsonl"))
}
fn read_sessions(dir: &Path) -> Result<BTreeMap<String, Session>> {
    let mut data = BTreeMap::new();
    for file in fs::read_dir(dir)? {
        let path = file?.path();
        if path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let mut lines = text.lines();
        let header = lines.next().context("missing session header")?;
        let value: serde_json::Value = serde_json::from_str(header)?;
        let (chat_key, metadata, legacy) = if value.get("record").is_some() {
            let Record::Header {
                version,
                chat_key,
                metadata,
            } = serde_json::from_value(value)?
            else {
                anyhow::bail!("missing session header");
            };
            ensure!(version == 1, "unsupported session version {version}");
            (chat_key, metadata, true)
        } else {
            let header: SessionRecord = serde_json::from_value(value)?;
            ensure!(
                header.kind == "system" && header.subtype == "session_metadata",
                "missing session metadata"
            );
            ensure!(
                header.version == 2,
                "unsupported session version {}",
                header.version
            );
            (header.session_id, header.metadata, false)
        };
        ensure!(
            session_path(dir, &chat_key) == path,
            "session filename does not match key"
        );
        let mut entries = Vec::new();
        let mut parent = None;
        let mut ids = std::collections::HashSet::new();
        for line in lines {
            let entry = if legacy {
                let Record::Entry { entry } = serde_json::from_str(line)? else {
                    anyhow::bail!("unexpected session header");
                };
                entry
            } else {
                let record: MessageRecord = serde_json::from_str(line)
                    .with_context(|| format!("invalid session record in {}", path.display()))?;
                ensure!(record.parent_uuid == parent, "broken session message chain");
                ensure!(ids.insert(record.uuid), "duplicate session message UUID");
                parent = Some(record.uuid);
                record.into_entry()?
            };
            ensure!(
                entry.chat_key == chat_key,
                "entry belongs to another session"
            );
            entries.push(entry);
        }
        data.insert(
            chat_key,
            Session {
                entries,
                meta: metadata,
            },
        );
    }
    Ok(data)
}
fn write_session(dir: &Path, key: &str, session: &Session) -> Result<()> {
    let path = session_path(dir, key);
    let temporary = dir.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        write_session_records(&mut file, key, session)?;
        file.sync_all()?;
        fs::rename(&temporary, &path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn write_session_records(writer: impl Write, key: &str, session: &Session) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer(
        &mut writer,
        &SessionRecord {
            kind: "system".into(),
            subtype: "session_metadata".into(),
            version: 2,
            session_id: key.to_owned(),
            metadata: session.meta.clone(),
        },
    )?;
    writer.write_all(b"\n")?;
    let mut parent = None;
    for entry in &session.entries {
        let record = MessageRecord::from_entry(entry, parent)?;
        parent = Some(record.uuid);
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(key: &str, content: &str, turn_index: i64) -> SessionEntry {
        SessionEntry {
            id: 0,
            chat_key: key.into(),
            entry_kind: SessionEntryKind::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            metadata: Some(serde_json::json!({"parts": [{"type":"text","text":"example"}]})),
            turn_index,
            created_at: 100,
        }
    }
    #[derive(Default)]
    struct CountingWriter {
        bytes: Vec<u8>,
        writes: usize,
    }

    impl Write for CountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn multiline_session() -> Session {
        let text = "example \"quoted\" line\n".repeat(64);
        Session {
            entries: (0..1000)
                .map(|index| entry("example", &text, index))
                .collect(),
            ..Session::default()
        }
    }

    #[test]
    fn session_serialization_batches_small_writes() {
        let session = multiline_session();
        let mut writer = CountingWriter::default();
        write_session_records(&mut writer, "example", &session).unwrap();
        assert!(
            writer.writes < session.entries.len(),
            "{} writes for {} records",
            writer.writes,
            session.entries.len()
        );
        let lines: Vec<_> = writer
            .bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(lines.len(), session.entries.len() + 1);
        for (line, entry) in lines[1..].iter().zip(&session.entries) {
            let record: MessageRecord = serde_json::from_slice(line).unwrap();
            assert_eq!(record.message.content, entry.content);
        }
    }

    #[test]
    fn large_multiline_sessions_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let session = multiline_session();
        write_session(directory.path(), "example", &session).unwrap();
        assert_eq!(
            read_sessions(directory.path()).unwrap()["example"].entries,
            session.entries
        );
    }

    #[test]
    fn session_serialization_propagates_buffer_flush_errors() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("write failed"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(write_session_records(FailingWriter, "example", &Session::default()).is_err());
    }

    #[tokio::test]
    async fn message_envelopes_preserve_fields_and_stable_parent_links() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        let mut assistant = entry("example", "answer", 1);
        assistant.entry_kind = SessionEntryKind::Assistant;
        assistant.tool_calls =
            Some(serde_json::json!([{ "id": "call", "name": "example", "arguments": {} }]));
        let entries = [entry("example", "question", 0), assistant];
        store.rewrite("example", &entries).await.unwrap();
        let path = session_path(dir.path(), "example");
        let text = fs::read_to_string(&path).unwrap();
        let records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records[0]["type"], "system");
        assert_eq!(records[1]["type"], "user");
        assert_eq!(records[1]["sessionId"], "example");
        assert!(records[1]["parentUuid"].is_null());
        assert_eq!(records[2]["parentUuid"], records[1]["uuid"]);
        assert_eq!(records[2]["message"]["role"], "assistant");
        assert_eq!(records[2]["message"]["content"], "answer");
        assert!(records[2]["timestamp"].as_str().unwrap().ends_with('Z'));
        assert_eq!(store.load("example").await.unwrap(), entries);
        store.update_title("example", "Title").await.unwrap();
        let updated = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().nth(1), updated.lines().nth(1));
        // A branch into an existing conversation can contain identical entries.
        store.branch("example", "example").await.unwrap();
        assert_eq!(store.load("example").await.unwrap().len(), 4);
    }
    #[tokio::test]
    async fn reads_previous_jsonl_format_and_upgrades_on_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = session_path(dir.path(), "example");
        let old = format!(
            "{}\n{}\n",
            serde_json::to_string(&Record::Header {
                version: 1,
                chat_key: "example".into(),
                metadata: Metadata::default()
            })
            .unwrap(),
            serde_json::to_string(&Record::Entry {
                entry: entry("example", "saved", 0)
            })
            .unwrap()
        );
        fs::write(&path, old).unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        assert_eq!(store.load("example").await.unwrap()[0].content, "saved");
        store.update_title("example", "Updated").await.unwrap();
        let text = fs::read_to_string(path).unwrap();
        let header: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(header["version"], 2);
        assert_eq!(store.load("example").await.unwrap()[0].content, "saved");
    }
    #[tokio::test]
    async fn rejects_broken_parent_chain_without_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        store
            .rewrite(
                "example",
                &[entry("example", "one", 0), entry("example", "two", 1)],
            )
            .await
            .unwrap();
        let path = session_path(dir.path(), "example");
        let text = fs::read_to_string(&path).unwrap();
        let mut records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        records[2]["parentUuid"] = serde_json::Value::Null;
        let broken = records.iter().map(|r| format!("{r}\n")).collect::<String>();
        fs::write(&path, &broken).unwrap();
        assert!(store.update_title("example", "Title").await.is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), broken);
    }
    #[tokio::test]
    async fn round_trip_rewrite_branch_and_prune() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        let key = "../".repeat(100);
        store
            .rewrite(&key, &[entry(&key, "hello\nworld", 0)])
            .await
            .unwrap();
        store
            .save_meta(
                &key,
                Some("Title"),
                Some("provider:model"),
                Some("/path/to/work"),
            )
            .await
            .unwrap();
        store
            .update_reasoning(&key, ReasoningLevel::High)
            .await
            .unwrap();
        let usage = SessionUsage {
            input_tokens: 123,
            ..SessionUsage::default()
        };
        store
            .update_session_usage(&key, &usage, None)
            .await
            .unwrap();
        let reopened = SessionStore::open(dir.path()).unwrap();
        assert_eq!(
            reopened.load(&key).await.unwrap()[0].content,
            "hello\nworld"
        );
        let summary = &reopened.list_sessions().await.unwrap()[0];
        assert_eq!(summary.total_usage.as_ref(), Some(&usage));
        assert_eq!(summary.title.as_deref(), Some("Title"));
        assert_eq!(
            reopened.reasoning(&key).await.unwrap(),
            Some(ReasoningLevel::High)
        );
        reopened
            .rewrite(&key, &[entry(&key, "changed", 0), entry(&key, "second", 1)])
            .await
            .unwrap();
        reopened
            .rewrite(&key, &[entry(&key, "shortened", 0)])
            .await
            .unwrap();
        assert_eq!(reopened.branch(&key, "branch").await.unwrap(), 1);
        assert_eq!(reopened.load("branch").await.unwrap()[0].chat_key, "branch");
        assert_eq!(reopened.prune(1).await.unwrap(), 2);
        assert!(reopened.list_sessions().await.unwrap().is_empty());
    }
    #[tokio::test]
    async fn independent_handles_do_not_lose_updates() {
        let dir = tempfile::tempdir().unwrap();
        let a = SessionStore::open(dir.path()).unwrap();
        let b = SessionStore::open(dir.path()).unwrap();
        let one = entry("same", "one", 0);
        let two = entry("same", "two", 1);
        let (a, b) = tokio::join!(a.save_entry(&one), b.save_entry(&two));
        a.unwrap();
        b.unwrap();
        assert_eq!(
            SessionStore::open(dir.path())
                .unwrap()
                .load("same")
                .await
                .unwrap()
                .len(),
            2
        );
    }
    #[tokio::test]
    async fn malformed_record_is_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        store
            .rewrite("key", &[entry("key", "saved", 0)])
            .await
            .unwrap();
        let path = session_path(dir.path(), "key");
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{broken");
        fs::write(&path, &bytes).unwrap();
        assert!(store.rewrite("key", &[]).await.is_err());
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
    #[cfg(feature = "session-sqlite-import")]
    #[tokio::test]
    async fn imports_old_schema_without_changing_source_or_resurrecting_deleted_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("sessions.db");
        {
            let conn = rusqlite::Connection::open(&source).unwrap();
            conn.execute_batch("CREATE TABLE sessions (id INTEGER PRIMARY KEY,chat_key TEXT,entry_kind TEXT,content TEXT,tool_calls TEXT,tool_call_id TEXT,tool_name TEXT,metadata TEXT,turn_index INTEGER,created_at INTEGER);
          CREATE TABLE session_meta(chat_key TEXT PRIMARY KEY,title TEXT,model TEXT,created_at INTEGER);
          INSERT INTO sessions VALUES (1,'old','{\"type\":\"user\"}','question',NULL,NULL,NULL,NULL,0,100);
          INSERT INTO session_meta VALUES ('old','Title','provider:model',100);").unwrap();
        }
        let original = fs::read(&source).unwrap();
        let store = SessionStore::open(dir.path()).unwrap();
        store.import_sqlite(&source).await.unwrap();
        assert_eq!(store.load("old").await.unwrap()[0].content, "question");
        assert_eq!(
            store.list_sessions().await.unwrap()[0].title.as_deref(),
            Some("Title")
        );
        assert_eq!(fs::read(&source).unwrap(), original);
        store.delete_session("old").await.unwrap();
        store.delete_meta("old").await.unwrap();
        let reopened = SessionStore::open(dir.path()).unwrap();
        reopened.import_sqlite(&source).await.unwrap();
        assert!(reopened.load("old").await.unwrap().is_empty());
    }
}
