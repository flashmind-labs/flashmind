//! Session persistence and picker.
//!
//! Wraps `flashmind_memory::SessionStore` and maps between
//! `ConversationEntry` and `SessionEntry` for SQLite storage.

use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{Frame, Terminal, TerminalOptions, Viewport, backend::CrosstermBackend};

use flashmind_core::{Conversation, ConversationEntry, EntryKind};
use flashmind_memory::rusqlite;
use flashmind_memory::session::schema::init_session_schema;
use flashmind_memory::session::{SessionEntry, SessionEntryKind, SessionStore};
use flashmind_memory::tokio_rusqlite;
use flashmind_types::{ContentPart, ToolCall};

// ---------------------------------------------------------------------------
// Sessions wrapper
// ---------------------------------------------------------------------------

/// High-level session manager backed by SQLite.
pub struct Sessions {
    store: SessionStore,
    conn: tokio_rusqlite::Connection,
}

impl Sessions {
    pub async fn connect(db_path: &Path) -> Result<Self> {
        let conn = tokio_rusqlite::Connection::open(db_path).await?;
        conn.call(|c| {
            init_session_schema(c)?;
            init_local_sessions_schema(c)?;
            Ok::<_, rusqlite::Error>(())
        })
        .await?;
        let store = SessionStore::new(conn.clone());
        Ok(Self { store, conn })
    }

    pub async fn save(&self, chat_key: &str, conversation: &Conversation) -> Result<()> {
        let entries: Vec<SessionEntry> = conversation
            .entries()
            .iter()
            .map(to_session_entry)
            .collect();
        self.store.delete_session(chat_key).await?;
        self.store.save_entries(&entries).await?;
        tracing::debug!("saved {} entries for {chat_key}", entries.len());
        Ok(())
    }

    pub async fn load(&self, chat_key: &str) -> Result<Option<Conversation>> {
        let entries = self.store.load(chat_key).await?;
        if entries.is_empty() {
            return Ok(None);
        }
        let mut conv = Conversation::new();
        for entry in &entries {
            if let Some(ce) = from_session_entry(entry) {
                conv.add(ce);
            }
        }
        Ok(Some(conv))
    }

    pub async fn delete(&self, chat_key: &str) -> Result<()> {
        self.store.delete_session(chat_key).await?;
        delete_local_session(&self.conn, chat_key).await?;
        Ok(())
    }

    pub async fn save_metadata(
        &self,
        key: &str,
        prompt: &str,
        model: &str,
        cwd: &str,
    ) -> Result<()> {
        save_local_session(&self.conn, key, prompt, model, cwd).await
    }

    pub async fn list_local(&self) -> Result<Vec<LocalSession>> {
        list_local_sessions(&self.conn).await
    }

    pub fn connection(&self) -> &tokio_rusqlite::Connection {
        &self.conn
    }
}

// ---------------------------------------------------------------------------
// ConversationEntry <-> SessionEntry mapping
// ---------------------------------------------------------------------------

fn to_session_entry(ce: &ConversationEntry) -> SessionEntry {
    let created_at = ce.timestamp.timestamp();
    let (entry_kind, content, tool_calls, tool_call_id, tool_name, metadata) = match &ce.kind {
        EntryKind::SystemPrompt(s) => (
            SessionEntryKind::SystemPrompt,
            s.clone(),
            None,
            None,
            None,
            None,
        ),
        EntryKind::Developer {
            content,
            tag,
            metadata,
        } => (
            SessionEntryKind::Developer { tag: tag.clone() },
            content.clone(),
            None,
            None,
            None,
            metadata.as_ref().and_then(|m| serde_json::to_value(m).ok()),
        ),
        EntryKind::User { content, parts } => (
            SessionEntryKind::User,
            content.clone(),
            None,
            None,
            None,
            parts.as_ref().and_then(|p| serde_json::to_value(p).ok()),
        ),
        EntryKind::Assistant {
            content,
            tool_calls,
        } => (
            SessionEntryKind::Assistant,
            content.clone(),
            tool_calls
                .as_ref()
                .and_then(|tc| serde_json::to_value(tc).ok()),
            None,
            None,
            None,
        ),
        EntryKind::Tool { call_id, output } => (
            SessionEntryKind::Tool,
            output.clone(),
            None,
            Some(call_id.clone()),
            None,
            None,
        ),
    };

    SessionEntry {
        id: 0,
        chat_key: String::new(),
        entry_kind,
        content,
        tool_calls,
        tool_call_id,
        tool_name,
        metadata,
        turn_index: 0,
        created_at,
    }
}

fn from_session_entry(se: &SessionEntry) -> Option<ConversationEntry> {
    let kind = match &se.entry_kind {
        SessionEntryKind::SystemPrompt => EntryKind::SystemPrompt(se.content.clone()),
        SessionEntryKind::Developer { tag } => EntryKind::Developer {
            content: se.content.clone(),
            tag: tag.clone(),
            metadata: se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_value(m.clone()).ok()),
        },
        SessionEntryKind::User => {
            let parts: Option<Vec<ContentPart>> = se
                .metadata
                .as_ref()
                .and_then(|m| serde_json::from_value(m.clone()).ok());
            EntryKind::User {
                content: se.content.clone(),
                parts,
            }
        }
        SessionEntryKind::Assistant => {
            let tool_calls: Option<Vec<ToolCall>> = se
                .tool_calls
                .as_ref()
                .and_then(|tc| serde_json::from_value(tc.clone()).ok());
            EntryKind::Assistant {
                content: se.content.clone(),
                tool_calls,
            }
        }
        SessionEntryKind::Tool => EntryKind::Tool {
            call_id: se.tool_call_id.clone().unwrap_or_default(),
            output: se.content.clone(),
        },
    };

    Some(ConversationEntry {
        kind,
        timestamp: chrono::DateTime::from_timestamp(se.created_at, 0).unwrap_or_else(Utc::now),
    })
}

// ---------------------------------------------------------------------------
// Local session metadata table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LocalSession {
    pub key: String,
    pub prompt: String,
    pub model: String,
    pub cwd: Option<String>,
    pub updated_at: i64,
    pub title: Option<String>,
}

fn init_local_sessions_schema(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS local_sessions (
            key        TEXT PRIMARY KEY,
            prompt     TEXT NOT NULL DEFAULT '',
            model      TEXT NOT NULL DEFAULT '',
            cwd        TEXT NOT NULL DEFAULT '',
            updated_at INTEGER NOT NULL DEFAULT 0,
            title      TEXT
        );",
    )?;
    // Migration: add title column if it doesn't exist
    let _ = conn.execute_batch("ALTER TABLE local_sessions ADD COLUMN title TEXT;");
    Ok(())
}

async fn save_local_session(
    conn: &tokio_rusqlite::Connection,
    key: &str,
    prompt: &str,
    model: &str,
    cwd: &str,
) -> Result<()> {
    let key = key.to_string();
    let prompt = prompt.chars().take(200).collect::<String>();
    let model = model.to_string();
    let cwd = cwd.to_string();
    let now = Utc::now().timestamp();

    conn.call(move |c| {
        c.execute(
            "INSERT INTO local_sessions (key, prompt, model, cwd, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(key) DO UPDATE SET prompt = ?2, model = ?3, cwd = ?4, updated_at = ?5",
            rusqlite::params![key, prompt, model, cwd, now],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

async fn delete_local_session(conn: &tokio_rusqlite::Connection, key: &str) -> Result<()> {
    let key = key.to_string();
    conn.call(move |c| {
        c.execute(
            "DELETE FROM local_sessions WHERE key = ?1",
            rusqlite::params![key],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

async fn list_local_sessions(conn: &tokio_rusqlite::Connection) -> Result<Vec<LocalSession>> {
    let rows = conn
        .call(|c| {
            let mut stmt = c.prepare(
                "SELECT key, prompt, model, cwd, updated_at, title
                 FROM local_sessions
                 ORDER BY updated_at DESC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    let cwd: String = row.get(3)?;
                    Ok(LocalSession {
                        key: row.get(0)?,
                        prompt: row.get(1)?,
                        model: row.get(2)?,
                        cwd: if cwd.is_empty() { None } else { Some(cwd) },
                        updated_at: row.get(4)?,
                        title: row.get(5)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok::<_, rusqlite::Error>(rows)
        })
        .await?;
    Ok(rows)
}

pub async fn set_title(
    conn: &tokio_rusqlite::Connection,
    key: &str,
    title: Option<&str>,
) -> Result<()> {
    let key = key.to_string();
    let title = title.map(|s| s.to_string());
    conn.call(move |c| {
        c.execute(
            "UPDATE local_sessions SET title = ?2 WHERE key = ?1",
            rusqlite::params![key, title],
        )?;
        Ok::<_, rusqlite::Error>(())
    })
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session picker
// ---------------------------------------------------------------------------

/// Result of the session picker.
pub enum SessionPick {
    /// Resume an existing session.
    Resume(String),
    /// Start a new session with the given key.
    New(String),
}

/// Render a single row in the session picker.
fn render_row(entry: Option<&LocalSession>, is_selected: bool, max_width: u16) -> Line<'static> {
    let arrow = if is_selected { " > " } else { "   " };
    let key_style = Style::default().fg(if is_selected {
        Color::Green
    } else {
        Color::DarkGray
    });
    let prompt_style = Style::default().fg(if is_selected {
        Color::White
    } else {
        Color::Gray
    });

    let mut spans: Vec<Span> = vec![Span::styled(arrow, key_style)];

    let raw_prompt;
    if let Some(e) = entry {
        spans.push(Span::styled(format!("{} ", e.key), key_style));

        if let Some(ref cwd) = e.cwd {
            let path = Path::new(cwd);
            let parts: Vec<_> = path.iter().collect();
            let short = if parts.len() >= 2 {
                format!(
                    "({}/{})",
                    parts[parts.len() - 2].to_string_lossy(),
                    parts.last().unwrap().to_string_lossy()
                )
            } else {
                format!("({})", parts[0].to_string_lossy())
            };
            spans.push(Span::styled(
                format!("{short}  "),
                Style::default().fg(Color::DarkGray),
            ));
        } else {
            spans.push(Span::styled(" ", key_style));
        }

        raw_prompt = if let Some(ref title) = e.title {
            title.clone()
        } else {
            e.prompt.split_whitespace().collect::<Vec<_>>().join(" ")
        };
    } else {
        spans.push(Span::styled(
            "new session  ",
            key_style.add_modifier(Modifier::BOLD),
        ));
        raw_prompt = "New session".to_string();
    }

    let prefix_width: usize = spans.iter().map(|s| s.width()).sum();
    let max_prompt = (max_width as usize).saturating_sub(prefix_width + 1);
    let prompt_display = if raw_prompt.len() > max_prompt {
        format!(
            "{}\u{2026}",
            truncate_utf8(&raw_prompt, max_prompt.saturating_sub(1))
        )
    } else {
        raw_prompt
    };
    spans.push(Span::styled(prompt_display, prompt_style));

    Line::from(spans)
}

fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}

#[derive(PartialEq, Eq)]
enum PickerMode {
    Navigate,
    Search,
}

struct PickerState {
    entries: Vec<LocalSession>,
    query: String,
    filtered: Vec<usize>,
    selected: usize,
    confirm_delete: bool,
    confirm_yes: bool,
    mode: PickerMode,
    deleted_keys: Vec<String>,
}

impl PickerState {
    fn new(entries: Vec<LocalSession>) -> Self {
        let mut state = Self {
            entries,
            query: String::new(),
            filtered: Vec::new(),
            selected: 0,
            confirm_delete: false,
            confirm_yes: true,
            mode: PickerMode::Navigate,
            deleted_keys: Vec::new(),
        };
        state.recompute_filter();
        if !state.filtered.is_empty() {
            state.selected = 1;
        }
        state
    }

    fn recompute_filter(&mut self) {
        let q = self.query.to_lowercase();
        let mut indices: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                if q.is_empty() {
                    return true;
                }
                let e = &self.entries[i];
                e.key.to_lowercase().contains(&q)
                    || e.prompt.to_lowercase().contains(&q)
                    || e.cwd
                        .as_ref()
                        .is_some_and(|c| c.to_lowercase().contains(&q))
            })
            .collect();

        indices.sort_by_key(|&i| {
            -(self.entries[i]
                .updated_at
                .max(self.entries[i].key.parse::<i64>().unwrap_or(0)))
        });
        self.filtered = indices;

        let total = self.total();
        if self.selected >= total {
            self.selected = total.saturating_sub(1);
        }
    }

    fn total(&self) -> usize {
        self.filtered.len() + 1
    }

    fn selected_entry_idx(&self) -> Option<usize> {
        if self.selected == 0 {
            None
        } else {
            self.filtered.get(self.selected - 1).copied()
        }
    }

    fn select_current(&self) -> Option<SessionPick> {
        if self.selected == 0 {
            return Some(SessionPick::New(new_session_key()));
        }
        self.selected_entry_idx()
            .map(|idx| SessionPick::Resume(self.entries[idx].key.clone()))
    }

    fn handle_key(&mut self, key: event::KeyEvent) -> Option<SessionPick> {
        if self.confirm_delete {
            self.handle_delete_key(key)
        } else if self.mode == PickerMode::Search {
            self.handle_search_key(key)
        } else {
            self.handle_normal_key(key)
        }
    }

    fn handle_normal_key(&mut self, key: event::KeyEvent) -> Option<SessionPick> {
        match key.code {
            KeyCode::Char('/') => {
                self.mode = PickerMode::Search;
            }
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') if self.selected + 1 < self.total() => {
                self.selected += 1;
            }
            KeyCode::Char('d')
                if self.selected_entry_idx().is_some()
                    && !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.confirm_delete = true;
                self.confirm_yes = true;
            }
            KeyCode::Enter => return self.select_current(),
            KeyCode::Esc | KeyCode::Char('q') => {
                return Some(SessionPick::New(new_session_key()));
            }
            KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Some(SessionPick::New(new_session_key()));
            }
            _ => {}
        }
        None
    }

    fn handle_search_key(&mut self, key: event::KeyEvent) -> Option<SessionPick> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        match key.code {
            KeyCode::Char('c' | 'd') if ctrl => {
                return Some(SessionPick::New(new_session_key()));
            }
            KeyCode::Esc => self.mode = PickerMode::Navigate,
            KeyCode::Char('u') if ctrl && !self.query.is_empty() => {
                self.query.clear();
                self.recompute_filter();
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down if self.selected + 1 < self.total() => self.selected += 1,
            KeyCode::Enter => return self.select_current(),
            KeyCode::Backspace if self.query.pop().is_some() => {
                self.recompute_filter();
            }
            KeyCode::Char(c) if !ctrl && !alt => {
                self.query.push(c);
                self.recompute_filter();
                if self.selected == 0 && !self.filtered.is_empty() {
                    self.selected = 1;
                }
            }
            _ => {}
        }
        None
    }

    fn handle_delete_key(&mut self, key: event::KeyEvent) -> Option<SessionPick> {
        match key.code {
            KeyCode::Left | KeyCode::Right | KeyCode::Char('h') | KeyCode::Char('l') => {
                self.confirm_yes = !self.confirm_yes;
            }
            KeyCode::Char('y') => {
                self.confirm_yes = true;
                self.delete_selected();
                self.confirm_delete = false;
            }
            KeyCode::Char('n') | KeyCode::Esc => self.confirm_delete = false,
            KeyCode::Enter => {
                if self.confirm_yes {
                    self.delete_selected();
                }
                self.confirm_delete = false;
            }
            KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Some(SessionPick::New(new_session_key()));
            }
            _ => {}
        }
        None
    }

    fn delete_selected(&mut self) {
        let Some(entry_idx) = self.selected_entry_idx() else {
            return;
        };
        let key = self.entries[entry_idx].key.clone();
        self.deleted_keys.push(key);
        self.entries.remove(entry_idx);
        self.recompute_filter();
    }

    fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // title
                Constraint::Length(1), // search bar
                Constraint::Min(1),    // list
                Constraint::Length(1), // footer
            ])
            .split(area);

        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " Select a session to resume",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))),
            chunks[0],
        );

        // Search bar
        let in_search = self.mode == PickerMode::Search && !self.confirm_delete;
        let prompt_prefix = " / ";
        let search_line = Line::from(vec![
            Span::styled(
                prompt_prefix,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(self.query.clone(), Style::default().fg(Color::White)),
        ]);
        frame.render_widget(Paragraph::new(search_line), chunks[1]);
        if in_search {
            let cursor_x = chunks[1].x
                + prompt_prefix.chars().count() as u16
                + self.query.chars().count() as u16;
            frame.set_cursor_position((cursor_x, chunks[1].y));
        }

        // List
        let visible = chunks[2].height as usize;
        let total = self.total();
        let scroll = self.selected.saturating_sub(visible.saturating_sub(1));

        let lines: Vec<Line> = (scroll..total)
            .take(visible)
            .map(|i| {
                let entry = if i == 0 {
                    None
                } else {
                    self.filtered.get(i - 1).map(|&idx| &self.entries[idx])
                };
                render_row(entry, i == self.selected, area.width)
            })
            .collect();

        frame.render_widget(Paragraph::new(lines), chunks[2]);

        // Footer
        let footer_line = if self.confirm_delete {
            let dim = Style::default().fg(Color::DarkGray);
            let yes_style = if self.confirm_yes {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                dim
            };
            let no_style = if !self.confirm_yes {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                dim
            };
            Line::from(vec![
                Span::styled(" Delete this session? ", Style::default().fg(Color::Red)),
                Span::styled(
                    if self.confirm_yes { "[Yes]" } else { " Yes " },
                    yes_style,
                ),
                Span::styled("  ", dim),
                Span::styled(
                    if !self.confirm_yes { "[No]" } else { " No " },
                    no_style,
                ),
            ])
        } else {
            let counts = if self.query.is_empty() {
                format!("{} sessions", self.entries.len())
            } else {
                format!("{}/{} matches", self.filtered.len(), self.entries.len())
            };
            let hints = if in_search {
                format!(
                    " typing filters \u{00b7} \u{2191}/\u{2193} move \u{00b7} Enter select \u{00b7} ^U clear \u{00b7} Esc done \u{00b7} {}",
                    counts
                )
            } else {
                format!(
                    " \u{2191}/\u{2193}/j/k move \u{00b7} Enter select \u{00b7} / search \u{00b7} d delete \u{00b7} Esc new \u{00b7} {}",
                    counts
                )
            };
            Line::from(Span::styled(hints, Style::default().fg(Color::DarkGray)))
        };
        frame.render_widget(Paragraph::new(footer_line), chunks[3]);
    }
}

fn setup_terminal() -> anyhow::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;
    let (_, height) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
    Ok(Terminal::with_options(
        CrosstermBackend::new(std::io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(height),
        },
    )?)
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) {
    let _ = disable_raw_mode();
    let _ = terminal.clear();
}

/// Interactive session picker using ratatui inline viewport.
/// Returns the pick and any session keys deleted during the picker.
pub fn pick_session(sessions: &[LocalSession]) -> std::io::Result<(SessionPick, Vec<String>)> {
    if sessions.is_empty() {
        return Ok((SessionPick::New(new_session_key()), Vec::new()));
    }

    let Ok(mut terminal) = setup_terminal() else {
        return Ok((SessionPick::New(new_session_key()), Vec::new()));
    };

    let mut state = PickerState::new(sessions.to_vec());

    let result = loop {
        terminal.draw(|frame| state.draw(frame)).ok();

        if let Ok(Event::Key(key)) = event::read()
            && let Some(pick) = state.handle_key(key)
        {
            break pick;
        }
    };

    let deleted = std::mem::take(&mut state.deleted_keys);
    teardown_terminal(&mut terminal);

    Ok((result, deleted))
}

pub fn new_session_key() -> String {
    format!("{}", Utc::now().timestamp())
}
