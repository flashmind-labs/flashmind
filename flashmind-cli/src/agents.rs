//! Custom named agents.
//!
//! An agent is a saved persona — a name, a system prompt, and an optional model
//! and reasoning level. Agents are persisted one-per-file as JSON under
//! `{config}/agents/` (see [`crate::paths`]) and can be invoked from the REPL with `/{name}`, which
//! persistently swaps the active session's system prompt (and model/reasoning).
//! Markdown personas are imported through [`flashmind_tools::persona`]; saved
//! ones override them.
//!
//! Creation goes through a guided flow ([`create_agent_flow`]) where pressing
//! Tab twice while drafting the system prompt runs a one-off "prompt enricher"
//! LLM call ([`enrich_prompt`]) that rewrites the draft into a sharper prompt.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use flashmind_core::Agent;
use flashmind_tools::persona::{
    self, MarkdownPersonas, Persona, PersonaSource, delegate_instructions,
};
use flashmind_tui::Tui;
use flashmind_tui::styles::{S_AGENT, S_DIM};
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker};
use flashmind_types::{CompletionRequest, LlmProvider, Message, Model, ReasoningLevel};

use crate::config::Config;
use crate::interactive::{ModelChoice, apply_model, complete_text, slash_commands};
use crate::paths::config_dir;
use crate::setup::run_choice;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// The result of switching to an agent — what the caller must reflect in the
/// status bar / loop state.
pub struct AppliedAgent {
    /// Set when the agent specified a model and it was swapped in.
    pub model: Option<ModelChoice>,
    /// Set when the agent specified a reasoning level.
    pub reasoning: Option<ReasoningLevel>,
}

// ---------------------------------------------------------------------------
// Disk store
// ---------------------------------------------------------------------------

/// Directory holding agent JSON files (`{config}/agents/`).
pub fn agents_dir() -> Result<PathBuf> {
    let dir = config_dir()?.join("agents");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Turn an agent name into a safe file stem (lowercase, alnum/`-`/`_` only).
fn slugify(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

/// Slash command names, which personas may not use.
fn reserved_names() -> Vec<String> {
    slash_commands()
        .iter()
        .map(|c| c.name.trim_start_matches('/').to_owned())
        .collect()
}

/// Imported Markdown personas plus saved JSON personas (one file per agent),
/// with saved ones taking precedence.
#[derive(Clone, Debug)]
pub struct AgentStore {
    dir: PathBuf,
    markdown: MarkdownPersonas,
}

impl AgentStore {
    /// Open the store rooted at `{config}/agents/`.
    pub fn open() -> Result<Self> {
        Ok(Self::new(
            agents_dir()?,
            crate::paths::markdown_agent_dirs()?,
        ))
    }

    fn new(dir: PathBuf, markdown_dirs: Vec<PathBuf>) -> Self {
        Self {
            dir,
            markdown: MarkdownPersonas::new(markdown_dirs).with_reserved_names(reserved_names()),
        }
    }

    /// Saved and imported agents, sorted by name. Invalid files are skipped.
    pub fn list(&self) -> Vec<Persona> {
        // Explicitly saved Flashmind personas override imported definitions.
        persona::merge(self.markdown.load().into_iter().chain(self.saved()))
    }

    fn saved(&self) -> Vec<Persona> {
        let mut defs: Vec<Persona> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter_map(|t| serde_json::from_str(&t).ok())
            .collect();
        defs.sort_by_key(|d| d.name.to_lowercase());
        defs
    }

    pub(crate) fn instructions(&self, can_delegate: bool) -> String {
        if !can_delegate {
            return String::new();
        }
        delegate_instructions(&self.list())
    }

    /// Persist an agent, overwriting any existing file with the same name.
    pub fn save(&self, def: &Persona) -> Result<()> {
        let path = self.dir.join(format!("{}.json", slugify(&def.name)));
        let json = serde_json::to_string_pretty(def)?;
        std::fs::write(&path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Delete an agent by name. Returns `true` if a file was removed.
    pub fn delete(&self, name: &str) -> Result<bool> {
        let Some(def) = self
            .saved()
            .into_iter()
            .find(|def| def.name.eq_ignore_ascii_case(name))
        else {
            return Ok(false);
        };
        let path = self.dir.join(format!("{}.json", slugify(&def.name)));
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl PersonaSource for AgentStore {
    fn personas(&self) -> Vec<Persona> {
        self.list()
    }
}

/// Validate a candidate agent name, returning an error message if invalid.
fn validate_name(name: &str) -> std::result::Result<(), String> {
    persona::validate_name(name, &reserved_names()).map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------------
// Switching to an agent
// ---------------------------------------------------------------------------

/// Switch the active session to `def`'s model and reasoning level when the
/// agent specifies them. The caller sets the system prompt. Returns what
/// changed so the caller can update its status bar.
pub async fn apply_agent(
    agent: &mut Agent,
    config: &Config,
    def: &Persona,
    tui: &mut Tui,
) -> Result<AppliedAgent> {
    for notice in &def.notices {
        tui.println(&ratatui::text::Line::from(format!("  {notice}")))?;
    }
    let model = match &def.model {
        Some(m) => Some(apply_model(agent, m, config).await?),
        None => None,
    };
    if let Some(level) = def.reasoning {
        agent.llm_mut().reasoning = level;
    }

    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Agent: {}", def.name),
        S_AGENT,
    )))?;

    Ok(AppliedAgent {
        model,
        reasoning: def.reasoning,
    })
}

// ---------------------------------------------------------------------------
// Guided creation flow
// ---------------------------------------------------------------------------

/// Run the guided agent-creation flow. `name`/`initial_prompt` pre-fill the
/// respective steps (e.g. from `/agents-create name draft...`). Returns the
/// saved agent, or `None` if the user cancelled.
pub async fn create_agent_flow(
    tui: &mut Tui,
    agent: &mut Agent,
    store: &AgentStore,
    name: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<Option<Persona>> {
    // 1. Name
    let name = match name.map(str::trim).filter(|s| !s.is_empty()) {
        Some(n) => n.to_string(),
        None => {
            let mut picker = ChoicePicker::new(
                "Agent name".into(),
                vec![ChoiceOption {
                    label: "name".into(),
                    accepts_input: true,
                }],
            );
            match run_choice(tui, &mut picker)? {
                Some(resp) if !resp.input.trim().is_empty() => resp.input.trim().to_string(),
                _ => {
                    tui.println(&ratatui::text::Line::from("  Cancelled."))?;
                    return Ok(None);
                }
            }
        }
    };
    if let Err(e) = validate_name(&name) {
        tui.println(&ratatui::text::Line::from(format!("  Invalid name: {e}")))?;
        return Ok(None);
    }

    // 2. System prompt — custom editor with Tab-Tab enrichment.
    let mut editor = PromptEditor::new(
        format!("System prompt for '{name}'"),
        initial_prompt.unwrap_or("").to_string(),
    );
    let system_prompt = loop {
        match run_prompt_editor(tui, &mut editor)? {
            Some(PromptAction::Submit) => {
                if editor.buffer.trim().is_empty() {
                    tui.println(&ratatui::text::Line::from(
                        "  System prompt cannot be empty. Cancelled.",
                    ))?;
                    return Ok(None);
                }
                break editor.buffer.trim().to_string();
            }
            Some(PromptAction::Enrich) => {
                if editor.buffer.trim().is_empty() {
                    continue;
                }
                tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                    "  Enriching prompt…",
                    S_DIM,
                )))?;
                match enrich_prompt(agent.provider_arc(), &agent.llm().model, &editor.buffer).await
                {
                    Ok(better) => editor.buffer = better,
                    Err(e) => {
                        tui.println(&ratatui::text::Line::from(format!("  Enrich failed: {e}")))?;
                    }
                }
            }
            None => {
                tui.println(&ratatui::text::Line::from("  Cancelled."))?;
                return Ok(None);
            }
        }
    };

    // 3. Snapshot the current model + reasoning so the agent is reproducible.
    let def = Persona {
        name: name.clone(),
        description: None,
        notices: Vec::new(),
        system_prompt,
        model: Some(agent.llm().model.to_string()),
        reasoning: Some(agent.llm().reasoning),
    };
    store.save(&def)?;

    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Created agent '{name}'. Invoke it with /{name}"),
        S_AGENT,
    )))?;
    Ok(Some(def))
}

// ---------------------------------------------------------------------------
// Prompt enricher
// ---------------------------------------------------------------------------

const ENRICH_SYSTEM: &str = "You are an expert prompt engineer. Rewrite the user's \
draft into a clear, specific, well-structured system prompt for an AI agent. \
Preserve the original intent, sharpen the role, add concrete guidance on tone and \
behavior, and keep it concise. Reply with ONLY the improved system prompt — no \
preamble, no commentary, no surrounding quotes.";

/// One-off LLM call that rewrites `draft` into a stronger system prompt.
pub async fn enrich_prompt(
    provider: &Arc<dyn LlmProvider>,
    model: &Model,
    draft: &str,
) -> Result<String> {
    let mut request = CompletionRequest::new(
        model.clone(),
        vec![
            Message::system(ENRICH_SYSTEM),
            Message::user(format!("Draft system prompt:\n\n{draft}")),
        ],
    );
    request.max_tokens = Some(1024);

    let out = complete_text(provider, request).await;
    if out.is_empty() {
        anyhow::bail!("enricher returned empty output");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Prompt editor widget
// ---------------------------------------------------------------------------

/// Action returned by the prompt editor's key loop.
enum PromptAction {
    /// Enter pressed — accept the current buffer.
    Submit,
    /// Tab pressed twice — enrich the current buffer.
    Enrich,
}

/// Minimal multi-line text editor used to draft a system prompt. Tracks a
/// "last key was Tab" flag so two consecutive Tabs trigger enrichment.
struct PromptEditor {
    title: String,
    buffer: String,
    last_was_tab: bool,
}

impl PromptEditor {
    fn new(title: String, buffer: String) -> Self {
        Self {
            title,
            buffer,
            last_was_tab: false,
        }
    }

    fn lines(&self) -> Vec<ratatui::text::Line<'static>> {
        use ratatui::text::{Line, Span};
        let mut lines = vec![Line::from(Span::styled(
            format!("  {}", self.title),
            S_AGENT,
        ))];
        if self.buffer.is_empty() {
            lines.push(Line::from(Span::styled("  (type a draft prompt)", S_DIM)));
        } else {
            for l in self.buffer.split('\n') {
                lines.push(Line::from(format!("  {l}")));
            }
        }
        lines.push(Line::from(Span::styled(
            "  Enter: save · Tab Tab: enrich · Esc: cancel",
            S_DIM,
        )));
        lines
    }

    /// Handle a key. Returns `Some(action)` when the loop should yield to the
    /// async caller (submit/enrich); `None` otherwise (kept editing).
    fn handle_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> Option<PromptAction> {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};

        if key.code == KeyCode::Tab {
            if self.last_was_tab {
                self.last_was_tab = false;
                return Some(PromptAction::Enrich);
            }
            self.last_was_tab = true;
            return None;
        }
        self.last_was_tab = false;

        match key.code {
            KeyCode::Enter => return Some(PromptAction::Submit),
            KeyCode::Char(c) => self.buffer.push(c),
            KeyCode::Backspace => {
                self.buffer.pop();
            }
            _ => {}
        }
        // Ctrl+J inserts a newline for multi-line prompts.
        if key.code == KeyCode::Char('j') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.buffer.pop(); // undo the 'j' pushed above
            self.buffer.push('\n');
        }
        None
    }
}

/// Draw/read loop for the prompt editor. Returns `Some(action)` on submit or
/// enrich (caller keeps the editor to continue), or `None` if cancelled (Esc /
/// Ctrl+D).
fn run_prompt_editor(
    tui: &mut Tui,
    editor: &mut PromptEditor,
) -> std::io::Result<Option<PromptAction>> {
    use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};

    let mut screen = tui.fullscreen()?;
    loop {
        let lines = editor.lines();
        let cursor = lines.len().saturating_sub(2);
        screen.draw_list(lines, cursor, 1, 1)?;
        if !event::poll(std::time::Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = screen.read_event()? else {
            continue;
        };
        if key.code == KeyCode::Esc
            || (key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL))
        {
            return Ok(None);
        }
        if let Some(action) = editor.handle_key(key) {
            return Ok(Some(action));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn temp_store() -> (AgentStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("flsh-agents-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        (AgentStore::new(dir.clone(), Vec::new()), dir)
    }

    fn def(name: &str) -> Persona {
        Persona {
            name: name.to_string(),
            description: None,
            notices: Vec::new(),
            system_prompt: format!("You are {name}."),
            model: Some("ollama:llama3.2".into()),
            reasoning: Some(ReasoningLevel::Off),
        }
    }

    #[test]
    fn store_round_trip() {
        let (store, dir) = temp_store();
        assert!(store.list().is_empty());
        assert!(store.instructions(true).is_empty());

        store.save(&def("Reviewer")).unwrap();
        store.save(&def("planner")).unwrap();

        let names: Vec<String> = store.list().into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["planner".to_string(), "Reviewer".to_string()]); // sorted

        // case-insensitive lookup
        let got = store.get("reviewer").expect("found");
        assert_eq!(got.system_prompt, "You are Reviewer.");
        assert_eq!(got.model.as_deref(), Some("ollama:llama3.2"));

        assert!(store.delete("REVIEWER").unwrap());
        assert!(store.get("reviewer").is_none());
        assert!(!store.delete("nope").unwrap());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_overwrites_same_name() {
        let (store, dir) = temp_store();
        store.save(&def("bot")).unwrap();
        let mut d = def("bot");
        d.system_prompt = "changed".into();
        store.save(&d).unwrap();
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.get("bot").unwrap().system_prompt, "changed");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn slugify_is_filesystem_safe() {
        assert_eq!(slugify("Code Reviewer"), "code-reviewer");
        assert_eq!(slugify("my_agent-2"), "my_agent-2");
        assert_eq!(slugify("a/b\\c"), "a-b-c");
    }

    #[test]
    fn validate_name_rules() {
        assert!(validate_name("reviewer").is_ok());
        assert!(validate_name("").is_err());
        assert!(validate_name("with space").is_err());
        assert!(validate_name("a/b").is_err());
        assert!(validate_name("dot.name").is_err());
        // reserved slash commands rejected (case-insensitive)
        assert!(validate_name("model").is_err());
        assert!(validate_name("Agents").is_err());
    }

    const PERSONA: &str = "---\nname: reviewer\ndescription: |\n  Review code carefully.\nmodel: inherit\n---\nYou are a meticulous reviewer.\n\nKeep feedback brief.\n";

    #[test]
    fn saved_personas_override_imports_and_preserve_sources() {
        let temp = tempfile::tempdir().unwrap();
        let user = temp.path().join("user/.claude/agents");
        let project = temp.path().join("project/.agents");
        let saved = temp.path().join("saved");
        for dir in [&user, &project, &saved] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(user.join("reviewer.md"), PERSONA).unwrap();
        let project_text = PERSONA.replace("meticulous reviewer", "project reviewer");
        std::fs::write(project.join("reviewer.md"), &project_text).unwrap();
        std::fs::write(
            project.join("model.md"),
            PERSONA.replace("name: reviewer", "name: model"),
        )
        .unwrap();
        let store = AgentStore::new(saved, vec![user, project.clone()]);
        assert_eq!(store.list().len(), 1);
        let instructions = store.instructions(true);
        assert!(instructions.contains("- reviewer: Review code carefully."));
        assert!(store.instructions(false).is_empty());
        let mut def = store.get("REVIEWER").unwrap();
        assert!(def.system_prompt.contains("project reviewer"));
        assert!(!store.delete("reviewer").unwrap());
        def.system_prompt = "Saved persona".into();
        store.save(&def).unwrap();
        assert_eq!(
            store.get("reviewer").unwrap().system_prompt,
            "Saved persona"
        );
        assert!(store.delete("reviewer").unwrap());
        assert!(
            store
                .get("reviewer")
                .unwrap()
                .system_prompt
                .contains("project reviewer")
        );
        assert_eq!(
            std::fs::read_to_string(project.join("reviewer.md")).unwrap(),
            project_text
        );
    }

    #[test]
    fn editor_tab_twice_enriches() {
        let mut ed = PromptEditor::new("t".into(), String::new());
        let tab = || KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        let ch = |c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);

        assert!(ed.handle_key(ch('h')).is_none());
        assert!(ed.handle_key(tab()).is_none()); // first Tab: armed
        assert!(matches!(ed.handle_key(tab()), Some(PromptAction::Enrich)));
        assert_eq!(ed.buffer, "h");

        // A non-Tab key between tabs resets the arm.
        assert!(ed.handle_key(tab()).is_none());
        assert!(ed.handle_key(ch('i')).is_none());
        assert!(ed.handle_key(tab()).is_none()); // armed again, not enrich
        assert_eq!(ed.buffer, "hi");

        assert!(matches!(
            ed.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PromptAction::Submit)
        ));
    }

    #[test]
    fn editor_ctrl_j_inserts_newline() {
        let mut ed = PromptEditor::new("t".into(), String::new());
        ed.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        ed.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        ed.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE));
        assert_eq!(ed.buffer, "a\nb");
    }
}
