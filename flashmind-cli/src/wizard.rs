//! LLM-assisted configuration editor with diff-based approval.
//!
//! Registers a custom `str_replace` tool that returns `ToolResult::Interrupt`
//! instead of writing directly. The wizard loop shows the diff to the user
//! and waits for y/n approval before applying each change.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ratatui::crossterm::event::{Event, KeyCode};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use flashmind_core::{Agent, Conversation, ConversationEntry};
use flashmind_types::tool::{Tool, ToolContext, ToolRegistry, ToolResult};
use flashmind_types::{AgentInput, LlmProvider};

use crate::config::Config;
use crate::display::DisplayLog;
use crate::tui::{TuiApp, TuiState};

// ---------------------------------------------------------------------------
// Sensitive key masking
// ---------------------------------------------------------------------------

const SENSITIVE_KEYS: &[&str] = &[
    "api_key",
    "bot_token",
    "app_token",
    "jwt_secret",
    "hmac_secret",
    "password",
    "secret",
    "token",
];

fn mask_credentials(config: &str) -> String {
    let mut out = String::with_capacity(config.len());
    for line in config.lines() {
        let trimmed = line.trim();
        if let Some((key, _)) = trimmed.split_once('=') {
            let key = key.trim().trim_start_matches('#').trim();
            if SENSITIVE_KEYS.contains(&key) {
                let eq_pos = line.find('=').unwrap();
                out.push_str(&line[..=eq_pos]);
                out.push_str(" \"***\"");
                out.push('\n');
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn restore_credentials(original: &str, new: &str) -> String {
    use std::collections::HashMap;

    let mut credentials: HashMap<String, String> = HashMap::new();
    let mut current_section = String::new();

    for line in original.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].to_string();
            continue;
        }
        if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim();
            if SENSITIVE_KEYS.contains(&key) {
                let value = value.trim();
                let value = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .unwrap_or(value);
                let full_key = if current_section.is_empty() {
                    key.to_string()
                } else {
                    format!("{}.{}", current_section, key)
                };
                credentials.insert(full_key, value.to_string());
            }
        }
    }

    let mut result = String::with_capacity(new.len());
    let mut current_section = String::new();

    for line in new.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].to_string();
            result.push_str(line);
            result.push('\n');
            continue;
        }
        if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim();
            if SENSITIVE_KEYS.contains(&key) && value.trim() == r#""***""# {
                let full_key = if current_section.is_empty() {
                    key.to_string()
                } else {
                    format!("{}.{}", current_section, key)
                };
                if let Some(original_value) = credentials.get(&full_key) {
                    let eq_pos = line.find('=').unwrap();
                    result.push_str(&line[..=eq_pos]);
                    result.push_str(&format!(" \"{}\"", original_value));
                    result.push('\n');
                    continue;
                }
            }
        }
        result.push_str(line);
        result.push('\n');
    }

    result
}

// ---------------------------------------------------------------------------
// SetupStrReplaceTool
// ---------------------------------------------------------------------------

struct SetupStrReplaceTool {
    original_config: String,
}

impl SetupStrReplaceTool {
    fn preview_replace(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
    ) -> std::result::Result<(String, String), String> {
        let current_content =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read file: {}", e))?;

        let Some(start_pos) = current_content.find(old_string) else {
            return Err("old_string not found in file".to_string());
        };

        if current_content[start_pos + 1..].find(old_string).is_some() {
            return Err(
                "old_string matches multiple times — include more context to make it unique"
                    .to_string(),
            );
        }

        let new_content = format!(
            "{}{}{}",
            &current_content[..start_pos],
            new_string,
            &current_content[start_pos + old_string.len()..]
        );

        let diff = similar::TextDiff::from_lines(&current_content, &new_content)
            .unified_diff()
            .context_radius(3)
            .to_string();

        Ok((new_content, diff))
    }
}

#[async_trait]
impl Tool for SetupStrReplaceTool {
    fn name(&self) -> &str {
        "str_replace"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file. Shows diff for user approval before applying. The old_string must match exactly once."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        #[derive(serde::Deserialize)]
        struct Args {
            path: String,
            old_string: String,
            new_string: String,
        }

        let args: Args = serde_json::from_value(ctx.args.clone())?;

        match self.preview_replace(&args.path, &args.old_string, &args.new_string) {
            Ok((new_content, diff)) => {
                let content_to_write = if new_content.contains("\"***\"") {
                    restore_credentials(&self.original_config, &new_content)
                } else {
                    new_content
                };

                let payload = json!({
                    "path": args.path,
                    "content": content_to_write,
                    "diff": diff,
                });

                Ok(ToolResult::interrupt(ctx.tool_call_id, payload.to_string()))
            }
            Err(e) => Ok(ToolResult::failure(ctx.tool_call_id, e)),
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let path = args["path"].as_str().unwrap_or("?");
        format!("edit {}", path)
    }
}

// ---------------------------------------------------------------------------
// Standalone entry point (flashmind-cli setup)
// ---------------------------------------------------------------------------

pub async fn run(model_override: Option<flashmind_types::model::Model>) -> Result<()> {
    let config = Config::load()?;
    Config::init()?;

    let llm_config = config.build_llm_config(model_override.as_ref())?;
    let provider = config.build_provider_for(&llm_config.model.provider)?;

    let mut app = TuiApp::new()?;
    let mut key_rx = crate::tui::spawn_key_reader();
    let mut display_log = DisplayLog::new();

    run_setup(
        provider,
        llm_config,
        &mut app,
        &mut key_rx,
        &mut display_log,
    )
    .await
}

// ---------------------------------------------------------------------------
// Setup wizard — runs inside an existing TUI (also called from /setup)
// ---------------------------------------------------------------------------

pub async fn run_setup(
    provider: Arc<dyn LlmProvider>,
    llm_config: flashmind_types::AgentLlmConfig,
    app: &mut TuiApp<'_>,
    key_rx: &mut mpsc::UnboundedReceiver<Event>,
    display_log: &mut DisplayLog,
) -> Result<()> {
    let config_path = Config::config_path();
    let config_path_str = config_path.display().to_string();
    let current_config = std::fs::read_to_string(&config_path).unwrap_or_default();

    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(SetupStrReplaceTool {
        original_config: current_config.clone(),
    }));

    let mut agent = Agent::builder(provider)
        .llm(llm_config)
        .tools(tools)
        .build();

    agent.refresh_features().await;

    let masked = if current_config.is_empty() {
        "(empty — not yet configured)".to_string()
    } else {
        mask_credentials(&current_config)
    };

    let system_prompt = build_setup_prompt(&config_path_str, &masked);

    let mut conversation = Conversation::new();
    conversation.prepend(ConversationEntry::system(system_prompt));

    app.add_system_message("Setup Assistant — type your message, /quit to exit");

    // Kick off the agent with initial prompt
    let initial = "Briefly introduce yourself and ask what I'd like to configure.";
    conversation.prepend(ConversationEntry::user(initial));

    let mut pending_stream = true;

    loop {
        if pending_stream {
            pending_stream = false;

            let interrupt = {
                let mut tui_state = TuiState::new();
                let stream = agent.start(&mut conversation, AgentInput::Resume, None);
                app.stream_response(Box::pin(stream), &mut tui_state, key_rx, |ev| {
                    display_log.log_agent_event(ev)
                })
                .await?
            };

            if let Some(int) = interrupt {
                let result = handle_interrupt(&int.output, app, key_rx).await;
                conversation.add(ConversationEntry::tool(&int.tool_call_id, &result));
                pending_stream = true;
                continue;
            }
        }

        // Wait for user input
        let action = app.read_input(key_rx).await?;
        match action {
            crate::tui::TuiAction::Submit(text) => {
                if text == "/quit" || text == "/exit" || text == "/q" {
                    break;
                }
                app.add_user_message(&text);
                display_log.log_user(text.clone());
                conversation.add(ConversationEntry::user(&text));
                pending_stream = true;
            }
            crate::tui::TuiAction::Quit => break,
            crate::tui::TuiAction::Cancel => {
                app.add_system_message("[press Ctrl+D or /quit to exit setup]");
            }
            crate::tui::TuiAction::None => {}
        }
    }

    app.add_system_message("Setup complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Interrupt handler — show diff, wait for y/n
// ---------------------------------------------------------------------------

async fn handle_interrupt(
    output: &str,
    app: &mut TuiApp<'_>,
    key_rx: &mut mpsc::UnboundedReceiver<Event>,
) -> String {
    let Ok(payload) = serde_json::from_str::<Value>(output) else {
        return "error: invalid interrupt payload".to_string();
    };

    let diff = payload["diff"].as_str().unwrap_or("");
    let path = payload["path"].as_str().unwrap_or("?");
    let content = payload["content"].as_str().unwrap_or("");

    app.add_system_message(&format!("Proposed changes to {}:\n{}", path, diff));
    app.add_system_message("Apply? (y/n)");
    let _ = app.draw(None);

    loop {
        let Some(ev) = key_rx.recv().await else {
            return "rejected: input closed".to_string();
        };
        if let Event::Key(key_event) = &ev {
            match key_event.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    if let Err(e) = std::fs::write(path, content) {
                        let msg = format!("failed to write: {}", e);
                        app.add_system_message(&msg);
                        return msg;
                    }
                    let msg = format!("Applied changes to {}", path);
                    app.add_system_message(&msg);
                    let _ = app.draw(None);
                    return format!("{}\n\n{}", msg, diff);
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    app.add_system_message("Changes rejected.");
                    let _ = app.draw(None);
                    return "User rejected the changes.".to_string();
                }
                _ => continue,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setup system prompt
// ---------------------------------------------------------------------------

fn build_setup_prompt(config_path: &str, current_config: &str) -> String {
    format!(
        r#"You are the Flash Setup Assistant. Help the user configure their `{config_path}` file.

Their current config:
```toml
{current_config}
```

## Config Reference

The config file is TOML with these sections:

### [llm] — Global LLM settings
- `temperature` (decimal) — sampling temperature (0.0-2.0)
- `top_p`, `top_k`, `min_p` — sampling params
- `max_tokens` (int) — max output tokens
- `reasoning` ("on"/"off") — enable extended thinking

### [[llm.providers]] — LLM provider configs (first = default)
Each provider needs `name` and `model`. Some need `api_key`.

**Ollama** (local, no key):
```toml
[[llm.providers]]
name = "ollama"
model = "llama3.2"
# url = "http://localhost:11434"
# num_ctx = 128000
```

**OpenRouter** (100+ models, needs key from openrouter.ai):
```toml
[[llm.providers]]
name = "openrouter"
api_key = "sk-or-v1-..."
model = "anthropic/claude-sonnet-4"
```

**Anthropic** (direct API, needs key from console.anthropic.com):
```toml
[[llm.providers]]
name = "anthropic"
api_key = "sk-ant-..."
model = "claude-sonnet-4-20250514"
```

**OpenAI** (or compatible endpoints):
```toml
[[llm.providers]]
name = "openai"
api_key = "sk-..."
model = "gpt-4o"
# url = "https://custom-endpoint.com/v1"
```

Multiple providers can be configured. The first one is the default; switch with `/model provider:model`.

### [tools] — Tool configuration
- `brave_api_key` (string) — enables web search (get key from brave.com/search/api)
- `[[tools.forbidden]]` — block dangerous commands:
  ```toml
  [[tools.forbidden]]
  command = "rm -rf /"
  reason = "Dangerous"
  ```

### [agent] — Agent behavior
- `system_prompt` (string) — override the default system prompt
- `max_session_age_days` (int) — auto-expire old sessions

### SOUL.md
Place a `SOUL.md` file at `~/.flashmind/SOUL.md` for a custom system prompt (alternative to the config field).

## Your tool
You have a `str_replace` tool for editing the config file. Each edit shows a diff that the user must approve before it's applied. Use it for all config changes — never ask the user to edit the file manually.

## Instructions
1. Ask what the user wants to set up (provider, tools, behavior).
2. Guide them step by step. Ask for API keys, model preferences, etc.
3. Use `str_replace` on `{config_path}` to make changes.
4. After each approved change, confirm and suggest next steps.
5. Be concise — no walls of text.
6. Credentials shown as "***" are masked for safety. Do not change masked values unless the user provides a new key."#
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_credentials() {
        let input = r#"[llm.providers.openrouter]
api_key = "sk-or-v1-abc123"
model = "anthropic/claude-sonnet-4"

[tools]
brave_api_key = "BSA-secret"
"#;
        let masked = mask_credentials(input);
        assert!(masked.contains(r#"api_key = "***""#));
        assert!(!masked.contains("sk-or-v1"));
        assert!(masked.contains("anthropic/claude-sonnet-4"));
    }

    #[test]
    fn test_restore_credentials() {
        let original = r#"[llm.providers.openrouter]
api_key = "sk-or-v1-abc123"
model = "anthropic/claude-sonnet-4"
"#;
        let new_config = r#"[llm.providers.openrouter]
api_key = "***"
model = "anthropic/claude-sonnet-4"
"#;
        let restored = restore_credentials(original, new_config);
        assert!(restored.contains(r#"api_key = "sk-or-v1-abc123""#));
        assert!(!restored.contains("***"));
    }

    #[test]
    fn test_restore_credentials_no_original() {
        let original = "";
        let new_config = r#"[telegram]
bot_token = "***"
"#;
        let restored = restore_credentials(original, new_config);
        assert!(restored.contains(r#"bot_token = "***""#));
    }
}
