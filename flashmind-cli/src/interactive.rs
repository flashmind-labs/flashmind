use std::io;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use rust_decimal::Decimal;

use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry};
use flashmind_memory::session::SessionStore;
use flashmind_memory::session::SessionSummary;
use flashmind_tui::styles::{S_AGENT, S_DIM, S_TOOL_FAIL, S_TOOL_OK, S_USER_ECHO};
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker, ChoicePickerAction, StatusInfo};
use flashmind_tui::{Repl, Tui};
use flashmind_types::llm::TokenUsage;
use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, CompactionReason, CompletionRequest, ContentPart,
    LlmError, LlmErrorKind, LlmProvider, Message, Model, ModelPricing, Provider, ReasoningLevel,
    SamplingParams, StreamEvent, TurnStatus,
};

use crate::config::Config;
use crate::provider::{build_provider, fetch_pricing};
use crate::session::{format_session_age, load_conversation_from, new_session_key, save_turn};
use crate::setup::{run_choice, run_choice_action};

// ---------------------------------------------------------------------------
// Slash command list (for autocomplete)
// ---------------------------------------------------------------------------

const SLASH_COMMANDS: &[&str] = &[
    "/clear",
    "/compact",
    "/export",
    "/fork",
    "/help",
    "/mcp",
    "/memory",
    "/model",
    "/new",
    "/rename",
    "/retry",
    "/sessions",
    "/system",
    "/thinking",
    "/undo",
];

// ---------------------------------------------------------------------------
// One-shot mode
// ---------------------------------------------------------------------------

pub async fn run_oneshot(
    agent: &mut Agent,
    conversation: &mut Conversation,
    prompt: String,
) -> Result<()> {
    let cancel = CancellationToken::new();
    let stream = agent.start(conversation, cancel, AgentInput::user(prompt), None);
    tokio::pin!(stream);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {e}");
                std::process::exit(1);
            }
            AgentEvent::Done(_) => {
                println!();
                return Ok(());
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Resume mode
// ---------------------------------------------------------------------------

pub async fn run_resume(cli: &crate::Cli, config: &Config) -> Result<()> {
    let store = crate::session::open_session_store().await?;
    let mut sessions = store.list_sessions().await?;

    if sessions.is_empty() {
        println!("No sessions to resume.");
        return Ok(());
    }

    let mut tui = Tui::new();

    let options: Vec<ChoiceOption> = sessions.iter().map(session_choice_option).collect();

    let previews: Vec<String> = sessions
        .iter()
        .map(|s| s.first_message.clone().unwrap_or_default())
        .collect();

    let mut picker =
        ChoicePicker::new("Select a session to resume:".into(), options).with_previews(previews);

    let resp = loop {
        match run_choice_action(&mut tui, &mut picker)? {
            ChoicePickerAction::Select(r) => break r,
            ChoicePickerAction::Cancel => return Ok(()),
            ChoicePickerAction::Delete(idx) => {
                let key = &sessions[idx].chat_key;
                store.delete_session(key).await?;
                sessions.remove(idx);
                picker.remove(idx);
                if sessions.is_empty() {
                    println!("No sessions to resume.");
                    return Ok(());
                }
            }
        }
    };

    let session = &sessions[resp.selected];
    let chat_key = session.chat_key.clone();

    // Restore model from session metadata, fall back to CLI/config default
    let model: Model = session
        .model
        .as_deref()
        .and_then(|m| m.parse().ok())
        .unwrap_or_else(|| {
            crate::provider::resolve_model(cli, config)
                .unwrap_or_else(|_| "ollama:llama3.2".parse().unwrap())
        });
    let provider = build_provider(&model, config)?;
    let (mut tools, tool_sync, skill_index) = crate::tools::build_tools(config).await;

    let memory_store = crate::memory::open_memory_store(config).await?;
    if let Some(ref ms) = memory_store {
        crate::memory::register_memory_tools(&mut tools, ms);
    }

    let base_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(crate::SYSTEM_PROMPT);
    let mut system_prompt = base_prompt.to_string();
    if memory_store.is_some() {
        system_prompt.push_str(&format!("\n\n{}", flashmind_prompts::MEMORY_INSTRUCTIONS));
    }
    system_prompt.push_str(&format!("\n\n{}", flashmind_prompts::SKILL_INSTRUCTIONS));
    system_prompt.push_str(&skill_index.0);

    let reasoning = config.reasoning.unwrap_or(ReasoningLevel::Off);
    let llm_config = AgentLlmConfig::new(model.clone()).with_reasoning(reasoning);

    let mut agent = Agent::builder(provider.clone())
        .tools(tools)
        .llm(llm_config)
        .auto_compact(true)
        .build()
        .await;

    let pricing = fetch_pricing(&provider, &model).await;
    let context_window = provider.context_window(&model).await;

    let mut conversation = load_conversation_from(&store, &chat_key, &system_prompt).await?;

    {
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        let title = session
            .title
            .as_deref()
            .unwrap_or(&chat_key[..chat_key.len().min(20)]);
        tui.println(&Line::from(Span::styled(
            format!("  session restored: {title}"),
            Style::default().fg(Color::Green),
        )))?;
        tui.println(&Line::default())?;

        // Replay conversation history
        for entry in conversation.entries() {
            if entry.is_system() {
                continue;
            }
            if entry.is_user() {
                let text = entry.content();
                for line in text.split('\n') {
                    tui.println(&Line::from(Span::styled(line.to_string(), S_USER_ECHO)))?;
                }
                tui.println(&Line::default())?;
            } else if entry.is_assistant() {
                if let Some(reasoning) = entry.reasoning() {
                    for line in reasoning.lines() {
                        tui.println(&Line::from(Span::styled(line.to_string(), S_DIM)))?;
                    }
                }
                let content = entry.content();
                if !content.is_empty() {
                    let lines = flashmind_tui::markdown_render::render_to_lines(content);
                    for line in &lines {
                        tui.println(line)?;
                    }
                    tui.println(&Line::default())?;
                }
            } else if entry.is_tool() {
                let content = entry.content();
                let first_line = content.lines().next().unwrap_or("");
                let preview = if first_line.len() > 120 {
                    format!("{}…", &first_line[..120])
                } else {
                    first_line.to_string()
                };
                tui.println(&Line::from(vec![
                    Span::styled("✓ ", S_TOOL_OK),
                    Span::styled(preview, S_DIM),
                ]))?;
            }
        }
    }

    run_interactive(
        &mut agent,
        &mut conversation,
        &store,
        config,
        SessionState {
            session_key: chat_key,
            model: model.clone(),
            reasoning,
            pricing,
            context_window,
            system_prompt,
            skip_banner: true,
        },
        &tool_sync,
    )
    .await?;
    tool_sync.shutdown().await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive REPL
// ---------------------------------------------------------------------------

fn configured_providers(config: &Config) -> Vec<(&'static str, Provider)> {
    let mut providers = Vec::new();
    if config.ollama_url.is_some() || cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        providers.push(("Ollama", Provider::Ollama));
    }
    if config.openrouter_api_key.is_some() {
        providers.push(("OpenRouter", Provider::OpenRouter));
    }
    if config.anthropic_api_key.is_some() {
        providers.push(("Anthropic", Provider::Anthropic));
    }
    if config.openai_api_key.is_some() {
        providers.push(("OpenAI", Provider::OpenAi));
    }
    providers
}

async fn handle_model_command(
    args: &str,
    agent: &mut Agent,
    config: &Config,
    tui: &mut Tui,
) -> Result<Option<(Model, ModelPricing, Option<u32>)>> {
    let input = args.trim();

    if !input.is_empty() {
        let model: Model = input
            .parse()
            .with_context(|| format!("invalid model format: {input}"))?;
        let provider = build_provider(&model, config)?;
        let pricing = fetch_pricing(&provider, &model).await;
        let context_window = provider.context_window(&model).await;

        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("  Switched to {}", model.name()),
            S_AGENT,
        )))?;

        agent.set_provider(provider);
        agent.llm_mut().model = model.clone();
        agent.refresh_features().await;

        return Ok(Some((model, pricing, context_window)));
    }

    let providers = configured_providers(config);
    if providers.is_empty() {
        tui.println(&ratatui::text::Line::from(
            "  No providers configured. Run `flsh setup` first.",
        ))?;
        return Ok(None);
    }

    let chosen_provider = if providers.len() == 1 {
        providers[0].1
    } else {
        let current = agent.llm().model.provider;
        let options: Vec<ChoiceOption> = providers
            .iter()
            .map(|(label, p)| {
                let suffix = if *p == current { " ◀" } else { "" };
                ChoiceOption {
                    label: format!("{label}{suffix}"),
                    accepts_input: false,
                }
            })
            .collect();
        let mut picker = ChoicePicker::new("Provider".into(), options);
        match run_choice(tui, &mut picker)? {
            Some(resp) => providers[resp.selected].1,
            None => return Ok(None),
        }
    };

    let tmp_provider = build_provider(
        &Model {
            provider: chosen_provider,
            model: flashmind_types::AliasedModel {
                name: String::new(),
                real_name: None,
            },
        },
        config,
    )?;

    tui.println(&ratatui::text::Line::from(format!(
        "  Fetching models from {}...",
        tmp_provider.name()
    )))?;
    let models = tmp_provider
        .list_models()
        .await
        .context("failed to list models")?;
    if models.is_empty() {
        tui.println(&ratatui::text::Line::from("  No models found."))?;
        return Ok(None);
    }

    let model_idx = match crate::setup::run_model_picker(tui, &models)? {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let chosen = &models[model_idx];
    let prefix = match chosen_provider {
        Provider::Ollama => "ollama",
        Provider::OpenRouter => "openrouter",
        Provider::Anthropic => "anthropic",
        Provider::OpenAi => "openai",
        _ => "ollama",
    };
    let model: Model = format!("{prefix}:{}", chosen.id).parse()?;
    let pricing = fetch_pricing(&tmp_provider, &model).await;
    let context_window = tmp_provider.context_window(&model).await;

    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Switched to {}", model.name()),
        S_AGENT,
    )))?;

    agent.set_provider(tmp_provider);
    agent.llm_mut().model = model.clone();
    agent.refresh_features().await;

    Ok(Some((model, pricing, context_window)))
}

pub struct SessionState {
    pub session_key: String,
    pub model: Model,
    pub reasoning: ReasoningLevel,
    pub pricing: ModelPricing,
    pub context_window: Option<u32>,
    pub system_prompt: String,
    pub skip_banner: bool,
}

pub async fn run_interactive(
    agent: &mut Agent,
    conversation: &mut Conversation,
    store: &SessionStore,
    config: &Config,
    state: SessionState,
    tool_sync: &flashmind_tools::tool_sync::ToolSync,
) -> Result<()> {
    use flashmind_tui::{ReplConfig, ReplEvent};

    let mut tui = Tui::new();
    if !state.skip_banner {
        let tool_names: Vec<&str> = agent.tools().list();
        print_banner(&mut tui, state.model.name(), state.reasoning, &tool_names)?;
    }

    let history_file = crate::config::config_dir().ok().map(|d| d.join("history"));
    let repl_config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: None,
        history_file,
        available_commands: SLASH_COMMANDS.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };

    let mut repl = Repl::new(repl_config);
    let mut session_key = state.session_key;
    let mut current_model = state.model;
    let mut current_reasoning = state.reasoning;
    let mut current_pricing = state.pricing;
    let mut current_context_window = state.context_window;
    let mut system_prompt = state.system_prompt;
    let mut title_generated = false;
    let mut turn_count: usize = 0;

    repl.set_status(StatusInfo {
        model: current_model.name().to_string(),
        thinking: Some(current_reasoning),
        ..Default::default()
    });

    let mut total_cost = Decimal::ZERO;

    while let ReplEvent::UserInput(text, pasted_images) = repl.read_input()? {
        // Slash command dispatch
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
            match cmd {
                "model" => {
                    if let Some((new_model, new_pricing, new_cw)) =
                        handle_model_command(args, agent, config, &mut tui).await?
                    {
                        current_model = new_model;
                        current_pricing = new_pricing;
                        current_context_window = new_cw;
                        repl.set_status(StatusInfo {
                            model: current_model.name().to_string(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window.map(|cw| (0, cw)),
                        });
                    }
                    continue;
                }
                "thinking" => {
                    let input = args.trim();
                    let level = if input.is_empty() {
                        let options = vec![
                            ChoiceOption {
                                label: "Off".into(),
                                accepts_input: false,
                            },
                            ChoiceOption {
                                label: "Low".into(),
                                accepts_input: false,
                            },
                            ChoiceOption {
                                label: "Medium".into(),
                                accepts_input: false,
                            },
                            ChoiceOption {
                                label: "High".into(),
                                accepts_input: false,
                            },
                        ];
                        let mut picker = ChoicePicker::new(
                            format!("Thinking (current: {current_reasoning})"),
                            options,
                        );
                        run_choice(&mut tui, &mut picker)?.map(|resp| match resp.selected {
                            1 => ReasoningLevel::Low,
                            2 => ReasoningLevel::Medium,
                            3 => ReasoningLevel::High,
                            _ => ReasoningLevel::Off,
                        })
                    } else {
                        match input {
                            "off" => Some(ReasoningLevel::Off),
                            "low" => Some(ReasoningLevel::Low),
                            "medium" | "on" => Some(ReasoningLevel::Medium),
                            "high" => Some(ReasoningLevel::High),
                            _ => {
                                tui.println(&ratatui::text::Line::from(
                                    "  Usage: /thinking [off|low|medium|high]",
                                ))?;
                                None
                            }
                        }
                    };
                    if let Some(level) = level {
                        agent.llm_mut().reasoning = level;
                        current_reasoning = level;
                        repl.set_status(StatusInfo {
                            model: current_model.name().to_string(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window.map(|cw| (0, cw)),
                        });
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Thinking: {level}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "new" => {
                    if turn_count > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    session_key = new_session_key();
                    *conversation = Conversation::with_system(&system_prompt);
                    title_generated = false;
                    turn_count = 0;
                    total_cost = Decimal::ZERO;
                    repl.set_status(StatusInfo {
                        model: current_model.name().to_string(),
                        thinking: Some(current_reasoning),
                        ..Default::default()
                    });
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  New session started",
                        S_AGENT,
                    )))?;
                    continue;
                }
                "clear" => {
                    *conversation = Conversation::with_system(&system_prompt);
                    turn_count = 0;
                    total_cost = Decimal::ZERO;
                    save_turn(store, &session_key, conversation).await?;
                    repl.set_status(StatusInfo {
                        model: current_model.name().to_string(),
                        thinking: Some(current_reasoning),
                        ..Default::default()
                    });
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  Conversation cleared",
                        S_AGENT,
                    )))?;
                    continue;
                }
                "compact" => {
                    let before = conversation.entries().len();
                    if let Err(e) = conversation
                        .compact_with_llm(agent.provider(), &agent.llm().model)
                        .await
                    {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Compaction failed: {e}"),
                            S_TOOL_FAIL,
                        )))?;
                    } else {
                        let after = conversation.entries().len();
                        save_turn(store, &session_key, conversation).await?;
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Compacted: {before} entries -> {after}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "undo" => {
                    let entries = conversation.entries_mut();
                    // Pop from the end: remove tool results, assistant, and user entries for the last turn
                    let before = entries.len();
                    while entries
                        .last()
                        .is_some_and(|e| e.is_tool() || e.is_assistant())
                    {
                        entries.pop();
                    }
                    // Remove the last user message
                    if entries.last().is_some_and(|e| e.is_user()) {
                        entries.pop();
                    }
                    let removed = before - entries.len();
                    if removed > 0 {
                        turn_count = turn_count.saturating_sub(1);
                        save_turn(store, &session_key, conversation).await?;
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Removed last turn ({removed} entries)"),
                            S_AGENT,
                        )))?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            "  Nothing to undo",
                            S_DIM,
                        )))?;
                    }
                    continue;
                }
                "retry" => {
                    let extra = args.trim();
                    // Find the last user message before popping
                    let last_user_msg = conversation
                        .entries()
                        .iter()
                        .rev()
                        .find(|e| e.is_user())
                        .map(|e| e.content().to_string());
                    // Pop last turn (same as /undo)
                    let entries = conversation.entries_mut();
                    while entries
                        .last()
                        .is_some_and(|e| e.is_tool() || e.is_assistant())
                    {
                        entries.pop();
                    }
                    if entries.last().is_some_and(|e| e.is_user()) {
                        entries.pop();
                    }
                    if let Some(original) = last_user_msg {
                        let retry_msg = if extra.is_empty() {
                            original
                        } else {
                            format!("{original}\n\n{extra}")
                        };
                        conversation.add(ConversationEntry::user(&retry_msg));
                        conversation.mark_turn_start();
                        run_turn_loop(
                            agent,
                            conversation,
                            &mut repl,
                            &mut total_cost,
                            &current_pricing,
                            store,
                            &session_key,
                        )
                        .await?;
                        tool_sync.sync(agent.tools_mut());
                        repl.set_status(StatusInfo {
                            model: current_model.name().to_string(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window
                                .and_then(|cw| repl.last_usage().map(|u| (u.prompt_tokens, cw))),
                        });
                        save_turn(store, &session_key, conversation).await?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            "  Nothing to retry",
                            S_DIM,
                        )))?;
                    }
                    continue;
                }
                "sessions" => {
                    if turn_count > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    let mut sessions = store.list_sessions().await?;
                    if sessions.is_empty() {
                        tui.println(&ratatui::text::Line::from("  No sessions saved."))?;
                        continue;
                    }
                    let options: Vec<ChoiceOption> = sessions
                        .iter()
                        .map(|s| {
                            let mut opt = session_choice_option(s);
                            if s.chat_key == session_key {
                                opt.label.push_str(" ◀");
                            }
                            opt
                        })
                        .collect();
                    let previews: Vec<String> = sessions
                        .iter()
                        .map(|s| s.first_message.clone().unwrap_or_default())
                        .collect();
                    let mut picker = ChoicePicker::new("Switch session:".into(), options)
                        .with_previews(previews);
                    let selected = loop {
                        match run_choice_action(&mut tui, &mut picker)? {
                            ChoicePickerAction::Select(r) => break Some(r),
                            ChoicePickerAction::Cancel => break None,
                            ChoicePickerAction::Delete(idx) => {
                                let key = &sessions[idx].chat_key;
                                if *key == session_key {
                                    continue;
                                }
                                store.delete_session(key).await?;
                                sessions.remove(idx);
                                picker.remove(idx);
                                if sessions.is_empty() {
                                    break None;
                                }
                            }
                        }
                    };
                    if let Some(resp) = selected {
                        let chosen = &sessions[resp.selected];
                        if chosen.chat_key != session_key {
                            session_key = chosen.chat_key.clone();
                            *conversation =
                                load_conversation_from(store, &session_key, &system_prompt).await?;
                            title_generated = chosen.title.is_some();
                            turn_count = conversation
                                .entries()
                                .iter()
                                .filter(|e| e.is_user())
                                .count();
                            total_cost = Decimal::ZERO;
                            repl.set_status(StatusInfo {
                                model: current_model.name().to_string(),
                                thinking: Some(current_reasoning),
                                ..Default::default()
                            });
                            let title = chosen
                                .title
                                .as_deref()
                                .unwrap_or(&session_key[..session_key.len().min(20)]);
                            tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                                format!("  Switched to: {title}"),
                                S_AGENT,
                            )))?;
                        }
                    }
                    continue;
                }
                "rename" => {
                    let new_title = args.trim();
                    if new_title.is_empty() {
                        tui.println(&ratatui::text::Line::from("  Usage: /rename <new title>"))?;
                    } else {
                        store.update_title(&session_key, new_title).await?;
                        title_generated = true;
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Session renamed to: {new_title}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "system" => {
                    let input = args.trim();
                    if input.is_empty() {
                        let current = conversation
                            .entries()
                            .first()
                            .filter(|e| e.is_system())
                            .map(|e| e.content())
                            .unwrap_or("(none)");
                        tui.println(&ratatui::text::Line::default())?;
                        for line in current.lines() {
                            tui.println(&ratatui::text::Line::from(format!("  {line}")))?;
                        }
                        tui.println(&ratatui::text::Line::default())?;
                    } else {
                        system_prompt = input.to_string();
                        conversation.set_system(&system_prompt);
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            "  System prompt updated",
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "export" => {
                    let path = if args.trim().is_empty() {
                        let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
                        format!("flsh-export-{ts}.md")
                    } else {
                        args.trim().to_string()
                    };
                    let mut output = String::new();
                    for entry in conversation.entries() {
                        match &entry.kind {
                            flashmind_core::EntryKind::SystemPrompt(s) => {
                                output.push_str("## System Prompt\n\n");
                                output.push_str(s);
                                output.push_str("\n\n---\n\n");
                            }
                            flashmind_core::EntryKind::User { content, .. } => {
                                output.push_str("## User\n\n");
                                output.push_str(content);
                                output.push_str("\n\n");
                            }
                            flashmind_core::EntryKind::Assistant { content, .. } => {
                                output.push_str("## Assistant\n\n");
                                output.push_str(content);
                                output.push_str("\n\n");
                            }
                            flashmind_core::EntryKind::Tool { call_id, output: o } => {
                                output.push_str(&format!(
                                    "### Tool Result ({call_id})\n\n```\n{o}\n```\n\n"
                                ));
                            }
                            flashmind_core::EntryKind::Developer { content, tag, .. } => {
                                let tag_str = tag.as_deref().unwrap_or("developer");
                                output.push_str(&format!("### {tag_str}\n\n{content}\n\n"));
                            }
                        }
                    }
                    if let Err(e) = std::fs::write(&path, &output) {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Export failed: {e}"),
                            S_TOOL_FAIL,
                        )))?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Exported to {path}"),
                            S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "fork" => {
                    save_turn(store, &session_key, conversation).await?;
                    let new_key = new_session_key();
                    let count = store.branch(&session_key, &new_key).await?;
                    // Copy metadata with a "(fork)" suffix
                    let sessions = store.list_sessions().await?;
                    let old_title = sessions
                        .iter()
                        .find(|s| s.chat_key == session_key)
                        .and_then(|s| s.title.as_deref())
                        .unwrap_or("untitled");
                    let fork_title = format!("{old_title} (fork)");
                    store
                        .save_meta(
                            &new_key,
                            Some(&fork_title),
                            Some(&current_model.to_string()),
                        )
                        .await?;
                    session_key = new_key;
                    title_generated = true;
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        format!("  Forked session ({count} entries) — {fork_title}"),
                        S_AGENT,
                    )))?;
                    continue;
                }
                "mcp" => {
                    let mcp_dir = crate::config::config_dir()
                        .map(|d| d.join("mcp"))
                        .unwrap_or_default();
                    if mcp_dir.exists() {
                        let entries: Vec<_> = std::fs::read_dir(&mcp_dir)
                            .into_iter()
                            .flatten()
                            .filter_map(|e| e.ok())
                            .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
                            .collect();
                        if entries.is_empty() {
                            tui.println(&ratatui::text::Line::from(
                                "  No MCP servers configured.",
                            ))?;
                        } else {
                            tui.println(&ratatui::text::Line::default())?;
                            for entry in &entries {
                                let name = entry
                                    .path()
                                    .file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();
                                tui.println(&ratatui::text::Line::from(format!("    {name}")))?;
                            }
                            tui.println(&ratatui::text::Line::default())?;
                        }
                    } else {
                        tui.println(&ratatui::text::Line::from("  No MCP servers configured."))?;
                    }
                    continue;
                }
                "memory" => {
                    let query = args.trim();
                    if query.is_empty() {
                        tui.println(&ratatui::text::Line::from(
                            "  Usage: /memory <search query>",
                        ))?;
                    } else {
                        // Memory search is handled by the agent — pass it through
                        // by falling through to normal message handling
                    }
                    if args.trim().is_empty() {
                        continue;
                    }
                    // Fall through: send as a regular message asking for memory recall
                }
                "help" => {
                    tui.println(&ratatui::text::Line::default())?;
                    let cmds = [
                        ("/model [provider:name]", "Switch model"),
                        ("/thinking [off|low|med|high]", "Set reasoning level"),
                        ("/new", "Start a new session"),
                        ("/clear", "Clear conversation"),
                        ("/compact", "Compress conversation context"),
                        ("/undo", "Remove last turn"),
                        (
                            "/retry [extra context]",
                            "Redo last turn, optionally with more context",
                        ),
                        ("/sessions", "List and switch sessions"),
                        ("/rename <title>", "Rename current session"),
                        ("/system [prompt]", "View or set system prompt"),
                        ("/export [path]", "Export conversation to markdown"),
                        ("/fork", "Fork current session"),
                        ("/mcp", "Show MCP servers"),
                        ("/memory <query>", "Search memory"),
                        ("/help", "Show this help"),
                        ("", ""),
                        ("Ctrl+V", "Paste image from clipboard"),
                        ("Ctrl+L", "Clear screen"),
                    ];
                    for (cmd, desc) in &cmds {
                        tui.println(&ratatui::text::Line::from(format!("  {:<30} {desc}", cmd)))?;
                    }
                    tui.println(&ratatui::text::Line::default())?;
                    continue;
                }
                _ => {} // unknown slash commands fall through to the agent
            }
        }

        // Add user message to conversation
        {
            if pasted_images.is_empty() {
                conversation.add(ConversationEntry::user(&text));
            } else {
                let parts: Vec<ContentPart> = pasted_images
                    .into_iter()
                    .map(|img| ContentPart::Image {
                        media_type: img.media_type,
                        data: img.data,
                    })
                    .collect();
                conversation.add(ConversationEntry::user_with_parts(&text, parts));
            }
        }
        conversation.mark_turn_start();

        run_turn_loop(
            agent,
            conversation,
            &mut repl,
            &mut total_cost,
            &current_pricing,
            store,
            &session_key,
        )
        .await?;
        tool_sync.sync(agent.tools_mut());
        turn_count += 1;

        repl.set_status(StatusInfo {
            model: current_model.name().to_string(),
            thinking: Some(current_reasoning),
            cost: Some(total_cost),
            context: current_context_window
                .and_then(|cw| repl.last_usage().map(|u| (u.prompt_tokens, cw))),
        });

        save_turn(store, &session_key, conversation).await?;

        // Generate session title after the first exchange
        if turn_count == 1 && !title_generated {
            title_generated = true;
            store
                .save_meta(&session_key, None, Some(&current_model.to_string()))
                .await?;
            let provider = agent.provider_arc().clone();
            let model = agent.llm().model.clone();
            let first_user = conversation
                .entries()
                .iter()
                .find(|e| e.is_user())
                .map(|e| e.content().to_string());
            let first_assistant = conversation
                .entries()
                .iter()
                .find(|e| e.is_assistant())
                .map(|e| {
                    let c = e.content();
                    if c.len() > 200 {
                        format!("{}...", &c[..200])
                    } else {
                        c.to_string()
                    }
                });
            if let (Some(user_msg), Some(asst_msg)) = (first_user, first_assistant) {
                let store_clone = store.clone();
                let key = session_key.clone();
                tokio::spawn(async move {
                    if let Ok(title) = generate_title(&provider, &model, &user_msg, &asst_msg).await
                    {
                        let _ = store_clone.update_title(&key, &title).await;
                    }
                });
            }
        }
    }

    // Final save on exit
    if turn_count > 0 {
        save_turn(store, &session_key, conversation).await?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Turn-based agent loop
// ---------------------------------------------------------------------------

/// Run the agent turn loop: stream one LLM turn, execute tool calls, repeat.
async fn run_turn_loop(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    total_cost: &mut Decimal,
    pricing: &ModelPricing,
    store: &SessionStore,
    session_key: &str,
) -> Result<()> {
    let cancel = CancellationToken::new();
    let mut compactions: u8 = 0;
    let mut error_compactions: u8 = 0;
    let mut stream_retries: u8 = 0;

    repl.mark_turn_start();

    loop {
        // Inject any messages the user submitted during the previous turn.
        drain_pending_inputs(repl, conversation);

        // Stream one LLM turn (text/reasoning deltas).
        // Scoped so the mutable borrows on agent+conversation are released
        // before we touch them for tool execution.
        let (cancelled, result) = {
            let mut turn = agent.run_turn(conversation, &cancel);
            let cancelled = repl.stream_events(&cancel, &mut turn).await?;
            let result = turn
                .take_result()
                .unwrap_or_else(|| Err(anyhow::anyhow!("stream ended without result")));
            (cancelled, result)
        };

        if cancelled {
            conversation.add(ConversationEntry::assistant("User interrupted the task"));
            repl.finish_turn()?;
            break;
        }

        match result {
            Ok(TurnStatus::Done { usage, .. }) => {
                accrue_cost(total_cost, pricing, usage.into());
                repl.finish_turn()?;
                break;
            }

            Ok(TurnStatus::Continue { usage, .. }) => {
                accrue_cost(total_cost, pricing, usage.into());
                compactions = 0;
                continue;
            }

            Ok(TurnStatus::ToolCalls {
                tool_calls, usage, ..
            }) => {
                let token_usage: TokenUsage = usage.into();
                accrue_cost(total_cost, pricing, token_usage.clone());
                compactions = 0;
                repl.emit_event(&AgentEvent::Usage(token_usage))?;

                if execute_tools(agent, conversation, repl, &tool_calls, &cancel).await? {
                    break; // tool interrupted
                }
                // Persist after tool execution for crash recovery
                let _ = save_turn(store, session_key, conversation).await;
                continue;
            }

            Ok(TurnStatus::Interrupted { .. }) => {
                repl.finish_turn()?;
                break;
            }

            Ok(TurnStatus::CompactionNeeded { usage, reason, .. }) => {
                accrue_cost(total_cost, pricing, usage.into());
                compactions += 1;
                if compactions > 2 {
                    repl.emit_event(&AgentEvent::Error(
                        "Context too small — compaction loop".into(),
                    ))?;
                    repl.finish_turn()?;
                    break;
                }
                compact(agent, conversation, repl, reason).await?;
                if matches!(reason, CompactionReason::ContextThreshold(_)) {
                    repl.finish_turn()?;
                    break;
                }
                continue;
            }

            Err(e) => {
                if try_recover(
                    agent,
                    conversation,
                    repl,
                    &cancel,
                    &e,
                    &mut error_compactions,
                    &mut stream_retries,
                )
                .await?
                {
                    continue;
                }
                repl.emit_event(&AgentEvent::Error(e.to_string()))?;
                repl.finish_turn()?;
                break;
            }
        }
    }

    conversation.strip_agent_progress();
    conversation.strip_memories();
    conversation.strip_reminders();
    Ok(())
}

// ---------------------------------------------------------------------------
// Turn-loop helpers
// ---------------------------------------------------------------------------

fn accrue_cost(total: &mut Decimal, pricing: &ModelPricing, usage: TokenUsage) {
    if let Some(cost) = usage.cost(pricing) {
        *total += cost;
    }
}

fn drain_pending_inputs(repl: &mut Repl<'_>, conversation: &mut Conversation) {
    while let Some((text, images)) = repl.take_pending_input() {
        if images.is_empty() {
            conversation.add(ConversationEntry::user(&text));
        } else {
            let parts: Vec<ContentPart> = images
                .into_iter()
                .map(|img| ContentPart::Image {
                    media_type: img.media_type,
                    data: img.data,
                })
                .collect();
            conversation.add(ConversationEntry::user_with_parts(&text, parts));
        }
        conversation.mark_turn_start();
        repl.mark_turn_start();
    }
}

/// Execute tool calls. Returns `true` if a tool interrupted (caller should break).
async fn execute_tools(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    tool_calls: &[flashmind_types::ToolCall],
    cancel: &CancellationToken,
) -> Result<bool, io::Error> {
    for tc in tool_calls {
        let humanized = agent.tools().humanize(tc);
        repl.emit_event(&AgentEvent::ToolStart {
            name: tc.name.clone(),
            id: tc.id.clone(),
            humanized,
        })?;

        let start = Instant::now();
        let result = repl
            .run_tool_ui(cancel, agent.tools().execute(tc, None, cancel))
            .await?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        for diff in result.diffs() {
            repl.emit_event(&AgentEvent::FileDiff {
                path: diff.path.clone(),
                diff: diff.diff.clone(),
            })?;
        }

        if result.is_interrupt() {
            repl.emit_event(&AgentEvent::Interrupted {
                tool_call_id: tc.id.clone(),
                tool_name: tc.name.clone(),
                output: result.output(),
                payload: result.payload().cloned(),
            })?;
            repl.finish_turn()?;
            return Ok(true);
        }

        conversation.add(ConversationEntry::tool(&tc.id, result.output()));

        repl.emit_event(&AgentEvent::ToolResult {
            name: tc.name.clone(),
            id: tc.id.clone(),
            output: result.output().to_string(),
            success: result.is_success(),
            elapsed_ms,
            sources: result.sources().to_vec(),
        })?;
    }
    Ok(false)
}

async fn compact(
    agent: &Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    reason: CompactionReason,
) -> Result<()> {
    let provider = agent.provider_arc().clone();
    let model = agent.llm().model.clone();

    match reason {
        CompactionReason::OutputLength => {
            repl.emit_event(&AgentEvent::Status(
                "Response truncated — compacting...".into(),
            ))?;
            if let Err(e) = conversation.compact_with_llm(&*provider, &model).await {
                tracing::warn!("Output-length compaction failed: {e}");
            }
        }
        CompactionReason::ContextThreshold(prompt_tokens) => {
            let stream = flashmind_core::compaction::try_compact(
                conversation,
                prompt_tokens,
                agent.context_window(),
                &*provider,
                &model,
            );
            tokio::pin!(stream);
            while let Some(ev) = stream.next().await {
                repl.emit_event(&ev)?;
            }
        }
    }
    Ok(())
}

/// Attempt error recovery (compaction, stream retry, binary strip).
/// Returns `true` if recovery succeeded and the caller should `continue`.
async fn try_recover(
    agent: &mut Agent,
    conversation: &mut Conversation,
    repl: &mut Repl<'_>,
    cancel: &CancellationToken,
    error: &anyhow::Error,
    error_compactions: &mut u8,
    stream_retries: &mut u8,
) -> Result<bool, io::Error> {
    let llm_err = error.downcast_ref::<LlmError>();

    if llm_err.is_some_and(|e| e.is_recoverable()) && *error_compactions < 2 {
        *error_compactions += 1;
        let provider = agent.provider_arc().clone();
        let model = agent.llm().model.clone();
        let mut stream = flashmind_core::handle_llm_error(
            conversation,
            &*provider,
            &model,
            cancel,
            *error_compactions,
        );
        while let Some(ev) = stream.next().await {
            repl.emit_event(&ev)?;
        }
        return Ok(true);
    }

    if llm_err.is_some_and(|e| e.kind == LlmErrorKind::StreamError) && *stream_retries < 3 {
        *stream_retries += 1;
        repl.emit_event(&AgentEvent::Status(
            "Connection dropped — retrying...".into(),
        ))?;
        return Ok(true);
    }

    if conversation.strip_binary_parts() > 0 {
        repl.emit_event(&AgentEvent::Status(
            "Stripped images/documents — retrying...".into(),
        ))?;
        return Ok(true);
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// Title generation
// ---------------------------------------------------------------------------

async fn generate_title(
    provider: &Arc<dyn LlmProvider>,
    model: &Model,
    user_msg: &str,
    assistant_msg: &str,
) -> Result<String> {
    let request = CompletionRequest {
        model: model.clone(),
        messages: vec![
            Message::system(
                "Generate a short title (3-6 words) for the following conversation. Reply with ONLY the title, nothing else.",
            ),
            Message::user(format!("User: {user_msg}\n\nAssistant: {assistant_msg}")),
        ],
        tools: vec![],
        max_tokens: Some(30),
        reasoning: ReasoningLevel::Off,
        sampling: SamplingParams::default(),
        modalities: vec![],
        audio_config: None,
        image_config: None,
        user: None,
        provider_preferences: None,
    };

    let stream = provider.complete(request);
    tokio::pin!(stream);

    let mut title = String::new();
    while let Some(event) = stream.next().await {
        if let Ok(StreamEvent::ContentDelta(delta)) = event {
            title.push_str(&delta);
        }
    }

    let title = title.trim().trim_matches('"').trim().to_string();
    if title.is_empty() {
        anyhow::bail!("empty title");
    }
    Ok(title)
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

pub fn print_banner(
    tui: &mut Tui,
    model_display: &str,
    reasoning: ReasoningLevel,
    tool_names: &[&str],
) -> io::Result<()> {
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    let dim = S_DIM;
    let bold = Style::default().add_modifier(Modifier::BOLD);

    tui.println(&Line::default())?;
    tui.println(&Line::from(vec![
        Span::styled(
            "  flsh",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" — ", dim),
        Span::styled(model_display, bold),
    ]))?;

    let mut info_spans = Vec::new();
    info_spans.push(Span::styled("  ", dim));

    if reasoning.is_on() {
        info_spans.push(Span::styled(
            format!("thinking: {reasoning}"),
            Style::default().fg(Color::Yellow),
        ));
        info_spans.push(Span::styled("  •  ", dim));
    }

    info_spans.push(Span::styled(
        "Esc to cancel, Ctrl-D to quit, /help for commands",
        dim,
    ));
    tui.println(&Line::from(info_spans))?;

    if !tool_names.is_empty() {
        let (term_w, _) = ratatui::crossterm::terminal::size().unwrap_or((80, 24));
        let max_w = term_w as usize;
        let mut line = String::from("  ");
        for (i, name) in tool_names.iter().enumerate() {
            let sep = if i > 0 { ", " } else { "" };
            if line.len() + sep.len() + name.len() > max_w {
                tui.println(&Line::from(Span::styled(line.clone(), dim)))?;
                line = format!("  {name}");
            } else {
                line.push_str(sep);
                line.push_str(name);
            }
        }
        if !line.trim().is_empty() {
            tui.println(&Line::from(Span::styled(line, dim)))?;
        }
    }

    tui.println(&Line::default())?;

    Ok(())
}

fn session_choice_option(s: &SessionSummary) -> ChoiceOption {
    let age = format_session_age(s.last_updated);
    let title = s
        .title
        .as_deref()
        .unwrap_or(&s.chat_key[..s.chat_key.len().min(20)]);
    let model_info = s
        .model
        .as_deref()
        .map(|m| format!(" [{m}]"))
        .unwrap_or_default();
    ChoiceOption {
        label: format!("{title}{model_info} ({} msgs, {age})", s.entry_count),
        accepts_input: false,
    }
}
