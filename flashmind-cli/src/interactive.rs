use std::io;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::StreamExt;

use flashmind_core::{Agent, CancellationToken, Conversation};
use flashmind_memory::session::SessionStore;
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker, StatusInfo};
use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, CompletionRequest, ContentPart, LlmProvider, Message,
    Model, ModelPricing, Provider, ReasoningLevel, SamplingParams, StreamEvent,
};

use crate::config::Config;
use crate::provider::{build_provider, fetch_pricing};
use crate::session::{format_session_age, load_conversation_from, new_session_key, save_turn};
use crate::setup::run_choice;

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
    let sessions = store.list_sessions().await?;

    if sessions.is_empty() {
        println!("No sessions to resume.");
        return Ok(());
    }

    let mut tui = flashmind_tui::Tui::new();

    let options: Vec<ChoiceOption> = sessions
        .iter()
        .map(|s| {
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
        })
        .collect();

    let mut picker = ChoicePicker::new("Select a session to resume:".into(), options);
    let Some(resp) = run_choice(&mut tui, &mut picker)? else {
        return Ok(());
    };

    let session = &sessions[resp.selected];
    let chat_key = session.chat_key.clone();

    let model = crate::provider::resolve_model(cli, config)?;
    let provider = build_provider(&model, config)?;
    let (mut tools, tool_sync) = crate::tools::build_tools(config).await;

    let memory_store = crate::memory::open_memory_store(config).await?;
    if let Some(ref ms) = memory_store {
        crate::memory::register_memory_tools(&mut tools, ms);
    }

    let base_prompt = cli
        .system
        .as_deref()
        .or(config.system_prompt.as_deref())
        .unwrap_or(crate::SYSTEM_PROMPT);
    let system_prompt = if memory_store.is_some() {
        format!(
            "{base_prompt}\n\n{}",
            flashmind_prompts::MEMORY_INSTRUCTIONS
        )
    } else {
        base_prompt.to_string()
    };

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

    let model_display = model.name();
    print_banner(
        &mut tui,
        model_display,
        reasoning,
        agent.tools().list().len(),
    )?;

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
    }

    run_interactive(
        &mut agent,
        &mut conversation,
        &store,
        config,
        SessionState {
            session_key: chat_key,
            model_display: model_display.to_string(),
            reasoning,
            pricing,
            context_window,
            system_prompt,
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
    tui: &mut flashmind_tui::Tui,
) -> Result<Option<(String, ModelPricing, Option<u32>)>> {
    let input = args.trim();

    if !input.is_empty() {
        let model: Model = input
            .parse()
            .with_context(|| format!("invalid model format: {input}"))?;
        let provider = build_provider(&model, config)?;
        let pricing = fetch_pricing(&provider, &model).await;
        let context_window = provider.context_window(&model).await;
        let display = model.name().to_string();

        agent.set_provider(provider);
        agent.llm_mut().model = model;
        agent.refresh_features().await;

        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("  Switched to {display}"),
            flashmind_tui::styles::S_AGENT,
        )))?;
        return Ok(Some((display, pricing, context_window)));
    }

    let providers = configured_providers(config);
    if providers.is_empty() {
        tui.println(&ratatui::text::Line::from(
            "  No providers configured. Run `flashmind setup` first.",
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
    let display = model.name().to_string();

    agent.set_provider(tmp_provider);
    agent.llm_mut().model = model;
    agent.refresh_features().await;

    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
        format!("  Switched to {display}"),
        flashmind_tui::styles::S_AGENT,
    )))?;
    Ok(Some((display, pricing, context_window)))
}

pub struct SessionState {
    pub session_key: String,
    pub model_display: String,
    pub reasoning: ReasoningLevel,
    pub pricing: ModelPricing,
    pub context_window: Option<u32>,
    pub system_prompt: String,
}

pub async fn run_interactive(
    agent: &mut Agent,
    conversation: &mut Conversation,
    store: &SessionStore,
    config: &Config,
    state: SessionState,
    tool_sync: &flashmind_tools::tool_sync::ToolSync,
) -> Result<()> {
    use flashmind_tui::{Repl, ReplConfig, ReplEvent};

    let mut tui = flashmind_tui::Tui::new();
    print_banner(
        &mut tui,
        &state.model_display,
        state.reasoning,
        agent.tools().list().len(),
    )?;

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
    let mut current_model = state.model_display;
    let mut current_reasoning = state.reasoning;
    let mut current_pricing = state.pricing;
    let mut current_context_window = state.context_window;
    let mut system_prompt = state.system_prompt;
    let mut title_generated = false;
    let mut turn_count: usize = 0;

    repl.set_status(StatusInfo {
        model: current_model.clone(),
        thinking: Some(current_reasoning),
        ..Default::default()
    });

    let mut total_cost = rust_decimal::Decimal::ZERO;

    while let ReplEvent::UserInput(text, pasted_images) = repl.read_input()? {
        // Slash command dispatch
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
            match cmd {
                "model" => {
                    if let Some((display, new_pricing, new_cw)) =
                        handle_model_command(args, agent, config, &mut tui).await?
                    {
                        current_model = display;
                        current_pricing = new_pricing;
                        current_context_window = new_cw;
                        repl.set_status(StatusInfo {
                            model: current_model.clone(),
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
                            model: current_model.clone(),
                            thinking: Some(current_reasoning),
                            cost: Some(total_cost),
                            context: current_context_window.map(|cw| (0, cw)),
                        });
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Thinking: {level}"),
                            flashmind_tui::styles::S_AGENT,
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
                    total_cost = rust_decimal::Decimal::ZERO;
                    repl.set_status(StatusInfo {
                        model: current_model.clone(),
                        thinking: Some(current_reasoning),
                        ..Default::default()
                    });
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  New session started",
                        flashmind_tui::styles::S_AGENT,
                    )))?;
                    continue;
                }
                "clear" => {
                    *conversation = Conversation::with_system(&system_prompt);
                    turn_count = 0;
                    total_cost = rust_decimal::Decimal::ZERO;
                    save_turn(store, &session_key, conversation).await?;
                    repl.set_status(StatusInfo {
                        model: current_model.clone(),
                        thinking: Some(current_reasoning),
                        ..Default::default()
                    });
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        "  Conversation cleared",
                        flashmind_tui::styles::S_AGENT,
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
                            flashmind_tui::styles::S_TOOL_FAIL,
                        )))?;
                    } else {
                        let after = conversation.entries().len();
                        save_turn(store, &session_key, conversation).await?;
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Compacted: {before} entries -> {after}"),
                            flashmind_tui::styles::S_AGENT,
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
                            flashmind_tui::styles::S_AGENT,
                        )))?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            "  Nothing to undo",
                            flashmind_tui::styles::S_DIM,
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
                        // Re-send as a normal turn (fall through below)
                        conversation.mark_turn_start();
                        let cancel = CancellationToken::new();
                        let stream = agent.start(
                            conversation,
                            cancel.clone(),
                            AgentInput::user(retry_msg),
                            None,
                        );
                        repl.stream_response(cancel, Box::pin(stream)).await?;
                        tool_sync.sync(agent.tools_mut());
                        // turn_count stays same (we removed one, added one)
                        if let Some(usage) = repl.last_usage() {
                            if let Some(turn_cost) = usage.cost(&current_pricing) {
                                total_cost += turn_cost;
                            }
                            repl.set_status(StatusInfo {
                                model: current_model.clone(),
                                thinking: Some(current_reasoning),
                                cost: Some(total_cost),
                                context: current_context_window.map(|cw| (usage.prompt_tokens, cw)),
                            });
                        }
                        save_turn(store, &session_key, conversation).await?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            "  Nothing to retry",
                            flashmind_tui::styles::S_DIM,
                        )))?;
                    }
                    continue;
                }
                "sessions" => {
                    if turn_count > 0 {
                        save_turn(store, &session_key, conversation).await?;
                    }
                    let sessions = store.list_sessions().await?;
                    if sessions.is_empty() {
                        tui.println(&ratatui::text::Line::from("  No sessions saved."))?;
                        continue;
                    }
                    let options: Vec<ChoiceOption> = sessions
                        .iter()
                        .map(|s| {
                            let age = format_session_age(s.last_updated);
                            let title = s
                                .title
                                .as_deref()
                                .unwrap_or(&s.chat_key[..s.chat_key.len().min(20)]);
                            let current = if s.chat_key == session_key {
                                " ◀"
                            } else {
                                ""
                            };
                            ChoiceOption {
                                label: format!("{title}{current} ({} msgs, {age})", s.entry_count),
                                accepts_input: false,
                            }
                        })
                        .collect();
                    let mut picker = ChoicePicker::new("Switch session:".into(), options);
                    if let Some(resp) = run_choice(&mut tui, &mut picker)? {
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
                            total_cost = rust_decimal::Decimal::ZERO;
                            repl.set_status(StatusInfo {
                                model: current_model.clone(),
                                thinking: Some(current_reasoning),
                                ..Default::default()
                            });
                            let title = chosen
                                .title
                                .as_deref()
                                .unwrap_or(&session_key[..session_key.len().min(20)]);
                            tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                                format!("  Switched to: {title}"),
                                flashmind_tui::styles::S_AGENT,
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
                            flashmind_tui::styles::S_AGENT,
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
                            flashmind_tui::styles::S_AGENT,
                        )))?;
                    }
                    continue;
                }
                "export" => {
                    let path = if args.trim().is_empty() {
                        let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
                        format!("flashmind-export-{ts}.md")
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
                            flashmind_tui::styles::S_TOOL_FAIL,
                        )))?;
                    } else {
                        tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                            format!("  Exported to {path}"),
                            flashmind_tui::styles::S_AGENT,
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
                        .save_meta(&new_key, Some(&fork_title), Some(&current_model))
                        .await?;
                    session_key = new_key;
                    title_generated = true;
                    tui.println(&ratatui::text::Line::from(ratatui::text::Span::styled(
                        format!("  Forked session ({count} entries) — {fork_title}"),
                        flashmind_tui::styles::S_AGENT,
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

        conversation.mark_turn_start();
        let cancel = CancellationToken::new();
        let input = if pasted_images.is_empty() {
            AgentInput::user(text)
        } else {
            let parts: Vec<ContentPart> = pasted_images
                .into_iter()
                .map(|img| ContentPart::Image {
                    media_type: img.media_type,
                    data: img.data,
                })
                .collect();
            AgentInput::User {
                content: text,
                context: None,
                parts: Some(parts),
            }
        };
        let stream = agent.start(conversation, cancel.clone(), input, None);
        repl.stream_response(cancel, Box::pin(stream)).await?;
        tool_sync.sync(agent.tools_mut());
        turn_count += 1;

        if let Some(usage) = repl.last_usage() {
            if let Some(turn_cost) = usage.cost(&current_pricing) {
                total_cost += turn_cost;
            }
            repl.set_status(StatusInfo {
                model: current_model.clone(),
                thinking: Some(current_reasoning),
                cost: Some(total_cost),
                context: current_context_window.map(|cw| (usage.prompt_tokens, cw)),
            });
        }

        save_turn(store, &session_key, conversation).await?;

        // Generate session title after the first exchange
        if turn_count == 1 && !title_generated {
            title_generated = true;
            store
                .save_meta(&session_key, None, Some(&current_model))
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
    tui: &mut flashmind_tui::Tui,
    model_display: &str,
    reasoning: ReasoningLevel,
    tool_count: usize,
) -> io::Result<()> {
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    let dim = flashmind_tui::styles::S_DIM;
    let bold = Style::default().add_modifier(Modifier::BOLD);

    tui.println(&Line::default())?;
    tui.println(&Line::from(vec![
        Span::styled(
            "  flashmind",
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

    info_spans.push(Span::styled(format!("{tool_count} tools"), dim));
    info_spans.push(Span::styled("  •  ", dim));
    info_spans.push(Span::styled(
        "Esc to cancel, Ctrl-D to quit, /help for commands",
        dim,
    ));
    tui.println(&Line::from(info_spans))?;
    tui.println(&Line::default())?;

    Ok(())
}
