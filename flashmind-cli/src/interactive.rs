use std::io;

use anyhow::{Context, Result};
use futures::StreamExt;

use flashmind_core::{Agent, CancellationToken, Conversation};
use flashmind_memory::session::SessionStore;
use flashmind_tui::widgets::{ChoiceOption, ChoicePicker, StatusInfo};
use flashmind_types::{
    AgentEvent, AgentInput, AgentLlmConfig, Model, ModelPricing, Provider, ReasoningLevel,
};

use crate::config::Config;
use crate::provider::{build_provider, fetch_pricing};
use crate::session::save_turn;
use crate::setup::run_choice;

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
            let age = crate::session::format_session_age(s.last_updated);
            ChoiceOption {
                label: format!("{} ({} entries, {})", s.chat_key, s.entry_count, age),
                accepts_input: false,
            }
        })
        .collect();

    let mut picker = ChoicePicker::new("Select a session to resume:".into(), options);
    let Some(resp) = run_choice(&mut tui, &mut picker)? else {
        return Ok(());
    };

    let chat_key = &sessions[resp.selected].chat_key;

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

    let mut conversation = Conversation::new();
    conversation.set_system(&system_prompt);
    let entries = store.load(chat_key).await?;
    for entry in &entries {
        if matches!(
            entry.entry_kind,
            flashmind_memory::session::SessionEntryKind::SystemPrompt
        ) {
            continue;
        }
        conversation.add(crate::session::session_to_conv(entry));
    }

    let model_display = model.name();
    print_banner(&mut tui, model_display, reasoning)?;

    {
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        tui.println(&Line::from(Span::styled(
            format!("  session restored: {chat_key}"),
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
            model_display: model_display.to_string(),
            reasoning,
            pricing,
            context_window,
        },
        &tool_sync,
    )
    .await?;
    save_turn(&store, &conversation).await?;
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
    pub model_display: String,
    pub reasoning: ReasoningLevel,
    pub pricing: ModelPricing,
    pub context_window: Option<u32>,
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
    print_banner(&mut tui, &state.model_display, state.reasoning)?;

    let repl_config = ReplConfig {
        prompt: "▸".to_string(),
        greeting: None,
        ..Default::default()
    };

    let mut repl = Repl::new(repl_config);
    let mut current_model = state.model_display;
    let mut current_reasoning = state.reasoning;
    let mut current_pricing = state.pricing;
    let mut current_context_window = state.context_window;

    repl.set_status(StatusInfo {
        model: current_model.clone(),
        thinking: Some(current_reasoning),
        ..Default::default()
    });

    let mut total_cost = rust_decimal::Decimal::ZERO;

    while let ReplEvent::UserInput(text) = repl.read_input()? {
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
                "help" => {
                    tui.println(&ratatui::text::Line::default())?;
                    tui.println(&ratatui::text::Line::from(
                        "  /model [provider:name]     Switch model (interactive picker if no args)",
                    ))?;
                    tui.println(&ratatui::text::Line::from(
                        "  /thinking [off|low|med|high]  Show or set reasoning level",
                    ))?;
                    tui.println(&ratatui::text::Line::from(
                        "  /help                      Show this help",
                    ))?;
                    tui.println(&ratatui::text::Line::default())?;
                    continue;
                }
                _ => {}
            }
        }

        conversation.mark_turn_start();
        let cancel = CancellationToken::new();
        let stream = agent.start(conversation, cancel.clone(), AgentInput::user(text), None);
        repl.stream_response(cancel, Box::pin(stream)).await?;
        tool_sync.sync(agent.tools_mut());

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

        save_turn(store, conversation).await?;
    }

    Ok(())
}

pub fn print_banner(
    tui: &mut flashmind_tui::Tui,
    model_display: &str,
    reasoning: ReasoningLevel,
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
        info_spans.push(Span::styled("  ", dim));
    }

    info_spans.push(Span::styled("Esc to cancel, Ctrl-D to quit", dim));
    tui.println(&Line::from(info_spans))?;
    tui.println(&Line::default())?;

    Ok(())
}
