use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use async_trait::async_trait;
use flashmind_core::subagent::AgentManager;
use flashmind_skills::{
    DiskSkillProvider, SkillInstallTool, SkillListTool, SkillLoadTool, SkillProvider, SkillRunTool,
    SkillRunner, SkillSaveTool,
};
use flashmind_tools::ToolBuilder;
use flashmind_tools::protected::ProtectedPaths;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use flashmind_types::tool::{InterruptPayload, Tool, ToolContext, ToolResult};
use flashmind_types::{AgentLlmConfig, LlmProvider, Model};

// ---------------------------------------------------------------------------
// StringPayload — simple InterruptPayload for serialized JSON
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct StringPayload(pub String);

impl InterruptPayload for StringPayload {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn display_output(&self) -> String {
        self.0.clone()
    }
}

// ---------------------------------------------------------------------------
// ProposeChoiceTool
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ProposeChoiceArgs {
    title: String,
    options: Vec<OptionArg>,
}

#[derive(Debug, Deserialize)]
struct OptionArg {
    label: String,
    #[serde(default)]
    accepts_input: bool,
}

#[derive(serde::Serialize, Deserialize)]
pub struct ChoiceProposal {
    pub title: String,
    pub options: Vec<ChoiceProposalOption>,
}

#[derive(serde::Serialize, Deserialize)]
pub struct ChoiceProposalOption {
    pub label: String,
    #[serde(default)]
    pub accepts_input: bool,
}

pub struct ProposeChoiceTool;

#[async_trait]
impl Tool for ProposeChoiceTool {
    fn name(&self) -> &str {
        "propose_choice"
    }

    fn description(&self) -> &str {
        "Present numbered options to the user for selection. The last option can accept free-form text input."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title/question for the choice"
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": { "type": "string", "description": "Option text" },
                            "accepts_input": {
                                "type": "boolean",
                                "description": "If true, user can type free-form text. Only use on the last option.",
                                "default": false
                            }
                        },
                        "required": ["label"]
                    },
                    "description": "List of options. The last one may have accepts_input: true for free-form input."
                }
            },
            "required": ["title", "options"]
        })
    }

    fn humanize(&self, args: &Value) -> String {
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("choice");
        let count = args
            .get("options")
            .and_then(|v| v.as_array())
            .map_or(0, |a| a.len());
        format!("Propose \"{title}\" ({count} options)")
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ProposeChoiceArgs = ctx.parse_args("propose_choice")?;

        let mut options: Vec<ChoiceProposalOption> = args
            .options
            .into_iter()
            .map(|o| ChoiceProposalOption {
                label: o.label,
                accepts_input: o.accepts_input,
            })
            .collect();

        if !options.iter().any(|o| o.accepts_input) {
            options.push(ChoiceProposalOption {
                label: "Tell Flash what to do...".into(),
                accepts_input: true,
            });
        }

        let proposal = ChoiceProposal {
            title: args.title,
            options,
        };

        Ok(ToolResult::interrupt(
            ctx.tool_call_id,
            Arc::new(StringPayload(serde_json::to_string(&proposal)?)),
        ))
    }
}

use crate::config::{Config, config_dir};

pub fn mcp_config_dir() -> Result<PathBuf> {
    let dir = config_dir()?.join("mcp");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn compute_skill_dirs(config: &Config) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(ref skill_dirs) = config.skill_dirs {
        for d in skill_dirs {
            let expanded = if let Some(rest) = d.strip_prefix("~/") {
                if let Some(home) = dirs::home_dir() {
                    home.join(rest)
                } else {
                    PathBuf::from(d)
                }
            } else {
                PathBuf::from(d)
            };
            dirs.push(expanded);
        }
    }

    if let Ok(cd) = config_dir() {
        let default_dir = cd.join("skills");
        let _ = std::fs::create_dir_all(&default_dir);
        if !dirs.contains(&default_dir) {
            dirs.push(default_dir);
        }
    }

    let cwd_skills = std::env::current_dir().unwrap_or_default().join("skills");
    if cwd_skills.is_dir() && !dirs.contains(&cwd_skills) {
        dirs.push(cwd_skills);
    }

    // Claude Code backwards compat: ~/.claude/commands/
    if let Some(home) = dirs::home_dir() {
        let claude_global = home.join(".claude").join("commands");
        if claude_global.is_dir() && !dirs.contains(&claude_global) {
            dirs.push(claude_global);
        }
    }

    // Claude Code backwards compat: .claude/commands/ (project-local)
    let claude_local = std::env::current_dir()
        .unwrap_or_default()
        .join(".claude")
        .join("commands");
    if claude_local.is_dir() && !dirs.contains(&claude_local) {
        dirs.push(claude_local);
    }

    dirs
}

pub struct SkillIndex(pub String);

pub async fn build_tools(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
) -> (
    flashmind_types::ToolRegistry,
    flashmind_tools::tool_sync::ToolSync,
    SkillIndex,
    Arc<RwLock<DiskSkillProvider>>,
    Arc<SkillRunner>,
) {
    let mcp_config_dir = config_dir()
        .map(|d| d.join("mcp"))
        .unwrap_or_else(|_| PathBuf::from(".flashmind/mcp"));
    let _ = std::fs::create_dir_all(&mcp_config_dir);
    let mcp_provider = flashmind_tools::mcp::McpDiskConfig::new(mcp_config_dir);

    let protected = Arc::new(ProtectedPaths::new(&PathBuf::from("/")));

    let vision_model: Option<Model> = config.vision_model.as_deref().and_then(|s| s.parse().ok());

    let manager = Arc::new(AgentManager::new(4, 2));

    let mut builder = ToolBuilder::new()
        .file_ops(vision_model.clone(), &protected)
        .bash(vec![], &protected, vec![], None)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .time()
        .subagents(manager, provider, Some(llm))
        .mcp(mcp_provider, None);

    if let Some(ref vm) = vision_model
        && let Ok(p) = crate::provider::build_provider(vm, config)
    {
        let mut providers = std::collections::HashMap::new();
        providers.insert(vm.provider, p);
        builder = builder.with_providers(std::sync::Arc::new(providers));
    }

    let (mut registry, sync) = builder.build_with_sync().await;

    registry.remove("file_delete");
    registry.remove("file_list");
    registry.remove("process");
    registry.remove("mcp_add");
    registry.remove("mcp_remove");
    registry.remove("mcp_list");
    registry.remove("mcp_auth");

    // Skill tools
    let skill_dirs = compute_skill_dirs(config);
    let provider = match DiskSkillProvider::discover(skill_dirs).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("skill discovery failed: {e}");
            DiskSkillProvider::discover(vec![]).await.unwrap()
        }
    };
    let provider = Arc::new(RwLock::new(provider));
    let runner = Arc::new(SkillRunner::new(Duration::from_secs(30)));

    registry.register(Arc::new(ProposeChoiceTool));
    registry.register(Arc::new(SkillListTool {
        provider: provider.clone(),
    }));
    registry.register(Arc::new(SkillLoadTool {
        provider: provider.clone(),
    }));
    registry.register(Arc::new(SkillRunTool {
        provider: provider.clone(),
        runner: runner.clone(),
    }));
    registry.register(Arc::new(SkillInstallTool {
        provider: provider.clone(),
    }));
    registry.register(Arc::new(SkillSaveTool {
        provider: provider.clone(),
    }));

    let skill_index = provider.read().await.skill_index();

    (registry, sync, SkillIndex(skill_index), provider, runner)
}

pub async fn build_tools_full(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
) -> (
    flashmind_types::ToolRegistry,
    flashmind_tools::tool_sync::ToolSync,
    SkillIndex,
    Arc<RwLock<DiskSkillProvider>>,
    Arc<SkillRunner>,
) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let protected = Arc::new(ProtectedPaths::new(&cwd));

    let mcp_config_dir = config_dir()
        .map(|d| d.join("mcp"))
        .unwrap_or_else(|_| PathBuf::from(".flashmind/mcp"));
    let _ = std::fs::create_dir_all(&mcp_config_dir);
    let mcp_provider = flashmind_tools::mcp::McpDiskConfig::new(mcp_config_dir);

    let skill_dirs = compute_skill_dirs(config);
    let skill_provider = match DiskSkillProvider::discover(skill_dirs).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("skill discovery failed: {e}");
            DiskSkillProvider::discover(vec![]).await.unwrap()
        }
    };
    let skill_provider = Arc::new(RwLock::new(skill_provider));
    let runner = Arc::new(SkillRunner::new(Duration::from_secs(30)));

    let vision_model: Option<Model> = config.vision_model.as_deref().and_then(|s| s.parse().ok());

    let manager = Arc::new(AgentManager::new(4, 2));

    let mut builder = ToolBuilder::new()
        .file_ops(vision_model.clone(), &protected)
        .bash(vec![], &protected, vec![], None)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .time()
        .skills(skill_provider.clone(), runner.clone())
        .subagents(manager, provider, Some(llm))
        .mcp(mcp_provider, None);

    if let Some(ref vm) = vision_model
        && let Ok(p) = crate::provider::build_provider(vm, config)
    {
        let mut providers = std::collections::HashMap::new();
        providers.insert(vm.provider, p);
        builder = builder.with_providers(std::sync::Arc::new(providers));
    }

    let (mut registry, sync) = builder.build_with_sync().await;

    registry.remove("file_delete");
    registry.remove("file_list");
    registry.remove("process");
    registry.remove("mcp_add");
    registry.remove("mcp_remove");
    registry.remove("mcp_list");
    registry.remove("mcp_auth");

    registry.register(Arc::new(ProposeChoiceTool));

    let skill_index = skill_provider.read().await.skill_index();

    (
        registry,
        sync,
        SkillIndex(skill_index),
        skill_provider,
        runner,
    )
}

pub async fn run_mcp(cmd: crate::McpCommand) -> Result<()> {
    use flashmind_tools::mcp::{McpConfigProvider, McpDiskConfig, McpRegistry, McpServerConfig};

    let config_provider = McpDiskConfig::new(mcp_config_dir()?);
    let registry = McpRegistry::new(config_provider.clone(), None);

    match cmd {
        crate::McpCommand::Add {
            name,
            command,
            args,
            url,
            env,
        } => {
            if command.is_none() && url.is_none() {
                bail!("must provide either --command or --url");
            }
            let server_config = McpServerConfig {
                name: name.clone(),
                command,
                args,
                url,
                env: env.into_iter().collect(),
                ..Default::default()
            };
            registry.add(server_config.clone()).await?;
            match registry.connect(server_config).await {
                Ok(tools) => {
                    print_mcp_connected(&name, &tools);
                }
                Err(e)
                    if e.downcast_ref::<flashmind_tools::mcp::McpAuthRequired>()
                        .is_some() =>
                {
                    println!("Server '{name}' requires authentication, starting OAuth flow…");
                    match run_mcp_oauth(&registry, &name).await {
                        Ok(tools) => print_mcp_connected(&name, &tools),
                        Err(auth_err) => {
                            println!("Authentication failed: {auth_err}");
                            println!("You can retry with: flsh mcp auth {name}");
                        }
                    }
                }
                Err(e) => {
                    println!("Saved '{name}' but failed to connect: {e}");
                    println!("It will retry on next launch.");
                }
            }
        }
        crate::McpCommand::Auth { name } => {
            let config = config_provider
                .list_configs()
                .await?
                .into_iter()
                .find(|c| c.name == name)
                .ok_or_else(|| anyhow::anyhow!("MCP server '{name}' not found"))?;
            registry.add(config).await?;
            match run_mcp_oauth(&registry, &name).await {
                Ok(tools) => print_mcp_connected(&name, &tools),
                Err(e) => bail!("Authentication failed for '{name}': {e}"),
            }
        }
        crate::McpCommand::Remove { name } => {
            config_provider.delete_config(&name).await?;
            println!("Removed '{name}'");
        }
        crate::McpCommand::Permissions { name } => {
            run_mcp_permissions(&config_provider, &name).await?;
        }
        crate::McpCommand::Import { claude } => {
            if claude {
                run_mcp_import_claude(&config_provider).await?;
            } else {
                bail!("specify a source to import from (e.g. --claude)");
            }
        }
        crate::McpCommand::List => {
            let configs = config_provider.list_configs().await?;
            if configs.is_empty() {
                println!("No MCP servers configured.");
            } else {
                for cfg in &configs {
                    let transport = if cfg.command.is_some() {
                        format!("stdio: {}", cfg.command.as_deref().unwrap_or("?"))
                    } else if let Some(url) = &cfg.url {
                        format!("http: {url}")
                    } else {
                        "unknown".into()
                    };
                    let tool_count = cfg.cached_tools.len();
                    println!(
                        "  {:<20} {transport}  ({tool_count} cached tools)",
                        cfg.name
                    );
                }
            }
        }
    }

    registry.shutdown_all().await;
    Ok(())
}

async fn run_mcp_import_claude(
    config_provider: &flashmind_tools::mcp::McpDiskConfig,
) -> Result<()> {
    use flashmind_tools::mcp::McpConfigProvider;
    use flashmind_tui::Tui;
    use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use std::collections::HashMap;

    #[derive(serde::Deserialize)]
    struct ClaudeDesktopConfig {
        #[serde(default, rename = "mcpServers")]
        mcp_servers: HashMap<String, ClaudeMcpEntry>,
    }

    #[derive(serde::Deserialize)]
    struct ClaudeCodeConfig {
        #[serde(default)]
        projects: HashMap<String, ClaudeCodeProject>,
    }

    #[derive(serde::Deserialize)]
    struct ClaudeCodeProject {
        #[serde(default, rename = "mcpServers")]
        mcp_servers: HashMap<String, ClaudeMcpEntry>,
    }

    #[derive(serde::Deserialize)]
    struct ClaudeSettingsConfig {
        #[serde(default, rename = "mcpServers")]
        mcp_servers: HashMap<String, ClaudeMcpEntry>,
    }

    #[derive(serde::Deserialize, Clone)]
    struct ClaudeMcpEntry {
        command: Option<String>,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        url: Option<String>,
    }

    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;

    // (name, entry, source label)
    let mut discovered: Vec<(String, ClaudeMcpEntry, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut insert = |name: String, entry: ClaudeMcpEntry, source: String| {
        if seen.insert(name.clone()) {
            discovered.push((name, entry, source));
        }
    };

    // 1. Claude Desktop config
    let desktop_path = home
        .join("Library")
        .join("Application Support")
        .join("Claude")
        .join("claude_desktop_config.json");
    if let Ok(content) = std::fs::read_to_string(&desktop_path)
        && let Ok(config) = serde_json::from_str::<ClaudeDesktopConfig>(&content)
    {
        for (name, entry) in config.mcp_servers {
            insert(name, entry, "Desktop".into());
        }
    }

    // 2. Claude Code project configs (~/.claude.json)
    let code_path = home.join(".claude.json");
    if let Ok(content) = std::fs::read_to_string(&code_path)
        && let Ok(config) = serde_json::from_str::<ClaudeCodeConfig>(&content)
    {
        for (project_path, project) in config.projects {
            for (name, entry) in project.mcp_servers {
                let label = if project_path == home.display().to_string() {
                    "Code (global)".into()
                } else {
                    format!(
                        "Code ({})",
                        std::path::Path::new(&project_path)
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or(project_path.clone())
                    )
                };
                insert(name, entry, label);
            }
        }
    }

    // 3. Claude Code settings files (~/.claude/settings.json, settings.local.json)
    for settings_name in ["settings.json", "settings.local.json"] {
        let settings_path = home.join(".claude").join(settings_name);
        if let Ok(content) = std::fs::read_to_string(&settings_path)
            && let Ok(config) = serde_json::from_str::<ClaudeSettingsConfig>(&content)
        {
            for (name, entry) in config.mcp_servers {
                insert(name, entry, format!("Code ({settings_name})"));
            }
        }
    }

    if discovered.is_empty() {
        println!("No MCP servers found in Claude configuration.");
        return Ok(());
    }

    // Filter out entries with no transport and mark existing ones
    let existing: std::collections::HashSet<String> = config_provider
        .list_configs()
        .await?
        .into_iter()
        .map(|c| c.name)
        .collect();

    // (name, entry, source, importable, selected)
    let mut items: Vec<(String, ClaudeMcpEntry, String, bool, bool)> = discovered
        .into_iter()
        .map(|(name, entry, source)| {
            let importable =
                !existing.contains(&name) && (entry.command.is_some() || entry.url.is_some());
            (name, entry, source, importable, importable)
        })
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));

    // Interactive picker
    let mut tui = Tui::new();
    let _raw = tui.raw_mode()?;
    let mut drawn: u16 = 0;
    let mut selected: usize = 0;

    loop {
        let mut lines: Vec<Line<'_>> = Vec::new();
        lines.push(Line::from(Span::styled(
            "  Import MCP servers from Claude",
            Style::default().add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(
            "  Space: toggle  |  Enter: import  |  a: all  |  n: none  |  Esc: cancel",
        ));
        lines.push(Line::from(""));

        for (i, (name, entry, source, importable, checked)) in items.iter().enumerate() {
            let transport = entry
                .command
                .as_deref()
                .unwrap_or(entry.url.as_deref().unwrap_or("?"));

            if !importable {
                let reason = if existing.contains(name) {
                    "exists"
                } else {
                    "no transport"
                };
                let label = format!("    -  {name:<20} {source:<18} ({reason})");
                lines.push(Line::from(Span::styled(
                    label,
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                let marker = if *checked { "[x]" } else { "[ ]" };
                let cursor = if i == selected { ">" } else { " " };
                let style = if i == selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                let label = format!("  {cursor} {marker} {name:<20} {source:<18} ({transport})");
                lines.push(Line::from(Span::styled(label, style)));
            }
        }

        lines.push(Line::from(""));
        let import_count = items.iter().filter(|i| i.4).count();
        lines.push(Line::from(format!(
            "  {import_count} server{} selected",
            if import_count == 1 { "" } else { "s" }
        )));

        drawn = tui.redraw_lines(&lines, drawn)?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if selected > 0 {
                        selected -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') if selected + 1 < items.len() => {
                    selected += 1;
                }
                KeyCode::Char(' ') => {
                    if items[selected].3 {
                        items[selected].4 = !items[selected].4;
                    }
                }
                KeyCode::Char('a') => {
                    for item in &mut items {
                        if item.3 {
                            item.4 = true;
                        }
                    }
                }
                KeyCode::Char('n') => {
                    for item in &mut items {
                        item.4 = false;
                    }
                }
                KeyCode::Enter => {
                    tui.erase(drawn)?;
                    break;
                }
                KeyCode::Esc | KeyCode::Char('q') => {
                    tui.erase(drawn)?;
                    println!("Cancelled.");
                    return Ok(());
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    tui.erase(drawn)?;
                    println!("Cancelled.");
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    let mut imported = 0;
    for (name, entry, _, _, checked) in &items {
        if !checked {
            continue;
        }
        let server_config = flashmind_tools::mcp::McpServerConfig {
            name: name.clone(),
            command: entry.command.clone(),
            args: entry.args.clone(),
            url: entry.url.clone(),
            env: entry.env.clone(),
            ..Default::default()
        };
        config_provider.save_config(&server_config).await?;
        let transport = entry
            .command
            .as_deref()
            .unwrap_or(entry.url.as_deref().unwrap_or("?"));
        println!("  {name:<20} imported ({transport})");
        imported += 1;
    }

    if imported == 0 {
        println!("Nothing to import.");
    } else {
        println!("\n{imported} server{} imported.", if imported == 1 { "" } else { "s" });
    }
    Ok(())
}

/// Interactive toggle UI for managing per-server tool approval requirements.
///
/// Displays a list of cached tools with checkbox toggles. Space toggles
/// approval requirement for the selected tool, Enter saves changes, Esc cancels.
async fn run_mcp_permissions(
    config_provider: &flashmind_tools::mcp::McpDiskConfig,
    server_name: &str,
) -> Result<()> {
    use flashmind_tools::mcp::McpConfigProvider;
    use flashmind_tui::Tui;
    use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    let configs = config_provider.list_configs().await?;
    let mut config = configs
        .into_iter()
        .find(|c| c.name == server_name)
        .ok_or_else(|| anyhow::anyhow!("MCP server '{server_name}' not found"))?;

    if config.cached_tools.is_empty() {
        bail!(
            "No cached tools for '{server_name}'. Connect first with: flsh mcp add {server_name} ..."
        );
    }

    let restricted_set: std::collections::HashSet<String> =
        config.restricted_tools.iter().cloned().collect();

    // Build toggle state: Vec<(tool_name, description, requires_approval)>
    let mut tool_states: Vec<(String, String, bool)> = config
        .cached_tools
        .iter()
        .map(|t| {
            let restricted = restricted_set.contains(&t.name);
            let desc = t.description.clone().unwrap_or_default();
            (t.name.clone(), desc, restricted)
        })
        .collect();
    tool_states.sort_by(|a, b| a.0.cmp(&b.0));

    let mut tui = Tui::new();
    let _raw = tui.raw_mode()?;
    let mut drawn: u16 = 0;
    let mut selected: usize = 0;

    loop {
        let mut lines: Vec<Line<'_>> = Vec::new();
        lines.push(Line::from(Span::styled(
            format!("  Tool permissions for '{server_name}'"),
            Style::default().add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(
            "  Space: toggle  |  Enter: save  |  Esc/q: cancel",
        ));
        lines.push(Line::from(""));

        for (i, (name, desc, restricted)) in tool_states.iter().enumerate() {
            let marker = if *restricted { "[x]" } else { "[ ]" };
            let cursor = if i == selected { ">" } else { " " };
            let style = if i == selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let label = if desc.is_empty() {
                format!("{cursor} {marker} {name}")
            } else {
                let truncated: String = desc.chars().take(50).collect();
                format!("{cursor} {marker} {name} — {truncated}")
            };
            lines.push(Line::from(Span::styled(label, style)));
        }

        lines.push(Line::from(""));
        let restricted_count = tool_states.iter().filter(|(_, _, r)| *r).count();
        lines.push(Line::from(format!(
            "  {restricted_count}/{} tools require approval",
            tool_states.len()
        )));

        drawn = tui.redraw_lines(&lines, drawn)?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    selected = selected.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') if selected + 1 < tool_states.len() => {
                    selected += 1;
                }
                KeyCode::Char(' ') => {
                    tool_states[selected].2 = !tool_states[selected].2;
                }
                KeyCode::Enter => {
                    tui.erase(drawn)?;
                    break;
                }
                KeyCode::Esc | KeyCode::Char('q') => {
                    tui.erase(drawn)?;
                    println!("Cancelled.");
                    return Ok(());
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    tui.erase(drawn)?;
                    println!("Cancelled.");
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    // Persist updated restricted_tools
    config.restricted_tools = tool_states
        .iter()
        .filter(|(_, _, restricted)| *restricted)
        .map(|(name, _, _)| name.clone())
        .collect();

    config_provider.save_config(&config).await?;

    let count = config.restricted_tools.len();
    if count == 0 {
        println!("All tools for '{server_name}' are allowed (no approval required).");
    } else {
        println!(
            "Updated '{server_name}': {} tool(s) require approval: {}",
            count,
            config.restricted_tools.join(", ")
        );
    }

    Ok(())
}

fn print_mcp_connected(name: &str, tools: &[flashmind_tools::mcp::McpToolDef]) {
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    println!(
        "Connected to '{name}' — {} tools: {}",
        tools.len(),
        names.join(", ")
    );
}

async fn run_mcp_oauth(
    registry: &flashmind_tools::mcp::McpRegistry,
    server_name: &str,
) -> Result<Vec<flashmind_tools::mcp::McpToolDef>> {
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let redirect_uri = format!("http://localhost:{port}/callback");

    let auth_url = registry.start_auth(server_name, &redirect_uri).await?;

    println!("Opening browser for authentication…");
    println!("{auth_url}");
    let _ = std::process::Command::new("open").arg(&auth_url).spawn();

    let (mut stream, _) = listener.accept().await?;
    let mut buf = vec![0u8; 4096];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("");

    let full_url = format!("http://localhost:{port}{path}");
    let parsed = url::Url::parse(&full_url)?;
    let mut code = None;
    let mut state = None;
    for (k, v) in parsed.query_pairs() {
        match k.as_ref() {
            "code" => code = Some(v.into_owned()),
            "state" => state = Some(v.into_owned()),
            _ => {}
        }
    }

    let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n\
        <html><body><h2>Authenticated! You can close this tab.</h2></body></html>";
    tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes()).await?;

    let code = code.ok_or_else(|| anyhow::anyhow!("no 'code' in OAuth callback"))?;
    let state = state.ok_or_else(|| anyhow::anyhow!("no 'state' in OAuth callback"))?;

    registry.complete_auth(server_name, &code, &state).await?;

    let tools = registry
        .current_mcp_tools()
        .remove(server_name)
        .unwrap_or_default();
    Ok(tools)
}
