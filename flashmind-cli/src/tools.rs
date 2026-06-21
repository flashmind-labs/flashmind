use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use flashmind_core::subagent::AgentManager;
use flashmind_skills::{
    DiskSkillProvider, SkillInstallTool, SkillListTool, SkillLoadTool, SkillProvider, SkillRunTool,
    SkillRunner, SkillSaveTool,
};
use flashmind_tools::ToolBuilder;
use flashmind_tools::protected::ProtectedPaths;
use tokio::sync::RwLock;

use flashmind_types::{AgentLlmConfig, LlmProvider, Model};

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

    let vision_model: Option<Model> = config
        .vision_model
        .as_deref()
        .and_then(|s| s.parse().ok());

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

    if let Some(ref vm) = vision_model {
        if let Ok(p) = crate::provider::build_provider(vm, config) {
            let mut providers = std::collections::HashMap::new();
            providers.insert(vm.provider, p);
            builder = builder.with_providers(std::sync::Arc::new(providers));
        }
    }

    let (mut registry, sync) = builder.build_with_sync().await;

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

    let vision_model: Option<Model> = config
        .vision_model
        .as_deref()
        .and_then(|s| s.parse().ok());

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

    if let Some(ref vm) = vision_model {
        if let Ok(p) = crate::provider::build_provider(vm, config) {
            let mut providers = std::collections::HashMap::new();
            providers.insert(vm.provider, p);
            builder = builder.with_providers(std::sync::Arc::new(providers));
        }
    }

    let (registry, sync) = builder.build_with_sync().await;

    let skill_index = skill_provider.read().await.skill_index();

    (registry, sync, SkillIndex(skill_index), skill_provider, runner)
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
