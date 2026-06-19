use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use flashmind_skills::{
    DiskSkillProvider, SkillListTool, SkillLoadTool, SkillProvider, SkillRunTool, SkillRunner,
    SkillSaveTool,
};
use flashmind_tools::ToolBuilder;
use flashmind_tools::protected::ProtectedPaths;
use tokio::sync::RwLock;

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
) -> (
    flashmind_types::ToolRegistry,
    flashmind_tools::tool_sync::ToolSync,
    SkillIndex,
) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let protected = Arc::new(ProtectedPaths::new(&cwd));

    let mcp_config_dir = config_dir()
        .map(|d| d.join("mcp"))
        .unwrap_or_else(|_| PathBuf::from(".flashmind/mcp"));
    let _ = std::fs::create_dir_all(&mcp_config_dir);
    let mcp_provider = flashmind_tools::mcp::McpDiskConfig::new(mcp_config_dir);

    let (mut registry, sync) = ToolBuilder::new()
        .core(None, &protected)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .mcp(mcp_provider, None)
        .build_with_sync()
        .await;

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
        runner,
    }));
    registry.register(Arc::new(SkillSaveTool {
        provider: provider.clone(),
    }));

    let skill_index = provider.read().await.skill_index();

    (registry, sync, SkillIndex(skill_index))
}

pub async fn build_tools_full(
    config: &Config,
) -> (
    flashmind_types::ToolRegistry,
    flashmind_tools::tool_sync::ToolSync,
) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let protected = Arc::new(ProtectedPaths::new(&cwd));

    let mcp_config_dir = config_dir()
        .map(|d| d.join("mcp"))
        .unwrap_or_else(|_| PathBuf::from(".flashmind/mcp"));
    let _ = std::fs::create_dir_all(&mcp_config_dir);
    let mcp_provider = flashmind_tools::mcp::McpDiskConfig::new(mcp_config_dir);

    ToolBuilder::new()
        .file_ops(None, &protected)
        .bash(vec![], &protected, vec![], None)
        .search(
            config.brave_api_key.clone(),
            config.firecrawl_api_key.clone(),
        )
        .time()
        .mcp(mcp_provider, None)
        .build_with_sync()
        .await
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
            config_provider.save_config(&server_config).await?;
            match registry.connect(server_config).await {
                Ok(tools) => {
                    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                    println!(
                        "Connected to '{name}' — {} tools: {}",
                        tools.len(),
                        names.join(", ")
                    );
                }
                Err(e) => {
                    println!("Saved '{name}' but failed to connect: {e}");
                    println!("It will retry on next launch.");
                }
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
