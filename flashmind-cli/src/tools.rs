use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock as SyncRwLock};

use flashmind_core::subagent::AgentManager;
use flashmind_skills::{DiskSkillProvider, SkillProvider};
use flashmind_tools::protected::ProtectedPaths;
use flashmind_tools::{ToolBuilder, ToolSync};
use flashmind_types::tool::{CommandAllowList, ToolRegistry};
use flashmind_types::{AgentLlmConfig, LlmProvider, Model};
use tokio::sync::RwLock;

use crate::config::Config;
use crate::file_tools::register_file_tools;
use crate::paths::config_dir;

struct CommandPolicy {
    patterns: SyncRwLock<Vec<glob::Pattern>>,
    skip_permissions: bool,
}

impl CommandPolicy {
    fn from_config(patterns: Option<&Vec<String>>, skip_permissions: bool) -> Self {
        let patterns = patterns
            .into_iter()
            .flatten()
            .filter_map(|pattern| match glob::Pattern::new(pattern) {
                Ok(pattern) => Some(pattern),
                Err(error) => {
                    tracing::warn!(%error, "ignored invalid command allowlist pattern");
                    None
                }
            })
            .collect();
        Self {
            patterns: SyncRwLock::new(patterns),
            skip_permissions,
        }
    }
}

impl CommandAllowList for CommandPolicy {
    fn is_allowed(&self, command: &str) -> bool {
        if self.skip_permissions {
            return true;
        }
        self.patterns
            .read()
            .map(|patterns| patterns.iter().any(|pattern| pattern.matches(command)))
            .unwrap_or(false)
    }

    fn add_session_pattern(&self, pattern: &str) {
        let Ok(pattern) = glob::Pattern::new(pattern) else {
            tracing::warn!("ignored invalid command approval pattern");
            return;
        };
        if let Ok(mut patterns) = self.patterns.write() {
            patterns.push(pattern);
        }
    }
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

    let mut roots = vec![std::env::current_dir().unwrap_or_default()];
    if let Some(home) = dirs::home_dir() {
        roots.push(home);
    }
    for root in roots {
        for relative in [".agents/skills", ".claude/skills", ".claude/commands"] {
            let path = root.join(relative);
            if path.is_dir() && !dirs.contains(&path) {
                dirs.push(path);
            }
        }
    }

    dirs
}

pub struct ToolSetup {
    pub registry: ToolRegistry,
    pub sync: ToolSync,
    pub skill_index: String,
    pub skill_provider: Arc<RwLock<DiskSkillProvider>>,
}

pub fn skill_instructions(index: &str) -> String {
    if index.is_empty() {
        return String::new();
    }
    format!(
        "\n\n## Skills\n\nUse a listed skill when its description matches the task. Read its source file with `read` before using it. Follow its instructions and use `exec` for scripts, setting the working directory to the skill's directory when required. The user can activate a skill with `/{{skill}}`.\n{index}"
    )
}

async fn build_coding_tools(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    skip_permissions: bool,
    instructions: String,
) -> (ToolRegistry, ToolSync) {
    let protected = Arc::new(ProtectedPaths::new(Path::new("/")));
    let vision_model: Option<Model> = config.vision_model.as_deref().and_then(|s| s.parse().ok());
    let manager = Arc::new(AgentManager::new(4, 2));
    let mut builder = ToolBuilder::new()
        .file_ops(vision_model.clone(), &protected)
        .bash(
            vec![],
            &protected,
            vec![],
            Some(Arc::new(CommandPolicy::from_config(
                config.command_allowlist.as_ref(),
                skip_permissions,
            ))),
        )
        .subagents_with_instructions(manager, provider, Some(llm), instructions);

    if let Some(ref vm) = vision_model
        && let Ok(provider) = crate::provider::build_provider(&vm.provider, config)
    {
        let mut providers = HashMap::new();
        providers.insert(vm.provider.clone(), provider);
        builder = builder.with_providers(Arc::new(providers));
    }

    let (mut registry, sync) = builder.build_with_sync().await;
    register_file_tools(&mut registry, &protected);
    sync.sync(&mut registry).await;
    (registry, sync)
}

pub async fn build_tools(
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    llm: AgentLlmConfig,
    skip_permissions: bool,
    allowed: Option<&[String]>,
) -> ToolSetup {
    let skill_provider = match DiskSkillProvider::discover(compute_skill_dirs(config)).await {
        Ok(provider) => provider,
        Err(error) => {
            tracing::warn!("skill discovery failed: {error}");
            DiskSkillProvider::discover(vec![])
                .await
                .expect("Skill discovery with no directories cannot fail.")
        }
    };
    let skill_index = skill_index(&skill_provider);
    let (mut registry, sync) = build_coding_tools(
        config,
        provider,
        llm,
        skip_permissions,
        skill_instructions(&skill_index),
    )
    .await;
    if let Some(allowed) = allowed
        && !allowed.iter().any(|name| name == "all")
    {
        let names: Vec<String> = allowed
            .iter()
            .filter_map(|name| registry.get(name).map(|tool| tool.name().to_string()))
            .collect();
        registry.retain(|name| names.iter().any(|allowed| allowed == name));
    }
    sync.sync(&mut registry).await;

    ToolSetup {
        registry,
        sync,
        skill_index,
        skill_provider: Arc::new(RwLock::new(skill_provider)),
    }
}

pub fn skill_index(provider: &impl SkillProvider) -> String {
    provider
        .list()
        .iter()
        .map(|skill| {
            format!(
                "- /{}: {} (source: `{}`)",
                skill.meta.name,
                skill
                    .meta
                    .description
                    .as_deref()
                    .unwrap_or("no description"),
                skill.source_path.display(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod permission_tests {
    use super::*;
    use flashmind_llm::OllamaProvider;

    #[tokio::test]
    async fn coding_toolkit_contains_only_files_shell_and_subagents() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let provider = Arc::new(OllamaProvider::new(None, None).unwrap());
        let llm = AgentLlmConfig::new("ollama:example".parse().unwrap());
        let (registry, _) =
            build_coding_tools(&Config::default(), provider, llm, false, String::new()).await;
        assert_eq!(
            registry.list(),
            vec![
                "agent_status",
                "agent_terminate",
                "agent_wait",
                "delegate",
                "edit",
                "exec",
                "find",
                "image_read",
                "process",
                "read",
                "write",
            ],
        );
    }

    #[test]
    fn skip_permissions_bypasses_shell_allowlist_only_when_enabled() {
        assert!(!CommandPolicy::from_config(None, false).is_allowed("echo example"));
        assert!(CommandPolicy::from_config(None, true).is_allowed("echo example"));
        let patterns = vec!["cargo test*".to_string()];
        let policy = CommandPolicy::from_config(Some(&patterns), false);
        assert!(policy.is_allowed("cargo test -p example"));
        assert!(!policy.is_allowed("echo example"));
    }
}
