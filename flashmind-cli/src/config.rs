//! Configuration loaded from `~/.flashmind/config.toml`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use flashmind_core::AgentManager;
use flashmind_llm::{AnthropicProvider, OllamaProvider, OpenAiProvider, OpenRouterProvider};
use flashmind_skills::{SkillRegistry, SkillRunner};
use flashmind_tools::ToolBuilder;
use flashmind_tools::mcp::McpDiskConfig;
use flashmind_tools::protected::ProtectedPaths;
use flashmind_tools::tool_sync::ToolSync;
use flashmind_types::model::{Model, Provider, ReasoningLevel, SamplingParams};
use flashmind_types::tool::ToolRegistry;
use flashmind_types::{AgentLlmConfig, InjectQueue, LlmProvider};

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub memory: Option<MemoryConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub temperature: Option<Decimal>,
    #[serde(default)]
    pub top_p: Option<Decimal>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<Decimal>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: Provider,
    #[serde(default)]
    pub api_key: Option<String>,
    pub model: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsConfig {
    #[serde(default)]
    pub forbidden: Vec<ForbiddenConfig>,
    #[serde(default)]
    pub brave_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenConfig {
    pub command: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub max_session_age_days: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default)]
    pub embedding: Option<flashmind_memory::embeddings::EmbeddingProviderConfig>,
}

/// Everything produced by [`Config::build_tools`].
#[allow(dead_code)]
pub struct ToolSet {
    pub tools: ToolRegistry,
    pub tool_sync: ToolSync,
    pub inject_queue: Arc<InjectQueue>,
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

impl Config {
    pub fn base_dir() -> PathBuf {
        dirs::home_dir()
            .expect("no home directory")
            .join(".flashmind")
    }

    pub fn config_path() -> PathBuf {
        Self::base_dir().join("config.toml")
    }

    pub fn db_path() -> PathBuf {
        Self::base_dir().join("flashmind.db")
    }

    pub fn sessions_dir() -> PathBuf {
        Self::base_dir().join("sessions")
    }

    pub fn session_display_path(key: &str) -> PathBuf {
        Self::sessions_dir().join(format!("{key}.jsonl"))
    }

    pub fn log_dir() -> PathBuf {
        Self::base_dir().join("logs")
    }

    pub fn soul_path() -> PathBuf {
        Self::base_dir().join("SOUL.md")
    }

    pub fn skills_dir() -> PathBuf {
        Self::base_dir().join("skills")
    }

    pub fn mcp_dir() -> PathBuf {
        Self::base_dir().join("mcp")
    }
}

// ---------------------------------------------------------------------------
// Load / Init
// ---------------------------------------------------------------------------

impl Config {
    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn init() -> Result<PathBuf> {
        let base = Self::base_dir();
        std::fs::create_dir_all(&base)?;
        std::fs::create_dir_all(Self::sessions_dir())?;
        std::fs::create_dir_all(Self::log_dir())?;
        std::fs::create_dir_all(Self::skills_dir())?;
        std::fs::create_dir_all(Self::mcp_dir())?;

        let config_path = Self::config_path();
        if !config_path.exists() {
            std::fs::write(&config_path, DEFAULT_CONFIG)?;
            tracing::info!("created {}", config_path.display());
        }
        if !Self::soul_path().exists() {
            std::fs::write(Self::soul_path(), DEFAULT_SOUL)?;
        }
        Ok(config_path)
    }
}

// ---------------------------------------------------------------------------
// Provider / Model / Tools
// ---------------------------------------------------------------------------

impl Config {
    pub fn provider_config_for(&self, provider: &Provider) -> Option<&ProviderConfig> {
        self.llm.providers.iter().find(|p| p.name == *provider)
    }

    pub fn active_provider_config(&self) -> Result<&ProviderConfig> {
        self.llm
            .providers
            .first()
            .context("no LLM providers configured — add [[llm.providers]] to config.toml")
    }

    pub fn build_provider_for(&self, provider: &Provider) -> Result<Arc<dyn LlmProvider>> {
        let pc = self
            .provider_config_for(provider)
            .with_context(|| format!("no config for provider {provider} — add [[llm.providers]] with name = \"{provider}\""))?;
        self.make_provider(pc)
    }

    fn make_provider(&self, pc: &ProviderConfig) -> Result<Arc<dyn LlmProvider>> {
        let provider: Arc<dyn LlmProvider> = match pc.name {
            Provider::Ollama => Arc::new(OllamaProvider::new(pc.url.clone(), pc.num_ctx)),
            Provider::OpenRouter => {
                let key = pc.api_key.clone().context("openrouter requires api_key")?;
                Arc::new(OpenRouterProvider::new(key))
            }
            Provider::Anthropic => {
                let key = pc.api_key.clone().context("anthropic requires api_key")?;
                let limiter = Arc::new(
                    ratelimit::Ratelimiter::builder(50)
                        .max_tokens(50)
                        .initial_available(50)
                        .build()
                        .expect("rate limiter"),
                );
                Arc::new(AnthropicProvider::new(key, limiter))
            }
            Provider::OpenAi => {
                let key = pc.api_key.clone();
                Arc::new(OpenAiProvider::new(
                    pc.url.clone(),
                    key,
                    Default::default(),
                    false,
                    None,
                ))
            }
            Provider::Connect => bail!("connect provider not supported in flashmind-cli"),
        };
        Ok(provider)
    }

    pub fn build_model(&self, model_override: Option<&Model>) -> Result<Model> {
        if let Some(m) = model_override {
            return Ok(m.clone());
        }
        let pc = self.active_provider_config()?;
        let full = format!("{}:{}", pc.name, pc.model);
        full.parse()
            .with_context(|| format!("invalid model: {full}"))
    }

    pub fn build_llm_config(&self, model_override: Option<&Model>) -> Result<AgentLlmConfig> {
        let model = self.build_model(model_override)?;
        let reasoning = match self.llm.reasoning.as_deref() {
            Some("on") => ReasoningLevel::On,
            _ => ReasoningLevel::Off,
        };
        let sampling = SamplingParams {
            temperature: self.llm.temperature,
            top_p: self.llm.top_p,
            top_k: self.llm.top_k,
            min_p: self.llm.min_p,
            ..Default::default()
        };
        Ok(AgentLlmConfig {
            model,
            max_tokens: self.llm.max_tokens,
            reasoning,
            sampling,
        })
    }

    pub async fn build_tools(
        &self,
        provider: Arc<dyn LlmProvider>,
        llm: &AgentLlmConfig,
    ) -> Result<ToolSet> {
        let protected = Arc::new(ProtectedPaths::new(&Self::base_dir()));
        let secrets = self.collect_secrets();

        // Subagent manager
        let inject_queue = InjectQueue::new();
        let manager = Arc::new(AgentManager::new(inject_queue.clone(), 8, 3));

        // MCP config
        let mcp_provider = McpDiskConfig::new(Self::mcp_dir());

        let builder = ToolBuilder::new()
            .file_ops(None, &protected)
            .bash(secrets, &protected)
            .search(self.tools.brave_api_key.clone(), None)
            .http()
            .time()
            .sqlite()
            .json()
            .models()
            .subagents(manager, provider.clone(), Some(llm.clone()))
            .mcp(mcp_provider, None);

        // Skills
        let skill_registry = Arc::new(RwLock::new(SkillRegistry::new(vec![Self::skills_dir()])));
        let skill_runner = Arc::new(SkillRunner::new(Duration::from_secs(300)));

        let (mut tools, tool_sync) = builder.build_with_sync().await;

        // Register skills tools
        tools.register(Arc::new(flashmind_skills::SkillListTool {
            registry: skill_registry.clone(),
        }));
        tools.register(Arc::new(flashmind_skills::SkillLoadTool {
            registry: skill_registry.clone(),
        }));
        tools.register(Arc::new(flashmind_skills::SkillRunTool {
            registry: skill_registry.clone(),
            runner: skill_runner,
        }));
        tools.register(Arc::new(flashmind_skills::SkillInstallTool {
            registry: skill_registry,
        }));

        // Memory tools — auto-detect embedding provider from config or LLM provider
        if let Some(embedder) = self.build_embedder() {
            let dim = embedder.dimensions();
            let db = flashmind_memory::DbStore::connect(&Self::db_path(), dim)
                .await
                .context("opening memory database")?;
            crate::memory::register_tools(&mut tools, db, embedder);
        }

        Ok(ToolSet {
            tools,
            tool_sync,
            inject_queue,
        })
    }

    fn build_embedder(&self) -> Option<std::sync::Arc<dyn flashmind_memory::EmbeddingProvider>> {
        let mem = self.memory.as_ref()?;
        let emb_config = mem.embedding.as_ref()?;
        let fallback_key = self.llm.providers.iter().find_map(|p| p.api_key.as_deref());
        match flashmind_memory::embeddings::create_embedding_provider(emb_config, fallback_key) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!("failed to create embedding provider: {e}");
                None
            }
        }
    }

    pub fn system_prompt(&self) -> String {
        let mut prompt = if let Some(ref p) = self.agent.system_prompt {
            p.clone()
        } else {
            let soul = Self::soul_path();
            if soul.exists()
                && let Ok(text) = std::fs::read_to_string(&soul)
                && !text.trim().is_empty()
            {
                text
            } else {
                DEFAULT_SOUL.to_string()
            }
        };

        // Project instructions from CLAUDE.md / AGENTS.md
        let cwd = std::env::current_dir().unwrap_or_default();
        let project = build_project_instructions(&cwd);
        if !project.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&project);
        }

        // Git context
        let git = build_git_context(&cwd);
        if !git.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&git);
        }

        // Memory instructions (when embedder is available)
        if self.build_embedder().is_some() {
            prompt.push_str("\n\n");
            prompt.push_str(flashmind_prompts::MEMORY_INSTRUCTIONS);
        }

        prompt
    }

    fn collect_secrets(&self) -> Vec<String> {
        let mut secrets: Vec<String> = self
            .llm
            .providers
            .iter()
            .filter_map(|p| p.api_key.clone())
            .collect();
        if let Some(ref key) = self.tools.brave_api_key {
            secrets.push(key.clone());
        }
        secrets
    }
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Project instructions
// ---------------------------------------------------------------------------

fn build_project_instructions(workspace: &std::path::Path) -> String {
    let names = ["CLAUDE.md", "AGENTS.md"];
    let found: Vec<&str> = names
        .iter()
        .filter(|n| workspace.join(n).is_file())
        .copied()
        .collect();

    if found.is_empty() {
        return String::new();
    }

    let mut out = String::from("## Project Instructions\n\nThe following project files exist:\n\n");
    for f in &found {
        out.push_str(&format!("- `{f}` — read this on your first turn\n"));
    }
    out.push_str(
        "\n**You are a software developer working on this project.** \
         When the user asks you to change, add, fix, or configure anything — \
         including tools, limits, features, or behavior — they mean modify the \
         source code. Do not confuse yourself with the software being built.\n",
    );
    out
}

// ---------------------------------------------------------------------------
// Git context
// ---------------------------------------------------------------------------

fn build_git_context(cwd: &std::path::Path) -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(cwd)
        .output();
    if !matches!(output, Ok(ref o) if o.status.success()) {
        return String::new();
    }

    let mut parts = Vec::new();

    if let Ok(o) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
    {
        let root = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if let Some(name) = std::path::Path::new(&root).file_name() {
            parts.push(format!("Repository: {}", name.to_string_lossy()));
        }
    }

    if let Ok(o) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(cwd)
        .output()
    {
        let branch = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !branch.is_empty() {
            parts.push(format!("Branch: {branch}"));
        }
    }

    if let Ok(o) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
    {
        let status = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !status.is_empty() {
            let lines: Vec<&str> = status.lines().collect();
            parts.push(format!("{} uncommitted change(s)", lines.len()));
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    format!("## Git Context\n\n{}", parts.join("\n"))
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: &str = r#"# Flashmind CLI Configuration
# Documentation: https://github.com/flashmind-labs/flashmind

[llm]
# temperature = 0.6
# max_tokens = 4096
# reasoning = "off"

# Ollama — run local models (no API key needed)
[[llm.providers]]
name = "ollama"
model = "llama3.2"
# url = "http://localhost:11434"
# num_ctx = 128000

# OpenRouter — routes to 100+ models
# [[llm.providers]]
# name = "openrouter"
# api_key = "sk-or-v1-YOUR-KEY-HERE"
# model = "anthropic/claude-sonnet-4"

# Anthropic — direct API access
# [[llm.providers]]
# name = "anthropic"
# api_key = "sk-ant-YOUR-KEY-HERE"
# model = "claude-sonnet-4-20250514"

# OpenAI — OpenAI API or compatible endpoints
# [[llm.providers]]
# name = "openai"
# api_key = "sk-YOUR-KEY"
# model = "gpt-4o"

[tools]
# brave_api_key = "BSA..."
# [[tools.forbidden]]
# command = "rm -rf /"
# reason = "Dangerous"

[agent]
# system_prompt = "You are a helpful assistant."
# max_session_age_days = 30

# Memory — requires explicit embedding config to enable
# [memory.embedding]
# provider = "ollama"
# model = "nomic-embed-text"
# url = "http://localhost:11434"

# Skills are loaded from ~/.flashmind/skills/
# MCP server configs are stored in ~/.flashmind/mcp/
"#;

const DEFAULT_SOUL: &str = "\
You are Flash — a technical AI agent with full system access. \
Your prime directive is continuous learning, context protection, and result-oriented execution.
- Be concise. Lead with results.
- Default to execution and action over discussion.
- Use tools to explore, verify, and act — don't guess when you can check.
- Do not use emojis.
";
