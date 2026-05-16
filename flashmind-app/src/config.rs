//! Configuration loaded from `~/.flashmind/config.toml`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

/// Top-level application configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {
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
    pub name: flashmind_types::model::Provider,
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
    pub allowed: Vec<String>,
    #[serde(default)]
    pub brave_api_key: Option<String>,
    #[serde(default)]
    pub firecrawl_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenConfig {
    pub command: String,
    pub reason: String,
    #[serde(default)]
    pub reconsider: bool,
}

impl ForbiddenConfig {
    fn new(command: &str, reason: &str) -> Self {
        Self {
            command: command.to_string(),
            reason: reason.to_string(),
            reconsider: false,
        }
    }

    fn soft(command: &str, reason: &str) -> Self {
        Self {
            command: command.to_string(),
            reason: reason.to_string(),
            reconsider: true,
        }
    }
}

impl ToolsConfig {
    /// Built-in forbidden commands merged with user config.
    pub fn all_forbidden(&self) -> Vec<flashmind_types::tool::ForbiddenCmd> {
        let defaults = vec![
            ForbiddenConfig::new(
                "git reset --hard",
                "Never hard reset. You are doing something wrong, reconsider",
            ),
            ForbiddenConfig::soft(
                "git reset --soft",
                "Only git reset when completely necessary and approved by the user",
            ),
            ForbiddenConfig::new(
                "git push --force",
                "Never push force. You are doing something wrong, reconsider",
            ),
            ForbiddenConfig::new("git revert", "Never revert"),
            ForbiddenConfig::new("^cat", "Use file_read tool instead"),
            ForbiddenConfig::new(
                "^grep",
                "Use grep tool instead - it's faster and returns structured results",
            ),
            ForbiddenConfig::new("^rg", "Use grep tool instead"),
            ForbiddenConfig::new("^ls", "Use file_list tool instead"),
            ForbiddenConfig::new("^find", "Use glob tool instead"),
            ForbiddenConfig::new(
                "^sed",
                "Use text_replace or text_replace_regex tools instead",
            ),
            ForbiddenConfig::new("^awk", "Use text_replace_regex tool instead"),
            ForbiddenConfig::new("^rm ", "Delete individual files first, then use rmdir"),
            ForbiddenConfig::new("pkill", "You cannot pkill, use kill -2 instead"),
            ForbiddenConfig::new("^head", "Use read_lines instead"),
        ];

        defaults
            .into_iter()
            .chain(self.forbidden.iter().cloned())
            .map(|f| flashmind_types::tool::ForbiddenCmd {
                command: f.command,
                reason: f.reason,
                reconsider: f.reconsider,
            })
            .collect()
    }

    /// Built-in allowed command patterns merged with user config.
    pub fn all_allowed(&self) -> Vec<String> {
        let defaults: Vec<String> = [
            "echo *",
            "pwd",
            "which *",
            "whoami",
            "date *",
            "uname *",
            "cargo *",
            "rustc *",
            "git status*",
            "git log*",
            "git diff*",
            "git branch*",
            "git show*",
            "git remote*",
            "git rev-parse*",
            "npm run *",
            "npm test*",
            "node --version",
            "python --version",
            "python3 --version",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        defaults
            .into_iter()
            .chain(self.allowed.iter().cloned())
            .collect()
    }
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
    #[serde(default)]
    pub capture: Option<CaptureConfig>,
    #[serde(default)]
    pub contextual: Option<ContextualConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualConfig {
    #[serde(default = "default_true")]
    pub enable: bool,
    #[serde(default = "default_max_memories")]
    pub max_memories: usize,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
}

impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_memories: 5,
            similarity_threshold: 0.6,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_max_memories() -> usize {
    5
}

fn default_similarity_threshold() -> f32 {
    0.6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub debug: bool,
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

impl AppConfig {
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

impl AppConfig {
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
// Helpers
// ---------------------------------------------------------------------------

impl AppConfig {
    pub fn provider_config_for(
        &self,
        provider: &flashmind_types::model::Provider,
    ) -> Option<&ProviderConfig> {
        self.llm.providers.iter().find(|p| p.name == *provider)
    }

    pub fn active_provider_config(&self) -> Result<&ProviderConfig> {
        self.llm
            .providers
            .first()
            .context("no LLM providers configured — add [[llm.providers]] to config.toml")
    }

    pub fn collect_secrets(&self) -> Vec<String> {
        let mut secrets: Vec<String> = self
            .llm
            .providers
            .iter()
            .filter_map(|p| p.api_key.clone())
            .collect();
        if let Some(ref key) = self.tools.brave_api_key {
            secrets.push(key.clone());
        }
        if let Some(ref key) = self.tools.firecrawl_api_key {
            secrets.push(key.clone());
        }
        secrets
    }
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

pub const DEFAULT_CONFIG: &str = r#"# Flashmind CLI Configuration
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

# OpenAI — OpenAI API or compatible endpoints (vLLM, LiteLLM, etc.)
# [[llm.providers]]
# name = "openai"
# api_key = "sk-YOUR-KEY"
# model = "gpt-4o"
# url = "https://api.openai.com/"  # optional — custom base URL

[tools]
# brave_api_key = "BSA..."
# firecrawl_api_key = "fc-..."
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

# [memory.contextual]
# enable = true
# max_memories = 5
# similarity_threshold = 0.6

# [memory.capture]
# enable = true
# model = "ollama:llama3.2"
# debug = false

# Skills are loaded from ~/.flashmind/skills/
# MCP server configs are stored in ~/.flashmind/mcp/
"#;

pub const DEFAULT_SOUL: &str = "\
You are Flash — a technical AI agent with full system access. \
Your prime directive is continuous learning, context protection, and result-oriented execution.
- Be concise. Lead with results.
- Default to execution and action over discussion.
- Use tools to explore, verify, and act — don't guess when you can check.
- Do not use emojis.
";
