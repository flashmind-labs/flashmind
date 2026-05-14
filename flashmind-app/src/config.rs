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
    pub brave_api_key: Option<String>,
    #[serde(default)]
    pub firecrawl_api_key: Option<String>,
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
    #[serde(default)]
    pub capture: Option<CaptureConfig>,
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
