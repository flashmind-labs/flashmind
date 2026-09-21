use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

use flashmind_types::ReasoningLevel;

#[derive(Deserialize, Default)]
pub struct Config {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub reasoning: Option<ReasoningLevel>,

    pub openrouter_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub ollama_url: Option<String>,

    pub brave_api_key: Option<String>,
    pub firecrawl_api_key: Option<String>,

    pub memory_provider: Option<String>,
    pub memory_model: Option<String>,

    pub vision_model: Option<String>,

    /// Shell commands the agent may execute without prompting. Patterns use glob syntax.
    pub command_allowlist: Option<Vec<String>>,

    pub skill_dirs: Option<Vec<String>>,

    /// Pet companion: a pet name (e.g. `"cat"`), `"random"` for a random pet
    /// per session, or `"off"`/`"none"` to disable. `None` defaults to random.
    pub pet: Option<String>,
}

pub fn config_dir() -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .context("cannot determine home directory")?
        .join(".flashmind");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

const DEFAULT_CONFIG: &str = r#"# model = "ollama:llama3.2"
# system_prompt = "You are a helpful coding assistant."
# reasoning = "off"  # off, low, medium, high

# openrouter_api_key = ""
# anthropic_api_key = ""
# openai_api_key = ""
# ollama_url = "http://localhost:11434"

# brave_api_key = ""
# firecrawl_api_key = ""

# memory_provider = "openrouter"  # openrouter or openai
# memory_model = "openai/text-embedding-3-small"

# vision_model = "ollama:llava"  # used by image_read when main model lacks vision

# command_allowlist = ["cargo test*", "git status"]

# skill_dirs = ["~/.flashmind/skills"]
"#;

pub fn load_config() -> Result<Config> {
    let path = config_dir()?.join("config.toml");
    let mut config: Config = if path.exists() {
        let text = std::fs::read_to_string(&path)?;
        toml::from_str(&text).with_context(|| format!("invalid config: {}", path.display()))?
    } else {
        std::fs::write(&path, DEFAULT_CONFIG)?;
        Config::default()
    };

    macro_rules! env_override {
        ($field:ident, $var:literal) => {
            if config.$field.is_none() {
                config.$field = std::env::var($var).ok().filter(|s| !s.is_empty());
            }
        };
    }
    env_override!(openrouter_api_key, "OPENROUTER_API_KEY");
    env_override!(anthropic_api_key, "ANTHROPIC_API_KEY");
    env_override!(openai_api_key, "OPENAI_API_KEY");
    env_override!(openai_base_url, "OPENAI_BASE_URL");
    env_override!(ollama_url, "OLLAMA_URL");
    env_override!(brave_api_key, "BRAVE_API_KEY");
    env_override!(firecrawl_api_key, "FIRECRAWL_API_KEY");

    Ok(config)
}
