use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::DocumentMut;

use flashmind_types::{ProviderPreferences, ReasoningLevel};

use crate::paths::config_dir;

#[derive(Clone, Deserialize, Serialize, Default)]
pub struct OpenRouterConfig {
    pub routing: Option<ProviderPreferences>,
}

#[derive(Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub reasoning: Option<ReasoningLevel>,

    pub openrouter_api_key: Option<String>,
    #[serde(default)]
    pub openrouter: OpenRouterConfig,
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub ollama_url: Option<String>,

    pub vision_model: Option<String>,

    /// Shell commands the agent may execute without prompting. Patterns use glob syntax.
    pub command_allowlist: Option<Vec<String>>,

    pub skill_dirs: Option<Vec<String>>,

    /// Pet companion: a pet name (e.g. `"cat"`), `"random"` for a random pet
    /// per session, or `"off"`/`"none"` to disable. `None` defaults to random.
    pub pet: Option<String>,

    /// Compact the conversation automatically; `false` waits for the context
    /// window to overflow. `None` means on. `/autocompact` saves it.
    pub auto_compact: Option<bool>,

    /// Compact once the prompt passes this many tokens, if that is earlier
    /// than the model's default trigger. `/autocompact` saves it.
    pub auto_compact_tokens: Option<u32>,
}

const DEFAULT_CONFIG: &str = r#"# model = "ollama:llama3.2"
# system_prompt = "You are a helpful coding assistant."
# reasoning = "off"  # off, low, medium, high

# openrouter_api_key = ""
# anthropic_api_key = ""
# openai_api_key = ""
# ollama_url = "http://localhost:11434"

# vision_model = "ollama:llava"  # used by image_read when main model lacks vision

# command_allowlist = ["cargo test*", "git status"]

# skill_dirs = ["~/my-skills"]  # in addition to the skills/ dir next to this file

# auto_compact = true          # false waits for the context window to overflow
# auto_compact_tokens = 80000  # compact earlier than the model's default trigger

# [openrouter.routing]
# zdr = true
# sort = "throughput"  # price, throughput, latency
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

    Ok(config)
}

/// Change `config.toml` through `edit`, keeping its comments and layout.
pub fn update_config(edit: impl FnOnce(&mut DocumentMut)) -> Result<()> {
    let path = config_dir()?.join("config.toml");
    let text = if path.exists() {
        std::fs::read_to_string(&path)?
    } else {
        String::new()
    };
    let mut doc: DocumentMut = text
        .parse()
        .with_context(|| format!("invalid config: {}", path.display()))?;
    edit(&mut doc);
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ProviderSort;

    #[test]
    fn routing_round_trip_preserves_openrouter_options() {
        let config: Config = toml::from_str(
            r#"
            model = "openrouter:example/model"
            [openrouter.routing]
            zdr = true
            sort = "throughput"
            data_collection = "deny"
            allow_fallbacks = true
            order = ["example-provider"]
        "#,
        )
        .unwrap();
        let routing = config.openrouter.routing.unwrap();
        assert_eq!(routing.zdr, Some(true));
        assert_eq!(routing.sort, Some(ProviderSort::Throughput));
        assert_eq!(routing.order, ["example-provider"]);
        let saved = format!(
            "model = \"openrouter:example/model\"\n[openrouter.routing]\n{}",
            toml::to_string(&routing).unwrap()
        );
        let restored: Config = toml::from_str(&saved).unwrap();
        assert_eq!(
            serde_json::to_value(restored.openrouter.routing).unwrap(),
            serde_json::to_value(routing).unwrap()
        );
    }

    #[test]
    fn openrouter_settings_are_optional() {
        let config: Config = toml::from_str("model = \"ollama:example\"").unwrap();
        assert!(config.openrouter.routing.is_none());
    }
}
