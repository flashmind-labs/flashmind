use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU64;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::DocumentMut;

use flashmind_types::{Model, ProviderPreferences, ReasoningLevel};

use crate::paths::config_dir;

#[derive(Clone, Deserialize, Serialize, Default)]
pub struct OpenRouterConfig {
    pub routing: Option<ProviderPreferences>,
    pub stream_start_timeout_secs: Option<NonZeroU64>,
    pub stream_idle_timeout_secs: Option<NonZeroU64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct AutoCompactSettings {
    pub enabled: bool,
    pub tokens: Option<u32>,
}

impl Default for AutoCompactSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tokens: None,
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Default)]
pub struct Config {
    #[serde(default)]
    pub lsp: BTreeMap<String, flashmind_tools::lsp::LanguageServerConfig>,
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

    /// Enables the Firecrawl tools: web search, scrape, crawl and map. Absent
    /// means no web tools are registered at all.
    pub firecrawl_api_key: Option<String>,

    pub vision_model: Option<String>,

    pub skill_dirs: Option<Vec<String>>,

    /// Skills whose full instructions go into every system prompt, instead of
    /// only their name in the skill index.
    pub always_skills: Option<Vec<String>>,

    /// Pet companion: a pet name (e.g. `"cat"`), `"random"` for a random pet
    /// per session, or `"off"`/`"none"` to disable. `None` defaults to random.
    pub pet: Option<String>,

    /// Display rate for streamed text and reasoning. `None` uses the UI default;
    /// `0` displays output as fast as it arrives.
    pub stream_chars_per_second: Option<u32>,
    /// Minimum display time before switching tools. `None` uses the UI default;
    /// `0` disables tool display pacing.
    pub tool_display_interval_ms: Option<u64>,

    /// Compact the conversation automatically; `false` waits for the context
    /// window to overflow. `None` means on. `/autocompact` saves it.
    pub auto_compact: Option<bool>,

    /// Compact once the prompt passes this many tokens, if that is earlier
    /// than the model's default trigger. `/autocompact` saves it.
    pub auto_compact_tokens: Option<u32>,

    /// A model rule replaces the global rule. Unlisted models use the global rule.
    #[serde(default)]
    pub auto_compact_models: HashMap<Model, AutoCompactSettings>,

    /// Subagents that may run at once; `0` removes the limit. `None` means
    /// [`DEFAULT_MAX_AGENTS`].
    pub max_agents: Option<usize>,
}

/// Subagents that may run at once unless `max_agents` says otherwise.
pub const DEFAULT_MAX_AGENTS: usize = 16;

impl Config {
    pub fn global_auto_compact(&self) -> AutoCompactSettings {
        AutoCompactSettings {
            enabled: self.auto_compact.unwrap_or(true),
            tokens: self.auto_compact_tokens,
        }
    }

    pub fn auto_compact_for(&self, model: &Model) -> AutoCompactSettings {
        self.auto_compact_models
            .get(model)
            .copied()
            .unwrap_or_else(|| self.global_auto_compact())
    }

    /// The concurrent subagent limit, with `0` meaning unlimited.
    pub fn max_agents(&self) -> usize {
        match self.max_agents.unwrap_or(DEFAULT_MAX_AGENTS) {
            0 => usize::MAX,
            limit => limit,
        }
    }
}

const DEFAULT_CONFIG: &str = r#"# model = "ollama:llama3.2"
# system_prompt = "You are a helpful coding assistant."
# reasoning = "off"  # off, low, medium, high

# openrouter_api_key = ""
# anthropic_api_key = ""
# openai_api_key = ""
# ollama_url = "http://localhost:11434"

# firecrawl_api_key = ""  # enables the web search, scrape, crawl and map tools

# vision_model = "ollama:llava"  # used by image_read when main model lacks vision

# skill_dirs = ["~/my-skills"]  # in addition to the skills/ dir next to this file
# always_skills = ["i-have-adhd"]  # skills loaded into every session's system prompt
# Or create .<skill-name>-always next to this file or in Claude's config directory.

# auto_compact = true          # false waits for the context window to overflow
# auto_compact_tokens = 80000  # compact earlier than the model's default trigger

# max_agents = 16  # subagents running at once; 0 for no limit

# stream_chars_per_second = 400  # display speed; 0 for unlimited
# tool_display_interval_ms = 500  # minimum time before switching tools; 0 for no delay

# [auto_compact_models."openrouter:example/model"]
# enabled = true
# tokens = 120000  # omit to use this model's default trigger

# [openrouter]
# stream_start_timeout_secs = 300  # maximum wait for the first streamed output
# stream_idle_timeout_secs = 30  # maximum silence after output starts

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
    env_override!(firecrawl_api_key, "FIRECRAWL_API_KEY");

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
    fn display_rate_is_optional_and_accepts_zero_for_unlimited() {
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.stream_chars_per_second, None);
        for rate in [0, 100, 400] {
            let config: Config =
                toml::from_str(&format!("stream_chars_per_second = {rate}")).unwrap();
            assert_eq!(config.stream_chars_per_second, Some(rate));
            let restored: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
            assert_eq!(restored.stream_chars_per_second, Some(rate));
        }
        assert!(toml::from_str::<Config>("stream_chars_per_second = -1").is_err());
    }

    #[test]
    fn tool_display_interval_is_optional_and_accepts_zero() {
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.tool_display_interval_ms, None);
        for interval in [0, 500, 1000] {
            let config: Config =
                toml::from_str(&format!("tool_display_interval_ms = {interval}")).unwrap();
            assert_eq!(config.tool_display_interval_ms, Some(interval));
            let restored: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
            assert_eq!(restored.tool_display_interval_ms, Some(interval));
        }
        assert!(toml::from_str::<Config>("tool_display_interval_ms = -1").is_err());
    }

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
    fn max_agents_defaults_to_sixteen_and_zero_lifts_the_limit() {
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.max_agents(), DEFAULT_MAX_AGENTS);
        let config: Config = toml::from_str("max_agents = 0").unwrap();
        assert_eq!(config.max_agents(), usize::MAX);
        let config: Config = toml::from_str("max_agents = 32").unwrap();
        assert_eq!(config.max_agents(), 32);
    }

    #[test]
    fn openrouter_settings_are_optional() {
        let config: Config = toml::from_str("model = \"ollama:example\"").unwrap();
        assert!(config.openrouter.routing.is_none());
        assert!(config.openrouter.stream_start_timeout_secs.is_none());
        assert!(config.openrouter.stream_idle_timeout_secs.is_none());
    }

    #[test]
    fn stream_idle_timeout_is_configurable_and_rejects_zero() {
        let config: Config = toml::from_str("[openrouter]\nstream_idle_timeout_secs = 45").unwrap();
        assert_eq!(
            config.openrouter.stream_idle_timeout_secs.unwrap().get(),
            45
        );
        assert!(toml::from_str::<Config>("[openrouter]\nstream_idle_timeout_secs = 0").is_err());
    }

    #[test]
    fn stream_start_timeout_is_configurable_and_rejects_zero() {
        let config: Config =
            toml::from_str("[openrouter]\nstream_start_timeout_secs = 90").unwrap();
        assert_eq!(
            config.openrouter.stream_start_timeout_secs.unwrap().get(),
            90
        );
        assert!(toml::from_str::<Config>("[openrouter]\nstream_start_timeout_secs = 0").is_err());
    }
}
