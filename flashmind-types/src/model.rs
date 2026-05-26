//! Model identifiers, provider variants, and sampling parameters.
//!
//! The [`Model`] type parses from a `"provider:model-name"` string via `FromStr`
//! (e.g., `"openrouter:anthropic/claude-sonnet-4"`) and serialises back to the same format.
//! Aliased models (`"fast,gpt-4.1"`) let users define shorthand names that resolve
//! to real model IDs at API-call time.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Model`] | Fully-qualified model ID (`provider:name`); parses bidirectionally |
//! | [`Provider`] | LLM backend variant (OpenRouter, Ollama, Anthropic, OpenAI, Connect) |
//! | [`AliasedModel`] | Model name with optional aliasing (`name,real_name`) |
//! | [`ReasoningLevel`] | Extended chain-of-thought effort (Off / Low / Medium / High) |
//! | [`SamplingParams`] | Extended sampling config (top_p, top_k, min_p, penalties) |
//! | [`AgentLlmConfig`] | Complete LLM config for an agent turn (model, temp, reasoning, sampling) |

use std::fmt;
use std::str::FromStr;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::ParseError;

/// Extended chain-of-thought / reasoning effort level.
///
/// Controls how much "thinking" budget the model gets. Not all providers
/// support granular levels; when unsupported, any non-[`Off`](Self::Off)
/// variant is treated as reasoning-enabled with a provider-chosen budget.
///
/// `"on"` deserialises as [`Medium`](Self::Medium) for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningLevel {
    #[default]
    /// Reasoning/thinking is disabled. The model responds directly.
    Off,
    /// Minimal reasoning budget.
    Low,
    /// Moderate reasoning budget (default when reasoning is simply "enabled").
    #[serde(alias = "on")]
    Medium,
    /// Maximum reasoning budget.
    High,
}

impl ReasoningLevel {
    /// Returns `true` for any level that enables reasoning.
    pub fn is_on(&self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// LLM backend variant. Used to route completion requests and select the
/// correct [`crate::llm::LlmProvider`] implementation from the registry.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Hash,
    Default,
    strum::Display,
    strum::EnumString,
    strum::EnumIter,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    OpenRouter,
    #[default]
    Ollama,
    Anthropic,
    OpenAi,
    /// Proxy to a remote Flash connect server (client-side use).
    Connect,
}

impl Provider {
    /// Hard-coded default model ID used when none is configured.
    ///
    /// | Provider     | Default Model                       |
    /// |--------------|-------------------------------------|
    /// | `OpenRouter` | `anthropic/claude-sonnet-4`         |
    /// | `Ollama`     | `qwen3:8b`                          |
    /// | `Anthropic`  | `claude-sonnet-4-20250514`          |
    /// | `OpenAi`     | `gpt-4.1`                           |
    /// | `Connect`    | `flashone-229b`                     |
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenRouter => "anthropic/claude-sonnet-4",
            Provider::Ollama => "qwen3:8b",
            Provider::Anthropic => "claude-sonnet-4-20250514",
            Provider::OpenAi => "gpt-4.1",
            Provider::Connect => "flashone-229b",
        }
    }
}

/// A model name with optional aliasing.
///
/// Strings like `"fast-gpt4, gpt-4.1"` parse to an `AliasedModel` where:
/// - `name = "fast-gpt4"` — used in config and display
/// - `real_name = Some("gpt-4.1")` — resolved when building API requests
///
/// If no comma is present, `real_name` is `None` and both fields are identical.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasedModel {
    /// Display / config name (may be a user-defined alias).
    pub name: String,
    /// Actual model ID sent to the API if this is an alias, otherwise `None`.
    pub real_name: Option<String>,
}

impl AliasedModel {
    /// The name used when probing provider capabilities (e.g. context window lookup).
    pub fn capability_name(&self) -> &str {
        self.real_name.as_deref().unwrap_or(&self.name)
    }
}

impl Serialize for AliasedModel {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for AliasedModel {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl FromStr for AliasedModel {
    type Err = ParseError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (name, real_name) = if let Some((alias, real)) = s.split_once(',') {
            (alias.to_string(), Some(real.to_string()))
        } else {
            (s.to_string(), None)
        };
        Ok(AliasedModel { name, real_name })
    }
}

impl fmt::Display for AliasedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.real_name {
            Some(real) => write!(f, "{},{}", self.name, real),
            None => write!(f, "{}", self.name),
        }
    }
}

/// Fully-qualified model identifier combining a backend [`Provider`] with an [`AliasedModel`].
///
/// String format: `"provider:name"` (e.g. `"openrouter:anthropic/claude-sonnet-4-20250514"`).
/// Parses bidirectionally via `FromStr` / `Display`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model {
    pub provider: Provider,
    pub model: AliasedModel,
}

impl Serialize for Model {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider, self.model)
    }
}

impl FromStr for Model {
    type Err = ParseError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (provider_str, model_part) = s.split_once(':').ok_or_else(|| {
            ParseError::new(format!(
                "Invalid model format '{}': expected 'provider:name'",
                s
            ))
        })?;
        let provider: Provider = provider_str.parse().map_err(|_| {
            ParseError::new(format!(
                "Unknown provider '{}' in model string",
                provider_str
            ))
        })?;

        let model: AliasedModel = model_part.parse()?;

        Ok(Model { provider, model })
    }
}

impl Model {
    /// Returns the display/config name of the model.
    pub fn name(&self) -> &str {
        &self.model.name
    }

    /// Returns the resolved name used for capability lookups. If this is an alias,
    /// returns the real underlying model name; otherwise returns the display name.
    pub fn capability_name(&self) -> &str {
        self.model.capability_name()
    }
}

/// Extended sampling parameters beyond just temperature.
///
/// All fields are optional; unset fields use the model's default.
/// Serialises to a flat TOML table. Merge via [`merge`](Self::merge).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Sampling temperature (0.0–2.0). Lower values make output more deterministic; higher values increase creativity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Decimal>,
    /// Nucleus sampling threshold (0.0–1.0). Only tokens within the top-p probability mass are considered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Decimal>,
    /// Hard cutoff: only sample from the K most likely next tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Dynamic minimum probability filter (0.0–1.0). Tokens below this absolute probability are excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<Decimal>,
    /// Additive penalty for tokens already present in the output (-2.0 to 2.0). Positive values discourage repetition of seen tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<Decimal>,
    /// Multiplicative penalty on repeated tokens (1.0–2.0). Values above 1.0 make repeated tokens less likely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<Decimal>,
}

// ---------------------------------------------------------------------------
// Provider Preferences
// ---------------------------------------------------------------------------

/// Quantization level for model weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Bf16,
    Fp16,
    Fp32,
    Fp8,
    Int4,
    Int8,
    /// Catch-all for future/unknown quantization levels.
    #[serde(untagged)]
    Other(String),
}

/// How to sort provider candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderSort {
    Price,
    Throughput,
    Latency,
}

/// Provider-level routing preferences (OpenRouter extension).
///
/// Controls which inference providers are selected and how they are ranked.
/// See <https://openrouter.ai/docs/features/provider-routing>.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderPreferences {
    /// Preferred provider ordering (e.g. `["DeepInfra", "Together"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    /// Allowed quantization levels. Empty means no filter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantizations: Vec<Quantization>,
    /// Whether to fall back to other providers if preferred ones are unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Only route to providers that support all request parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Opt out of provider-level data collection (`"deny"` to opt out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<String>,
    /// Sort providers by this criterion before selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
    /// Providers to exclude from routing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
}

impl SamplingParams {
    /// Override `self` with values from `other`, keeping `self`'s value when
    /// `other` has a field set to `None`. Useful for profile layering.
    ///
    /// The semantics are simple: for each field, if `other` has a `Some` value it
    /// wins; otherwise `self`'s value (if any) is retained. This makes it ideal
    /// for stacking an overlay profile on top of a base profile — the overlay
    /// only changes what it explicitly sets.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            min_p: other.min_p.or(self.min_p),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            repetition_penalty: other.repetition_penalty.or(self.repetition_penalty),
        }
    }
}

impl fmt::Display for SamplingParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut params = Vec::new();
        if let Some(t) = self.temperature {
            params.push(format!("t={}", t));
        }
        if let Some(top_p) = self.top_p {
            params.push(format!("top_p={}", top_p));
        }
        if let Some(top_k) = self.top_k {
            params.push(format!("top_k={}", top_k));
        }
        if let Some(min_p) = self.min_p {
            params.push(format!("min_p={}", min_p));
        }
        if let Some(pp) = self.presence_penalty {
            params.push(format!("p_penalty={}", pp));
        }
        if let Some(rp) = self.repetition_penalty {
            params.push(format!("r_penalty={}", rp));
        }
        if params.is_empty() {
            write!(f, "(none)")
        } else {
            write!(f, "{}", params.join(", "))
        }
    }
}

/// Complete LLM configuration for an agent turn.
///
/// Passed to [`LlmProvider::complete`](crate::llm::LlmProvider::complete)
/// and cloned into each `Started` event so listeners know the current config.
#[derive(Debug, Clone)]
pub struct AgentLlmConfig {
    /// The model to use for this turn (includes provider prefix).
    pub model: Model,
    /// Maximum output tokens. If `None`, the provider decides.
    pub max_tokens: Option<u32>,
    /// Whether reasoning/thinking mode is enabled.
    pub reasoning: ReasoningLevel,
    /// Extended sampling parameters (temperature, top_p, penalties).
    pub sampling: SamplingParams,
    /// Opaque user identifier for per-user tracking in provider dashboards.
    pub user: Option<String>,
    /// Provider routing preferences (used by OpenRouter).
    pub provider_preferences: Option<ProviderPreferences>,
}

impl AgentLlmConfig {
    /// Create a default LLM configuration. Use `with_model()` to set the model.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams::default(),
            user: None,
            provider_preferences: None,
        }
    }

    /// Set the user identifier for per-user tracking in provider dashboards.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set provider routing preferences (used by OpenRouter).
    pub fn with_provider_preferences(mut self, prefs: ProviderPreferences) -> Self {
        self.provider_preferences = Some(prefs);
        self
    }

    /// Swap in a different model while preserving all other parameters.
    pub fn with_model(&self, model: Model) -> Self {
        Self {
            model,
            max_tokens: self.max_tokens,
            reasoning: self.reasoning.clone(),
            sampling: self.sampling.clone(),
            user: self.user.clone(),
            provider_preferences: self.provider_preferences.clone(),
        }
    }
}

impl fmt::Display for AgentLlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sampling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn parse_model_round_trip() {
        let model: Model = "openrouter:anthropic/claude-sonnet-4".parse().unwrap();
        assert_eq!(model.provider, Provider::OpenRouter);
        assert_eq!(model.model.name, "anthropic/claude-sonnet-4");
        assert_eq!(model.to_string(), "openrouter:anthropic/claude-sonnet-4");
    }

    #[test]
    fn parse_aliased_model() {
        let aliased: AliasedModel = "fast,gpt-4.1".parse().unwrap();
        assert_eq!(aliased.name, "fast");
        assert_eq!(aliased.real_name.as_deref(), Some("gpt-4.1"));
        assert_eq!(aliased.capability_name(), "gpt-4.1");
    }

    #[test]
    fn parse_model_invalid_format() {
        let result = "no-colon-here".parse::<Model>();
        assert!(result.is_err());
    }

    #[test]
    fn parse_model_unknown_provider() {
        let result = "foobar:some-model".parse::<Model>();
        assert!(result.is_err());
    }

    #[test]
    fn sampling_params_merge() {
        let base = SamplingParams {
            top_p: Some(dec!(0.9)),
            top_k: Some(40),
            ..Default::default()
        };
        let overlay = SamplingParams {
            top_p: Some(dec!(0.8)),
            min_p: Some(dec!(0.05)),
            ..Default::default()
        };
        let merged = base.merge(&overlay);
        assert_eq!(merged.top_p, Some(dec!(0.8)));
        assert_eq!(merged.top_k, Some(40));
        assert_eq!(merged.min_p, Some(dec!(0.05)));
    }

    #[test]
    fn sampling_params_display_empty() {
        let params = SamplingParams::default();
        assert_eq!(params.to_string(), "(none)");
    }

    #[test]
    fn agent_llm_config_display() {
        let config = AgentLlmConfig {
            model: "ollama:qwen3:8b".parse().unwrap(),
            max_tokens: None,
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams {
                temperature: Some(dec!(0.7)),
                top_p: Some(dec!(0.9)),
                ..Default::default()
            },
            user: None,
            provider_preferences: None,
        };
        let display = config.to_string();
        assert!(display.contains("t=0.7"));
        assert!(display.contains("top_p=0.9"));
    }

    #[test]
    fn agent_llm_config_with_model() {
        let config = AgentLlmConfig {
            model: "ollama:test".parse().unwrap(),
            max_tokens: Some(1000),
            reasoning: ReasoningLevel::Medium,
            sampling: SamplingParams {
                temperature: Some(dec!(0.5)),
                ..Default::default()
            },
            user: None,
            provider_preferences: None,
        };
        let new_model: Model = "anthropic:claude-sonnet-4-20250514".parse().unwrap();
        let updated = config.with_model(new_model.clone());
        assert_eq!(updated.model, new_model);
        assert_eq!(updated.sampling.temperature, Some(dec!(0.5)));
        assert_eq!(updated.max_tokens, Some(1000));
    }

    #[test]
    fn provider_default_models() {
        assert!(!Provider::Ollama.default_model().is_empty());
        assert!(!Provider::OpenRouter.default_model().is_empty());
        assert!(!Provider::Anthropic.default_model().is_empty());
        assert!(!Provider::OpenAi.default_model().is_empty());
    }

    #[test]
    fn reasoning_level_variants() {
        assert!(!ReasoningLevel::Off.is_on());
        assert!(ReasoningLevel::Low.is_on());
        assert!(ReasoningLevel::Medium.is_on());
        assert!(ReasoningLevel::High.is_on());
    }

    #[test]
    fn reasoning_level_on_deserialises_as_medium() {
        let level: ReasoningLevel = serde_json::from_str(r#""on""#).unwrap();
        assert_eq!(level, ReasoningLevel::Medium);
    }

    #[test]
    fn model_serde_round_trip() {
        let model: Model = "ollama:llama3.2".parse().unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(model, deserialized);
    }
}
