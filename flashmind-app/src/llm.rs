//! LLM config builder methods.

use anyhow::{Context, Result};

use flashmind_types::AgentLlmConfig;
use flashmind_types::model::{Model, ReasoningLevel, SamplingParams};

use crate::config::AppConfig;

// ---------------------------------------------------------------------------
// Model / LLM config
// ---------------------------------------------------------------------------

impl AppConfig {
    /// Resolve the active [`Model`], optionally overridden by `model_override`.
    pub fn build_model(&self, model_override: Option<&Model>) -> Result<Model> {
        if let Some(m) = model_override {
            return Ok(m.clone());
        }
        let pc = self.active_provider_config()?;
        let full = format!("{}:{}", pc.name, pc.model);
        full.parse()
            .with_context(|| format!("invalid model: {full}"))
    }

    /// Build the full [`AgentLlmConfig`] from the config file.
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
}
