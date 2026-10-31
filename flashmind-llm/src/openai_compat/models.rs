//! Model catalogs returned by OpenAI-compatible endpoints.

use chrono::DateTime;
use flashmind_types::{ModelCapabilities, ModelInfo, ModelPricing};
use rust_decimal::Decimal;
use serde::Deserialize;

use super::OpenAiCompatProfile;
use crate::oss_capabilities;

#[derive(Clone)]
pub(crate) struct CatalogModel {
    pub info: ModelInfo,
    pub reasoning_efforts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum ModelsResponse {
    Object { data: Vec<ModelEntry> },
    List(Vec<ModelEntry>),
}

impl ModelsResponse {
    pub(crate) fn entries(self) -> Vec<ModelEntry> {
        match self {
            Self::Object { data } => data,
            Self::List(data) => data,
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct ModelEntry {
    pub id: String,
    created: Option<i64>,
    max_model_len: Option<u32>,
    context_window: Option<u32>,
    context_length: Option<u32>,
    max_completion_tokens: Option<u32>,
    display_name: Option<String>,
    #[serde(default)]
    pricing: Prices,
    model_spec: Option<VeniceSpec>,
}

impl ModelEntry {
    pub(crate) fn context_length(&self) -> Option<u32> {
        self.model_spec
            .as_ref()
            .and_then(|spec| spec.available_context_tokens)
            .or(self.context_length)
            .or(self.context_window)
            .or(self.max_model_len)
    }

    pub(crate) fn reasoning_efforts(&self) -> Vec<String> {
        self.model_spec
            .as_ref()
            .map(|spec| spec.capabilities.reasoning_effort_options.clone())
            .unwrap_or_default()
    }

    pub(crate) fn into_info(self, profile: OpenAiCompatProfile) -> ModelInfo {
        let context_length = self.context_length();
        let mut capabilities = oss_capabilities::get_oss_capabilities_or_default(&self.id);
        let pricing = match profile {
            OpenAiCompatProfile::Venice => {
                if let Some(spec) = &self.model_spec {
                    spec.capabilities.apply(&mut capabilities);
                    spec.pricing.per_token()
                } else {
                    ModelPricing::default()
                }
            }
            OpenAiCompatProfile::Together => self.pricing.per_token(),
            _ => ModelPricing::default(),
        };
        if profile == OpenAiCompatProfile::DeepSeek {
            capabilities.tool_calling = true;
            capabilities.reasoning = true;
        }
        if profile == OpenAiCompatProfile::Groq && self.id.starts_with("openai/gpt-oss-") {
            capabilities.reasoning = true;
            capabilities.tool_calling = true;
        }
        ModelInfo {
            id: self.id,
            name: self.display_name,
            created_at: self
                .created
                .and_then(|created| DateTime::from_timestamp(created, 0)),
            context_length,
            max_completion_tokens: self.max_completion_tokens,
            categories: capabilities.categories(),
            capabilities,
            pricing,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeniceSpec {
    available_context_tokens: Option<u32>,
    #[serde(default)]
    capabilities: VeniceCapabilities,
    #[serde(default)]
    pricing: Prices,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeniceCapabilities {
    supports_function_calling: Option<bool>,
    supports_vision: Option<bool>,
    supports_reasoning: Option<bool>,
    supports_audio_input: Option<bool>,
    supports_video_input: Option<bool>,
    #[serde(default)]
    reasoning_effort_options: Vec<String>,
}

impl VeniceCapabilities {
    fn apply(&self, caps: &mut ModelCapabilities) {
        if let Some(value) = self.supports_function_calling {
            caps.tool_calling = value;
        }
        if let Some(value) = self.supports_vision {
            caps.images = value;
        }
        if let Some(value) = self.supports_reasoning {
            caps.reasoning = value;
        }
        if let Some(value) = self.supports_audio_input {
            caps.audio = value;
        }
        if let Some(value) = self.supports_video_input {
            caps.video = value;
        }
    }
}

#[derive(Default, Deserialize)]
struct Prices {
    input: Option<Price>,
    output: Option<Price>,
    cache_input: Option<Price>,
    cache_write: Option<Price>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Price {
    Usd { usd: Decimal },
    Number(Decimal),
}

impl Price {
    fn per_token(&self) -> Decimal {
        let price = match self {
            Self::Usd { usd } => usd,
            Self::Number(value) => value,
        };
        price / Decimal::from(1_000_000)
    }
}

impl Prices {
    fn per_token(&self) -> ModelPricing {
        ModelPricing {
            prompt: self.input.as_ref().map(Price::per_token),
            completion: self.output.as_ref().map(Price::per_token),
            cache_read: self.cache_input.as_ref().map(Price::per_token),
            cache_write: self.cache_write.as_ref().map(Price::per_token),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn venice_catalog_has_context_capabilities_and_per_token_cache_prices() {
        let entry: ModelEntry = serde_json::from_value(json!({
            "id": "example-model", "model_spec": {
                "availableContextTokens": 400000,
                "capabilities": { "supportsFunctionCalling": true, "supportsVision": true,
                    "supportsReasoning": true, "reasoningEffortOptions": ["low", "high"] },
                "pricing": { "input": {"usd": 2}, "output": {"usd": 8},
                    "cache_input": {"usd": 0.2}, "cache_write": {"usd": 2.5} }
            }
        }))
        .unwrap();
        assert_eq!(entry.reasoning_efforts(), ["low", "high"]);
        let info = entry.into_info(OpenAiCompatProfile::Venice);
        assert_eq!(info.context_length, Some(400000));
        assert!(
            info.capabilities.images
                && info.capabilities.tool_calling
                && info.capabilities.reasoning
        );
        assert_eq!(info.pricing.prompt, Some(Decimal::new(2, 6)));
        assert_eq!(info.pricing.completion, Some(Decimal::new(8, 6)));
        assert_eq!(info.pricing.cache_read, Some(Decimal::new(2, 7)));
        assert_eq!(info.pricing.cache_write, Some(Decimal::new(25, 7)));
    }

    #[test]
    fn together_accepts_array_catalog_and_groq_uses_context_window() {
        let list: ModelsResponse = serde_json::from_value(json!([{
            "id": "example-model", "display_name": "Example", "context_length": 64000,
            "pricing": {"input": 0.5, "output": 2}
        }]))
        .unwrap();
        let info = list
            .entries()
            .remove(0)
            .into_info(OpenAiCompatProfile::Together);
        assert_eq!(info.context_length, Some(64000));
        assert_eq!(info.name.as_deref(), Some("Example"));
        assert_eq!(info.pricing.prompt, Some(Decimal::new(5, 7)));
        let list: ModelsResponse = serde_json::from_value(json!({"data": [{
            "id": "openai/gpt-oss-20b", "context_window": 128000, "max_completion_tokens": 8000
        }]}))
        .unwrap();
        let info = list
            .entries()
            .remove(0)
            .into_info(OpenAiCompatProfile::Groq);
        assert_eq!(info.context_length, Some(128000));
        assert_eq!(info.max_completion_tokens, Some(8000));
        assert!(info.capabilities.reasoning);
        assert_eq!(info.pricing.prompt, None);
    }
}
