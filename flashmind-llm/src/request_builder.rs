//! Shared request-building logic for OpenAI-compatible providers.
//!
//! Both [`crate::openai::OpenAiProvider`] and [`crate::openrouter::OpenRouterProvider`]
//! follow the OpenAI chat completions wire format. This module extracts the common
//! request construction — sampling params, tool definitions, modality handling,
//! gemma-4 special tokens, image/audio config — into a single
//! [`build_openai_compat_request`] function.
//!
//! Provider-specific tweaks (field stripping, reasoning config) are applied
//! after the shared builder returns.

use crate::wire_types::{
    ApiAudioConfig, ApiImageConfig, ApiMessage, ApiSamplingParams, ApiTool, ChatTemplateKwargs,
    StreamOptions, to_api_messages, to_api_tools,
};
use flashmind_types::CompletionRequest;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Provider-specific overrides applied during shared request construction.
///
/// Each provider creates one of these before calling [`build_openai_compat_request`]
/// to customise which sampling parameters are included and which model-name
/// patterns trigger special handling.
#[derive(Default)]
pub struct RequestConfig {
    /// When `true`, `top_k`, `min_p`, and `repetition_penalty` are stripped
    /// from the sampling params (OpenAI's API rejects these unknown fields).
    pub strip_extended_sampling: bool,

    /// When `true`, `presence_penalty` is forced to `None`.
    /// Used for Kimi/Moonshot models that only accept `presence_penalty=0`.
    pub strip_presence_penalty: bool,
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// The shared portion of an OpenAI-compatible chat completion request.
///
/// Contains every field that both OpenAI and OpenRouter include verbatim.
/// Provider-specific extensions (e.g. OpenRouter's `reasoning` / `include_reasoning`)
/// are added by the caller after receiving this struct.
#[derive(Serialize)]
pub struct OpenAiCompatRequest {
    /// Model identifier (e.g. `"gpt-4.1"`, `"anthropic/claude-sonnet-4"`).
    pub model: String,
    /// Message history in wire format.
    pub messages: Vec<ApiMessage>,
    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Sampling params flattened into the top-level object.
    #[serde(flatten)]
    pub sampling: ApiSamplingParams,
    pub stream: bool,
    pub stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    pub chat_template_kwargs: ChatTemplateKwargs,
    pub parallel_tool_calls: bool,
    /// Requested output modalities (`text`, `audio`, `image`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ApiAudioConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<ApiImageConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Metadata extracted during request building that providers may need for
/// their own extensions (e.g. to decide whether to attach reasoning config).
pub struct RequestMeta {
    /// Whether the request has image output modality.
    pub has_image_output: bool,
    /// Whether reasoning/thinking mode was requested.
    pub reasoning_on: bool,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build the shared portion of an OpenAI-compatible chat completion request.
///
/// Handles:
/// - Message and tool conversion to wire format
/// - Sampling parameter construction (with image-output bypass)
/// - Gemma-4 `skip_special_tokens` detection
/// - Modality, audio, and image config mapping
/// - Tool choice and parallel tool call settings
///
/// The caller receives both the request body and [`RequestMeta`] so it can
/// layer on provider-specific fields (OpenRouter reasoning, etc.) before
/// serialising.
pub fn build_openai_compat_request(
    request: &CompletionRequest,
    config: &RequestConfig,
) -> (OpenAiCompatRequest, RequestMeta) {
    let messages = to_api_messages(&request.messages);
    let tools = to_api_tools(request.tools.clone());

    // Use capability_name for model lookup (handles aliases via real_name)
    let model_name = request.model.capability_name();

    // Gemma-4 models require skip_special_tokens=false when thinking is enabled.
    // Both "gemma-4" and "gemma4" patterns are checked for compatibility.
    let is_gemma_4 = model_name.contains("gemma-4") || model_name.contains("gemma4");
    let reasoning_on = request.reasoning.is_on();
    let include_special_tokens = is_gemma_4 && reasoning_on;

    let has_image_output = request
        .modalities
        .contains(&flashmind_types::Modality::Image);

    let sampling = if has_image_output {
        ApiSamplingParams::default()
    } else {
        ApiSamplingParams {
            temperature: request.sampling.temperature,
            max_tokens: request.max_tokens,
            top_p: request.sampling.top_p,
            top_k: if config.strip_extended_sampling {
                None
            } else {
                request.sampling.top_k
            },
            min_p: if config.strip_extended_sampling {
                None
            } else {
                request.sampling.min_p
            },
            presence_penalty: if config.strip_presence_penalty {
                None
            } else {
                request.sampling.presence_penalty
            },
            repetition_penalty: if config.strip_extended_sampling {
                None
            } else {
                request.sampling.repetition_penalty
            },
        }
    };

    let api_request = OpenAiCompatRequest {
        model: request.model.name().to_string(),
        messages,
        tool_choice: if tools.is_empty() {
            None
        } else {
            Some("auto".into())
        },
        tools,
        sampling,
        stream: true,
        stream_options: StreamOptions::default(),
        skip_special_tokens: if include_special_tokens && !has_image_output {
            Some(false)
        } else {
            None
        },
        chat_template_kwargs: ChatTemplateKwargs {
            enable_thinking: if has_image_output {
                false
            } else {
                reasoning_on
            },
        },
        parallel_tool_calls: !has_image_output,
        modalities: if request.modalities.is_empty() {
            None
        } else {
            Some(
                request
                    .modalities
                    .iter()
                    .map(|m| match m {
                        flashmind_types::Modality::Text => "text".into(),
                        flashmind_types::Modality::Audio => "audio".into(),
                        flashmind_types::Modality::Image => "image".into(),
                    })
                    .collect(),
            )
        },
        audio: request.audio_config.as_ref().map(|c| ApiAudioConfig {
            voice: c.voice.clone(),
            format: c.format.to_string(),
        }),
        image_config: request.image_config.as_ref().map(|c| ApiImageConfig {
            aspect_ratio: c.aspect_ratio.clone(),
            size: c.size.clone(),
            super_resolution_references: vec![],
        }),
        user: request.user.clone(),
    };

    let meta = RequestMeta {
        has_image_output,
        reasoning_on,
    };

    (api_request, meta)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::model::ReasoningLevel;
    use flashmind_types::{CompletionRequest, Message, Modality, SamplingParams};
    use rust_decimal::Decimal;
    use std::str::FromStr;

    fn minimal_request() -> CompletionRequest {
        CompletionRequest {
            messages: vec![Message::user("hello")],
            tools: vec![],
            model: "openai:test-model".parse().unwrap(),
            max_tokens: Some(100),
            reasoning: ReasoningLevel::Off,
            sampling: SamplingParams::default(),
            modalities: vec![],
            audio_config: None,
            image_config: None,
            user: None,
            provider_preferences: None,
        }
    }

    #[test]
    fn default_config_preserves_all_sampling() {
        let mut req = minimal_request();
        req.sampling.top_k = Some(50);
        req.sampling.min_p = Some(Decimal::from_str("0.1").unwrap());
        req.sampling.repetition_penalty = Some(Decimal::from_str("1.2").unwrap());
        req.sampling.presence_penalty = Some(Decimal::from_str("0.5").unwrap());

        let config = RequestConfig::default();
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert_eq!(built.sampling.top_k, Some(50));
        assert_eq!(
            built.sampling.min_p,
            Some(Decimal::from_str("0.1").unwrap())
        );
        assert_eq!(
            built.sampling.repetition_penalty,
            Some(Decimal::from_str("1.2").unwrap())
        );
        assert_eq!(
            built.sampling.presence_penalty,
            Some(Decimal::from_str("0.5").unwrap())
        );
    }

    #[test]
    fn strip_extended_sampling_removes_fields() {
        let mut req = minimal_request();
        req.sampling.top_k = Some(50);
        req.sampling.min_p = Some(Decimal::from_str("0.1").unwrap());
        req.sampling.repetition_penalty = Some(Decimal::from_str("1.2").unwrap());
        req.sampling.presence_penalty = Some(Decimal::from_str("0.5").unwrap());

        let config = RequestConfig {
            strip_extended_sampling: true,
            strip_presence_penalty: false,
        };
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert_eq!(built.sampling.top_k, None);
        assert_eq!(built.sampling.min_p, None);
        assert_eq!(built.sampling.repetition_penalty, None);
        // presence_penalty is NOT stripped by strip_extended_sampling
        assert_eq!(
            built.sampling.presence_penalty,
            Some(Decimal::from_str("0.5").unwrap())
        );
    }

    #[test]
    fn strip_presence_penalty_removes_field() {
        let mut req = minimal_request();
        req.sampling.presence_penalty = Some(Decimal::from_str("0.5").unwrap());

        let config = RequestConfig {
            strip_extended_sampling: false,
            strip_presence_penalty: true,
        };
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert_eq!(built.sampling.presence_penalty, None);
    }

    #[test]
    fn image_output_resets_sampling_to_default() {
        let mut req = minimal_request();
        req.sampling.temperature = Some(Decimal::from_str("0.9").unwrap());
        req.modalities = vec![Modality::Image];

        let config = RequestConfig::default();
        let (built, meta) = build_openai_compat_request(&req, &config);

        assert!(meta.has_image_output);
        assert_eq!(built.sampling.temperature, None);
        assert_eq!(built.sampling.max_tokens, None);
        assert!(!built.parallel_tool_calls);
        assert!(!built.chat_template_kwargs.enable_thinking);
    }

    #[test]
    fn empty_tools_produces_no_tool_choice() {
        let req = minimal_request();
        let config = RequestConfig::default();
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert!(built.tools.is_empty());
        assert!(built.tool_choice.is_none());
    }

    #[test]
    fn tools_present_sets_auto_tool_choice() {
        let mut req = minimal_request();
        req.tools = vec![flashmind_types::ToolDefinition {
            name: "test".into(),
            description: "a test tool".into(),
            parameters: serde_json::json!({}),
        }];

        let config = RequestConfig::default();
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert_eq!(built.tools.len(), 1);
        assert_eq!(built.tool_choice.as_deref(), Some("auto"));
    }

    #[test]
    fn gemma4_with_reasoning_sets_skip_special_tokens() {
        let mut req = minimal_request();
        req.model = "openai:gemma-4-27b".parse().unwrap();
        req.reasoning = ReasoningLevel::On;

        let config = RequestConfig::default();
        let (built, meta) = build_openai_compat_request(&req, &config);

        assert!(meta.reasoning_on);
        assert_eq!(built.skip_special_tokens, Some(false));
        assert!(built.chat_template_kwargs.enable_thinking);
    }

    #[test]
    fn gemma4_no_hyphen_variant_also_detected() {
        let mut req = minimal_request();
        req.model = "openai:gemma4-27b".parse().unwrap();
        req.reasoning = ReasoningLevel::On;

        let config = RequestConfig::default();
        let (built, _meta) = build_openai_compat_request(&req, &config);

        assert_eq!(built.skip_special_tokens, Some(false));
    }
}
