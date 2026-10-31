//! Provider-specific behavior on the shared chat completions transport.

use flashmind_types::{CompletionRequest, ReasoningLevel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strum::EnumString;
use url::Url;

pub(crate) mod fireworks_images;
pub(crate) mod models;
pub(crate) mod openai_edit;
pub(crate) mod together_video;
pub(crate) mod venice_edit;
pub(crate) mod venice_images;
pub(crate) mod venice_video;
pub(crate) mod zai_images;

/// Selects documented request and response conventions for a compatible API.
/// Unknown endpoints use `Generic`; proxies can select a profile explicitly.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenAiCompatProfile {
    #[default]
    Generic,
    #[serde(rename = "openai")]
    #[strum(serialize = "openai")]
    OpenAi,
    Venice,
    Groq,
    Together,
    Fireworks,
    #[serde(rename = "deepseek")]
    #[strum(serialize = "deepseek")]
    DeepSeek,
    Moonshot,
    Zai,
    Vllm,
}

impl OpenAiCompatProfile {
    /// Public API base URL for a known provider.
    pub fn default_url(self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some("https://api.openai.com/v1"),
            Self::Venice => Some("https://api.venice.ai/api/v1"),
            Self::Groq => Some("https://api.groq.com/openai/v1"),
            Self::Together => Some("https://api.together.xyz/v1"),
            Self::Fireworks => Some("https://api.fireworks.ai/inference/v1"),
            Self::DeepSeek => Some("https://api.deepseek.com/v1"),
            Self::Moonshot => Some("https://api.moonshot.ai/v1"),
            Self::Zai => Some("https://api.z.ai/api/paas/v4"),
            Self::Generic | Self::Vllm => None,
        }
    }

    /// Infer only from an exact public API hostname, never a user-chosen alias.
    pub fn for_url(url: &Url) -> Self {
        match url.host_str() {
            Some("api.openai.com") => Self::OpenAi,
            Some("api.venice.ai") => Self::Venice,
            Some("api.groq.com") => Self::Groq,
            Some("api.together.xyz" | "api.together.ai") => Self::Together,
            Some("api.fireworks.ai") => Self::Fireworks,
            Some("api.deepseek.com") => Self::DeepSeek,
            Some("api.moonshot.ai") => Self::Moonshot,
            Some("api.z.ai") => Self::Zai,
            _ => Self::Generic,
        }
    }

    pub(crate) fn cache_routing(self, configured: Option<bool>) -> bool {
        configured.unwrap_or(matches!(
            self,
            Self::OpenAi | Self::Venice | Self::Fireworks
        ))
    }

    pub(crate) fn prepare_body(
        self,
        body: &mut Value,
        request: &CompletionRequest,
        cache_routing: Option<bool>,
        reasoning_efforts: &[String],
    ) {
        let Some(fields) = body.as_object_mut() else {
            return;
        };
        // These are inference-server extensions, not shared OpenAI fields.
        if self != Self::Vllm {
            fields.remove("chat_template_kwargs");
            fields.remove("skip_special_tokens");
        }
        if !matches!(self, Self::Vllm | Self::Together | Self::Fireworks) {
            for field in ["top_k", "min_p", "repetition_penalty"] {
                fields.remove(field);
            }
        }
        if matches!(
            self,
            Self::Generic | Self::DeepSeek | Self::Moonshot | Self::Zai
        ) {
            fields.remove("parallel_tool_calls");
        }
        if self == Self::Zai {
            fields.remove("stream_options");
        }
        if self.cache_routing(cache_routing)
            && self != Self::Fireworks
            && let Some(session) = &request.session_id
        {
            fields.insert("prompt_cache_key".into(), json!(session.as_str()));
        }
        let model = request.model.capability_name();
        let enabled = request.reasoning.is_on();
        let effort = request.reasoning.to_string();
        match self {
            Self::OpenAi if enabled => {
                fields.insert("reasoning_effort".into(), json!(effort));
            }
            Self::Venice => {
                fields.insert(
                    "venice_parameters".into(),
                    json!({
                        "disable_thinking": !enabled,
                        "strip_thinking_response": false,
                    }),
                );
                if reasoning_efforts.contains(&effort) {
                    fields.insert("reasoning_effort".into(), json!(effort));
                }
            }
            Self::Groq if model.starts_with("openai/gpt-oss-") => {
                // GPT-OSS always reasons and accepts only low, medium, high.
                let effort = match request.reasoning {
                    ReasoningLevel::Off | ReasoningLevel::Low => "low",
                    ReasoningLevel::Medium => "medium",
                    _ => "high",
                };
                fields.insert("reasoning_effort".into(), json!(effort));
                fields.insert("include_reasoning".into(), json!(enabled));
            }
            Self::DeepSeek => {
                fields.insert(
                    "thinking".into(),
                    json!({"type": if enabled { "enabled" } else { "disabled" }}),
                );
                if enabled {
                    let effort = match request.reasoning {
                        ReasoningLevel::Low => "low",
                        _ => "high",
                    };
                    fields.insert("reasoning_effort".into(), json!(effort));
                    fields.remove("temperature");
                    fields.remove("presence_penalty");
                }
            }
            Self::Zai
                if enabled
                    && (model.starts_with("glm-5")
                        || model.starts_with("glm-4.6")
                        || model.starts_with("glm-4.7")) =>
            {
                fields.insert("thinking".into(), json!({"type": "enabled"}));
            }
            Self::Moonshot if enabled && model.starts_with("kimi-k3.1") => {
                let effort = match request.reasoning {
                    ReasoningLevel::Low => "low",
                    _ => "high",
                };
                fields.insert("reasoning_effort".into(), json!(effort));
            }
            _ => {}
        }
        // Thinking traces are part of the next tool request on these backends.
        // Other providers continue to receive only the ordinary message fields.
        if matches!(
            self,
            Self::DeepSeek | Self::Together | Self::Fireworks | Self::Zai | Self::Moonshot
        ) && let Some(messages) = fields.get_mut("messages").and_then(Value::as_array_mut)
        {
            for (message, source) in messages.iter_mut().zip(&request.messages) {
                if let Some(reasoning) = &source.reasoning {
                    message["reasoning_content"] = json!(reasoning);
                }
            }
        }
    }
}

/// Preserve gateway prefixes and accept both API roots and versioned base URLs.
pub(crate) fn endpoint_url(base: &Url, endpoint: &str) -> Url {
    let mut url = base.clone();
    let prefix = base.path().trim_end_matches('/');
    let endpoint = endpoint.trim_start_matches('/');
    let endpoint = if prefix.ends_with("/v1") || prefix.ends_with("/v4") {
        endpoint.strip_prefix("v1/").unwrap_or(endpoint)
    } else {
        endpoint
    };
    url.set_path(&format!("{prefix}/{endpoint}"));
    url
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_builder::{RequestConfig, build_openai_compat_request};
    use flashmind_types::{Message, SessionId};

    fn body(profile: OpenAiCompatProfile, model: &str, cache: Option<bool>) -> Value {
        let mut assistant = Message::assistant("answer");
        assistant.reasoning = Some("retained reasoning".into());
        let mut request = CompletionRequest::new(model.parse().unwrap(), vec![assistant])
            .with_session_id("example-session".parse::<SessionId>().unwrap());
        request.reasoning = ReasoningLevel::High;
        let (base, _) = build_openai_compat_request(&request, &RequestConfig::default());
        let mut body = serde_json::to_value(base).unwrap();
        profile.prepare_body(&mut body, &request, cache, &["low".into(), "high".into()]);
        body
    }

    #[test]
    fn endpoints_preserve_provider_prefixes_and_versions() {
        for (base, expected) in [
            ("https://api.venice.ai/api/v1", "/api/v1/chat/completions"),
            (
                "https://api.groq.com/openai/v1/",
                "/openai/v1/chat/completions",
            ),
            (
                "https://api.fireworks.ai/inference/v1",
                "/inference/v1/chat/completions",
            ),
            ("https://api.deepseek.com", "/v1/chat/completions"),
            ("https://api.together.xyz/v1", "/v1/chat/completions"),
            ("https://api.moonshot.ai/v1", "/v1/chat/completions"),
            (
                "https://api.z.ai/api/paas/v4",
                "/api/paas/v4/chat/completions",
            ),
            (
                "https://proxy.example/gateway/v1",
                "/gateway/v1/chat/completions",
            ),
        ] {
            let base = Url::parse(base).unwrap();
            assert_eq!(endpoint_url(&base, "v1/chat/completions").path(), expected);
        }
    }

    #[test]
    fn media_endpoints_preserve_versioned_provider_prefixes() {
        for base in [
            "https://api.venice.ai/api/v1",
            "https://api.groq.com/openai/v1",
            "https://api.openai.com/v1",
        ] {
            let base = Url::parse(base).unwrap();
            for path in [
                "images/generations",
                "audio/speech",
                "audio/transcriptions",
                "audio/voices",
            ] {
                let expected = format!("{}/{}", base.path(), path);
                assert_eq!(endpoint_url(&base, &format!("/v1/{path}")).path(), expected);
            }
        }
    }

    #[test]
    fn known_profiles_have_matching_defaults_and_unknown_hosts_are_conservative() {
        for profile in [
            OpenAiCompatProfile::OpenAi,
            OpenAiCompatProfile::Venice,
            OpenAiCompatProfile::Groq,
            OpenAiCompatProfile::Together,
            OpenAiCompatProfile::Fireworks,
            OpenAiCompatProfile::DeepSeek,
            OpenAiCompatProfile::Moonshot,
            OpenAiCompatProfile::Zai,
        ] {
            let url = Url::parse(profile.default_url().unwrap()).unwrap();
            assert_eq!(OpenAiCompatProfile::for_url(&url), profile);
        }
        for url in [
            "https://api.venice.ai.example/v1",
            "https://proxy.example/venice/v1",
        ] {
            assert_eq!(
                OpenAiCompatProfile::for_url(&Url::parse(url).unwrap()),
                OpenAiCompatProfile::Generic
            );
        }
        let generic = body(OpenAiCompatProfile::Generic, "custom:example", None);
        for field in [
            "prompt_cache_key",
            "reasoning_effort",
            "chat_template_kwargs",
            "parallel_tool_calls",
        ] {
            assert!(generic.get(field).is_none(), "unexpected {field}");
        }
        assert!(generic["messages"][0].get("reasoning_content").is_none());
    }

    #[test]
    fn moonshot_and_zai_keep_reasoning_for_tool_turns() {
        let zai = body(OpenAiCompatProfile::Zai, "zai:glm-5.3", None);
        assert_eq!(zai["thinking"]["type"], "enabled");
        assert_eq!(
            zai["messages"][0]["reasoning_content"],
            "retained reasoning"
        );
        let kimi = body(
            OpenAiCompatProfile::Moonshot,
            "moonshot:kimi-k3.1-pro",
            None,
        );
        assert_eq!(kimi["reasoning_effort"], "high");
        assert_eq!(
            kimi["messages"][0]["reasoning_content"],
            "retained reasoning"
        );
    }

    #[test]
    fn cache_hints_are_provider_specific_and_can_be_disabled() {
        for profile in [OpenAiCompatProfile::OpenAi, OpenAiCompatProfile::Venice] {
            assert_eq!(
                body(profile, "custom:example", None)["prompt_cache_key"],
                "example-session"
            );
            assert!(
                body(profile, "custom:example", Some(false))
                    .get("prompt_cache_key")
                    .is_none()
            );
        }
        for profile in [
            OpenAiCompatProfile::Groq,
            OpenAiCompatProfile::Together,
            OpenAiCompatProfile::DeepSeek,
            OpenAiCompatProfile::Fireworks,
        ] {
            assert!(
                body(profile, "custom:example", None)
                    .get("prompt_cache_key")
                    .is_none()
            );
        }
        assert_eq!(
            body(OpenAiCompatProfile::Generic, "custom:example", Some(true))["prompt_cache_key"],
            "example-session"
        );
    }

    #[test]
    fn reasoning_is_mapped_and_history_is_only_sent_to_supported_profiles() {
        let deepseek = body(OpenAiCompatProfile::DeepSeek, "deepseek:example", None);
        assert_eq!(deepseek["thinking"]["type"], "enabled");
        assert_eq!(deepseek["reasoning_effort"], "high");
        assert!(deepseek.get("temperature").is_none());
        for profile in [
            OpenAiCompatProfile::DeepSeek,
            OpenAiCompatProfile::Together,
            OpenAiCompatProfile::Fireworks,
        ] {
            assert_eq!(
                body(profile, "custom:example", None)["messages"][0]["reasoning_content"],
                "retained reasoning"
            );
        }
        let groq = body(OpenAiCompatProfile::Groq, "groq:openai/gpt-oss-20b", None);
        assert_eq!(groq["reasoning_effort"], "high");
        assert_eq!(groq["include_reasoning"], true);
        assert!(
            body(OpenAiCompatProfile::Groq, "groq:plain-model", None)
                .get("reasoning_effort")
                .is_none()
        );
        let venice = body(OpenAiCompatProfile::Venice, "venice:example", None);
        assert_eq!(venice["reasoning_effort"], "high");
        assert_eq!(venice["venice_parameters"]["disable_thinking"], false);
    }
}
