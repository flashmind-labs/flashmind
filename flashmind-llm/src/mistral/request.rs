//! Chat completion request body for Mistral.
//!
//! The shared OpenAI-compatible builder does the message, tool and sampling
//! conversion. This module keeps only the fields Mistral documents, so
//! OpenAI-only extensions (`stream_options`, `chat_template_kwargs`, `min_p`,
//! ...) never reach it.

use serde::Serialize;
use serde_json::Value;

use crate::request_builder::{RequestConfig, build_openai_compat_request};
use crate::wire_types::{ApiContent, ApiMessage, ApiSamplingParams, ApiTool};
use flashmind_types::CompletionRequest;
use flashmind_types::model::ReasoningLevel;

/// Mistral's `reasoning_effort` values. Its own models document only these two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum ReasoningEffort {
    None,
    High,
}

impl ReasoningEffort {
    /// Pick the effort for a request. Mistral models think by default, so an
    /// explicit `none` is sent when reasoning is off and the model is known to
    /// reason. Models without reasoning get no field at all.
    pub(super) fn for_request(level: ReasoningLevel, model_reasons: bool) -> Option<Self> {
        match (level.is_on(), model_reasons) {
            (true, _) => Some(Self::High),
            (false, true) => Some(Self::None),
            (false, false) => None,
        }
    }
}

#[derive(Serialize)]
pub(super) struct MistralChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(flatten)]
    sampling: ApiSamplingParams,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
}

/// Build the Mistral chat request for `request`.
///
/// `model_reasons` says whether `/v1/models` lists the model as a reasoning
/// model; it decides whether reasoning-off is sent explicitly.
pub(super) fn build_chat_request(
    request: &CompletionRequest,
    model_reasons: bool,
) -> MistralChatRequest {
    let (base, _meta) = build_openai_compat_request(request, &RequestConfig::default());
    let mut sampling = base.sampling;
    // Mistral has no min_p.
    sampling.min_p = None;
    let has_tools = !base.tools.is_empty();
    MistralChatRequest {
        model: request.model.capability_name().to_string(),
        messages: base.messages.into_iter().map(mistral_message).collect(),
        tools: base.tools,
        tool_choice: base.tool_choice,
        parallel_tool_calls: has_tools.then_some(base.parallel_tool_calls),
        sampling,
        stream: true,
        reasoning_effort: ReasoningEffort::for_request(request.reasoning, model_reasons),
        prompt_cache_key: request
            .session_id
            .as_ref()
            .map(|session| session.as_str().to_string()),
    }
}

/// Mistral has no `developer` role. Like other non-OpenAI endpoints, send it
/// as a user turn wrapped in `<system>` tags, since these messages can come
/// last and Mistral expects a user or tool message there.
fn mistral_message(mut message: ApiMessage) -> ApiMessage {
    if message.role == "developer" {
        message.role = "user".into();
        if let ApiContent::Text(text) = &mut message.content {
            *text = format!("<system>\n{text}\n</system>");
        }
    }
    message
}

/// Serialize the request, rewriting `input_audio` parts into Mistral's shape:
/// a base64 string rather than an object with `data` and `format`.
pub(super) fn to_json(request: &MistralChatRequest) -> serde_json::Result<Vec<u8>> {
    let mut body = serde_json::to_value(request)?;
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if let Some(audio) = part.get_mut("input_audio")
                    && let Some(data) = audio.get_mut("data").map(Value::take)
                {
                    *audio = data;
                }
            }
        }
    }
    serde_json::to_vec(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::message::ContentPart;
    use flashmind_types::{Message, SessionId, ToolDefinition};
    use rust_decimal::Decimal;
    use serde_json::json;

    fn request(reasoning: ReasoningLevel) -> CompletionRequest {
        let mut request = CompletionRequest::new(
            "mistral:mistral-medium-3-5".parse().unwrap(),
            vec![Message::system("Be brief."), Message::user("hello")],
        );
        request.reasoning = reasoning;
        request
    }

    fn body(request: &CompletionRequest, model_reasons: bool) -> Value {
        let bytes = to_json(&build_chat_request(request, model_reasons)).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn request_keeps_only_fields_mistral_accepts() {
        let mut request = request(ReasoningLevel::Off);
        request.sampling.temperature = Some(Decimal::new(3, 1));
        request.sampling.min_p = Some(Decimal::new(1, 1));
        request.sampling.top_k = Some(40);
        request.max_tokens = Some(512);
        request.user = Some("someone".into());
        let body = body(&request, false);
        assert_eq!(
            body,
            json!({
                "model": "mistral-medium-3-5",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hello"},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
                "top_k": 40,
                "stream": true,
            })
        );
    }

    #[test]
    fn tools_set_auto_choice_and_parallel_calls() {
        let mut request = request(ReasoningLevel::Off);
        request.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Weather for a city".into(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }];
        let body = body(&request, false);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn reasoning_maps_to_reasoning_effort() {
        assert_eq!(
            body(&request(ReasoningLevel::Low), false)["reasoning_effort"],
            "high"
        );
        assert_eq!(
            body(&request(ReasoningLevel::High), true)["reasoning_effort"],
            "high"
        );
        assert_eq!(
            body(&request(ReasoningLevel::Off), true)["reasoning_effort"],
            "none"
        );
        assert!(
            body(&request(ReasoningLevel::Off), false)
                .get("reasoning_effort")
                .is_none()
        );
    }

    #[test]
    fn session_id_becomes_prompt_cache_key() {
        let session = SessionId::new();
        let request = request(ReasoningLevel::Off).with_session_id(session.clone());
        assert_eq!(body(&request, false)["prompt_cache_key"], session.as_str());
    }

    #[test]
    fn developer_messages_become_user_turns() {
        let mut request = request(ReasoningLevel::Off);
        request.messages.push(Message::developer("Mode changed"));
        let body = body(&request, false);
        assert_eq!(
            body["messages"][2],
            json!({"role": "user", "content": "<system>\nMode changed\n</system>"})
        );
    }

    #[test]
    fn images_and_audio_use_mistral_content_chunks() {
        let mut message = Message::user("What is this?");
        message.parts = Some(vec![
            ContentPart::Image {
                media_type: "image/png".into(),
                data: "aW1n".into(),
            },
            ContentPart::Audio {
                media_type: "audio/mp3".into(),
                filename: "clip.mp3".into(),
                data: "YXVkaW8=".into(),
            },
        ]);
        let mut request = request(ReasoningLevel::Off);
        request.messages = vec![message];
        let body = body(&request, false);
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
                {"type": "input_audio", "input_audio": "YXVkaW8="},
            ])
        );
    }
}
