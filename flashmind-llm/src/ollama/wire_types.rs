//! Ollama-specific wire types for the native `/api/chat` format.
//!
//! <https://github.com/ollama/ollama/blob/main/docs/api.md>
//!
//! Includes request/response structs, NDJSON streaming types,
//! and model capability helpers.

use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize, Serializer};
use url::Url;

fn serialize_optional_decimal<S>(value: &Option<Decimal>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(d) => {
            let f: f64 = (*d).try_into().map_err(serde::ser::Error::custom)?;
            serializer.serialize_f64(f)
        }
        None => serializer.serialize_none(),
    }
}

// ============================================================================
// Model Capabilities
// ============================================================================

#[derive(Deserialize)]
pub(super) struct ShowResponse {
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
    #[serde(default)]
    pub parameters: Option<String>,
}

/// Extract context window size from /api/show response.
pub(super) fn extract_context_window(show: &ShowResponse) -> Option<u32> {
    // Check model_info for context length keys (arch-prefixed or bare)
    if let Some(ref info) = show.model_info
        && let Some(obj) = info.as_object()
    {
        for (key, val) in obj {
            if key.ends_with("context_length")
                && let Some(n) = val.as_u64()
            {
                return Some(n as u32);
            }
        }
    }
    // Check parameters string for num_ctx
    if let Some(ref params) = show.parameters {
        for line in params.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("num_ctx") {
                let val = rest.trim();
                if let Ok(n) = val.parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Check model_info for vision-related metadata (projector, encoder, vision blocks).
///
/// Vision models in Ollama's model_info typically have keys like:
/// - `llava.projector_type` / `*.projector_type` — CLIP-based vision projector
/// - `gemma3.vision.block_count` / `*.vision.*` — vision encoder blocks
/// - `clip.has_vision_encoder` — explicit vision marker
///
/// This catches models that support images but don't report "vision" in capabilities
/// (custom Modelfiles, community quants, older Ollama versions).
pub(super) fn has_vision_metadata(show: &ShowResponse) -> bool {
    let Some(ref info) = show.model_info else {
        return false;
    };
    let Some(obj) = info.as_object() else {
        return false;
    };

    obj.keys().any(|k| {
        let k = k.to_ascii_lowercase();
        k.contains("projector") || k.contains(".vision.")
    })
}

/// Query /api/show and return the full response.
pub(super) async fn query_show(client: &Client, url: Url, model: &str) -> Option<ShowResponse> {
    let resp = client
        .post(url)
        .json(&serde_json::json!({ "model": model }))
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => r.json::<ShowResponse>().await.ok(),
        _ => None,
    }
}

// ============================================================================
// Native API Wire Types — Ollama /api/chat format
// ============================================================================

#[derive(Serialize)]
pub(super) struct NativeRequest {
    pub model: String,
    pub messages: Vec<NativeMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<NativeTool>>,
    pub think: bool,
    pub options: NativeOptions,
}

#[derive(Serialize)]
pub(super) struct NativeOptions {
    #[serde(skip_serializing_if = "Option::is_none", with = "rust_decimal::serde::float_option")]
    pub temperature: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    /// Context window size override. Ollama defaults to 2048 or model-specific value,
    /// but often caps at 32K to save memory. Set explicitly to use larger context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub min_p: Option<Decimal>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_decimal"
    )]
    pub presence_penalty: Option<Decimal>,
    /// Maps from config `repetition_penalty` to Ollama's `repeat_penalty`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "repeat_penalty",
        serialize_with = "serialize_optional_decimal"
    )]
    pub repetition_penalty: Option<Decimal>,
}

#[derive(Serialize)]
pub(super) struct NativeMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<NativeToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct NativeToolCall {
    pub function: NativeFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct NativeFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool definition in Ollama's native format (same as OpenAI).
#[derive(Serialize)]
pub(super) struct NativeTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: NativeToolFunction,
}

#[derive(Serialize)]
pub(super) struct NativeToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ============================================================================
// NDJSON Streaming Types
// ============================================================================

#[derive(Deserialize)]
pub(super) struct NativeChunk {
    pub message: Option<NativeChunkMessage>,
    pub done: bool,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct NativeChunkMessage {
    #[allow(dead_code)]
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<NativeToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_ndjson_chunk_parsing() {
        let json = r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: NativeChunk = serde_json::from_str(json).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        assert_eq!(msg.content.as_deref(), Some("Hello"));
        assert!(msg.thinking.is_none());
    }

    #[test]
    fn test_ndjson_chunk_with_thinking() {
        let json = r#"{"message":{"role":"assistant","content":"","thinking":"Let me think..."},"done":false}"#;
        let chunk: NativeChunk = serde_json::from_str(json).unwrap();
        let msg = chunk.message.unwrap();
        assert_eq!(msg.thinking.as_deref(), Some("Let me think..."));
        assert_eq!(msg.content.as_deref(), Some(""));
    }

    #[test]
    fn test_ndjson_chunk_done_with_usage() {
        let json = r#"{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":42,"eval_count":100}"#;
        let chunk: NativeChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, Some(42));
        assert_eq!(chunk.eval_count, Some(100));
    }

    #[test]
    fn test_ndjson_chunk_done_without_usage() {
        let json = r#"{"done":true}"#;
        let chunk: NativeChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.done);
        assert!(chunk.prompt_eval_count.is_none());
        assert!(chunk.eval_count.is_none());
        assert!(chunk.message.is_none());
    }

    #[test]
    fn test_ndjson_chunk_with_tool_calls() {
        let json = r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"file_read","arguments":{"path":"/tmp/test"}}}]},"done":false}"#;
        let chunk: NativeChunk = serde_json::from_str(json).unwrap();
        let msg = chunk.message.unwrap();
        let tool_calls = msg.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "file_read");
        assert_eq!(
            tool_calls[0].function.arguments,
            serde_json::json!({"path": "/tmp/test"})
        );
    }

    #[test]
    fn test_native_request_serialization() {
        let request = NativeRequest {
            model: "llama3.2".into(),
            messages: vec![NativeMessage {
                role: "user".into(),
                content: "Hello".into(),
                tool_calls: None,
                images: None,
            }],
            stream: true,
            tools: None,
            think: false,
            options: NativeOptions {
                temperature: Some(Decimal::from_str("0.7").unwrap()),
                num_predict: None,
                num_ctx: None,
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                repetition_penalty: None,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "llama3.2");
        assert_eq!(json["stream"], true);
        assert!(json.get("tools").is_none()); // skip_serializing_if = None
        assert_eq!(json["think"], false);
        assert_eq!(json["options"]["temperature"], 0.7);
        assert!(json["options"].get("num_predict").is_none());
    }

    #[test]
    fn test_native_request_with_think() {
        let request = NativeRequest {
            model: "qwq".into(),
            messages: vec![],
            stream: true,
            tools: None,
            think: true,
            options: NativeOptions {
                temperature: Some(Decimal::from_str("0.5").unwrap()),
                num_predict: Some(4096),
                num_ctx: None,
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                repetition_penalty: None,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["think"], true);
        assert_eq!(json["options"]["num_predict"], 4096);
    }

    #[test]
    fn test_native_request_with_tools() {
        let request = NativeRequest {
            model: "llama3.2".into(),
            messages: vec![],
            stream: true,
            tools: Some(vec![NativeTool {
                tool_type: "function".into(),
                function: NativeToolFunction {
                    name: "bash".into(),
                    description: "Run a command".into(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            think: false,
            options: NativeOptions {
                temperature: Some(Decimal::from_str("0.7").unwrap()),
                num_predict: None,
                num_ctx: None,
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                repetition_penalty: None,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
    }

    #[test]
    fn test_has_vision_metadata_projector() {
        let show = ShowResponse {
            capabilities: vec![],
            model_info: Some(serde_json::json!({
                "llava.projector_type": "mlp",
                "general.architecture": "llava"
            })),
            parameters: None,
        };
        assert!(has_vision_metadata(&show));
    }

    #[test]
    fn test_has_vision_metadata_vision_blocks() {
        let show = ShowResponse {
            capabilities: vec![],
            model_info: Some(serde_json::json!({
                "gemma3.vision.block_count": 27,
                "general.architecture": "gemma3"
            })),
            parameters: None,
        };
        assert!(has_vision_metadata(&show));
    }

    #[test]
    fn test_has_vision_metadata_none() {
        let show = ShowResponse {
            capabilities: vec![],
            model_info: Some(serde_json::json!({
                "general.architecture": "llama",
                "llama.context_length": 131072
            })),
            parameters: None,
        };
        assert!(!has_vision_metadata(&show));
    }

    #[test]
    fn test_has_vision_metadata_no_model_info() {
        let show = ShowResponse {
            capabilities: vec![],
            model_info: None,
            parameters: None,
        };
        assert!(!has_vision_metadata(&show));
    }
}
