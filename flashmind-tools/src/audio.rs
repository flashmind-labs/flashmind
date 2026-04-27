//! Text-to-Speech tool — generates audio from text via LLM provider TTS APIs.
//! Includes `say` (tts) and `list_voices` tools.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};
use tracing::debug;

use crate::utils::truncate_utf8;
use flashmind_types::llm::{AudioFormat, LlmProvider, TtsRequest};
use flashmind_types::model::{Model, Provider};
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

pub type ProviderRegistry = Arc<std::collections::HashMap<Provider, Arc<dyn LlmProvider>>>;

/// TTS audio configuration — model and default voice.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AudioConfig {
    #[serde(default)]
    pub model: Option<Model>,
    #[serde(default)]
    pub voice: Option<String>,
}

/// Default voices for OpenAI TTS models.
const DEFAULT_VOICES: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

/// Arguments for the TTS tool.
#[derive(serde::Deserialize)]
struct SayArgs {
    text: String,
    model: Option<Model>,
    voice: Option<String>,
    output_format: Option<AudioFormat>,
}

pub struct SayTool {
    audio_config: AudioConfig,
    audio_dir: PathBuf,
    providers: ProviderRegistry,
}

impl SayTool {
    pub fn new(audio_config: AudioConfig, audio_dir: PathBuf, providers: ProviderRegistry) -> Self {
        Self {
            audio_config,
            audio_dir,
            providers,
        }
    }
}

#[async_trait]
impl Tool for SayTool {
    fn name(&self) -> &str {
        "say"
    }

    fn description(&self) -> &str {
        "Convert text to speech. After generating, ALWAYS include the `@filename` marker from the result in your response so the audio is sent to the user. For Telegram, use output_format='opus' for native voice notes. To send to a different chat instead, use the message tool with the file path.\n\nTo list available voices, use `list_voices`."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to convert to speech"
                },
                "model": {
                    "type": "string",
                    "description": "TTS model (e.g., 'openai:tts-1', 'openai:tts-1-hd'). Uses config default if not specified."
                },
                "voice": {
                    "type": "string",
                    "description": "Voice to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["mp3", "wav", "opus", "aac", "flac"],
                    "description": "Output format. Defaults to mp3."
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: SayArgs = ctx.parse_args(self.name())?;

        let model = args.model.or_else(|| self.audio_config.model.clone());

        let Some(model) = model else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No TTS model configured. Set [llm.audio] model in config or pass model argument (e.g., 'openai:tts-1').",
            ));
        };

        let voice = args.voice.or_else(|| self.audio_config.voice.clone());
        let Some(voice) = voice else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No TTS model configured. Set [llm.audio] voice in config or pass voice argument (use list_voices tool).",
            ));
        };

        let output_format = args.output_format.unwrap_or_default();

        debug!(text_len = args.text.len(), %model, %voice, %output_format, "TTS requested");

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        let request = TtsRequest {
            model: model.name().to_string(),
            input: args.text,
            voice,
            response_format: output_format,
        };

        let bytes = match provider.text_to_speech(request).await {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("TTS failed: {}", e),
                ));
            }
        };

        // Save to file
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let filename = format!("speech_{}.{}", timestamp, output_format);
        let output_path = self.audio_dir.join(&filename);

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        if let Err(e) = std::fs::write(&output_path, &bytes) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Failed to save audio file: {}", e),
            ));
        }

        let size_kb = bytes.len() / 1024;
        debug!(path = %output_path.display(), size_kb, "TTS audio saved");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Audio saved to {} ({}KB). \
                 To send it to the user, include @{} in your response. \
                 To send it to a different chat, use the message tool with the file path.",
                output_path.display(),
                size_kb,
                output_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio"),
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        if text.is_empty() {
            format!("Speaking with model: {}", model)
        } else {
            format!("Speaking: {}", truncate_utf8(text, 50))
        }
    }
}

/// Tool to list available TTS voices for a model.
pub struct ListVoicesTool {
    audio_config: AudioConfig,
    providers: ProviderRegistry,
}

impl ListVoicesTool {
    pub fn new(audio_config: AudioConfig, providers: ProviderRegistry) -> Self {
        Self {
            audio_config,
            providers,
        }
    }
}

#[derive(serde::Deserialize)]
struct ListVoicesArgs {
    model: Option<Model>,
}

#[async_trait]
impl Tool for ListVoicesTool {
    fn name(&self) -> &str {
        "list_voices"
    }

    fn description(&self) -> &str {
        "List available TTS voices for a model. If no model specified, uses the configured default."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "TTS model (e.g., 'openai:tts-1', 'ollama:voxtral-4b-tts'). Uses config default if not specified."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ListVoicesArgs = ctx.parse_args(self.name())?;

        let model = args.model.or_else(|| self.audio_config.model.clone());

        let Some(model) = model else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No TTS model configured. Set [llm.audio] model in config or pass model argument.",
            ));
        };

        // Check provider supports TTS
        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        // Try to get voices from provider
        let voice_structs = provider.list_voices(model.name()).await.unwrap_or_default();
        if voice_structs.is_empty() {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "No voices found for model '{}'. Try using one of the default OpenAI voices: {}",
                    model,
                    DEFAULT_VOICES.join(", ")
                ),
            ));
        }

        let voice_list: Vec<String> = voice_structs.iter().map(|v| v.name.clone()).collect();
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Available voices for {}: {}", model, voice_list.join(", ")),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        format!("Listing TTS voices for model: {}", model)
    }
}
