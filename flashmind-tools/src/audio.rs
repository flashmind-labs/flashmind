//! Audio tools — TTS (text-to-speech) and STT (speech-to-text / transcription).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::utils::truncate_utf8;
use flashmind_types::StreamEvent;
use flashmind_types::llm::{AudioFormat, LlmProvider, SttRequest, TtsRequest};
use flashmind_types::model::{Model, Provider};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

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

// ============================================================================
// TTS Tool
// ============================================================================

#[derive(serde::Deserialize)]
struct TtsArgs {
    text: String,
    model: Option<Model>,
    voice: Option<String>,
    output_format: Option<AudioFormat>,
}

pub struct TtsTool {
    audio_config: AudioConfig,
    audio_dir: PathBuf,
    providers: ProviderRegistry,
}

impl TtsTool {
    pub fn new(audio_config: AudioConfig, audio_dir: PathBuf, providers: ProviderRegistry) -> Self {
        Self {
            audio_config,
            audio_dir,
            providers,
        }
    }
}

#[async_trait]
impl Tool for TtsTool {
    fn name(&self) -> &str {
        "tts"
    }

    fn description(&self) -> &str {
        "Convert text to speech. Streams audio chunks for real-time playback. After generating, ALWAYS include the `@filename` marker from the result in your response so the audio is sent to the user. For Telegram, use output_format='opus' for native voice notes.\n\nTo list available voices, use `list_voices`."
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
        let args: TtsArgs = ctx.parse_args(self.name())?;

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
                "No voice configured. Set [llm.audio] voice in config or pass voice argument (use list_voices tool).",
            ));
        };

        let output_format = args.output_format.unwrap_or_default();

        tracing::debug!(text_len = args.text.len(), %model, %voice, %output_format, "TTS requested");

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

        let mut stream = provider.text_to_speech(request);

        use base64::Engine;
        use futures::StreamExt;
        let mut audio_bytes = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::AudioDelta { data, .. }) => {
                    if let Ok(chunk) = base64::engine::general_purpose::STANDARD.decode(&data) {
                        audio_bytes.extend_from_slice(&chunk);
                    }
                }
                Ok(StreamEvent::Finished(_)) => break,
                Err(e) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("TTS failed: {e}"),
                    ));
                }
                _ => {}
            }
        }

        if audio_bytes.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "TTS produced no audio data.",
            ));
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let filename = format!("speech_{timestamp}.{output_format}");
        let output_path = self.audio_dir.join(&filename);

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        std::fs::write(&output_path, &audio_bytes)?;

        let size_kb = audio_bytes.len() / 1024;
        tracing::debug!(path = %output_path.display(), size_kb, "TTS audio saved");

        let fname = output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Audio saved to {} ({size_kb}KB). Include @{fname} in your response to send it.",
                output_path.display(),
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        format!("Speaking: {}", truncate_utf8(text, 50))
    }
}

// ============================================================================
// STT (Transcribe) Tool
// ============================================================================

#[derive(serde::Deserialize)]
struct TranscribeArgs {
    file_path: String,
    model: Option<Model>,
    language: Option<String>,
}

pub struct TranscribeTool {
    default_model: Option<Model>,
    providers: ProviderRegistry,
}

impl TranscribeTool {
    pub fn new(default_model: Option<Model>, providers: ProviderRegistry) -> Self {
        Self {
            default_model,
            providers,
        }
    }
}

#[async_trait]
impl Tool for TranscribeTool {
    fn name(&self) -> &str {
        "transcribe"
    }

    fn description(&self) -> &str {
        "Transcribe audio to text (speech-to-text). Accepts audio files (mp3, wav, opus, flac, m4a, webm)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the audio file to transcribe"
                },
                "model": {
                    "type": "string",
                    "description": "STT model (e.g., 'openai:whisper-1'). Uses config default if not specified."
                },
                "language": {
                    "type": "string",
                    "description": "Language hint (ISO 639-1, e.g., 'en', 'es', 'fr')"
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: TranscribeArgs = ctx.parse_args(self.name())?;

        let model = args.model.or_else(|| self.default_model.clone());
        let Some(model) = model else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No STT model configured. Pass a model argument (e.g., 'openai:whisper-1').",
            ));
        };

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        let path = std::path::Path::new(&args.file_path);
        if !path.exists() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("File not found: {}", args.file_path),
            ));
        }

        let audio = std::fs::read(path)?;
        let media_type = match path.extension().and_then(|e| e.to_str()) {
            Some("mp3") => "audio/mpeg",
            Some("wav") => "audio/wav",
            Some("opus" | "ogg") => "audio/ogg",
            Some("flac") => "audio/flac",
            Some("m4a") => "audio/mp4",
            Some("webm") => "audio/webm",
            _ => "audio/mpeg",
        }
        .to_string();

        tracing::debug!(path = %args.file_path, size_kb = audio.len() / 1024, %model, "STT requested");

        let request = SttRequest {
            model: model.name().to_string(),
            audio,
            media_type,
            language: args.language,
        };

        let mut stream = provider.transcribe(request);

        use futures::StreamExt;
        let mut text = String::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ContentDelta(delta)) => {
                    text.push_str(&delta);
                }
                Ok(StreamEvent::Finished(_)) => break,
                Err(e) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Transcription failed: {e}"),
                    ));
                }
                _ => {}
            }
        }

        if text.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Transcription produced no text.",
            ));
        }

        Ok(ToolResult::success(ctx.tool_call_id, text))
    }

    fn humanize(&self, args: &Value) -> String {
        let path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        format!("Transcribing: {}", truncate_utf8(path, 60))
    }
}

// ============================================================================
// List Voices Tool
// ============================================================================

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

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

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
