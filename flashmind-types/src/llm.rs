//! LLM wire types — request / response structures and the [`LlmProvider`] trait.
//!
//! This module defines everything needed to communicate with an LLM backend:
//! the provider trait, completion requests and responses, streaming events,
//! model capabilities, and multimodal support (audio, images, video).
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`LlmProvider`] | Trait implemented by each LLM backend (OpenRouter, Anthropic, Ollama…) |
//! | [`CompletionRequest`] | Single completion request sent to a provider |
//! | [`CompletionResponse`] | Non-streaming response wrapper |
//! | [`StreamEvent`] | Incremental events from the provider's stream |
//! | [`CompletionStream`] | Alias for `Pin<Box<dyn Stream<Item = Result<StreamEvent>>>>` |
//! | [`FinishReason`] | Why generation stopped (stop, length, tool_calls, content_filter) |
//! | [`ToolDefinition`] | OpenAI-compatible JSON schema for tool calling |
//! | [`ModelCapabilities`] | Feature flags (tool_calling, images, reasoning, audio…) |
//! | [`TokenUsage`] | Token consumption reported by the provider |
//! | [`Modality`] | Output modality: text, audio, or image |
//! | [`TtsRequest`], [`SttRequest`], [`VideoGenRequest`] | Multimodal operation requests |

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::event::TurnUsage;
use crate::message::Message;
use crate::model::{Model, Provider, ReasoningLevel, SamplingParams};

/// OpenAI-compatible JSON schema for a tool exposed to the LLM.
///
/// Mirrors the structure sent in the `tools` array of chat completion requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name — must match the [`Tool::name`](crate::tool::Tool::name) implementation.
    pub name: String,
    /// Description shown to the model to help it decide when to call this tool.
    pub description: String,
    /// JSON Schema object describing the parameters this tool accepts.
    pub parameters: serde_json::Value,
}

/// A single completion request sent to an LLM provider.
///
/// This is the primary input type for [`LlmProvider::complete`]. The agent runtime
/// assembles this from the conversation history, active tools, and LLM config.
#[derive(Debug)]
pub struct CompletionRequest {
    /// Model to use (includes provider prefix).
    pub model: Model,
    /// Message history in wire format.
    pub messages: Vec<Message>,
    /// Tools available to the model in this turn.
    pub tools: Vec<ToolDefinition>,
    /// Maximum output tokens. If `None`, the provider decides.
    pub max_tokens: Option<u32>,
    /// Whether reasoning/thinking mode is enabled.
    pub reasoning: ReasoningLevel,
    /// Sampling parameters (temperature, top_p, top_k, min_p, penalties).
    pub sampling: SamplingParams,
    /// Output modalities (e.g. text+audio, image). Empty means text-only.
    pub modalities: Vec<Modality>,
    /// Audio output configuration. Only used when modalities includes [`Modality::Audio`].
    pub audio_config: Option<AudioOutputConfig>,
    /// Image generation configuration. Only used when modalities includes [`Modality::Image`].
    pub image_config: Option<ImageGenConfig>,
    /// Opaque user identifier for per-user tracking in provider dashboards.
    pub user: Option<String>,
}

/// Why the LLM stopped generating (used in stream and non-stream responses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop token or end of response.
    Stop,
    /// Model requested one or more tool calls.
    ToolCalls,
    /// Hit `max_tokens` limit without natural stopping point.
    Length,
    /// Output was filtered by content policy.
    ContentFilter,
}

/// High-level category for filtering models by use-case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum ModelCategory {
    /// General text chat / instruction-following.
    Chat,
    /// Advanced reasoning / chain-of-thought.
    Reasoning,
    /// Can accept images as input.
    Vision,
    /// Generates images from text.
    ImageGeneration,
    /// Generates audio / text-to-speech.
    Tts,
    /// Accepts audio input / speech-to-text.
    Stt,
    /// Generates video.
    VideoGeneration,
    /// Accepts video as input.
    VideoInput,
}

/// Pricing information for a model, including input/output token costs and caching discounts.
#[derive(Debug, Clone, Default)]
pub struct ModelPricing {
    /// Cost per input token (USD).
    pub prompt: Option<f64>,
    /// Cost per output token (USD).
    pub completion: Option<f64>,
    /// Cost per image generated (USD).
    pub image: Option<f64>,
    /// Cost per cached input token read (USD).
    pub cache_read: Option<f64>,
}

/// Metadata about an available LLM model, including capabilities, pricing, and context window.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID used in API requests.
    pub id: String,
    /// Human-readable display name.
    pub name: Option<String>,
    /// Context window size in tokens.
    pub context_length: Option<u32>,
    /// Maximum output tokens allowed.
    pub max_completion_tokens: Option<u32>,
    /// Capability flags.
    pub capabilities: ModelCapabilities,
    /// High-level categories (chat, vision, reasoning, etc.).
    pub categories: Vec<ModelCategory>,
    /// Per-token pricing.
    pub pricing: ModelPricing,
}

/// Feature flags describing what a model supports.
///
/// Defaults to all `false` via `#[derive(Default)]` — nothing is assumed.
/// Providers override these after inspecting the model's actual capabilities.
/// Use [`all()`](Self::all()) for a curated "full-featured text model" preset.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    /// Can invoke tools/function calling.
    pub tool_calling: bool,
    /// Can accept images as input (vision).
    pub images: bool,
    /// Can accept documents (PDFs, etc.) as input.
    pub documents: bool,
    /// Can accept video as input.
    pub video: bool,
    /// Can accept audio as input (speech-to-text).
    pub audio: bool,
    /// Supports extended chain-of-thought / reasoning mode.
    pub reasoning: bool,
    // Output capabilities
    /// Can generate audio output (text-to-speech inline).
    pub audio_output: bool,
    /// Can generate images as output.
    pub image_generation: bool,
    /// Can generate video as output.
    pub video_generation: bool,
}

impl ModelCapabilities {
    /// Derive high-level categories from capability flags.
    pub fn categories(&self) -> Vec<ModelCategory> {
        let mut cats = vec![ModelCategory::Chat];
        if self.reasoning {
            cats.push(ModelCategory::Reasoning);
        }
        if self.images {
            cats.push(ModelCategory::Vision);
        }
        if self.image_generation {
            cats.push(ModelCategory::ImageGeneration);
        }
        if self.audio_output {
            cats.push(ModelCategory::Tts);
        }
        if self.audio {
            cats.push(ModelCategory::Stt);
        }
        if self.video_generation {
            cats.push(ModelCategory::VideoGeneration);
        }
        if self.video {
            cats.push(ModelCategory::VideoInput);
        }
        cats
    }

    /// Curated "full-featured text model" capability preset.
    ///
    /// Enables common features (tool calling, vision, reasoning, audio input)
    /// but keeps output-generation capabilities (`audio_output`, `image_generation`,
    /// `video_generation`) disabled, as those require dedicated models.
    pub fn all() -> Self {
        Self {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: true,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        }
    }
}

/// Token usage counts for a single completion request or turn.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    /// Tokens consumed by the prompt (input messages).
    pub prompt_tokens: u32,
    /// Tokens generated by the model (output/response).
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
}

impl From<TurnUsage> for TokenUsage {
    fn from(u: TurnUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.prompt_tokens + u.completion_tokens,
        }
    }
}

/// Non-streaming completion response wrapper.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The assistant's response message.
    pub message: Message,
    /// Token usage for this completion.
    pub usage: TokenUsage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// Events emitted incrementally by the provider's stream.
///
/// Each variant corresponds to a server-side SSE/data-line event type.
/// Consumers assemble these into higher-level [`AgentEvent`](crate::event::AgentEvent)s.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental reasoning/chain-of-thought token.
    ReasoningDelta(String),
    /// Incremental assistant text token.
    ContentDelta(String),
    /// The LLM started requesting a tool call at `index`.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// Incremental JSON argument string for the tool at `index`.
    ToolCallDelta { index: usize, arguments: String },
    /// Token usage snapshot mid-stream (some providers emit this).
    Usage(TokenUsage),
    /// Incremental audio output chunk (raw PCM/MP3 bytes).
    AudioDelta { data: Vec<u8>, format: String },
    /// Out-of-band file delivered by the provider (e.g. generated image).
    FileAttachment {
        filename: String,
        media_type: String,
        data: Vec<u8>,
    },
    /// Stream has ended; payload is the final finish reason.
    Finished(FinishReason),
}

/// Type alias for a stream of completion events from an LLM provider.
///
/// The stream yields `StreamEvent`s (content deltas, tool calls, finish reasons) and is consumed by the agent loop.
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + 'static>>;

/// A voice available for text-to-speech generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// Unique voice identifier (e.g., `"alloy"`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
}

/// Parameters for a text-to-speech request.
pub struct TtsRequest {
    /// TTS model identifier (e.g., `"tts-1"`).
    pub model: String,
    /// Text to synthesize into speech.
    pub input: String,
    /// Voice preset ID (e.g., `"alloy"`, `"echo"`).
    pub voice: String,
    /// Desired output audio format.
    pub response_format: AudioFormat,
}

/// Parameters for a speech-to-text (transcription) request.
pub struct SttRequest {
    /// STT model identifier (e.g., `"whisper-1"`).
    pub model: String,
    /// Raw audio bytes.
    pub audio: Vec<u8>,
    /// MIME type of the audio (e.g. "audio/mp3", "audio/wav").
    pub media_type: String,
    /// Optional language hint (ISO 639-1, e.g. "en").
    pub language: Option<String>,
}

/// Output audio format for TTS responses.
///
/// Serialises to lowercase strings matching common audio file extensions.
#[derive(
    Debug, Clone, Copy, Default, strum::EnumString, strum::Display, Serialize, Deserialize,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MP3 — most widely supported, good compression.
    #[default]
    Mp3,
    /// WAV — uncompressed PCM, highest quality.
    Wav,
    /// Opus — efficient codec optimized for speech.
    Opus,
    /// AAC — Advanced Audio Coding, common in mobile ecosystems.
    Aac,
    /// FLAC — lossless compression.
    Flac,
}

/// Output modality requested in a completion.
///
/// Multiple modalities can be requested simultaneously (e.g., text + audio).
/// An empty list means text-only output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, strum::Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Modality {
    /// Plain text output (default when no modalities specified).
    Text,
    /// Inline audio output (real-time voice).
    Audio,
    /// Image output (generated during completion).
    Image,
}

/// Configuration for audio output in multimodal completions.
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    /// Voice preset ID (e.g., `"alloy"`).
    pub voice: String,
    /// Output audio format.
    pub format: AudioFormat,
}

/// Configuration for image generation requests.
#[derive(Debug, Clone, Default)]
pub struct ImageGenConfig {
    /// Aspect ratio for the generated image (e.g., `"16:9"`).
    pub aspect_ratio: Option<String>,
    /// Absolute dimensions (e.g., `"1024x1024"`).
    pub size: Option<String>,
}

impl ImageGenConfig {
    /// Read an image file and return it as a `data:<mime>;base64,...` URI.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file on disk.
    ///
    /// # Returns
    ///
    /// A data URI string with auto-detected MIME type, or an I/O error if the file cannot be read.
    pub fn data_uri_from_path(path: &std::path::Path) -> std::io::Result<String> {
        let bytes = std::fs::read(path)?;
        let mime = mime_from_extension(path);
        Ok(Self::data_uri_from_bytes(&bytes, mime))
    }

    /// Encode raw bytes as a `data:<mime>;base64,...` URI.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw binary data (e.g., image file contents).
    /// * `mime` - MIME type string (e.g., `"image/png"`).
    ///
    /// # Returns
    ///
    /// A data URI string with the provided MIME type and base64-encoded payload.
    pub fn data_uri_from_bytes(bytes: &[u8], mime: &str) -> String {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        format!("data:{mime};base64,{b64}")
    }
}

/// Infer MIME type from a file path's extension.
///
/// # Arguments
///
/// * `path` - File path whose extension is inspected.
///
/// # Returns
///
/// A static MIME type string corresponding to the extension, or `"application/octet-stream"`
/// if the extension is unrecognized or missing.
pub fn mime_from_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

/// Parameters for generating an image via a generative model.
#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    /// Image generation model identifier.
    pub model: String,
    /// Text description of the desired image.
    pub prompt: String,
    /// Image dimensions (e.g., `"1024x1024"`).
    pub size: Option<String>,
    /// Aspect ratio (e.g., `"16:9"`).
    pub aspect_ratio: Option<String>,
    /// Quality preset (e.g., `"standard"`, `"hd"`).
    pub quality: Option<String>,
    /// Style preset (e.g., `"natural"`, `"vivid"`).
    pub style: Option<String>,
    /// Number of images to generate.
    pub n: Option<u32>,
}

/// Parameters for generating a video from text prompts or image frames.
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    /// Video generation model identifier.
    pub model: String,
    /// Text description of the desired video.
    pub description: String,
    /// Video resolution (e.g., `"720p"`, `"1080p"`).
    pub resolution: Option<String>,
    /// Aspect ratio (e.g., `"16:9"`).
    pub aspect_ratio: Option<String>,
    /// Duration in seconds.
    pub duration: Option<u32>,
    /// Whether to generate accompanying audio.
    pub generate_audio: Option<bool>,
    /// Image URLs for first/last frame (image-to-video).
    pub frame_images: Vec<FrameImage>,
    /// Reference image URLs for style guidance.
    pub input_references: Vec<String>,
}

/// An image used as a frame reference in video generation.
#[derive(Debug, Clone)]
pub struct FrameImage {
    /// URL or data URI of the frame image.
    pub url: String,
    /// Frame position — `"first_frame"` or `"last_frame"`.
    pub frame_type: String,
}

/// Trait implemented by each LLM backend (OpenRouter, Anthropic, Ollama, etc.).
///
/// Providers are registered in a [`ProviderRegistry`] at startup and selected
/// per-turn based on the active model's [`Model::provider`](crate::model::Model::provider) field.
///
/// # Required method
///
/// Only [`complete`](Self::complete) is required — it returns a stream of [`StreamEvent`]s.
/// All other methods have default implementations suitable for simple providers.
///
/// # Concurrency
///
/// Implementations must be `Send + Sync` since they're shared via `Arc` across async tasks.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Human-readable name of this provider (for logs and error messages).
    fn name(&self) -> &str;

    /// Which [`crate::model::Provider`] variant this instance implements.
    fn provider(&self) -> Provider;

    /// Query the provider for the context window size of `model`. Return `None`
    /// if the provider doesn't expose this information.
    async fn context_window(&self, _model: &Model) -> Option<u32> {
        None
    }

    /// Probe or return hard-coded capabilities for `model`. Used to detect features like reasoning or multimodal support.
    async fn capabilities(&self, _model: &Model) -> ModelCapabilities {
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            ..Default::default()
        }
    }

    /// List models available from this provider. Return `None` if unsupported.
    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        None
    }

    /// Issue a completion request and return a stream of [`StreamEvent`]s.
    ///
    /// This is the core method that every provider must implement. The returned
    /// stream should yield incremental events (text deltas, tool calls, usage)
    /// and terminate with a [`StreamEvent::Finished`] event.
    fn complete(&self, request: CompletionRequest) -> CompletionStream;

    /// Synthesise speech from text as a stream of [`StreamEvent::AudioDelta`] chunks,
    /// ending with [`StreamEvent::Finished`]. Consumers can play chunks as they arrive
    /// for real-time playback.
    fn text_to_speech(&self, _request: TtsRequest) -> CompletionStream {
        let name = self.name().to_string();
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!(
                "Provider '{name}' does not support text-to-speech"
            ))
        }))
    }

    /// Transcribe audio to text. Returns a stream of [`StreamEvent::ContentDelta`] chunks,
    /// ending with [`StreamEvent::Finished`].
    fn transcribe(&self, _request: SttRequest) -> CompletionStream {
        let name = self.name().to_string();
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!(
                "Provider '{name}' does not support speech-to-text"
            ))
        }))
    }

    /// List available voices for TTS. Return `None` if unsupported.
    async fn list_voices(&self, _model: &str) -> Option<Vec<Voice>> {
        None
    }

    /// Generate an image via a dedicated image API (e.g. DALL-E).
    /// Default implementation returns an error.
    fn generate_image(&self, _request: ImageGenRequest) -> CompletionStream {
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!(
                "Provider does not support image generation"
            ))
        }))
    }

    // TODO: add video editing support (video in + prompt → edited video out).
    // Runway has the most mature API; Kling also supports it.
    // Neither is available via OpenRouter — needs direct provider impls.

    /// Generate a video. Returns a stream that yields progress updates as
    /// [`StreamEvent::ContentDelta`] and the final video as [`StreamEvent::FileAttachment`].
    /// Default implementation returns an error.
    fn generate_video(&self, _request: VideoGenRequest) -> CompletionStream {
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!(
                "Provider does not support video generation"
            ))
        }))
    }

    /// Update URL routing table (for providers like Ollama with multiple backends).
    fn update_routing(&self, _routing: &HashMap<String, String>) {}

    /// Wrap `self` in an `Arc<dyn LlmProvider>` for use with [`Agent`](crate) builders.
    fn into_arc(self) -> Arc<dyn LlmProvider>
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

/// Shared registry mapping each [`crate::model::Provider`] to its concrete instance.
///
/// Created once at startup via the CLI; cloned into every agent so they can
/// switch providers without reconnecting.
pub type ProviderRegistry = Arc<HashMap<Provider, Arc<dyn LlmProvider>>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TurnUsage;

    #[test]
    fn token_usage_from_turn_usage() {
        let turn = TurnUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
        };
        let token: TokenUsage = turn.into();
        assert_eq!(token.prompt_tokens, 100);
        assert_eq!(token.completion_tokens, 50);
        assert_eq!(token.total_tokens, 150);
    }

    #[test]
    fn token_usage_default_is_zero() {
        let usage = TokenUsage::default();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn model_capabilities_all() {
        let caps = ModelCapabilities::all();
        assert!(caps.tool_calling);
        assert!(caps.images);
        assert!(caps.reasoning);
    }

    #[test]
    fn model_capabilities_default_is_conservative() {
        let caps = ModelCapabilities::default();
        assert!(!caps.tool_calling);
        assert!(!caps.reasoning);
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::ToolCalls.to_string(), "tool_calls");
        assert_eq!(FinishReason::Length.to_string(), "length");
    }

    #[test]
    fn finish_reason_from_str() {
        let stop: FinishReason = "stop".parse().unwrap();
        assert_eq!(stop, FinishReason::Stop);
        let tc: FinishReason = "tool_calls".parse().unwrap();
        assert_eq!(tc, FinishReason::ToolCalls);
    }

    #[test]
    fn tool_definition_serde() {
        let def = ToolDefinition {
            name: "file_read".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        };
        let json = serde_json::to_string(&def).unwrap();
        let parsed: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "file_read");
    }

    #[test]
    fn audio_format_default() {
        assert_eq!(AudioFormat::default().to_string(), "mp3");
    }
}
