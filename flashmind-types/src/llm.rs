//! LLM wire types — request / response structures and the [`LlmProvider`] trait.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use rust_decimal::Decimal;
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
    /// Sampling temperature (0.0 = deterministic, higher = more creative).
    pub temperature: Decimal,
    /// Maximum output tokens. If `None`, the provider decides.
    pub max_tokens: Option<u32>,
    /// Whether reasoning/thinking mode is enabled.
    pub reasoning: ReasoningLevel,
    /// Extended sampling parameters (top_p, top_k, min_p, penalties).
    pub sampling: SamplingParams,
    /// Output modalities (e.g. text+audio, image). When `None`, text-only.
    pub modalities: Option<Vec<Modality>>,
    /// Audio output configuration. Only used when modalities includes [`Modality::Audio`].
    pub audio_config: Option<AudioOutputConfig>,
    /// Image generation configuration. Only used when modalities includes [`Modality::Image`].
    pub image_config: Option<ImageGenConfig>,
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

/// Pricing per token (USD).
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

/// Static metadata about a known model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    /// Human-readable display name.
    pub name: Option<String>,
    pub context_length: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub capabilities: ModelCapabilities,
    pub categories: Vec<ModelCategory>,
    pub pricing: ModelPricing,
}

/// Feature flags describing what a model supports.
///
/// Defaults to conservative values (`tool_calling`, `images`, `documents` only);
/// providers override these after inspecting the model's actual capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub tool_calling: bool,
    pub images: bool,
    pub documents: bool,
    pub video: bool,
    pub audio: bool,
    pub reasoning: bool,
    // Output capabilities
    pub audio_output: bool,
    pub image_generation: bool,
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

    /// Maximum feature set — assumes all capabilities are available.
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

/// Token consumption as reported by the provider.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
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
    pub message: Message,
    pub usage: TokenUsage,
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
    /// Incremental audio output chunk (base64-encoded).
    AudioDelta {
        data: String,
        format: String,
    },
    /// Out-of-band file delivered by the provider (e.g. generated image).
    FileAttachment {
        filename: String,
        media_type: String,
        /// Base64-encoded payload.
        data: String,
    },
    /// Stream has ended; payload is the final finish reason.
    Finished(FinishReason),
}

/// Alias for the boxed async stream type used by all providers.
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + 'static>>;

/// Available voice preset for TTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    pub name: String,
}

/// Text-to-speech request sent to providers that support TTS.
pub struct TtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub response_format: AudioFormat,
}

/// Speech-to-text (transcription) request.
pub struct SttRequest {
    pub model: String,
    /// Raw audio bytes.
    pub audio: Vec<u8>,
    /// MIME type of the audio (e.g. "audio/mp3", "audio/wav").
    pub media_type: String,
    /// Optional language hint (ISO 639-1, e.g. "en").
    pub language: Option<String>,
}

/// Output audio format for TTS responses.
#[derive(
    Debug, Clone, Copy, Default, strum::EnumString, strum::Display, Serialize, Deserialize,
)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    #[default]
    Mp3,
    Wav,
    Opus,
    Aac,
    Flac,
}

/// Output modality requested in a completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Audio,
    Image,
}

/// Configuration for audio output in completions.
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    pub voice: String,
    pub format: AudioFormat,
}

/// Configuration for image generation in completions.
#[derive(Debug, Clone, Default)]
pub struct ImageGenConfig {
    pub aspect_ratio: Option<String>,
    pub size: Option<String>,
    /// URLs of reference images for style/quality guidance (max 4).
    pub reference_images: Vec<String>,
}

/// Video generation request.
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    pub model: String,
    pub description: String,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub duration: Option<u32>,
    pub generate_audio: Option<bool>,
    /// Image URLs for first/last frame (image-to-video).
    pub frame_images: Vec<FrameImage>,
    /// Reference image URLs for style guidance.
    pub input_references: Vec<String>,
}

/// A frame image for image-to-video generation.
#[derive(Debug, Clone)]
pub struct FrameImage {
    pub url: String,
    /// `"first_frame"` or `"last_frame"`.
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
            Err(anyhow::anyhow!("Provider '{name}' does not support text-to-speech"))
        }))
    }

    /// Transcribe audio to text. Returns a stream of [`StreamEvent::ContentDelta`] chunks,
    /// ending with [`StreamEvent::Finished`].
    fn transcribe(&self, _request: SttRequest) -> CompletionStream {
        let name = self.name().to_string();
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!("Provider '{name}' does not support speech-to-text"))
        }))
    }

    /// List available voices for TTS. Return `None` if unsupported.
    async fn list_voices(&self, _model: &str) -> Option<Vec<Voice>> {
        None
    }

    /// Generate a video. Returns a stream that yields progress updates as
    /// [`StreamEvent::ContentDelta`] and the final video as [`StreamEvent::FileAttachment`].
    /// Default implementation returns an error.
    fn generate_video(&self, _request: VideoGenRequest) -> CompletionStream {
        Box::pin(futures::stream::once(async move {
            Err(anyhow::anyhow!("Provider does not support video generation"))
        }))
    }

    /// Update URL routing table (for providers like Ollama with multiple backends).
    fn update_routing(&self, _routing: &HashMap<String, String>) {}
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
