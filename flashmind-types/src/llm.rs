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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A single completion request sent to an LLM provider.
#[derive(Debug)]
pub struct CompletionRequest {
    pub model: Model,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub temperature: Decimal,
    pub max_tokens: Option<u32>,
    pub reasoning: ReasoningLevel,
    pub sampling: SamplingParams,
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

/// Static metadata about a known model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: Option<u32>,
    pub capabilities: ModelCapabilities,
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
}

impl ModelCapabilities {
    /// Maximum feature set — assumes all capabilities are available.
    pub fn all() -> Self {
        Self {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: true,
            reasoning: true,
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

/// Trait implemented by each LLM backend (OpenRouter, Anthropic, Ollama, etc.).
///
/// Providers are registered in a [`ProviderRegistry`] at startup and selected
/// per-turn based on the active model's [`Model::provider`] field.
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

    /// Probe or return hard-coded capabilities for `model`.
    async fn capabilities(&self, _model: &Model) -> ModelCapabilities {
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
        }
    }

    /// List models available from this provider. Return `None` if unsupported.
    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        None
    }

    /// Issue a completion request and return a stream of [`StreamEvent`]s.
    fn complete(&self, request: CompletionRequest) -> CompletionStream;

    /// Synthesise speech from text. Default implementation returns an error.
    async fn text_to_speech(&self, _request: TtsRequest) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("Provider '{}' does not support text-to-speech", self.name())
    }

    /// List available voices for TTS. Return `None` if unsupported.
    async fn list_voices(&self, _model: &str) -> Option<Vec<Voice>> {
        None
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
