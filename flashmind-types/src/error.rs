//! Error types for the Flashmind framework.
//!
//! Contains [`ParseError`] for `FromStr` failures and [`LlmError`] for typed
//! classification of LLM provider errors.

use thiserror::Error;

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

/// A string failed to parse into a typed variant (model ID, ChatKey, etc.).
///
/// The public [`String`](#structfield.0) field contains the human-readable error
/// message produced during parsing.
///
/// This type is used as the `Err` for `FromStr` on [`Model`](crate::model::Model),
/// [`AliasedModel`](crate::model::AliasedModel), and [`Provider`](crate::model::Provider).
///
/// # Example
///
/// ```ignore
/// use std::str::FromStr;
/// use crate::model::Model;
/// use crate::error::ParseError;
///
/// let result = Model::from_str("invalid-format");
/// assert!(matches!(result, Err(ParseError(_))));
///
/// if let Err(ParseError(msg)) = result {
///     eprintln!("parse failed: {msg}");
/// }
/// ```
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParseError(pub String);

impl ParseError {
    /// Construct a `ParseError` with the given message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

// ---------------------------------------------------------------------------
// LlmError
// ---------------------------------------------------------------------------

/// Classification of LLM provider errors.
///
/// Providers emit this as a concrete error type inside `anyhow::Error` so that
/// the agent loop can downcast and match on [`LlmErrorKind`] instead of
/// fragile string matching.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct LlmError {
    /// Semantic category of the error.
    pub kind: LlmErrorKind,
    /// Human-readable error message (typically includes the provider response body).
    pub message: String,
}

/// Semantic category for an [`LlmError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmErrorKind {
    /// The prompt exceeds the model's context window.
    ContextLengthExceeded,
    /// The model produced a tool call that could not be parsed (malformed JSON, etc.).
    MalformedToolCall,
    /// The provider returned a rate-limit response that was not resolved by retries.
    RateLimited,
    /// Any other provider error that does not fit a specific category.
    Other,
}

impl LlmError {
    /// Create a new `LlmError` with the given kind and message.
    pub fn new(kind: LlmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Classify an error message string into the appropriate [`LlmErrorKind`].
    ///
    /// This inspects the message for well-known patterns returned by LLM APIs
    /// (OpenAI, Anthropic, Ollama, etc.) and picks the most specific kind.
    pub fn classify(message: impl Into<String>) -> Self {
        let message = message.into();
        let kind = Self::classify_message(&message);
        Self { kind, message }
    }

    /// Determine the [`LlmErrorKind`] for an error message.
    fn classify_message(msg: &str) -> LlmErrorKind {
        // Context length patterns from various providers
        if msg.contains("maximum context length")
            || msg.contains("context_length_exceeded")
            || msg.contains("too many tokens")
            || msg.contains("exceeds the model's context")
            || msg.contains("reduce the length of the input")
            || msg.contains("maximum input length")
            || msg.contains("prompt is too long")
            || msg.contains("input too long")
        {
            return LlmErrorKind::ContextLengthExceeded;
        }

        // Malformed tool call / JSON parse failure
        if msg.contains("failed to parse JSON")
            || msg.contains("invalid_request_error") && msg.contains("tool")
        {
            return LlmErrorKind::MalformedToolCall;
        }

        // Rate limiting (post-retry — send_with_retry already handles transient 429s)
        if msg.contains("rate_limit") || msg.contains("Rate limit") {
            return LlmErrorKind::RateLimited;
        }

        LlmErrorKind::Other
    }

    /// Returns `true` if this error is recoverable through compaction or
    /// conversation truncation.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.kind,
            LlmErrorKind::ContextLengthExceeded | LlmErrorKind::MalformedToolCall
        )
    }
}

impl LlmErrorKind {
    /// Human-readable label for this error kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ContextLengthExceeded => "context_length_exceeded",
            Self::MalformedToolCall => "malformed_tool_call",
            Self::RateLimited => "rate_limited",
            Self::Other => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_context_length_openai() {
        let err = LlmError::classify(
            "OpenAI error: API error 400: {\"error\":{\"message\":\"This model's maximum context length is 128000 tokens\",\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\"}}",
        );
        assert_eq!(err.kind, LlmErrorKind::ContextLengthExceeded);
        assert!(err.is_recoverable());
    }

    #[test]
    fn classify_context_length_anthropic() {
        let err = LlmError::classify(
            "Anthropic error: API error 400: prompt is too long: 200000 tokens > 200000 maximum",
        );
        assert_eq!(err.kind, LlmErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn classify_context_length_too_many_tokens() {
        let err = LlmError::classify("error: too many tokens in the input");
        assert_eq!(err.kind, LlmErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn classify_context_length_reduce_input() {
        let err = LlmError::classify("Please reduce the length of the input messages");
        assert_eq!(err.kind, LlmErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn classify_malformed_tool_call() {
        let err = LlmError::classify("failed to parse JSON for tool call arguments");
        assert_eq!(err.kind, LlmErrorKind::MalformedToolCall);
        assert!(err.is_recoverable());
    }

    #[test]
    fn classify_rate_limited() {
        let err = LlmError::classify("Rate limit exceeded, please retry after 30s");
        assert_eq!(err.kind, LlmErrorKind::RateLimited);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn classify_other() {
        let err = LlmError::classify("Internal server error");
        assert_eq!(err.kind, LlmErrorKind::Other);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn llm_error_display() {
        let err = LlmError::new(LlmErrorKind::ContextLengthExceeded, "too long");
        assert_eq!(err.to_string(), "too long");
    }

    #[test]
    fn llm_error_downcast_from_anyhow() {
        let anyhow_err: anyhow::Error =
            LlmError::new(LlmErrorKind::ContextLengthExceeded, "test").into();
        let llm_err = anyhow_err.downcast_ref::<LlmError>().unwrap();
        assert_eq!(llm_err.kind, LlmErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn llm_error_kind_as_str() {
        assert_eq!(
            LlmErrorKind::ContextLengthExceeded.as_str(),
            "context_length_exceeded"
        );
        assert_eq!(
            LlmErrorKind::MalformedToolCall.as_str(),
            "malformed_tool_call"
        );
        assert_eq!(LlmErrorKind::RateLimited.as_str(), "rate_limited");
        assert_eq!(LlmErrorKind::Other.as_str(), "other");
    }
}
