//! Lightweight error types for parsing — `ParseError` only.

use thiserror::Error;

/// A string failed to parse into a typed variant (model ID, ChatKey, etc.).
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParseError(pub String);

impl ParseError {
    /// Construct a `ParseError` with the given message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}
