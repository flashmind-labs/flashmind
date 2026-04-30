//! Lightweight error types for parsing — `ParseError` only.
//!
//! This error is the `Err` type for `FromStr` implementations on [`Model`](crate::model::Model),
//! [`AliasedModel`](crate::model::AliasedModel), and [`Provider`](crate::model::Provider)
//! parsing.

use thiserror::Error;

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
