//! Stable session identities for provider cache routing.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Reuse one ID for every request in a conversation, including after resume.
/// Create a new ID for a new conversation or convert its existing UUID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionId(String);

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionIdError {
    #[error("session ID must not be empty")]
    Empty,
    #[error("session ID exceeds 256 characters")]
    TooLong,
}

impl SessionId {
    pub const MAX_LEN: usize = 256;

    /// Generate an ID once when starting a conversation.
    pub fn new() -> Self {
        Uuid::new_v4().into()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for SessionId {
    fn from(value: Uuid) -> Self {
        Self(value.to_string())
    }
}

impl TryFrom<String> for SessionId {
    type Error = SessionIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(SessionIdError::Empty);
        }
        if value.chars().count() > Self::MAX_LEN {
            return Err(SessionIdError::TooLong);
        }
        Ok(Self(value))
    }
}

impl FromStr for SessionId {
    type Err = SessionIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.to_owned().try_into()
    }
}

impl From<SessionId> for String {
    fn from(value: SessionId) -> Self {
        value.0
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_existing_uuid_and_saved_id() {
        let uuid = Uuid::new_v4();
        let id = SessionId::from(uuid);
        assert_eq!(id.as_str(), uuid.to_string());
        let saved = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<SessionId>(&saved).unwrap(), id);
        assert_eq!(
            "saved-session".parse::<SessionId>().unwrap().as_str(),
            "saved-session"
        );
    }

    #[test]
    fn enforces_length_when_parsing_and_deserializing() {
        assert_eq!("".parse::<SessionId>(), Err(SessionIdError::Empty));
        assert!("x".repeat(256).parse::<SessionId>().is_ok());
        assert!("é".repeat(256).parse::<SessionId>().is_ok());
        let too_long = "x".repeat(257);
        assert_eq!(too_long.parse::<SessionId>(), Err(SessionIdError::TooLong));
        assert!(serde_json::from_value::<SessionId>(serde_json::json!(too_long)).is_err());
    }
}
