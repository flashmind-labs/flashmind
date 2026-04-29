use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSet {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    pub scopes: Vec<String>,
}

impl TokenSet {
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => Utc::now() + chrono::Duration::minutes(5) >= exp,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unexpired_token() {
        let ts = TokenSet {
            access_token: "abc".into(),
            refresh_token: None,
            expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
            scopes: vec!["read".into()],
        };
        assert!(!ts.is_expired());
    }

    #[test]
    fn expired_token() {
        let ts = TokenSet {
            access_token: "abc".into(),
            refresh_token: None,
            expires_at: Some(Utc::now() - chrono::Duration::hours(1)),
            scopes: vec![],
        };
        assert!(ts.is_expired());
    }

    #[test]
    fn within_buffer_counts_as_expired() {
        let ts = TokenSet {
            access_token: "abc".into(),
            refresh_token: None,
            expires_at: Some(Utc::now() + chrono::Duration::minutes(3)),
            scopes: vec![],
        };
        assert!(ts.is_expired());
    }

    #[test]
    fn no_expiry_never_expires() {
        let ts = TokenSet {
            access_token: "abc".into(),
            refresh_token: None,
            expires_at: None,
            scopes: vec![],
        };
        assert!(!ts.is_expired());
    }
}
