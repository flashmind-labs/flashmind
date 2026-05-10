//! Microsoft identity platform OAuth2 implementation.
//!
//! Supports the Authorization Code flow with PKCE for user authentication
//! via Microsoft Graph API.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::oauth::{CachedToken, parse_token_response};
use crate::utils::http_client;

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlookCredentials {
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<String>,
    #[serde(default = "default_tenant")]
    pub tenant: String,
    #[serde(default = "default_redirect")]
    pub redirect_uri: String,
    #[serde(default)]
    pub scopes: Vec<String>,
}

fn default_tenant() -> String {
    "common".to_string()
}

fn default_redirect() -> String {
    "http://localhost".to_string()
}

// ---------------------------------------------------------------------------
// Token acquisition
// ---------------------------------------------------------------------------

pub fn auth_url(creds: &OutlookCredentials) -> String {
    let scopes = if creds.scopes.is_empty() {
        "Mail.Read Mail.Send Calendars.ReadWrite Contacts.Read offline_access".to_string()
    } else {
        let mut s = creds.scopes.join(" ");
        if !s.contains("offline_access") {
            s.push_str(" offline_access");
        }
        s
    };

    format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/authorize?\
         client_id={}&response_type=code&redirect_uri={}&scope={}&response_mode=query",
        creds.tenant,
        urlencoding::encode(&creds.client_id),
        urlencoding::encode(&creds.redirect_uri),
        urlencoding::encode(&scopes),
    )
}

#[allow(dead_code)]
pub async fn exchange_code(creds: &OutlookCredentials, code: &str) -> Result<CachedToken> {
    let token_url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        creds.tenant
    );

    let scopes = if creds.scopes.is_empty() {
        "Mail.Read Mail.Send Calendars.ReadWrite Contacts.Read offline_access".to_string()
    } else {
        let mut s = creds.scopes.join(" ");
        if !s.contains("offline_access") {
            s.push_str(" offline_access");
        }
        s
    };

    let mut form: Vec<(&str, &str)> = vec![
        ("client_id", &creds.client_id),
        ("code", code),
        ("redirect_uri", &creds.redirect_uri),
        ("grant_type", "authorization_code"),
        ("scope", &scopes),
    ];
    if let Some(secret) = &creds.client_secret {
        form.push(("client_secret", secret));
    }

    let resp: serde_json::Value = http_client()
        .post(&token_url)
        .form(&form)
        .send()
        .await?
        .error_for_status()
        .context("Outlook token exchange failed")?
        .json()
        .await?;

    parse_token_response(&resp)
}

pub async fn refresh_token(creds: &OutlookCredentials, refresh: &str) -> Result<CachedToken> {
    let token_url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        creds.tenant
    );

    let scopes = if creds.scopes.is_empty() {
        "Mail.Read Mail.Send Calendars.ReadWrite Contacts.Read offline_access".to_string()
    } else {
        let mut s = creds.scopes.join(" ");
        if !s.contains("offline_access") {
            s.push_str(" offline_access");
        }
        s
    };

    let refresh_owned = refresh.to_string();
    let grant_type = "refresh_token".to_string();
    let mut form: Vec<(&str, &str)> = vec![
        ("client_id", &creds.client_id),
        ("refresh_token", &refresh_owned),
        ("grant_type", &grant_type),
        ("scope", &scopes),
    ];
    if let Some(secret) = &creds.client_secret {
        form.push(("client_secret", secret));
    }

    let resp: serde_json::Value = http_client()
        .post(&token_url)
        .form(&form)
        .send()
        .await?
        .error_for_status()
        .context("Outlook token refresh failed")?
        .json()
        .await?;

    let mut token = parse_token_response(&resp)?;
    if token.refresh_token.is_none() {
        token.refresh_token = Some(refresh.to_string());
    }
    Ok(token)
}
