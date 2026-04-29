use std::collections::HashMap;

use anyhow::{Result, bail};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use url::Url;

use super::credentials::TokenSet;

// ---------------------------------------------------------------------------
// Provider presets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OAuthProviderPreset {
    Google,
    Microsoft,
}

impl OAuthProviderPreset {
    fn defaults(&self) -> (&str, &str) {
        match self {
            Self::Google => (
                "https://accounts.google.com/o/oauth2/v2/auth",
                "https://oauth2.googleapis.com/token",
            ),
            Self::Microsoft => (
                "https://login.microsoftonline.com/common/oauth2/v2/authorize",
                "https://login.microsoftonline.com/common/oauth2/v2/token",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthProviderConfig {
    pub preset: Option<OAuthProviderPreset>,
    pub client_id: String,
    pub client_secret: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
}

impl OAuthProviderConfig {
    pub fn auth_url(&self) -> Result<String> {
        if let Some(ref url) = self.auth_url {
            return Ok(url.clone());
        }
        if let Some(preset) = self.preset {
            return Ok(preset.defaults().0.to_string());
        }
        bail!("OAuth provider has no auth_url and no preset")
    }

    pub fn token_url(&self) -> Result<String> {
        if let Some(ref url) = self.token_url {
            return Ok(url.clone());
        }
        if let Some(preset) = self.preset {
            return Ok(preset.defaults().1.to_string());
        }
        bail!("OAuth provider has no token_url and no preset")
    }
}

// ---------------------------------------------------------------------------
// PKCE
// ---------------------------------------------------------------------------

pub struct PkceChallenge {
    pub code_verifier: String,
    pub code_challenge: String,
}

pub fn generate_pkce() -> PkceChallenge {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).expect("getrandom failed");
    let code_verifier = URL_SAFE_NO_PAD.encode(buf);

    let digest = ring::digest::digest(&ring::digest::SHA256, code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest.as_ref());

    PkceChallenge {
        code_verifier,
        code_challenge,
    }
}

// ---------------------------------------------------------------------------
// Auth URL
// ---------------------------------------------------------------------------

pub fn build_auth_url(
    provider: &OAuthProviderConfig,
    redirect_uri: &str,
    scopes: &[String],
    state: &str,
    pkce: &PkceChallenge,
) -> Result<String> {
    let mut url = Url::parse(&provider.auth_url()?)?;
    url.query_pairs_mut()
        .append_pair("client_id", &provider.client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", &scopes.join(" "))
        .append_pair("state", state)
        .append_pair("code_challenge", &pkce.code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");
    Ok(url.to_string())
}

// ---------------------------------------------------------------------------
// Token exchange
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    scope: Option<String>,
}

pub async fn exchange_code(
    provider: &OAuthProviderConfig,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<TokenSet> {
    let client = reqwest::Client::new();
    let resp = client
        .post(&provider.token_url()?)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", &provider.client_id),
            ("client_secret", &provider.client_secret),
            ("code_verifier", code_verifier),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Token exchange failed ({status}): {body}");
    }

    let tr: TokenResponse = resp.json().await?;
    let expires_at = tr
        .expires_in
        .map(|s| Utc::now() + chrono::Duration::seconds(s));
    let scopes = tr
        .scope
        .map(|s| s.split_whitespace().map(String::from).collect())
        .unwrap_or_default();

    Ok(TokenSet {
        access_token: tr.access_token,
        refresh_token: tr.refresh_token,
        expires_at,
        scopes,
    })
}

pub async fn refresh_token(provider: &OAuthProviderConfig, refresh: &str) -> Result<TokenSet> {
    let client = reqwest::Client::new();
    let resp = client
        .post(&provider.token_url()?)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
            ("client_id", &provider.client_id),
            ("client_secret", &provider.client_secret),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Token refresh failed ({status}): {body}");
    }

    let tr: TokenResponse = resp.json().await?;
    let expires_at = tr
        .expires_in
        .map(|s| Utc::now() + chrono::Duration::seconds(s));
    let scopes = tr
        .scope
        .map(|s| s.split_whitespace().map(String::from).collect())
        .unwrap_or_default();

    Ok(TokenSet {
        access_token: tr.access_token,
        refresh_token: tr.refresh_token.or_else(|| Some(refresh.to_string())),
        expires_at,
        scopes,
    })
}

// ---------------------------------------------------------------------------
// Local callback server (for standalone / CLI OAuth flows)
// ---------------------------------------------------------------------------

pub async fn run_local_callback(port: u16) -> Result<(String, HashMap<String, String>)> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let actual_port = listener.local_addr()?.port();
    let redirect_uri = format!("http://localhost:{actual_port}/oauth/callback");

    let (tx, rx) = oneshot::channel::<HashMap<String, String>>();
    let mut tx = Some(tx);

    tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            let _ = handle_callback_connection(stream, &mut tx).await;
        }
    });

    let params = rx.await?;
    Ok((redirect_uri, params))
}

async fn handle_callback_connection(
    mut stream: tokio::net::TcpStream,
    tx: &mut Option<oneshot::Sender<HashMap<String, String>>>,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = vec![0u8; 4096];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    let params = if let Some(query) = path.split('?').nth(1) {
        url::form_urlencoded::parse(query.as_bytes())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    } else {
        HashMap::new()
    };

    let body =
        "<html><body><h2>Authorization complete.</h2><p>You can close this tab.</p></body></html>";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await?;
    stream.shutdown().await?;

    if let Some(sender) = tx.take() {
        let _ = sender.send(params);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_generates_valid_lengths() {
        let p = generate_pkce();
        assert!(p.code_verifier.len() >= 43);
        assert!(p.code_challenge.len() >= 43);
        assert_ne!(p.code_verifier, p.code_challenge);
    }

    #[test]
    fn build_auth_url_includes_all_params() {
        let provider = OAuthProviderConfig {
            preset: Some(OAuthProviderPreset::Google),
            client_id: "test-client".into(),
            client_secret: "secret".into(),
            auth_url: None,
            token_url: None,
        };
        let pkce = generate_pkce();
        let url = build_auth_url(
            &provider,
            "http://localhost:8080/callback",
            &["openid".into(), "email".into()],
            "test-state",
            &pkce,
        )
        .unwrap();

        assert!(url.contains("accounts.google.com"));
        assert!(url.contains("client_id=test-client"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("state=test-state"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("openid+email") || url.contains("openid%20email"));
    }

    #[test]
    fn preset_defaults() {
        let (auth, token) = OAuthProviderPreset::Google.defaults();
        assert!(auth.contains("google.com"));
        assert!(token.contains("googleapis.com"));

        let (auth, token) = OAuthProviderPreset::Microsoft.defaults();
        assert!(auth.contains("microsoftonline.com"));
        assert!(token.contains("microsoftonline.com"));
    }

    #[test]
    fn custom_urls_override_preset() {
        let provider = OAuthProviderConfig {
            preset: Some(OAuthProviderPreset::Google),
            client_id: "id".into(),
            client_secret: "secret".into(),
            auth_url: Some("https://custom.auth/authorize".into()),
            token_url: Some("https://custom.auth/token".into()),
        };
        assert_eq!(
            provider.auth_url().unwrap(),
            "https://custom.auth/authorize".to_string()
        );
        assert_eq!(
            provider.token_url().unwrap(),
            "https://custom.auth/token".to_string()
        );
    }

    #[tokio::test]
    async fn local_callback_receives_params() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let callback = tokio::spawn(run_local_callback(port));

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let _ = client
            .get(format!(
                "http://localhost:{port}/oauth/callback?code=abc123&state=xyz"
            ))
            .send()
            .await;

        let (redirect_uri, params) = callback.await.unwrap().unwrap();
        assert!(redirect_uri.contains(&port.to_string()));
        assert_eq!(params.get("code").unwrap(), "abc123");
        assert_eq!(params.get("state").unwrap(), "xyz");
    }
}
