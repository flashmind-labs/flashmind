use std::io::Write;
use std::mem::take;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header, jwk::JwkSet};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use toml_edit::{Item, Table, Value};
use url::Url;
use uuid::Uuid;

use crate::ApiKey;
use crate::config::OpenAiChatAuth;
use crate::paths::config_dir;

const ISSUER: &str = "https://auth.openai.com";
const RESOURCE: &str = "https://api.openai.com/v1";
const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    id_token: String,
    expires_in: i64,
    scope: String,
    token_type: String,
}

#[derive(Deserialize)]
struct Identity {
    sub: String,
    nonce: String,
    email: Option<String>,
}

fn secret() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

fn secure_host_id() -> Result<String> {
    let path = config_dir()?.join("openai-host-id");
    if path.exists() {
        return Ok(std::fs::read_to_string(path)?.trim().to_owned());
    }
    let id = format!("urn:uuid:{}", Uuid::new_v4());
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            file.write_all(id.as_bytes())?;
            Ok(id)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            Ok(std::fs::read_to_string(path)?.trim().to_owned())
        }
        Err(error) => Err(error.into()),
    }
}

async fn validate_identity(
    client: &reqwest::Client,
    id_token: &str,
    client_id: &str,
    nonce: &str,
) -> Result<Identity> {
    let header = decode_header(id_token).context("invalid OpenAI ID token")?;
    ensure!(
        header.alg == Algorithm::RS256,
        "unsupported OpenAI ID token algorithm"
    );
    let kid = header.kid.context("OpenAI ID token has no key ID")?;
    let keys: JwkSet = client
        .get("https://auth.openai.com/.well-known/jwks.json")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let key = keys
        .find(&kid)
        .context("OpenAI ID token signing key not found")?;
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_issuer(&[ISSUER]);
    validation.set_audience(&[client_id]);
    let identity = decode::<Identity>(id_token, &DecodingKey::from_jwk(key)?, &validation)?.claims;
    ensure!(identity.nonce == nonce, "OpenAI ID token nonce mismatch");
    ensure!(!identity.sub.is_empty(), "OpenAI ID token has no subject");
    Ok(identity)
}

fn save_credentials(
    host_id: &str,
    client_id: &str,
    identity: &Identity,
    token: &TokenResponse,
) -> Result<()> {
    let path = config_dir()?.join("config.toml");
    save_credentials_at(&path, host_id, client_id, identity, token)
}

fn credential_path(config_path: &Path, auth: &OpenAiChatAuth) -> Result<PathBuf> {
    let directory = config_path.parent().context("invalid config path")?;
    Ok(match &auth.file {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => directory.join(path),
        None => directory.join("chatgpt.json"),
    })
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let parent = path.parent().context("invalid credential path")?;
    std::fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    temporary.write_all(data)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .context("could not save OpenAI credentials")?;
    Ok(())
}

fn save_credentials_at(
    path: &Path,
    host_id: &str,
    client_id: &str,
    identity: &Identity,
    token: &TokenResponse,
) -> Result<()> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error.into()),
    };
    let config: crate::config::Config =
        toml::from_str(&contents).context("invalid flsh configuration")?;
    let previous = config
        .providers
        .get("openai")
        .and_then(|entry| entry.chat_auth.clone())
        .unwrap_or_default();
    let credentials = OpenAiChatAuth {
        host_id: Some(host_id.to_owned()),
        client_id: Some(client_id.to_owned()),
        subject: Some(identity.sub.clone()),
        email: identity.email.clone(),
        access_token: Some(ApiKey(token.access_token.clone())),
        refresh_token: Some(ApiKey(token.refresh_token.clone())),
        id_token: Some(ApiKey(token.id_token.clone())),
        expires_at: Some(
            chrono::Utc::now()
                .timestamp()
                .checked_add(token.expires_in)
                .context("invalid token expiry")?,
        ),
        scope: Some(token.scope.clone()),
        ..Default::default()
    };
    let json_path = credential_path(path, &previous)?;
    atomic_write(&json_path, &serde_json::to_vec(&credentials)?)?;
    strip_inline_credentials(path, &contents)
}

fn strip_inline_credentials(path: &Path, contents: &str) -> Result<()> {
    let mut doc: toml_edit::DocumentMut = contents.parse().context("invalid flsh configuration")?;
    let providers = ensure_table(doc.as_table_mut(), "providers")?;
    let openai = ensure_table(providers, "openai")?;
    let auth = ensure_table(openai, "chat_auth")?;
    let configured = OpenAiChatAuth {
        file: auth.get("file").and_then(Item::as_str).map(PathBuf::from),
        ..Default::default()
    };
    let file = credential_path(path, &configured)?;
    auth["file"] = toml_edit::value(
        file.to_str()
            .context("credential path is not valid UTF-8")?,
    );
    for key in [
        "host_id",
        "client_id",
        "subject",
        "email",
        "access_token",
        "refresh_token",
        "id_token",
        "expires_at",
        "scope",
    ] {
        auth.remove(key);
    }
    atomic_write(path, doc.to_string().as_bytes())
}

fn load_credentials_at(
    config_path: &Path,
    auth: &OpenAiChatAuth,
) -> Result<Option<OpenAiChatAuth>> {
    let json_path = credential_path(config_path, auth)?;
    let credentials = match std::fs::read(&json_path) {
        Ok(data) => {
            Some(serde_json::from_slice(&data).context("invalid ChatGPT credentials file")?)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if auth.access_token.is_some() {
                let mut migrated = auth.clone();
                migrated.file = None;
                atomic_write(&json_path, &serde_json::to_vec(&migrated)?)?;
                Some(migrated)
            } else {
                None
            }
        }
        Err(error) => return Err(error.into()),
    };
    if auth.access_token.is_some()
        || (credentials.is_some() && auth.file.as_ref().is_none_or(|path| !path.is_absolute()))
    {
        let contents = std::fs::read_to_string(config_path)?;
        strip_inline_credentials(config_path, &contents)?;
    }
    Ok(credentials)
}

fn ensure_table<'a>(parent: &'a mut Table, name: &str) -> Result<&'a mut Table> {
    if !parent.contains_key(name) {
        parent.insert(name, Item::Table(Table::new()));
    }
    let (mut key, item) = parent
        .get_key_value_mut(name)
        .context("missing configuration table")?;
    if let Item::Value(Value::InlineTable(inline)) = item {
        let prefix = key.leaf_decor().prefix().cloned();
        let suffix = inline.decor().suffix().cloned();
        let mut table = take(inline).into_table();
        key.leaf_decor_mut().clear();
        if let Some(prefix) = prefix {
            table.decor_mut().set_prefix(prefix);
        }
        if let Some(suffix) = suffix {
            table.decor_mut().set_suffix(suffix);
        }
        *item = Item::Table(table);
    }
    item.as_table_mut()
        .with_context(|| format!("{name} must be a table"))
}

pub async fn login() -> Result<()> {
    let existing = tokio::task::spawn_blocking(|| {
        let _lock = lock_credentials()?;
        let config = crate::config::load_config()?;
        let auth = config
            .providers
            .get("openai")
            .and_then(|provider| provider.chat_auth.clone())
            .unwrap_or_default();
        load_credentials_at(&config_dir()?.join("config.toml"), &auth)
    })
    .await??;
    let host_id = tokio::task::spawn_blocking(secure_host_id).await??;
    let state = secret();
    let nonce = secret();
    let verifier = secret();
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
    let redirect = format!(
        "http://{}:{}/auth/callback",
        Ipv4Addr::LOCALHOST,
        listener.local_addr()?.port()
    );
    let mut url = Url::parse("https://auth.openai.com/api/accounts/authorize")?;
    let saved_client = existing.as_ref().and_then(|auth| auth.client_id.as_deref());
    url.query_pairs_mut().extend_pairs([
        ("client_id", saved_client.unwrap_or("dynamic_agent_client")),
        ("ext_agent_host_id", &host_id),
        ("response_type", "code"),
        ("redirect_uri", &redirect),
        (
            "scope",
            "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
        ),
        ("resource", RESOURCE),
        ("state", &state),
        ("nonce", &nonce),
        ("code_challenge_method", "S256"),
        ("code_challenge", &challenge),
    ]);
    if saved_client.is_none() {
        url.query_pairs_mut().append_pair("agent_name_hint", "flsh");
    }
    println!("Continue with ChatGPT by opening this URL in a browser on this machine:\n{url}");
    let (mut socket, _) = tokio::time::timeout(Duration::from_secs(300), listener.accept())
        .await
        .context("OpenAI sign-in timed out")??;
    let request = tokio::time::timeout(Duration::from_secs(10), async {
        let mut buffer = Vec::new();
        while !buffer.ends_with(b"\r\n\r\n") {
            ensure!(buffer.len() < 16384, "OpenAI callback is too large");
            buffer.push(
                socket
                    .read_u8()
                    .await
                    .context("incomplete OpenAI callback")?,
            );
        }
        String::from_utf8(buffer).context("invalid callback")
    })
    .await
    .context("OpenAI callback timed out")??;
    let path = request
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("GET "))
        .and_then(|line| line.split_whitespace().next())
        .context("invalid callback request")?;
    let callback = Url::parse(&format!("http://{}{path}", Ipv4Addr::LOCALHOST))?;
    ensure!(callback.path() == "/auth/callback", "invalid callback path");
    let mut params = std::collections::HashMap::new();
    for (key, value) in callback.query_pairs() {
        ensure!(
            params
                .insert(key.into_owned(), value.into_owned())
                .is_none(),
            "duplicate callback parameter"
        );
    }
    ensure!(
        params.get("state").map(String::as_str) == Some(state.as_str()),
        "OpenAI sign-in state mismatch"
    );
    if let Some(error) = params.get("error") {
        bail!("OpenAI sign-in declined: {error}");
    }
    let code = params
        .get("code")
        .context("OpenAI callback has no authorization code")?;
    let returned_client = params.get("client_id").map(String::as_str);
    if let (Some(saved), Some(returned)) = (saved_client, returned_client) {
        ensure!(saved == returned, "OpenAI returned a different client ID");
    }
    let client_id = returned_client
        .or(saved_client)
        .context("OpenAI callback has no client ID")?;
    ensure!(
        client_id != "dynamic_agent_client",
        "OpenAI did not issue a client ID"
    );
    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nSign-in received. Return to your terminal.\n").await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let token: TokenResponse = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", client_id),
            ("code", code),
            ("code_verifier", &verifier),
            ("redirect_uri", &redirect),
            ("resource", RESOURCE),
        ])
        .send()
        .await?
        .error_for_status()
        .context("OpenAI token exchange failed")?
        .json()
        .await?;
    ensure!(
        token.token_type.eq_ignore_ascii_case("bearer"),
        "unexpected OpenAI token type"
    );
    ensure!(
        token
            .scope
            .split_whitespace()
            .any(|scope| scope == "chatgpt.tokens.use.direct"),
        "ChatGPT plan access was not granted"
    );
    ensure!(token.expires_in > 0, "invalid OpenAI token expiry");
    let identity = validate_identity(&client, &token.id_token, client_id, &nonce).await?;
    if let Some(subject) = existing.as_ref().and_then(|auth| auth.subject.as_deref()) {
        ensure!(
            subject == identity.sub,
            "OpenAI returned a different account"
        );
    }
    let client_id = client_id.to_owned();
    tokio::task::spawn_blocking(move || {
        let _lock = lock_credentials()?;
        save_credentials(&host_id, &client_id, &identity, &token)
    })
    .await??;
    println!("ChatGPT plan connected for OpenAI chat.");
    Ok(())
}

/// Supplies plan credentials to the Responses provider.
pub struct PlanTokens;

#[derive(Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: i64,
    scope: Option<String>,
    token_type: String,
}

fn lock_credentials() -> Result<std::fs::File> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(config_dir()?.join("openai-auth.lock"))?;
    file.lock()?;
    Ok(file)
}

async fn current_access_token() -> Result<secrecy::SecretString> {
    let _lock = tokio::task::spawn_blocking(lock_credentials).await??;
    let auth = tokio::task::spawn_blocking(|| {
        let config = crate::config::load_config()?;
        let auth = config
            .providers
            .get("openai")
            .and_then(|provider| provider.chat_auth.as_ref())
            .context("Run flsh login openai first")?;
        load_credentials_at(&config_dir()?.join("config.toml"), auth)?
            .context("Run flsh login openai first")
    })
    .await??;
    let access = auth
        .access_token
        .context("OpenAI access token missing; run flsh login openai")?;
    if !needs_refresh(auth.expires_at, chrono::Utc::now().timestamp()) {
        return Ok(access.0.into());
    }
    let client_id = auth
        .client_id
        .context("OpenAI client ID missing; run flsh login openai")?;
    let refresh = auth
        .refresh_token
        .context("OpenAI refresh token missing; run flsh login openai")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let updated = request_refresh(&client, TOKEN_URL, &client_id, &refresh.0).await?;
    ensure!(
        updated.token_type.eq_ignore_ascii_case("bearer") && updated.expires_in > 0,
        "invalid OpenAI refresh response"
    );
    let scope = updated
        .scope
        .or(auth.scope)
        .context("OpenAI token scope missing")?;
    ensure!(
        scope
            .split_whitespace()
            .any(|scope| scope == "chatgpt.tokens.use.direct"),
        "ChatGPT plan access was not granted"
    );
    let identity = Identity {
        sub: auth.subject.context("OpenAI subject missing")?,
        nonce: String::new(),
        email: auth.email,
    };
    let host_id = auth.host_id.context("OpenAI host ID missing")?;
    let token = TokenResponse {
        access_token: updated.access_token,
        refresh_token: updated.refresh_token.unwrap_or(refresh.0),
        id_token: auth
            .id_token
            .map(|key| key.0)
            .context("OpenAI ID token missing")?,
        expires_in: updated.expires_in,
        scope,
        token_type: updated.token_type,
    };
    let secret = secrecy::SecretString::from(token.access_token.clone());
    tokio::task::spawn_blocking(move || save_credentials(&host_id, &client_id, &identity, &token))
        .await??;
    Ok(secret)
}

fn needs_refresh(expires_at: Option<i64>, now: i64) -> bool {
    expires_at.is_none_or(|expiry| expiry <= now.saturating_add(60))
}

async fn request_refresh(
    client: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    refresh_token: &str,
) -> Result<RefreshResponse> {
    Ok(client
        .post(token_url)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", client_id),
            ("refresh_token", refresh_token),
            ("resource", RESOURCE),
        ])
        .send()
        .await?
        .error_for_status()
        .context("OpenAI token refresh failed; run flsh login openai")?
        .json()
        .await?)
}

#[async_trait::async_trait]
impl flashmind_llm::responses::AccessTokenSource for PlanTokens {
    async fn access_token(&self) -> Result<secrecy::SecretString> {
        tokio::spawn(current_access_token())
            .await
            .context("OpenAI token refresh task failed")?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refreshes_before_expiry_and_when_expiry_is_missing() {
        assert!(!needs_refresh(Some(1061), 1000));
        assert!(needs_refresh(Some(1060), 1000));
        assert!(needs_refresh(Some(999), 1000));
        assert!(needs_refresh(None, 1000));
    }

    #[tokio::test]
    async fn refresh_request_uses_refresh_grant_and_receives_rotated_token() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = format!("http://{}/token", listener.local_addr().unwrap());
        let server = async {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap();
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(str::to_owned)
                })
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0; length];
            socket.read_exact(&mut body).await.unwrap();
            let params: std::collections::HashMap<_, _> =
                url::form_urlencoded::parse(&body).into_owned().collect();
            assert_eq!(params["grant_type"], "refresh_token");
            assert_eq!(params["client_id"], "example-client");
            assert_eq!(params["refresh_token"], "example-refresh");
            assert_eq!(params["resource"], RESOURCE);
            let response = r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600,"token_type":"Bearer"}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
        };
        let client = reqwest::Client::new();
        let (result, ()) = tokio::join!(
            request_refresh(&client, &url, "example-client", "example-refresh"),
            server
        );
        let result = result.unwrap();
        assert_eq!(result.access_token, "new-access");
        assert_eq!(result.refresh_token.as_deref(), Some("new-refresh"));
    }

    #[test]
    fn credentials_preserve_api_key_and_are_private() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(&path, "[providers.openai]\napi_key = 'example-key'\n").unwrap();
        let identity = Identity {
            sub: "example-subject".into(),
            nonce: "example-nonce".into(),
            email: None,
        };
        let token = TokenResponse {
            access_token: "example-access".into(),
            refresh_token: "example-refresh".into(),
            id_token: "example-id".into(),
            expires_in: 3600,
            scope: "chatgpt.tokens.use.direct".into(),
            token_type: "Bearer".into(),
        };
        save_credentials_at(
            &path,
            "urn:uuid:example",
            "example-client",
            &identity,
            &token,
        )
        .unwrap();
        let config: crate::config::Config =
            toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            config.api_key_for(&flashmind_types::Provider::OpenAi),
            Some("example-key")
        );
        let auth = config.providers["openai"].chat_auth.as_ref().unwrap();
        assert!(auth.access_token.is_none());
        assert_eq!(
            auth.file.as_deref(),
            Some(directory.path().join("chatgpt.json").as_path())
        );
        let saved = load_credentials_at(&path, auth).unwrap().unwrap();
        assert_eq!(saved.access_token.as_ref().unwrap().0, "example-access");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(directory.path().join("chatgpt.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn credentials_use_tables_for_new_and_inline_configurations() {
        let identity = Identity {
            sub: "example-subject".into(),
            nonce: "example-nonce".into(),
            email: None,
        };
        let token = TokenResponse {
            access_token: "example-access".into(),
            refresh_token: "example-refresh".into(),
            id_token: "example-id".into(),
            expires_in: 3600,
            scope: "chatgpt.tokens.use.direct".into(),
            token_type: "Bearer".into(),
        };
        for input in [
            "",
            "[providers.openai]\napi_key = 'example-key'\n",
            "[providers]\nopenai = { api_key = 'example-key', chat_auth = { access_token = 'old' } }\n",
            "providers = { openai = { api_key = 'example-key' }, other = { api_key = 'other-key' } }\n",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("config.toml");
            std::fs::write(&path, input).unwrap();
            save_credentials_at(
                &path,
                "urn:uuid:example",
                "example-client",
                &identity,
                &token,
            )
            .unwrap();
            let output = std::fs::read_to_string(&path).unwrap();
            assert!(output.contains("[providers.openai.chat_auth]"), "{output}");
            let config: crate::config::Config = toml::from_str(&output).unwrap();
            if !input.is_empty() {
                assert_eq!(
                    config.api_key_for(&flashmind_types::Provider::OpenAi),
                    Some("example-key")
                );
            }
            assert_eq!(
                load_credentials_at(
                    &path,
                    config.providers["openai"].chat_auth.as_ref().unwrap()
                )
                .unwrap()
                .unwrap()
                .access_token
                .as_ref()
                .unwrap()
                .0,
                "example-access"
            );
            if input.contains("other-key") {
                assert!(config.providers.contains_key("other"));
            }
        }
    }

    #[test]
    fn migrates_inline_tokens_to_configured_json_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(&path, "[providers.openai]\napi_key = 'example-key'\n[providers.openai.chat_auth]\nfile = 'credentials.json'\naccess_token = 'example-access'\nrefresh_token = 'example-refresh'\nexpires_at = 123\n").unwrap();
        let config: crate::config::Config =
            toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let auth = config.providers["openai"].chat_auth.as_ref().unwrap();
        let migrated = load_credentials_at(&path, auth).unwrap().unwrap();
        assert_eq!(migrated.refresh_token.unwrap().0, "example-refresh");
        let output = std::fs::read_to_string(&path).unwrap();
        assert!(!output.contains("example-access"));
        assert!(!output.contains("example-refresh"));
        assert!(output.contains("credentials.json"));
        let saved: OpenAiChatAuth = serde_json::from_slice(
            &std::fs::read(directory.path().join("credentials.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(saved.expires_at, Some(123));
        let config: crate::config::Config = toml::from_str(&output).unwrap();
        assert_eq!(
            config.api_key_for(&flashmind_types::Provider::OpenAi),
            Some("example-key")
        );
    }

    #[test]
    fn existing_empty_table_gains_the_default_file_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(&path, "[providers.openai.chat_auth]\n").unwrap();
        std::fs::write(
            directory.path().join("chatgpt.json"),
            r#"{"access_token":"example-access"}"#,
        )
        .unwrap();
        let config: crate::config::Config =
            toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let auth = config.providers["openai"].chat_auth.as_ref().unwrap();
        assert!(load_credentials_at(&path, auth).unwrap().is_some());
        let output = std::fs::read_to_string(&path).unwrap();
        let updated: crate::config::Config = toml::from_str(&output).unwrap();
        assert_eq!(
            updated.providers["openai"]
                .chat_auth
                .as_ref()
                .unwrap()
                .file
                .as_deref(),
            Some(directory.path().join("chatgpt.json").as_path())
        );
    }
}
