//! `flsh mcp login`: OAuth sign-in to a remote MCP server.
//!
//! The browser returns to a loopback listener. When the browser runs on
//! another machine, the address it ends up on can be pasted instead.

use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use flashmind_tools::mcp::{McpRegistry, McpServerConfig};
use std::io::BufRead;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use url::Url;

use crate::config::load_config;

const CALLBACK_PATH: &str = "/callback";
const SIGN_IN_TIMEOUT: Duration = Duration::from_secs(600);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_REQUEST: usize = 16 * 1024;

pub(super) async fn run(name: &str) -> Result<()> {
    let config = load_config()?;
    let server = config
        .mcp_servers
        .iter()
        .find(|server| server.name == name)
        .with_context(|| format!("no MCP server named '{name}'"))?;
    let url = server
        .url
        .clone()
        .with_context(|| format!("MCP server '{name}' runs locally and needs no sign-in"))?;

    let credentials = super::credentials()?;
    // Keep the current sign-in until the new one replaces it.
    let saved = credentials.load_credentials(name).await?;
    private_dir(&super::data_dir()?.join("mcp"))?;
    let registry = McpRegistry::new(credentials, None);
    registry
        .add(McpServerConfig {
            name: name.into(),
            url: Some(url),
            client_id: server.client_id.clone(),
            scopes: server.scopes.clone(),
            credentials: saved,
            ..Default::default()
        })
        .await?;

    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
    let redirect = format!(
        "http://{}:{}{CALLBACK_PATH}",
        Ipv4Addr::LOCALHOST,
        listener.local_addr()?.port()
    );
    let auth_url = registry.start_auth(name, &redirect).await?;
    println!("Sign in to '{name}' by opening this URL in a browser:\n\n{auth_url}\n");
    println!(
        "If the browser is on another machine, it will fail to load the last page. \
         Paste that page's address here and press Enter."
    );

    let params = tokio::time::timeout(SIGN_IN_TIMEOUT, async {
        tokio::select! {
            params = from_browser(&listener) => params,
            params = from_terminal() => params,
        }
    })
    .await
    .context("sign-in timed out")??;
    let callback = Callback::from_params(params)?;
    registry
        .complete_auth(
            name,
            &callback.code,
            &callback.state,
            callback.issuer.as_deref(),
        )
        .await?;
    let tools = registry.current_mcp_tools().get(name).map_or(0, Vec::len);
    registry.shutdown_all().await;
    println!("Signed in to '{name}': {tools} tools.");
    Ok(())
}

/// The saved sign-in holds tokens: only the user may read the directory.
fn private_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Serve the loopback listener until the browser arrives at the callback path.
async fn from_browser(listener: &TcpListener) -> Result<HashMap<String, String>> {
    loop {
        let (mut socket, _) = listener.accept().await?;
        let request = match tokio::time::timeout(REQUEST_TIMEOUT, read_head(&mut socket)).await {
            Ok(Ok(request)) => request,
            _ => continue,
        };
        let target = request
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("GET "))
            .and_then(|line| line.split_whitespace().next());
        let params = target.and_then(|target| {
            callback_params(&format!("http://{}{target}", Ipv4Addr::LOCALHOST)).ok()
        });
        let Some(params) = params else {
            // Browsers also ask for a favicon; only the callback ends the wait.
            let _ = socket
                .write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        };
        let _ = socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n\
                  Sign-in received. Return to your terminal.\n",
            )
            .await;
        return Ok(params);
    }
}

async fn read_head(socket: &mut TcpStream) -> Result<String> {
    let mut buffer = Vec::new();
    while !buffer.ends_with(b"\r\n\r\n") {
        ensure!(buffer.len() < MAX_REQUEST, "callback request is too large");
        buffer.push(socket.read_u8().await?);
    }
    String::from_utf8(buffer).context("invalid callback request")
}

/// Read a pasted callback address. Without a terminal this never finishes,
/// leaving the browser as the only way back.
///
/// The read runs on its own thread: a pending Tokio stdin read would keep the
/// runtime from shutting down after the browser finished the sign-in.
async fn from_terminal() -> Result<HashMap<String, String>> {
    let (sender, mut lines) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    while let Some(line) = lines.recv().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match callback_params(line) {
            Ok(params) => return Ok(params),
            Err(error) => eprintln!("{error:#}. Paste the full address, starting with http."),
        }
    }
    std::future::pending().await
}

/// The query of a callback address, rejecting other paths and repeated keys.
fn callback_params(address: &str) -> Result<HashMap<String, String>> {
    let url = Url::parse(address).context("not a URL")?;
    ensure!(
        url.path() == CALLBACK_PATH,
        "not the sign-in callback address"
    );
    let mut params = HashMap::new();
    for (key, value) in url.query_pairs() {
        let key = key.into_owned();
        ensure!(
            !params.contains_key(&key),
            "repeated callback parameter '{key}'"
        );
        params.insert(key, value.into_owned());
    }
    Ok(params)
}

#[derive(Debug)]
struct Callback {
    code: String,
    state: String,
    /// RFC 9207 issuer. Servers that advertise it, such as Fastmail, require it back.
    issuer: Option<String>,
}

impl Callback {
    fn from_params(mut params: HashMap<String, String>) -> Result<Self> {
        if let Some(error) = params.remove("error") {
            match params.remove("error_description") {
                Some(description) => bail!("sign-in declined: {error}: {description}"),
                None => bail!("sign-in declined: {error}"),
            }
        }
        Ok(Self {
            code: params
                .remove("code")
                .context("callback has no authorization code")?,
            state: params.remove("state").context("callback has no state")?,
            issuer: params.remove("iss"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_keeps_code_state_and_issuer() {
        let params = callback_params(
            "http://127.0.0.1:4321/callback?code=abc&state=xyz&iss=https%3A%2F%2Fauth.example.com",
        )
        .unwrap();
        let callback = Callback::from_params(params).unwrap();
        assert_eq!(callback.code, "abc");
        assert_eq!(callback.state, "xyz");
        assert_eq!(callback.issuer.as_deref(), Some("https://auth.example.com"));
    }

    #[test]
    fn callback_without_issuer_is_accepted() {
        let params = callback_params("http://127.0.0.1:1/callback?code=a&state=b").unwrap();
        assert_eq!(Callback::from_params(params).unwrap().issuer, None);
    }

    #[test]
    fn other_addresses_are_rejected() {
        assert!(callback_params("http://127.0.0.1:1/favicon.ico").is_err());
        assert!(callback_params("not a url").is_err());
        assert!(callback_params("http://127.0.0.1:1/callback?code=a&code=b").is_err());
    }

    #[test]
    fn declined_sign_in_reports_the_reason() {
        let params = callback_params(
            "http://127.0.0.1:1/callback?error=access_denied&error_description=User+said+no&state=b",
        )
        .unwrap();
        let error = Callback::from_params(params).unwrap_err().to_string();
        assert_eq!(error, "sign-in declined: access_denied: User said no");
    }

    #[test]
    fn missing_code_is_an_error() {
        let params = callback_params("http://127.0.0.1:1/callback?state=b").unwrap();
        assert!(Callback::from_params(params).is_err());
    }

    #[tokio::test]
    async fn browser_callback_ends_the_wait_after_other_requests() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let browser = async move {
            for path in ["/favicon.ico", "/callback?code=a&state=b"] {
                let mut socket = TcpStream::connect((Ipv4Addr::LOCALHOST, port))
                    .await
                    .unwrap();
                socket
                    .write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
                    .await
                    .unwrap();
                let mut response = String::new();
                socket.read_to_string(&mut response).await.unwrap();
                assert!(response.starts_with("HTTP/1.1"), "{response}");
            }
        };
        let (params, ()) = tokio::join!(from_browser(&listener), browser);
        let params = params.unwrap();
        assert_eq!(params["code"], "a");
        assert_eq!(params["state"], "b");
    }
}
