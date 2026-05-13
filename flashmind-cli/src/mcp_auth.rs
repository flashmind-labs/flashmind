//! Browser-based OAuth flow for MCP servers.

use std::time::Duration;

use anyhow::Result;
use flashmind_tools::mcp::McpRegistry;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::time::timeout;

/// Run the browser-based OAuth flow for an MCP server.
///
/// Calls `start_auth` to get the authorization URL, opens the browser,
/// listens on `localhost:19836` for the callback, and exchanges the code.
pub async fn browser_oauth(registry: &McpRegistry, server: &str) -> Result<()> {
    let redirect_uri = "http://localhost:19836/callback";
    let auth_url = registry
        .start_auth(server, redirect_uri)
        .await
        .map_err(|e| anyhow::anyhow!("OAuth setup failed: {e}"))?;

    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(&auth_url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open")
        .arg(&auth_url)
        .spawn();

    let listener = TcpListener::bind("127.0.0.1:19836")
        .await
        .map_err(|e| anyhow::anyhow!("failed to bind callback port: {e}"))?;

    let (code, state) = timeout(Duration::from_secs(300), accept_callback(&listener))
        .await
        .map_err(|_| anyhow::anyhow!("OAuth timed out — no callback received within 5 minutes"))?
        .map_err(|e| anyhow::anyhow!("OAuth callback error: {e}"))?;

    if code.is_empty() {
        anyhow::bail!("OAuth callback received but no authorization code found");
    }

    registry.complete_auth(server, &code, &state).await?;

    Ok(())
}

async fn accept_callback(listener: &TcpListener) -> Result<(String, String)> {
    let (tx, rx) = oneshot::channel::<(String, String)>();
    let tx = std::sync::Mutex::new(Some(tx));

    let (stream, _) = listener.accept().await?;
    let io = TokioIo::new(stream);

    http1::Builder::new()
        .serve_connection(
            io,
            service_fn(|req: Request<Incoming>| {
                let tx = tx.lock().unwrap().take();
                async move {
                    let (code, state) = extract_params(&req);

                    if let Some(tx) = tx {
                        let _ = tx.send((code, state));
                    }

                    let body = "<html><body><h2>Authentication successful</h2>\
                        <p>You can close this tab.</p></body></html>";
                    Ok::<_, std::convert::Infallible>(
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "text/html")
                            .header("Connection", "close")
                            .body(body.to_string())
                            .unwrap(),
                    )
                }
            }),
        )
        .await
        .map_err(|e| anyhow::anyhow!("HTTP error: {e}"))?;

    let (code, state) = rx
        .await
        .map_err(|_| anyhow::anyhow!("callback handler dropped without sending"))?;

    Ok((code, state))
}

fn extract_params(req: &Request<Incoming>) -> (String, String) {
    let mut code = String::new();
    let mut state = String::new();
    if let Some(query) = req.uri().query() {
        for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
            match k.as_ref() {
                "code" => code = v.into_owned(),
                "state" => state = v.into_owned(),
                _ => {}
            }
        }
    }
    (code, state)
}
