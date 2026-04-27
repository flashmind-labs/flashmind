//! MCP client — stdio and HTTP/SSE transport, initialize handshake, tool discovery, and tool calling.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::wire::{
    self, ClientCapabilities, ClientInfo, InitializeParams, InitializeResult, JsonRpcResponse,
    McpToolDef, RequestId, ServerInfo, ToolCallParams, ToolCallResult, ToolsListResult,
};

/// MCP client supporting stdio and HTTP/SSE transports.
pub struct McpClient {
    /// Stdio transport — `None` for HTTP/SSE.
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,

    /// HTTP/SSE transport.
    http_url: Option<String>,
    http_client: Option<reqwest::Client>,

    /// Monotonically increasing request ID counter.
    id_counter: AtomicU64,

    /// Server info received during the `initialize` handshake.
    pub server_info: Option<ServerInfo>,

    /// Tools discovered via `tools/list`.
    pub tools: Vec<McpToolDef>,
}

impl McpClient {
    /// Spawn a child process and connect via stdio (piped stdin/stdout, null stderr).
    ///
    /// If `command` is not an absolute path, attempts to resolve it through the
    /// user's login shell PATH (which may differ from the daemon's PATH).
    pub async fn spawn_stdio(command: &str, args: &[&str], env: &[(&str, &str)]) -> Result<Self> {
        let resolved = if !command.contains('/') {
            resolve_command(command).unwrap_or_else(|| command.to_owned())
        } else {
            command.to_owned()
        };

        let mut cmd = Command::new(&resolved);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        for (key, val) in env {
            cmd.env(key, val);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn MCP server process: {resolved}"))?;

        let stdin = child.stdin.take().context("child stdin unavailable")?;
        let stdout = child.stdout.take().context("child stdout unavailable")?;

        tracing::debug!(command, "spawned MCP stdio server");

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout: Some(BufReader::new(stdout)),
            http_url: None,
            http_client: None,
            id_counter: AtomicU64::new(1),
            server_info: None,
            tools: Vec::new(),
        })
    }

    /// Create an HTTP/SSE client pointed at `url`.
    pub async fn connect_sse(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let http_client = crate::utils::http_client();
        tracing::debug!(%url, "created MCP HTTP/SSE client");

        Ok(Self {
            child: None,
            stdin: None,
            stdout: None,
            http_url: Some(url),
            http_client: Some(http_client),
            id_counter: AtomicU64::new(1),
            server_info: None,
            tools: Vec::new(),
        })
    }

    /// Return the next request ID and advance the counter.
    pub fn next_id(&self) -> u64 {
        self.id_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a JSON-RPC request and return the response.
    pub async fn request(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> Result<JsonRpcResponse> {
        let id = self.next_id();
        let req = wire::JsonRpcRequest::new(RequestId::Num(id), method, params);

        if self.stdin.is_some() {
            self.request_stdio(req).await
        } else {
            self.request_http(req).await
        }
    }

    /// Send a request over the stdio transport and parse the response.
    ///
    /// Loops to skip server-initiated notifications (JSON-RPC messages without
    /// an `id` field) that MCP servers may send between request and response.
    async fn request_stdio(&mut self, req: wire::JsonRpcRequest) -> Result<JsonRpcResponse> {
        let line = serde_json::to_string(&req).context("failed to serialize JSON-RPC request")?;
        let request_id = req.id.clone();
        tracing::debug!(method = %req.method, "→ stdio request");

        self.send_raw(&line).await?;

        loop {
            let resp_line = self.read_line().await?;

            // Try to parse as a generic JSON value first to check for notifications.
            let value: Value =
                serde_json::from_str(&resp_line).context("failed to parse JSON from stdio")?;

            // Notifications have a `method` field but no `id` — skip them.
            if value.get("method").is_some() && value.get("id").is_none() {
                tracing::debug!(
                    method = value.get("method").and_then(|m| m.as_str()).unwrap_or("?"),
                    "← skipping server notification"
                );
                continue;
            }

            // Verify the response ID matches our request.
            if let Some(resp_id) = value.get("id") {
                let matches = match (&request_id, resp_id) {
                    (wire::RequestId::Num(n), Value::Number(rn)) => rn.as_u64() == Some(*n),
                    (wire::RequestId::Str(s), Value::String(rs)) => s == rs,
                    _ => false,
                };

                if !matches {
                    tracing::debug!("← skipping response with mismatched id");
                    continue;
                }
            }

            tracing::debug!("← stdio response");
            let resp: JsonRpcResponse = serde_json::from_value(value)
                .context("failed to parse JSON-RPC response from stdio")?;
            return Ok(resp);
        }
    }

    /// Send a request over HTTP/SSE and parse the response.
    async fn request_http(&mut self, req: wire::JsonRpcRequest) -> Result<JsonRpcResponse> {
        let url = self
            .http_url
            .as_deref()
            .context("no HTTP URL configured")?
            .to_owned();
        let client = self
            .http_client
            .as_ref()
            .context("no HTTP client configured")?;

        tracing::debug!(method = %req.method, %url, "→ HTTP request");

        let resp = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .with_context(|| format!("HTTP request to {url} failed"))?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .context("failed to read HTTP response body")?;

        if !status.is_success() {
            bail!("MCP HTTP server returned {status}: {body}");
        }

        tracing::debug!("← HTTP response");

        let parsed: JsonRpcResponse =
            serde_json::from_str(&body).context("failed to parse JSON-RPC response from HTTP")?;
        Ok(parsed)
    }

    /// Write a raw line (appending `\n`) to the child's stdin and flush.
    pub async fn send_raw(&mut self, line: &str) -> Result<()> {
        let stdin = self.stdin.as_mut().context("stdin not available")?;
        stdin
            .write_all(line.as_bytes())
            .await
            .context("write to stdin failed")?;
        stdin
            .write_all(b"\n")
            .await
            .context("write newline to stdin failed")?;
        stdin.flush().await.context("flush stdin failed")?;
        Ok(())
    }

    /// Read one line from the child's stdout (errors on EOF).
    pub async fn read_line(&mut self) -> Result<String> {
        let stdout = self.stdout.as_mut().context("stdout not available")?;
        let mut line = String::new();
        let n = stdout
            .read_line(&mut line)
            .await
            .context("read from stdout failed")?;

        if n == 0 {
            bail!("MCP server closed stdout (EOF)");
        }

        Ok(line
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_owned())
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    /// No-op for HTTP transport.
    pub async fn notify(&mut self, method: &str) -> Result<()> {
        if self.stdin.is_none() {
            return Ok(());
        }

        #[derive(serde::Serialize)]
        struct Notification<'a> {
            jsonrpc: &'a str,
            method: &'a str,
        }

        let notif = Notification {
            jsonrpc: "2.0",
            method,
        };
        let line = serde_json::to_string(&notif).context("failed to serialize notification")?;
        tracing::debug!(%method, "→ notification");
        self.send_raw(&line).await
    }

    /// Perform the MCP `initialize` handshake, then discover available tools.
    pub async fn initialize(&mut self) -> Result<()> {
        let params = InitializeParams {
            protocol_version: "2025-03-26".to_owned(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "flash".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        };

        let params_value =
            serde_json::to_value(params).context("failed to serialize InitializeParams")?;

        let resp = self
            .request("initialize", Some(params_value))
            .await
            .context("initialize request failed")?;

        if let Some(err) = resp.error {
            bail!("MCP initialize error {}: {}", err.code, err.message);
        }

        let result_value = resp.result.context("initialize response missing result")?;
        let result: InitializeResult = serde_json::from_value(result_value)
            .context("failed to deserialize InitializeResult")?;

        tracing::info!(
            server = %result.server_info.name,
            protocol = %result.protocol_version,
            "MCP server initialized"
        );

        self.server_info = Some(result.server_info);

        self.notify("notifications/initialized").await?;

        self.refresh_tools().await
    }

    /// Fetch and store the list of tools advertised by the MCP server.
    pub async fn refresh_tools(&mut self) -> Result<()> {
        let resp = self
            .request("tools/list", None)
            .await
            .context("tools/list request failed")?;

        if let Some(err) = resp.error {
            bail!("MCP tools/list error {}: {}", err.code, err.message);
        }

        let result_value = resp.result.context("tools/list response missing result")?;
        let result: ToolsListResult = serde_json::from_value(result_value)
            .context("failed to deserialize ToolsListResult")?;

        self.tools = result.tools;
        tracing::info!(count = self.tools.len(), "refreshed MCP tool list");

        Ok(())
    }

    /// Call a named tool with the given arguments and return the result.
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<ToolCallResult> {
        let params = ToolCallParams {
            name: name.to_owned(),
            arguments,
        };

        let params_value =
            serde_json::to_value(params).context("failed to serialize ToolCallParams")?;

        let resp = self
            .request("tools/call", Some(params_value))
            .await
            .with_context(|| format!("tools/call failed for tool '{name}'"))?;

        if let Some(err) = resp.error {
            bail!("MCP tools/call error {}: {}", err.code, err.message);
        }

        let result_value = resp
            .result
            .with_context(|| format!("tools/call response missing result for '{name}'"))?;

        let result: ToolCallResult =
            serde_json::from_value(result_value).context("failed to deserialize ToolCallResult")?;

        if result.is_error {
            tracing::warn!(tool = name, "MCP tool reported is_error=true");
        }

        Ok(result)
    }

    /// Shut down the client: close stdin and kill the child process if running.
    pub async fn shutdown(&mut self) {
        drop(self.stdin.take());

        if let Some(mut child) = self.child.take()
            && let Err(e) = child.kill().await
        {
            tracing::warn!(error = %e, "failed to kill MCP server process");
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Drop stdin so the child process receives EOF.
        drop(self.stdin.take());
    }
}

/// Resolve a bare command name to an absolute path using the user's login shell.
///
/// Daemons (launchd/systemd) inherit a minimal PATH that won't include
/// user-installed tools like `uvx`, `npx`, etc. This asks the user's
/// default shell for the full path.
fn resolve_command(command: &str) -> Option<String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned());

    // `sh -l -c "which <cmd>"` sources login profile → full user PATH.
    let output = std::process::Command::new(&shell)
        .args(["-l", "-c", &format!("which {command}")])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?.trim().to_owned();

    if path.is_empty() || !path.starts_with('/') {
        return None;
    }

    tracing::debug!(command, resolved = %path, "resolved MCP command via login shell");
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stdio_connect_echo_server() {
        // Use `cat` as a trivial echo "server"
        let mut client = McpClient::spawn_stdio("cat", &[], &[]).await.unwrap();
        let req = wire::JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: wire::RequestId::Num(1),
            method: "test".into(),
            params: None,
        };
        let line = serde_json::to_string(&req).unwrap();
        client.send_raw(&line).await.unwrap();
        let resp_line = client.read_line().await.unwrap();
        assert!(resp_line.contains("\"method\":\"test\""));
    }

    #[tokio::test]
    async fn test_next_id_increments() {
        let client = McpClient::spawn_stdio("cat", &[], &[]).await.unwrap();
        let id1 = client.next_id();
        let id2 = client.next_id();
        assert_eq!(id1 + 1, id2);
    }

    #[test]
    fn test_resolve_command_finds_common_binaries() {
        // `ls` exists on every unix system — resolve_command should find it.
        let resolved = resolve_command("ls");
        assert!(resolved.is_some(), "should resolve 'ls'");
        assert!(
            resolved.unwrap().starts_with('/'),
            "resolved path should be absolute"
        );
    }

    #[test]
    fn test_resolve_command_returns_none_for_nonexistent() {
        let resolved = resolve_command("definitely_not_a_real_binary_abc123");
        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_command_skips_absolute_paths() {
        // spawn_stdio only calls resolve_command for bare names (no '/').
        // Verify resolve_command still works if called with an absolute path.
        let resolved = resolve_command("/bin/ls");
        // `which /bin/ls` returns the path itself on most shells.
        assert!(resolved.is_some());
    }
}
