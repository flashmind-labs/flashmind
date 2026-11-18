//! Stdio MCP connections owned by one agent session.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use flashmind_types::tool::{Tool, ToolContext, ToolRegistry, ToolResult};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, Tool as RemoteTool};
use rmcp::service::{Peer, RoleClient};
use rmcp::transport::TokioChildProcess;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;

use super::config::McpConfigProvider;
use super::transport::{McpService, client_info, connect_http};
use super::types::call_result_from_rmcp;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const CALL_TIMEOUT: Duration = Duration::from_secs(300);

/// An MCP server for one session: a local process started with `command`, or
/// a remote server at `url`. Arguments and environment values are passed
/// literally.
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionServerConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Streamable HTTP endpoint of a remote server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// OAuth client ID, for servers without dynamic client registration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// OAuth scopes to request when signing in.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}

impl SessionServerConfig {
    /// The server's transport, or an error unless exactly one of `command`
    /// and `url` is set and the other transport's settings are absent.
    pub fn transport(&self) -> Result<SessionTransport<'_>> {
        match (&self.command, &self.url) {
            (Some(command), None) => {
                ensure!(
                    self.client_id.is_none() && self.scopes.is_empty(),
                    "MCP server '{}' runs a command, so it takes no OAuth settings",
                    self.name
                );
                Ok(SessionTransport::Stdio { command })
            }
            (None, Some(url)) => {
                ensure!(
                    self.args.is_empty() && self.env.is_empty(),
                    "MCP server '{}' has a URL, so it takes no args or env",
                    self.name
                );
                Ok(SessionTransport::Http { url })
            }
            _ => anyhow::bail!("MCP server '{}' needs either a command or a URL", self.name),
        }
    }
}

/// How a session reaches a server.
pub enum SessionTransport<'a> {
    Stdio { command: &'a Path },
    Http { url: &'a str },
}

/// Keeps connections alive until session shutdown. Dropping it cancels all services.
#[derive(Default)]
pub struct SessionMcp {
    services: Vec<McpService>,
}

impl SessionMcp {
    /// Initialize and discover every server before publishing any of its tools.
    /// Failure drops all connections opened by this call.
    ///
    /// Remote servers sign in with the OAuth credentials kept by `credentials`;
    /// without stored credentials, or when they no longer work, the error is
    /// [`McpAuthRequired`](super::McpAuthRequired).
    pub async fn connect(
        servers: &[SessionServerConfig],
        cwd: &Path,
        credentials: Option<&Arc<dyn McpConfigProvider>>,
        tools: &mut ToolRegistry,
    ) -> Result<Self> {
        let mut names = HashSet::new();
        for server in servers {
            ensure!(valid_name(&server.name), "invalid MCP server name");
            server.transport()?;
            ensure!(
                names.insert(&server.name),
                "duplicate MCP server name: {}",
                server.name
            );
        }
        let mut session = Self::default();
        let mut wrappers: Vec<Arc<dyn Tool>> = Vec::new();
        let mut tool_names = HashSet::new();
        let connected: Result<()> = async {
            for server in servers {
                let service = match server.transport()? {
                    SessionTransport::Stdio { command } => {
                        connect_stdio(server, command, cwd).await?
                    }
                    SessionTransport::Http { url } => {
                        let credentials = credentials.with_context(|| {
                            format!("MCP server '{}' has a URL, but this session cannot sign in to remote servers", server.name)
                        })?;
                        connect_http(credentials, &server.name, url, None, None, false).await?
                    }
                };
                session.services.push(service);
                let service = session.services.last().context("missing MCP connection")?;
                let discovered = timeout(CONNECT_TIMEOUT, service.list_all_tools())
                    .await
                    .context("MCP tool discovery timed out")?
                    .with_context(|| {
                        format!("could not discover tools for MCP server '{}'", server.name)
                    })?;
                for tool in discovered {
                    ensure!(valid_name(&tool.name), "invalid MCP tool name");
                    let name = format!("mcp__{}__{}", server.name, tool.name);
                    ensure!(name.len() <= 64, "MCP tool name exceeds 64 bytes: {name}");
                    ensure!(
                        tools.get(&name).is_none() && tool_names.insert(name.clone()),
                        "duplicate MCP tool name: {name}"
                    );
                    wrappers.push(Arc::new(SessionTool {
                        name,
                        tool,
                        peer: service.peer().clone(),
                    }));
                }
            }
            Ok(())
        }
        .await;
        if let Err(error) = connected {
            session.shutdown().await;
            return Err(error);
        }
        for wrapper in wrappers {
            tools.register(wrapper);
        }
        Ok(session)
    }

    /// Close transports and reap their child processes.
    pub async fn shutdown(&mut self) {
        for service in self.services.drain(..) {
            if let Err(error) = service.cancel().await {
                tracing::warn!(%error, "MCP shutdown failed");
            }
        }
    }
}

async fn connect_stdio(
    server: &SessionServerConfig,
    command: &Path,
    cwd: &Path,
) -> Result<McpService> {
    let mut command = Command::new(command);
    command
        .args(&server.args)
        .envs(&server.env)
        .current_dir(cwd);
    let (transport, _) = TokioChildProcess::builder(command)
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("could not spawn MCP server '{}'", server.name))?;
    timeout(CONNECT_TIMEOUT, client_info().serve(transport))
        .await
        .context("MCP initialization timed out")?
        .with_context(|| format!("could not initialize MCP server '{}'", server.name))
}

/// Whether `name` can be an MCP server or tool name: ASCII letters, digits,
/// underscores and hyphens.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

struct SessionTool {
    name: String,
    tool: RemoteTool,
    peer: Peer<RoleClient>,
}

#[async_trait]
impl Tool for SessionTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        self.tool.description.as_deref().unwrap_or_default()
    }
    fn parameters(&self) -> Value {
        Value::Object(self.tool.input_schema.as_ref().clone())
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let arguments = match &ctx.args {
            Value::Null => Default::default(),
            Value::Object(arguments) => arguments.clone(),
            _ => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "MCP arguments must be an object",
                ));
            }
        };
        let request =
            CallToolRequestParams::new(self.tool.name.to_string()).with_arguments(arguments);
        // A failed call may already have performed a side effect. Never retry it.
        let result = tokio::select! {
            result = timeout(CALL_TIMEOUT, self.peer.call_tool(request)) => result,
            _ = ctx.cancel_token().cancelled() => {
                return Ok(ToolResult::failure(ctx.tool_call_id, "MCP call cancelled; completion is unknown"));
            }
        };
        Ok(match result {
            Ok(Ok(result)) => call_result_from_rmcp(result).into_tool_result(ctx.tool_call_id),
            Ok(Err(error)) => {
                ToolResult::failure(ctx.tool_call_id, format!("MCP call failed: {error}"))
            }
            Err(_) => ToolResult::failure(
                ctx.tool_call_id,
                "MCP call timed out; completion is unknown",
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::ToolCall;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::mcp::{McpAuthRequired, McpDiskConfig};

    fn remote() -> SessionServerConfig {
        SessionServerConfig {
            name: "remote".into(),
            // Nothing listens on the discard port, so connecting fails at once.
            url: Some("http://127.0.0.1:9/mcp".into()),
            ..Default::default()
        }
    }

    #[test]
    fn transport_needs_exactly_one_of_command_and_url() {
        let stdio = SessionServerConfig {
            name: "local".into(),
            command: Some("server".into()),
            ..Default::default()
        };
        assert!(matches!(
            stdio.transport().unwrap(),
            SessionTransport::Stdio { .. }
        ));
        assert!(matches!(
            remote().transport().unwrap(),
            SessionTransport::Http { .. }
        ));

        let neither = SessionServerConfig {
            name: "x".into(),
            ..Default::default()
        };
        let both = SessionServerConfig {
            command: Some("server".into()),
            ..remote()
        };
        let remote_with_env = SessionServerConfig {
            env: BTreeMap::from([("A".into(), "1".into())]),
            ..remote()
        };
        let local_with_scopes = SessionServerConfig {
            scopes: vec!["mail".into()],
            ..stdio
        };
        for server in [neither, both, remote_with_env, local_with_scopes] {
            assert!(server.transport().is_err());
        }
    }

    #[tokio::test]
    async fn remote_server_without_sign_in_requires_auth() {
        let dir = tempfile::tempdir().unwrap();
        let credentials: Arc<dyn McpConfigProvider> = McpDiskConfig::new(dir.path().into());
        let mut tools = ToolRegistry::default();
        let error = SessionMcp::connect(&[remote()], dir.path(), Some(&credentials), &mut tools)
            .await
            .err()
            .unwrap();
        let required = error.downcast_ref::<McpAuthRequired>().unwrap();
        assert_eq!(required.server, "remote");

        let error = SessionMcp::connect(&[remote()], dir.path(), None, &mut tools)
            .await
            .err()
            .unwrap();
        assert!(error.downcast_ref::<McpAuthRequired>().is_none());
        assert!(tools.list().is_empty());
    }

    #[tokio::test]
    async fn discovers_pages_calls_tools_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("server.py");
        let pid_file = dir.path().join("pid");
        tokio::fs::write(&source, include_str!("test_server.py"))
            .await
            .unwrap();
        let server = SessionServerConfig {
            name: "fixture".into(),
            command: Some("python3".into()),
            args: vec![
                source.to_str().unwrap().into(),
                pid_file.to_str().unwrap().into(),
            ],
            env: BTreeMap::from([("MCP_MARKER".into(), "literal $value".into())]),
            ..Default::default()
        };
        let mut tools = ToolRegistry::default();
        let mut session =
            SessionMcp::connect(std::slice::from_ref(&server), dir.path(), None, &mut tools)
                .await
                .unwrap();
        assert!(tools.get("mcp__fixture__fail").is_some());
        let cancel = CancellationToken::new();
        let mut call = ToolCall {
            id: "call".into(),
            name: "mcp__fixture__start_agent".into(),
            arguments: json!({"task": "work"}),
        };
        let result = tools.execute(&call, None, &cancel).await;
        assert!(result.is_success(), "{}", result.output());
        let output: Value = serde_json::from_str(&result.output()).unwrap();
        assert_eq!(output["marker"], "literal $value");
        assert_eq!(output["cwd"], dir.path().to_str().unwrap());
        assert_eq!(output["task"], "work");
        call.name = "mcp__fixture__fail".into();
        assert!(!tools.execute(&call, None, &cancel).await.is_success());
        call.name = "mcp__fixture__start_agent".into();
        call.arguments = json!({"task": "rpc-error"});
        assert!(
            tools
                .execute(&call, None, &cancel)
                .await
                .output()
                .contains("call failed")
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        call.arguments = json!({"task": "hang"});
        assert!(
            tools
                .execute(&call, None, &cancelled)
                .await
                .output()
                .contains("cancelled")
        );
        tokio::time::pause();
        assert!(
            tools
                .execute(&call, None, &cancel)
                .await
                .output()
                .contains("timed out")
        );
        tokio::time::resume();
        call.arguments = json!([]);
        assert!(!tools.execute(&call, None, &cancel).await.is_success());
        let pid = tokio::fs::read_to_string(&pid_file).await.unwrap();
        session.shutdown().await;
        #[cfg(target_os = "linux")]
        assert!(!Path::new("/proc").join(pid).exists());
        call.arguments = json!({});
        assert!(!tools.execute(&call, None, &cancel).await.is_success());

        let mut broken = server;
        broken.env.insert("FAIL_DISCOVERY".into(), "1".into());
        let mut tools = ToolRegistry::default();
        assert!(
            SessionMcp::connect(&[broken], dir.path(), None, &mut tools)
                .await
                .is_err()
        );
        assert!(tools.list().is_empty());
        #[cfg(target_os = "linux")]
        assert!(
            !Path::new("/proc")
                .join(tokio::fs::read_to_string(&pid_file).await.unwrap())
                .exists()
        );
    }
}
