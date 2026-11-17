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

use super::transport::{McpService, client_info};
use super::types::call_result_from_rmcp;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const CALL_TIMEOUT: Duration = Duration::from_secs(300);

/// A local MCP process. Arguments and environment values are passed literally.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StdioServerConfig {
    pub name: String,
    pub command: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// Keeps connections alive until session shutdown. Dropping it cancels all services.
#[derive(Default)]
pub struct SessionMcp {
    services: Vec<McpService>,
}

impl SessionMcp {
    /// Initialize and discover every server before publishing any of its tools.
    /// Failure drops all connections opened by this call.
    pub async fn connect(
        servers: &[StdioServerConfig],
        cwd: &Path,
        tools: &mut ToolRegistry,
    ) -> Result<Self> {
        let mut names = HashSet::new();
        for server in servers {
            ensure!(valid_component(&server.name), "invalid MCP server name");
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
                let mut command = Command::new(&server.command);
                command
                    .args(&server.args)
                    .envs(&server.env)
                    .current_dir(cwd);
                let (transport, _) = TokioChildProcess::builder(command)
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .with_context(|| format!("could not spawn MCP server '{}'", server.name))?;
                let service = timeout(CONNECT_TIMEOUT, client_info().serve(transport))
                    .await
                    .context("MCP initialization timed out")?
                    .with_context(|| {
                        format!("could not initialize MCP server '{}'", server.name)
                    })?;
                session.services.push(service);
                let service = session.services.last().context("missing MCP connection")?;
                let discovered = timeout(CONNECT_TIMEOUT, service.list_all_tools())
                    .await
                    .context("MCP tool discovery timed out")?
                    .with_context(|| {
                        format!("could not discover tools for MCP server '{}'", server.name)
                    })?;
                for tool in discovered {
                    ensure!(valid_component(&tool.name), "invalid MCP tool name");
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

fn valid_component(name: &str) -> bool {
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

    #[tokio::test]
    async fn discovers_pages_calls_tools_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("server.py");
        let pid_file = dir.path().join("pid");
        tokio::fs::write(&source, include_str!("test_server.py"))
            .await
            .unwrap();
        let server = StdioServerConfig {
            name: "fixture".into(),
            command: "python3".into(),
            args: vec![
                source.to_str().unwrap().into(),
                pid_file.to_str().unwrap().into(),
            ],
            env: BTreeMap::from([("MCP_MARKER".into(), "literal $value".into())]),
        };
        let mut tools = ToolRegistry::default();
        let mut session =
            SessionMcp::connect(std::slice::from_ref(&server), dir.path(), &mut tools)
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
            SessionMcp::connect(&[broken], dir.path(), &mut tools)
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
