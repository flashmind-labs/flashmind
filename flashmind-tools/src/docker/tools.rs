//! Docker `Tool` trait implementations.
//!
//! Nine tools covering container lifecycle, image management, and exec.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use bollard::Docker;
use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, LogsOptions, RemoveContainerOptions,
    StopContainerOptions,
};
use bollard::exec::{CreateExecOptions, StartExecResults};
use bollard::image::{CreateImageOptions, ListImagesOptions};
use bollard::models::PortBinding;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format bytes into a human-readable size string.
fn human_size(bytes: i64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// docker_list_containers
// ---------------------------------------------------------------------------

/// List Docker containers, optionally including stopped ones.
pub struct DockerListContainersTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct ListContainersArgs {
    all: Option<bool>,
}

#[async_trait]
impl Tool for DockerListContainersTool {
    fn name(&self) -> &str {
        "docker_list_containers"
    }

    fn description(&self) -> &str {
        "List Docker containers. By default includes stopped containers."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "all": {
                    "type": "boolean",
                    "description": "Include stopped containers (default: true)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: ListContainersArgs = parse_args(self.name(), ctx.args)?;
        let all = args.all.unwrap_or(true);

        let containers = self
            .client
            .list_containers(Some(ListContainersOptions::<String> {
                all,
                ..Default::default()
            }))
            .await
            .context("listing containers")?;

        let mut out = String::new();
        if containers.is_empty() {
            out.push_str("No containers found.");
        } else {
            let _ = writeln!(
                out,
                "{:<14} {:<25} {:<20} {:<30} PORTS",
                "CONTAINER ID", "IMAGE", "STATUS", "NAMES"
            );
            for c in &containers {
                let id =
                    c.id.as_deref()
                        .map(|s| &s[..s.len().min(12)])
                        .unwrap_or("—");
                let image = c.image.as_deref().unwrap_or("—");
                let status = c.status.as_deref().unwrap_or("—");
                let names = c
                    .names
                    .as_ref()
                    .map(|ns| {
                        ns.iter()
                            .map(|n| n.trim_start_matches('/'))
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_else(|| "—".into());
                let ports = c
                    .ports
                    .as_ref()
                    .map(|ps| {
                        ps.iter()
                            .map(|p| {
                                let private = p.private_port;
                                if let Some(pub_port) = p.public_port {
                                    format!("{pub_port}->{private}")
                                } else {
                                    format!("{private}")
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "{:<14} {:<25} {:<20} {:<30} {}",
                    id, image, status, names, ports
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Docker containers".into()
    }
}

// ---------------------------------------------------------------------------
// docker_inspect_container
// ---------------------------------------------------------------------------

/// Inspect a Docker container and display key details.
pub struct DockerInspectContainerTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct InspectContainerArgs {
    id: String,
}

#[async_trait]
impl Tool for DockerInspectContainerTool {
    fn name(&self) -> &str {
        "docker_inspect_container"
    }

    fn description(&self) -> &str {
        "Inspect a Docker container and show its configuration, state, and network settings."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Container ID or name"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: InspectContainerArgs = parse_args(self.name(), ctx.args)?;

        let info = self
            .client
            .inspect_container(&args.id, None)
            .await
            .context("inspecting container")?;

        let mut out = String::new();

        // Identity
        let _ = writeln!(out, "Id: {}", info.id.as_deref().unwrap_or("—"));
        let _ = writeln!(out, "Name: {}", info.name.as_deref().unwrap_or("—"));

        // State
        if let Some(state) = &info.state {
            let _ = writeln!(out, "\n--- State ---");
            let _ = writeln!(
                out,
                "Status: {}",
                state
                    .status
                    .as_ref()
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| "—".into())
            );
            let _ = writeln!(
                out,
                "Running: {}",
                state
                    .running
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "—".into())
            );
            let _ = writeln!(
                out,
                "StartedAt: {}",
                state.started_at.as_deref().unwrap_or("—")
            );
            if let Some(finished) = &state.finished_at
                && !finished.is_empty()
                && finished != "0001-01-01T00:00:00Z"
            {
                let _ = writeln!(out, "FinishedAt: {finished}");
            }
            if let Some(exit_code) = state.exit_code {
                let _ = writeln!(out, "ExitCode: {exit_code}");
            }
        }

        // Config
        if let Some(config) = &info.config {
            let _ = writeln!(out, "\n--- Config ---");
            let _ = writeln!(out, "Image: {}", config.image.as_deref().unwrap_or("—"));
            if let Some(cmd) = &config.cmd {
                let _ = writeln!(out, "Cmd: {}", cmd.join(" "));
            }
            if let Some(env) = &config.env {
                let _ = writeln!(out, "Env:");
                for var in env {
                    let _ = writeln!(out, "  {var}");
                }
            }
        }

        // Network
        if let Some(net) = &info.network_settings {
            let _ = writeln!(out, "\n--- Network ---");
            if let Some(ip) = &net.ip_address
                && !ip.is_empty()
            {
                let _ = writeln!(out, "IPAddress: {ip}");
            }
            if let Some(ports) = &net.ports {
                for (container_port, bindings) in ports {
                    if let Some(binds) = bindings {
                        for b in binds {
                            let host_port = b.host_port.as_deref().unwrap_or("?");
                            let _ = writeln!(out, "Port: {host_port} -> {container_port}");
                        }
                    }
                }
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Inspecting container {id}")
    }
}

// ---------------------------------------------------------------------------
// docker_container_logs
// ---------------------------------------------------------------------------

/// Retrieve logs from a Docker container.
pub struct DockerContainerLogsTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct ContainerLogsArgs {
    id: String,
    tail: Option<i64>,
    since: Option<i64>,
}

#[async_trait]
impl Tool for DockerContainerLogsTool {
    fn name(&self) -> &str {
        "docker_container_logs"
    }

    fn description(&self) -> &str {
        "Retrieve recent logs from a Docker container."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Container ID or name"
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of lines from the end of the logs (default: 100)"
                },
                "since": {
                    "type": "integer",
                    "description": "Unix timestamp — only return logs since this time"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let cancel = ctx.cancel_token().clone();
        let call_id = ctx.tool_call_id.to_string();

        if cancel.is_cancelled() {
            bail!("cancelled");
        }

        let args: ContainerLogsArgs = parse_args(self.name(), ctx.args)?;
        let tail = args.tail.unwrap_or(100);

        let mut stream = self.client.logs(
            &args.id,
            Some(LogsOptions::<String> {
                stdout: true,
                stderr: true,
                tail: tail.to_string(),
                since: args.since.unwrap_or(0),
                ..Default::default()
            }),
        );

        const MAX_LOG_BYTES: usize = 100_000;
        let mut out = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(log) => {
                    let _ = write!(out, "{log}");
                }
                Err(e) => {
                    let _ = writeln!(out, "[error reading logs: {e}]");
                    break;
                }
            }
            if out.len() > MAX_LOG_BYTES {
                out.truncate(out.floor_char_boundary(MAX_LOG_BYTES));
                out.push_str("\n... (logs truncated)");
                break;
            }
            if cancel.is_cancelled() {
                let _ = writeln!(out, "\n[cancelled]");
                break;
            }
        }

        if out.is_empty() {
            out.push_str("(no logs)");
        }

        Ok(ToolResult::success(&call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Fetching logs from container {id}")
    }
}

// ---------------------------------------------------------------------------
// docker_list_images
// ---------------------------------------------------------------------------

/// List Docker images on the host.
pub struct DockerListImagesTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct ListImagesArgs {
    all: Option<bool>,
}

#[async_trait]
impl Tool for DockerListImagesTool {
    fn name(&self) -> &str {
        "docker_list_images"
    }

    fn description(&self) -> &str {
        "List Docker images on the host."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "all": {
                    "type": "boolean",
                    "description": "Include intermediate images (default: false)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: ListImagesArgs = parse_args(self.name(), ctx.args)?;
        let all = args.all.unwrap_or(false);

        let images = self
            .client
            .list_images(Some(ListImagesOptions::<String> {
                all,
                ..Default::default()
            }))
            .await
            .context("listing images")?;

        let mut out = String::new();
        if images.is_empty() {
            out.push_str("No images found.");
        } else {
            let _ = writeln!(out, "{:<14} {:<50} SIZE", "IMAGE ID", "REPOSITORY:TAG");
            for img in &images {
                let id = img.id.strip_prefix("sha256:").unwrap_or(&img.id);
                let short_id = &id[..id.len().min(12)];
                let tags = if img.repo_tags.is_empty() {
                    "<none>".into()
                } else {
                    img.repo_tags.join(", ")
                };
                let size = human_size(img.size);
                let _ = writeln!(out, "{:<14} {:<50} {}", short_id, tags, size);
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Docker images".into()
    }
}

// ---------------------------------------------------------------------------
// docker_container_exec
// ---------------------------------------------------------------------------

/// Execute a command inside a running Docker container.
pub struct DockerContainerExecTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct ContainerExecArgs {
    id: String,
    command: CommandArg,
}

/// Accepts either a single string or an array of strings for the command.
#[derive(Deserialize)]
#[serde(untagged)]
enum CommandArg {
    Single(String),
    Array(Vec<String>),
}

impl CommandArg {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec!["/bin/sh".into(), "-c".into(), s],
            Self::Array(v) => v,
        }
    }
}

const EXEC_OUTPUT_MAX: usize = 10_000;

#[async_trait]
impl Tool for DockerContainerExecTool {
    fn name(&self) -> &str {
        "docker_container_exec"
    }

    fn description(&self) -> &str {
        "Execute a command inside a running Docker container and return its output. \
         A string command is executed via `/bin/sh -c`; an array is executed directly."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Container ID or name"
                },
                "command": {
                    "description": "Command to execute — a string (run via sh -c) or array of strings",
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ]
                }
            },
            "required": ["id", "command"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let cancel = ctx.cancel_token().clone();
        let call_id = ctx.tool_call_id.to_string();

        if cancel.is_cancelled() {
            bail!("cancelled");
        }

        let args: ContainerExecArgs = parse_args(self.name(), ctx.args)?;
        let cmd = args.command.into_vec();

        let exec = self
            .client
            .create_exec(
                &args.id,
                CreateExecOptions {
                    cmd: Some(cmd),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    ..Default::default()
                },
            )
            .await
            .context("creating exec")?;

        let result = self
            .client
            .start_exec(&exec.id, None)
            .await
            .context("starting exec")?;

        let mut out = String::new();
        match result {
            StartExecResults::Attached { mut output, .. } => {
                while let Some(chunk) = output.next().await {
                    match chunk {
                        Ok(log) => {
                            let _ = write!(out, "{log}");
                        }
                        Err(e) => {
                            let _ = writeln!(out, "[error: {e}]");
                            break;
                        }
                    }
                    if out.len() > EXEC_OUTPUT_MAX {
                        out.truncate(EXEC_OUTPUT_MAX);
                        out.push_str("\n... (output truncated)");
                        break;
                    }
                    if cancel.is_cancelled() {
                        out.push_str("\n[cancelled]");
                        break;
                    }
                }
            }
            StartExecResults::Detached => {
                out.push_str("Exec started in detached mode.");
            }
        }

        if out.is_empty() {
            out.push_str("(no output)");
        }

        Ok(ToolResult::success(&call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Executing command in container {id}")
    }
}

// ---------------------------------------------------------------------------
// docker_create_container
// ---------------------------------------------------------------------------

/// Create (and optionally start) a new Docker container.
pub struct DockerCreateContainerTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct CreateContainerArgs {
    image: String,
    name: Option<String>,
    env: Option<Vec<String>>,
    cmd: Option<Vec<String>>,
    ports: Option<HashMap<String, String>>,
    start: Option<bool>,
}

#[async_trait]
impl Tool for DockerCreateContainerTool {
    fn name(&self) -> &str {
        "docker_create_container"
    }

    fn description(&self) -> &str {
        "Create a new Docker container from an image. Optionally starts it immediately."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Docker image (e.g. \"nginx:latest\")"
                },
                "name": {
                    "type": "string",
                    "description": "Optional container name"
                },
                "env": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Environment variables as KEY=VALUE strings"
                },
                "cmd": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Command to run in the container"
                },
                "ports": {
                    "type": "object",
                    "description": "Port mappings: container_port -> host_port (e.g. {\"80\": \"8080\"})",
                    "additionalProperties": { "type": "string" }
                },
                "start": {
                    "type": "boolean",
                    "description": "Start the container after creation (default: true)"
                }
            },
            "required": ["image"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: CreateContainerArgs = parse_args(self.name(), ctx.args)?;
        let should_start = args.start.unwrap_or(true);

        // Build port bindings if provided.
        let mut exposed_ports = HashMap::new();
        let mut port_bindings: HashMap<String, Option<Vec<PortBinding>>> = HashMap::new();

        if let Some(ref ports) = args.ports {
            for (container_port, host_port) in ports {
                let key = if container_port.contains('/') {
                    container_port.clone()
                } else {
                    format!("{container_port}/tcp")
                };
                exposed_ports.insert(key.clone(), HashMap::new());
                port_bindings.insert(
                    key,
                    Some(vec![PortBinding {
                        host_ip: Some("0.0.0.0".into()),
                        host_port: Some(host_port.clone()),
                    }]),
                );
            }
        }

        let host_config = if !port_bindings.is_empty() {
            Some(bollard::models::HostConfig {
                port_bindings: Some(port_bindings),
                ..Default::default()
            })
        } else {
            None
        };

        let config: Config<String> = Config {
            image: Some(args.image.clone()),
            env: args.env.clone(),
            cmd: args.cmd.clone(),
            exposed_ports: if exposed_ports.is_empty() {
                None
            } else {
                Some(exposed_ports)
            },
            host_config,
            ..Default::default()
        };

        let options = args.name.as_ref().map(|n| CreateContainerOptions {
            name: n.clone(),
            platform: None,
        });

        let response = self
            .client
            .create_container(options, config)
            .await
            .context("creating container")?;

        let container_id = &response.id;
        let mut out = format!(
            "Created container {}",
            &container_id[..container_id.len().min(12)]
        );

        if should_start {
            self.client
                .start_container::<String>(container_id, None)
                .await
                .context("starting container")?;
            let _ = write!(out, " (started)");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let image = args.get("image").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Creating container from {image}")
    }
}

// ---------------------------------------------------------------------------
// docker_stop_container
// ---------------------------------------------------------------------------

/// Stop a running Docker container.
pub struct DockerStopContainerTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct StopContainerArgs {
    id: String,
    timeout: Option<i64>,
}

#[async_trait]
impl Tool for DockerStopContainerTool {
    fn name(&self) -> &str {
        "docker_stop_container"
    }

    fn description(&self) -> &str {
        "Stop a running Docker container."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Container ID or name"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Seconds to wait before killing the container (default: 10)"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: StopContainerArgs = parse_args(self.name(), ctx.args)?;

        self.client
            .stop_container(
                &args.id,
                Some(StopContainerOptions {
                    t: args.timeout.unwrap_or(10),
                }),
            )
            .await
            .context("stopping container")?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Container {} stopped.", args.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Stopping container {id}")
    }
}

// ---------------------------------------------------------------------------
// docker_remove_container
// ---------------------------------------------------------------------------

/// Remove a Docker container.
pub struct DockerRemoveContainerTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct RemoveContainerArgs {
    id: String,
    force: Option<bool>,
}

#[async_trait]
impl Tool for DockerRemoveContainerTool {
    fn name(&self) -> &str {
        "docker_remove_container"
    }

    fn description(&self) -> &str {
        "Remove a Docker container. Use force to remove a running container."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Container ID or name"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force removal even if the container is running (default: false)"
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }

        let args: RemoveContainerArgs = parse_args(self.name(), ctx.args)?;

        self.client
            .remove_container(
                &args.id,
                Some(RemoveContainerOptions {
                    force: args.force.unwrap_or(false),
                    ..Default::default()
                }),
            )
            .await
            .context("removing container")?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Container {} removed.", args.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Removing container {id}")
    }
}

// ---------------------------------------------------------------------------
// docker_pull_image
// ---------------------------------------------------------------------------

/// Pull a Docker image from a registry.
pub struct DockerPullImageTool {
    /// Shared Docker client.
    pub client: Arc<Docker>,
}

#[derive(Deserialize)]
struct PullImageArgs {
    image: String,
}

#[async_trait]
impl Tool for DockerPullImageTool {
    fn name(&self) -> &str {
        "docker_pull_image"
    }

    fn description(&self) -> &str {
        "Pull a Docker image from a registry (e.g. \"nginx:latest\")."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image name with optional tag (e.g. \"nginx:latest\")"
                }
            },
            "required": ["image"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let cancel = ctx.cancel_token().clone();
        let call_id = ctx.tool_call_id.to_string();

        if cancel.is_cancelled() {
            bail!("cancelled");
        }

        let args: PullImageArgs = parse_args(self.name(), ctx.args)?;

        let mut stream = self.client.create_image(
            Some(CreateImageOptions {
                from_image: args.image.clone(),
                ..Default::default()
            }),
            None,
            None,
        );

        let mut last_status = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        last_status = status;
                    }
                }
                Err(e) => {
                    bail!("pull failed: {e}");
                }
            }
            if cancel.is_cancelled() {
                bail!("cancelled during pull");
            }
        }

        let msg = if last_status.is_empty() {
            format!("Pulled image {}.", args.image)
        } else {
            format!("Pulled image {}: {last_status}", args.image)
        };

        Ok(ToolResult::success(&call_id, msg))
    }

    fn humanize(&self, args: &Value) -> String {
        let image = args.get("image").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Pulling image {image}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> Arc<Docker> {
        // Connect to a fake HTTP endpoint — tests only validate schemas, names,
        // and humanize output, not actual Docker calls.
        Arc::new(
            Docker::connect_with_http("http://localhost:1", 4, bollard::API_DEFAULT_VERSION)
                .expect("docker client for tests"),
        )
    }

    #[test]
    fn tool_names() {
        let c = make_client();
        assert_eq!(
            DockerListContainersTool { client: c.clone() }.name(),
            "docker_list_containers"
        );
        assert_eq!(
            DockerInspectContainerTool { client: c.clone() }.name(),
            "docker_inspect_container"
        );
        assert_eq!(
            DockerContainerLogsTool { client: c.clone() }.name(),
            "docker_container_logs"
        );
        assert_eq!(
            DockerListImagesTool { client: c.clone() }.name(),
            "docker_list_images"
        );
        assert_eq!(
            DockerContainerExecTool { client: c.clone() }.name(),
            "docker_container_exec"
        );
        assert_eq!(
            DockerCreateContainerTool { client: c.clone() }.name(),
            "docker_create_container"
        );
        assert_eq!(
            DockerStopContainerTool { client: c.clone() }.name(),
            "docker_stop_container"
        );
        assert_eq!(
            DockerRemoveContainerTool { client: c.clone() }.name(),
            "docker_remove_container"
        );
        assert_eq!(
            DockerPullImageTool { client: c.clone() }.name(),
            "docker_pull_image"
        );
    }

    #[test]
    fn humanize_tools() {
        let c = make_client();

        assert_eq!(
            DockerListContainersTool { client: c.clone() }.humanize(&json!({})),
            "Listing Docker containers"
        );
        assert_eq!(
            DockerListImagesTool { client: c.clone() }.humanize(&json!({})),
            "Listing Docker images"
        );

        let h =
            DockerInspectContainerTool { client: c.clone() }.humanize(&json!({ "id": "abc123" }));
        assert!(h.contains("abc123"));

        let h = DockerContainerLogsTool { client: c.clone() }.humanize(&json!({ "id": "myapp" }));
        assert!(h.contains("myapp"));

        let h = DockerContainerExecTool { client: c.clone() }.humanize(&json!({ "id": "web" }));
        assert!(h.contains("web"));

        let h = DockerCreateContainerTool { client: c.clone() }
            .humanize(&json!({ "image": "nginx:latest" }));
        assert!(h.contains("nginx:latest"));

        let h = DockerStopContainerTool { client: c.clone() }.humanize(&json!({ "id": "runner" }));
        assert!(h.contains("runner"));

        let h = DockerRemoveContainerTool { client: c.clone() }.humanize(&json!({ "id": "old" }));
        assert!(h.contains("old"));

        let h = DockerPullImageTool { client: c.clone() }.humanize(&json!({ "image": "redis:7" }));
        assert!(h.contains("redis:7"));
    }

    #[test]
    fn parameters_are_objects() {
        let c = make_client();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(DockerListContainersTool { client: c.clone() }),
            Box::new(DockerInspectContainerTool { client: c.clone() }),
            Box::new(DockerContainerLogsTool { client: c.clone() }),
            Box::new(DockerListImagesTool { client: c.clone() }),
            Box::new(DockerContainerExecTool { client: c.clone() }),
            Box::new(DockerCreateContainerTool { client: c.clone() }),
            Box::new(DockerStopContainerTool { client: c.clone() }),
            Box::new(DockerRemoveContainerTool { client: c.clone() }),
            Box::new(DockerPullImageTool { client: c.clone() }),
        ];

        for tool in &tools {
            let params = tool.parameters();
            assert_eq!(
                params.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "tool '{}' parameters should have type: object",
                tool.name()
            );
            assert!(
                params.get("properties").is_some(),
                "tool '{}' parameters should have properties",
                tool.name()
            );
        }
    }

    #[test]
    fn required_fields_present() {
        let c = make_client();

        let inspect = DockerInspectContainerTool { client: c.clone() };
        let required: Vec<String> = inspect.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(required.contains(&"id".to_string()));

        let exec = DockerContainerExecTool { client: c.clone() };
        let required: Vec<String> = exec.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(required.contains(&"id".to_string()));
        assert!(required.contains(&"command".to_string()));

        let create = DockerCreateContainerTool { client: c.clone() };
        let required: Vec<String> = create.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(required.contains(&"image".to_string()));

        let pull = DockerPullImageTool { client: c.clone() };
        let required: Vec<String> = pull.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(required.contains(&"image".to_string()));
    }

    #[test]
    fn human_size_formatting() {
        assert_eq!(human_size(500), "500 B");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(10_485_760), "10.0 MB");
        assert_eq!(human_size(1_610_612_736), "1.5 GB");
    }
}
