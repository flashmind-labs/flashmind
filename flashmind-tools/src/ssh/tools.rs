//! SSH `Tool` trait implementations.
//!
//! Three tools: [`SshExecTool`] for remote command execution, [`SshUploadTool`]
//! for uploading local files, and [`SshDownloadTool`] for downloading remote files.

use std::sync::Arc;

use async_trait::async_trait;
use russh::ChannelMsg;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult};

use super::{SshProfile, connect};

/// Maximum output length before truncation (chars).
const MAX_OUTPUT_CHARS: usize = 50_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Look up a profile by name from the shared list.
fn find_profile<'a>(profiles: &'a [SshProfile], name: &str) -> anyhow::Result<&'a SshProfile> {
    profiles.iter().find(|p| p.name == name).ok_or_else(|| {
        let known: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        anyhow::anyhow!(
            "Unknown SSH profile '{name}'. Available profiles: {}",
            known.join(", ")
        )
    })
}

/// Truncate a string to `MAX_OUTPUT_CHARS`, appending a notice if truncated.
fn truncate(s: &str) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        s.to_string()
    } else {
        let mut out = s[..MAX_OUTPUT_CHARS].to_string();
        out.push_str("\n\n[output truncated]");
        out
    }
}

// ---------------------------------------------------------------------------
// ssh_exec
// ---------------------------------------------------------------------------

/// Execute a command on a remote host via SSH.
///
/// Connects to the named profile, runs the command, and returns combined
/// stdout/stderr output together with the exit status.
pub struct SshExecTool {
    /// Available SSH connection profiles.
    pub profiles: Arc<Vec<SshProfile>>,
    /// When `true` the tool description emphasises read-only usage.
    pub readonly: bool,
}

#[derive(Deserialize)]
struct ExecArgs {
    profile: String,
    command: String,
}

#[async_trait]
impl Tool for SshExecTool {
    fn name(&self) -> &str {
        "ssh_exec"
    }

    fn description(&self) -> &str {
        if self.readonly {
            "Execute a read-only command on a remote host via SSH. \
             Use this for inspecting system state, reading logs, checking \
             status, etc. Avoid commands that modify the remote system."
        } else {
            "Execute a command on a remote host via SSH. \
             Returns stdout, stderr, and the exit status."
        }
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "description": "Name of the SSH profile to connect to"
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute on the remote host"
                }
            },
            "required": ["profile", "command"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: ExecArgs = ctx.parse_args(self.name())?;
        let profile = find_profile(&self.profiles, &args.profile)?;

        debug!(profile = %args.profile, command = %args.command, "ssh_exec");

        let handle = connect(profile).await?;
        let mut channel = handle.channel_open_session().await?;
        channel.exec(true, args.command.as_bytes()).await?;

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exit_status: Option<u32> = None;

        loop {
            if ctx.cancel_token().is_cancelled() {
                let _ = channel.close().await;
                return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
            }

            match channel.wait().await {
                Some(ChannelMsg::Data { data }) => stdout.extend_from_slice(&data),
                Some(ChannelMsg::ExtendedData { data, ext: 1 }) => {
                    stderr.extend_from_slice(&data);
                }
                Some(ChannelMsg::ExitStatus { exit_status: code }) => {
                    exit_status = Some(code);
                }
                None => break,
                _ => {}
            }
        }

        let stdout_str = String::from_utf8_lossy(&stdout);
        let stderr_str = String::from_utf8_lossy(&stderr);
        let code = exit_status.unwrap_or(0);

        let mut out = String::new();
        if !stdout_str.is_empty() {
            out.push_str(&stdout_str);
        }
        if !stderr_str.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("[stderr]\n");
            out.push_str(&stderr_str);
        }
        out.push_str(&format!("\n\n[exit status: {code}]"));

        let out = truncate(&out);

        if code == 0 {
            Ok(ToolResult::success(ctx.tool_call_id, out))
        } else {
            Ok(ToolResult::failure(ctx.tool_call_id, out))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let profile = args.get("profile").and_then(|v| v.as_str()).unwrap_or("?");
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Running `{command}` on {profile} via SSH")
    }
}

// ---------------------------------------------------------------------------
// ssh_upload
// ---------------------------------------------------------------------------

/// Upload a local file to a remote host via SSH.
///
/// Streams the file content through `cat > remote_path` over an exec channel.
pub struct SshUploadTool {
    /// Available SSH connection profiles.
    pub profiles: Arc<Vec<SshProfile>>,
}

#[derive(Deserialize)]
struct UploadArgs {
    profile: String,
    local_path: String,
    remote_path: String,
}

#[async_trait]
impl Tool for SshUploadTool {
    fn name(&self) -> &str {
        "ssh_upload"
    }

    fn description(&self) -> &str {
        "Upload a local file to a remote host via SSH. \
         The file is streamed through an exec channel using `cat > path`."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "description": "Name of the SSH profile to connect to"
                },
                "local_path": {
                    "type": "string",
                    "description": "Path to the local file to upload"
                },
                "remote_path": {
                    "type": "string",
                    "description": "Destination path on the remote host"
                }
            },
            "required": ["profile", "local_path", "remote_path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        let args: UploadArgs = ctx.parse_args(self.name())?;
        let profile = find_profile(&self.profiles, &args.profile)?;

        debug!(
            profile = %args.profile,
            local = %args.local_path,
            remote = %args.remote_path,
            "ssh_upload"
        );

        let data = tokio::fs::read(&args.local_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read local file '{}': {e}", args.local_path))?;
        let size = data.len();

        let handle = connect(profile).await?;
        let mut channel = handle.channel_open_session().await?;

        // Use shell-escaped path in the cat command
        let cmd = format!("cat > '{}'", args.remote_path.replace('\'', "'\\''"));
        channel.exec(true, cmd.as_bytes()).await?;
        channel.data(&data[..]).await?;
        channel.eof().await?;

        // Wait for the channel to close so we capture exit status
        let mut exit_status: Option<u32> = None;
        loop {
            match channel.wait().await {
                Some(ChannelMsg::ExitStatus { exit_status: code }) => {
                    exit_status = Some(code);
                }
                None => break,
                _ => {}
            }
        }

        let code = exit_status.unwrap_or(0);
        if code == 0 {
            Ok(ToolResult::success(
                ctx.tool_call_id,
                format!(
                    "Uploaded {} ({size} bytes) to {}:{}",
                    args.local_path, args.profile, args.remote_path
                ),
            ))
        } else {
            Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Upload failed with exit status {code}"),
            ))
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let profile = args.get("profile").and_then(|v| v.as_str()).unwrap_or("?");
        let local = args
            .get("local_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let remote = args
            .get("remote_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        format!("Uploading {local} to {profile}:{remote}")
    }
}

// ---------------------------------------------------------------------------
// ssh_download
// ---------------------------------------------------------------------------

/// Download a file from a remote host via SSH.
///
/// Reads the remote file through `cat path` and writes it locally.
pub struct SshDownloadTool {
    /// Available SSH connection profiles.
    pub profiles: Arc<Vec<SshProfile>>,
}

#[derive(Deserialize)]
struct DownloadArgs {
    profile: String,
    remote_path: String,
    local_path: String,
}

#[async_trait]
impl Tool for SshDownloadTool {
    fn name(&self) -> &str {
        "ssh_download"
    }

    fn description(&self) -> &str {
        "Download a file from a remote host via SSH. \
         The file is read through `cat path` and saved locally."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "description": "Name of the SSH profile to connect to"
                },
                "remote_path": {
                    "type": "string",
                    "description": "Path to the file on the remote host"
                },
                "local_path": {
                    "type": "string",
                    "description": "Local destination path to save the file"
                }
            },
            "required": ["profile", "remote_path", "local_path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
        }

        const MAX_DOWNLOAD_SIZE: usize = 100 * 1024 * 1024; // 100 MB

        let args: DownloadArgs = ctx.parse_args(self.name())?;
        let profile = find_profile(&self.profiles, &args.profile)?;

        debug!(
            profile = %args.profile,
            remote = %args.remote_path,
            local = %args.local_path,
            "ssh_download"
        );

        let handle = connect(profile).await?;
        let mut channel = handle.channel_open_session().await?;

        let cmd = format!("cat '{}'", args.remote_path.replace('\'', "'\\''"));
        channel.exec(true, cmd.as_bytes()).await?;

        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        let mut exit_status: Option<u32> = None;

        loop {
            if ctx.cancel_token().is_cancelled() {
                let _ = channel.close().await;
                return Ok(ToolResult::failure(ctx.tool_call_id, "Cancelled"));
            }

            match channel.wait().await {
                Some(ChannelMsg::Data { ref data }) => {
                    stdout_buf.extend_from_slice(data);
                    if stdout_buf.len() > MAX_DOWNLOAD_SIZE {
                        let _ = channel.close().await;
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!(
                                "Download aborted: file exceeds {}MB limit",
                                MAX_DOWNLOAD_SIZE / (1024 * 1024)
                            ),
                        ));
                    }
                }
                Some(ChannelMsg::ExtendedData { data: d, ext: 1 }) => {
                    stderr_buf.extend_from_slice(&d);
                }
                Some(ChannelMsg::ExitStatus { exit_status: code }) => {
                    exit_status = Some(code);
                }
                None => break,
                _ => {}
            }
        }

        let code = exit_status.unwrap_or(0);
        if code != 0 {
            let err = String::from_utf8_lossy(&stderr_buf);
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Remote cat failed (exit {code}): {err}"),
            ));
        }

        let size = stdout_buf.len();
        tokio::fs::write(&args.local_path, &stdout_buf)
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to write local file '{}': {e}", args.local_path)
            })?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Downloaded {}:{} to {} ({size} bytes)",
                args.profile, args.remote_path, args.local_path
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let profile = args.get("profile").and_then(|v| v.as_str()).unwrap_or("?");
        let remote = args
            .get("remote_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let local = args
            .get("local_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        format!("Downloading {profile}:{remote} to {local}")
    }
}
