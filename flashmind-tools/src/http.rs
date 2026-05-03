//! HTTP request tool for making arbitrary HTTP requests.

use async_trait::async_trait;
use metrics;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

use tracing::{Instrument, info_span};
use url::Url;
use uuid::Uuid;

use crate::utils::http_client;
use crate::utils::truncate_utf8;
use flashmind_types::tool::ToolContext;
use flashmind_types::tool::{Tool, ToolResult};

const SPILL_THRESHOLD: usize = 50 * 1024;
const HEAD_PREVIEW_BYTES: usize = 4 * 1024;
const NO_WORKSPACE_TRUNCATE: usize = 50 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 30;

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_unspecified()
                // 169.254.169.254 — cloud metadata
                || v4.octets() == [169, 254, 169, 254]
        }
        IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
    }
}

fn validate_url(raw: &str) -> Result<Url, String> {
    let parsed = Url::parse(raw).map_err(|e| format!("Invalid URL: {e}"))?;

    match parsed.scheme() {
        "http" | "https" => {}
        s => return Err(format!("Unsupported scheme: {s}")),
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| "URL has no host".to_string())?;

    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_private_ip(ip) {
            return Err(format!(
                "Blocked: {host} resolves to a private/reserved address"
            ));
        }
    } else {
        // Hostname-based checks
        let lower = host.to_ascii_lowercase();
        if lower == "localhost"
            || lower.ends_with(".local")
            || lower.ends_with(".internal")
            || lower == "metadata.google.internal"
        {
            return Err(format!("Blocked: {host} is a private/reserved hostname"));
        }
    }

    Ok(parsed)
}

#[derive(Deserialize)]
struct HttpRequestArgs {
    url: String,
    method: Option<String>,
    headers: Option<HashMap<String, String>>,
    body: Option<String>,
    timeout_secs: Option<u64>,
    /// Save response body to this file path instead of returning as text.
    download_to: Option<String>,
}

pub struct HttpRequestTool {
    client: reqwest::Client,
}

impl HttpRequestTool {
    pub fn new() -> Self {
        Self {
            client: http_client(),
        }
    }
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SpillDecision {
    Inline,
    Truncate,
    Spill,
}

fn decide_spill(len: usize, has_workspace: bool) -> SpillDecision {
    if len <= SPILL_THRESHOLD {
        SpillDecision::Inline
    } else if !has_workspace {
        SpillDecision::Truncate
    } else {
        SpillDecision::Spill
    }
}

/// Map a Content-Type header to a file extension and binary flag.
fn ext_for_content_type(ct: &str) -> (&'static str, bool) {
    let mime = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    match mime.as_str() {
        "application/json" => return ("json", false),
        "text/html" | "application/xhtml+xml" => return ("html", false),
        "text/xml" | "application/xml" => return ("xml", false),
        "text/csv" => return ("csv", false),
        "text/javascript" | "application/javascript" => return ("js", false),
        "text/css" => return ("css", false),
        "text/markdown" => return ("md", false),
        "application/pdf" => return ("pdf", true),
        "application/zip" => return ("zip", true),
        "application/gzip" => return ("gz", true),
        "image/png" => return ("png", true),
        "image/jpeg" => return ("jpg", true),
        "image/gif" => return ("gif", true),
        "image/webp" => return ("webp", true),
        "image/svg+xml" => return ("svg", false),
        _ => {}
    }

    if mime.ends_with("+json") {
        return ("json", false);
    }
    if mime.ends_with("+xml") {
        return ("xml", false);
    }
    if mime.starts_with("text/") {
        return ("txt", false);
    }
    if mime.starts_with("image/")
        || mime.starts_with("video/")
        || mime.starts_with("audio/")
        || mime == "application/octet-stream"
        || mime.is_empty()
    {
        return ("bin", true);
    }

    ("bin", true)
}

/// Write `bytes` to `~/.flashagent/.http_cache/{uuid8}.{ext}` and return the full path.
fn cache_dir() -> std::io::Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| std::io::Error::other("could not determine home directory"))?;
    Ok(home.join(".flashagent/.http_cache"))
}

async fn spill_to_cache(bytes: &[u8], ext: &str) -> std::io::Result<PathBuf> {
    let dir = cache_dir()?;
    tokio::fs::create_dir_all(&dir).await?;
    let id = &Uuid::new_v4().to_string()[..8];
    let path = dir.join(format!("{id}.{ext}"));
    tokio::fs::write(&path, bytes).await?;
    Ok(path)
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Make an HTTP request to an API endpoint. Supports GET, POST, PUT, DELETE, PATCH with headers and JSON/text body. Returns status code, headers, and response body. Large responses are spilled to `~/.flashagent/.http_cache/` and can be paginated with `file_read`/`read_lines`. Use `download_to` to save the response to a chosen path. For fetching web pages, use web_fetch instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP method (default: GET)"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs"
                },
                "body": {
                    "type": "string",
                    "description": "Optional request body"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default: 30)"
                },
                "download_to": {
                    "type": "string",
                    "description": "Save response body to this file path instead of returning as text. Useful for downloading binaries, archives, etc."
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();
        let args: HttpRequestArgs = ctx.parse_args(self.name())?;

        if let Err(reason) = validate_url(&args.url) {
            return Ok(ToolResult::failure(ctx.tool_call_id, reason));
        }

        let method = args.method.as_deref().unwrap_or("GET");
        let timeout_secs = args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);

        tracing::debug!(method, url = %args.url, "http: sending request");

        let request = match method {
            "GET" => self.client.get(&args.url),
            "POST" => self.client.post(&args.url),
            "PUT" => self.client.put(&args.url),
            "DELETE" => self.client.delete(&args.url),
            "PATCH" => self.client.patch(&args.url),
            other => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!(
                        "Unsupported HTTP method: {}. Use GET, POST, PUT, DELETE, or PATCH.",
                        other
                    ),
                ));
            }
        };

        let mut request = request;

        if let Some(headers) = &args.headers {
            for (key, value) in headers {
                request = request.header(key.as_str(), value.as_str());
            }
        }

        if let Some(body) = &args.body {
            request = request.body(body.clone());
        }

        let timeout = Duration::from_secs(timeout_secs);

        match tokio::time::timeout(
            timeout,
            request.send().instrument(info_span!(
                target: "prompt_trace", "step",
                step = "api_call",
                detail = format!("{} {} timeout={}s", method, args.url, timeout_secs).as_str(),
            )),
        )
        .await
        {
            Ok(Ok(response)) => {
                let status = response.status().as_u16();
                metrics::counter!("tools.http.calls").increment(1);
                tracing::debug!(status, url = %args.url, "http: response received");
                let content_type = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown")
                    .to_string();
                let content_length = response
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown")
                    .to_string();

                if let Some(download_path) = &args.download_to {
                    let path = crate::file_ops::resolve_path(download_path, ctx.working_dir);

                    // Block absolute paths outside working directory
                    if let Some(wd) = ctx.working_dir {
                        let canonical_wd = std::fs::canonicalize(wd).unwrap_or_else(|_| wd.clone());
                        let canonical_path = if path.exists() {
                            std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone())
                        } else if let Some(parent) = path.parent() {
                            std::fs::canonicalize(parent)
                                .map(|p| p.join(path.file_name().unwrap_or_default()))
                                .unwrap_or_else(|_| path.clone())
                        } else {
                            path.clone()
                        };
                        if !canonical_path.starts_with(&canonical_wd) {
                            return Ok(ToolResult::failure(
                                ctx.tool_call_id,
                                format!(
                                    "download_to path must be within the working directory: {}",
                                    wd.display()
                                ),
                            ));
                        }
                    }

                    match tokio::fs::write(&path, response.bytes().await?).await {
                        Ok(_) => {
                            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                            Ok(ToolResult::success(
                                ctx.tool_call_id,
                                format!(
                                    "Downloaded {} bytes to {}\n[{}]({})",
                                    size,
                                    path.display(),
                                    path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
                                    path.display()
                                ),
                            ))
                        }
                        Err(e) => Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!("Failed to write file: {}", e),
                        )),
                    }
                } else {
                    let bytes = match response.bytes().await {
                        Ok(b) => b,
                        Err(e) => {
                            return Ok(ToolResult::failure(
                                ctx.tool_call_id,
                                format!("HTTP {}\nFailed to read response body: {}", status, e),
                            ));
                        }
                    };

                    let len = bytes.len();
                    let (ext, is_binary) = ext_for_content_type(&content_type);
                    let decision = decide_spill(len, ctx.working_dir.is_some());

                    let output = match decision {
                        SpillDecision::Inline => {
                            let body = String::from_utf8_lossy(&bytes);
                            format!(
                                "HTTP {}\nContent-Type: {}\nContent-Length: {}\n\n{}",
                                status, content_type, content_length, body
                            )
                        }
                        SpillDecision::Truncate => {
                            let body = String::from_utf8_lossy(&bytes);
                            let slice = truncate_utf8(&body, NO_WORKSPACE_TRUNCATE);
                            format!(
                                "HTTP {}\nContent-Type: {}\nContent-Length: {}\n\n{}\n\n[truncated to 50KB]",
                                status, content_type, content_length, slice
                            )
                        }
                        SpillDecision::Spill => match spill_to_cache(&bytes, ext).await {
                            Ok(path) => {
                                let display = path.display();
                                if is_binary {
                                    format!(
                                        "HTTP {}\nContent-Type: {}\nContent-Length: {}\n\nBinary response ({} bytes) saved to: {}\nTip: next time, pass `download_to` to `http_request` for binary downloads.\nNo preview available (binary content).",
                                        status, content_type, content_length, len, display
                                    )
                                } else {
                                    let lossy = String::from_utf8_lossy(&bytes);
                                    let preview = truncate_utf8(&lossy, HEAD_PREVIEW_BYTES);
                                    let preview_len = preview.len();
                                    format!(
                                        "HTTP {}\nContent-Type: {}\nContent-Length: {}\n\nResponse body was {} bytes (exceeds {}-byte inline limit).\nFull body written to: {} ({} bytes)\n\nUse `read_lines` (path=\"{}\", from_line=N, to_line=M) to paginate,\nor `file_read` (path=\"{}\") for the whole file (auto-truncates at 1000 lines).\nDo NOT retry this request — the body is already on disk.\n\n--- HEAD PREVIEW (first {} bytes) ---\n{}\n--- END PREVIEW ---",
                                        status,
                                        content_type,
                                        content_length,
                                        len,
                                        SPILL_THRESHOLD,
                                        display,
                                        len,
                                        display,
                                        display,
                                        preview_len,
                                        preview,
                                    )
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "http: spill_to_cache failed, falling back to truncation");
                                let body = String::from_utf8_lossy(&bytes);
                                let slice = truncate_utf8(&body, NO_WORKSPACE_TRUNCATE);
                                format!(
                                    "HTTP {}\nContent-Type: {}\nContent-Length: {}\n\n{}\n\n[truncated to 50KB]\n\n[note: failed to write .http_cache ({}); response was truncated]",
                                    status, content_type, content_length, slice, e
                                )
                            }
                        },
                    };

                    Ok(ToolResult::success(ctx.tool_call_id, output))
                }
            }
            Ok(Err(e)) => {
                metrics::counter!("tools.http.calls").increment(1);
                metrics::histogram!("tools.http.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                tracing::warn!(error = %e, url = %args.url, "http: request failed");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Request failed: {}", e),
                ))
            }
            Err(_) => {
                metrics::counter!("tools.http.calls").increment(1);
                metrics::histogram!("tools.http.duration_seconds")
                    .record(start.elapsed().as_secs_f64());
                tracing::warn!(timeout_secs, url = %args.url, "http: request timed out");
                Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Request timed out after {} seconds", timeout_secs),
                ))
            }
        }
    }

    fn humanize(&self, args: &Value) -> String {
        let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let method = args.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
        let download_to = args.get("download_to").and_then(|v| v.as_str());
        if let Some(dest) = download_to {
            format!("Downloading {url} into {dest}")
        } else {
            format!("{method} {url}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name() {
        assert_eq!(HttpRequestTool::new().name(), "http_request");
    }

    #[test]
    fn test_default_method() {
        let args: HttpRequestArgs =
            serde_json::from_value(json!({ "url": "https://example.com" })).unwrap();
        assert_eq!(args.method, None);
        assert_eq!(args.method.as_deref().unwrap_or("GET"), "GET");
    }

    #[test]
    fn test_parameters_schema() {
        let tool = HttpRequestTool::new();
        let params = tool.parameters();
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "url");
        assert!(params["properties"]["url"].is_object());
        assert!(params["properties"]["method"].is_object());
        assert!(params["properties"]["headers"].is_object());
        assert!(params["properties"]["body"].is_object());
        assert!(params["properties"]["timeout_secs"].is_object());
    }

    #[test]
    fn ext_for_content_type_known_text_types() {
        assert_eq!(ext_for_content_type("application/json"), ("json", false));
        assert_eq!(
            ext_for_content_type("application/json; charset=utf-8"),
            ("json", false)
        );
        assert_eq!(ext_for_content_type("text/html"), ("html", false));
        assert_eq!(
            ext_for_content_type("application/xhtml+xml"),
            ("html", false)
        );
        assert_eq!(ext_for_content_type("text/xml"), ("xml", false));
        assert_eq!(ext_for_content_type("application/xml"), ("xml", false));
        assert_eq!(ext_for_content_type("text/csv"), ("csv", false));
        assert_eq!(
            ext_for_content_type("application/javascript"),
            ("js", false)
        );
        assert_eq!(ext_for_content_type("text/css"), ("css", false));
        assert_eq!(ext_for_content_type("text/markdown"), ("md", false));
    }

    #[test]
    fn ext_for_content_type_suffix_structured_syntax() {
        assert_eq!(
            ext_for_content_type("application/vnd.api+json"),
            ("json", false)
        );
        assert_eq!(
            ext_for_content_type("application/atom+xml; charset=utf-8"),
            ("xml", false)
        );
    }

    #[test]
    fn ext_for_content_type_binary() {
        assert_eq!(ext_for_content_type("application/pdf"), ("pdf", true));
        assert_eq!(ext_for_content_type("application/zip"), ("zip", true));
        assert_eq!(ext_for_content_type("application/gzip"), ("gz", true));
        assert_eq!(ext_for_content_type("image/png"), ("png", true));
        assert_eq!(ext_for_content_type("image/jpeg"), ("jpg", true));
        assert_eq!(ext_for_content_type("image/webp"), ("webp", true));
        assert_eq!(ext_for_content_type("image/svg+xml"), ("svg", false));
        assert_eq!(
            ext_for_content_type("application/octet-stream"),
            ("bin", true)
        );
        assert_eq!(ext_for_content_type("video/mp4"), ("bin", true));
        assert_eq!(ext_for_content_type("audio/mpeg"), ("bin", true));
    }

    #[test]
    fn ext_for_content_type_unknown_text_and_empty() {
        assert_eq!(ext_for_content_type("text/plain"), ("txt", false));
        assert_eq!(ext_for_content_type("text/x-something"), ("txt", false));
        assert_eq!(ext_for_content_type(""), ("bin", true));
        assert_eq!(ext_for_content_type("unknown"), ("bin", true));
        assert_eq!(ext_for_content_type("unknown/thing"), ("bin", true));
    }

    #[test]
    fn decide_spill_at_boundaries() {
        assert_eq!(
            decide_spill(SPILL_THRESHOLD - 1, true),
            SpillDecision::Inline
        );
        assert_eq!(decide_spill(SPILL_THRESHOLD, true), SpillDecision::Inline);
        assert_eq!(
            decide_spill(SPILL_THRESHOLD + 1, true),
            SpillDecision::Spill
        );
        assert_eq!(
            decide_spill(SPILL_THRESHOLD + 1, false),
            SpillDecision::Truncate
        );
        assert_eq!(decide_spill(0, true), SpillDecision::Inline);
        assert_eq!(decide_spill(0, false), SpillDecision::Inline);
    }

    #[tokio::test]
    async fn spill_to_cache_writes_file() {
        let body = vec![0x41u8; 200 * 1024];

        let path = spill_to_cache(&body, "json").await.unwrap();

        assert!(path.exists());
        assert_eq!(path.parent().unwrap().file_name().unwrap(), ".http_cache");
        assert_eq!(path.extension().unwrap(), "json");

        let stem = path.file_stem().unwrap().to_str().unwrap();
        assert_eq!(stem.len(), 8);
        assert!(stem.chars().all(|c| c.is_ascii_hexdigit()));

        let read_back = tokio::fs::read(&path).await.unwrap();
        assert_eq!(read_back, body);

        // Verify path is under home/.flashagent/.http_cache
        let home = dirs::home_dir().unwrap();
        let expected_dir = home.join(".flashagent/.http_cache");
        assert!(
            path.starts_with(&expected_dir),
            "path {} should be under {}",
            path.display(),
            expected_dir.display()
        );
    }

    #[test]
    fn validate_url_allows_public() {
        assert!(validate_url("https://api.example.com/v1/data").is_ok());
        assert!(validate_url("http://93.184.216.34/page").is_ok());
    }

    #[test]
    fn validate_url_blocks_private_ips() {
        assert!(validate_url("http://127.0.0.1/admin").is_err());
        assert!(validate_url("http://10.0.0.1/internal").is_err());
        assert!(validate_url("http://192.168.1.1/router").is_err());
        assert!(validate_url("http://172.16.0.1/private").is_err());
        assert!(validate_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_url("http://0.0.0.0/").is_err());
    }

    #[test]
    fn validate_url_blocks_private_hostnames() {
        assert!(validate_url("http://localhost/admin").is_err());
        assert!(validate_url("http://myhost.local/api").is_err());
        assert!(validate_url("http://metadata.google.internal/").is_err());
    }

    #[test]
    fn validate_url_blocks_bad_schemes() {
        assert!(validate_url("ftp://example.com/file").is_err());
        assert!(validate_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn validate_url_rejects_invalid() {
        assert!(validate_url("not a url").is_err());
        assert!(validate_url("").is_err());
    }
}
