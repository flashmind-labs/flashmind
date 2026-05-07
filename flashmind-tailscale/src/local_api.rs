//! Unix socket HTTP client for the Tailscale local API (`tailscaled`).
//!
//! Communicates directly with the Tailscale daemon over its Unix domain socket,
//! bypassing the CLI for faster and richer responses.

use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, Uri};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;

use crate::error::TailscaleError;
use crate::status::{RawStatus, TailscaleStatus};
use crate::types::{PingResult, RawCertDomains, RawWhoisResponse, WhoisResult};

// ---------------------------------------------------------------------------
// Known socket paths
// ---------------------------------------------------------------------------

const SOCKET_LINUX: &str = "/var/run/tailscale/tailscaled.sock";
const SOCKET_MACOS_OPENSOURCE: &str = "/var/run/tailscale/tailscaled.sock";

fn socket_macos_appstore() -> Option<PathBuf> {
    home::home_dir().map(|h| {
        h.join("Library/Group Containers/io.tailscale.ipn.macos/daemon-socket")
    })
}

fn known_socket_paths() -> Vec<PathBuf> {
    let mut paths = vec![
        PathBuf::from(SOCKET_LINUX),
        PathBuf::from(SOCKET_MACOS_OPENSOURCE),
    ];
    if let Some(p) = socket_macos_appstore() {
        paths.push(p);
    }
    paths.dedup();
    paths
}

// ---------------------------------------------------------------------------
// LocalApi
// ---------------------------------------------------------------------------

/// HTTP client that talks to `tailscaled` via its Unix domain socket.
pub struct LocalApi {
    socket_path: PathBuf,
}

impl LocalApi {
    /// Create a client using an explicit socket path.
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// Auto-detect the socket by probing known paths.
    ///
    /// Returns `None` if no socket is found.
    pub fn detect() -> Option<Self> {
        for path in known_socket_paths() {
            if path.exists() {
                return Some(Self::new(path));
            }
        }
        None
    }

    /// The socket path this client is connected to.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Check if the socket is reachable by issuing a lightweight status request.
    pub async fn is_reachable(&self) -> bool {
        self.get("/localapi/v0/status").await.is_ok()
    }

    // -------------------------------------------------------------------
    // High-level endpoints
    // -------------------------------------------------------------------

    /// `GET /localapi/v0/status` — full tailnet status.
    pub async fn status(&self) -> Result<TailscaleStatus, TailscaleError> {
        let body = self.get("/localapi/v0/status").await?;
        let raw: RawStatus = serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse status: {e}")))?;
        Ok(raw.into_status())
    }

    /// `GET /localapi/v0/whois?addr=<addr>` — identify a Tailscale peer by IP:port.
    pub async fn whois(&self, addr: &str) -> Result<WhoisResult, TailscaleError> {
        let path = format!(
            "/localapi/v0/whois?addr={}",
            urlencoded(addr)
        );
        let body = self.get(&path).await?;
        let raw: RawWhoisResponse = serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse whois: {e}")))?;
        Ok(raw.into_result())
    }

    /// `GET /localapi/v0/prefs` — current daemon preferences.
    pub async fn prefs(&self) -> Result<serde_json::Value, TailscaleError> {
        let body = self.get("/localapi/v0/prefs").await?;
        serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse prefs: {e}")))
    }

    /// `POST /localapi/v0/ping` — ping a Tailscale peer.
    pub async fn ping(&self, ip: &str, ping_type: &str) -> Result<PingResult, TailscaleError> {
        let path = format!(
            "/localapi/v0/ping?ip={}&type={}",
            urlencoded(ip),
            urlencoded(ping_type),
        );
        let body = self.post(&path, Bytes::new()).await?;
        serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse ping: {e}")))
    }

    /// `GET /localapi/v0/dns-config` — current DNS configuration.
    pub async fn dns_config(&self) -> Result<serde_json::Value, TailscaleError> {
        let body = self.get("/localapi/v0/dns-config").await?;
        serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse dns-config: {e}")))
    }

    /// `GET /localapi/v0/cert-domains` — domains for which certs can be issued.
    pub async fn cert_domains(&self) -> Result<Vec<String>, TailscaleError> {
        let body = self.get("/localapi/v0/cert-domains").await?;
        let raw: RawCertDomains = serde_json::from_slice(&body)
            .map_err(|e| TailscaleError::Api(format!("failed to parse cert-domains: {e}")))?;
        Ok(raw.domains)
    }

    // -------------------------------------------------------------------
    // Low-level HTTP helpers
    // -------------------------------------------------------------------

    /// Issue a GET request to the daemon.
    pub async fn get(&self, path: &str) -> Result<Bytes, TailscaleError> {
        let resp = self.request("GET", path, Full::new(Bytes::new())).await?;
        read_response(resp).await
    }

    /// Issue a POST request to the daemon.
    pub async fn post(&self, path: &str, body: Bytes) -> Result<Bytes, TailscaleError> {
        let resp = self.request("POST", path, Full::new(body)).await?;
        read_response(resp).await
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        body: Full<Bytes>,
    ) -> Result<Response<Incoming>, TailscaleError> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| TailscaleError::SocketError(format!("{}: {e}", self.socket_path.display())))?;

        let io = TokioIo::new(stream);

        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(|e| TailscaleError::SocketError(format!("handshake failed: {e}")))?;

        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::debug!("tailscale local API connection closed: {e}");
            }
        });

        let uri: Uri = path
            .parse()
            .map_err(|e| TailscaleError::Api(format!("invalid path: {e}")))?;

        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("Host", "local-tailscaled.sock")
            .body(body)
            .map_err(|e| TailscaleError::Api(format!("failed to build request: {e}")))?;

        sender
            .send_request(req)
            .await
            .map_err(|e| TailscaleError::SocketError(format!("request failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn read_response(resp: Response<Incoming>) -> Result<Bytes, TailscaleError> {
    let status = resp.status();
    let body = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| TailscaleError::SocketError(format!("failed to read body: {e}")))?
        .to_bytes();

    if !status.is_success() {
        let text = String::from_utf8_lossy(&body);
        return Err(TailscaleError::Api(format!(
            "local API returned {status}: {text}"
        )));
    }

    Ok(body)
}

fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push(char::from(HEX[(b >> 4) as usize]));
                out.push(char::from(HEX[(b & 0x0f) as usize]));
            }
        }
    }
    out
}

const HEX: &[u8; 16] = b"0123456789ABCDEF";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_paths_are_non_empty() {
        let paths = known_socket_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn detect_returns_none_when_no_socket() {
        // In CI/test environments the socket typically doesn't exist.
        // This just ensures detect() doesn't panic.
        let _ = LocalApi::detect();
    }

    #[test]
    fn urlencoded_passthrough() {
        assert_eq!(urlencoded("100.64.0.1"), "100.64.0.1");
    }

    #[test]
    fn urlencoded_special_chars() {
        assert_eq!(urlencoded("100.64.0.1:443"), "100.64.0.1%3A443");
    }

    #[test]
    fn urlencoded_spaces() {
        assert_eq!(urlencoded("hello world"), "hello%20world");
    }

    #[test]
    fn socket_path_accessor() {
        let api = LocalApi::new(PathBuf::from("/tmp/test.sock"));
        assert_eq!(api.socket_path(), Path::new("/tmp/test.sock"));
    }
}
