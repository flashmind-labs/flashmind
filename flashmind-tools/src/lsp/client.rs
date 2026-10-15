//! One running rust-analyzer process and its JSON-RPC connection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::BufReader;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::AsyncMutex;
use super::protocol::{CONTENT_MODIFIED, RpcError, read_message, write_message};

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>>>;

const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const CONTENT_MODIFIED_RETRIES: usize = 3;

/// Text last sent to the server for one open document.
struct OpenDocument {
    version: i32,
    text: String,
}

/// A connection to one language server process rooted at a workspace.
pub(crate) struct LspClient {
    root: PathBuf,
    child: AsyncMutex<Child>,
    stdin: Arc<AsyncMutex<ChildStdin>>,
    pending: Pending,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    quiescent: watch::Receiver<bool>,
    documents: AsyncMutex<HashMap<PathBuf, OpenDocument>>,
    started: Instant,
    last_used: Mutex<Instant>,
}

impl LspClient {
    /// Spawn the server and complete the `initialize` handshake.
    ///
    /// Indexing continues in the background; use [`wait_ready`](Self::wait_ready)
    /// before trusting results.
    pub(crate) async fn start(command: &Path, root: &Path) -> anyhow::Result<Arc<Self>> {
        let mut child = Command::new(command)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to start {}", command.display()))?;
        let stdin = child.stdin.take().context("server stdin was not piped")?;
        let stdout = child.stdout.take().context("server stdout was not piped")?;

        let stdin = Arc::new(AsyncMutex::new(stdin));
        let pending: Pending = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let (quiescent_tx, quiescent) = watch::channel(false);
        tokio::spawn(read_loop(ReadLoop {
            stdout,
            stdin: stdin.clone(),
            pending: pending.clone(),
            alive: alive.clone(),
            quiescent: quiescent_tx,
        }));

        let now = Instant::now();
        let client = Arc::new(Self {
            root: root.to_path_buf(),
            child: AsyncMutex::new(child),
            stdin,
            pending,
            next_id: AtomicU64::new(1),
            alive,
            quiescent,
            documents: AsyncMutex::default(),
            started: now,
            last_used: Mutex::new(now),
        });

        let root_uri = Url::from_directory_path(root)
            .map_err(|()| anyhow::anyhow!("not an absolute path: {}", root.display()))?;
        let name = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri.as_str(),
            "workspaceFolders": [{"uri": root_uri.as_str(), "name": name}],
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": true},
                    "implementation": {"linkSupport": true},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": true},
                },
                "workspace": {"symbol": {}},
                "experimental": {"serverStatusNotification": true},
            },
            "initializationOptions": {
                // `cargo check` on save would fight the agent's own builds for the target dir.
                "checkOnSave": false,
                // The agent edits files on disk, so let the server watch them.
                "files": {"watcher": "server"},
                "workspace": {"symbol": {"search": {"kind": "all_symbols", "limit": 256}}},
            },
        });
        let never = CancellationToken::new();
        client
            .request_with_timeout("initialize", params, &never, INITIALIZE_TIMEOUT)
            .await
            .context("rust-analyzer did not initialize")?;
        client.notify("initialized", json!({})).await?;
        Ok(client)
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    /// How long the server has been running.
    pub(crate) fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    pub(crate) fn touch(&self) {
        *self
            .last_used
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Instant::now();
    }

    pub(crate) fn idle_for(&self) -> Duration {
        self.last_used
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .elapsed()
    }

    /// Wait until the server reports it has finished indexing.
    ///
    /// Returns `false` on timeout, cancellation, or if the server exits.
    pub(crate) async fn wait_ready(&self, timeout: Duration, cancel: &CancellationToken) -> bool {
        let mut quiescent = self.quiescent.clone();
        tokio::select! {
            ready = quiescent.wait_for(|ready| *ready) => ready.is_ok(),
            () = tokio::time::sleep(timeout) => false,
            () = cancel.cancelled() => false,
        }
    }

    /// Send a request, retrying when the server reports the content changed under it.
    pub(crate) async fn request(
        &self,
        method: &str,
        params: Value,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Value> {
        let mut attempt = 0;
        loop {
            match self
                .request_with_timeout(method, params.clone(), cancel, REQUEST_TIMEOUT)
                .await
            {
                Err(error)
                    if attempt < CONTENT_MODIFIED_RETRIES
                        && error
                            .downcast_ref::<RpcError>()
                            .is_some_and(|e| e.code == CONTENT_MODIFIED) =>
                {
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    async fn request_with_timeout(
        &self,
        method: &str,
        params: Value,
        cancel: &CancellationToken,
        timeout: Duration,
    ) -> anyhow::Result<Value> {
        if !self.is_alive() {
            bail!("rust-analyzer has exited");
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, tx);
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if let Err(error) = self.send(&message).await {
            self.forget(id);
            return Err(error);
        }

        tokio::select! {
            response = rx => match response {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(error)) => Err(anyhow::Error::new(error)),
                Err(_) => bail!("rust-analyzer exited during {method}"),
            },
            () = cancel.cancelled() => {
                self.cancel(id).await;
                bail!("{method} was cancelled")
            }
            () = tokio::time::sleep(timeout) => {
                self.cancel(id).await;
                bail!("{method} timed out after {}s", timeout.as_secs())
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> anyhow::Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    async fn send(&self, message: &Value) -> anyhow::Result<()> {
        let mut stdin = self.stdin.lock().await;
        write_message(&mut *stdin, message)
            .await
            .context("failed to write to rust-analyzer")
    }

    fn forget(&self, id: u64) {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id);
    }

    async fn cancel(&self, id: u64) {
        self.forget(id);
        // Best effort: the server may already have answered or exited.
        let _ = self.notify("$/cancelRequest", json!({"id": id})).await;
    }

    /// Make the server's view of a Rust file match the disk.
    ///
    /// Opens the file on first use and sends the full text again when it has
    /// changed since. Non-Rust files are left to the server's file watcher.
    pub(crate) async fn sync_document(&self, path: &Path) -> anyhow::Result<()> {
        if path.extension().is_none_or(|ext| ext != "rs") {
            return Ok(());
        }
        let text = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("failed to read {}", path.display()))?;
        let uri = file_uri(path)?;

        let mut documents = self.documents.lock().await;
        match documents.get_mut(path) {
            Some(open) if open.text == text => Ok(()),
            Some(open) => {
                open.version += 1;
                let params = json!({
                    "textDocument": {"uri": uri.as_str(), "version": open.version},
                    "contentChanges": [{"text": text}],
                });
                open.text = text;
                self.notify("textDocument/didChange", params).await
            }
            None => {
                let params = json!({
                    "textDocument": {
                        "uri": uri.as_str(),
                        "languageId": "rust",
                        "version": 1,
                        "text": text,
                    },
                });
                documents.insert(path.to_path_buf(), OpenDocument { version: 1, text });
                self.notify("textDocument/didOpen", params).await
            }
        }
    }

    /// Ask the server to exit, then kill it if it has not.
    pub(crate) async fn shutdown(&self) {
        if self.is_alive() {
            let never = CancellationToken::new();
            let _ = self
                .request_with_timeout("shutdown", Value::Null, &never, SHUTDOWN_TIMEOUT)
                .await;
            let _ = self.notify("exit", Value::Null).await;
        }
        let mut child = self.child.lock().await;
        if tokio::time::timeout(SHUTDOWN_TIMEOUT, child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }
}

/// Convert an absolute path to a `file://` URI.
pub(crate) fn file_uri(path: &Path) -> anyhow::Result<Url> {
    Url::from_file_path(path)
        .map_err(|()| anyhow::anyhow!("not an absolute path: {}", path.display()))
}

#[derive(Deserialize)]
struct ServerStatus {
    quiescent: bool,
}

struct ReadLoop {
    stdout: ChildStdout,
    stdin: Arc<AsyncMutex<ChildStdin>>,
    pending: Pending,
    alive: Arc<AtomicBool>,
    quiescent: watch::Sender<bool>,
}

/// Route server messages until the process closes stdout.
async fn read_loop(state: ReadLoop) {
    let mut reader = BufReader::new(state.stdout);
    loop {
        let message = match read_message(&mut reader).await {
            Ok(Some(message)) => message,
            Ok(None) => break,
            Err(error) => {
                tracing::warn!(%error, "rust-analyzer sent an unreadable message");
                break;
            }
        };
        let id = message.get("id").cloned();
        let method = message.get("method").and_then(Value::as_str);
        match (id, method) {
            // A request from the server. The client advertises almost nothing, so
            // a null result is a valid answer to everything it can ask.
            (Some(id), Some(_)) => {
                let reply = json!({"jsonrpc": "2.0", "id": id, "result": null});
                let mut stdin = state.stdin.lock().await;
                if write_message(&mut *stdin, &reply).await.is_err() {
                    break;
                }
            }
            (Some(id), None) => {
                let Some(id) = id.as_u64() else { continue };
                let sender = state
                    .pending
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&id);
                let Some(sender) = sender else { continue };
                let result = match message.get("error") {
                    Some(error) => Err(RpcError::deserialize(error).unwrap_or(RpcError {
                        code: 0,
                        message: error.to_string(),
                    })),
                    None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                };
                let _ = sender.send(result);
            }
            (None, Some("experimental/serverStatus")) => {
                if let Some(status) = message
                    .get("params")
                    .and_then(|params| ServerStatus::deserialize(params).ok())
                {
                    state.quiescent.send_replace(status.quiescent);
                }
            }
            _ => {}
        }
    }
    state.alive.store(false, Ordering::Release);
    // Dropping the senders wakes every waiting request with an error.
    state
        .pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}
