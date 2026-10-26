//! One language server process and its JSON-RPC connection.

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
use super::protocol::{
    CONTENT_MODIFIED, Diagnostic, PublishedDiagnostics, RpcError, read_message, write_message,
};
use super::servers::{LanguageServerConfig, language_id};

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
    diagnostics: Arc<Mutex<HashMap<String, PublishedDiagnostics>>>,
    diagnostic_updates: watch::Receiver<()>,
    pull_diagnostics: AtomicBool,
    started: Instant,
    last_used: Mutex<Instant>,
}

impl LspClient {
    /// Spawn the server and complete the `initialize` handshake.
    ///
    /// Indexing continues in the background; use [`wait_ready`](Self::wait_ready)
    /// before trusting results.
    pub(crate) async fn start(
        config: &LanguageServerConfig,
        root: &Path,
        rust: bool,
    ) -> anyhow::Result<Arc<Self>> {
        let command = config
            .command
            .first()
            .context("language server command is empty")?;
        let mut child = Command::new(command)
            .args(&config.command[1..])
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to start {}", command))?;
        let stdin = child.stdin.take().context("server stdin was not piped")?;
        let stdout = child.stdout.take().context("server stdout was not piped")?;

        let stdin = Arc::new(AsyncMutex::new(stdin));
        let pending: Pending = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let (quiescent_tx, quiescent) = watch::channel(!rust);
        let diagnostics = Arc::new(Mutex::new(HashMap::new()));
        let (diagnostic_tx, diagnostic_updates) = watch::channel(());
        tokio::spawn(read_loop(ReadLoop {
            stdout,
            root: root.to_path_buf(),
            stdin: stdin.clone(),
            pending: pending.clone(),
            alive: alive.clone(),
            quiescent: quiescent_tx,
            diagnostics: diagnostics.clone(),
            diagnostic_updates: diagnostic_tx,
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
            diagnostics,
            diagnostic_updates,
            pull_diagnostics: AtomicBool::new(false),
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
                    "publishDiagnostics": {"versionSupport": true},
                    "diagnostic": {},
                    "definition": {"linkSupport": true},
                    "implementation": {"linkSupport": true},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": true},
                },
                "workspace": {"symbol": {}},
                "experimental": {"serverStatusNotification": true},
            },
            "initializationOptions": if !config.initialization.is_null() { config.initialization.clone() } else if rust { json!({
                // `cargo check` on save would fight the agent's own builds for the target dir.
                "checkOnSave": false,
                // The agent edits files on disk, so let the server watch them.
                "files": {"watcher": "server"},
                "workspace": {"symbol": {"search": {"kind": "all_symbols", "limit": 256}}},
            }) } else { Value::Null },
        });
        let never = CancellationToken::new();
        let initialized = client
            .request_with_timeout("initialize", params, &never, INITIALIZE_TIMEOUT)
            .await
            .context("language server did not initialize")?;
        client.pull_diagnostics.store(
            initialized
                .pointer("/capabilities/diagnosticProvider")
                .is_some_and(|value| !value.is_null() && value != &Value::Bool(false)),
            Ordering::Release,
        );
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
            bail!("language server has exited");
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
                Err(_) => bail!("language server exited during {method}"),
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
            .context("failed to write to language server")
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

    /// Make the server's view of a file match the disk.
    ///
    /// Opens the file on first use and sends the full text again when it has
    /// changed since.
    pub(crate) async fn sync_document(&self, path: &Path) -> anyhow::Result<()> {
        let text = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("failed to read {}", path.display()))?;
        let uri = file_uri(path)?;

        let mut documents = self.documents.lock().await;
        match documents.get_mut(path) {
            Some(open) if open.text == text => Ok(()),
            Some(open) => {
                self.diagnostics
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(uri.as_str());
                open.version = open
                    .version
                    .checked_add(1)
                    .context("document version overflow")?;
                let params = json!({
                    "textDocument": {"uri": uri.as_str(), "version": open.version},
                    "contentChanges": [{"text": text}],
                });
                self.notify("textDocument/didChange", params).await?;
                open.text = text;
                Ok(())
            }
            None => {
                let params = json!({
                    "textDocument": {
                        "uri": uri.as_str(),
                        "languageId": language_id(path),
                        "version": 1,
                        "text": text,
                    },
                });
                self.notify("textDocument/didOpen", params).await?;
                documents.insert(path.to_path_buf(), OpenDocument { version: 1, text });
                Ok(())
            }
        }
    }

    pub(super) async fn diagnostics(
        &self,
        path: &Path,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Vec<Diagnostic>> {
        let uri = file_uri(path)?;
        let mut updates = self.diagnostic_updates.clone();
        self.sync_document(path).await?;
        if self.pull_diagnostics.load(Ordering::Acquire) {
            let response = self
                .request(
                    "textDocument/diagnostic",
                    json!({"textDocument": {"uri": uri.as_str()}}),
                    cancel,
                )
                .await?;
            return serde_json::from_value(
                response
                    .get("items")
                    .cloned()
                    .context("language server returned no diagnostic items")?,
            )
            .context("invalid diagnostics response");
        }
        let version = self
            .documents
            .lock()
            .await
            .get(path)
            .map(|document| document.version);
        let wait = async {
            loop {
                if let Some(published) = self
                    .diagnostics
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get(uri.as_str())
                    .filter(|published| published.version.is_none() || published.version == version)
                    .cloned()
                {
                    return Ok(published.diagnostics);
                }
                updates
                    .changed()
                    .await
                    .context("language server stopped publishing diagnostics")?;
            }
        };
        tokio::select! {
            result = wait => result,
            () = cancel.cancelled() => bail!("diagnostics cancelled"),
            () = tokio::time::sleep(Duration::from_secs(5)) => bail!("language server has not published diagnostics for this file; run the project's check command"),
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
    root: PathBuf,
    stdin: Arc<AsyncMutex<ChildStdin>>,
    pending: Pending,
    alive: Arc<AtomicBool>,
    quiescent: watch::Sender<bool>,
    diagnostics: Arc<Mutex<HashMap<String, PublishedDiagnostics>>>,
    diagnostic_updates: watch::Sender<()>,
}

/// Route server messages until the process closes stdout.
async fn read_loop(state: ReadLoop) {
    let mut reader = BufReader::new(state.stdout);
    loop {
        let message = match read_message(&mut reader).await {
            Ok(Some(message)) => message,
            Ok(None) => break,
            Err(error) => {
                tracing::warn!(%error, "language server sent an unreadable message");
                break;
            }
        };
        let id = message.get("id").cloned();
        let method = message.get("method").and_then(Value::as_str);
        match (id, method) {
            (Some(id), Some(method)) => {
                let result = match method {
                    "workspace/configuration" => {
                        let items = message.pointer("/params/items").and_then(Value::as_array);
                        Value::Array(
                            items
                                .map(|items| vec![Value::Null; items.len()])
                                .unwrap_or_default(),
                        )
                    }
                    "workspace/workspaceFolders" => match Url::from_directory_path(&state.root) {
                        Ok(uri) => {
                            json!([{"uri": uri.as_str(), "name": state.root.file_name().and_then(|name| name.to_str()).unwrap_or("workspace")}])
                        }
                        Err(()) => Value::Null,
                    },
                    _ => Value::Null,
                };
                let reply = json!({"jsonrpc": "2.0", "id": id, "result": result});
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
            (None, Some("textDocument/publishDiagnostics")) => {
                if let Some(params) = message.get("params") {
                    match PublishedDiagnostics::deserialize(params) {
                        Ok(published) => {
                            state
                                .diagnostics
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .insert(published.uri.clone(), published);
                            state.diagnostic_updates.send_replace(());
                        }
                        Err(error) => {
                            tracing::warn!(%error, "language server sent invalid diagnostics")
                        }
                    }
                }
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
