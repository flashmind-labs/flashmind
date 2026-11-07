//! Optional local session presence. The connection remains open between turns.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

const MAX_LINE: usize = 16 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const RETRY_MIN: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(5);

#[derive(Clone, PartialEq, Serialize)]
struct Registration {
    version: u8,
    agent: &'static str,
    session_id: String,
    cwd: PathBuf,
    pid: u32,
    title: Option<String>,
    started_at: u64,
}

impl Registration {
    fn new(session_id: &str, cwd: &Path) -> Self {
        Self {
            version: 1,
            agent: "flsh",
            session_id: session_id.to_owned(),
            cwd: std::path::absolute(cwd).unwrap_or_else(|_| cwd.to_path_buf()),
            pid: std::process::id(),
            title: None,
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    fn line(&self) -> io::Result<Vec<u8>> {
        if !self.cwd.is_absolute() {
            return Err(invalid("presence directory must be absolute"));
        }
        let mut line = serde_json::to_vec(self)?;
        line.push(b'\n');
        if line.len() > MAX_LINE {
            return Err(invalid("presence registration exceeds the line limit"));
        }
        Ok(line)
    }
}

/// Cloned by the session task while its handle owns the connection lifetime.
#[derive(Clone)]
pub(crate) struct Updates(watch::Sender<Registration>);

impl Updates {
    pub(crate) fn cwd(&self, cwd: &Path) {
        let Ok(cwd) = std::path::absolute(cwd) else {
            return;
        };
        self.0.send_if_modified(|registration| {
            if registration.cwd == cwd {
                return false;
            }
            registration.cwd = cwd;
            true
        });
    }
}

/// Dropping the owner cancels retries and releases its socket on every exit path.
pub(crate) struct Presence {
    updates: Updates,
    task: JoinHandle<()>,
}

impl Presence {
    pub(crate) fn open(session_id: &str, cwd: &Path) -> Self {
        Self::with_locator(session_id, cwd, None)
    }

    pub(crate) fn with_locator(session_id: &str, cwd: &Path, locator: Option<PathBuf>) -> Self {
        Self::start(Registration::new(session_id, cwd), locator)
    }

    fn start(registration: Registration, locator: Option<PathBuf>) -> Self {
        let (sender, receiver) = watch::channel(registration);
        Self {
            updates: Updates(sender),
            task: tokio::spawn(run(receiver, locator)),
        }
    }

    pub(crate) fn updates(&self) -> Updates {
        self.updates.clone()
    }

    pub(crate) fn switch(&self, session_id: &str, cwd: &Path) {
        self.updates
            .0
            .send_replace(Registration::new(session_id, cwd));
    }

    pub(crate) async fn shutdown(mut self) {
        self.task.abort();
        let _ = (&mut self.task).await;
    }
}

impl Drop for Presence {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Deserialize)]
struct Locator {
    version: u8,
    transport: Transport,
    address: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Transport {
    Unix,
    Pipe,
}

#[derive(Deserialize)]
struct Ack {
    version: u8,
    ok: bool,
}

trait IpcStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> IpcStream for T {}
type Connection = BufReader<Box<dyn IpcStream>>;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn locator_path() -> io::Result<PathBuf> {
    resolve_locator(std::env::var_os("BEEP_PRESENCE_FILE"), dirs::config_dir())
}

fn resolve_locator(
    override_path: Option<std::ffi::OsString>,
    config: Option<PathBuf>,
) -> io::Result<PathBuf> {
    if let Some(path) = override_path {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err(invalid("BEEP_PRESENCE_FILE must be absolute"));
        }
        return Ok(path);
    }
    config
        .map(|directory| directory.join("beep-node").join("presence.json"))
        .ok_or_else(|| invalid("cannot locate the presence configuration"))
}

async fn connect(path: Option<&Path>, registration: &Registration) -> io::Result<Connection> {
    let path = match path {
        Some(path) => path.to_path_buf(),
        None => locator_path()?,
    };
    let mut bytes = Vec::new();
    tokio::fs::File::open(path)
        .await?
        .take((MAX_LINE + 1) as u64)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > MAX_LINE {
        return Err(invalid("presence locator exceeds the size limit"));
    }
    let locator: Locator = serde_json::from_slice(&bytes)?;
    if locator.version != 1 {
        return Err(invalid("unsupported presence locator version"));
    }
    let stream: Box<dyn IpcStream> = match locator.transport {
        #[cfg(unix)]
        Transport::Unix if Path::new(&locator.address).is_absolute() => {
            Box::new(tokio::net::UnixStream::connect(&locator.address).await?)
        }
        #[cfg(windows)]
        Transport::Pipe if locator.address.starts_with(r"\\.\pipe\") => {
            Box::new(tokio::net::windows::named_pipe::ClientOptions::new().open(&locator.address)?)
        }
        _ => return Err(invalid("unsupported or nonlocal presence address")),
    };
    let mut connection = BufReader::new(stream);
    connection.write_all(&registration.line()?).await?;
    let ack: Ack = serde_json::from_slice(&read_line(&mut connection).await?)?;
    if ack.version != 1 || !ack.ok {
        return Err(invalid("presence registration rejected"));
    }
    Ok(connection)
}

async fn read_line(reader: &mut (impl AsyncBufRead + Unpin)) -> io::Result<Vec<u8>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let end = available.iter().position(|byte| *byte == b'\n');
        let count = end.map_or(available.len(), |index| index + 1);
        if line.len() + count > MAX_LINE {
            return Err(invalid("presence response exceeds the line limit"));
        }
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        if end.is_some() {
            return Ok(line);
        }
    }
}

async fn connected(
    connection: &mut Connection,
    updates: &mut watch::Receiver<Registration>,
    mut registration: Registration,
) -> io::Result<()> {
    let mut byte = [0];
    loop {
        let next = updates.borrow_and_update().clone();
        if next.session_id != registration.session_id {
            return Ok(());
        }
        if next != registration {
            let line = next.line()?;
            // A change during a blocked write reconnects instead of leaving
            // a partial JSON line on the connection.
            tokio::select! {
                _ = updates.changed() => return Ok(()),
                result = timeout(CONNECT_TIMEOUT, connection.write_all(&line)) => result??,
            }
            registration = next;
        }
        tokio::select! {
            changed = updates.changed() => if changed.is_err() { return Ok(()) },
            result = connection.read(&mut byte) => {
                result?;
                return Err(io::ErrorKind::ConnectionAborted.into());
            }
        }
    }
}

async fn run(mut updates: watch::Receiver<Registration>, locator: Option<PathBuf>) {
    let mut retry = RETRY_MIN;
    'sessions: loop {
        let registration = updates.borrow_and_update().clone();
        let result = {
            let connecting = timeout(CONNECT_TIMEOUT, connect(locator.as_deref(), &registration));
            tokio::pin!(connecting);
            loop {
                tokio::select! {
                    changed = updates.changed() => {
                        if changed.is_err() { return }
                        if updates.borrow().session_id != registration.session_id {
                            retry = RETRY_MIN;
                            continue 'sessions;
                        }
                        // Finish the ACK for this session, then send its latest metadata.
                    }
                    result = &mut connecting => break result,
                }
            }
        };
        let result = match result {
            Ok(Ok(mut connection)) => {
                retry = RETRY_MIN;
                match connected(&mut connection, &mut updates, registration).await {
                    Ok(()) => continue,
                    Err(error) => error,
                }
            }
            Ok(Err(error)) => error,
            Err(error) => error.into(),
        };
        tracing::debug!(%result, "local session presence unavailable");
        tokio::select! {
            changed = updates.changed() => {
                if changed.is_err() { return }
                retry = RETRY_MIN;
            }
            _ = sleep(retry) => retry = (retry * 2).min(RETRY_MAX),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
