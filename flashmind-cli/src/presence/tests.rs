use super::*;

#[test]
fn discovery_uses_platform_config_or_an_absolute_override() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config");
    let override_path = directory.path().join("presence.json");
    assert_eq!(
        resolve_locator(None, Some(config.clone())).unwrap(),
        config.join("beep-node/presence.json")
    );
    assert_eq!(
        resolve_locator(Some(override_path.clone().into_os_string()), None).unwrap(),
        override_path
    );
    assert!(resolve_locator(Some("relative.json".into()), Some(config)).is_err());
    assert!(resolve_locator(None, None).is_err());
}

#[test]
fn registration_is_bounded_and_contains_the_session_identity() {
    let directory = tempfile::tempdir().unwrap();
    let mut registration = Registration::new("session_example", directory.path());
    let line = registration.line().unwrap();
    assert_eq!(line.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&line).unwrap();
    assert_eq!(value["version"], 1);
    assert_eq!(value["agent"], "flsh");
    assert_eq!(value["session_id"], "session_example");
    assert_eq!(value["cwd"], directory.path().to_str().unwrap());
    assert_eq!(value["pid"], std::process::id());
    assert!(value["title"].is_null());
    assert!(value["started_at"].as_u64().unwrap() > 0);
    registration.session_id = "x".repeat(MAX_LINE);
    assert!(registration.line().is_err());
}

#[tokio::test]
async fn response_lines_are_bounded_and_require_a_newline() {
    let mut valid = BufReader::new(&b"{\"version\":1,\"ok\":true}\n"[..]);
    assert!(read_line(&mut valid).await.is_ok());
    let bytes = vec![b'x'; MAX_LINE + 1];
    let mut oversized = BufReader::new(bytes.as_slice());
    assert_eq!(
        read_line(&mut oversized).await.unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    let mut truncated = BufReader::new(&b"{\"version\":1"[..]);
    assert_eq!(
        read_line(&mut truncated).await.unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
}

#[cfg(unix)]
pub(crate) struct Receiver {
    directory: tempfile::TempDir,
    listener: tokio::net::UnixListener,
}

#[cfg(unix)]
impl Receiver {
    pub(crate) fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("receiver.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let receiver = Self {
            directory,
            listener,
        };
        receiver.advertise(&socket);
        receiver
    }

    pub(crate) fn locator(&self) -> PathBuf {
        self.directory.path().join("presence.json")
    }

    fn advertise(&self, address: &Path) {
        std::fs::write(
            self.locator(),
            serde_json::to_vec(&serde_json::json!({
                "version": 1, "transport": "unix", "address": address
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn restart(&mut self) {
        let socket = self.directory.path().join("restarted.sock");
        self.listener = tokio::net::UnixListener::bind(&socket).unwrap();
        self.advertise(&socket);
    }

    async fn unacked(&self) -> (BufReader<tokio::net::UnixStream>, serde_json::Value) {
        let (stream, _) = timeout(Duration::from_secs(5), self.listener.accept())
            .await
            .unwrap()
            .unwrap();
        let mut stream = BufReader::new(stream);
        let registration = next(&mut stream).await;
        (stream, registration)
    }

    pub(crate) async fn accept(&self) -> (BufReader<tokio::net::UnixStream>, serde_json::Value) {
        let (mut stream, registration) = self.unacked().await;
        stream
            .write_all(b"{\"version\":1,\"ok\":true}\n")
            .await
            .unwrap();
        (stream, registration)
    }

    fn open(&self, session: &str) -> Presence {
        Presence::with_locator(session, self.directory.path(), Some(self.locator()))
    }
}

#[cfg(unix)]
async fn next(stream: &mut BufReader<tokio::net::UnixStream>) -> serde_json::Value {
    let line = timeout(Duration::from_secs(5), read_line(stream))
        .await
        .unwrap()
        .unwrap();
    serde_json::from_slice(&line).unwrap()
}

#[cfg(unix)]
pub(crate) async fn eof(stream: &mut BufReader<tokio::net::UnixStream>) {
    let result = timeout(Duration::from_secs(5), stream.read(&mut [0]))
        .await
        .unwrap();
    // Unix can report a reset when the peer drops with an unread ACK.
    match result {
        Ok(0) => {}
        Err(error) if error.kind() == io::ErrorKind::ConnectionReset => {}
        other => panic!("expected a closed connection, got {other:?}"),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn idle_session_updates_metadata_on_its_connection_and_shutdown_sends_eof() {
    let receiver = Receiver::new();
    let presence = receiver.open("session_example");
    let (mut connection, initial) = receiver.accept().await;
    let latest = receiver.directory.path().join("other-folder");
    presence.updates().cwd(&latest);
    let updated = next(&mut connection).await;
    assert_eq!(updated["cwd"], latest.to_str().unwrap());
    assert_eq!(updated["session_id"], initial["session_id"]);
    assert_eq!(updated["started_at"], initial["started_at"]);
    presence.shutdown().await;
    eof(&mut connection).await;
}

#[cfg(unix)]
#[tokio::test]
async fn switch_closes_the_old_connection_and_registers_the_actual_session() {
    let receiver = Receiver::new();
    let presence = receiver.open("session_first");
    let (mut first, _) = receiver.accept().await;
    let latest = receiver.directory.path().join("other-folder");
    presence.switch("session_second", &latest);
    eof(&mut first).await;
    let (mut second, registration) = receiver.accept().await;
    assert_eq!(registration["session_id"], "session_second");
    assert_eq!(registration["cwd"], latest.to_str().unwrap());
    presence.shutdown().await;
    eof(&mut second).await;
}

#[cfg(unix)]
#[tokio::test]
async fn eof_reconnects_rereads_locator_and_replays_latest_metadata() {
    let mut receiver = Receiver::new();
    let presence = receiver.open("session_example");
    let (first, initial) = receiver.accept().await;
    let latest = receiver.directory.path().join("other-folder");
    presence.updates().cwd(&latest);
    receiver.restart();
    drop(first);
    let (mut second, registration) = receiver.accept().await;
    assert_eq!(registration["session_id"], initial["session_id"]);
    assert_eq!(registration["started_at"], initial["started_at"]);
    assert_eq!(registration["cwd"], latest.to_str().unwrap());
    presence.shutdown().await;
    eof(&mut second).await;
}

#[cfg(unix)]
#[tokio::test]
async fn updates_interrupt_an_outstanding_acknowledgment() {
    let receiver = Receiver::new();
    let presence = receiver.open("session_first");
    let (mut first, _) = receiver.unacked().await;
    presence.switch("session_second", receiver.directory.path());
    eof(&mut first).await;
    let (mut second, registration) = receiver.accept().await;
    assert_eq!(registration["session_id"], "session_second");
    presence.shutdown().await;
    eof(&mut second).await;
}

#[cfg(unix)]
#[tokio::test]
async fn updates_interrupt_retry_backoff_after_rejection() {
    let receiver = Receiver::new();
    let presence = receiver.open("session_first");
    let (mut first, _) = receiver.unacked().await;
    first
        .write_all(b"{\"version\":1,\"ok\":false}\n")
        .await
        .unwrap();
    eof(&mut first).await;
    presence.switch("session_second", receiver.directory.path());
    let (mut second, registration) = timeout(Duration::from_millis(500), receiver.accept())
        .await
        .unwrap();
    assert_eq!(registration["session_id"], "session_second");
    presence.shutdown().await;
    eof(&mut second).await;
}

#[cfg(unix)]
#[tokio::test]
async fn drop_closes_the_socket_even_with_an_updater_alive() {
    let receiver = Receiver::new();
    let presence = receiver.open("session_example");
    let updater = presence.updates();
    let (mut stream, _) = receiver.accept().await;
    drop(presence);
    eof(&mut stream).await;
    updater.cwd(receiver.directory.path());
}

#[cfg(unix)]
#[tokio::test]
async fn shutdown_cancels_a_missing_node_and_a_pending_ack() {
    let receiver = Receiver::new();
    let missing = Presence::with_locator(
        "session_missing",
        receiver.directory.path(),
        Some(receiver.directory.path().join("missing.json")),
    );
    let task = missing.task.abort_handle();
    missing.shutdown().await;
    assert!(task.is_finished());
    let pending = receiver.open("session_pending");
    let (mut stream, _) = receiver.unacked().await;
    pending.shutdown().await;
    eof(&mut stream).await;
}
