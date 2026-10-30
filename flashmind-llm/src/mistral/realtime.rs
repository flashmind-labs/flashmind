//! Realtime transcription over Mistral's WebSocket API.
//!
//! The session lives at `wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=...`.
//! Mistral sends `session.created`, the client sets the audio format with
//! `session.update`, then streams `input_audio.append` messages (base64 PCM)
//! and closes the input with `input_audio.flush` and `input_audio.end`.
//! Mistral answers with `transcription.text.delta`, `transcription.language`
//! and finally `transcription.done`, or with `error`.

use std::error::Error as StdError;
use std::fmt;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use futures_util::{SinkExt, Stream};
use reqwest::header::{AUTHORIZATION, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_rustls::TlsConnector;
use url::Url;
use yawc::{Frame, HttpRequestBuilder, OpCode, Options, TcpWebSocket, WebSocket};

use crate::http::websocket_tls_config;

/// Sample rate of the audio the caller pushes.
pub const REALTIME_SAMPLE_RATE: u32 = 16_000;
const ENCODING: &str = "pcm_s16le";
/// Mistral's limit on decoded audio per `input_audio.append`.
const MAX_APPEND_BYTES: usize = 262_144;
/// Audio chunks queued between the caller and the socket.
const AUDIO_QUEUE: usize = 64;
/// Events queued between the socket and the caller.
const EVENT_QUEUE: usize = 256;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Settings for a realtime transcription session.
#[derive(Debug, Clone)]
pub struct RealtimeTranscriptionRequest {
    /// Realtime model, such as `voxtral-mini-transcribe-realtime-2602`.
    pub model: String,
    /// ISO 639-1 language of the audio. Mistral detects it when unset.
    pub language: Option<String>,
    /// How long the model waits for context before transcribing. Higher is
    /// more accurate and slower. Mistral picks a default when unset.
    pub target_streaming_delay_ms: Option<u32>,
}

impl RealtimeTranscriptionRequest {
    /// A session for `model` with Mistral's defaults.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            language: None,
            target_streaming_delay_ms: None,
        }
    }

    /// Pin the audio language (ISO 639-1, e.g. `"en"`).
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the streaming delay in milliseconds.
    pub fn with_target_streaming_delay_ms(mut self, delay_ms: u32) -> Self {
        self.target_streaming_delay_ms = Some(delay_ms);
        self
    }
}

/// An event from a realtime transcription session.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TranscriptionEvent {
    /// More transcript text. Append it to what came before.
    TextDelta(String),
    /// The language Mistral detected in the audio.
    Language(String),
    /// The session finished. Carries the whole transcript. The stream ends
    /// after this event.
    Done(TranscriptionDone),
}

/// The final transcript of a realtime session.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionDone {
    /// Full transcript text.
    pub text: String,
    /// Detected or requested language, when Mistral reports one.
    pub language: Option<String>,
}

/// An `error` message from Mistral's realtime API. Event stream errors can be
/// downcast to it.
#[derive(Debug, Clone)]
pub struct RealtimeTranscriptionError {
    /// Mistral's internal error code.
    pub code: Option<i64>,
    /// Human-readable message.
    pub message: String,
}

impl fmt::Display for RealtimeTranscriptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(
                f,
                "Mistral realtime transcription error {code}: {}",
                self.message
            ),
            None => write!(f, "Mistral realtime transcription error: {}", self.message),
        }
    }
}

impl StdError for RealtimeTranscriptionError {}

/// An open realtime transcription session.
///
/// [`split`](Self::split) it into the [`AudioSink`] that takes audio and the
/// [`TranscriptionEvents`] stream. Dropping the event stream ends the session.
pub struct RealtimeTranscription {
    request_id: String,
    audio: AudioSink,
    events: TranscriptionEvents,
}

impl RealtimeTranscription {
    /// Mistral's request ID for this session, for support and logs.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Separate the audio input from the event stream.
    pub fn split(self) -> (AudioSink, TranscriptionEvents) {
        (self.audio, self.events)
    }
}

enum AudioCommand {
    Append(Bytes),
    Flush,
    End,
}

/// Takes 16 kHz mono `pcm_s16le` audio for a realtime session.
///
/// The queue is bounded: [`send`](Self::send) waits while the socket is
/// behind. Dropping the sink ends the input like [`finish`](Self::finish).
pub struct AudioSink {
    commands: mpsc::Sender<AudioCommand>,
}

impl AudioSink {
    /// Queue a chunk of audio. Fails once the session has ended.
    pub async fn send(&self, pcm: Bytes) -> anyhow::Result<()> {
        self.command(AudioCommand::Append(pcm)).await
    }

    /// Ask Mistral to transcribe the audio sent so far without waiting for
    /// more context, for example when the speaker pauses.
    pub async fn flush(&self) -> anyhow::Result<()> {
        self.command(AudioCommand::Flush).await
    }

    /// End the input. Mistral then sends the rest of the transcript and
    /// [`TranscriptionEvent::Done`].
    pub async fn finish(self) -> anyhow::Result<()> {
        self.command(AudioCommand::End).await
    }

    async fn command(&self, command: AudioCommand) -> anyhow::Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| anyhow!("the realtime transcription session has ended"))
    }
}

/// Events of a realtime session. Ends after [`TranscriptionEvent::Done`] or
/// the first error. Dropping it stops the session's background task.
pub struct TranscriptionEvents {
    events: mpsc::Receiver<anyhow::Result<TranscriptionEvent>>,
    task: JoinHandle<()>,
}

impl Stream for TranscriptionEvents {
    type Item = anyhow::Result<TranscriptionEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}

impl Drop for TranscriptionEvents {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ---------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(tag = "type")]
enum ClientMessage<'a> {
    #[serde(rename = "session.update")]
    SessionUpdate { session: SessionUpdate<'a> },
    #[serde(rename = "input_audio.append")]
    AudioAppend { audio: String },
    #[serde(rename = "input_audio.flush")]
    AudioFlush,
    #[serde(rename = "input_audio.end")]
    AudioEnd,
}

#[derive(Serialize)]
struct SessionUpdate<'a> {
    audio_format: AudioFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_streaming_delay_ms: Option<u32>,
}

#[derive(Serialize)]
struct AudioFormat {
    encoding: &'static str,
    sample_rate: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "session.created")]
    SessionCreated { session: SessionInfo },
    #[serde(rename = "error")]
    Error { error: ServerError },
    #[serde(rename = "transcription.language")]
    Language { audio_language: String },
    #[serde(rename = "transcription.text.delta")]
    TextDelta { text: String },
    #[serde(rename = "transcription.done")]
    Done {
        text: String,
        #[serde(default)]
        language: Option<String>,
    },
    /// `session.updated`, `transcription.segment`, turn events and anything newer.
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct SessionInfo {
    request_id: String,
}

#[derive(Deserialize)]
struct ServerError {
    /// A string, or an object with a `detail` string.
    message: Value,
    #[serde(default)]
    code: Option<i64>,
}

impl From<ServerError> for RealtimeTranscriptionError {
    fn from(error: ServerError) -> Self {
        let message = match error.message {
            Value::String(message) => message,
            Value::Object(object) => match object.get("detail") {
                Some(Value::String(detail)) => detail.clone(),
                _ => Value::Object(object).to_string(),
            },
            other => other.to_string(),
        };
        Self {
            code: error.code,
            message,
        }
    }
}

/// A decoded server message.
#[derive(Debug, PartialEq)]
enum Incoming {
    SessionCreated { request_id: String },
    Event(TranscriptionEvent),
    Ignored,
}

fn decode(text: &str) -> Result<Incoming, RealtimeTranscriptionError> {
    let message = match serde_json::from_str::<ServerMessage>(text) {
        Ok(message) => message,
        Err(error) => {
            tracing::warn!(%error, bytes = text.len(), "Unreadable Mistral realtime message");
            return Ok(Incoming::Ignored);
        }
    };
    Ok(match message {
        ServerMessage::SessionCreated { session } => Incoming::SessionCreated {
            request_id: session.request_id,
        },
        ServerMessage::Error { error } => return Err(error.into()),
        ServerMessage::Language { audio_language } => {
            Incoming::Event(TranscriptionEvent::Language(audio_language))
        }
        ServerMessage::TextDelta { text } => Incoming::Event(TranscriptionEvent::TextDelta(text)),
        ServerMessage::Done { text, language } => {
            Incoming::Event(TranscriptionEvent::Done(TranscriptionDone {
                text,
                language,
            }))
        }
        ServerMessage::Other => Incoming::Ignored,
    })
}

fn encode(message: &ClientMessage<'_>) -> String {
    serde_json::to_string(message)
        .expect("client messages hold only strings and integers, which always serialize")
}

fn session_update(request: &RealtimeTranscriptionRequest) -> String {
    encode(&ClientMessage::SessionUpdate {
        session: SessionUpdate {
            audio_format: AudioFormat {
                encoding: ENCODING,
                sample_rate: REALTIME_SAMPLE_RATE,
            },
            language: request.language.as_deref(),
            target_streaming_delay_ms: request.target_streaming_delay_ms,
        },
    })
}

/// `input_audio.append` messages for `pcm`, split to Mistral's size limit.
fn audio_appends(pcm: &[u8]) -> impl Iterator<Item = String> + '_ {
    pcm.chunks(MAX_APPEND_BYTES).map(|chunk| {
        encode(&ClientMessage::AudioAppend {
            audio: STANDARD.encode(chunk),
        })
    })
}

fn realtime_url(base_url: &Url, model: &str) -> anyhow::Result<Url> {
    let mut url = base_url.clone();
    let scheme = match url.scheme() {
        "https" | "wss" => "wss",
        "http" | "ws" => "ws",
        other => bail!("unsupported Mistral URL scheme '{other}'"),
    };
    url.set_scheme(scheme)
        .map_err(|()| anyhow!("cannot use '{scheme}' with the Mistral URL"))?;
    url.set_path("v1/audio/transcriptions/realtime");
    url.query_pairs_mut().clear().append_pair("model", model);
    Ok(url)
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub(super) async fn connect(
    base_url: &Url,
    api_key: &SecretString,
    request: RealtimeTranscriptionRequest,
) -> anyhow::Result<RealtimeTranscription> {
    let url = realtime_url(base_url, &request.model)?;
    let mut auth = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
        .context("the Mistral API key is not a valid header value")?;
    auth.set_sensitive(true);

    let connect = WebSocket::connect(url)
        .with_connector(TlsConnector::from(websocket_tls_config()))
        .with_options(socket_options())
        .with_request(HttpRequestBuilder::new().header(AUTHORIZATION, auth));
    let mut socket = tokio::time::timeout(HANDSHAKE_TIMEOUT, connect)
        .await
        .context("timed out connecting to Mistral realtime transcription")?
        .context("connecting to Mistral realtime transcription")?;

    let request_id = tokio::time::timeout(HANDSHAKE_TIMEOUT, wait_for_session(&mut socket))
        .await
        .context("timed out waiting for the Mistral realtime session")??;
    socket
        .send(Frame::text(session_update(&request)))
        .await
        .context("configuring the Mistral realtime session")?;
    tracing::debug!(request_id = %request_id, model = %request.model, "Mistral realtime session open");

    let (commands_tx, commands_rx) = mpsc::channel(AUDIO_QUEUE);
    let (events_tx, events_rx) = mpsc::channel(EVENT_QUEUE);
    let task = tokio::spawn(run_session(socket, commands_rx, events_tx));
    Ok(RealtimeTranscription {
        request_id,
        audio: AudioSink {
            commands: commands_tx,
        },
        events: TranscriptionEvents {
            events: events_rx,
            task,
        },
    })
}

/// Socket settings: fast permessage-deflate, and no Nagle delay on audio.
fn socket_options() -> Options {
    Options::default()
        .with_low_latency_compression()
        .with_no_delay()
}

/// Read until `session.created` and return its request ID.
async fn wait_for_session(socket: &mut TcpWebSocket) -> anyhow::Result<String> {
    loop {
        let frame = socket
            .next_frame()
            .await
            .context("reading the Mistral realtime handshake")?;
        match frame.opcode() {
            OpCode::Text => {
                if let Incoming::SessionCreated { request_id } = decode(frame.as_str())? {
                    return Ok(request_id);
                }
            }
            OpCode::Close => bail!(
                "Mistral closed the realtime session: {}",
                describe_close(&frame)
            ),
            _ => {}
        }
    }
}

fn describe_close(frame: &Frame) -> String {
    let code = frame
        .close_code()
        .map_or_else(|| "no code".to_owned(), |code| format!("{code:?}"));
    match frame.close_reason() {
        Ok(Some(reason)) if !reason.is_empty() => format!("{code}, {reason}"),
        _ => code,
    }
}

/// Pump audio to the socket and events to the caller until the session ends,
/// fails, or the caller drops the event stream.
async fn run_session(
    mut socket: TcpWebSocket,
    mut commands: mpsc::Receiver<AudioCommand>,
    events: mpsc::Sender<anyhow::Result<TranscriptionEvent>>,
) {
    let mut input_open = true;
    let outcome: anyhow::Result<()> = loop {
        tokio::select! {
            () = events.closed() => break Ok(()),
            command = commands.recv(), if input_open => {
                let messages: Vec<String> = match command {
                    Some(AudioCommand::Append(pcm)) => audio_appends(&pcm).collect(),
                    Some(AudioCommand::Flush) => vec![encode(&ClientMessage::AudioFlush)],
                    Some(AudioCommand::End) | None => {
                        input_open = false;
                        vec![
                            encode(&ClientMessage::AudioFlush),
                            encode(&ClientMessage::AudioEnd),
                        ]
                    }
                };
                if let Err(error) = send_all(&mut socket, messages).await {
                    break Err(error);
                }
            }
            frame = socket.next_frame() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(error) if error.is_closed() => {
                        break Err(anyhow!("the Mistral realtime connection closed before the session finished"));
                    }
                    Err(error) => {
                        break Err(anyhow::Error::new(error).context("reading from Mistral realtime transcription"));
                    }
                };
                match frame.opcode() {
                    OpCode::Text => {}
                    OpCode::Close => {
                        break Err(anyhow!(
                            "Mistral closed the realtime session before it finished: {}",
                            describe_close(&frame)
                        ));
                    }
                    _ => continue,
                }
                match decode(frame.as_str()) {
                    Ok(Incoming::Event(event)) => {
                        let done = matches!(event, TranscriptionEvent::Done(_));
                        if events.send(Ok(event)).await.is_err() || done {
                            break Ok(());
                        }
                    }
                    Ok(Incoming::SessionCreated { .. } | Incoming::Ignored) => {}
                    Err(error) => break Err(error.into()),
                }
            }
        }
    };

    if let Err(error) = outcome {
        tracing::debug!(
            error = format!("{error:#}"),
            "Mistral realtime session failed"
        );
        // The caller may already have dropped the stream; then nobody needs the error.
        let _ = events.send(Err(error)).await;
    }
    // A clean close is a courtesy to the server. The connection is dropped
    // either way, so a failed or slow close changes nothing for the caller.
    let _ = tokio::time::timeout(CLOSE_TIMEOUT, socket.close()).await;
}

async fn send_all(socket: &mut TcpWebSocket, messages: Vec<String>) -> anyhow::Result<()> {
    for message in messages {
        socket
            .send(Frame::text(message))
            .await
            .context("sending audio to Mistral realtime transcription")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Mutex;

    use futures_util::StreamExt;

    use super::*;
    use hyper::Request;
    use hyper::body::Incoming as IncomingBody;
    use hyper::header::SEC_WEBSOCKET_EXTENSIONS;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use yawc::UpgradeFut;

    use crate::MistralProvider;

    fn json(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn session_update_sets_pcm_format_and_options() {
        let request = RealtimeTranscriptionRequest::new("voxtral-mini-transcribe-realtime-2602")
            .with_language("fr")
            .with_target_streaming_delay_ms(480);
        assert_eq!(
            json(&session_update(&request)),
            json!({
                "type": "session.update",
                "session": {
                    "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                    "language": "fr",
                    "target_streaming_delay_ms": 480,
                },
            })
        );
        let bare = RealtimeTranscriptionRequest::new("voxtral-mini-transcribe-realtime-2602");
        assert_eq!(
            json(&session_update(&bare))["session"],
            json!({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        );
    }

    #[test]
    fn audio_is_base64_and_split_at_the_size_limit() {
        let pcm = vec![7u8; MAX_APPEND_BYTES + 10];
        let messages: Vec<Value> = audio_appends(&pcm).map(|m| json(&m)).collect();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], "input_audio.append");
        let first = STANDARD
            .decode(messages[0]["audio"].as_str().unwrap())
            .unwrap();
        let second = STANDARD
            .decode(messages[1]["audio"].as_str().unwrap())
            .unwrap();
        assert_eq!(first.len(), MAX_APPEND_BYTES);
        assert_eq!(second, vec![7u8; 10]);
        assert_eq!(audio_appends(&[]).count(), 0);
    }

    #[test]
    fn control_messages_carry_only_their_type() {
        assert_eq!(
            json(&encode(&ClientMessage::AudioFlush)),
            json!({"type": "input_audio.flush"})
        );
        assert_eq!(
            json(&encode(&ClientMessage::AudioEnd)),
            json!({"type": "input_audio.end"})
        );
    }

    #[test]
    fn server_messages_decode() {
        assert_eq!(
            decode(r#"{"type":"session.created","session":{"request_id":"req-1","model":"voxtral-mini-transcribe-realtime-2602","audio_format":{"encoding":"pcm_s16le","sample_rate":16000},"target_streaming_delay_ms":null}}"#).unwrap(),
            Incoming::SessionCreated { request_id: "req-1".into() }
        );
        assert_eq!(
            decode(r#"{"type":"transcription.text.delta","text":" Hello"}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::TextDelta(" Hello".into()))
        );
        assert_eq!(
            decode(r#"{"type":"transcription.language","audio_language":"en"}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::Language("en".into()))
        );
        assert_eq!(
            decode(r#"{"type":"transcription.done","model":"voxtral-mini-transcribe-realtime-2602","text":"Hello world.","usage":{"prompt_audio_seconds":2},"language":null,"segments":[]}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::Done(TranscriptionDone {
                text: "Hello world.".into(),
                language: None,
            }))
        );
        for ignored in [
            r#"{"type":"session.updated","session":{"request_id":"req-1"}}"#,
            r#"{"type":"transcription.segment","text":"Hi","start":0.0,"end":0.4}"#,
            r#"{"type":"turn.end"}"#,
            "not json",
        ] {
            assert_eq!(decode(ignored).unwrap(), Incoming::Ignored);
        }
    }

    #[test]
    fn server_errors_keep_code_and_message() {
        let error =
            decode(r#"{"type":"error","error":{"message":"Invalid audio format","code":3001}}"#)
                .unwrap_err();
        assert_eq!(error.code, Some(3001));
        assert_eq!(error.message, "Invalid audio format");
        let error =
            decode(r#"{"type":"error","error":{"message":{"detail":"Model not found"},"code":1}}"#)
                .unwrap_err();
        assert_eq!(error.message, "Model not found");
    }

    #[test]
    fn realtime_url_switches_to_websocket_scheme() {
        let url = realtime_url(
            &Url::parse("https://api.mistral.ai/").unwrap(),
            "voxtral-mini-transcribe-realtime-2602",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=voxtral-mini-transcribe-realtime-2602"
        );
        let local = realtime_url(&Url::parse("http://127.0.0.1:9/").unwrap(), "m").unwrap();
        assert_eq!(local.scheme(), "ws");
    }

    /// What the fake server saw from the client.
    struct Seen {
        authorization: Option<String>,
        extensions: Option<String>,
        messages: Vec<Value>,
    }

    /// The handshake request's details and the pending upgrade.
    struct Handshake {
        authorization: Option<String>,
        extensions: Option<String>,
        upgrade: UpgradeFut,
    }

    /// A one-connection server that plays Mistral's side of a session.
    async fn fake_mistral(script: Vec<&'static str>) -> (String, JoinHandle<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (handshake_tx, handshake_rx) = oneshot::channel();
            // service_fn takes an Fn, but only one request ever arrives.
            let handshake_tx = Mutex::new(Some(handshake_tx));
            let service = service_fn(move |mut request: Request<IncomingBody>| {
                assert_eq!(request.uri().path(), "/v1/audio/transcriptions/realtime");
                let header = |name| {
                    request
                        .headers()
                        .get(name)
                        .map(|value: &HeaderValue| value.to_str().unwrap().to_string())
                };
                let authorization = header(AUTHORIZATION);
                let extensions = header(SEC_WEBSOCKET_EXTENSIONS);
                let (response, upgrade) =
                    WebSocket::upgrade_with_options(&mut request, socket_options()).unwrap();
                let handshake = Handshake {
                    authorization,
                    extensions,
                    upgrade,
                };
                let sender = handshake_tx.lock().unwrap().take();
                assert!(sender.unwrap().send(handshake).is_ok());
                async move { Ok::<_, Infallible>(response) }
            });
            tokio::spawn(
                http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .with_upgrades(),
            );
            let handshake = handshake_rx.await.unwrap();
            let mut socket = handshake.upgrade.await.unwrap();
            socket
                .send(Frame::text(
                    r#"{"type":"session.created","session":{"request_id":"req-42","model":"m","audio_format":{"encoding":"pcm_s16le","sample_rate":16000}}}"#,
                ))
                .await
                .unwrap();
            let mut messages = Vec::new();
            // Read until the client ends its input.
            while let Ok(frame) = socket.next_frame().await {
                if frame.opcode() == OpCode::Text {
                    let message = json(frame.as_str());
                    let end = message["type"] == "input_audio.end";
                    messages.push(message);
                    if end {
                        break;
                    }
                }
            }
            for line in script {
                socket.send(Frame::text(line)).await.unwrap();
            }
            // Wait for the client's close.
            while let Ok(frame) = socket.next_frame().await {
                if frame.opcode() == OpCode::Close {
                    break;
                }
            }
            Seen {
                authorization: handshake.authorization,
                extensions: handshake.extensions,
                messages,
            }
        });
        (url, server)
    }

    #[tokio::test]
    async fn session_streams_audio_and_events() {
        let (url, server) = fake_mistral(vec![
            r#"{"type":"transcription.language","audio_language":"en"}"#,
            r#"{"type":"transcription.text.delta","text":"Hello"}"#,
            r#"{"type":"transcription.text.delta","text":" world."}"#,
            r#"{"type":"transcription.done","model":"m","text":"Hello world.","usage":{},"language":"en"}"#,
        ])
        .await;
        let provider = MistralProvider::new("test-key")
            .with_base_url(&url)
            .unwrap();
        let session = provider
            .realtime_transcription(RealtimeTranscriptionRequest::new("m").with_language("en"))
            .await
            .unwrap();
        assert_eq!(session.request_id(), "req-42");
        let (audio, events) = session.split();
        audio.send(Bytes::from_static(&[1, 0, 2, 0])).await.unwrap();
        audio.finish().await.unwrap();

        let events: Vec<TranscriptionEvent> = events.map(|event| event.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![
                TranscriptionEvent::Language("en".into()),
                TranscriptionEvent::TextDelta("Hello".into()),
                TranscriptionEvent::TextDelta(" world.".into()),
                TranscriptionEvent::Done(TranscriptionDone {
                    text: "Hello world.".into(),
                    language: Some("en".into()),
                }),
            ]
        );

        let seen = server.await.unwrap();
        assert_eq!(seen.authorization.as_deref(), Some("Bearer test-key"));
        assert!(
            seen.extensions
                .as_deref()
                .is_some_and(|value| value.contains("permessage-deflate")),
            "client should offer compression, got {:?}",
            seen.extensions
        );
        let types: Vec<&str> = seen
            .messages
            .iter()
            .map(|message| message["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            types,
            [
                "session.update",
                "input_audio.append",
                "input_audio.flush",
                "input_audio.end"
            ]
        );
        assert_eq!(seen.messages[0]["session"]["language"], "en");
        assert_eq!(seen.messages[1]["audio"], STANDARD.encode([1u8, 0, 2, 0]));
    }

    #[tokio::test]
    async fn server_error_ends_the_stream_with_an_error() {
        let (url, _server) = fake_mistral(vec![
            r#"{"type":"transcription.text.delta","text":"Hel"}"#,
            r#"{"type":"error","error":{"message":"Audio too long","code":3002}}"#,
        ])
        .await;
        let provider = MistralProvider::new("test-key")
            .with_base_url(&url)
            .unwrap();
        let (audio, events) = provider
            .realtime_transcription(RealtimeTranscriptionRequest::new("m"))
            .await
            .unwrap()
            .split();
        // Dropping the sink ends the input, as finish() does.
        drop(audio);
        let events: Vec<_> = events.collect().await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Ok(TranscriptionEvent::TextDelta(text)) if text == "Hel"));
        let error = events[1].as_ref().unwrap_err();
        let error = error.downcast_ref::<RealtimeTranscriptionError>().unwrap();
        assert_eq!(error.code, Some(3002));
    }
}
