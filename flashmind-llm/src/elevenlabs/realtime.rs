//! Realtime transcription over ElevenLabs's Scribe WebSocket API.
//!
//! The session connects to `wss://api.elevenlabs.io/v1/speech-to-text/realtime`
//! with `xi-api-key` in the upgrade header and session options in the query
//! string. ElevenLabs opens with `session_started`, the client pushes
//! `input_audio_chunk` messages (base64 audio with `commit: false` or `true`),
//! and ElevenLabs replies with `partial_transcript`, `committed_transcript`,
//! optional `committed_transcript_with_timestamps`, `warning`, or a typed
//! error message.

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
use reqwest::header::HeaderValue;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_rustls::TlsConnector;
use url::Url;
use yawc::close::CloseCode;
use yawc::{Frame, HttpRequestBuilder, OpCode, Options, TcpWebSocket, WebSocket};

use crate::http::websocket_tls_config;

/// Default sample rate (16 kHz) when [`AudioEncoding::Pcm16000`] is used.
pub const REALTIME_SAMPLE_RATE: u32 = 16_000;
const API_KEY_HEADER: &str = "xi-api-key";
/// Upper bound on decoded audio bytes per `input_audio_chunk` frame.
const MAX_CHUNK_BYTES: usize = 262_144;
const AUDIO_QUEUE: usize = 64;
const EVENT_QUEUE: usize = 256;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const FINAL_COMMIT_TIMEOUT: Duration = Duration::from_secs(5);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Audio encoding accepted by ElevenLabs realtime speech-to-text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioEncoding {
    /// 8 kHz signed 16-bit little-endian mono PCM (`pcm_8000`).
    Pcm8000,
    /// 16 kHz signed 16-bit little-endian mono PCM (`pcm_16000`).
    #[default]
    Pcm16000,
    /// 22.05 kHz signed 16-bit little-endian mono PCM (`pcm_22050`).
    Pcm22050,
    /// 24 kHz signed 16-bit little-endian mono PCM (`pcm_24000`).
    Pcm24000,
    /// 44.1 kHz signed 16-bit little-endian mono PCM (`pcm_44100`).
    Pcm44100,
    /// 48 kHz signed 16-bit little-endian mono PCM (`pcm_48000`).
    Pcm48000,
    /// 8 kHz G.711 mu-law (`ulaw_8000`).
    Ulaw8000,
}

impl AudioEncoding {
    /// Wire identifier used in the `audio_format` query parameter.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pcm8000 => "pcm_8000",
            Self::Pcm16000 => "pcm_16000",
            Self::Pcm22050 => "pcm_22050",
            Self::Pcm24000 => "pcm_24000",
            Self::Pcm44100 => "pcm_44100",
            Self::Pcm48000 => "pcm_48000",
            Self::Ulaw8000 => "ulaw_8000",
        }
    }

    /// Sample rate in Hz associated with this encoding.
    pub fn sample_rate(self) -> u32 {
        match self {
            Self::Pcm8000 | Self::Ulaw8000 => 8_000,
            Self::Pcm16000 => 16_000,
            Self::Pcm22050 => 22_050,
            Self::Pcm24000 => 24_000,
            Self::Pcm44100 => 44_100,
            Self::Pcm48000 => 48_000,
        }
    }
}

impl fmt::Display for AudioEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How ElevenLabs segments speech into committed transcripts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitStrategy {
    /// Segments commit only when [`AudioSink::commit`] or [`AudioSink::finish`] is called.
    #[default]
    Manual,
    /// Voice activity detection commits automatically after silence.
    Vad,
    /// Commits when the model predicts the speaker finished their turn (supported models only).
    TurnPrediction,
}

impl CommitStrategy {
    /// Wire identifier used in the `commit_strategy` query parameter.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Vad => "vad",
            Self::TurnPrediction => "turn_prediction",
        }
    }
}

impl fmt::Display for CommitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Settings for an ElevenLabs realtime transcription session.
#[derive(Debug, Clone)]
pub struct RealtimeTranscriptionRequest {
    /// Realtime model ID, such as `scribe_v3_live` or `scribe_v2_realtime`.
    pub model: String,
    /// Input audio encoding. Defaults to [`AudioEncoding::Pcm16000`].
    pub audio_encoding: AudioEncoding,
    /// Segmentation commit strategy. Defaults to [`CommitStrategy::Manual`].
    pub commit_strategy: CommitStrategy,
    /// Optional ISO 639-1 or ISO 639-3 language code.
    pub language_code: Option<String>,
    /// Optional text context attached to the first audio chunk sent in the session.
    pub previous_text: Option<String>,
    /// Silence threshold in seconds for VAD commits.
    pub vad_silence_threshold_secs: Option<f64>,
    /// Voice activity detection threshold (`0.0..=1.0`).
    pub vad_threshold: Option<f64>,
    /// Minimum speech duration in milliseconds for VAD.
    pub min_speech_duration_ms: Option<u32>,
    /// Minimum silence duration in milliseconds for VAD.
    pub min_silence_duration_ms: Option<u32>,
    /// Request `committed_transcript_with_timestamps` events with word timings.
    pub include_timestamps: bool,
    /// Include detected `language_code` in `committed_transcript_with_timestamps`.
    pub include_language_detection: bool,
    /// Strip filler words and disfluencies from committed transcripts.
    pub no_verbatim: bool,
    /// Set `false` to request zero-retention mode (enterprise plans).
    pub enable_logging: Option<bool>,
}

impl RealtimeTranscriptionRequest {
    /// Create a request for `model` using 16 kHz mono PCM and manual commits.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            audio_encoding: AudioEncoding::default(),
            commit_strategy: CommitStrategy::default(),
            language_code: None,
            previous_text: None,
            vad_silence_threshold_secs: None,
            vad_threshold: None,
            min_speech_duration_ms: None,
            min_silence_duration_ms: None,
            include_timestamps: false,
            include_language_detection: false,
            no_verbatim: false,
            enable_logging: None,
        }
    }

    /// Set the input audio encoding and sample rate.
    pub fn with_audio_encoding(mut self, encoding: AudioEncoding) -> Self {
        self.audio_encoding = encoding;
        self
    }

    /// Set how speech segments are committed.
    pub fn with_commit_strategy(mut self, strategy: CommitStrategy) -> Self {
        self.commit_strategy = strategy;
        self
    }

    /// Pin the spoken language (ISO 639-1 or ISO 639-3).
    pub fn with_language_code(mut self, language_code: impl Into<String>) -> Self {
        self.language_code = Some(language_code.into());
        self
    }

    /// Provide preceding text context alongside the first audio chunk.
    pub fn with_previous_text(mut self, text: impl Into<String>) -> Self {
        self.previous_text = Some(text.into());
        self
    }

    /// Set the VAD silence threshold in seconds.
    pub fn with_vad_silence_threshold_secs(mut self, secs: f64) -> Self {
        self.vad_silence_threshold_secs = Some(secs);
        self
    }

    /// Set the VAD sensitivity threshold.
    pub fn with_vad_threshold(mut self, threshold: f64) -> Self {
        self.vad_threshold = Some(threshold);
        self
    }

    /// Set the minimum speech duration in milliseconds for VAD.
    pub fn with_min_speech_duration_ms(mut self, duration_ms: u32) -> Self {
        self.min_speech_duration_ms = Some(duration_ms);
        self
    }

    /// Set the minimum silence duration in milliseconds for VAD.
    pub fn with_min_silence_duration_ms(mut self, duration_ms: u32) -> Self {
        self.min_silence_duration_ms = Some(duration_ms);
        self
    }

    /// Enable or disable word-level timestamps on committed transcripts.
    pub fn with_timestamps(mut self, include: bool) -> Self {
        self.include_timestamps = include;
        self
    }

    /// Enable or disable language detection on committed transcripts.
    pub fn with_language_detection(mut self, include: bool) -> Self {
        self.include_language_detection = include;
        self
    }

    /// Enable or disable removal of filler words and disfluencies.
    pub fn with_no_verbatim(mut self, no_verbatim: bool) -> Self {
        self.no_verbatim = no_verbatim;
        self
    }

    /// Explicitly enable or disable server request logging.
    pub fn with_logging(mut self, enable_logging: bool) -> Self {
        self.enable_logging = Some(enable_logging);
        self
    }
}

/// An event emitted during an ElevenLabs realtime transcription session.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TranscriptionEvent {
    /// Interim transcript for the current uncommitted segment. Replaces earlier partials.
    Partial(String),
    /// Finalized transcript for a committed speech segment.
    Committed(String),
    /// Committed transcript carrying word timings or detected language.
    CommittedWithTimestamps(CommittedTranscript),
    /// Non-fatal server warning.
    Warning(String),
}

/// Committed segment details from `committed_transcript_with_timestamps`.
#[derive(Debug, Clone, PartialEq)]
pub struct CommittedTranscript {
    /// Finalized segment text.
    pub text: String,
    /// Detected or configured language code, when reported.
    pub language_code: Option<String>,
    /// Word-level timing entries, when reported.
    pub words: Vec<TranscriptionWord>,
}

/// Classification of a [`TranscriptionWord`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscriptionWordKind {
    /// Spoken word token.
    Word,
    /// Whitespace between tokens.
    Spacing,
    /// Non-speech sound event.
    AudioEvent,
    /// Unrecognized token type reported by a newer API version.
    Other(String),
}

/// Word-level timing metadata in a [`CommittedTranscript`].
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionWord {
    /// Transcribed token text.
    pub text: String,
    /// Token start offset in seconds.
    pub start: Option<f64>,
    /// Token end offset in seconds.
    pub end: Option<f64>,
    /// Token kind (`word`, `spacing`, `audio_event`).
    pub kind: TranscriptionWordKind,
    /// Speaker identifier when diarization is active.
    pub speaker_id: Option<String>,
    /// Log-probability confidence score.
    pub logprob: Option<f64>,
}

/// Category of an ElevenLabs realtime transcription error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealtimeErrorKind {
    /// Generic server error (`error`).
    Error,
    /// Invalid or unauthorized API key (`auth_error`).
    Auth,
    /// Account quota exceeded (`quota_exceeded`).
    QuotaExceeded,
    /// Explicit commit sent too soon or with too little audio (`commit_throttled`).
    CommitThrottled,
    /// Account must accept model or product terms (`unaccepted_terms`).
    UnacceptedTerms,
    /// Too many concurrent or rapid requests (`rate_limited`).
    RateLimited,
    /// Server input queue overflowed (`queue_overflow`).
    QueueOverflow,
    /// Server capacity exhausted (`resource_exhausted`).
    ResourceExhausted,
    /// Maximum session duration reached (`session_time_limit_exceeded`).
    SessionTimeLimitExceeded,
    /// Malformed audio chunk or parameters (`input_error`).
    Input,
    /// Invalid request configuration (`invalid_request`).
    InvalidRequest,
    /// Audio chunk exceeded size limit (`chunk_size_exceeded`).
    ChunkSizeExceeded,
    /// Not enough audio activity to commit or keep open (`insufficient_audio_activity`).
    InsufficientAudioActivity,
    /// Internal transcriber failure (`transcriber_error`).
    Transcriber,
}

impl RealtimeErrorKind {
    /// Wire `message_type` string from ElevenLabs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Auth => "auth_error",
            Self::QuotaExceeded => "quota_exceeded",
            Self::CommitThrottled => "commit_throttled",
            Self::UnacceptedTerms => "unaccepted_terms",
            Self::RateLimited => "rate_limited",
            Self::QueueOverflow => "queue_overflow",
            Self::ResourceExhausted => "resource_exhausted",
            Self::SessionTimeLimitExceeded => "session_time_limit_exceeded",
            Self::Input => "input_error",
            Self::InvalidRequest => "invalid_request",
            Self::ChunkSizeExceeded => "chunk_size_exceeded",
            Self::InsufficientAudioActivity => "insufficient_audio_activity",
            Self::Transcriber => "transcriber_error",
        }
    }
}

/// An error message returned by ElevenLabs's realtime transcription API.
///
/// Errors yielded by [`TranscriptionEvents`] or [`ElevenLabsProvider::realtime_transcription`](super::ElevenLabsProvider::realtime_transcription)
/// can be downcast to this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeTranscriptionError {
    /// Specific error category reported by ElevenLabs.
    pub kind: RealtimeErrorKind,
    /// Human-readable error message.
    pub message: String,
}

impl fmt::Display for RealtimeTranscriptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElevenLabs realtime transcription {} ({}): {}",
            if self.kind == RealtimeErrorKind::Error {
                "failed"
            } else {
                "error"
            },
            self.kind.as_str(),
            self.message
        )
    }
}

impl StdError for RealtimeTranscriptionError {}

/// An open ElevenLabs realtime transcription session.
///
/// Call [`split`](Self::split) to obtain the [`AudioSink`] and
/// [`TranscriptionEvents`] stream. Dropping [`TranscriptionEvents`] aborts the
/// background session task immediately.
pub struct RealtimeTranscription {
    session_id: String,
    audio: AudioSink,
    events: TranscriptionEvents,
}

impl RealtimeTranscription {
    /// ElevenLabs's `session_id` from `session_started`.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Separate the audio sink from the incoming event stream.
    pub fn split(self) -> (AudioSink, TranscriptionEvents) {
        (self.audio, self.events)
    }
}

enum AudioCommand {
    Append(Bytes),
    Commit,
    Finish,
    Close,
}

/// Sends audio chunks to an open ElevenLabs realtime session.
///
/// Dropping the sink behaves like [`finish`](Self::finish): any uncommitted
/// audio is committed before the WebSocket closes. Use [`close`](Self::close)
/// to close immediately without committing buffered audio.
pub struct AudioSink {
    commands: mpsc::Sender<AudioCommand>,
}

impl AudioSink {
    /// Queue a chunk of raw audio matching the session's [`AudioEncoding`].
    pub async fn send(&self, audio: Bytes) -> anyhow::Result<()> {
        self.command(AudioCommand::Append(audio)).await
    }

    /// Commit the current speech segment while keeping the session open for more audio.
    pub async fn commit(&self) -> anyhow::Result<()> {
        self.command(AudioCommand::Commit).await
    }

    /// Commit any remaining audio, wait for the final committed transcript, and close.
    pub async fn finish(self) -> anyhow::Result<()> {
        self.command(AudioCommand::Finish).await
    }

    /// Close the WebSocket without committing buffered audio.
    pub async fn close(self) -> anyhow::Result<()> {
        self.command(AudioCommand::Close).await
    }

    async fn command(&self, command: AudioCommand) -> anyhow::Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| anyhow!("the realtime transcription session has ended"))
    }
}

/// Incoming events from an ElevenLabs realtime session.
///
/// Ends once [`AudioSink::finish`] or [`AudioSink::close`] completes, or after
/// the first fatal error. Dropping this stream aborts the background task.
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
struct InputAudioChunk<'a> {
    message_type: &'static str,
    audio_base_64: String,
    commit: bool,
    sample_rate: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_text: Option<&'a str>,
}

#[derive(Deserialize)]
#[serde(tag = "message_type")]
enum ServerMessage {
    #[serde(rename = "session_started")]
    SessionStarted { session_id: String },
    #[serde(rename = "partial_transcript")]
    PartialTranscript { text: String },
    #[serde(rename = "committed_transcript")]
    CommittedTranscript { text: String },
    #[serde(rename = "committed_transcript_with_timestamps")]
    CommittedTranscriptWithTimestamps {
        text: String,
        #[serde(default)]
        language_code: Option<String>,
        #[serde(default)]
        words: Option<Vec<WireWord>>,
    },
    #[serde(rename = "warning")]
    Warning { warning: String },
    #[serde(rename = "error")]
    Error { error: String },
    #[serde(rename = "auth_error")]
    AuthError { error: String },
    #[serde(rename = "quota_exceeded")]
    QuotaExceeded { error: String },
    #[serde(rename = "commit_throttled")]
    CommitThrottled { error: String },
    #[serde(rename = "unaccepted_terms")]
    UnacceptedTerms { error: String },
    #[serde(rename = "rate_limited")]
    RateLimited { error: String },
    #[serde(rename = "queue_overflow")]
    QueueOverflow { error: String },
    #[serde(rename = "resource_exhausted")]
    ResourceExhausted { error: String },
    #[serde(rename = "session_time_limit_exceeded")]
    SessionTimeLimitExceeded { error: String },
    #[serde(rename = "input_error")]
    InputError { error: String },
    #[serde(rename = "invalid_request")]
    InvalidRequest { error: String },
    #[serde(rename = "chunk_size_exceeded")]
    ChunkSizeExceeded { error: String },
    #[serde(rename = "insufficient_audio_activity")]
    InsufficientAudioActivity { error: String },
    #[serde(rename = "transcriber_error")]
    TranscriberError { error: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireWord {
    text: String,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    speaker_id: Option<String>,
    #[serde(default)]
    logprob: Option<f64>,
}

impl From<WireWord> for TranscriptionWord {
    fn from(word: WireWord) -> Self {
        let kind = match word.kind.as_deref() {
            Some("word") | None => TranscriptionWordKind::Word,
            Some("spacing") => TranscriptionWordKind::Spacing,
            Some("audio_event") => TranscriptionWordKind::AudioEvent,
            Some(other) => TranscriptionWordKind::Other(other.to_owned()),
        };
        Self {
            text: word.text,
            start: word.start,
            end: word.end,
            kind,
            speaker_id: word.speaker_id,
            logprob: word.logprob,
        }
    }
}

#[derive(Debug, PartialEq)]
enum Incoming {
    SessionStarted { session_id: String },
    Event(TranscriptionEvent),
    Ignored,
}

fn decode(text: &str) -> Result<Incoming, RealtimeTranscriptionError> {
    let message = match serde_json::from_str::<ServerMessage>(text) {
        Ok(message) => message,
        Err(error) => {
            tracing::warn!(%error, bytes = text.len(), "Unreadable ElevenLabs realtime message");
            return Ok(Incoming::Ignored);
        }
    };
    let err = |kind, message| Err(RealtimeTranscriptionError { kind, message });
    match message {
        ServerMessage::SessionStarted { session_id } => Ok(Incoming::SessionStarted { session_id }),
        ServerMessage::PartialTranscript { text } => {
            Ok(Incoming::Event(TranscriptionEvent::Partial(text)))
        }
        ServerMessage::CommittedTranscript { text } => {
            Ok(Incoming::Event(TranscriptionEvent::Committed(text)))
        }
        ServerMessage::CommittedTranscriptWithTimestamps {
            text,
            language_code,
            words,
        } => Ok(Incoming::Event(
            TranscriptionEvent::CommittedWithTimestamps(CommittedTranscript {
                text,
                language_code,
                words: words
                    .unwrap_or_default()
                    .into_iter()
                    .map(TranscriptionWord::from)
                    .collect(),
            }),
        )),
        ServerMessage::Warning { warning } => {
            Ok(Incoming::Event(TranscriptionEvent::Warning(warning)))
        }
        ServerMessage::Error { error } => err(RealtimeErrorKind::Error, error),
        ServerMessage::AuthError { error } => err(RealtimeErrorKind::Auth, error),
        ServerMessage::QuotaExceeded { error } => err(RealtimeErrorKind::QuotaExceeded, error),
        ServerMessage::CommitThrottled { error } => err(RealtimeErrorKind::CommitThrottled, error),
        ServerMessage::UnacceptedTerms { error } => err(RealtimeErrorKind::UnacceptedTerms, error),
        ServerMessage::RateLimited { error } => err(RealtimeErrorKind::RateLimited, error),
        ServerMessage::QueueOverflow { error } => err(RealtimeErrorKind::QueueOverflow, error),
        ServerMessage::ResourceExhausted { error } => {
            err(RealtimeErrorKind::ResourceExhausted, error)
        }
        ServerMessage::SessionTimeLimitExceeded { error } => {
            err(RealtimeErrorKind::SessionTimeLimitExceeded, error)
        }
        ServerMessage::InputError { error } => err(RealtimeErrorKind::Input, error),
        ServerMessage::InvalidRequest { error } => err(RealtimeErrorKind::InvalidRequest, error),
        ServerMessage::ChunkSizeExceeded { error } => {
            err(RealtimeErrorKind::ChunkSizeExceeded, error)
        }
        ServerMessage::InsufficientAudioActivity { error } => {
            err(RealtimeErrorKind::InsufficientAudioActivity, error)
        }
        ServerMessage::TranscriberError { error } => err(RealtimeErrorKind::Transcriber, error),
        ServerMessage::Other => Ok(Incoming::Ignored),
    }
}

fn encode_chunk(
    audio: &[u8],
    commit: bool,
    sample_rate: u32,
    previous_text: Option<&str>,
) -> String {
    serde_json::to_string(&InputAudioChunk {
        message_type: "input_audio_chunk",
        audio_base_64: STANDARD.encode(audio),
        commit,
        sample_rate,
        previous_text,
    })
    .expect("audio chunk fields always serialize")
}

fn audio_chunks(audio: &[u8], sample_rate: u32, previous_text: &mut Option<String>) -> Vec<String> {
    audio
        .chunks(MAX_CHUNK_BYTES)
        .map(|chunk| {
            let prev = previous_text.take();
            encode_chunk(chunk, false, sample_rate, prev.as_deref())
        })
        .collect()
}

fn commit_chunk(sample_rate: u32, previous_text: &mut Option<String>) -> String {
    let prev = previous_text.take();
    encode_chunk(&[], true, sample_rate, prev.as_deref())
}

fn realtime_url(base_url: &Url, request: &RealtimeTranscriptionRequest) -> anyhow::Result<Url> {
    let mut url = base_url.clone();
    let scheme = match url.scheme() {
        "https" | "wss" => "wss",
        "http" | "ws" => "ws",
        other => bail!("unsupported ElevenLabs URL scheme '{other}'"),
    };
    url.set_scheme(scheme)
        .map_err(|()| anyhow!("cannot use '{scheme}' with the ElevenLabs URL"))?;

    let prefix = url.path().trim_end_matches('/');
    let prefix = prefix.strip_suffix("/v1").unwrap_or(prefix);
    url.set_path(&format!("{prefix}/v1/speech-to-text/realtime"));

    {
        let mut query = url.query_pairs_mut();
        query.clear();
        query.append_pair("model_id", &request.model);
        query.append_pair("audio_format", request.audio_encoding.as_str());
        query.append_pair("commit_strategy", request.commit_strategy.as_str());
        if let Some(language_code) = &request.language_code {
            query.append_pair("language_code", language_code);
        }
        if let Some(secs) = request.vad_silence_threshold_secs {
            query.append_pair("vad_silence_threshold_secs", &secs.to_string());
        }
        if let Some(threshold) = request.vad_threshold {
            query.append_pair("vad_threshold", &threshold.to_string());
        }
        if let Some(ms) = request.min_speech_duration_ms {
            query.append_pair("min_speech_duration_ms", &ms.to_string());
        }
        if let Some(ms) = request.min_silence_duration_ms {
            query.append_pair("min_silence_duration_ms", &ms.to_string());
        }
        if request.include_timestamps {
            query.append_pair("include_timestamps", "true");
        }
        if request.include_language_detection {
            query.append_pair("include_language_detection", "true");
        }
        if request.no_verbatim {
            query.append_pair("no_verbatim", "true");
        }
        if let Some(enable_logging) = request.enable_logging {
            query.append_pair(
                "enable_logging",
                if enable_logging { "true" } else { "false" },
            );
        }
    }

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
    let url = realtime_url(base_url, &request)?;
    let mut key_header = HeaderValue::from_str(api_key.expose_secret())
        .context("the ElevenLabs API key is not a valid header value")?;
    key_header.set_sensitive(true);

    let connect = WebSocket::connect(url)
        .with_connector(TlsConnector::from(websocket_tls_config()))
        .with_options(socket_options())
        .with_request(HttpRequestBuilder::new().header(API_KEY_HEADER, key_header));
    let mut socket = tokio::time::timeout(HANDSHAKE_TIMEOUT, connect)
        .await
        .context("timed out connecting to ElevenLabs realtime transcription")?
        .context("connecting to ElevenLabs realtime transcription")?;

    let session_id = tokio::time::timeout(HANDSHAKE_TIMEOUT, wait_for_session(&mut socket))
        .await
        .context("timed out waiting for the ElevenLabs realtime session")??;
    tracing::debug!(
        session_id = %session_id,
        model = %request.model,
        "ElevenLabs realtime session open"
    );

    let config = SessionConfig {
        sample_rate: request.audio_encoding.sample_rate(),
        previous_text: request.previous_text,
        expects_timestamps_event: request.include_timestamps || request.include_language_detection,
    };
    let (commands_tx, commands_rx) = mpsc::channel(AUDIO_QUEUE);
    let (events_tx, events_rx) = mpsc::channel(EVENT_QUEUE);
    let task = tokio::spawn(run_session(socket, config, commands_rx, events_tx));
    Ok(RealtimeTranscription {
        session_id,
        audio: AudioSink {
            commands: commands_tx,
        },
        events: TranscriptionEvents {
            events: events_rx,
            task,
        },
    })
}

fn socket_options() -> Options {
    Options::default()
        .with_low_latency_compression()
        .with_no_delay()
}

async fn wait_for_session(socket: &mut TcpWebSocket) -> anyhow::Result<String> {
    loop {
        let frame = socket
            .next_frame()
            .await
            .context("reading the ElevenLabs realtime handshake")?;
        match frame.opcode() {
            OpCode::Text => {
                if let Incoming::SessionStarted { session_id } = decode(frame.as_str())? {
                    return Ok(session_id);
                }
            }
            OpCode::Close => bail!(
                "ElevenLabs closed the realtime session: {}",
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

struct SessionConfig {
    sample_rate: u32,
    previous_text: Option<String>,
    expects_timestamps_event: bool,
}

async fn run_session(
    mut socket: TcpWebSocket,
    mut config: SessionConfig,
    mut commands: mpsc::Receiver<AudioCommand>,
    events: mpsc::Sender<anyhow::Result<TranscriptionEvent>>,
) {
    let mut input_open = true;
    let mut uncommitted_bytes: usize = 0;
    let mut drain_deadline: Option<Instant> = None;

    let outcome: anyhow::Result<()> = loop {
        let drain_timeout = async {
            match drain_deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            () = events.closed() => break Ok(()),
            () = drain_timeout => break Ok(()),
            command = commands.recv(), if input_open => {
                let messages = match command {
                    Some(AudioCommand::Append(pcm)) => {
                        if pcm.is_empty() {
                            Vec::new()
                        } else {
                            uncommitted_bytes = uncommitted_bytes.saturating_add(pcm.len());
                            audio_chunks(&pcm, config.sample_rate, &mut config.previous_text)
                        }
                    }
                    Some(AudioCommand::Commit) => {
                        uncommitted_bytes = 0;
                        vec![commit_chunk(config.sample_rate, &mut config.previous_text)]
                    }
                    Some(AudioCommand::Close) => break Ok(()),
                    Some(AudioCommand::Finish) | None => {
                        input_open = false;
                        if uncommitted_bytes == 0 {
                            break Ok(());
                        }
                        uncommitted_bytes = 0;
                        drain_deadline = Some(Instant::now() + FINAL_COMMIT_TIMEOUT);
                        vec![commit_chunk(config.sample_rate, &mut config.previous_text)]
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
                        if drain_deadline.is_some() {
                            break Ok(());
                        }
                        break Err(anyhow!("the ElevenLabs realtime connection closed before the session finished"));
                    }
                    Err(error) => {
                        break Err(anyhow::Error::new(error).context("reading from ElevenLabs realtime transcription"));
                    }
                };
                match frame.opcode() {
                    OpCode::Text => {}
                    OpCode::Close => {
                        if drain_deadline.is_some() && matches!(frame.close_code(), None | Some(CloseCode::Normal)) {
                            break Ok(());
                        }
                        break Err(anyhow!(
                            "ElevenLabs closed the realtime session before it finished: {}",
                            describe_close(&frame)
                        ));
                    }
                    _ => continue,
                }
                match decode(frame.as_str()) {
                    Ok(Incoming::Event(event)) => {
                        let committed = matches!(event, TranscriptionEvent::Committed(_));
                        let detailed = matches!(event, TranscriptionEvent::CommittedWithTimestamps(_));
                        if committed || detailed {
                            uncommitted_bytes = 0;
                        }
                        let finished_drain = drain_deadline.is_some()
                            && (detailed || (committed && !config.expects_timestamps_event));
                        if events.send(Ok(event)).await.is_err() || finished_drain {
                            break Ok(());
                        }
                    }
                    Ok(Incoming::SessionStarted { .. } | Incoming::Ignored) => {}
                    Err(error)
                        if drain_deadline.is_some()
                            && matches!(
                                error.kind,
                                RealtimeErrorKind::CommitThrottled
                                    | RealtimeErrorKind::InsufficientAudioActivity
                            ) =>
                    {
                        break Ok(());
                    }
                    Err(error) => break Err(error.into()),
                }
            }
        }
    };

    if let Err(error) = outcome {
        tracing::debug!(
            error = format!("{error:#}"),
            "ElevenLabs realtime session failed"
        );
        let _ = events.send(Err(error)).await;
    }
    let _ = tokio::time::timeout(CLOSE_TIMEOUT, socket.close()).await;
}

async fn send_all(socket: &mut TcpWebSocket, messages: Vec<String>) -> anyhow::Result<()> {
    for message in messages {
        socket
            .send(Frame::text(message))
            .await
            .context("sending audio to ElevenLabs realtime transcription")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Mutex;

    use futures_util::StreamExt;
    use hyper::Request;
    use hyper::body::Incoming as IncomingBody;
    use hyper::header::SEC_WEBSOCKET_EXTENSIONS;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use yawc::UpgradeFut;

    use super::*;
    use crate::ElevenLabsProvider;

    fn parse_json(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn realtime_url_encodes_query_and_preserves_gateway_prefix() {
        let request = RealtimeTranscriptionRequest::new("scribe_v3_live")
            .with_audio_encoding(AudioEncoding::Pcm24000)
            .with_commit_strategy(CommitStrategy::Vad)
            .with_language_code("en")
            .with_vad_silence_threshold_secs(0.9)
            .with_timestamps(true)
            .with_language_detection(true)
            .with_no_verbatim(true)
            .with_logging(false);
        let url =
            realtime_url(&Url::parse("https://api.elevenlabs.io/").unwrap(), &request).unwrap();
        assert_eq!(url.scheme(), "wss");
        assert_eq!(url.path(), "/v1/speech-to-text/realtime");
        let params: Vec<(String, String)> = url.query_pairs().into_owned().collect();
        assert_eq!(
            params,
            vec![
                ("model_id".into(), "scribe_v3_live".into()),
                ("audio_format".into(), "pcm_24000".into()),
                ("commit_strategy".into(), "vad".into()),
                ("language_code".into(), "en".into()),
                ("vad_silence_threshold_secs".into(), "0.9".into()),
                ("include_timestamps".into(), "true".into()),
                ("include_language_detection".into(), "true".into()),
                ("no_verbatim".into(), "true".into()),
                ("enable_logging".into(), "false".into()),
            ]
        );

        let gateway = realtime_url(
            &Url::parse("http://127.0.0.1:9999/proxy/v1").unwrap(),
            &RealtimeTranscriptionRequest::new("scribe_v2_realtime"),
        )
        .unwrap();
        assert_eq!(gateway.scheme(), "ws");
        assert_eq!(gateway.path(), "/proxy/v1/speech-to-text/realtime");
    }

    #[test]
    fn audio_chunks_attach_previous_text_once_and_split_large_buffers() {
        let pcm = vec![3u8; MAX_CHUNK_BYTES + 8];
        let mut previous = Some("Earlier sentence.".to_owned());
        let messages: Vec<Value> = audio_chunks(&pcm, 16_000, &mut previous)
            .into_iter()
            .map(|m| parse_json(&m))
            .collect();
        assert_eq!(messages.len(), 2);
        assert!(previous.is_none());
        assert_eq!(messages[0]["message_type"], "input_audio_chunk");
        assert_eq!(messages[0]["commit"], false);
        assert_eq!(messages[0]["sample_rate"], 16_000);
        assert_eq!(messages[0]["previous_text"], "Earlier sentence.");
        assert!(messages[1].get("previous_text").is_none());
        let first = STANDARD
            .decode(messages[0]["audio_base_64"].as_str().unwrap())
            .unwrap();
        let second = STANDARD
            .decode(messages[1]["audio_base_64"].as_str().unwrap())
            .unwrap();
        assert_eq!(first.len(), MAX_CHUNK_BYTES);
        assert_eq!(second, vec![3u8; 8]);
    }

    #[test]
    fn server_events_and_errors_decode() {
        assert_eq!(
            decode(r#"{"message_type":"session_started","session_id":"sess-7","config":{"sample_rate":16000}}"#).unwrap(),
            Incoming::SessionStarted {
                session_id: "sess-7".into(),
            }
        );
        assert_eq!(
            decode(r#"{"message_type":"partial_transcript","text":"Hel"}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::Partial("Hel".into()))
        );
        assert_eq!(
            decode(r#"{"message_type":"committed_transcript","text":"Hello world."}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::Committed("Hello world.".into()))
        );
        assert_eq!(
            decode(r#"{"message_type":"committed_transcript_with_timestamps","text":"Hi","language_code":"en","words":[{"text":"Hi","start":0.0,"end":0.3,"type":"word","logprob":-0.1}]}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::CommittedWithTimestamps(CommittedTranscript {
                text: "Hi".into(),
                language_code: Some("en".into()),
                words: vec![TranscriptionWord {
                    text: "Hi".into(),
                    start: Some(0.0),
                    end: Some(0.3),
                    kind: TranscriptionWordKind::Word,
                    speaker_id: None,
                    logprob: Some(-0.1),
                }],
            }))
        );
        assert_eq!(
            decode(r#"{"message_type":"warning","warning":"high background noise"}"#).unwrap(),
            Incoming::Event(TranscriptionEvent::Warning("high background noise".into()))
        );
        assert_eq!(
            decode(
                r#"{"message_type":"speakers_merged","from_speaker_id":"a","to_speaker_id":"b"}"#
            )
            .unwrap(),
            Incoming::Ignored
        );

        let auth =
            decode(r#"{"message_type":"auth_error","error":"Invalid API key"}"#).unwrap_err();
        assert_eq!(auth.kind, RealtimeErrorKind::Auth);
        assert_eq!(auth.message, "Invalid API key");
    }

    struct Seen {
        api_key: Option<String>,
        query: Option<String>,
        extensions: Option<String>,
        messages: Vec<Value>,
    }

    struct Handshake {
        api_key: Option<String>,
        query: Option<String>,
        extensions: Option<String>,
        upgrade: UpgradeFut,
    }

    async fn fake_elevenlabs(commit_reply: &'static str) -> (Url, JoinHandle<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (handshake_tx, handshake_rx) = oneshot::channel();
            let handshake_tx = Mutex::new(Some(handshake_tx));
            let service = service_fn(move |mut request: Request<IncomingBody>| {
                assert_eq!(request.uri().path(), "/v1/speech-to-text/realtime");
                let header = |name: &str| {
                    request
                        .headers()
                        .get(name)
                        .map(|value: &HeaderValue| value.to_str().unwrap().to_string())
                };
                let api_key = header(API_KEY_HEADER);
                let extensions = header(SEC_WEBSOCKET_EXTENSIONS.as_str());
                let query = request.uri().query().map(str::to_owned);
                let (response, upgrade) =
                    WebSocket::upgrade_with_options(&mut request, socket_options()).unwrap();
                let sender = handshake_tx.lock().unwrap().take().unwrap();
                assert!(
                    sender
                        .send(Handshake {
                            api_key,
                            query,
                            extensions,
                            upgrade,
                        })
                        .is_ok()
                );
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
                    r#"{"message_type":"session_started","session_id":"sess-42","config":{"sample_rate":16000,"model_id":"scribe_v3_live"}}"#,
                ))
                .await
                .unwrap();

            let mut messages = Vec::new();
            while let Ok(frame) = socket.next_frame().await {
                match frame.opcode() {
                    OpCode::Text => {
                        let msg = parse_json(frame.as_str());
                        let commit = msg["commit"].as_bool().unwrap_or(false);
                        messages.push(msg);
                        if !commit {
                            socket
                                .send(Frame::text(
                                    r#"{"message_type":"partial_transcript","text":"Hello"}"#,
                                ))
                                .await
                                .unwrap();
                        } else {
                            socket.send(Frame::text(commit_reply)).await.unwrap();
                        }
                    }
                    OpCode::Close => break,
                    _ => {}
                }
            }

            Seen {
                api_key: handshake.api_key,
                query: handshake.query,
                extensions: handshake.extensions,
                messages,
            }
        });
        (url, server)
    }

    #[tokio::test]
    async fn session_streams_audio_and_drains_final_commit() {
        let (url, server) =
            fake_elevenlabs(r#"{"message_type":"committed_transcript","text":"Hello world."}"#)
                .await;
        let provider = ElevenLabsProvider::new("secret-key")
            .unwrap()
            .with_base_url(url);
        let session = provider
            .realtime_transcription(
                RealtimeTranscriptionRequest::new("scribe_v3_live")
                    .with_commit_strategy(CommitStrategy::Vad)
                    .with_previous_text("Context."),
            )
            .await
            .unwrap();
        assert_eq!(session.session_id(), "sess-42");

        let (audio, events) = session.split();
        audio.send(Bytes::from_static(&[1, 0, 2, 0])).await.unwrap();
        audio.finish().await.unwrap();

        let collected: Vec<TranscriptionEvent> = events.map(|event| event.unwrap()).collect().await;
        assert_eq!(
            collected,
            vec![
                TranscriptionEvent::Partial("Hello".into()),
                TranscriptionEvent::Committed("Hello world.".into()),
            ]
        );

        let seen = server.await.unwrap();
        assert_eq!(seen.api_key.as_deref(), Some("secret-key"));
        assert!(
            seen.extensions
                .as_deref()
                .unwrap_or_default()
                .contains("permessage-deflate")
        );
        assert_eq!(
            seen.query.as_deref(),
            Some("model_id=scribe_v3_live&audio_format=pcm_16000&commit_strategy=vad")
        );
        assert_eq!(seen.messages.len(), 2);
        assert_eq!(
            seen.messages[0],
            json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": "AQACAA==",
                "commit": false,
                "sample_rate": 16000,
                "previous_text": "Context.",
            })
        );
        assert_eq!(
            seen.messages[1],
            json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": true,
                "sample_rate": 16000,
            })
        );
    }

    #[tokio::test]
    async fn finish_ignores_insufficient_audio_activity_on_trailing_silence() {
        let (url, server) = fake_elevenlabs(
            r#"{"message_type":"insufficient_audio_activity","error":"No speech activity"}"#,
        )
        .await;
        let provider = ElevenLabsProvider::new("secret-key")
            .unwrap()
            .with_base_url(url);
        let session = provider
            .realtime_transcription(RealtimeTranscriptionRequest::new("scribe_v3_live"))
            .await
            .unwrap();
        let (audio, events) = session.split();
        audio.send(Bytes::from_static(&[0, 0, 0, 0])).await.unwrap();
        audio.finish().await.unwrap();

        let collected: Vec<TranscriptionEvent> = events.map(|event| event.unwrap()).collect().await;
        assert_eq!(collected, vec![TranscriptionEvent::Partial("Hello".into())]);
        server.await.unwrap();
    }
}
