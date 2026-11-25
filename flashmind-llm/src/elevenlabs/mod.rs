//! ElevenLabs speech synthesis, voice discovery and realtime transcription
//! against `https://api.elevenlabs.io`.

mod provider;
#[cfg(feature = "elevenlabs-realtime")]
mod realtime;

pub use provider::ElevenLabsProvider;
#[cfg(feature = "elevenlabs-realtime")]
pub use realtime::{
    AudioEncoding, AudioSink, CommitStrategy, CommittedTranscript, REALTIME_SAMPLE_RATE,
    RealtimeErrorKind, RealtimeTranscription, RealtimeTranscriptionError,
    RealtimeTranscriptionRequest, TranscriptionEvent, TranscriptionEvents, TranscriptionWord,
    TranscriptionWordKind,
};
