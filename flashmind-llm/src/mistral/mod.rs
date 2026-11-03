//! Mistral provider: chat completions, batch transcription and realtime
//! transcription against `https://api.mistral.ai`.
//!
//! See <https://docs.mistral.ai/api> for the API reference.

mod models;
mod ocr;
mod provider;
#[cfg(feature = "mistral-realtime")]
mod realtime;
mod request;
mod speech;

pub use ocr::{OcrDocument, OcrRequest};
pub use provider::MistralProvider;
#[cfg(feature = "mistral-realtime")]
pub use realtime::{
    AudioSink, REALTIME_SAMPLE_RATE, RealtimeTranscription, RealtimeTranscriptionError,
    RealtimeTranscriptionRequest, TranscriptionDone, TranscriptionEvent, TranscriptionEvents,
};
