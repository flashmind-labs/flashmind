use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_types::SttRequest;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::OpenRouterProvider;

/// Timing detail requested from a transcription provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, strum::EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum TimestampGranularity {
    /// Start and end times for transcript segments.
    Segment,
    /// Start and end times for individual words.
    Word,
}

/// Structured output and provider-specific controls for OpenRouter transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionOptions {
    /// Request `verbose_json`, including metadata the provider supports.
    pub verbose_json: bool,
    /// Timing detail to request. Nonempty values also enable `verbose_json`.
    pub timestamp_granularities: Vec<TimestampGranularity>,
    /// Provider parameters keyed by endpoint slug, forwarded as `provider.options`.
    /// For example: `{"azure":{"diarization":{"enabled":true}}}`.
    pub provider_options: Map<String, Value>,
}

/// A complete transcription, preserving all fields returned by the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The complete transcript.
    pub text: String,
    /// Remaining response fields, including segments, words, speakers and usage.
    #[serde(flatten)]
    pub metadata: Map<String, Value>,
}

impl OpenRouterProvider {
    /// Transcribe audio without dropping timing, speaker or provider metadata.
    ///
    /// Diarization is enabled using the serving provider's own parameters in
    /// `options.provider_options`. Unsupported structured output returns an error.
    pub async fn transcribe_detailed(
        &self,
        request: SttRequest,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResponse> {
        let req = self.app.apply(
            self.client
                .post("https://openrouter.ai/api/v1/audio/transcriptions")
                .bearer_auth(&self.api_key),
        );
        transcribe(req, request, options).await
    }
}

async fn transcribe(
    req: RequestBuilder,
    request: SttRequest,
    options: TranscriptionOptions,
) -> Result<TranscriptionResponse> {
    let format = request
        .media_type
        .strip_prefix("audio/")
        .filter(|format| !format.is_empty())
        .context("transcription requires an audio MIME type")?;
    let mut body = json!({
        "model": request.model,
        "input_audio": {
            "data": STANDARD.encode(&request.audio),
            "format": format,
        },
        "response_format": if options.verbose_json || !options.timestamp_granularities.is_empty() {
            "verbose_json"
        } else {
            "json"
        },
    });
    if let Some(language) = request.language {
        body["language"] = language.into();
    }
    if !options.timestamp_granularities.is_empty() {
        body["timestamp_granularities"] = json!(options.timestamp_granularities);
    }
    if !options.provider_options.is_empty() {
        body["provider"] = json!({ "options": options.provider_options });
    }

    // Transcription is billed per request, so do not retry a failed upload.
    let response = req.json(&body).send().await.context("STT request failed")?;
    if !response.status().is_success() {
        let status = response.status();
        let message = response.text().await.context("cannot read STT API error")?;
        bail!("STT API error ({status}): {message}");
    }
    response.json().await.context("cannot parse STT response")
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use crate::http::http_client_builder;

    use super::*;

    const TRANSCRIPT: &str = r#"{
        "text":"Hello. Hi.",
        "language":"en",
        "duration":2.5,
        "segments":[
            {"start":0.0,"end":1.0,"text":"Hello.","speaker":0},
            {"start":1.5,"end":2.5,"text":"Hi.","speaker":1}
        ],
        "words":[{"word":"Hello.","start":0.0,"end":1.0,"speaker":0,"confidence":0.99}],
        "usage":{"seconds":2.5,"cost":0.0001},
        "provider_extension":{"tags":["example"]}
    }"#;

    fn request() -> SttRequest {
        SttRequest {
            model: "example/transcribe".into(),
            audio: b"example audio".to_vec(),
            media_type: "audio/mp3".into(),
            language: Some("en".into()),
        }
    }

    async fn read_body(stream: &mut TcpStream) -> Value {
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        let body_start = loop {
            let count = stream.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "request must contain headers");
            bytes.extend_from_slice(&buffer[..count]);
            assert!(bytes.len() < 65536, "test request must be bounded");
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let headers = String::from_utf8_lossy(&bytes[..body_start]);
        let length: usize = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .map(str::to_owned)
            })
            .unwrap()
            .parse()
            .unwrap();
        assert!(length < 65536, "test body must be bounded");
        while bytes.len() - body_start < length {
            let count = stream.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "request must contain its full body");
            bytes.extend_from_slice(&buffer[..count]);
        }
        let body: Value = serde_json::from_slice(&bytes[body_start..]).unwrap();
        body
    }

    #[tokio::test]
    async fn forwards_options_and_preserves_the_complete_response() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let body = read_body(&mut stream).await;
            assert_eq!(
                body,
                json!({
                    "model": "example/transcribe",
                    "input_audio": {"data": STANDARD.encode(b"example audio"), "format":"mp3"},
                    "language": "en",
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment", "word"],
                    "provider": {"options": {"azure": {"diarization": {"enabled": true}}}},
                })
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{TRANSCRIPT}",
                TRANSCRIPT.len(),
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        };
        let client = http_client_builder().no_proxy().build().unwrap();
        let options = TranscriptionOptions {
            timestamp_granularities: vec![
                TimestampGranularity::Segment,
                TimestampGranularity::Word,
            ],
            provider_options: serde_json::from_value(json!({
                "azure": {"diarization": {"enabled": true}}
            }))
            .unwrap(),
            ..Default::default()
        };
        let (_, response) = tokio::join!(
            server,
            transcribe(
                client.post(format!("http://{address}/")),
                request(),
                options
            ),
        );
        let response = response.unwrap();
        assert_eq!(response.text, "Hello. Hi.");
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            serde_json::from_str::<Value>(TRANSCRIPT).unwrap()
        );
    }

    #[tokio::test]
    async fn reports_provider_errors_without_falling_back_to_plain_text() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let body = read_body(&mut stream).await;
            assert_eq!(body["response_format"], "verbose_json");
            stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 24\r\nConnection: close\r\n\r\nverbose_json unsupported").await.unwrap();
        };
        let client = http_client_builder().no_proxy().build().unwrap();
        let (_, response) = tokio::join!(
            server,
            transcribe(
                client.post(format!("http://{address}/")),
                request(),
                TranscriptionOptions {
                    verbose_json: true,
                    ..Default::default()
                },
            )
        );
        let error = response.unwrap_err().to_string();
        assert!(error.contains("400 Bad Request"), "{error}");
        assert!(error.contains("verbose_json unsupported"), "{error}");
    }
}
