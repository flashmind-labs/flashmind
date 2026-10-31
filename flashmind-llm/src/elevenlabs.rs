//! ElevenLabs speech synthesis and catalog discovery.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use flashmind_types::{
    AudioFormat, CompletionRequest, CompletionStream, FinishReason, LlmProvider, ModelCapabilities,
    ModelCategory, ModelInfo, ModelPricing, Provider, StreamEvent, TtsRequest, Voice,
};
use futures_util::StreamExt;
use reqwest::{Client, Method};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use url::Url;

use crate::http::{add_extra_args, http_client_builder};

const DEFAULT_BASE_URL: &str = "https://api.elevenlabs.io/";
const MAX_VOICE_PAGES: usize = 100;

/// ElevenLabs text-to-speech provider.
pub struct ElevenLabsProvider {
    api_key: SecretString,
    base_url: Url,
    client: Client,
    extra_args: Map<String, Value>,
}

impl ElevenLabsProvider {
    /// Create a provider using the ElevenLabs API.
    pub fn new(api_key: impl Into<SecretString>) -> Result<Self> {
        Ok(Self {
            api_key: api_key.into(),
            base_url: Url::parse(DEFAULT_BASE_URL).context("invalid ElevenLabs base URL")?,
            client: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(120))
                .build()
                .context("building ElevenLabs HTTP client")?,
            extra_args: Map::new(),
        })
    }

    /// Send requests to a different API root.
    pub fn with_base_url(mut self, url: Url) -> Self {
        self.base_url = url;
        self
    }

    /// Add provider-specific fields to speech requests.
    pub fn with_extra_args(mut self, extra: Map<String, Value>) -> Self {
        self.extra_args = extra;
        self
    }

    fn url(&self, path: &str) -> Result<Url> {
        let mut url = self.base_url.clone();
        let prefix = url.path().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        let prefix = if path.starts_with("v2/") {
            prefix.strip_suffix("/v1").unwrap_or(prefix)
        } else {
            prefix
        };
        let path = if prefix.ends_with("/v1") {
            path.strip_prefix("v1/").unwrap_or(path)
        } else {
            path
        };
        url.set_path(&format!("{prefix}/{path}"));
        Ok(url)
    }

    fn request(&self, method: Method, url: Url) -> reqwest::RequestBuilder {
        self.client
            .request(method, url)
            .header("xi-api-key", self.api_key.expose_secret())
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        let mut voices = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..MAX_VOICE_PAGES {
            let mut url = self.url("v2/voices")?;
            url.query_pairs_mut().append_pair("page_size", "100");
            if let Some(token) = &token {
                url.query_pairs_mut().append_pair("next_page_token", token);
            }
            let response = self
                .request(Method::GET, url)
                .send()
                .await
                .context("ElevenLabs voice listing failed")?
                .error_for_status()
                .context("ElevenLabs voice listing failed")?;
            let page: VoicesPage = response
                .json()
                .await
                .context("invalid ElevenLabs voice list")?;
            voices.extend(page.voices.into_iter().map(|voice| Voice {
                id: voice.voice_id,
                name: voice.name,
            }));
            if !page.has_more {
                return Ok(voices);
            }
            let next = page
                .next_page_token
                .context("ElevenLabs voice list missing next page token")?;
            if token.as_ref() == Some(&next) {
                bail!("ElevenLabs voice pagination did not advance");
            }
            token = Some(next);
        }
        bail!("ElevenLabs voice listing exceeded the page limit")
    }
}

#[derive(Deserialize)]
struct VoicesPage {
    voices: Vec<VoiceEntry>,
    #[serde(default)]
    has_more: bool,
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
struct VoiceEntry {
    voice_id: String,
    name: String,
}

#[derive(Deserialize)]
struct SpeechModel {
    model_id: String,
    name: Option<String>,
    #[serde(default)]
    can_do_text_to_speech: bool,
}

#[async_trait]
impl LlmProvider for ElevenLabsProvider {
    fn name(&self) -> &str {
        "elevenlabs"
    }
    fn provider(&self) -> Provider {
        Provider::Custom("elevenlabs".into())
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        let url = self
            .url("v1/models")
            .inspect_err(|error| tracing::warn!(%error, "ElevenLabs model URL failed"))
            .ok()?;
        let models: Vec<SpeechModel> = self
            .request(Method::GET, url)
            .send()
            .await
            .inspect_err(|error| tracing::warn!(%error, "ElevenLabs model listing failed"))
            .ok()?
            .error_for_status()
            .inspect_err(|error| tracing::warn!(%error, "ElevenLabs model listing failed"))
            .ok()?
            .json()
            .await
            .inspect_err(|error| tracing::warn!(%error, "ElevenLabs model list invalid"))
            .ok()?;
        Some(
            models
                .into_iter()
                .filter(|model| model.can_do_text_to_speech)
                .map(|model| ModelInfo {
                    image: Default::default(),
                    id: model.model_id,
                    name: model.name,
                    created_at: None,
                    context_length: None,
                    max_completion_tokens: None,
                    capabilities: ModelCapabilities {
                        audio_output: true,
                        ..Default::default()
                    },
                    categories: vec![ModelCategory::Tts],
                    pricing: ModelPricing::default(),
                })
                .collect(),
        )
    }

    async fn list_voices(&self, _model: &str) -> Option<Vec<Voice>> {
        self.voices()
            .await
            .inspect_err(|error| tracing::warn!(%error, "ElevenLabs voice listing failed"))
            .ok()
    }

    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        Box::pin(futures_util::stream::once(async {
            Err(anyhow::anyhow!(
                "ElevenLabs does not provide chat completions"
            ))
        }))
    }

    fn text_to_speech(&self, request: TtsRequest) -> CompletionStream {
        let builder = self.url("v1/text-to-speech").and_then(|mut url| {
            url.path_segments_mut()
                .map_err(|_| anyhow::anyhow!("invalid ElevenLabs speech URL"))?
                .push(&request.voice);
            let output = match request.response_format {
                AudioFormat::Mp3 => "mp3_44100_128",
                AudioFormat::Wav => "wav_24000",
                AudioFormat::Opus => "opus_48000_128",
                AudioFormat::Aac | AudioFormat::Flac => {
                    bail!("ElevenLabs supports mp3, wav and opus output in flsh")
                }
            };
            url.query_pairs_mut().append_pair("output_format", output);
            Ok(self.request(Method::POST, url))
        });
        let extra = self.extra_args.clone();
        Box::pin(async_stream::try_stream! {
            let builder = builder?;
            let format = request.response_format.to_string();
            let body = add_extra_args(json!({"model_id": request.model, "text": request.input}), &extra)?;
            let response = builder.json(&body).send().await.context("ElevenLabs speech request failed")?;
            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.context("cannot read ElevenLabs speech error")?;
                Err(anyhow::anyhow!("ElevenLabs speech failed ({status}): {text}"))?;
            } else {
                let mut chunks = response.bytes_stream();
                while let Some(chunk) = chunks.next().await {
                    let data = chunk.context("ElevenLabs speech stream failed")?;
                    yield StreamEvent::AudioDelta { data: data.to_vec(), format: format.clone() };
                }
                yield StreamEvent::Finished(FinishReason::Stop);
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn lists_speech_models_and_paginated_voices_and_generates_audio() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let base = Url::parse(&format!(
            "http://{}/gateway/v1",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let provider = ElevenLabsProvider::new("example-key")
            .unwrap()
            .with_base_url(base)
            .with_extra_args(
                serde_json::from_value(json!({"voice_settings": {"stability": 0.5}})).unwrap(),
            );
        let serve = async {
            for (path, response, mime) in [
                (
                    "/gateway/v1/models",
                    r#"[{"model_id":"example-speech","name":"Example","can_do_text_to_speech":true},{"model_id":"example-other","can_do_text_to_speech":false}]"#,
                    "application/json",
                ),
                (
                    "/gateway/v2/voices?page_size=100",
                    r#"{"voices":[{"voice_id":"first","name":"First"}],"has_more":true,"next_page_token":"next"}"#,
                    "application/json",
                ),
                (
                    "/gateway/v2/voices?page_size=100&next_page_token=next",
                    r#"{"voices":[{"voice_id":"second","name":"Second"}],"has_more":false}"#,
                    "application/json",
                ),
                (
                    "/gateway/v1/text-to-speech/first?output_format=mp3_44100_128",
                    "example-audio",
                    "audio/mpeg",
                ),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut header_bytes = Vec::new();
                while !header_bytes.ends_with(b"\r\n\r\n") {
                    header_bytes.push(socket.read_u8().await.unwrap());
                }
                let headers = String::from_utf8(header_bytes).unwrap();
                assert!(headers.lines().next().unwrap().contains(path));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("xi-api-key: example-key\r\n")
                );
                if path.contains("text-to-speech") {
                    let headers = headers.to_ascii_lowercase();
                    let len: usize = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .unwrap()
                        .parse()
                        .unwrap();
                    let mut body = vec![0; len];
                    socket.read_exact(&mut body).await.unwrap();
                    let body: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(body["model_id"], "example-speech");
                    assert_eq!(body["text"], "hello");
                    assert_eq!(body["voice_settings"]["stability"], 0.5);
                }
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
            }
        };
        let exercise = async {
            let models = provider.list_models().await.unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].categories, [ModelCategory::Tts]);
            let voices = provider.list_voices("example-speech").await.unwrap();
            assert_eq!(
                voices
                    .iter()
                    .map(|voice| voice.id.as_str())
                    .collect::<Vec<_>>(),
                ["first", "second"]
            );
            let request = TtsRequest {
                model: "example-speech".into(),
                input: "hello".into(),
                voice: "first".into(),
                response_format: AudioFormat::Mp3,
            };
            let events = provider.text_to_speech(request).collect::<Vec<_>>().await;
            let events: Vec<_> = events.into_iter().collect::<Result<_, _>>().unwrap();
            assert!(events.iter().any(|event| matches!(event, StreamEvent::AudioDelta { data, format } if data == b"example-audio" && format == "mp3")));
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(exercise, serve);
        })
        .await
        .expect("The local endpoint should complete the requests.");
    }

    #[tokio::test]
    async fn rejects_unsupported_output_format_without_request() {
        let provider = ElevenLabsProvider::new("example-key").unwrap();
        let request = TtsRequest {
            model: "example".into(),
            input: "hello".into(),
            voice: "example".into(),
            response_format: AudioFormat::Aac,
        };
        let events = provider.text_to_speech(request).collect::<Vec<_>>().await;
        assert!(events[0].is_err());
    }
}
