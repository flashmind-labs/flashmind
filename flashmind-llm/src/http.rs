//! HTTP utilities including shared client, TLS config, retry logic, and rate limiting.
//!
//! Provides [`send_with_retry`] for automatic retry on 429/5xx responses with fixed-interval
//! backoff (2s between retries, 120s total budget). All provider modules use this instead of
//! raw `reqwest` calls to ensure consistent error handling and metrics.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use flashmind_types::{CompletionStream, FinishReason, StreamEvent, SttRequest, TtsRequest};
use metrics;
use ratelimit::Ratelimiter;
use reqwest::{Client, RequestBuilder, Response};
use serde::Deserialize;
use tokio_stream::StreamExt;

/// Maximum number of retry attempts for HTTP requests with [`send_with_retry`].
pub const MAX_RETRIES: u32 = 30;
const RETRY_INTERVAL_MS: u64 = 2000;
const RETRY_BUDGET_SECS: u64 = 120;

fn tls_config() -> rustls::ClientConfig {
    let root_store =
        rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    config.enable_early_data = true;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    config
}

static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    http_client_builder()
        .build()
        .expect("failed to build HTTP client")
});

/// Returns a clone of the shared HTTP client (cheap — just an Arc bump).
pub fn http_client() -> Client {
    HTTP_CLIENT.clone()
}

/// Create a pre-configured reqwest client builder with rustls TLS and compression enabled.
///
/// Use this when you need to customize the client beyond what [`http_client`] provides.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    let _ = rustls::crypto::ring::default_provider().install_default();
    Client::builder()
        .use_preconfigured_tls(tls_config())
        .user_agent("Flash/1.0 (Flashmind Labs)")
}

/// Send an HTTP request with automatic retry on transient failures.
///
/// Retries on:
/// - **Connection errors** (DNS failures, refused connections, etc.)
/// - **Timeouts** (connection timeout, request timeout)
/// - **429 Rate Limit** responses
/// - **5xx Server Errors**
///
/// Uses a fixed 2s retry interval or respects the `Retry-After` header if provided.
/// Gives up after 120 seconds total elapsed time or after MAX_RETRIES attempts.
pub async fn send_with_retry(
    build_request: impl Fn() -> RequestBuilder,
) -> Result<Response, reqwest::Error> {
    let start = std::time::Instant::now();
    let mut attempt = 0;

    loop {
        attempt += 1;
        let result = build_request().send().await;

        // Handle connection/timeout errors - retry immediately
        if let Err(e) = &result
            && (e.is_connect() || e.is_timeout())
        {
            metrics::counter!("llm.http.retries").increment(1);
            metrics::counter!("llm.http.retry_reason.connection").increment(1);
            let over_budget = start.elapsed().as_secs() >= RETRY_BUDGET_SECS;
            if !over_budget && attempt <= MAX_RETRIES {
                tracing::warn!(
                    attempt,
                    wait_ms = RETRY_INTERVAL_MS,
                    error = ?e,
                    "Connection failed, retrying"
                );
                tokio::time::sleep(Duration::from_millis(RETRY_INTERVAL_MS)).await;
                continue;
            }
        }

        let response = result?;
        let status = response.status();

        // Check if we should retry based on status code
        let over_budget = start.elapsed().as_secs() >= RETRY_BUDGET_SECS;
        let should_retry = !over_budget
            && (status.as_u16() == 429 || status.is_server_error())
            && attempt <= MAX_RETRIES;

        if should_retry {
            metrics::counter!("llm.http.retries").increment(1);
            if status.as_u16() == 429 {
                metrics::counter!("llm.http.retry_reason.rate_limit").increment(1);
            } else {
                metrics::counter!("llm.http.retry_reason.server_error").increment(1);
            }

            // Check for Retry-After header
            let wait_ms = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(|s| s * 1000)
                .unwrap_or(RETRY_INTERVAL_MS);

            tracing::warn!(
                attempt,
                wait_ms,
                status = status.as_u16(),
                "Retryable error ({}), retrying",
                status
            );

            tokio::time::sleep(Duration::from_millis(wait_ms)).await;
            continue;
        }

        return Ok(response);
    }
}

/// Create a token-bucket rate limiter for the given requests-per-minute.
pub fn create_rate_limiter(rpm: u32) -> Arc<Ratelimiter> {
    let rpm = rpm.max(1);
    tracing::debug!(rpm, "creating LLM rate limiter");

    Arc::new(
        Ratelimiter::builder(rpm as u64)
            .max_tokens(rpm as u64)
            .initial_available(rpm as u64)
            .build()
            .expect("LLM rate limiter"),
    )
}

/// Wait until the rate limiter allows a request.
pub async fn wait_for_rate_limit(limiter: &Ratelimiter) {
    use ratelimit::TryWaitError;
    loop {
        match limiter.try_wait() {
            Ok(()) => return,
            Err(TryWaitError::Insufficient(wait)) => tokio::time::sleep(wait).await,
            Err(_) => {
                tracing::warn!("unexpected ratelimit error, backing off");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// Send a text-to-speech request and stream the audio body as [`StreamEvent::AudioDelta`]s.
///
/// `req` carries the URL and auth; the JSON body is added here.
pub(crate) fn tts_stream(req: RequestBuilder, request: TtsRequest) -> CompletionStream {
    let format = request.response_format.to_string();

    Box::pin(async_stream::stream! {
        let body = serde_json::json!({
            "model": request.model,
            "input": request.input,
            "voice": request.voice,
            "response_format": &format,
        });

        let response = match req.json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                yield Err(anyhow::anyhow!("TTS request failed: {e}"));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            yield Err(anyhow::anyhow!("TTS API error ({status}): {text}"));
            return;
        }

        let mut byte_stream = response.bytes_stream();
        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    yield Ok(StreamEvent::AudioDelta { data: bytes.to_vec(), format: format.clone() });
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!("TTS stream error: {e}"));
                    return;
                }
            }
        }
        yield Ok(StreamEvent::Finished(FinishReason::Stop));
    })
}

/// Send a speech-to-text request as a multipart upload and yield the transcript.
///
/// `req` carries the URL and auth; the multipart form is added here.
pub(crate) fn stt_stream(req: RequestBuilder, request: SttRequest) -> CompletionStream {
    Box::pin(async_stream::stream! {
        let ext = request.media_type.split('/').next_back().unwrap_or("mp3");
        let filename = format!("audio.{ext}");

        let part = reqwest::multipart::Part::bytes(request.audio)
            .file_name(filename)
            .mime_str(&request.media_type)
            .unwrap();

        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model)
            .part("file", part);

        if let Some(lang) = request.language {
            form = form.text("language", lang);
        }

        let response = match req.multipart(form).send().await {
            Ok(r) => r,
            Err(e) => {
                yield Err(anyhow::anyhow!("STT request failed: {e}"));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            yield Err(anyhow::anyhow!("STT API error ({status}): {text}"));
            return;
        }

        #[derive(Deserialize)]
        struct TranscriptionResponse {
            text: String,
        }

        match response.json::<TranscriptionResponse>().await {
            Ok(resp) => {
                yield Ok(StreamEvent::ContentDelta(resp.text));
            }
            Err(e) => {
                yield Err(anyhow::anyhow!("Failed to parse transcription response: {e}"));
                return;
            }
        }
        yield Ok(StreamEvent::Finished(FinishReason::Stop));
    })
}
