//! HTTP utilities including shared client, TLS config, retry logic, and rate limiting.
//!
//! Provides [`send_with_retry`] for automatic retry on 429/5xx responses with fixed-interval
//! backoff (2s between retries, 120s total budget). All provider modules use this instead of
//! raw `reqwest` calls to ensure consistent error handling and metrics.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use chrono::{DateTime, Utc};
use flashmind_types::{CompletionStream, FinishReason, StreamEvent, SttRequest, TtsRequest};
use futures::StreamExt;
use metrics;
use ratelimit::Ratelimiter;
use reqwest::{Client, ClientBuilder, Error, RequestBuilder, Response};
use serde::Deserialize;
use tokio::time::Instant;

/// Maximum number of retries after the initial request with [`send_with_retry`].
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

static STREAMING_HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    client_builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(60))
        .build()
        .expect("the streaming HTTP client must initialize before requests can run")
});

/// Client without a total or read timeout. The stream reader owns its idle deadline.
pub(crate) fn streaming_http_client() -> Client {
    STREAMING_HTTP_CLIENT.clone()
}

/// Returns a clone of the shared HTTP client (cheap — just an Arc bump).
pub fn http_client() -> Client {
    HTTP_CLIENT.clone()
}

/// Create a pre-configured reqwest client builder with rustls TLS and compression enabled.
///
/// Use this when you need to customize the client beyond what [`http_client`] provides.
pub fn http_client_builder() -> ClientBuilder {
    client_builder().read_timeout(Duration::from_secs(RETRY_BUDGET_SECS))
}

fn client_builder() -> ClientBuilder {
    let _ = rustls::crypto::ring::default_provider().install_default();
    Client::builder()
        .use_preconfigured_tls(tls_config())
        .user_agent("Flash/1.0 (Flashmind Labs)")
        .connect_timeout(Duration::from_secs(30))
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
/// The 120s deadline covers attempts, waits, and the returned response body.
/// Gives up after MAX_RETRIES retries, or when the next wait would reach the deadline.
/// Set shorter total timeouts on the request builder: the deadline sets a
/// request timeout, which overrides the client's total timeout.
pub async fn send_with_retry(
    build_request: impl Fn() -> RequestBuilder,
) -> Result<Response, Error> {
    send_with_retry_budget(
        build_request,
        Duration::from_secs(RETRY_BUDGET_SECS),
        Duration::from_millis(RETRY_INTERVAL_MS),
    )
    .await
}

async fn send_with_retry_budget(
    build_request: impl Fn() -> RequestBuilder,
    budget: Duration,
    retry_interval: Duration,
) -> Result<Response, Error> {
    send_with_retry_mode(
        build_request,
        budget,
        retry_interval,
        ResponseBody::Buffered,
    )
    .await
}

enum ResponseBody {
    Buffered,
    Streaming,
}

/// Bound retries and response headers while leaving the stream's body deadline to its reader.
pub(crate) async fn send_stream_with_retry(
    build_request: impl Fn() -> RequestBuilder,
) -> anyhow::Result<Response> {
    send_stream_with_retry_budget(
        build_request,
        Duration::from_secs(RETRY_BUDGET_SECS),
        Duration::from_millis(RETRY_INTERVAL_MS),
    )
    .await
}

async fn send_stream_with_retry_budget(
    build_request: impl Fn() -> RequestBuilder,
    budget: Duration,
    retry_interval: Duration,
) -> anyhow::Result<Response> {
    use anyhow::Context;

    let response = tokio::time::timeout(
        budget,
        send_with_retry_mode(
            build_request,
            budget,
            retry_interval,
            ResponseBody::Streaming,
        ),
    )
    .await
    .context("LLM stream connection deadline exceeded")??;
    Ok(response)
}

async fn send_with_retry_mode(
    build_request: impl Fn() -> RequestBuilder,
    budget: Duration,
    retry_interval: Duration,
    body: ResponseBody,
) -> Result<Response, Error> {
    let deadline = Instant::now() + budget;
    let mut attempt = 0;

    loop {
        attempt += 1;
        let (client, request) = build_request().build_split();
        let mut request = request?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if matches!(body, ResponseBody::Buffered) {
            let timeout = request
                .timeout()
                .copied()
                .unwrap_or(remaining)
                .min(remaining);
            *request.timeout_mut() = Some(timeout);
        }
        let result = client.execute(request).await;

        let wait = match &result {
            Err(error) if error.is_connect() || error.is_timeout() => retry_interval,
            Ok(response)
                if response.status().as_u16() == 429 || response.status().is_server_error() =>
            {
                response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| retry_after_delay(value, Utc::now()))
                    .unwrap_or(retry_interval)
            }
            _ => return result,
        };

        if attempt > MAX_RETRIES || wait >= deadline.saturating_duration_since(Instant::now()) {
            return result;
        }

        metrics::counter!("llm.http.retries").increment(1);
        if let Err(error) = &result {
            metrics::counter!("llm.http.retry_reason.connection").increment(1);
            tracing::warn!(
                attempt,
                wait_ms = wait.as_millis(),
                error = ?error,
                "Connection failed, retrying"
            );
        } else if let Ok(response) = &result {
            let status = response.status();
            if status.as_u16() == 429 {
                metrics::counter!("llm.http.retry_reason.rate_limit").increment(1);
            } else {
                metrics::counter!("llm.http.retry_reason.server_error").increment(1);
            }
            tracing::warn!(
                attempt,
                wait_ms = wait.as_millis(),
                status = status.as_u16(),
                "Retryable response, retrying"
            );
        }

        tokio::time::sleep(wait).await;
        if Instant::now() >= deadline {
            return result;
        }
    }
}

fn retry_after_delay(value: &str, now: DateTime<Utc>) -> Option<Duration> {
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    DateTime::parse_from_rfc2822(value)
        .ok()
        .map(|date| date.signed_duration_since(now).to_std().unwrap_or_default())
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::net::Ipv4Addr;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::task::JoinHandle;
    use url::Url;

    use super::*;

    const OK_RESPONSE: &str = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
    const RETRY_RESPONSE: &str =
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    struct TestResponse {
        delay: Duration,
        response: &'static str,
    }

    struct TestServer {
        url: Url,
        task: JoinHandle<()>,
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn read_request(stream: &mut TcpStream) {
        let mut request = Vec::new();
        let mut bytes = [0; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let count = stream.read(&mut bytes).await.unwrap();
            assert_ne!(count, 0, "request ended before its headers");
            request.extend_from_slice(&bytes[..count]);
        }
    }

    async fn server(responses: Vec<TestResponse>) -> TestServer {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let task = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_request(&mut stream).await;
                tokio::time::sleep(response.delay).await;
                let _ = stream.write_all(response.response.as_bytes()).await;
            }
        });
        TestServer { url, task }
    }

    #[tokio::test]
    async fn retry_budget_bounds_all_attempts() {
        let server = server(vec![
            TestResponse {
                delay: Duration::from_millis(70),
                response: RETRY_RESPONSE,
            },
            TestResponse {
                delay: Duration::from_millis(150),
                response: OK_RESPONSE,
            },
        ])
        .await;
        let client = http_client_builder().no_proxy().build().unwrap();
        let attempts = Cell::new(0);
        let start = Instant::now();
        let error = send_with_retry_budget(
            || {
                attempts.set(attempts.get() + 1);
                client.get(server.url.clone())
            },
            Duration::from_millis(200),
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();
        assert!(error.is_timeout());
        assert_eq!(attempts.get(), 2);
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn request_headers_are_bounded_by_budget() {
        let server = server(vec![TestResponse {
            delay: Duration::from_secs(1),
            response: OK_RESPONSE,
        }])
        .await;
        let client = http_client_builder().no_proxy().build().unwrap();
        let start = Instant::now();
        let error = send_with_retry_budget(
            || client.get(server.url.clone()),
            Duration::from_millis(100),
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();
        assert!(error.is_timeout());
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn shorter_request_timeout_is_preserved() {
        let server = server(vec![TestResponse {
            delay: Duration::from_secs(1),
            response: OK_RESPONSE,
        }])
        .await;
        let client = http_client_builder().no_proxy().build().unwrap();
        let start = Instant::now();
        let error = send_with_retry_budget(
            || {
                client
                    .get(server.url.clone())
                    .timeout(Duration::from_millis(50))
            },
            Duration::from_secs(1),
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert!(error.is_timeout());
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn excessive_retry_after_returns_last_response_without_waiting() {
        let server = server(vec![TestResponse {
            delay: Duration::ZERO,
            response: "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 18446744073709551615\r\nContent-Length: 4\r\nConnection: close\r\n\r\nslow",
        }])
        .await;
        let client = http_client_builder().no_proxy().build().unwrap();
        let start = Instant::now();
        let response = send_with_retry_budget(
            || client.get(server.url.clone()),
            Duration::from_secs(1),
            Duration::from_millis(20),
        )
        .await
        .unwrap();
        assert_eq!(response.status().as_u16(), 429);
        assert_eq!(response.text().await.unwrap(), "slow");
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn active_stream_can_outlive_the_connection_budget() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = TestServer {
            url,
            task: tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_request(&mut stream).await;
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
                for byte in *b"abc" {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let _ = stream.write_all(&[byte]).await;
                }
            }),
        };
        let client = http_client_builder()
            .no_proxy()
            .read_timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        let response = send_stream_with_retry_budget(
            || client.get(server.url.clone()),
            Duration::from_millis(100),
            Duration::from_millis(20),
        )
        .await
        .unwrap();
        assert_eq!(response.text().await.unwrap(), "abc");
    }

    #[tokio::test]
    async fn stream_headers_and_retries_still_have_a_deadline() {
        let server = server(vec![
            TestResponse {
                delay: Duration::from_millis(70),
                response: RETRY_RESPONSE,
            },
            TestResponse {
                delay: Duration::from_secs(1),
                response: OK_RESPONSE,
            },
        ])
        .await;
        let client = client_builder().no_proxy().build().unwrap();
        let attempts = Cell::new(0);
        let start = Instant::now();
        let error = send_stream_with_retry_budget(
            || {
                attempts.set(attempts.get() + 1);
                client.get(server.url.clone())
            },
            Duration::from_millis(200),
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("connection deadline exceeded"));
        assert_eq!(attempts.get(), 2);
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn stalled_response_body_has_read_timeout() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = TestServer {
            url,
            task: tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_request(&mut stream).await;
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx")
                    .await
                    .unwrap();
                tokio::time::sleep(Duration::from_secs(1)).await;
            }),
        };
        let client = http_client_builder()
            .no_proxy()
            .read_timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let response = send_with_retry_budget(
            || client.get(server.url.clone()),
            Duration::from_secs(1),
            Duration::from_millis(20),
        )
        .await
        .unwrap();
        let start = Instant::now();
        let error = response.text().await.unwrap_err();
        assert!(error.is_timeout());
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn retry_after_accepts_dates_and_seconds_without_overflow() {
        let now = DateTime::parse_from_rfc2822("Wed, 21 Oct 2015 07:28:00 GMT")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(retry_after_delay("2", now), Some(Duration::from_secs(2)));
        assert_eq!(
            retry_after_delay("18446744073709551615", now),
            Some(Duration::from_secs(u64::MAX))
        );
        assert_eq!(
            retry_after_delay("Wed, 21 Oct 2015 07:28:03 GMT", now),
            Some(Duration::from_secs(3))
        );
        assert_eq!(
            retry_after_delay("Wed, 21 Oct 2015 07:27:00 GMT", now),
            Some(Duration::ZERO)
        );
        assert_eq!(retry_after_delay("invalid", now), None);
    }
}
