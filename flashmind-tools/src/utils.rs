//! Shared utilities for tool implementations.
//!
//! Mirrors `src/utils/` from the binary crate so tools can use the same
//! HTTP client, retry logic, and string helpers without a binary dependency.

use std::sync::LazyLock;
use std::time::Duration;

use reqwest::{Client, RequestBuilder, Response};

pub const MAX_RETRIES: u32 = 30;
const RETRY_INTERVAL_MS: u64 = 2000;
const RETRY_BUDGET_SECS: u64 = 120;

fn tls_config() -> rustls::ClientConfig {
    let root_store =
        rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth()
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

/// Create a pre-configured reqwest client builder with rustls TLS.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    Client::builder()
        .use_preconfigured_tls(tls_config())
        .user_agent("Flash/1.0 (Flashmind Labs)")
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(60))
}

/// Get a persistent (connection-pooled) HTTP client for repeated requests.
pub fn persistent_client() -> Client {
    http_client_builder()
        .pool_idle_timeout(Duration::from_secs(86400 * 7))
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .build()
        .expect("failed to build persistent HTTP client")
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

/// Response from [`send_with_retry_inspecting`] when the body was pre-read.
pub enum RetryOutcome {
    /// Normal response (body not consumed).
    Response(Response),
    /// The server returned an error whose body matched the abort predicate.
    /// The body has already been consumed and is returned here.
    Aborted {
        status: reqwest::StatusCode,
        body: String,
    },
}

/// Like [`send_with_retry`] but on server errors (5xx), reads the response
/// body and passes it to `abort_if`.  When the predicate returns `true`,
/// retries stop immediately and the pre-read body is returned as
/// [`RetryOutcome::Aborted`] so the caller doesn't have to re-fetch it.
///
/// Non-aborted responses (success, 4xx, exhausted retries) come back as
/// [`RetryOutcome::Response`] with the body unconsumed.
pub async fn send_with_retry_inspecting(
    build_request: impl Fn() -> RequestBuilder,
    abort_if: impl Fn(&str) -> bool,
) -> Result<RetryOutcome, reqwest::Error> {
    let start = std::time::Instant::now();
    let mut attempt = 0;

    loop {
        attempt += 1;
        let result = build_request().send().await;

        if let Err(e) = &result
            && (e.is_connect() || e.is_timeout())
        {
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

        let over_budget = start.elapsed().as_secs() >= RETRY_BUDGET_SECS;
        let retryable = !over_budget
            && (status.as_u16() == 429 || status.is_server_error())
            && attempt <= MAX_RETRIES;

        if retryable || status.is_server_error() {
            // Buffer body so we can inspect it before deciding to retry
            let body = response.text().await.unwrap_or_default();

            if status.is_server_error() && abort_if(&body) {
                return Ok(RetryOutcome::Aborted { status, body });
            }

            if retryable {
                tracing::warn!(
                    attempt,
                    wait_ms = RETRY_INTERVAL_MS,
                    status = status.as_u16(),
                    "Retryable error ({}), retrying",
                    status
                );
                tokio::time::sleep(Duration::from_millis(RETRY_INTERVAL_MS)).await;
                continue;
            }

            // Exhausted retries on server error — return the pre-read body
            return Ok(RetryOutcome::Aborted { status, body });
        }

        return Ok(RetryOutcome::Response(response));
    }
}

/// Truncate a string to at most `max` bytes, landing on a valid UTF-8 char boundary.
pub fn truncate_utf8(s: &str, max: usize) -> &str {
    &s[..s.floor_char_boundary(max)]
}

/// Strip ANSI escape codes from a string.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }

        match chars.next() {
            Some('[') => {
                for ch in chars.by_ref() {
                    if ch.is_ascii_alphabetic() || ch == '~' || ch == '@' {
                        break;
                    }
                }
            }
            Some(']') => {
                for ch in chars.by_ref() {
                    if ch == '\x07' {
                        break;
                    }
                    if ch == '\x1b' {
                        chars.next();
                        break;
                    }
                }
            }
            Some(ch) if ch.is_ascii_alphabetic() => {}
            _ => {}
        }
    }

    out
}

/// Create a rate limiter with the given requests-per-minute limit.
pub fn create_rate_limiter(rpm: u32) -> std::sync::Arc<ratelimit::Ratelimiter> {
    tracing::debug!(rpm, "creating LLM rate limiter");

    std::sync::Arc::new(
        ratelimit::Ratelimiter::builder(rpm as u64)
            .max_tokens(rpm as u64)
            .initial_available(rpm as u64)
            .build()
            .expect("LLM rate limiter"),
    )
}
