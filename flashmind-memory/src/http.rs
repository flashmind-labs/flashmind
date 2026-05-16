//! HTTP utilities including retry logic with exponential backoff.

use reqwest::{RequestBuilder, Response};
use std::time::Duration;

const MAX_RETRIES: u32 = 10;
const INITIAL_BACKOFF_MS: u64 = 1000;
const RETRY_BUDGET_SECS: u64 = 60;

/// Create a pre-configured reqwest client builder for embedding provider HTTP requests.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder().user_agent("Flash/1.0 (Flashmind Labs)")
}

/// Send an HTTP request with automatic retry on transient failures.
///
/// Retries on connection errors, timeouts, 429 rate limits, and 5xx server errors.
/// Uses exponential backoff (1s, 2s, 4s...) or respects `Retry-After` header.
pub async fn send_with_retry(
    build_request: impl Fn() -> RequestBuilder,
) -> Result<Response, reqwest::Error> {
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
                let wait_ms = INITIAL_BACKOFF_MS * (1 << (attempt - 1));
                tracing::warn!(attempt, wait_ms, error = ?e, "Connection failed, retrying");
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                continue;
            }
        }

        let response = result?;
        let status = response.status();

        let over_budget = start.elapsed().as_secs() >= RETRY_BUDGET_SECS;
        let should_retry = !over_budget
            && (status.as_u16() == 429 || status.is_server_error())
            && attempt <= MAX_RETRIES;

        if should_retry {
            let wait_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or_else(|| INITIAL_BACKOFF_MS * (1 << (attempt - 1)) / 1000)
                .max(1);

            tracing::warn!(
                attempt,
                wait_secs,
                status = status.as_u16(),
                "Retryable error, backoff"
            );
            tokio::time::sleep(Duration::from_secs(wait_secs)).await;
            continue;
        }

        return Ok(response);
    }
}
