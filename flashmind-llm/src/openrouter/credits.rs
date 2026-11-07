use std::time::Duration;

use anyhow::{Context, Result, bail};
use flashmind_types::{CreditBalance, CreditScope, CreditUnit};
use rust_decimal::Decimal;
use serde::Deserialize;

use super::OpenRouterProvider;

#[derive(Deserialize)]
struct Envelope<T> {
    data: T,
}

#[derive(Deserialize)]
struct Key {
    limit: Option<Decimal>,
    limit_remaining: Option<Decimal>,
    limit_reset: Option<String>,
    #[serde(default)]
    is_management_key: bool,
    effective_budget: Option<Budget>,
}

#[derive(Deserialize)]
struct Budget {
    limit_usd: Decimal,
    remaining_usd: Decimal,
    spend_usd: Decimal,
    reset_interval: Option<String>,
}

#[derive(Deserialize)]
struct Account {
    total_credits: Decimal,
    total_usage: Decimal,
}

impl OpenRouterProvider {
    async fn credit_request<T: serde::de::DeserializeOwned>(&self, url: String) -> Result<T> {
        let response = self
            .client
            .get(url)
            .bearer_auth(&self.api_key)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(reqwest::Error::without_url)
            .context("OpenRouter credit request failed")?;
        if !response.status().is_success() {
            bail!(
                "OpenRouter credit request returned HTTP {}",
                response.status()
            );
        }
        let response: Envelope<T> = response
            .json()
            .await
            .map_err(reqwest::Error::without_url)
            .context("invalid OpenRouter credit response")?;
        Ok(response.data)
    }

    pub(super) async fn fetch_credit_balance(&self, base: &str) -> Result<CreditBalance> {
        let key: Key = self.credit_request(format!("{base}key")).await?;
        if let Some(budget) = key.effective_budget {
            return Ok(CreditBalance {
                scope: CreditScope::Budget,
                unit: CreditUnit::Usd,
                remaining: Some(budget.remaining_usd),
                total: Some(budget.limit_usd),
                used: Some(budget.spend_usd),
                resets: budget.reset_interval,
            });
        }
        if key.is_management_key {
            let account: Account = self.credit_request(format!("{base}credits")).await?;
            return Ok(CreditBalance {
                scope: CreditScope::Account,
                unit: CreditUnit::Usd,
                remaining: account.total_credits.checked_sub(account.total_usage),
                total: Some(account.total_credits),
                used: Some(account.total_usage),
                resets: None,
            });
        }
        Ok(CreditBalance {
            scope: CreditScope::ApiKey,
            unit: CreditUnit::Usd,
            remaining: key.limit_remaining,
            total: key.limit,
            // Lifetime usage cannot be subtracted from a recurring limit.
            used: key
                .limit
                .zip(key.limit_remaining)
                .and_then(|(total, remaining)| total.checked_sub(remaining)),
            resets: key.limit_reset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn balance(responses: &[(&str, &str, &str)]) -> Result<CreditBalance> {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        let base = format!("http://{}/api/v1/", listener.local_addr()?);
        let server = async {
            for (path, status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    headers.push(socket.read_u8().await.unwrap());
                }
                let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
                assert!(headers.starts_with(&format!("get /api/v1/{path} http/1.1\r\n")));
                assert!(headers.contains("authorization: bearer example-key\r\n"));
                socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        };
        let provider = OpenRouterProvider::new("example-key".into());
        let (result, ()) = tokio::join!(provider.fetch_credit_balance(&base), server);
        result
    }

    #[tokio::test]
    async fn recurring_limit_uses_remaining_instead_of_lifetime_usage() {
        let result = balance(&[("key", "200 OK", r#"{"data":{"limit":10,"limit_remaining":7.123456,"usage":500,"limit_reset":"monthly"}}"#)]).await.unwrap();
        assert_eq!(result.scope, CreditScope::ApiKey);
        assert_eq!(result.remaining, Some(Decimal::new(7123456, 6)));
        assert_eq!(result.used, Some(Decimal::new(2876544, 6)));
        assert_eq!(result.total, Some(Decimal::TEN));
        assert_eq!(result.resets.as_deref(), Some("monthly"));
    }

    #[tokio::test]
    async fn uncapped_inference_key_does_not_claim_an_account_balance() {
        let result = balance(&[(
            "key",
            "200 OK",
            r#"{"data":{"limit":null,"limit_remaining":null,"usage":42}}"#,
        )])
        .await
        .unwrap();
        assert_eq!(result.remaining, None);
        assert_eq!(result.total, None);
        assert_eq!(result.used, None);
    }

    #[tokio::test]
    async fn effective_budget_takes_precedence_over_key_limit() {
        let result = balance(&[("key", "200 OK", r#"{"data":{"limit":100,"limit_remaining":90,"effective_budget":{"limit_usd":20,"remaining_usd":-0.5,"spend_usd":20.5,"reset_interval":"weekly"}}}"#)]).await.unwrap();
        assert_eq!(result.scope, CreditScope::Budget);
        assert_eq!(result.remaining, Some(Decimal::new(-5, 1)));
        assert_eq!(result.total, Some(Decimal::from(20)));
    }

    #[tokio::test]
    async fn management_key_can_read_account_credits() {
        let result = balance(&[
            ("key", "200 OK", r#"{"data":{"is_management_key":true}}"#),
            (
                "credits",
                "200 OK",
                r#"{"data":{"total_credits":100.5,"total_usage":25.75}}"#,
            ),
        ])
        .await
        .unwrap();
        assert_eq!(result.scope, CreditScope::Account);
        assert_eq!(result.remaining, Some(Decimal::new(7475, 2)));
        assert_eq!(result.total, Some(Decimal::new(1005, 1)));
    }

    #[tokio::test]
    async fn malformed_or_denied_balance_is_an_error_without_response_body() {
        let error = balance(&[("key", "403 Forbidden", "example-sensitive-body")])
            .await
            .unwrap_err();
        assert!(!error.to_string().contains("example-sensitive-body"));
        assert!(
            balance(&[("key", "200 OK", r#"{"data":{"limit":"invalid"}}"#)])
                .await
                .is_err()
        );
    }
}
