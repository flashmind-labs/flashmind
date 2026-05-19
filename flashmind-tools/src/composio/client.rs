//! HTTP client for the Composio REST API.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tracing::warn;
use url::Url;

use crate::utils::{http_client, send_with_retry};

use super::types::{ComposioToolDef, ExecuteResponse, ToolsListResponse};

const DEFAULT_BASE_URL: &str = "https://backend.composio.dev/api/v3.1";
const PAGE_LIMIT: usize = 100;

/// Thin HTTP client wrapping the Composio v3.1 REST API.
pub struct ComposioClient {
    api_key: String,
    base_url: Url,
    connected_account_id: Option<String>,
    http: reqwest::Client,
}

impl ComposioClient {
    /// Create a new client.
    ///
    /// `base_url` defaults to `https://backend.composio.dev/api/v3.1` when
    /// `None`.
    pub fn new(
        api_key: String,
        connected_account_id: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        let mut base = base_url
            .and_then(|u| Url::parse(&u).ok())
            .unwrap_or_else(|| Url::parse(DEFAULT_BASE_URL).unwrap());
        // Ensure trailing slash so join() works correctly.
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        Self {
            api_key,
            base_url: base,
            connected_account_id,
            http: http_client(),
        }
    }

    fn url(&self, path: &str) -> Url {
        self.base_url.join(path).expect("valid relative path")
    }

    /// Fetch all available tools, optionally filtered by toolkit slugs.
    ///
    /// Handles cursor-based pagination automatically.
    pub async fn list_tools(&self, toolkits: &[String]) -> Result<Vec<ComposioToolDef>> {
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let url = self.url("tools");

            let resp = send_with_retry(|| {
                let mut req = self
                    .http
                    .get(url.clone())
                    .header("x-api-key", &self.api_key)
                    .query(&[("limit", &PAGE_LIMIT.to_string())]);

                for toolkit in toolkits {
                    req = req.query(&[("toolkit_slug", toolkit)]);
                }

                if let Some(c) = &cursor {
                    req = req.query(&[("cursor", c)]);
                }

                req
            })
            .await
            .context("Composio list_tools request failed")?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                bail!("Composio API returned {status}: {body}");
            }

            let page: ToolsListResponse = resp
                .json()
                .await
                .context("failed to parse Composio tools response")?;

            let next = page.next_cursor.clone();

            for raw in page.items {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ComposioToolDef::from(raw)
                })) {
                    Ok(def) => all_tools.push(def),
                    Err(_) => warn!("skipping malformed Composio tool definition"),
                }
            }

            match next {
                Some(c) if !c.is_empty() => cursor = Some(c),
                _ => break,
            }
        }

        Ok(all_tools)
    }

    /// Execute a Composio tool by slug.
    pub(crate) async fn execute_tool(
        &self,
        slug: &str,
        arguments: Value,
    ) -> Result<ExecuteResponse> {
        let url = self.url(&format!("tools/execute/{slug}"));

        let mut body = json!({ "arguments": arguments });
        if let Some(id) = &self.connected_account_id {
            body["connected_account_id"] = json!(id);
        }

        let resp = send_with_retry(|| {
            self.http
                .post(url.clone())
                .header("x-api-key", &self.api_key)
                .json(&body)
        })
        .await
        .context("Composio execute_tool request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let err_body = resp.text().await.unwrap_or_default();
            bail!("Composio execute returned {status}: {err_body}");
        }

        resp.json()
            .await
            .context("failed to parse Composio execute response")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_base_url() {
        let client = ComposioClient::new("key".into(), None, None);
        assert_eq!(client.base_url.as_str(), "https://backend.composio.dev/api/v3.1/");
    }

    #[test]
    fn custom_base_url() {
        let client = ComposioClient::new(
            "key".into(),
            None,
            Some("https://custom.example.com/v1".into()),
        );
        assert_eq!(client.base_url.as_str(), "https://custom.example.com/v1/");
    }

    #[test]
    fn url_join() {
        let client = ComposioClient::new("key".into(), None, None);
        assert_eq!(
            client.url("tools").as_str(),
            "https://backend.composio.dev/api/v3.1/tools"
        );
        assert_eq!(
            client.url("tools/execute/GITHUB_CREATE_ISSUE").as_str(),
            "https://backend.composio.dev/api/v3.1/tools/execute/GITHUB_CREATE_ISSUE"
        );
    }
}
