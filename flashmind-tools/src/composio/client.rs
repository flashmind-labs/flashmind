//! HTTP client for the Composio REST API.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use reqwest::{Method, RequestBuilder, Response};
use serde_json::{Value, json};
use url::Url;

use crate::utils::{RetryOutcome, http_client, send_with_retry, send_with_retry_inspecting};

use super::types::{
    AuthConfigInfo, AuthConfigsListResponse, ComposioPage, ComposioSession, ComposioToolDef,
    ComposioToolkit, ComposioTriggerInstance, ComposioTriggerType, ConnectedAccountInfo,
    ConnectedAccountsListResponse, CreateSessionRequest, ExecuteResponse, ManageConnections,
    SessionLinkRequest, SessionLinkResponse, SessionToolkits, ToolkitsListResponse,
    ToolsListResponse, TriggerInstancesListResponse, TriggerLogsRequest, TriggerLogsResponse,
    TriggerTypeRaw, TriggerTypesListResponse, TriggerUpsertRequest, TriggerUpsertResponse,
};

const DEFAULT_BASE_URL: &str = "https://backend.composio.dev/api/v3.1";
const PAGE_LIMIT: usize = 100;

/// Thin HTTP client wrapping the Composio v3.1 REST API.
pub struct ComposioClient {
    api_key: String,
    base_url: Url,
    connected_account_id: Option<String>,
    entity_id: Option<String>,
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
            entity_id: None,
            http: http_client(),
        }
    }

    /// Set the entity ID (your platform's user identifier) sent with execute requests.
    pub fn with_entity_id(mut self, entity_id: String) -> Self {
        self.entity_id = Some(entity_id);
        self
    }

    /// Returns the API key this client was created with.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    fn url(&self, path: &str) -> Url {
        self.base_url.join(path).expect("valid relative path")
    }

    fn redact(&self, text: String) -> String {
        if !self.api_key.is_empty() && text.contains(&self.api_key) {
            text.replace(&self.api_key, "[REDACTED]")
        } else {
            text
        }
    }

    fn request(&self, method: Method, url: &Url) -> RequestBuilder {
        self.http
            .request(method, url.clone())
            .header("x-api-key", &self.api_key)
    }

    /// Return `resp` unchanged on success, otherwise bail with the redacted body
    /// as `Composio {label} returned {status}: {body}`.
    async fn check(&self, resp: Response, label: &str) -> Result<Response> {
        let status = resp.status();
        if !status.is_success() {
            let body = self.redact(resp.text().await.unwrap_or_default());
            bail!("Composio {label} returned {status}: {body}");
        }
        Ok(resp)
    }

    /// Follow cursor pagination on a tools endpoint and collect every tool.
    async fn list_tool_pages(
        &self,
        url: Url,
        toolkits: &[String],
        tags: &[&str],
        labels: PageLabels,
    ) -> Result<Vec<ComposioToolDef>> {
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let resp = send_with_retry(|| {
                let mut req = self
                    .request(Method::GET, &url)
                    .query(&[("limit", &PAGE_LIMIT.to_string())]);

                for toolkit in toolkits {
                    req = req.query(&[("toolkit_slug", toolkit)]);
                }

                for tag in tags {
                    req = req.query(&[("tags", tag)]);
                }

                if let Some(c) = &cursor {
                    req = req.query(&[("cursor", c)]);
                }

                req
            })
            .await
            .context(labels.request)?;

            let page: ToolsListResponse = self
                .check(resp, labels.status)
                .await?
                .json()
                .await
                .context(labels.parse)?;

            all_tools.extend(page.items.into_iter().map(ComposioToolDef::from));

            match page.next_cursor {
                Some(c) if !c.is_empty() => cursor = Some(c),
                _ => break,
            }
        }

        Ok(all_tools)
    }

    /// Fetch available tools, optionally filtered by toolkit slugs and tags.
    ///
    /// Handles cursor-based pagination automatically.
    pub async fn list_tools(&self, toolkits: &[String]) -> Result<Vec<ComposioToolDef>> {
        self.list_tools_tagged(toolkits, &[]).await
    }

    /// Fetch available tools filtered by toolkit slugs and tags (e.g. `"important"`).
    pub async fn list_tools_tagged(
        &self,
        toolkits: &[String],
        tags: &[&str],
    ) -> Result<Vec<ComposioToolDef>> {
        self.list_tool_pages(
            self.url("tools"),
            toolkits,
            tags,
            PageLabels {
                request: "Composio list_tools request failed",
                status: "API",
                parse: "failed to parse Composio tools response",
            },
        )
        .await
    }

    /// Fetch all available toolkits (apps) from the Composio catalogue.
    ///
    /// Handles cursor-based pagination automatically.
    pub async fn list_toolkits(&self) -> Result<Vec<ComposioToolkit>> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        let url = self.url("toolkits");

        loop {
            let resp = send_with_retry(|| {
                let mut req = self
                    .request(Method::GET, &url)
                    .query(&[("limit", &PAGE_LIMIT.to_string())]);

                if let Some(c) = &cursor {
                    req = req.query(&[("cursor", c)]);
                }

                req
            })
            .await
            .context("Composio list_toolkits request failed")?;

            let page: ToolkitsListResponse = self
                .check(resp, "API")
                .await?
                .json()
                .await
                .context("failed to parse Composio toolkits response")?;

            all.extend(page.items.into_iter().map(ComposioToolkit::from));

            match page.next_cursor {
                Some(c) if !c.is_empty() => cursor = Some(c),
                _ => break,
            }
        }

        Ok(all)
    }

    /// Fetch auth configs, optionally filtered by toolkit slug.
    ///
    /// Returns auth config items with `id`, `toolkit_slug`, and `type`
    /// ("default" or "custom").
    pub async fn list_auth_configs(&self, toolkit: Option<&str>) -> Result<Vec<AuthConfigInfo>> {
        let mut url = self.url("auth_configs");
        if let Some(tk) = toolkit {
            url.query_pairs_mut().append_pair("toolkit_slug", tk);
        }

        let resp = send_with_retry(|| self.request(Method::GET, &url))
            .await
            .context("Composio list_auth_configs request failed")?;

        let resp: AuthConfigsListResponse = self
            .check(resp, "list_auth_configs")
            .await?
            .json()
            .await
            .context("failed to parse Composio auth configs response")?;
        Ok(resp.items)
    }

    /// Create a session for a user, returning OAuth URLs for connecting apps.
    ///
    /// The returned [`ComposioSession`] contains:
    /// - `session_id` — for polling via [`get_session`](Self::get_session)
    /// - `mcp` — MCP server info scoped to this session
    /// - `tool_router_tools` — available tool slugs
    /// - `connection_urls` — OAuth URLs per toolkit (when `manage_connections` is enabled)
    /// - `connected_accounts` — already-connected account IDs per toolkit
    ///
    /// `toolkits` optionally restricts which apps are available. `callback_url`
    /// is where the user is redirected after completing OAuth (Composio appends
    /// `?status=success&connected_account_id=...`).
    pub async fn create_session(
        &self,
        user_id: &str,
        toolkits: Option<SessionToolkits>,
        callback_url: Option<&str>,
        auth_configs: Option<HashMap<String, String>>,
    ) -> Result<ComposioSession> {
        let url = self.url("tool_router/session");

        let body = CreateSessionRequest {
            user_id: user_id.into(),
            toolkits,
            manage_connections: Some(ManageConnections {
                enable: Some(true),
                callback_url: callback_url.map(Into::into),
            }),
            auth_configs,
        };

        let resp = send_with_retry(|| self.request(Method::POST, &url).json(&body))
            .await
            .context("Composio create_session request failed")?;

        self.check(resp, "create_session")
            .await?
            .json()
            .await
            .context("failed to parse Composio session response")
    }

    /// Initiate OAuth for a toolkit within a session.
    ///
    /// Calls `POST /tool_router/session/{session_id}/link` to get an OAuth
    /// redirect URL for the given toolkit.
    pub async fn session_link(
        &self,
        session_id: &str,
        toolkit: &str,
        callback_url: Option<&str>,
    ) -> Result<SessionLinkResponse> {
        let url = self.url(&format!("tool_router/session/{session_id}/link"));

        let body = SessionLinkRequest {
            toolkit: toolkit.into(),
            callback_url: callback_url.map(Into::into),
        };

        let resp = send_with_retry(|| self.request(Method::POST, &url).json(&body))
            .await
            .context("Composio session_link request failed")?;

        self.check(resp, "session_link")
            .await?
            .json()
            .await
            .context("failed to parse Composio session_link response")
    }

    /// Poll a session to check which apps the user has connected.
    ///
    /// After redirecting a user to the OAuth URLs from [`create_session`](Self::create_session),
    /// call this to check whether they've completed authorization. The returned
    /// session's `connected_accounts` map will be populated as apps are connected.
    pub async fn get_session(&self, session_id: &str) -> Result<ComposioSession> {
        let url = self.url(&format!("tool_router/session/{session_id}"));

        let resp = send_with_retry(|| self.request(Method::GET, &url))
            .await
            .context("Composio get_session request failed")?;

        self.check(resp, "get_session")
            .await?
            .json()
            .await
            .context("failed to parse Composio session response")
    }

    /// Fetch tools scoped to a session.
    ///
    /// Calls `GET /tool_router/session/{session_id}/tools` which returns only
    /// the tools available within the given session. Handles cursor-based
    /// pagination automatically.
    pub async fn list_session_tools(&self, session_id: &str) -> Result<Vec<ComposioToolDef>> {
        self.list_tool_pages(
            self.url(&format!("tool_router/session/{session_id}/tools")),
            &[],
            &[],
            PageLabels {
                request: "Composio list_session_tools request failed",
                status: "list_session_tools",
                parse: "failed to parse Composio session tools response",
            },
        )
        .await
    }

    /// Execute a tool within a session context.
    ///
    /// Calls `POST /tool_router/session/{session_id}/execute`. The session
    /// handles connected account resolution, so no explicit
    /// `connected_account_id` is needed.
    pub async fn execute_tool_in_session(
        &self,
        session_id: &str,
        slug: &str,
        arguments: Value,
    ) -> Result<ExecuteResponse> {
        let url = self.url(&format!("tool_router/session/{session_id}/execute"));

        let body = json!({
            "tool_slug": slug,
            "arguments": arguments,
        });

        let resp = send_with_retry(|| self.request(Method::POST, &url).json(&body))
            .await
            .context("Composio execute_tool_in_session request failed")?;

        self.check(resp, "session execute")
            .await?
            .json()
            .await
            .context("failed to parse Composio session execute response")
    }

    // ---------------------------------------------------------------------------
    // Triggers
    // ---------------------------------------------------------------------------

    /// Fetch a page of available trigger types, optionally filtered by toolkit slugs.
    ///
    /// Pass `cursor` from a previous response's [`ComposioPage::next_cursor`] to
    /// fetch subsequent pages, or `None` for the first page.
    pub async fn list_trigger_types(
        &self,
        toolkits: &[String],
        cursor: Option<&str>,
    ) -> Result<ComposioPage<ComposioTriggerType>> {
        let url = self.url("triggers_types");

        let resp = send_with_retry(|| {
            let mut req = self
                .request(Method::GET, &url)
                .query(&[("limit", &PAGE_LIMIT.to_string())]);

            for toolkit in toolkits {
                req = req.query(&[("toolkit_slug", toolkit)]);
            }

            if let Some(c) = cursor {
                req = req.query(&[("cursor", c)]);
            }

            req
        })
        .await
        .context("Composio list_trigger_types request failed")?;

        let page: TriggerTypesListResponse = self
            .check(resp, "API")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger types response")?;

        let items = page
            .items
            .into_iter()
            .map(ComposioTriggerType::from)
            .collect();
        let next_cursor = page.next_cursor.filter(|c| !c.is_empty());
        Ok(ComposioPage { items, next_cursor })
    }

    /// Fetch a single trigger type by slug.
    pub async fn get_trigger_type(&self, slug: &str) -> Result<ComposioTriggerType> {
        let url = self.url(&format!("triggers_types/{slug}"));

        let resp = send_with_retry(|| self.request(Method::GET, &url))
            .await
            .context("Composio get_trigger_type request failed")?;

        let raw: TriggerTypeRaw = self
            .check(resp, "API")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger type response")?;

        Ok(ComposioTriggerType::from(raw))
    }

    /// Create or update a trigger instance.
    ///
    /// `slug` is the trigger type (e.g. `"GITHUB_COMMIT_EVENT"`).
    /// `config` contains trigger-specific parameters (e.g. `{"owner": "org", "repo": "myrepo"}`).
    pub async fn create_trigger(
        &self,
        slug: &str,
        connected_account_id: &str,
        config: Value,
    ) -> Result<TriggerUpsertResponse> {
        let url = self.url(&format!("trigger_instances/{slug}/upsert"));

        let body = TriggerUpsertRequest {
            connected_account_id: connected_account_id.into(),
            trigger_config: config,
        };

        let resp = send_with_retry(|| self.request(Method::POST, &url).json(&body))
            .await
            .context("Composio create_trigger request failed")?;

        self.check(resp, "create_trigger")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger upsert response")
    }

    /// Fetch a page of trigger instances.
    ///
    /// When `include_disabled` is `true`, disabled triggers are included.
    /// Pass `cursor` from a previous response's [`ComposioPage::next_cursor`] to
    /// fetch subsequent pages, or `None` for the first page.
    pub async fn list_triggers(
        &self,
        include_disabled: bool,
        cursor: Option<&str>,
    ) -> Result<ComposioPage<ComposioTriggerInstance>> {
        let url = self.url("trigger_instances/active");

        let resp = send_with_retry(|| {
            let mut req = self.request(Method::GET, &url);

            if include_disabled {
                req = req.query(&[("include_disabled", "true")]);
            }

            if let Some(c) = cursor {
                req = req.query(&[("cursor", c)]);
            }

            req
        })
        .await
        .context("Composio list_triggers request failed")?;

        let page: TriggerInstancesListResponse = self
            .check(resp, "API")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger instances response")?;

        let next_cursor = page.next_cursor.filter(|c| !c.is_empty());
        Ok(ComposioPage {
            items: page.items,
            next_cursor,
        })
    }

    /// Fetch a single trigger instance by ID.
    pub async fn get_trigger(&self, trigger_id: &str) -> Result<ComposioTriggerInstance> {
        let url = self.url(&format!("trigger_instances/{trigger_id}"));

        let resp = send_with_retry(|| self.request(Method::GET, &url))
            .await
            .context("Composio get_trigger request failed")?;

        self.check(resp, "API")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger instance response")
    }

    /// Enable a disabled trigger instance.
    pub async fn enable_trigger(&self, trigger_id: &str) -> Result<()> {
        let url = self.url(&format!("trigger_instances/manage/{trigger_id}"));

        let resp = send_with_retry(|| {
            self.request(Method::PATCH, &url)
                .json(&json!({"status": "enable"}))
        })
        .await
        .context("Composio enable_trigger request failed")?;

        self.check(resp, "enable_trigger").await?;
        Ok(())
    }

    /// Disable an active trigger instance.
    pub async fn disable_trigger(&self, trigger_id: &str) -> Result<()> {
        let url = self.url(&format!("trigger_instances/manage/{trigger_id}"));

        let resp = send_with_retry(|| {
            self.request(Method::PATCH, &url)
                .json(&json!({"status": "disable"}))
        })
        .await
        .context("Composio disable_trigger request failed")?;

        self.check(resp, "disable_trigger").await?;
        Ok(())
    }

    /// Delete a trigger instance.
    pub async fn delete_trigger(&self, trigger_id: &str) -> Result<()> {
        let url = self.url(&format!("trigger_instances/{trigger_id}"));

        let resp = send_with_retry(|| self.request(Method::DELETE, &url))
            .await
            .context("Composio delete_trigger request failed")?;

        self.check(resp, "delete_trigger").await?;
        Ok(())
    }

    /// Retrieve trigger event logs.
    ///
    /// Use [`TriggerLogsRequest`] to filter by time range, status, and pagination.
    /// Set `include_payload` to `true` to get the full event payload (slower).
    pub async fn get_trigger_logs(
        &self,
        request: TriggerLogsRequest,
    ) -> Result<TriggerLogsResponse> {
        let url = self.url("internal/trigger/logs");

        let resp = send_with_retry(|| self.request(Method::POST, &url).json(&request))
            .await
            .context("Composio get_trigger_logs request failed")?;

        self.check(resp, "get_trigger_logs")
            .await?
            .json()
            .await
            .context("failed to parse Composio trigger logs response")
    }

    // ---------------------------------------------------------------------------
    // Connected accounts
    // ---------------------------------------------------------------------------

    /// List connected accounts, optionally filtered by toolkit slug.
    pub async fn list_connected_accounts(
        &self,
        toolkit: Option<&str>,
    ) -> Result<Vec<ConnectedAccountInfo>> {
        self.list_connected_accounts_for_user(toolkit, None).await
    }

    /// List connected accounts, optionally filtered by toolkit and/or user ID.
    pub async fn list_connected_accounts_for_user(
        &self,
        toolkit: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<ConnectedAccountInfo>> {
        let mut url = self.url("connected_accounts");
        if let Some(tk) = toolkit {
            url.query_pairs_mut().append_pair("toolkit_slug", tk);
        }
        if let Some(uid) = user_id {
            url.query_pairs_mut().append_pair("user_id", uid);
        }

        let resp = send_with_retry(|| self.request(Method::GET, &url))
            .await
            .context("Composio list_connected_accounts request failed")?;

        let resp: ConnectedAccountsListResponse = self
            .check(resp, "list_connected_accounts")
            .await?
            .json()
            .await
            .context("failed to parse Composio connected accounts response")?;
        Ok(resp.items)
    }

    /// Delete a connected account by its ID.
    pub async fn delete_connected_account(&self, account_id: &str) -> Result<()> {
        let url = self.url(&format!("connected_accounts/{account_id}"));

        let resp = send_with_retry(|| self.request(Method::DELETE, &url))
            .await
            .context("Composio delete_connected_account request failed")?;

        self.check(resp, "delete_connected_account").await?;
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Tool execution
    // ---------------------------------------------------------------------------

    /// Execute a Composio tool by slug.
    ///
    /// Uses [`send_with_retry_inspecting`] to bail immediately on non-transient
    /// server errors (e.g. expired/corrupted connected accounts) instead of
    /// burning the full retry budget.
    pub async fn execute_tool(&self, slug: &str, arguments: Value) -> Result<ExecuteResponse> {
        let url = self.url(&format!("tools/execute/{slug}"));

        let mut body = json!({ "arguments": arguments });
        if let Some(id) = &self.connected_account_id {
            body["connected_account_id"] = json!(id);
        }
        if let Some(id) = &self.entity_id {
            body["entity_id"] = json!(id);
        }

        let outcome = send_with_retry_inspecting(
            || self.request(Method::POST, &url).json(&body),
            is_non_transient_composio_error,
        )
        .await
        .context("Composio execute_tool request failed")?;

        match outcome {
            RetryOutcome::Aborted { status, body } => {
                let body = self.redact(body);
                bail!("Composio execute returned {status}: {body}");
            }
            RetryOutcome::Response(resp) => self
                .check(resp, "execute")
                .await?
                .json()
                .await
                .context("failed to parse Composio execute response"),
        }
    }
}

/// Error-context strings for one paginated tools endpoint.
struct PageLabels {
    /// Context attached when the request itself fails.
    request: &'static str,
    /// Label in the `Composio {label} returned {status}` error.
    status: &'static str,
    /// Context attached when a page fails to parse.
    parse: &'static str,
}

/// Composio error slugs that will never succeed on retry (corrupted tokens,
/// revoked access, etc.).  When one of these appears in a 5xx body we bail
/// immediately instead of burning the full retry budget.
fn is_non_transient_composio_error(body: &str) -> bool {
    const NON_TRANSIENT: &[&str] = &[
        "ConnectedAccount_InternalServerError",
        "Error decrypting connected account data",
    ];
    NON_TRANSIENT.iter().any(|slug| body.contains(slug))
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
        assert_eq!(
            client.base_url.as_str(),
            "https://backend.composio.dev/api/v3.1/"
        );
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
        assert_eq!(
            client.url("toolkits").as_str(),
            "https://backend.composio.dev/api/v3.1/toolkits"
        );
        assert_eq!(
            client.url("tool_router/session").as_str(),
            "https://backend.composio.dev/api/v3.1/tool_router/session"
        );
        assert_eq!(
            client.url("tool_router/session/sess_abc").as_str(),
            "https://backend.composio.dev/api/v3.1/tool_router/session/sess_abc"
        );
    }

    #[test]
    fn trigger_urls() {
        let client = ComposioClient::new("key".into(), None, None);
        assert_eq!(
            client.url("triggers_types").as_str(),
            "https://backend.composio.dev/api/v3.1/triggers_types"
        );
        assert_eq!(
            client
                .url("trigger_instances/GITHUB_COMMIT_EVENT/upsert")
                .as_str(),
            "https://backend.composio.dev/api/v3.1/trigger_instances/GITHUB_COMMIT_EVENT/upsert"
        );
        assert_eq!(
            client.url("trigger_instances/active").as_str(),
            "https://backend.composio.dev/api/v3.1/trigger_instances/active"
        );
        assert_eq!(
            client.url("trigger_instances/ti_abc123").as_str(),
            "https://backend.composio.dev/api/v3.1/trigger_instances/ti_abc123"
        );
    }
}
