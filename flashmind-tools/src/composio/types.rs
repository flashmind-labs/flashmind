//! Composio API response types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Tool definition fetched from `GET /tools`.
#[derive(Debug, Clone)]
pub struct ComposioToolDef {
    /// Composio tool slug (e.g. `"GITHUB_CREATE_ISSUE"`).
    pub slug: String,
    /// Human-readable display name.
    pub display_name: Option<String>,
    /// Description shown to the LLM.
    pub description: Option<String>,
    /// JSON Schema for input parameters.
    pub input_schema: Value,
    /// Toolkit this tool belongs to (e.g. `"github"`).
    pub toolkit_slug: Option<String>,
}

/// Nested toolkit reference in a tool definition.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolkitRef {
    pub slug: String,
}

/// Raw tool object from the paginated API response.
#[derive(Debug, Deserialize)]
pub(crate) struct ComposioToolRaw {
    pub slug: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, alias = "inputParameters")]
    pub input_parameters: Value,
    #[serde(default)]
    pub toolkit: Option<ToolkitRef>,
}

impl From<ComposioToolRaw> for ComposioToolDef {
    fn from(raw: ComposioToolRaw) -> Self {
        Self {
            slug: raw.slug,
            display_name: raw.name,
            description: raw.description,
            input_schema: raw.input_parameters,
            toolkit_slug: raw.toolkit.map(|t| t.slug),
        }
    }
}

/// Paginated response from `GET /tools`.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolsListResponse {
    #[serde(default)]
    pub items: Vec<ComposioToolRaw>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// Toolkits
// ---------------------------------------------------------------------------

/// Toolkit (app) available in the Composio catalogue.
#[derive(Debug, Clone)]
pub struct ComposioToolkit {
    /// Toolkit slug (e.g. `"github"`, `"slack"`).
    pub slug: String,
    /// Human-readable name.
    pub name: Option<String>,
    /// Description of what the toolkit provides.
    pub description: Option<String>,
}

/// Raw toolkit object from the paginated API response.
#[derive(Debug, Deserialize)]
pub(crate) struct ComposioToolkitRaw {
    pub slug: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

impl From<ComposioToolkitRaw> for ComposioToolkit {
    fn from(raw: ComposioToolkitRaw) -> Self {
        Self {
            slug: raw.slug,
            name: raw.name,
            description: raw.description,
        }
    }
}

/// Paginated response from `GET /toolkits`.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolkitsListResponse {
    #[serde(default)]
    pub items: Vec<ComposioToolkitRaw>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// Sessions (user connection management)
// ---------------------------------------------------------------------------

/// Request body for `POST /tool_router/session`.
#[derive(Debug, Serialize)]
pub(crate) struct CreateSessionRequest {
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toolkits: Option<SessionToolkits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manage_connections: Option<ManageConnections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_configs: Option<HashMap<String, String>>,
}

/// Toolkit allow/deny configuration for a session.
#[derive(Debug, Clone, Serialize)]
pub struct SessionToolkits {
    /// Only enable these toolkits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable: Option<Vec<String>>,
    /// Disable these toolkits (mutually exclusive with `enable`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable: Option<Vec<String>>,
}

/// Connection management options for a session.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ManageConnections {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
}

/// MCP server info nested inside a session response.
#[derive(Debug, Clone, Deserialize)]
pub struct McpInfo {
    /// MCP server URL scoped to this session.
    pub url: String,
}

/// Session configuration echoed back in the session response.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub user_id: Option<String>,
}

/// A user session returned by the Composio session API.
#[derive(Debug, Clone, Deserialize)]
pub struct ComposioSession {
    /// Unique session identifier.
    pub session_id: String,
    /// MCP server info scoped to this session.
    #[serde(default)]
    pub mcp: Option<McpInfo>,
    /// Tools available in this session.
    #[serde(default)]
    pub tool_router_tools: Vec<String>,
    /// Session configuration (user_id, toolkits, etc.).
    #[serde(default)]
    pub config: Option<SessionConfig>,
    /// OAuth URLs per toolkit for connecting apps.
    /// Populated when the session was created with `manage_connections.enable = true`.
    #[serde(default)]
    pub connection_urls: HashMap<String, String>,
    /// Connected account IDs per toolkit.
    /// Populated when the session was created with `manage_connections.enable = true`.
    #[serde(default)]
    pub connected_accounts: HashMap<String, Vec<String>>,
}

// ---------------------------------------------------------------------------
// Session link (initiate OAuth for a toolkit)
// ---------------------------------------------------------------------------

/// Request body for `POST /tool_router/session/{session_id}/link`.
#[derive(Debug, Serialize)]
pub(crate) struct SessionLinkRequest {
    pub toolkit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
}

/// Response from `POST /tool_router/session/{session_id}/link`.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionLinkResponse {
    pub connected_account_id: String,
    pub redirect_url: String,
    #[serde(default)]
    pub link_token: Option<String>,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Response from `POST /tools/execute/{slug}`.
#[derive(Debug, Deserialize)]
pub struct ExecuteResponse {
    /// Response payload from the executed tool.
    #[serde(default)]
    pub data: Value,
    /// Error message, if the execution failed.
    #[serde(default)]
    pub error: Option<String>,
    /// Whether the execution was successful.
    #[serde(default)]
    pub successful: Option<bool>,
    /// Unique identifier for the execution log (useful for debugging).
    #[serde(default)]
    pub log_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_tools_list() {
        let json = r#"{
            "items": [
                {
                    "slug": "GITHUB_CREATE_ISSUE",
                    "name": "Create Issue",
                    "description": "Create a new GitHub issue",
                    "input_parameters": {
                        "type": "object",
                        "properties": {
                            "owner": { "type": "string" },
                            "repo": { "type": "string" },
                            "title": { "type": "string" }
                        },
                        "required": ["owner", "repo", "title"]
                    },
                    "toolkit": { "slug": "github", "name": "github", "logo": "https://example.com/gh.png" }
                }
            ],
            "next_cursor": "abc123"
        }"#;

        let resp: ToolsListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.items.len(), 1);
        assert_eq!(resp.items[0].slug, "GITHUB_CREATE_ISSUE");
        assert_eq!(resp.items[0].name.as_deref(), Some("Create Issue"));
        assert_eq!(resp.items[0].toolkit.as_ref().unwrap().slug, "github");
        assert_eq!(resp.next_cursor.as_deref(), Some("abc123"));

        let def = ComposioToolDef::from(resp.items.into_iter().next().unwrap());
        assert_eq!(def.slug, "GITHUB_CREATE_ISSUE");
        assert_eq!(def.display_name.as_deref(), Some("Create Issue"));
        assert_eq!(def.toolkit_slug.as_deref(), Some("github"));
        assert_eq!(
            def.input_schema["required"],
            serde_json::json!(["owner", "repo", "title"])
        );
    }

    #[test]
    fn deserialize_execute_response_success() {
        let json = r#"{
            "data": { "id": 42, "url": "https://github.com/org/repo/issues/42" },
            "successful": true
        }"#;

        let resp: ExecuteResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.successful, Some(true));
        assert!(resp.error.is_none());
        assert_eq!(resp.data["id"], 42);
    }

    #[test]
    fn deserialize_toolkits_list() {
        let json = r#"{
            "items": [
                {
                    "slug": "github",
                    "name": "GitHub",
                    "description": "Manage repos, issues, and PRs"
                },
                {
                    "slug": "slack",
                    "name": "Slack"
                }
            ],
            "next_cursor": "xyz789"
        }"#;

        let resp: ToolkitsListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.items.len(), 2);
        assert_eq!(resp.items[0].slug, "github");
        assert_eq!(resp.items[0].name.as_deref(), Some("GitHub"));
        assert_eq!(
            resp.items[0].description.as_deref(),
            Some("Manage repos, issues, and PRs")
        );
        assert_eq!(resp.items[1].slug, "slack");
        assert!(resp.items[1].description.is_none());
        assert_eq!(resp.next_cursor.as_deref(), Some("xyz789"));

        let toolkit = ComposioToolkit::from(resp.items.into_iter().next().unwrap());
        assert_eq!(toolkit.slug, "github");
        assert_eq!(toolkit.name.as_deref(), Some("GitHub"));
    }

    #[test]
    fn deserialize_session() {
        let json = r#"{
            "session_id": "trs_abc123",
            "mcp": { "type": "http", "url": "https://app.composio.dev/tool_router/v3/trs_abc123/mcp" },
            "tool_router_tools": ["GITHUB_CREATE_AN_ISSUE", "GITHUB_LIST_REPO_ISSUES"],
            "config": { "user_id": "user_42" },
            "connection_urls": {
                "github": "https://connect.composio.dev/link/ln_gh_xyz",
                "slack": "https://connect.composio.dev/link/ln_sl_xyz"
            },
            "connected_accounts": {
                "gmail": ["ca_gmail_001"]
            }
        }"#;

        let session: ComposioSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.session_id, "trs_abc123");
        assert_eq!(
            session.mcp.as_ref().unwrap().url,
            "https://app.composio.dev/tool_router/v3/trs_abc123/mcp"
        );
        assert_eq!(session.tool_router_tools.len(), 2);
        assert_eq!(
            session.config.as_ref().unwrap().user_id.as_deref(),
            Some("user_42")
        );
        assert_eq!(session.connection_urls.len(), 2);
        assert!(session.connection_urls.contains_key("github"));
        assert_eq!(session.connected_accounts["gmail"], vec!["ca_gmail_001"]);
    }

    #[test]
    fn serialize_create_session_request() {
        let req = CreateSessionRequest {
            user_id: "user_42".into(),
            toolkits: Some(SessionToolkits {
                enable: Some(vec!["github".into()]),
                disable: None,
            }),
            manage_connections: Some(ManageConnections {
                enable: Some(true),
                callback_url: Some("https://myapp.com/callback".into()),
            }),
            auth_configs: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["user_id"], "user_42");
        assert_eq!(json["toolkits"]["enable"][0], "github");
        assert!(json["toolkits"].get("disable").is_none());
        assert_eq!(
            json["manage_connections"]["callback_url"],
            "https://myapp.com/callback"
        );
        assert!(json.get("auth_configs").is_none());
    }

    #[test]
    fn deserialize_execute_response_error() {
        let json = r#"{
            "data": {},
            "error": "Authentication required",
            "successful": false
        }"#;

        let resp: ExecuteResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.successful, Some(false));
        assert_eq!(resp.error.as_deref(), Some("Authentication required"));
    }
}
