//! Composio API response types.

use serde::Deserialize;
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

/// Raw tool object from the paginated API response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ComposioToolRaw {
    #[serde(alias = "slug")]
    pub name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
    #[serde(default, alias = "toolkitSlug")]
    pub toolkit_slug: Option<String>,
}

impl From<ComposioToolRaw> for ComposioToolDef {
    fn from(raw: ComposioToolRaw) -> Self {
        Self {
            slug: raw.name,
            display_name: raw.display_name,
            description: raw.description,
            input_schema: raw.input_schema,
            toolkit_slug: raw.toolkit_slug,
        }
    }
}

/// Paginated response from `GET /tools`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolsListResponse {
    #[serde(default, alias = "tools")]
    pub items: Vec<ComposioToolRaw>,
    #[serde(default, alias = "next_cursor")]
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
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolkitsListResponse {
    #[serde(default, alias = "toolkits")]
    pub items: Vec<ComposioToolkitRaw>,
    #[serde(default, alias = "next_cursor")]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Response from `POST /tools/execute/{slug}`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
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
                    "name": "GITHUB_CREATE_ISSUE",
                    "displayName": "Create Issue",
                    "description": "Create a new GitHub issue",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": { "type": "string" },
                            "repo": { "type": "string" },
                            "title": { "type": "string" }
                        },
                        "required": ["owner", "repo", "title"]
                    },
                    "toolkitSlug": "github"
                }
            ],
            "nextCursor": "abc123"
        }"#;

        let resp: ToolsListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.items.len(), 1);
        assert_eq!(resp.items[0].name, "GITHUB_CREATE_ISSUE");
        assert_eq!(resp.items[0].display_name.as_deref(), Some("Create Issue"));
        assert_eq!(resp.items[0].toolkit_slug.as_deref(), Some("github"));
        assert_eq!(resp.next_cursor.as_deref(), Some("abc123"));

        let def = ComposioToolDef::from(resp.items.into_iter().next().unwrap());
        assert_eq!(def.slug, "GITHUB_CREATE_ISSUE");
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
            "nextCursor": "xyz789"
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
