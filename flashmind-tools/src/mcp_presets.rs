//! Registry of known, trusted MCP servers with pre-configured connection details.
//!
//! When a user runs `mcp_add fastmail`, the preset system recognizes the name
//! and pre-fills the server URL and auth configuration, requiring only
//! credentials from the user.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A known, trusted MCP server with pre-configured connection details.
///
/// When a user runs `mcp_add <name>`, the preset system recognizes the name
/// and pre-fills the server URL and auth type, requiring only credentials.
pub struct McpPreset {
    /// Short identifier used to look up the preset (e.g. `"fastmail"`).
    pub name: &'static str,
    /// Human-readable summary shown when listing available presets.
    pub description: &'static str,
    /// The MCP endpoint URL to connect to.
    pub server_url: &'static str,
    /// How the server authenticates clients.
    pub auth_type: PresetAuthType,
}

/// Authentication mechanism required by an MCP preset.
pub enum PresetAuthType {
    /// Full OAuth 2.0 flow with authorization and token endpoints.
    OAuth2 {
        /// URL the user is redirected to for authorization.
        auth_url: &'static str,
        /// URL used to exchange the authorization code for tokens.
        token_url: &'static str,
        /// OAuth scopes to request.
        scopes: &'static [&'static str],
    },
    /// Static API key passed as a header.
    #[allow(dead_code)]
    ApiKey,
    /// Bearer token (e.g. app password or personal access token).
    #[allow(dead_code)]
    Bearer,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

static PRESETS: &[McpPreset] = &[McpPreset {
    name: "fastmail",
    description: "Fastmail email, contacts, and calendar via their official MCP server",
    server_url: "https://api.fastmail.com/mcp",
    auth_type: PresetAuthType::Bearer,
}];

/// Returns all registered MCP presets.
pub fn known_presets() -> &'static [McpPreset] {
    PRESETS
}

/// Look up a preset by name (case-insensitive).
pub fn find_preset(name: &str) -> Option<&'static McpPreset> {
    let name_lower = name.to_lowercase();
    PRESETS.iter().find(|p| p.name == name_lower)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_known_preset() {
        assert!(find_preset("fastmail").is_some());
        assert!(find_preset("Fastmail").is_some());
        assert!(find_preset("unknown").is_none());
    }

    #[test]
    fn presets_have_urls() {
        for preset in known_presets() {
            assert!(!preset.server_url.is_empty());
            assert!(!preset.name.is_empty());
        }
    }
}
