//! Registry of known, trusted MCP servers with pre-configured connection details.
//!
//! When a user runs `mcp_add fastmail`, the preset system recognizes the name
//! and pre-fills the server URL and auth configuration, requiring only
//! credentials from the user.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub struct McpPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub server_url: &'static str,
    pub auth_type: PresetAuthType,
}

pub enum PresetAuthType {
    OAuth2 {
        auth_url: &'static str,
        token_url: &'static str,
        scopes: &'static [&'static str],
    },
    #[allow(dead_code)]
    ApiKey,
    #[allow(dead_code)]
    Bearer,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

static PRESETS: &[McpPreset] = &[McpPreset {
    name: "fastmail",
    description: "Fastmail email, contacts, and calendar via their official MCP server",
    server_url: "https://api.fastmail.com/mcp/sse",
    auth_type: PresetAuthType::Bearer,
}];

pub fn known_presets() -> &'static [McpPreset] {
    PRESETS
}

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
