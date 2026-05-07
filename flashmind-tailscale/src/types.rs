//! Additional types for Tailscale local API responses.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Whois
// ---------------------------------------------------------------------------

/// Result of a whois lookup on a Tailscale IP address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhoisResult {
    pub node: PeerNode,
    pub user_profile: UserProfile,
}

/// A peer node returned by the whois endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerNode {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Addresses", default)]
    pub addresses: Vec<String>,
    #[serde(rename = "Online")]
    pub online: bool,
}

/// User profile attached to a whois result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    #[serde(rename = "ID")]
    pub id: u64,
    #[serde(rename = "LoginName")]
    pub login_name: String,
    #[serde(rename = "DisplayName")]
    pub display_name: String,
}

// ---------------------------------------------------------------------------
// Ping
// ---------------------------------------------------------------------------

/// Result of pinging a Tailscale peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResult {
    #[serde(rename = "IP")]
    pub ip: String,
    #[serde(rename = "LatencySeconds")]
    pub latency_seconds: f64,
    #[serde(rename = "NodeIP")]
    pub node_ip: String,
    #[serde(rename = "NodeName")]
    pub node_name: String,
    #[serde(rename = "Err", default)]
    pub err: Option<String>,
}

// ---------------------------------------------------------------------------
// Raw whois response shape
// ---------------------------------------------------------------------------

/// Raw JSON shape from `GET /localapi/v0/whois`.
#[derive(Deserialize)]
pub(crate) struct RawWhoisResponse {
    #[serde(rename = "Node")]
    pub node: PeerNode,
    #[serde(rename = "UserProfile")]
    pub user_profile: UserProfile,
}

impl RawWhoisResponse {
    pub fn into_result(self) -> WhoisResult {
        WhoisResult {
            node: self.node,
            user_profile: self.user_profile,
        }
    }
}

// ---------------------------------------------------------------------------
// Cert domains
// ---------------------------------------------------------------------------

/// Raw JSON shape from `GET /localapi/v0/cert-domains`.
#[derive(Deserialize)]
pub(crate) struct RawCertDomains {
    #[serde(rename = "Domains", default)]
    pub domains: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_whois_response() {
        let json = r#"{
            "Node": {
                "ID": "n123456",
                "Name": "peer-host.tail1234.ts.net.",
                "Addresses": ["100.64.0.10/32", "fd7a:115c:a1e0::a/128"],
                "Online": true
            },
            "UserProfile": {
                "ID": 12345,
                "LoginName": "user@example.com",
                "DisplayName": "Test User"
            }
        }"#;
        let raw: RawWhoisResponse = serde_json::from_str(json).unwrap();
        let result = raw.into_result();

        assert_eq!(result.node.id, "n123456");
        assert_eq!(result.node.name, "peer-host.tail1234.ts.net.");
        assert!(result.node.online);
        assert_eq!(result.user_profile.id, 12345);
        assert_eq!(result.user_profile.login_name, "user@example.com");
    }

    #[test]
    fn parse_ping_result() {
        let json = r#"{
            "IP": "100.64.0.10",
            "LatencySeconds": 0.0042,
            "NodeIP": "100.64.0.10",
            "NodeName": "peer-host",
            "Err": null
        }"#;
        let result: PingResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.ip, "100.64.0.10");
        assert!(result.latency_seconds < 0.01);
        assert!(result.err.is_none());
    }

    #[test]
    fn parse_ping_with_error() {
        let json = r#"{
            "IP": "100.64.0.99",
            "LatencySeconds": 0.0,
            "NodeIP": "",
            "NodeName": "",
            "Err": "timeout"
        }"#;
        let result: PingResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.err.as_deref(), Some("timeout"));
    }

    #[test]
    fn parse_cert_domains() {
        let json = r#"{"Domains": ["myhost.tail1234.ts.net"]}"#;
        let raw: RawCertDomains = serde_json::from_str(json).unwrap();
        assert_eq!(raw.domains, vec!["myhost.tail1234.ts.net"]);
    }
}
