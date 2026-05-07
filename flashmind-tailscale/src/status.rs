//! Tailscale status types parsed from `tailscale status --json` or the local API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------

/// A single device in the Tailscale network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    pub device_name: String,
    pub tailscale_ips: Vec<String>,
    pub dns_name: String,
    pub online: bool,
    pub os: String,
}

/// A peer device in the tailnet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStatus {
    pub id: String,
    pub host_name: String,
    pub dns_name: String,
    pub tailscale_ips: Vec<String>,
    pub online: bool,
    pub os: String,
}

/// Top-level status returned by [`TailscaleClient::status`](crate::TailscaleClient::status).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailscaleStatus {
    pub self_device: DeviceStatus,
    pub version: String,
    pub peers: Vec<PeerStatus>,
}

// ---------------------------------------------------------------------------

/// Raw JSON shape of a node in `tailscale status --json`.
#[derive(Deserialize)]
pub(crate) struct RawNode {
    #[serde(rename = "ID", default)]
    pub id: String,
    #[serde(rename = "HostName")]
    pub host_name: String,
    #[serde(rename = "TailscaleIPs", default)]
    pub tailscale_ips: Vec<String>,
    #[serde(rename = "DNSName")]
    pub dns_name: String,
    #[serde(rename = "Online")]
    pub online: bool,
    #[serde(rename = "OS")]
    pub os: String,
}

/// Root of `tailscale status --json`.
#[derive(Deserialize)]
pub(crate) struct RawStatus {
    #[serde(rename = "Self")]
    pub self_node: RawNode,
    #[serde(rename = "Version")]
    pub version: String,
    #[serde(rename = "Peer", default)]
    pub peer: HashMap<String, RawNode>,
}

impl RawStatus {
    pub fn into_status(self) -> TailscaleStatus {
        let peers = self
            .peer
            .into_values()
            .map(|n| PeerStatus {
                id: n.id,
                host_name: n.host_name,
                dns_name: n.dns_name,
                tailscale_ips: n.tailscale_ips,
                online: n.online,
                os: n.os,
            })
            .collect();

        TailscaleStatus {
            self_device: DeviceStatus {
                device_name: self.self_node.host_name,
                tailscale_ips: self.self_node.tailscale_ips,
                dns_name: self.self_node.dns_name,
                online: self.self_node.online,
                os: self.self_node.os,
            },
            version: self.version,
            peers,
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSON: &str = r#"{
        "Version": "1.62.0",
        "Self": {
            "ID": "nSELF",
            "HostName": "myhost",
            "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"],
            "DNSName": "myhost.tail1234.ts.net.",
            "Online": true,
            "OS": "linux"
        },
        "Peer": {
            "abc123": {
                "ID": "nABC123",
                "HostName": "peer1",
                "TailscaleIPs": ["100.64.0.2"],
                "DNSName": "peer1.tail1234.ts.net.",
                "Online": true,
                "OS": "darwin"
            }
        }
    }"#;

    #[test]
    fn parse_raw_status() {
        let raw: RawStatus = serde_json::from_str(SAMPLE_JSON).unwrap();
        let status = raw.into_status();

        assert_eq!(status.version, "1.62.0");
        assert_eq!(status.self_device.device_name, "myhost");
        assert_eq!(status.self_device.tailscale_ips.len(), 2);
        assert_eq!(status.self_device.tailscale_ips[0], "100.64.0.1");
        assert_eq!(status.self_device.dns_name, "myhost.tail1234.ts.net.");
        assert!(status.self_device.online);
        assert_eq!(status.self_device.os, "linux");
    }

    #[test]
    fn parse_peers() {
        let raw: RawStatus = serde_json::from_str(SAMPLE_JSON).unwrap();
        let status = raw.into_status();

        assert_eq!(status.peers.len(), 1);
        assert_eq!(status.peers[0].host_name, "peer1");
        assert_eq!(status.peers[0].id, "nABC123");
        assert!(status.peers[0].online);
        assert_eq!(status.peers[0].os, "darwin");
    }

    #[test]
    fn parse_no_peers() {
        let json = r#"{
            "Version": "1.62.0",
            "Self": {
                "HostName": "solo",
                "TailscaleIPs": ["100.64.0.1"],
                "DNSName": "solo.tail1234.ts.net.",
                "Online": true,
                "OS": "linux"
            }
        }"#;
        let raw: RawStatus = serde_json::from_str(json).unwrap();
        let status = raw.into_status();
        assert!(status.peers.is_empty());
    }

    #[test]
    fn parse_offline_device() {
        let json = r#"{
            "Version": "1.60.0",
            "Self": {
                "HostName": "laptop",
                "TailscaleIPs": ["100.64.0.5"],
                "DNSName": "laptop.tail9999.ts.net.",
                "Online": false,
                "OS": "darwin"
            }
        }"#;
        let raw: RawStatus = serde_json::from_str(json).unwrap();
        let status = raw.into_status();

        assert!(!status.self_device.online);
        assert_eq!(status.self_device.os, "darwin");
    }
}
