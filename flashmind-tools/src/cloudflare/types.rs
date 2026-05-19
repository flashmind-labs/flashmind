//! Response types for the Cloudflare REST API.
//!
//! All types derive `Deserialize` and use `#[serde(default)]` for optional
//! fields so that missing keys in partial API responses don't cause errors.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Zone
// ---------------------------------------------------------------------------

/// A Cloudflare zone (domain).
#[derive(Debug, Deserialize)]
pub struct Zone {
    /// Unique zone identifier.
    pub id: String,
    /// Domain name (e.g. `example.com`).
    pub name: String,
    /// Zone status (`active`, `pending`, `initializing`, etc.).
    #[serde(default)]
    pub status: String,
    /// Plan information for the zone.
    #[serde(default)]
    pub plan: Option<ZonePlan>,
}

/// Billing plan associated with a zone.
#[derive(Debug, Deserialize)]
pub struct ZonePlan {
    /// Plan name (e.g. `Free`, `Pro`, `Business`, `Enterprise`).
    #[serde(default)]
    pub name: String,
}

// ---------------------------------------------------------------------------
// DNS Record
// ---------------------------------------------------------------------------

/// A DNS record within a zone.
#[derive(Debug, Deserialize)]
pub struct DnsRecord {
    /// Unique record identifier.
    pub id: String,
    /// Record type (`A`, `AAAA`, `CNAME`, `MX`, `TXT`, etc.).
    #[serde(rename = "type")]
    pub record_type: String,
    /// DNS record name (e.g. `sub.example.com`).
    pub name: String,
    /// Record content (IP address, hostname, text, etc.).
    pub content: String,
    /// Time to live in seconds. `1` means automatic.
    #[serde(default)]
    pub ttl: u32,
    /// Whether the record is proxied through Cloudflare.
    #[serde(default)]
    pub proxied: bool,
}

// ---------------------------------------------------------------------------
// Worker Route
// ---------------------------------------------------------------------------

/// A Cloudflare Workers route mapping.
#[derive(Debug, Deserialize)]
pub struct WorkerRoute {
    /// Unique route identifier.
    pub id: String,
    /// URL pattern for the route (e.g. `example.com/*`).
    #[serde(default)]
    pub pattern: String,
    /// Name of the worker script bound to this route.
    #[serde(default)]
    pub script: Option<String>,
}

// ---------------------------------------------------------------------------
// Token Verification
// ---------------------------------------------------------------------------

/// Response from `GET /user/tokens/verify`.
#[derive(Debug, Deserialize)]
pub struct TokenVerifyResult {
    /// Token status (`active`, `expired`, `disabled`, etc.).
    pub status: String,
}

// ---------------------------------------------------------------------------
// Purge Cache
// ---------------------------------------------------------------------------

/// Response from a cache purge request.
#[derive(Debug, Deserialize)]
pub struct PurgeCacheResult {
    /// Unique purge identifier.
    pub id: String,
}
