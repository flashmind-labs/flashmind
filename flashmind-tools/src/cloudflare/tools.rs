//! Cloudflare `Tool` trait implementations.
//!
//! Eight tools covering zones, DNS records, worker routes, and cache purging.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use anyhow::bail;
use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::CloudflareClient;
use super::types::*;

// ---------------------------------------------------------------------------
// cloudflare_list_zones
// ---------------------------------------------------------------------------

/// List all zones (domains) in the Cloudflare account.
pub struct CloudflareListZonesTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[async_trait]
impl Tool for CloudflareListZonesTool {
    fn name(&self) -> &str {
        "cloudflare_list_zones"
    }

    fn description(&self) -> &str {
        "List all zones (domains) in the Cloudflare account."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let zones: Vec<Zone> = self.client.get("zones?per_page=50").await?;

        let mut out = String::new();
        if zones.is_empty() {
            out.push_str("No zones found.");
        } else {
            let _ = writeln!(out, "| ID | Name | Status | Plan |");
            let _ = writeln!(out, "|---|---|---|---|");
            for z in &zones {
                let plan = z.plan.as_ref().map(|p| p.name.as_str()).unwrap_or("—");
                let _ = writeln!(out, "| {} | {} | {} | {} |", z.id, z.name, z.status, plan,);
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Listing Cloudflare zones".into()
    }
}

// ---------------------------------------------------------------------------
// cloudflare_list_dns_records
// ---------------------------------------------------------------------------

/// List DNS records for a Cloudflare zone.
pub struct CloudflareListDnsRecordsTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct ListDnsRecordsArgs {
    zone_id: String,
}

#[async_trait]
impl Tool for CloudflareListDnsRecordsTool {
    fn name(&self) -> &str {
        "cloudflare_list_dns_records"
    }

    fn description(&self) -> &str {
        "List DNS records for a Cloudflare zone."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                }
            },
            "required": ["zone_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: ListDnsRecordsArgs = parse_args(self.name(), ctx.args)?;

        let path = format!("zones/{}/dns_records?per_page=100", args.zone_id);
        let records: Vec<DnsRecord> = self.client.get(&path).await?;

        let mut out = String::new();
        if records.is_empty() {
            out.push_str("No DNS records found.");
        } else {
            let _ = writeln!(out, "| ID | Type | Name | Content | TTL | Proxied |");
            let _ = writeln!(out, "|---|---|---|---|---|---|");
            for r in &records {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} |",
                    r.id, r.record_type, r.name, r.content, r.ttl, r.proxied,
                );
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let zone = args.get("zone_id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Listing DNS records for zone {zone}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_get_dns_record
// ---------------------------------------------------------------------------

/// Get a specific DNS record by zone and record ID.
pub struct CloudflareGetDnsRecordTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct GetDnsRecordArgs {
    zone_id: String,
    record_id: String,
}

#[async_trait]
impl Tool for CloudflareGetDnsRecordTool {
    fn name(&self) -> &str {
        "cloudflare_get_dns_record"
    }

    fn description(&self) -> &str {
        "Get a specific DNS record by zone and record ID."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                },
                "record_id": {
                    "type": "string",
                    "description": "DNS record identifier"
                }
            },
            "required": ["zone_id", "record_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: GetDnsRecordArgs = parse_args(self.name(), ctx.args)?;

        let path = format!("zones/{}/dns_records/{}", args.zone_id, args.record_id);
        let record: DnsRecord = self.client.get(&path).await?;

        let mut out = String::new();
        let _ = writeln!(out, "ID: {}", record.id);
        let _ = writeln!(out, "Type: {}", record.record_type);
        let _ = writeln!(out, "Name: {}", record.name);
        let _ = writeln!(out, "Content: {}", record.content);
        let _ = writeln!(out, "TTL: {}", record.ttl);
        let _ = writeln!(out, "Proxied: {}", record.proxied);

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let record = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        format!("Getting DNS record {record}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_list_worker_routes
// ---------------------------------------------------------------------------

/// List worker routes for a Cloudflare zone.
pub struct CloudflareListWorkerRoutesTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct ListWorkerRoutesArgs {
    zone_id: String,
}

#[async_trait]
impl Tool for CloudflareListWorkerRoutesTool {
    fn name(&self) -> &str {
        "cloudflare_list_worker_routes"
    }

    fn description(&self) -> &str {
        "List worker routes for a Cloudflare zone."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                }
            },
            "required": ["zone_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: ListWorkerRoutesArgs = parse_args(self.name(), ctx.args)?;

        let path = format!("zones/{}/workers/routes", args.zone_id);
        let routes: Vec<WorkerRoute> = self.client.get(&path).await?;

        let mut out = String::new();
        if routes.is_empty() {
            out.push_str("No worker routes found.");
        } else {
            let _ = writeln!(out, "| ID | Pattern | Script |");
            let _ = writeln!(out, "|---|---|---|");
            for r in &routes {
                let script = r.script.as_deref().unwrap_or("—");
                let _ = writeln!(out, "| {} | {} | {} |", r.id, r.pattern, script);
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let zone = args.get("zone_id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Listing worker routes for zone {zone}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_create_dns_record
// ---------------------------------------------------------------------------

/// Create a new DNS record in a Cloudflare zone.
pub struct CloudflareCreateDnsRecordTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct CreateDnsRecordArgs {
    zone_id: String,
    #[serde(rename = "type")]
    record_type: String,
    name: String,
    content: String,
    ttl: Option<u32>,
    proxied: Option<bool>,
}

#[async_trait]
impl Tool for CloudflareCreateDnsRecordTool {
    fn name(&self) -> &str {
        "cloudflare_create_dns_record"
    }

    fn description(&self) -> &str {
        "Create a new DNS record in a Cloudflare zone."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                },
                "type": {
                    "type": "string",
                    "description": "Record type (A, AAAA, CNAME, MX, TXT, etc.)"
                },
                "name": {
                    "type": "string",
                    "description": "DNS record name (e.g. sub.example.com or @ for root)"
                },
                "content": {
                    "type": "string",
                    "description": "Record content (IP address, hostname, text, etc.)"
                },
                "ttl": {
                    "type": "integer",
                    "description": "Time to live in seconds (1 = automatic, default: 1)"
                },
                "proxied": {
                    "type": "boolean",
                    "description": "Whether to proxy through Cloudflare (default: false)"
                }
            },
            "required": ["zone_id", "type", "name", "content"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: CreateDnsRecordArgs = parse_args(self.name(), ctx.args)?;

        let mut payload = json!({
            "type": args.record_type,
            "name": args.name,
            "content": args.content,
        });
        if let Some(ttl) = args.ttl {
            payload["ttl"] = json!(ttl);
        }
        if let Some(proxied) = args.proxied {
            payload["proxied"] = json!(proxied);
        }

        let path = format!("zones/{}/dns_records", args.zone_id);
        let record: DnsRecord = self.client.post(&path, &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Created {} record: {} -> {} (ID: {})",
                record.record_type, record.name, record.content, record.id,
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let rtype = args.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Creating {rtype} record for {name}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_update_dns_record
// ---------------------------------------------------------------------------

/// Update an existing DNS record in a Cloudflare zone.
pub struct CloudflareUpdateDnsRecordTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct UpdateDnsRecordArgs {
    zone_id: String,
    record_id: String,
    #[serde(rename = "type")]
    record_type: String,
    name: String,
    content: String,
    ttl: Option<u32>,
    proxied: Option<bool>,
}

#[async_trait]
impl Tool for CloudflareUpdateDnsRecordTool {
    fn name(&self) -> &str {
        "cloudflare_update_dns_record"
    }

    fn description(&self) -> &str {
        "Update an existing DNS record in a Cloudflare zone."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                },
                "record_id": {
                    "type": "string",
                    "description": "DNS record identifier"
                },
                "type": {
                    "type": "string",
                    "description": "Record type (A, AAAA, CNAME, MX, TXT, etc.)"
                },
                "name": {
                    "type": "string",
                    "description": "DNS record name"
                },
                "content": {
                    "type": "string",
                    "description": "Record content"
                },
                "ttl": {
                    "type": "integer",
                    "description": "Time to live in seconds (1 = automatic)"
                },
                "proxied": {
                    "type": "boolean",
                    "description": "Whether to proxy through Cloudflare"
                }
            },
            "required": ["zone_id", "record_id", "type", "name", "content"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: UpdateDnsRecordArgs = parse_args(self.name(), ctx.args)?;

        let mut payload = json!({
            "type": args.record_type,
            "name": args.name,
            "content": args.content,
        });
        if let Some(ttl) = args.ttl {
            payload["ttl"] = json!(ttl);
        }
        if let Some(proxied) = args.proxied {
            payload["proxied"] = json!(proxied);
        }

        let path = format!("zones/{}/dns_records/{}", args.zone_id, args.record_id);
        let record: DnsRecord = self.client.put(&path, &payload).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Updated {} record: {} -> {} (ID: {})",
                record.record_type, record.name, record.content, record.id,
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Updating DNS record for {name}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_delete_dns_record
// ---------------------------------------------------------------------------

/// Delete a DNS record from a Cloudflare zone.
pub struct CloudflareDeleteDnsRecordTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct DeleteDnsRecordArgs {
    zone_id: String,
    record_id: String,
}

#[async_trait]
impl Tool for CloudflareDeleteDnsRecordTool {
    fn name(&self) -> &str {
        "cloudflare_delete_dns_record"
    }

    fn description(&self) -> &str {
        "Delete a DNS record from a Cloudflare zone."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                },
                "record_id": {
                    "type": "string",
                    "description": "DNS record identifier"
                }
            },
            "required": ["zone_id", "record_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: DeleteDnsRecordArgs = parse_args(self.name(), ctx.args)?;

        let path = format!("zones/{}/dns_records/{}", args.zone_id, args.record_id);
        self.client.delete_req(&path).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Deleted DNS record {}", args.record_id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let record = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        format!("Deleting DNS record {record}")
    }
}

// ---------------------------------------------------------------------------
// cloudflare_purge_cache
// ---------------------------------------------------------------------------

/// Purge cached content for a Cloudflare zone.
pub struct CloudflarePurgeCacheTool {
    /// Shared Cloudflare API client.
    pub client: Arc<CloudflareClient>,
}

#[derive(Deserialize)]
struct PurgeCacheArgs {
    zone_id: String,
    purge_everything: Option<bool>,
    files: Option<Vec<String>>,
}

#[async_trait]
impl Tool for CloudflarePurgeCacheTool {
    fn name(&self) -> &str {
        "cloudflare_purge_cache"
    }

    fn description(&self) -> &str {
        "Purge cached content for a Cloudflare zone. Either purge everything \
         or specify individual URLs to purge."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "zone_id": {
                    "type": "string",
                    "description": "Zone identifier"
                },
                "purge_everything": {
                    "type": "boolean",
                    "description": "Purge all cached content (default: false)"
                },
                "files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of specific URLs to purge from cache"
                }
            },
            "required": ["zone_id"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            bail!("cancelled");
        }
        let args: PurgeCacheArgs = parse_args(self.name(), ctx.args)?;

        let mut payload = json!({});
        if args.purge_everything.unwrap_or(false) {
            payload["purge_everything"] = json!(true);
        } else if let Some(files) = &args.files {
            payload["files"] = json!(files);
        } else {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                "Please specify either `purge_everything: true` or a list of `files` to purge."
                    .to_string(),
            ));
        }

        let path = format!("zones/{}/purge_cache", args.zone_id);
        let result: PurgeCacheResult = self.client.post(&path, &payload).await?;

        let desc = if args.purge_everything.unwrap_or(false) {
            "all cached content".to_string()
        } else {
            format!(
                "{} file(s)",
                args.files.as_ref().map(|f| f.len()).unwrap_or(0)
            )
        };

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("Cache purge initiated for {desc} (ID: {})", result.id),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        if args
            .get("purge_everything")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            "Purging all cached content".into()
        } else {
            "Purging specific cached files".into()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> Arc<CloudflareClient> {
        Arc::new(CloudflareClient::new_for_test())
    }

    // -- name() tests -------------------------------------------------------

    #[test]
    fn tool_names() {
        let c = make_client();
        assert_eq!(
            CloudflareListZonesTool { client: c.clone() }.name(),
            "cloudflare_list_zones"
        );
        assert_eq!(
            CloudflareListDnsRecordsTool { client: c.clone() }.name(),
            "cloudflare_list_dns_records"
        );
        assert_eq!(
            CloudflareGetDnsRecordTool { client: c.clone() }.name(),
            "cloudflare_get_dns_record"
        );
        assert_eq!(
            CloudflareListWorkerRoutesTool { client: c.clone() }.name(),
            "cloudflare_list_worker_routes"
        );
        assert_eq!(
            CloudflareCreateDnsRecordTool { client: c.clone() }.name(),
            "cloudflare_create_dns_record"
        );
        assert_eq!(
            CloudflareUpdateDnsRecordTool { client: c.clone() }.name(),
            "cloudflare_update_dns_record"
        );
        assert_eq!(
            CloudflareDeleteDnsRecordTool { client: c.clone() }.name(),
            "cloudflare_delete_dns_record"
        );
        assert_eq!(
            CloudflarePurgeCacheTool { client: c.clone() }.name(),
            "cloudflare_purge_cache"
        );
    }

    // -- humanize() tests ---------------------------------------------------

    #[test]
    fn humanize_list_zones() {
        let c = make_client();
        let tool = CloudflareListZonesTool { client: c };
        assert_eq!(tool.humanize(&json!({})), "Listing Cloudflare zones");
    }

    #[test]
    fn humanize_list_dns_records() {
        let c = make_client();
        let tool = CloudflareListDnsRecordsTool { client: c };
        let h = tool.humanize(&json!({ "zone_id": "abc123" }));
        assert!(h.contains("abc123"));
    }

    #[test]
    fn humanize_get_dns_record() {
        let c = make_client();
        let tool = CloudflareGetDnsRecordTool { client: c };
        let h = tool.humanize(&json!({ "record_id": "rec456" }));
        assert!(h.contains("rec456"));
    }

    #[test]
    fn humanize_create_dns_record() {
        let c = make_client();
        let tool = CloudflareCreateDnsRecordTool { client: c };
        let h = tool.humanize(&json!({ "type": "A", "name": "sub.example.com" }));
        assert!(h.contains("A"));
        assert!(h.contains("sub.example.com"));
    }

    #[test]
    fn humanize_purge_cache_everything() {
        let c = make_client();
        let tool = CloudflarePurgeCacheTool { client: c };
        let h = tool.humanize(&json!({ "purge_everything": true }));
        assert!(h.contains("all"));
    }

    #[test]
    fn humanize_purge_cache_files() {
        let c = make_client();
        let tool = CloudflarePurgeCacheTool { client: c };
        let h = tool.humanize(&json!({ "files": ["https://example.com/a.js"] }));
        assert!(h.contains("specific"));
    }

    // -- parameters() schema tests ------------------------------------------

    #[test]
    fn parameters_are_objects() {
        let c = make_client();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(CloudflareListZonesTool { client: c.clone() }),
            Box::new(CloudflareListDnsRecordsTool { client: c.clone() }),
            Box::new(CloudflareGetDnsRecordTool { client: c.clone() }),
            Box::new(CloudflareListWorkerRoutesTool { client: c.clone() }),
            Box::new(CloudflareCreateDnsRecordTool { client: c.clone() }),
            Box::new(CloudflareUpdateDnsRecordTool { client: c.clone() }),
            Box::new(CloudflareDeleteDnsRecordTool { client: c.clone() }),
            Box::new(CloudflarePurgeCacheTool { client: c.clone() }),
        ];

        for tool in &tools {
            let params = tool.parameters();
            assert_eq!(
                params.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "tool '{}' parameters should have type: object",
                tool.name()
            );
            assert!(
                params.get("properties").is_some(),
                "tool '{}' parameters should have properties",
                tool.name()
            );
        }
    }

    #[test]
    fn required_fields_present() {
        let c = make_client();

        let list_dns = CloudflareListDnsRecordsTool { client: c.clone() };
        let required = list_dns.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"zone_id".to_string()));

        let get_dns = CloudflareGetDnsRecordTool { client: c.clone() };
        let required = get_dns.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"zone_id".to_string()));
        assert!(required.contains(&"record_id".to_string()));

        let create_dns = CloudflareCreateDnsRecordTool { client: c.clone() };
        let required = create_dns.parameters()["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"zone_id".to_string()));
        assert!(required.contains(&"type".to_string()));
        assert!(required.contains(&"name".to_string()));
        assert!(required.contains(&"content".to_string()));
    }
}
