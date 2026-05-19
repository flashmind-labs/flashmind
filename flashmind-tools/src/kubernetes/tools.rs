//! Kubernetes `Tool` trait implementations.
//!
//! Eleven tools covering pods, deployments, services, events, manifests,
//! scaling, deletion, and exec — modeled after common `kubectl` workflows.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::{Event, Pod, Service};
use kube::Client;
use kube::api::{Api, DeleteParams, ListParams, LogParams, Patch, PatchParams};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::Tool;
use flashmind_types::tool::{ToolContext, ToolResult, parse_args};

use super::types;

// ---------------------------------------------------------------------------
// k8s_list_pods
// ---------------------------------------------------------------------------

/// List pods in a namespace, formatted like `kubectl get pods`.
pub struct K8sListPodsTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ListPodsArgs {
    namespace: Option<String>,
    label_selector: Option<String>,
}

#[async_trait]
impl Tool for K8sListPodsTool {
    fn name(&self) -> &str {
        "k8s_list_pods"
    }

    fn description(&self) -> &str {
        "List Kubernetes pods in a namespace. Returns a table with name, ready \
         containers, status, restarts, and age."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                },
                "label_selector": {
                    "type": "string",
                    "description": "Label selector to filter pods (e.g. app=nginx)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ListPodsArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(namespace = ns, "k8s_list_pods");

        let api: Api<Pod> = Api::namespaced((*self.client).clone(), ns);
        let mut lp = ListParams::default();
        if let Some(sel) = &args.label_selector {
            lp = lp.labels(sel);
        }
        let pods = api.list(&lp).await?;

        let mut out = String::new();
        let _ = writeln!(out, "NAME\tREADY\tSTATUS\tRESTARTS\tAGE");
        for pod in &pods.items {
            let name = pod.metadata.name.as_deref().unwrap_or("<unknown>");
            let (ready, total) = types::container_ready_count(pod);
            let status = types::pod_status(pod);
            let restarts = types::total_restarts(pod);
            let age = pod
                .metadata
                .creation_timestamp
                .as_ref()
                .map(|t| types::format_age(&t.0.to_rfc3339()))
                .unwrap_or_else(|| "unknown".into());
            let _ = writeln!(out, "{name}\t{ready}/{total}\t{status}\t{restarts}\t{age}");
        }

        if pods.items.is_empty() {
            out.push_str("No pods found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let ns = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_namespace);
        format!("Listing pods in {ns}")
    }
}

// ---------------------------------------------------------------------------
// k8s_get_pod
// ---------------------------------------------------------------------------

/// Get detailed information about a single pod.
pub struct K8sGetPodTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct GetPodArgs {
    name: String,
    namespace: Option<String>,
}

#[async_trait]
impl Tool for K8sGetPodTool {
    fn name(&self) -> &str {
        "k8s_get_pod"
    }

    fn description(&self) -> &str {
        "Get detailed information about a Kubernetes pod including status, \
         conditions, containers, and recent events."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Pod name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: GetPodArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(pod = %args.name, namespace = ns, "k8s_get_pod");

        let api: Api<Pod> = Api::namespaced((*self.client).clone(), ns);
        let pod = api.get(&args.name).await?;

        let mut out = String::new();
        let name = pod.metadata.name.as_deref().unwrap_or("<unknown>");
        let namespace = pod.metadata.namespace.as_deref().unwrap_or(ns);
        let status = types::pod_status(&pod);
        let age = pod
            .metadata
            .creation_timestamp
            .as_ref()
            .map(|t| types::format_age(&t.0.to_rfc3339()))
            .unwrap_or_else(|| "unknown".into());

        let _ = writeln!(out, "Name:      {name}");
        let _ = writeln!(out, "Namespace: {namespace}");
        let _ = writeln!(out, "Status:    {status}");
        let _ = writeln!(out, "Age:       {age}");

        // Node
        if let Some(spec) = &pod.spec
            && let Some(node) = &spec.node_name
        {
            let _ = writeln!(out, "Node:      {node}");
        }

        // Conditions
        if let Some(status) = &pod.status
            && let Some(conditions) = &status.conditions
        {
            let _ = writeln!(out, "\nConditions:");
            for cond in conditions {
                let _ = writeln!(out, "  {}: {}", cond.type_, cond.status);
            }
        }

        // Containers
        if let Some(spec) = &pod.spec {
            let _ = writeln!(out, "\nContainers:");
            for container in &spec.containers {
                let _ = writeln!(out, "  {}:", container.name);
                let _ = writeln!(
                    out,
                    "    Image: {}",
                    container.image.as_deref().unwrap_or("—")
                );

                // State from status
                if let Some(pod_status) = &pod.status
                    && let Some(statuses) = &pod_status.container_statuses
                    && let Some(cs) = statuses.iter().find(|s| s.name == container.name)
                {
                    let state_str = if let Some(state) = &cs.state {
                        if state.running.is_some() {
                            "Running".to_string()
                        } else if let Some(w) = &state.waiting {
                            format!("Waiting ({})", w.reason.as_deref().unwrap_or("unknown"))
                        } else if let Some(t) = &state.terminated {
                            format!("Terminated ({})", t.reason.as_deref().unwrap_or("unknown"))
                        } else {
                            "Unknown".to_string()
                        }
                    } else {
                        "Unknown".to_string()
                    };
                    let _ = writeln!(out, "    State: {state_str}");
                    let _ = writeln!(out, "    Restarts: {}", cs.restart_count);
                }
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        format!("Getting pod {name}")
    }
}

// ---------------------------------------------------------------------------
// k8s_pod_logs
// ---------------------------------------------------------------------------

/// Retrieve logs from a pod container.
pub struct K8sPodLogsTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct PodLogsArgs {
    name: String,
    namespace: Option<String>,
    container: Option<String>,
    tail_lines: Option<i64>,
    since_seconds: Option<i64>,
}

#[async_trait]
impl Tool for K8sPodLogsTool {
    fn name(&self) -> &str {
        "k8s_pod_logs"
    }

    fn description(&self) -> &str {
        "Retrieve logs from a Kubernetes pod. Optionally specify a container, \
         tail lines, or a time window."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Pod name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                },
                "container": {
                    "type": "string",
                    "description": "Container name (required for multi-container pods)"
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Number of recent log lines to return (default 100)"
                },
                "since_seconds": {
                    "type": "integer",
                    "description": "Only return logs from the last N seconds"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: PodLogsArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(pod = %args.name, namespace = ns, "k8s_pod_logs");

        let api: Api<Pod> = Api::namespaced((*self.client).clone(), ns);
        let lp = LogParams {
            container: args.container.clone(),
            tail_lines: Some(args.tail_lines.unwrap_or(100)),
            since_seconds: args.since_seconds,
            ..Default::default()
        };
        let logs = api.logs(&args.name, &lp).await?;

        // Truncate very large logs.
        const MAX_CHARS: usize = 50_000;
        let out = if logs.len() > MAX_CHARS {
            let start = logs.ceil_char_boundary(logs.len() - MAX_CHARS);
            format!("[truncated to {MAX_CHARS} chars]\n{}", &logs[start..])
        } else if logs.is_empty() {
            "No logs found.".into()
        } else {
            logs
        };

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        format!("Fetching logs for pod {name}")
    }
}

// ---------------------------------------------------------------------------
// k8s_list_deployments
// ---------------------------------------------------------------------------

/// List deployments in a namespace, formatted like `kubectl get deployments`.
pub struct K8sListDeploymentsTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ListDeploymentsArgs {
    namespace: Option<String>,
    label_selector: Option<String>,
}

#[async_trait]
impl Tool for K8sListDeploymentsTool {
    fn name(&self) -> &str {
        "k8s_list_deployments"
    }

    fn description(&self) -> &str {
        "List Kubernetes deployments in a namespace. Returns a table with name, \
         ready replicas, up-to-date, available, and age."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                },
                "label_selector": {
                    "type": "string",
                    "description": "Label selector to filter deployments (e.g. app=nginx)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ListDeploymentsArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(namespace = ns, "k8s_list_deployments");

        let api: Api<Deployment> = Api::namespaced((*self.client).clone(), ns);
        let mut lp = ListParams::default();
        if let Some(sel) = &args.label_selector {
            lp = lp.labels(sel);
        }
        let deployments = api.list(&lp).await?;

        let mut out = String::new();
        let _ = writeln!(out, "NAME\tREADY\tUP-TO-DATE\tAVAILABLE\tAGE");
        for dep in &deployments.items {
            let name = dep.metadata.name.as_deref().unwrap_or("<unknown>");
            let status = dep.status.as_ref();
            let desired = dep.spec.as_ref().and_then(|s| s.replicas).unwrap_or(0);
            let ready = status.and_then(|s| s.ready_replicas).unwrap_or(0);
            let up_to_date = status.and_then(|s| s.updated_replicas).unwrap_or(0);
            let available = status.and_then(|s| s.available_replicas).unwrap_or(0);
            let age = dep
                .metadata
                .creation_timestamp
                .as_ref()
                .map(|t| types::format_age(&t.0.to_rfc3339()))
                .unwrap_or_else(|| "unknown".into());
            let _ = writeln!(
                out,
                "{name}\t{ready}/{desired}\t{up_to_date}\t{available}\t{age}"
            );
        }

        if deployments.items.is_empty() {
            out.push_str("No deployments found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let ns = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_namespace);
        format!("Listing deployments in {ns}")
    }
}

// ---------------------------------------------------------------------------
// k8s_get_deployment
// ---------------------------------------------------------------------------

/// Get detailed information about a single deployment.
pub struct K8sGetDeploymentTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct GetDeploymentArgs {
    name: String,
    namespace: Option<String>,
}

#[async_trait]
impl Tool for K8sGetDeploymentTool {
    fn name(&self) -> &str {
        "k8s_get_deployment"
    }

    fn description(&self) -> &str {
        "Get detailed information about a Kubernetes deployment including \
         strategy, replicas, conditions, and container specs."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Deployment name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: GetDeploymentArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(deployment = %args.name, namespace = ns, "k8s_get_deployment");

        let api: Api<Deployment> = Api::namespaced((*self.client).clone(), ns);
        let dep = api.get(&args.name).await?;

        let mut out = String::new();
        let name = dep.metadata.name.as_deref().unwrap_or("<unknown>");
        let namespace = dep.metadata.namespace.as_deref().unwrap_or(ns);
        let age = dep
            .metadata
            .creation_timestamp
            .as_ref()
            .map(|t| types::format_age(&t.0.to_rfc3339()))
            .unwrap_or_else(|| "unknown".into());

        let _ = writeln!(out, "Name:      {name}");
        let _ = writeln!(out, "Namespace: {namespace}");
        let _ = writeln!(out, "Age:       {age}");

        // Replicas
        if let Some(spec) = &dep.spec {
            let desired = spec.replicas.unwrap_or(1);
            let _ = writeln!(out, "Replicas:  {desired} desired");

            // Strategy
            if let Some(strategy) = &spec.strategy {
                let strat_type = strategy.type_.as_deref().unwrap_or("RollingUpdate");
                let _ = writeln!(out, "Strategy:  {strat_type}");
            }
        }

        // Status
        if let Some(status) = &dep.status {
            let ready = status.ready_replicas.unwrap_or(0);
            let available = status.available_replicas.unwrap_or(0);
            let updated = status.updated_replicas.unwrap_or(0);
            let _ = writeln!(
                out,
                "Status:    {ready} ready, {available} available, {updated} up-to-date"
            );

            // Conditions
            if let Some(conditions) = &status.conditions {
                let _ = writeln!(out, "\nConditions:");
                for cond in conditions {
                    let _ = writeln!(
                        out,
                        "  {}: {} ({})",
                        cond.type_,
                        cond.status,
                        cond.reason.as_deref().unwrap_or("—"),
                    );
                }
            }
        }

        // Containers in the template
        if let Some(spec) = &dep.spec
            && let Some(template_spec) = &spec.template.spec
        {
            let _ = writeln!(out, "\nContainers:");
            for container in &template_spec.containers {
                let _ = writeln!(out, "  {}:", container.name);
                let _ = writeln!(
                    out,
                    "    Image: {}",
                    container.image.as_deref().unwrap_or("—")
                );
                if let Some(ports) = &container.ports {
                    let port_strs: Vec<String> = ports
                        .iter()
                        .map(|p| {
                            let proto = p.protocol.as_deref().unwrap_or("TCP");
                            format!("{}/{proto}", p.container_port)
                        })
                        .collect();
                    let _ = writeln!(out, "    Ports: {}", port_strs.join(", "));
                }
            }
        }

        // Labels
        if let Some(labels) = &dep.metadata.labels
            && !labels.is_empty()
        {
            let _ = writeln!(out, "\nLabels:");
            for (k, v) in labels {
                let _ = writeln!(out, "  {k}={v}");
            }
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        format!("Getting deployment {name}")
    }
}

// ---------------------------------------------------------------------------
// k8s_list_services
// ---------------------------------------------------------------------------

/// List services in a namespace, formatted like `kubectl get svc`.
pub struct K8sListServicesTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ListServicesArgs {
    namespace: Option<String>,
}

#[async_trait]
impl Tool for K8sListServicesTool {
    fn name(&self) -> &str {
        "k8s_list_services"
    }

    fn description(&self) -> &str {
        "List Kubernetes services in a namespace. Returns a table with name, \
         type, cluster IP, external IP, ports, and age."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ListServicesArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(namespace = ns, "k8s_list_services");

        let api: Api<Service> = Api::namespaced((*self.client).clone(), ns);
        let services = api.list(&ListParams::default()).await?;

        let mut out = String::new();
        let _ = writeln!(out, "NAME\tTYPE\tCLUSTER-IP\tEXTERNAL-IP\tPORTS\tAGE");
        for svc in &services.items {
            let name = svc.metadata.name.as_deref().unwrap_or("<unknown>");
            let spec = svc.spec.as_ref();
            let svc_type = spec.and_then(|s| s.type_.as_deref()).unwrap_or("ClusterIP");
            let cluster_ip = spec
                .and_then(|s| s.cluster_ip.as_deref())
                .unwrap_or("<none>");
            let external_ip = spec
                .and_then(|s| s.external_ips.as_ref())
                .and_then(|ips| {
                    if ips.is_empty() {
                        None
                    } else {
                        Some(ips.join(","))
                    }
                })
                .or_else(|| {
                    svc.status
                        .as_ref()
                        .and_then(|s| s.load_balancer.as_ref())
                        .and_then(|lb| lb.ingress.as_ref())
                        .and_then(|ingress| {
                            let ips: Vec<String> = ingress
                                .iter()
                                .filter_map(|i| i.ip.clone().or_else(|| i.hostname.clone()))
                                .collect();
                            if ips.is_empty() {
                                None
                            } else {
                                Some(ips.join(","))
                            }
                        })
                })
                .unwrap_or_else(|| "<none>".into());
            let ports = spec
                .and_then(|s| s.ports.as_ref())
                .map(|ports| {
                    ports
                        .iter()
                        .map(|p| {
                            let proto = p.protocol.as_deref().unwrap_or("TCP");
                            if let Some(np) = p.node_port {
                                format!("{}:{np}/{proto}", p.port)
                            } else {
                                format!("{}/{proto}", p.port)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_else(|| "<none>".into());
            let age = svc
                .metadata
                .creation_timestamp
                .as_ref()
                .map(|t| types::format_age(&t.0.to_rfc3339()))
                .unwrap_or_else(|| "unknown".into());
            let _ = writeln!(
                out,
                "{name}\t{svc_type}\t{cluster_ip}\t{external_ip}\t{ports}\t{age}"
            );
        }

        if services.items.is_empty() {
            out.push_str("No services found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let ns = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_namespace);
        format!("Listing services in {ns}")
    }
}

// ---------------------------------------------------------------------------
// k8s_list_events
// ---------------------------------------------------------------------------

/// List recent events in a namespace.
pub struct K8sListEventsTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ListEventsArgs {
    namespace: Option<String>,
}

#[async_trait]
impl Tool for K8sListEventsTool {
    fn name(&self) -> &str {
        "k8s_list_events"
    }

    fn description(&self) -> &str {
        "List recent Kubernetes events in a namespace. Shows the last 50 events \
         with timestamp, type, reason, object, and message."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                }
            }
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ListEventsArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(namespace = ns, "k8s_list_events");

        let api: Api<Event> = Api::namespaced((*self.client).clone(), ns);
        let events = api.list(&ListParams::default()).await?;

        let mut items: Vec<&Event> = events.items.iter().collect();
        // Sort by last timestamp descending, take last 50.
        items.sort_by(|a, b| {
            let ts_a = a
                .last_timestamp
                .as_ref()
                .map(|t| t.0)
                .or_else(|| a.metadata.creation_timestamp.as_ref().map(|t| t.0));
            let ts_b = b
                .last_timestamp
                .as_ref()
                .map(|t| t.0)
                .or_else(|| b.metadata.creation_timestamp.as_ref().map(|t| t.0));
            ts_b.cmp(&ts_a)
        });
        items.truncate(50);

        let mut out = String::new();
        let _ = writeln!(out, "LAST SEEN\tTYPE\tREASON\tOBJECT\tMESSAGE");
        for event in &items {
            let last_seen = event
                .last_timestamp
                .as_ref()
                .map(|t| types::format_age(&t.0.to_rfc3339()))
                .or_else(|| {
                    event
                        .metadata
                        .creation_timestamp
                        .as_ref()
                        .map(|t| types::format_age(&t.0.to_rfc3339()))
                })
                .unwrap_or_else(|| "unknown".into());
            let type_ = event.type_.as_deref().unwrap_or("Normal");
            let reason = event.reason.as_deref().unwrap_or("—");
            let object = event.involved_object.name.as_deref().unwrap_or("<unknown>");
            let kind = event.involved_object.kind.as_deref().unwrap_or("");
            let message = event.message.as_deref().unwrap_or("");
            let _ = writeln!(
                out,
                "{last_seen}\t{type_}\t{reason}\t{kind}/{object}\t{message}"
            );
        }

        if items.is_empty() {
            out.push_str("No events found.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let ns = args
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_namespace);
        format!("Listing events in {ns}")
    }
}

// ---------------------------------------------------------------------------
// k8s_apply_manifest
// ---------------------------------------------------------------------------

/// Apply a YAML or JSON manifest using server-side apply.
pub struct K8sApplyManifestTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ApplyManifestArgs {
    manifest: String,
}

#[async_trait]
impl Tool for K8sApplyManifestTool {
    fn name(&self) -> &str {
        "k8s_apply_manifest"
    }

    fn description(&self) -> &str {
        "Apply a Kubernetes manifest (YAML or JSON) using server-side apply. \
         The manifest must include apiVersion, kind, and metadata."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "manifest": {
                    "type": "string",
                    "description": "Kubernetes resource manifest in YAML or JSON format"
                }
            },
            "required": ["manifest"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ApplyManifestArgs = parse_args(self.name(), ctx.args)?;
        debug!("k8s_apply_manifest");

        // Parse the manifest into a DynamicObject.
        let resource: kube::api::DynamicObject = serde_yaml::from_str(&args.manifest)
            .or_else(|_| serde_json::from_str(&args.manifest))
            .map_err(|e| anyhow::anyhow!("Failed to parse manifest: {e}"))?;

        // Extract type metadata to build the API resource.
        let types = resource
            .types
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Manifest must include apiVersion and kind"))?;

        let api_resource =
            kube::api::ApiResource::from_gvk(&kube::api::GroupVersionKind::try_from(types)?);

        let name = resource
            .metadata
            .name
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Manifest must include metadata.name"))?;

        let ns = resource
            .metadata
            .namespace
            .as_deref()
            .unwrap_or(&self.default_namespace);

        let api: Api<kube::api::DynamicObject> =
            Api::namespaced_with((*self.client).clone(), ns, &api_resource);

        let pp = PatchParams::apply("flashmind").force();
        let result = api.patch(name, &pp, &Patch::Apply(&resource)).await?;

        let kind = &types.kind;
        let result_name = result.metadata.name.as_deref().unwrap_or(name);
        let result_ns = result.metadata.namespace.as_deref().unwrap_or(ns);

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("{kind}/{result_name} applied in namespace {result_ns}"),
        ))
    }

    fn humanize(&self, _args: &Value) -> String {
        "Applying Kubernetes manifest".into()
    }
}

// ---------------------------------------------------------------------------
// k8s_delete_resource
// ---------------------------------------------------------------------------

/// Delete a Kubernetes resource by kind and name.
pub struct K8sDeleteResourceTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct DeleteResourceArgs {
    kind: String,
    name: String,
    namespace: Option<String>,
}

/// Map a user-friendly kind name to (group, version, kind).
fn kind_to_gvk(kind: &str) -> anyhow::Result<kube::api::GroupVersionKind> {
    let (group, version, k) = match kind.to_lowercase().as_str() {
        "pod" | "pods" => ("", "v1", "Pod"),
        "service" | "services" | "svc" => ("", "v1", "Service"),
        "deployment" | "deployments" | "deploy" => ("apps", "v1", "Deployment"),
        "statefulset" | "statefulsets" | "sts" => ("apps", "v1", "StatefulSet"),
        "daemonset" | "daemonsets" | "ds" => ("apps", "v1", "DaemonSet"),
        "replicaset" | "replicasets" | "rs" => ("apps", "v1", "ReplicaSet"),
        "configmap" | "configmaps" | "cm" => ("", "v1", "ConfigMap"),
        "secret" | "secrets" => ("", "v1", "Secret"),
        "namespace" | "namespaces" | "ns" => ("", "v1", "Namespace"),
        "job" | "jobs" => ("batch", "v1", "Job"),
        "cronjob" | "cronjobs" => ("batch", "v1", "CronJob"),
        "ingress" | "ingresses" | "ing" => ("networking.k8s.io", "v1", "Ingress"),
        "persistentvolumeclaim" | "persistentvolumeclaims" | "pvc" => {
            ("", "v1", "PersistentVolumeClaim")
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unknown resource kind: {other}. Supported: pod, service, deployment, \
                 statefulset, daemonset, replicaset, configmap, secret, namespace, \
                 job, cronjob, ingress, pvc"
            ));
        }
    };
    Ok(kube::api::GroupVersionKind::gvk(group, version, k))
}

#[async_trait]
impl Tool for K8sDeleteResourceTool {
    fn name(&self) -> &str {
        "k8s_delete_resource"
    }

    fn description(&self) -> &str {
        "Delete a Kubernetes resource by kind and name. Supports pods, \
         deployments, services, statefulsets, configmaps, secrets, jobs, \
         cronjobs, ingresses, and PVCs."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "Resource kind (e.g. pod, deployment, service, configmap)"
                },
                "name": {
                    "type": "string",
                    "description": "Resource name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                }
            },
            "required": ["kind", "name"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: DeleteResourceArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(kind = %args.kind, name = %args.name, namespace = ns, "k8s_delete_resource");

        let gvk = kind_to_gvk(&args.kind)?;
        let api_resource = kube::api::ApiResource::from_gvk(&gvk);
        let api: Api<kube::api::DynamicObject> =
            Api::namespaced_with((*self.client).clone(), ns, &api_resource);

        api.delete(&args.name, &DeleteParams::default()).await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!("{}/{} deleted from namespace {ns}", args.kind, args.name),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let kind = args
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("resource");
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        format!("Deleting {kind}/{name}")
    }
}

// ---------------------------------------------------------------------------
// k8s_scale_deployment
// ---------------------------------------------------------------------------

/// Scale a deployment to a given number of replicas.
pub struct K8sScaleDeploymentTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ScaleDeploymentArgs {
    name: String,
    namespace: Option<String>,
    replicas: i32,
}

#[async_trait]
impl Tool for K8sScaleDeploymentTool {
    fn name(&self) -> &str {
        "k8s_scale_deployment"
    }

    fn description(&self) -> &str {
        "Scale a Kubernetes deployment to a specified number of replicas."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Deployment name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                },
                "replicas": {
                    "type": "integer",
                    "description": "Desired number of replicas"
                }
            },
            "required": ["name", "replicas"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ScaleDeploymentArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(
            deployment = %args.name,
            namespace = ns,
            replicas = args.replicas,
            "k8s_scale_deployment"
        );

        let api: Api<Deployment> = Api::namespaced((*self.client).clone(), ns);
        let patch = json!({
            "spec": {
                "replicas": args.replicas
            }
        });
        api.patch(&args.name, &PatchParams::default(), &Patch::Merge(patch))
            .await?;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Deployment {} scaled to {} replicas in namespace {ns}",
                args.name, args.replicas
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let replicas = args.get("replicas").and_then(|v| v.as_i64()).unwrap_or(0);
        format!("Scaling {name} to {replicas} replicas")
    }
}

// ---------------------------------------------------------------------------
// k8s_exec_in_pod
// ---------------------------------------------------------------------------

/// Execute a command inside a running pod container.
pub struct K8sExecInPodTool {
    /// Shared Kubernetes API client.
    pub client: Arc<Client>,
    /// Fallback namespace when the caller omits one.
    pub default_namespace: String,
}

#[derive(Deserialize)]
struct ExecInPodArgs {
    name: String,
    namespace: Option<String>,
    container: Option<String>,
    command: Vec<String>,
}

#[async_trait]
impl Tool for K8sExecInPodTool {
    fn name(&self) -> &str {
        "k8s_exec_in_pod"
    }

    fn description(&self) -> &str {
        "Execute a command inside a running Kubernetes pod. Returns combined \
         stdout and stderr output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Pod name"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (defaults to configured namespace)"
                },
                "container": {
                    "type": "string",
                    "description": "Container name (required for multi-container pods)"
                },
                "command": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Command to execute as an array (e.g. [\"ls\", \"-la\"])"
                }
            },
            "required": ["name", "command"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        if ctx.cancel_token().is_cancelled() {
            return Ok(ToolResult::success(ctx.tool_call_id, "Cancelled."));
        }
        let args: ExecInPodArgs = parse_args(self.name(), ctx.args)?;
        let ns = args.namespace.as_deref().unwrap_or(&self.default_namespace);
        debug!(
            pod = %args.name,
            namespace = ns,
            command = ?args.command,
            "k8s_exec_in_pod"
        );

        let api: Api<Pod> = Api::namespaced((*self.client).clone(), ns);
        let mut ap = kube::api::AttachParams::default().stdout(true).stderr(true);
        if let Some(container) = &args.container {
            ap = ap.container(container);
        }

        let mut attached = api.exec(&args.name, &args.command, &ap).await?;

        // Collect output.
        let stdout = if let Some(stdout_reader) = attached.stdout() {
            let mut buf = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(
                &mut tokio::io::BufReader::new(stdout_reader),
                &mut buf,
            )
            .await?;
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::new()
        };

        let stderr = if let Some(stderr_reader) = attached.stderr() {
            let mut buf = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(
                &mut tokio::io::BufReader::new(stderr_reader),
                &mut buf,
            )
            .await?;
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::new()
        };

        attached.join().await?;

        let mut out = String::new();
        if !stdout.is_empty() {
            out.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            let _ = write!(out, "[stderr]\n{stderr}");
        }
        if out.is_empty() {
            out.push_str("Command completed with no output.");
        }

        Ok(ToolResult::success(ctx.tool_call_id, out))
    }

    fn humanize(&self, args: &Value) -> String {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let cmd = args
            .get("command")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "...".into());
        format!("Executing `{cmd}` in pod {name}")
    }
}
