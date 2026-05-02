//! Tool trait, registry, context, and result types for agent function calling.
//!
//! These types form the interface between the agent runtime and tool
//! implementations, enabling tools to be defined in external crates
//! without depending on the binary.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Tool`] | Trait for tools callable by the agent (name, description, params, execute) |
//! | [`ToolRegistry`] | Registry of available tools with aliases, gating, and on-demand loading |
//! | [`ToolContext`] | Per-invocation context passed to `execute()` — args, cancellation, event sink |
//! | [`ToolResult`] | Result of a tool execution returned to the LLM |
//! | [`FileDiff`] | A file diff produced by a tool |
//! | [`ForbiddenCmd`] | Server-side command restrictions (regex-based) |
//!
//! # Design principles
//!
//! - Tools are `Send + Sync` so they can be shared behind `Arc`
//! - The registry supports **aliases** (`alias("ls", "file_list")`) transparently
//! - **On-demand loading** exposes only essential tools initially; others activate via `tool_list`/`tool_load`
//! - **Gating** blocks destructive tools in read-only modes with path-based exemptions

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::event::{AgentEvent, Source};
use crate::llm::ToolDefinition;

// ---------------------------------------------------------------------------
// ForbiddenCmd
// ---------------------------------------------------------------------------

/// A forbidden command configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenCmd {
    /// Regex pattern to match against the full command line (command + args).
    pub command: String,
    /// Reason shown to user when blocked.
    pub reason: String,
    /// If true, inject a warning message and let the LLM reconsider instead of rejecting.
    /// Like tool-aware RAG, this will retry the turn with additional context.
    #[serde(default)]
    pub reconsider: bool,
}

impl ForbiddenCmd {
    /// Validate the forbidden command config.
    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.command.is_empty() {
            errors.push("forbidden command name cannot be empty".to_string());
        }
        if regex::Regex::new(&self.command).is_err() {
            errors.push(format!("invalid regex for name: '{}'", self.command));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// FileDiff
// ---------------------------------------------------------------------------

/// A file diff produced by a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: String,
    pub diff: String,
}

// ---------------------------------------------------------------------------
// ToolResult (tool execution result — distinct from message::ToolResult)
// ---------------------------------------------------------------------------

/// Result of a tool execution returned to the LLM.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
    pub success: bool,
    /// Source URLs referenced by the tool (e.g. search result links).
    pub sources: Vec<Source>,
    /// File diffs produced by the tool.
    pub diffs: Vec<FileDiff>,
    /// When true, the agent loop should stop and return this result to the
    /// caller for interactive handling (e.g. plan approval, choice picker).
    /// The `output` contains proposal data as JSON for the caller to interpret.
    pub interrupt: bool,
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            success: true,
            sources: Vec::new(),
            diffs: Vec::new(),
            interrupt: false,
        }
    }

    /// Create a failed tool result.
    pub fn failure(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            success: false,
            sources: Vec::new(),
            diffs: Vec::new(),
            interrupt: false,
        }
    }

    /// Create a successful tool result with file diffs.
    pub fn success_with_diffs(
        tool_call_id: &str,
        output: impl Into<String>,
        diffs: Vec<FileDiff>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            success: true,
            sources: Vec::new(),
            diffs,
            interrupt: false,
        }
    }

    /// Create a result that interrupts the agent loop for interactive handling.
    /// The `output` should contain proposal data as JSON.
    pub fn interrupt(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            success: true,
            sources: Vec::new(),
            diffs: Vec::new(),
            interrupt: true,
        }
    }

    /// Add source URLs to this result.
    pub fn with_sources(mut self, sources: Vec<Source>) -> Self {
        self.sources = sources;
        self
    }
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------

/// Deserialize tool arguments from a serde_json::Value into a typed struct.
/// Returns a typed error mentioning which tool failed to parse.
pub fn parse_args<T: DeserializeOwned>(tool_name: &str, args: Value) -> anyhow::Result<T> {
    serde_json::from_value(args)
        .map_err(|e| anyhow::anyhow!("Tool '{}': Invalid arguments: {}", tool_name, e))
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// Trait for tools that can be invoked by the agent.
///
/// Implementations define a name, description, JSON schema parameters, and async execution
/// logic. Tools are registered in a [`ToolRegistry`] and become available to the LLM during
/// conversation turns via function calling.
///
/// # Required methods
///
/// - [`name`](Self::name) — unique identifier used by the LLM to invoke the tool
/// - [`description`](Self::description) — shown to the LLM to decide when to call the tool
/// - [`parameters`](Self::parameters) — JSON Schema object describing expected arguments
/// - [`execute`](Self::execute) — async logic that runs when the tool is called
/// - [`humanize`](Self::humanize) — generates a display-friendly summary of a tool call
///
/// # Optional overrides
///
/// - [`timeout_secs`](Self::timeout_secs) — per-tool timeout (default: uses global default)
/// - [`max_output_bytes`](Self::max_output_bytes) — output size limit (default: 32 KiB)
/// - [`max_output_lines`](Self::max_output_lines) — output line count limit (default: 1000)
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique tool name (used for invocation). Must match the name the LLM sees in its function definitions.
    fn name(&self) -> &str;
    /// Human-readable description for the LLM. Used to help the model decide when to invoke this tool.
    fn description(&self) -> &str;
    /// JSON Schema object describing the tool's parameters.
    fn parameters(&self) -> Value;
    /// Execute the tool with given arguments. Returns a [`ToolResult`] on success.
    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult>;
    /// Optional per-tool timeout in seconds. Returns `None` to use the global default.
    fn timeout_secs(&self) -> Option<u64> {
        None
    }

    /// Maximum allowed output size in bytes. Tools that produce large payloads
    /// (e.g. base64-encoded images, arbitrary MCP responses) should return
    /// `usize::MAX` to bypass the limit.
    fn max_output_bytes(&self) -> usize {
        32 * 1024
    }

    /// Maximum allowed output line count. Tools that produce large payloads
    /// should return `usize::MAX` to bypass the limit.
    fn max_output_lines(&self) -> usize {
        1000
    }

    /// Generate a human-readable description of what this tool call does,
    /// suitable for display in a TUI or chat card (e.g. "Reading file Cargo.toml").
    fn humanize(&self, args: &Value) -> String;
}

// ---------------------------------------------------------------------------
// ToolContext
// ---------------------------------------------------------------------------

/// Per-invocation context passed to [`Tool::execute`] methods.
///
/// Carries everything a tool needs: the call ID for building results, raw JSON arguments,
/// cancellation support, and an event sink for streaming status updates back to the agent loop.
///
/// # Key fields
///
/// | Field | Purpose |
/// |-------|---------|
/// | [`tool_call_id`](Self::tool_call_id) | Echo into [`ToolResult::tool_call_id`] so results match calls |
/// | [`args`](Self::args) | Raw JSON arguments sent by the LLM — parse with [`parse_args`](Self::parse_args) |
/// | [`scope`](Self::scope) | Session identifier (e.g. `"telegram:12345"`) used for memory/tag scoping |
/// | [`working_dir`](Self::working_dir) | Chat workspace root; relative paths resolve against this |
/// | [`prompt_tokens`](Self::prompt_tokens) | Current prompt token count for context-aware decisions |
/// | [`context_window`](Self::context_window) | Active model's context window size |
pub struct ToolContext<'a> {
    /// The unique ID for this tool call. Echo into [`ToolResult::tool_call_id`] when returning.
    pub tool_call_id: &'a str,
    /// Raw JSON arguments from the LLM. Use [`parse_args`](Self::parse_args) to deserialize.
    pub args: Value,
    /// Scope identifier for the current agent session (e.g. `"telegram:12345"`).
    pub scope: &'a str,
    /// Resolved username from the identity system (e.g. `"dario"`).
    pub username: Option<&'a str>,
    /// Working directory for the current agent (typically chat workspace). Relative paths resolve here.
    pub working_dir: Option<&'a PathBuf>,
    /// Cancellation token for cooperative cancellation. Access via [`cancel_token`](Self::cancel_token) or [`child_token`](Self::child_token).
    cancel_token: &'a CancellationToken,
    /// Event sink for emitting `[AgentEvent]` updates during long-running operations.
    response_tx: &'a mpsc::Sender<AgentEvent>,
    /// Last known prompt token count (set by agent loop for interactive tools).
    pub prompt_tokens: u32,
    /// Context window size for the active model.
    pub context_window: u32,
}

impl<'a> ToolContext<'a> {
    /// Create a new tool context.
    pub fn new(
        tool_call_id: &'a str,
        args: Value,
        scope: &'a str,
        working_dir: Option<&'a PathBuf>,
        cancel_token: &'a CancellationToken,
        response_tx: &'a mpsc::Sender<AgentEvent>,
    ) -> Self {
        Self {
            tool_call_id,
            args,
            scope,
            username: None,
            working_dir,
            cancel_token,
            response_tx,
            prompt_tokens: 0,
            context_window: 0,
        }
    }

    /// Set the resolved username for this tool context.
    pub fn with_username(mut self, username: Option<&'a str>) -> Self {
        self.username = username;
        self
    }

    /// Attach prompt token usage and context window info. Called by the agent loop.
    pub fn with_context_usage(mut self, prompt_tokens: u32, context_window: u32) -> Self {
        self.prompt_tokens = prompt_tokens;
        self.context_window = context_window;
        self
    }

    /// Parse the raw JSON args into a typed struct. Fails with a descriptive error if invalid.
    pub fn parse_args<T: DeserializeOwned>(&self, tool_name: &str) -> anyhow::Result<T> {
        parse_args(tool_name, self.args.clone())
    }

    /// Access the cancellation token for cooperative cancellation checks.
    ///
    /// For long-running operations, prefer [`child_token`](Self::child_token) to get
    /// a derived token that can be passed to async sub-operations.
    pub fn cancel_token(&self) -> &CancellationToken {
        self.cancel_token
    }

    /// Create a child cancellation token. Cancels when the parent turn is cancelled.
    /// Use this for spawned tasks so they're automatically cancelled when the user aborts.
    pub fn child_token(&self) -> CancellationToken {
        self.cancel_token.child_token()
    }

    /// Access the event sink for emitting [`AgentEvent`] updates during execution.
    pub fn response_tx(&self) -> &mpsc::Sender<AgentEvent> {
        self.response_tx
    }

    /// Check whether an absolute path starts with the working directory.
    ///
    /// Returns `Err(ToolResult)` with a failure message nudging the agent to use
    /// relative paths instead of absolute ones rooted in the cwd.
    pub fn check_absolute_path(&self, path: &str) -> std::result::Result<(), ToolResult> {
        if let Some(wdir) = self.working_dir {
            let p = Path::new(path);
            if p.is_absolute() && p.starts_with(wdir) {
                let rel = p.strip_prefix(wdir).unwrap_or(p);
                let suggestion = if rel.as_os_str().is_empty() {
                    ".".to_string()
                } else {
                    rel.display().to_string()
                };

                return Err(ToolResult::failure(
                    self.tool_call_id,
                    format!(
                        "You are already in `{}`. Use the relative path `{}` instead.",
                        wdir.display(),
                        suggestion,
                    ),
                ));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// Registry of tools available to an agent.
///
/// When on-demand loading is enabled ([`enable_on_demand`](Self::enable_on_demand)),
/// only active tools appear in [`definitions`](Self::definitions).
/// [`get`](Self::get) always resolves any registered tool regardless.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    aliases: HashMap<String, String>,
    /// When `Some`, only these tools are exposed in `definitions()`.
    active: Option<HashSet<String>>,
    /// Forbidden shell commands (enforced server-side for both local and remote).
    forbidden: Vec<ForbiddenCmd>,
    /// Tools gated in the current mode. Calls to these tools return the error
    /// message instead of executing, unless the path argument (if any) falls
    /// under an allowed prefix (e.g. `docs/` in plan mode).
    gated: HashMap<String, String>,
    /// Path prefixes exempt from the gate (relative to working dir).
    gate_allowed_prefixes: Vec<PathBuf>,
}

impl ToolRegistry {
    /// Create an empty registry with no tools, aliases, or forbidden commands.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            aliases: HashMap::new(),
            active: None,
            forbidden: Vec::new(),
            gated: HashMap::new(),
            gate_allowed_prefixes: Vec::new(),
        }
    }

    /// Set the forbidden commands list (checked server-side before routing to client).
    pub fn set_forbidden(&mut self, forbidden: Vec<ForbiddenCmd>) {
        self.forbidden = forbidden;
    }

    /// Returns the forbidden commands list for server-side enforcement.
    pub fn forbidden(&self) -> &[ForbiddenCmd] {
        &self.forbidden
    }

    /// Add a forbidden command pattern to the list (session-only).
    pub fn add_forbidden(&mut self, cmd: ForbiddenCmd) {
        self.forbidden.push(cmd);
    }

    /// Gate a set of tools so they return an error message instead of executing.
    /// Path-based tools (file_write, str_replace, etc.) are exempt when the
    /// target path falls under one of the allowed prefixes.
    pub fn set_gate(&mut self, tools: &[&str], message: String, allowed_prefixes: Vec<PathBuf>) {
        self.gated.clear();
        for name in tools {
            self.gated.insert(name.to_string(), message.clone());
        }
        self.gate_allowed_prefixes = allowed_prefixes;
    }

    /// Remove all tool gates.
    pub fn clear_gate(&mut self) {
        self.gated.clear();
        self.gate_allowed_prefixes.clear();
    }

    /// Check if a tool call is gated. Returns `Some(error_message)` if the tool
    /// should be blocked, `None` if it should proceed.
    pub fn check_gate(&self, tool_name: &str, args: &serde_json::Value) -> Option<&str> {
        let msg = self.gated.get(tool_name)?;

        if !self.gate_allowed_prefixes.is_empty()
            && let Some(path_str) = args
                .get("path")
                .or_else(|| args.get("file"))
                .and_then(|v| v.as_str())
        {
            let path = Path::new(path_str);
            if self
                .gate_allowed_prefixes
                .iter()
                .any(|prefix| path.starts_with(prefix))
            {
                return None;
            }
        }

        Some(msg.as_str())
    }

    /// Remove a forbidden pattern by exact match. Returns true if removed.
    pub fn remove_forbidden(&mut self, pattern: &str) -> bool {
        let before = self.forbidden.len();
        self.forbidden.retain(|fc| fc.command != pattern);
        self.forbidden.len() != before
    }

    /// Register a tool (replaces existing tool with same name).
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        tracing::debug!(tool = tool.name(), "registered tool");
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Remove a tool by name. Returns true if the tool was present.
    pub fn remove(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Remove every tool whose name starts with one of `prefixes`. Aliases
    /// pointing at removed tools are also dropped.
    pub fn strip_prefixes(&mut self, prefixes: &[&str]) {
        let to_remove: Vec<String> = self
            .tools
            .keys()
            .filter(|name| prefixes.iter().any(|p| name.starts_with(p)))
            .cloned()
            .collect();
        for name in to_remove {
            self.tools.remove(&name);
        }
        let tools = &self.tools;
        self.aliases.retain(|_, target| tools.contains_key(target));
    }

    /// Register an alias so that lookups for `alias` resolve to `target`.
    /// Aliases are not exposed in `definitions()` or `list()`.
    pub fn alias(&mut self, alias: &str, target: &str) {
        tracing::debug!(alias, target, "registered tool alias");
        self.aliases.insert(alias.to_string(), target.to_string());
    }

    /// Look up a tool by name (checks aliases if direct lookup fails).
    /// Always resolves any registered tool, even if not in the active set.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        let result = self.tools.get(name).or_else(|| {
            if let Some(target) = self.aliases.get(name) {
                tracing::debug!(
                    alias = name,
                    target = target.as_str(),
                    "resolved tool alias"
                );
                self.tools.get(target)
            } else {
                None
            }
        });

        if result.is_none() {
            tracing::warn!(tool = name, "tool not found in registry");
        }

        result
    }

    /// Returns true if a tool with the given name is registered (without logging).
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// List all registered tool names (sorted).
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List all registered tools with their names and descriptions (sorted by name).
    pub fn list_with_descriptions(&self) -> Vec<(&str, &str)> {
        let mut items: Vec<(&str, &str)> = self
            .tools
            .values()
            .map(|t| (t.name(), t.description()))
            .collect();
        items.sort_by_key(|(name, _)| *name);
        items
    }

    /// Enable on-demand tool loading. Only `essential` tools (plus
    /// `tool_list` and `tool_load`) will appear in `definitions()`.
    /// Other tools remain registered and executable via `get()`.
    pub fn enable_on_demand(&mut self, essential: &[&str]) {
        tracing::debug!(
            essential_count = essential.len(),
            total_registered = self.tools.len(),
            "enabling on-demand tool loading"
        );
        let mut active: HashSet<String> = essential.iter().map(|s| s.to_string()).collect();
        active.insert("tool_list".to_string());
        active.insert("tool_load".to_string());
        self.active = Some(active);
    }

    /// Disable on-demand tool loading. All registered tools will appear
    /// in `definitions()` again.
    pub fn disable_on_demand(&mut self) {
        tracing::debug!("disabling on-demand tool loading");
        self.active = None;
    }

    /// Activate a tool so it appears in subsequent `definitions()` calls.
    /// No-op if on-demand loading is not enabled.
    pub fn activate(&mut self, name: &str) {
        if let Some(ref mut active) = self.active {
            tracing::debug!(tool = name, "activating on-demand tool");
            active.insert(name.to_string());
        }
    }

    /// Returns true if on-demand loading is enabled.
    pub fn is_on_demand(&self) -> bool {
        self.active.is_some()
    }

    /// Register proxy tool definitions from a remote client.
    /// These appear in `definitions()` so the LLM can call them.
    pub fn register_proxy_defs(&mut self, defs: &[ToolDefinition]) {
        for def in defs {
            if self.contains(&def.name) {
                continue; // Don't override server-side tools
            }
            tracing::debug!(tool = def.name, "registered proxy tool from client");
            self.register(Arc::new(ProxyTool {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.parameters.clone(),
            }));
        }
    }

    /// Generate LLM tool definitions (filtered by active set if enabled).
    /// When on-demand loading is active, synthetic definitions for `tool_list`
    /// and `tool_load` are injected automatically.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .values()
            .filter(|t| self.active.as_ref().is_none_or(|a| a.contains(t.name())))
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect();

        // Inject meta-tool definitions when on-demand is active
        if self.active.is_some() {
            defs.push(ToolDefinition {
                name: "tool_list".into(),
                description: "List all available tools with descriptions. Use this to discover \
                              tools you can load."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            });
            defs.push(ToolDefinition {
                name: "tool_load".into(),
                description: "Load additional tools to make them available for use.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tools": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Names of tools to activate (from tool_list output)"
                        }
                    },
                    "required": ["tools"]
                }),
            });
        }

        defs.sort_by(|a, b| a.name.cmp(&b.name));

        tracing::debug!(
            count = defs.len(),
            on_demand = self.active.is_some(),
            "generated tool definitions"
        );

        defs
    }
}

/// A stub tool registered for client-provided proxy tools.
struct ProxyTool {
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl Tool for ProxyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::failure(
            ctx.tool_call_id,
            format!(
                "Tool '{}' should be executed by the remote client",
                self.name
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        // Summarize proxy tool arguments (truncated JSON keys).
        let Some(obj) = args.as_object() else {
            return String::new();
        };
        let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        let summary = keys.join(", ");
        if summary.len() > 100 {
            format!("{}...", &summary[..97])
        } else {
            summary
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool(&'static str);

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "dummy"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({})
        }
        async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success(ctx.tool_call_id, "ok"))
        }

        fn humanize(&self, _args: &Value) -> String {
            "Dummy tool".to_string()
        }
    }

    #[test]
    fn test_registry_empty() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_registry_definitions_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.definitions().is_empty());
    }

    #[test]
    fn test_on_demand_filtering() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool("exec")));
        registry.register(Arc::new(DummyTool("slack_react")));
        registry.register(Arc::new(DummyTool("browser")));

        // Before enabling: all tools visible
        assert_eq!(registry.definitions().len(), 3);
        assert!(!registry.is_on_demand());

        // Enable on-demand with only "exec" as essential
        registry.enable_on_demand(&["exec"]);
        assert!(registry.is_on_demand());

        // "exec" + synthetic tool_list/tool_load visible
        let defs = registry.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"exec"));
        assert!(names.contains(&"tool_list"));
        assert!(names.contains(&"tool_load"));
        assert!(!names.contains(&"browser"));
        assert!(!names.contains(&"slack_react"));
        assert_eq!(names.len(), 3); // exec + tool_list + tool_load

        // Activate browser
        registry.activate("browser");
        let defs = registry.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"browser"));
        assert!(names.contains(&"exec"));

        // get() still resolves non-active tools
        assert!(registry.get("slack_react").is_some());
    }

    #[test]
    fn test_gate_blocks_tool() {
        let mut registry = ToolRegistry::new();
        registry.set_gate(&["file_write"], "blocked".into(), vec![]);

        let args = serde_json::json!({"path": "/src/main.rs", "content": "x"});
        assert_eq!(registry.check_gate("file_write", &args), Some("blocked"));
        assert_eq!(registry.check_gate("file_read", &args), None);
    }

    #[test]
    fn test_gate_allows_prefix() {
        let mut registry = ToolRegistry::new();
        registry.set_gate(
            &["file_write"],
            "blocked".into(),
            vec![PathBuf::from("/project/docs")],
        );

        let allowed = serde_json::json!({"path": "/project/docs/plan.md", "content": "x"});
        assert_eq!(registry.check_gate("file_write", &allowed), None);

        let blocked = serde_json::json!({"path": "/project/src/main.rs", "content": "x"});
        assert_eq!(registry.check_gate("file_write", &blocked), Some("blocked"));
    }

    #[test]
    fn test_gate_clear() {
        let mut registry = ToolRegistry::new();
        registry.set_gate(&["file_write"], "blocked".into(), vec![]);
        registry.clear_gate();

        let args = serde_json::json!({"path": "/src/main.rs"});
        assert_eq!(registry.check_gate("file_write", &args), None);
    }
}
