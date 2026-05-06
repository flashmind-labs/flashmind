//! Tool trait, registry, context, and result types for agent function calling.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`Tool`] | Trait for tools callable by the agent (name, description, params, execute) |
//! | [`ToolRegistry`] | Registry of available tools with aliases |
//! | [`ToolContext`] | Per-invocation context passed to `execute()` |
//! | [`ToolResult`] | Result of a tool execution returned to the LLM |
//! | [`FileDiff`] | A file diff produced by a tool |
//! | [`ForbiddenCmd`] | Command restriction pattern (regex-based) |

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::event::Source;
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

/// A single line in a file diff — either added or removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffLine {
    Added { line: u64, content: String },
    Removed { line: u64, content: String },
}

/// A file diff produced by a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: String,
    pub diff: Vec<DiffLine>,
}

// ---------------------------------------------------------------------------
// ToolResult (tool execution result — distinct from message::ToolResult)
// ---------------------------------------------------------------------------

/// Result of a tool execution, returned by [`Tool::execute`] and fed back to
/// the LLM as a `"role": "tool"` message.
///
/// Three variants cover the full lifecycle of a tool call:
///
/// - **`Success`** — The tool ran without error. The `output` field contains
///   the text payload sent back to the model. Optional `sources` (URLs the
///   agent can cite) and `diffs` (unified diffs for file-mutating tools) may
///   be attached.
/// - **`Failure`** — The tool encountered an error (invalid arguments, I/O
///   failure, etc.). The `output` field carries a human- and LLM-readable
///   error description. Unlike `Success`, no sources or diffs are included.
/// - **`Interrupt`** — The tool needs user interaction before continuing
///   (e.g. plan approval, choice picker). The agent loop should halt and
///   surface the `output` (structured as JSON) to the caller for handling.
///
/// # Construction helpers
///
/// Prefer the associated constructors on the `impl` block rather than building
/// variants directly:
///
/// ```ignore
/// use flashmind_types::tool::ToolResult;
///
/// // Basic success
/// ToolResult::success(call_id, "file written")
///
/// // Failure with error message
/// ToolResult::failure(call_id, "permission denied")
///
/// // Success with file diffs attached
/// ToolResult::success_with_diffs(call_id, "replaced", vec![diff])
///
/// // Interrupt for interactive prompt
/// ToolResult::interrupt(call_id, json!({"type": "choice", ...}).to_string())
///
/// // Chain .with_sources() onto any Success
/// ToolResult::success(call_id, results).with_sources(srcs)
/// ```
///
/// # Relation to [`message::ToolResult`](crate::message::ToolResult)
///
/// This type lives in the *tool* module and represents the raw outcome of
/// executing a tool. It is converted into a [`Message`](crate::message::Message)
/// with `role: "tool"` (via [`AgentEvent::ToolResult`](crate::event::AgentEvent::ToolResult))
/// before being appended to the conversation history sent to the LLM.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Tool executed successfully.
    ///
    /// - `tool_call_id` — echoes the ID from the original tool call so the LLM
    ///   can match this result to the invocation.
    /// - `output` — the primary text payload returned to the model. Keep it
    ///   concise but informative; this is what the agent reads next.
    /// - `sources` — optional list of URLs/references that support the output
    ///   (used by search and web-fetch tools).
    /// - `diffs` — optional list of per-file unified diffs produced by this
    ///   tool call (used by file-write and text-replace tools).
    Success {
        tool_call_id: String,
        output: String,
        sources: Vec<Source>,
        diffs: Vec<FileDiff>,
    },
    /// Tool execution failed.
    ///
    /// The `output` field should contain a clear error message describing what
    /// went wrong and, when possible, how the agent might recover (e.g. "file
    /// not found — check the path").
    Failure {
        tool_call_id: String,
        output: String,
    },
    /// The agent loop should stop and return this result to the caller for
    /// interactive handling (e.g. plan approval, choice picker). The `output`
    /// contains proposal data as JSON for the caller to interpret.
    Interrupt {
        tool_call_id: String,
        output: String,
    },
}

impl ToolResult {
    /// Create a basic success result with no sources or diffs.
    pub fn success(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self::Success {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            sources: Vec::new(),
            diffs: Vec::new(),
        }
    }

    /// Create a failure result with an error message.
    pub fn failure(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self::Failure {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
        }
    }

    /// Create a success result that includes file diffs for tools that
    /// mutate content (file write, text replace, etc.).
    pub fn success_with_diffs(
        tool_call_id: &str,
        output: impl Into<String>,
        diffs: Vec<FileDiff>,
    ) -> Self {
        Self::Success {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
            sources: Vec::new(),
            diffs,
        }
    }

    /// Create an interrupt result that pauses the agent loop for interactive
    /// user handling (plan approval, choice selection, etc.).
    pub fn interrupt(tool_call_id: &str, output: impl Into<String>) -> Self {
        Self::Interrupt {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
        }
    }

    /// Attach source URLs to a `Success` result. Returns `self` unchanged if
    /// the variant is not `Success`.
    ///
    /// This is designed for method chaining:
    /// ```ignore
    /// ToolResult::success(id, "results").with_sources(srcs)
    /// ```
    pub fn with_sources(mut self, sources: Vec<Source>) -> Self {
        if let Self::Success {
            sources: ref mut s, ..
        } = self
        {
            *s = sources;
        }
        self
    }

    /// Return the tool call ID echoed from the original invocation.
    pub fn tool_call_id(&self) -> &str {
        match self {
            Self::Success { tool_call_id, .. }
            | Self::Failure { tool_call_id, .. }
            | Self::Interrupt { tool_call_id, .. } => tool_call_id,
        }
    }

    /// Return the text output payload (success result, error message, or
    /// interrupt data).
    pub fn output(&self) -> &str {
        match self {
            Self::Success { output, .. }
            | Self::Failure { output, .. }
            | Self::Interrupt { output, .. } => output,
        }
    }

    /// Check whether this result represents a successful execution.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Check whether this result is an interrupt requiring user interaction.
    pub fn is_interrupt(&self) -> bool {
        matches!(self, Self::Interrupt { .. })
    }

    /// Return the attached source references, or an empty slice for
    /// non-success variants.
    pub fn sources(&self) -> &[Source] {
        match self {
            Self::Success { sources, .. } => sources,
            _ => &[],
        }
    }

    /// Return the attached file diffs, or an empty slice for non-success
    /// variants.
    pub fn diffs(&self) -> &[FileDiff] {
        match self {
            Self::Success { diffs, .. } => diffs,
            _ => &[],
        }
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
///
/// # Optional overrides
///
/// - [`humanize`](Self::humanize) — display-friendly summary (default: comma-separated arg keys)
/// - [`timeout_secs`](Self::timeout_secs) — per-tool timeout (default: uses global default)
/// - [`max_output_bytes`](Self::max_output_bytes) — output size limit (default: 256 KiB)
/// - [`max_output_lines`](Self::max_output_lines) — output line count limit (default: 10,000)
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
        256 * 1024
    }

    /// Maximum allowed output line count. Tools that produce large payloads
    /// should return `usize::MAX` to bypass the limit.
    fn max_output_lines(&self) -> usize {
        10_000
    }

    /// Generate a human-readable description of what this tool call does,
    /// suitable for display in a TUI or chat card (e.g. "Reading file Cargo.toml").
    fn humanize(&self, args: &Value) -> String {
        let Some(obj) = args.as_object() else {
            return self.name().to_string();
        };
        let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        if keys.is_empty() {
            return self.name().to_string();
        }
        let summary = keys.join(", ");
        if summary.len() > 80 {
            let truncated: String = summary.chars().take(77).collect();
            format!("{truncated}…")
        } else {
            summary
        }
    }
}

// ---------------------------------------------------------------------------
// ToolContext
// ---------------------------------------------------------------------------

/// Per-invocation context passed to [`Tool::execute`] methods.
///
/// Carries everything a tool needs: the call ID for building results, raw JSON arguments,
/// cancellation support, and the working directory for path resolution.
///
/// # Key fields
///
/// | Field | Purpose |
/// |-------|---------|
/// | [`tool_call_id`](Self::tool_call_id) | Echo into [`ToolResult::tool_call_id`] so results match calls |
/// | [`args`](Self::args) | Raw JSON arguments sent by the LLM — parse with [`parse_args`](Self::parse_args) |
/// | [`working_dir`](Self::working_dir) | Chat workspace root; relative paths resolve against this |
pub struct ToolContext<'a> {
    /// The unique ID for this tool call. Echo into [`ToolResult::tool_call_id`] when returning.
    pub tool_call_id: &'a str,
    /// Raw JSON arguments from the LLM. Use [`parse_args`](Self::parse_args) to deserialize.
    pub args: Value,
    /// Working directory for the current agent (typically chat workspace). Relative paths resolve here.
    pub working_dir: Option<&'a PathBuf>,
    /// Cancellation token for cooperative cancellation. Access via [`cancel_token`](Self::cancel_token) or [`child_token`](Self::child_token).
    cancel_token: &'a CancellationToken,
}

impl<'a> ToolContext<'a> {
    pub fn new(
        tool_call_id: &'a str,
        args: Value,
        working_dir: Option<&'a PathBuf>,
        cancel_token: &'a CancellationToken,
    ) -> Self {
        Self {
            tool_call_id,
            args,
            working_dir,
            cancel_token,
        }
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
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    aliases: HashMap<String, String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            aliases: HashMap::new(),
        }
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

    /// Execute a tool call from the LLM.
    ///
    /// Looks up the tool by name, builds a [`ToolContext`], and runs it. Returns
    /// a [`ToolResult::Failure`] if the tool is not found or if execution errors.
    /// If the tool defines [`timeout_secs`](Tool::timeout_secs), the call is
    /// wrapped in a timeout and the cancel token is cancelled on expiry.
    pub async fn execute(
        &self,
        call: &crate::ToolCall,
        working_dir: Option<&PathBuf>,
        cancel_token: &CancellationToken,
    ) -> ToolResult {
        let Some(tool) = self.get(&call.name) else {
            return ToolResult::failure(&call.id, format!("Unknown tool: {}", call.name));
        };

        let child_token = cancel_token.child_token();
        let ctx = ToolContext::new(&call.id, call.arguments.clone(), working_dir, &child_token);
        let fut = tool.execute(ctx);

        let outcome = if let Some(secs) = tool.timeout_secs() {
            match tokio::time::timeout(std::time::Duration::from_secs(secs), fut).await {
                Ok(r) => r,
                Err(_) => {
                    child_token.cancel();
                    return ToolResult::failure(
                        &call.id,
                        format!("Tool '{}' timed out after {secs}s", call.name),
                    );
                }
            }
        } else {
            fut.await
        };

        match outcome {
            Ok(result) => result,
            Err(e) => ToolResult::failure(&call.id, format!("Tool error: {e:#}")),
        }
    }

    /// Human-readable summary of a tool call's arguments.
    pub fn humanize(&self, call: &crate::ToolCall) -> String {
        self.get(&call.name)
            .map(|t| t.humanize(&call.arguments))
            .unwrap_or_else(|| call.name.clone())
    }

    /// Generate LLM tool definitions for all registered tools.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect();

        defs.sort_by(|a, b| a.name.cmp(&b.name));

        tracing::debug!(count = defs.len(), "generated tool definitions");

        defs
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
    fn test_registry_lookup() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool("exec")));
        registry.register(Arc::new(DummyTool("slack_react")));
        registry.register(Arc::new(DummyTool("browser")));

        assert_eq!(registry.definitions().len(), 3);

        // get() resolves tools by name
        assert!(registry.get("slack_react").is_some());
    }
}
