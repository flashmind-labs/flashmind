//! Fluent builder for spawning agents.

use std::sync::Arc;

use flashmind_types::{AgentLlmConfig, LlmProvider, Model, ToolRegistry};

// ---------------------------------------------------------------------------
// SpawnBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for configuring and spawning an agent.
///
/// At minimum, a `task` description and `provider` are required. Everything
/// else has sensible defaults inherited from the parent agent.
///
/// # Example
///
/// ```rust,ignore
/// let id = manager.spawn(
///     SpawnBuilder::new("research pricing models", provider.clone())
///         .system_prompt("You are a research assistant.")
///         .max_iterations(50)
/// ).await?;
/// ```
pub struct SpawnBuilder {
    pub(crate) task: String,
    pub(crate) provider: Arc<dyn LlmProvider>,
    pub(crate) name: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) model: Option<Model>,
    pub(crate) llm: Option<AgentLlmConfig>,
    pub(crate) tools: Option<ToolRegistry>,
    pub(crate) strip_prefixes: Vec<String>,
    pub(crate) max_iterations: Option<usize>,
}

impl SpawnBuilder {
    /// Create a builder with the required task description and provider.
    pub fn new(task: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            task: task.into(),
            provider,
            name: None,
            system_prompt: None,
            model: None,
            llm: None,
            tools: None,
            strip_prefixes: Vec::new(),
            max_iterations: Some(100),
        }
    }

    /// Set a human-readable name for the agent (e.g. "Pacifist").
    ///
    /// Named agents can be addressed by name in `communicate`, `agent_status`,
    /// `agent_wait`, and `agent_terminate`. Names must be unique among active agents.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a custom system prompt for the agent.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Override the LLM model (uses parent's default otherwise).
    pub fn model(mut self, model: Model) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the full LLM configuration (model, temperature, reasoning).
    pub fn llm(mut self, llm: AgentLlmConfig) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Provide a specific tool registry. If not set, inherits the parent's
    /// registry (with any `strip_prefixes` applied).
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Remove tools matching these name prefixes from the inherited registry.
    ///
    /// Only applies when `tools()` is not explicitly set — the agent
    /// inherits the parent's registry minus the stripped prefixes.
    pub fn strip_prefixes(mut self, prefixes: Vec<String>) -> Self {
        self.strip_prefixes = prefixes;
        self
    }

    /// Maximum number of iterations before the agent stops.
    /// Defaults to 100. Pass `None` for unlimited.
    pub fn max_iterations(mut self, max: impl Into<Option<usize>>) -> Self {
        self.max_iterations = max.into();
        self
    }
}
