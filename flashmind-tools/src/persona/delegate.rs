//! Start workers from personas.

use std::sync::Arc;

use async_trait::async_trait;
use flashmind_core::SpawnBuilder;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::{AgentLlmConfig, LlmProvider, Model};
use serde::Deserialize;
use serde_json::{Value, json};

use super::catalog::{PersonaSource, catalog};
use super::definition::{Persona, PersonaError};

impl Persona {
    /// A [`SpawnBuilder`] for a worker running this persona.
    ///
    /// The worker starts from `llm` (the parent's routing and cache settings)
    /// with the persona's reasoning level and model applied on top. Add
    /// instructions, tools or a name to the builder before passing it to
    /// [`AgentManager::spawn`](flashmind_core::AgentManager::spawn).
    pub fn spawn_builder(
        &self,
        task: impl Into<String>,
        provider: Arc<dyn LlmProvider>,
        llm: &AgentLlmConfig,
    ) -> Result<SpawnBuilder, PersonaError> {
        let mut llm = llm.clone();
        if let Some(reasoning) = self.reasoning {
            llm.reasoning = reasoning;
        }
        let mut builder = SpawnBuilder::new(task, provider)
            .persona(self.name.clone())
            .system_prompt(self.system_prompt.clone())
            .llm(llm);
        if let Some(model) = &self.model {
            let parsed: Model = model.parse().map_err(|error| PersonaError::Model {
                model: model.clone(),
                reason: format!("{error}"),
            })?;
            builder = builder.model(parsed);
        }
        Ok(builder)
    }
}

/// Wraps the `delegate` tool so the model can pick a persona.
///
/// Adds a `persona` argument to the wrapped tool's schema listing the
/// available personas. When it is set, the persona's system prompt, model and
/// reasoning level are filled in before the call reaches the wrapped tool; an
/// explicit `model` argument still wins. Personas are looked up on every
/// call, so edits apply without a restart. Notices from importing the persona
/// are appended to a successful result.
///
/// `inner` is normally [`DelegateTool`](crate::subagent::DelegateTool).
pub struct PersonaDelegate {
    inner: Arc<dyn Tool>,
    personas: Arc<dyn PersonaSource>,
    default_llm: AgentLlmConfig,
}

impl PersonaDelegate {
    /// Wrap `inner`, reading personas from `personas`. `default_llm` is used
    /// when a call carries no parent configuration.
    pub fn new(
        inner: Arc<dyn Tool>,
        personas: Arc<dyn PersonaSource>,
        default_llm: AgentLlmConfig,
    ) -> Self {
        Self {
            inner,
            personas,
            default_llm,
        }
    }
}

#[derive(Deserialize)]
struct PersonaArgs {
    persona: Option<String>,
}

#[async_trait]
impl Tool for PersonaDelegate {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        "Spawn a background subagent for a task. Select a named persona to use its personality and instructions. Use agent_status or agent_wait to follow its progress."
    }

    fn parameters(&self) -> Value {
        let mut schema = self.inner.parameters();
        let catalog = catalog(&self.personas.personas());
        schema["properties"]["persona"] = json!({
            "type": "string",
            "description": format!("Optional named persona (case-insensitive). Uses its system prompt and default model/reasoning. An explicit model overrides its model and must use the current provider. Cannot combine with system_prompt. Available personas:\n{catalog}")
        });
        schema
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        crate::utils::validate_tool_args(&ctx.args, &self.parameters())?;
        let args: PersonaArgs = ctx.parse_args("delegate")?;
        let Some(name) = args.persona else {
            return self.inner.execute(ctx).await;
        };
        let Some(def) = self.personas.get(&name) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Unknown persona: {name}"),
            ));
        };
        if ctx
            .args
            .get("system_prompt")
            .is_some_and(|value| !value.is_null())
        {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Use either persona or system_prompt, not both.",
            ));
        }
        ctx.args["system_prompt"] = json!(def.system_prompt);
        if ctx.args.get("model").is_none_or(Value::is_null)
            && let Some(model) = def.model
        {
            ctx.args["model"] = json!(model);
        }
        // Preserve the active session's routing and cache settings.
        let mut llm = ctx.llm.unwrap_or(&self.default_llm).clone();
        if let Some(reasoning) = def.reasoning {
            llm.reasoning = reasoning;
        }
        let notices = def.notices;
        let result = self.inner.execute(ctx.with_llm(&llm)).await?;
        if result.is_success() && !notices.is_empty() {
            return Ok(ToolResult::success(
                result.tool_call_id(),
                format!("{}\n{}", result.output(), notices.join("\n")),
            ));
        }
        Ok(result)
    }

    fn humanize(&self, args: &Value) -> String {
        self.inner.humanize(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    use flashmind_core::{AgentManager, CancellationToken};
    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, LlmProvider, Model, ModelCapabilities,
        Provider, ReasoningLevel, Role, SessionId, StreamEvent,
    };
    use tempfile::TempDir;

    use crate::persona::{MarkdownPersonas, merge};
    use crate::subagent::DelegateTool;

    struct RecordingProvider(Mutex<Vec<CompletionRequest>>);

    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        async fn capabilities(&self, _model: &Model) -> ModelCapabilities {
            ModelCapabilities {
                reasoning: true,
                ..Default::default()
            }
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.0.lock().unwrap().push(request);
            Box::pin(futures_util::stream::iter([
                Ok(StreamEvent::ContentDelta("Reviewed.".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    /// Markdown imports with host-saved personas layered on top.
    struct Layered {
        markdown: MarkdownPersonas,
        saved: Mutex<Vec<Persona>>,
    }

    impl PersonaSource for Layered {
        fn personas(&self) -> Vec<Persona> {
            let saved = self.saved.lock().unwrap().clone();
            merge(self.markdown.load().into_iter().chain(saved))
        }
    }

    struct Fixture {
        temp: TempDir,
        tool: PersonaDelegate,
        personas: Arc<Layered>,
        manager: Arc<AgentManager>,
        provider: Arc<RecordingProvider>,
    }

    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let imports = temp.path().join(".agents");
            fs::create_dir_all(&imports).unwrap();
            fs::write(imports.join("reviewer.md"), "---\nname: reviewer\ndescription: Review code carefully\nmodel: ollama:review-model\ntools: Read\n---\nYou are a meticulous reviewer.").unwrap();
            let personas = Arc::new(Layered {
                markdown: MarkdownPersonas::new([imports]),
                saved: Mutex::new(Vec::new()),
            });
            let manager = Arc::new(AgentManager::new(2, 1));
            let provider = Arc::new(RecordingProvider(Mutex::new(Vec::new())));
            let llm = AgentLlmConfig::new("ollama:parent".parse().unwrap());
            let inner = Arc::new(
                DelegateTool::new(manager.clone(), provider.clone())
                    .with_llm(llm.clone())
                    .with_instructions("Shared skill instructions.".into()),
            );
            let tool = PersonaDelegate::new(inner, personas.clone(), llm);
            Self {
                temp,
                tool,
                personas,
                manager,
                provider,
            }
        }

        async fn execute(&self, args: Value) -> ToolResult {
            let cancel = CancellationToken::new();
            self.tool
                .execute(ToolContext::new("call", args, None, &cancel))
                .await
                .unwrap()
        }
    }

    #[tokio::test]
    async fn markdown_persona_reaches_the_spawned_worker() {
        let fixture = Fixture::new();
        let schema = fixture.tool.parameters().to_string();
        assert!(schema.contains("reviewer: Review code carefully"));
        let result = fixture
            .execute(json!({"task": "Review this change", "persona": "REVIEWER", "name": "worker"}))
            .await;
        assert!(result.is_success(), "{}", result.output());
        assert!(result.output().contains("Unsupported agent settings"));
        let id = fixture.manager.resolve("worker").await.unwrap();
        assert_eq!(
            fixture.manager.conversation(id).unwrap().persona.as_deref(),
            Some("REVIEWER")
        );
        fixture.manager.wait(id).await.unwrap();
        let requests = fixture.provider.0.lock().unwrap();
        let request = &requests[0];
        assert_eq!(request.model.to_string(), "ollama:review-model");
        let system = &request
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .unwrap()
            .content;
        assert!(system.starts_with("You are a meticulous reviewer."));
        assert!(system.contains("Shared skill instructions."));
        assert_eq!(
            request
                .messages
                .iter()
                .find(|m| m.role == Role::User)
                .unwrap()
                .content,
            "Review this change"
        );
    }

    #[tokio::test]
    async fn saved_persona_keeps_active_routing_and_allows_model_override() {
        let fixture = Fixture::new();
        let mut def = fixture.personas.get("reviewer").unwrap();
        def.system_prompt = "Saved personality.".into();
        def.reasoning = Some(ReasoningLevel::Low);
        fixture.personas.saved.lock().unwrap().push(def);
        let llm = AgentLlmConfig::new("ollama:active-parent".parse().unwrap())
            .with_session_id("active-session".parse::<SessionId>().unwrap());
        let cancel = CancellationToken::new();
        let result = fixture.tool.execute(ToolContext::new("call", json!({
            "task": "Review", "persona": "reviewer", "name": "worker", "model": "ollama:override"
        }), None, &cancel).with_llm(&llm)).await.unwrap();
        assert!(result.is_success(), "{}", result.output());
        fixture
            .manager
            .wait(fixture.manager.resolve("worker").await.unwrap())
            .await
            .unwrap();
        let requests = fixture.provider.0.lock().unwrap();
        let request = &requests[0];
        assert_eq!(request.model.to_string(), "ollama:override");
        assert_eq!(request.reasoning, ReasoningLevel::Low);
        assert_eq!(request.session_id, llm.session_id);
        assert!(
            request
                .messages
                .iter()
                .any(|m| m.role == Role::System && m.content.starts_with("Saved personality."))
        );
    }

    #[tokio::test]
    async fn invalid_selections_do_not_spawn_workers() {
        let fixture = Fixture::new();
        for args in [
            json!({"task": "Review", "persona": "missing"}),
            json!({"task": "Review", "persona": "reviewer", "system_prompt": "Other prompt"}),
            json!({"task": "Review", "persona": "reviewer", "model": "openrouter:example/other"}),
        ] {
            let result = fixture.execute(args).await;
            assert!(!result.is_success(), "{}", result.output());
            assert!(fixture.manager.all_statuses().await.is_empty());
        }
    }

    #[tokio::test]
    async fn reloads_edited_personas_and_preserves_plain_delegation() {
        let fixture = Fixture::new();
        fs::write(fixture.temp.path().join(".agents/reviewer.md"), "---\nname: reviewer\ndescription: Updated description\nmodel: inherit\n---\nUpdated personality.").unwrap();
        assert!(
            fixture
                .tool
                .parameters()
                .to_string()
                .contains("Updated description")
        );
        for (name, args) in [
            (
                "persona-worker",
                json!({"task": "Review", "persona": "reviewer", "name": "persona-worker"}),
            ),
            (
                "plain-worker",
                json!({"task": "Review", "system_prompt": "Ad hoc prompt.", "name": "plain-worker"}),
            ),
        ] {
            let result = fixture.execute(args).await;
            assert!(result.is_success(), "{}", result.output());
            fixture
                .manager
                .wait(fixture.manager.resolve(name).await.unwrap())
                .await
                .unwrap();
        }
        let requests = fixture.provider.0.lock().unwrap();
        for (request, prompt) in requests
            .iter()
            .zip(["Updated personality.", "Ad hoc prompt."])
        {
            assert_eq!(request.model.to_string(), "ollama:parent");
            assert!(
                request
                    .messages
                    .iter()
                    .any(|m| m.role == Role::System && m.content.starts_with(prompt))
            );
        }
    }

    #[tokio::test]
    async fn spawn_builder_runs_the_persona_through_the_manager() {
        let fixture = Fixture::new();
        let mut def = fixture.personas.get("reviewer").unwrap();
        def.reasoning = Some(ReasoningLevel::Low);
        let llm = AgentLlmConfig::new("ollama:parent".parse().unwrap());
        let builder = def
            .spawn_builder("Review", fixture.provider.clone(), &llm)
            .unwrap()
            .name("direct");
        let id = fixture.manager.spawn(builder).await.unwrap();
        assert_eq!(
            fixture.manager.conversation(id).unwrap().persona.as_deref(),
            Some("reviewer")
        );
        fixture.manager.wait(id).await.unwrap();
        let requests = fixture.provider.0.lock().unwrap();
        let request = &requests[0];
        assert_eq!(request.model.to_string(), "ollama:review-model");
        assert_eq!(request.reasoning, ReasoningLevel::Low);
        assert!(
            request.messages.iter().any(|m| m.role == Role::System
                && m.content.starts_with("You are a meticulous reviewer."))
        );

        def.model = Some("no-provider".into());
        assert!(matches!(
            def.spawn_builder("Review", fixture.provider.clone(), &llm),
            Err(PersonaError::Model { .. })
        ));
    }
}
