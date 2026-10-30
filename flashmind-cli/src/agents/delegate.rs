//! Resolve CLI personas before delegating to the shared subagent tool.

use std::sync::Arc;

use async_trait::async_trait;
use flashmind_types::AgentLlmConfig;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

use super::AgentStore;

pub(crate) struct PersonaDelegate {
    inner: Arc<dyn Tool>,
    store: AgentStore,
    default_llm: AgentLlmConfig,
}

impl PersonaDelegate {
    pub(crate) fn new(
        inner: Arc<dyn Tool>,
        store: AgentStore,
        default_llm: AgentLlmConfig,
    ) -> Self {
        Self {
            inner,
            store,
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
        let catalog = self.store.catalog();
        schema["properties"]["persona"] = json!({
            "type": "string",
            "description": format!("Optional named persona (case-insensitive). Uses its system prompt and default model/reasoning. An explicit model overrides its model and must use the current provider. Cannot combine with system_prompt. Available personas:\n{catalog}")
        });
        schema
    }

    async fn execute(&self, mut ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: PersonaArgs = ctx.parse_args("delegate")?;
        let Some(name) = args.persona else {
            return self.inner.execute(ctx).await;
        };
        let Some(def) = self.store.get(&name) else {
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
    use flashmind_tools::subagent::DelegateTool;
    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, LlmProvider, Provider, ReasoningLevel,
        Role, StreamEvent,
    };
    use flashmind_types::{Model, ModelCapabilities, SessionId};
    use tempfile::TempDir;

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

    struct Fixture {
        temp: TempDir,
        tool: PersonaDelegate,
        manager: Arc<AgentManager>,
        provider: Arc<RecordingProvider>,
    }

    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let imports = temp.path().join(".agents");
            let saved = temp.path().join("saved");
            fs::create_dir_all(&imports).unwrap();
            fs::create_dir_all(&saved).unwrap();
            fs::write(imports.join("reviewer.md"), "---\nname: reviewer\ndescription: Review code carefully\nmodel: ollama:review-model\ntools: Read\n---\nYou are a meticulous reviewer.").unwrap();
            let store = AgentStore {
                dir: saved,
                markdown_dirs: vec![imports],
            };
            let manager = Arc::new(AgentManager::new(2, 1));
            let provider = Arc::new(RecordingProvider(Mutex::new(Vec::new())));
            let llm = AgentLlmConfig::new("ollama:parent".parse().unwrap());
            let inner = Arc::new(
                DelegateTool::new(manager.clone(), provider.clone())
                    .with_llm(llm.clone())
                    .with_instructions("Shared skill instructions.".into()),
            );
            let tool = PersonaDelegate::new(inner, store, llm);
            Self {
                temp,
                tool,
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
        let mut def = fixture.tool.store.get("reviewer").unwrap();
        def.system_prompt = "Saved personality.".into();
        def.reasoning = Some(ReasoningLevel::Low);
        fixture.tool.store.save(&def).unwrap();
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
}
