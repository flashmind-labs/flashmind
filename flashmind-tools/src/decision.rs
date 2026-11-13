//! Typed decisions across supported decision providers.

use std::sync::Arc;

use async_trait::async_trait;
pub use flashmind_llm::decisions::DecisionProvider;
use flashmind_llm::decisions::{DecisionApi, NativeDecisionProvider};
use flashmind_llm::{JevRequest, OpenRouterProvider};
use flashmind_types::ApiKey;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use serde_json::{Value, json};

/// Agent tool sharing the request and response contract of `flsh decide`.
pub struct DecisionTool {
    providers: Vec<(ApiKey, Arc<dyn DecisionProvider>)>,
}

impl DecisionTool {
    pub fn new(provider: Arc<dyn DecisionProvider>) -> Self {
        Self {
            providers: vec![(provider.api_key(), provider)],
        }
    }

    pub(crate) fn from_keys(
        mut read_key: impl FnMut(ApiKey) -> Option<String>,
    ) -> anyhow::Result<Option<Self>> {
        let mut providers: Vec<(ApiKey, Arc<dyn DecisionProvider>)> = Vec::new();
        for key_kind in [ApiKey::OpenRouter, ApiKey::OpenAi, ApiKey::Gemini] {
            let Some(key) = read_key(key_kind).filter(|key| !key.trim().is_empty()) else {
                continue;
            };
            let provider: Arc<dyn DecisionProvider> = match key_kind {
                ApiKey::OpenRouter => Arc::new(OpenRouterProvider::new(key)),
                ApiKey::OpenAi => Arc::new(NativeDecisionProvider::new(DecisionApi::OpenAi, key)?),
                ApiKey::Gemini => Arc::new(NativeDecisionProvider::new(DecisionApi::Gemini, key)?),
                _ => continue,
            };
            providers.push((key_kind, provider));
        }
        Ok((!providers.is_empty()).then_some(Self { providers }))
    }
}

#[async_trait]
impl Tool for DecisionTool {
    fn name(&self) -> &str {
        "decision"
    }

    fn description(&self) -> &str {
        "Run typed decisions on context supplied as state. Use OpenRouter for any namespaced decision model (such as cloudflare/clef), or native OpenAI or Gemini. Defaults to the first configured provider in that order and its default model. Ask named yes/no (noul), classification (choice), or ordered rating (score) questions. Returns JSON with typed answers, model, provider, ID, and usage."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["openrouter", "openai", "gemini"], "description": "Backend to charge. Must have a configured API key."},
                "model": {"type": "string", "description": "Decision model ID. Defaults: OpenRouter jev-latest, OpenAI gpt-6-luna, Gemini gemini-decision-1."},
                "state": {"description": "Context to evaluate: a string, object, or array", "type": ["string", "object", "array"]},
                "questions": {
                    "type": "object",
                    "description": "Questions keyed by caller-chosen identifiers",
                    "additionalProperties": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"const": "noul"},
                                    "instructions": {"type": "string"},
                                    "criteria": {"type": ["object", "null"], "additionalProperties": {"type": "string"}}
                                },
                                "required": ["type", "instructions"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"const": "choice"},
                                    "instructions": {"type": "string"},
                                    "criteria": {"type": "object", "description": "Named options mapped to their meanings", "additionalProperties": {"type": "string"}}
                                },
                                "required": ["type", "instructions", "criteria"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"const": "score"},
                                    "instructions": {"type": "string"},
                                    "criteria": {"type": "array", "description": "Ordered rating legend", "items": {"type": "string"}}
                                },
                                "required": ["type", "instructions", "criteria"]
                            }
                        ]
                    }
                }
            },
            "required": ["state", "questions"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(120)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let provider = match ctx.args.get("provider") {
            Some(value) => {
                let name = value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("provider must be a string"))?;
                self.providers
                    .iter()
                    .find(|(key, _)| key.to_string() == name)
                    .map(|(_, provider)| provider)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Decision provider '{name}' is unsupported or has no configured API key"
                        )
                    })?
            }
            None => &self.providers[0].1,
        };
        let mut args = ctx.args.clone();
        if let Some(object) = args.as_object_mut() {
            object
                .entry("model")
                .or_insert_with(|| json!(provider.default_model()));
        }
        let request: JevRequest = serde_json::from_value(args)?;
        let response = provider.decide(request).await?;
        Ok(ToolResult::success(
            ctx.tool_call_id,
            serde_json::to_string(&response)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_llm::{JEV_LATEST_MODEL, JevQuestion, JevResponse};
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    #[derive(Default)]
    struct FakeProvider {
        requests: Mutex<Vec<JevRequest>>,
        fail: bool,
    }

    #[async_trait]
    impl DecisionProvider for FakeProvider {
        fn api_key(&self) -> ApiKey {
            ApiKey::OpenRouter
        }

        async fn decide(&self, request: JevRequest) -> anyhow::Result<JevResponse> {
            if self.fail {
                anyhow::bail!("decision unavailable");
            }
            let model = request.model.clone();
            self.requests.lock().unwrap().push(request);
            Ok(JevResponse {
                id: Some("decision-id".into()),
                model,
                provider: Some("test".into()),
                answers: [
                    ("invoice".into(), json!(true)),
                    ("category".into(), json!("invoice")),
                    ("priority".into(), json!(1)),
                ]
                .into(),
                usage: Default::default(),
            })
        }
    }

    fn request() -> Value {
        json!({
            "state": {"document": "Invoice for services"},
            "questions": {
                "invoice": {"type": "noul", "instructions": "Is this an invoice?"},
                "category": {"type": "choice", "instructions": "Classify", "criteria": {"invoice": "An invoice", "other": "Something else"}},
                "priority": {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "high"]}
            }
        })
    }

    #[tokio::test]
    async fn forwards_typed_questions_and_returns_json_response() {
        let provider = Arc::new(FakeProvider::default());
        let tool = DecisionTool::new(provider.clone());
        let cancel = CancellationToken::new();
        let result = tool
            .execute(ToolContext::new("call", request(), None, &cancel))
            .await
            .unwrap();
        assert!(result.is_success());
        assert_eq!(result.tool_call_id(), "call");
        let response: JevResponse = serde_json::from_str(&result.output()).unwrap();
        assert_eq!(response.answers["invoice"], true);
        assert_eq!(response.answers["category"], "invoice");
        assert_eq!(response.answers["priority"], 1);
        assert_eq!(response.id.as_deref(), Some("decision-id"));
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[0].model, JEV_LATEST_MODEL);
        assert_eq!(requests[0].state, request()["state"]);
        assert!(matches!(
            requests[0].questions["invoice"],
            JevQuestion::Noul { .. }
        ));
        assert!(matches!(
            requests[0].questions["category"],
            JevQuestion::Choice { .. }
        ));
        assert!(matches!(
            requests[0].questions["priority"],
            JevQuestion::Score { .. }
        ));
    }

    #[tokio::test]
    async fn preserves_explicit_model_and_rejects_invalid_questions_before_dispatch() {
        let provider = Arc::new(FakeProvider::default());
        let tool = DecisionTool::new(provider.clone());
        let cancel = CancellationToken::new();
        let mut args = request();
        args["model"] = json!("jev-1.13");
        tool.execute(ToolContext::new("call", args, None, &cancel))
            .await
            .unwrap();
        assert_eq!(provider.requests.lock().unwrap()[0].model, "jev-1.13");
        for args in [
            json!(null),
            json!({"state": "context"}),
            json!({"state": [], "questions": {"bad": {"type": "choice", "instructions": "Classify"}}}),
        ] {
            assert!(
                tool.execute(ToolContext::new("call", args, None, &cancel))
                    .await
                    .is_err()
            );
        }
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn propagates_provider_errors() {
        let tool = DecisionTool::new(Arc::new(FakeProvider {
            fail: true,
            ..Default::default()
        }));
        let cancel = CancellationToken::new();
        let error = tool
            .execute(ToolContext::new("call", request(), None, &cancel))
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "decision unavailable");
    }

    #[test]
    fn schema_exposes_native_request_and_all_question_types() {
        let tool = DecisionTool::new(Arc::new(FakeProvider::default()));
        let schema = tool.parameters();
        assert_eq!(tool.name(), "decision");
        assert_eq!(schema["required"], json!(["state", "questions"]));
        assert_eq!(
            schema["properties"]["provider"]["enum"],
            json!(["openrouter", "openai", "gemini"])
        );
        let variants = schema["properties"]["questions"]["additionalProperties"]["oneOf"]
            .as_array()
            .unwrap();
        assert_eq!(
            variants
                .iter()
                .map(|v| v["properties"]["type"]["const"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["noul", "choice", "score"]
        );
    }
    #[test]
    fn key_selection_collects_supported_providers_in_priority_order() {
        let tool = DecisionTool::from_keys(|_| Some("test-key".into()))
            .unwrap()
            .unwrap();
        assert_eq!(
            tool.providers
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>(),
            [ApiKey::OpenRouter, ApiKey::OpenAi, ApiKey::Gemini]
        );
        for (key, expected_model) in [
            (ApiKey::OpenRouter, JEV_LATEST_MODEL),
            (ApiKey::OpenAi, "gpt-6-luna"),
            (ApiKey::Gemini, "gemini-decision-1"),
        ] {
            let tool =
                DecisionTool::from_keys(|candidate| (candidate == key).then(|| "test-key".into()))
                    .unwrap()
                    .unwrap();
            assert_eq!(tool.providers[0].0, key);
            assert_eq!(tool.providers[0].1.default_model(), expected_model);
        }
    }

    #[tokio::test]
    async fn explicit_provider_dispatches_without_fallback() {
        let first = Arc::new(FakeProvider::default());
        let second = Arc::new(FakeProvider::default());
        let tool = DecisionTool {
            providers: vec![
                (ApiKey::OpenRouter, first.clone()),
                (ApiKey::Gemini, second.clone()),
            ],
        };
        let cancel = CancellationToken::new();
        let mut args = request();
        args["provider"] = json!("gemini");
        tool.execute(ToolContext::new("call", args, None, &cancel))
            .await
            .unwrap();
        assert!(first.requests.lock().unwrap().is_empty());
        assert_eq!(second.requests.lock().unwrap().len(), 1);
        for value in [json!("openai"), json!("unsupported"), json!(null)] {
            let mut args = request();
            args["provider"] = value;
            assert!(
                tool.execute(ToolContext::new("call", args, None, &cancel))
                    .await
                    .is_err()
            );
        }
        assert!(first.requests.lock().unwrap().is_empty());
        assert_eq!(second.requests.lock().unwrap().len(), 1);
    }
}
