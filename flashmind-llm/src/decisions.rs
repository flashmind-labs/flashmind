//! Decision backends sharing the existing Jev request and response types.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, ensure};
use async_trait::async_trait;
use flashmind_types::{ApiKey, LlmError, LlmErrorKind};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::http::{http_client_builder, send_with_retry};
use crate::{JEV_LATEST_MODEL, JevQuestion, JevRequest, JevResponse, JevUsage, OpenRouterProvider};

/// A backend for the shared typed decision contract.
#[async_trait]
pub trait DecisionProvider: Send + Sync {
    fn api_key(&self) -> ApiKey;

    fn default_model(&self) -> &str {
        JEV_LATEST_MODEL
    }

    async fn decide(&self, request: JevRequest) -> anyhow::Result<JevResponse>;
}

#[async_trait]
impl DecisionProvider for OpenRouterProvider {
    fn api_key(&self) -> ApiKey {
        ApiKey::OpenRouter
    }

    async fn decide(&self, request: JevRequest) -> anyhow::Result<JevResponse> {
        OpenRouterProvider::decide(self, request).await
    }
}

/// Native APIs whose request envelopes differ from OpenRouter's.
#[derive(Clone, Copy)]
pub enum DecisionApi {
    OpenAi,
    Gemini,
}

impl DecisionApi {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
        }
    }
}

/// Native decision client with provider-specific wire conversion.
pub struct NativeDecisionProvider {
    api: DecisionApi,
    client: Client,
    key: SecretString,
    endpoint: String,
}

impl NativeDecisionProvider {
    pub fn new(api: DecisionApi, key: String) -> anyhow::Result<Self> {
        let endpoint = match api {
            DecisionApi::OpenAi => "https://api.openai.com/v1/decisions",
            DecisionApi::Gemini => "https://generativelanguage.googleapis.com/v1beta/decisions",
        };
        Ok(Self {
            api,
            client: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(120))
                .build()?,
            key: key.into(),
            endpoint: endpoint.into(),
        })
    }

    /// Override the endpoint for tests or compatible proxies.
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }

    fn payload(&self, request: &JevRequest) -> anyhow::Result<Value> {
        match self.api {
            DecisionApi::Gemini => {
                // Objects are context data, not Gemini Content blocks. Serialize
                // them as text rather than interpreting arbitrary keys as media.
                let input = match &request.state {
                    Value::String(text) => text.clone(),
                    state => serde_json::to_string(state)?,
                };
                let mut questions = serde_json::to_value(&request.questions)?;
                for (name, question) in &request.questions {
                    if let JevQuestion::Noul {
                        instructions,
                        criteria: Some(criteria),
                    } = question
                    {
                        questions[name]["instructions"] = json!(format!(
                            "{instructions}\nCriteria: {}",
                            serde_json::to_string(criteria)?
                        ));
                        if let Some(object) = questions[name].as_object_mut() {
                            object.remove("criteria");
                        }
                    }
                }
                Ok(json!({"model": request.model, "input": input, "questions": questions}))
            }
            DecisionApi::OpenAi => {
                let input = match &request.state {
                    Value::String(text) => text.clone(),
                    state => serde_json::to_string(state)?,
                };
                let questions: Vec<Value> = request.questions.iter().map(|(name, question)| {
                    match question {
                        JevQuestion::Noul { instructions, criteria } => {
                            // Predicates have no criteria field. Include explicit
                            // yes/no meanings in the question instructions.
                            let instructions = match criteria {
                                Some(criteria) => format!("{instructions}\nCriteria: {}", serde_json::to_string(criteria)?),
                                None => instructions.clone(),
                            };
                            Ok(json!({"name": name, "type": "predicate", "instructions": instructions}))
                        }
                        JevQuestion::Choice { instructions, criteria } => {
                            let choices: Vec<Value> = criteria.iter().map(|(value, description)| json!({"value": value, "description": description})).collect();
                            Ok(json!({"name": name, "type": "choice", "instructions": instructions, "choices": choices}))
                        }
                        JevQuestion::Score { instructions, criteria } => {
                            let levels: Vec<Value> = criteria.iter().map(|label| json!({"label": label})).collect();
                            Ok(json!({"name": name, "type": "score", "instructions": instructions, "levels": levels}))
                        }
                    }
                }).collect::<anyhow::Result<_>>()?;
                Ok(json!({"model": request.model, "input": input, "questions": questions}))
            }
        }
    }

    fn parse_response(&self, response: Value, request: &JevRequest) -> anyhow::Result<JevResponse> {
        match self.api {
            DecisionApi::OpenAi => {
                #[derive(Deserialize)]
                struct Response {
                    model: String,
                    answers: Vec<Value>,
                    #[serde(default)]
                    id: Option<String>,
                    usage: JevUsage,
                }
                let response: Response = serde_json::from_value(response)?;
                let mut answers = BTreeMap::new();
                for answer in response.answers {
                    let name = answer
                        .get("name")
                        .and_then(Value::as_str)
                        .context("decision answer is missing its question name")?
                        .to_owned();
                    ensure!(
                        request.questions.contains_key(&name),
                        "decision answer has an unknown question name"
                    );
                    ensure!(!answers.contains_key(&name), "duplicate decision answer");
                    answers.insert(name, answer);
                }
                ensure!(
                    answers.len() == request.questions.len(),
                    "decision response is missing answers"
                );
                Ok(JevResponse {
                    id: response.id,
                    model: response.model,
                    provider: Some(self.api.name().into()),
                    answers,
                    usage: response.usage,
                })
            }
            DecisionApi::Gemini => {
                #[derive(Deserialize)]
                struct Usage {
                    total_input_tokens: u32,
                    total_output_tokens: u32,
                }
                #[derive(Deserialize)]
                struct Response {
                    answers: BTreeMap<String, Value>,
                    usage: Usage,
                }
                let response: Response = serde_json::from_value(response)?;
                ensure!(
                    response.answers.keys().eq(request.questions.keys()),
                    "decision response question names do not match request"
                );
                Ok(JevResponse {
                    id: None,
                    model: request.model.clone(),
                    provider: Some(self.api.name().into()),
                    answers: response.answers,
                    usage: JevUsage {
                        input_tokens: response.usage.total_input_tokens,
                        output_tokens: response.usage.total_output_tokens,
                        cost: None,
                    },
                })
            }
        }
    }
}

#[async_trait]
impl DecisionProvider for NativeDecisionProvider {
    fn api_key(&self) -> ApiKey {
        match self.api {
            DecisionApi::OpenAi => ApiKey::OpenAi,
            DecisionApi::Gemini => ApiKey::Gemini,
        }
    }

    fn default_model(&self) -> &str {
        match self.api {
            DecisionApi::OpenAi => "gpt-6-luna",
            DecisionApi::Gemini => "gemini-decision-1",
        }
    }

    async fn decide(&self, request: JevRequest) -> anyhow::Result<JevResponse> {
        let payload = self.payload(&request)?;
        let response = send_with_retry(|| {
            let builder = self.client.post(&self.endpoint).json(&payload);
            match self.api {
                DecisionApi::OpenAi => builder.bearer_auth(self.key.expose_secret()),
                DecisionApi::Gemini => builder.header("x-goog-api-key", self.key.expose_secret()),
            }
        })
        .await?;
        let status = response.status();
        if !status.is_success() {
            // Never copy upstream bodies, which can echo credentials or state.
            let kind = if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                LlmErrorKind::RateLimited
            } else {
                LlmErrorKind::Other
            };
            return Err(LlmError::new(
                kind,
                format!(
                    "{} decision error: HTTP {}",
                    self.api.name(),
                    status.as_u16()
                ),
            )
            .into());
        }
        self.parse_response(response.json().await?, &request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::task::JoinHandle;

    fn request() -> JevRequest {
        serde_json::from_value(json!({
            "model": "test-model", "state": {"ticket": "Payment failed"},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "Blocking", "false": "Can wait"}},
                "team": {"type": "choice", "instructions": "Who handles this?", "criteria": {"billing": "Payment problems", "other": "Other problems"}},
                "severity": {"type": "score", "instructions": "Rate severity", "criteria": ["low", "high"]}
            }
        })).unwrap()
    }

    async fn server(status: &str, body: Value) -> (String, JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/decisions", listener.local_addr().unwrap());
        let body = body.to_string();
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut data = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let n = stream.read(&mut buffer).await.unwrap();
                assert!(n > 0, "request ended before its body arrived");
                data.extend_from_slice(&buffer[..n]);
                let text = String::from_utf8_lossy(&data);
                if let Some(end) = text.find("\r\n\r\n") {
                    let length: usize = text[..end]
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if data.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            stream.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8(data).unwrap()
        });
        (endpoint, task)
    }

    #[tokio::test]
    async fn openai_translates_questions_and_preserves_native_answers_and_usage() {
        let body = json!({"model": "test-model", "answers": [
            {"name": "urgent", "type": "predicate", "probability": 0.9},
            {"name": "team", "type": "choice", "choice": "billing", "probabilities": [{"value": "billing", "probability": 0.8}]},
            {"name": "severity", "type": "score", "score": 0.8, "probabilities": [{"value": 1, "label": "high", "probability": 0.8}]}
        ], "usage": {"input_tokens": 50, "output_tokens": 0}});
        let (endpoint, task) = server("200 OK", body.clone()).await;
        let provider = NativeDecisionProvider::new(DecisionApi::OpenAi, "test-key".into())
            .unwrap()
            .with_endpoint(endpoint);
        let response = provider.decide(request()).await.unwrap();
        let wire = task.await.unwrap();
        assert!(
            wire.to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert!(!wire.to_ascii_lowercase().contains("x-goog-api-key"));
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(payload["input"], r#"{"ticket":"Payment failed"}"#);
        let questions = payload["questions"].as_array().unwrap();
        let predicate = questions.iter().find(|v| v["name"] == "urgent").unwrap();
        assert_eq!(predicate["type"], "predicate");
        assert!(
            predicate["instructions"]
                .as_str()
                .unwrap()
                .contains("Blocking")
        );
        let choice = questions.iter().find(|v| v["name"] == "team").unwrap();
        assert_eq!(
            choice["choices"][0],
            json!({"value": "billing", "description": "Payment problems"})
        );
        let score = questions.iter().find(|v| v["name"] == "severity").unwrap();
        assert_eq!(
            score["levels"],
            json!([{"label": "low"}, {"label": "high"}])
        );
        assert_eq!(response.answers["urgent"], body["answers"][0]);
        assert_eq!(response.answers["team"], body["answers"][1]);
        assert_eq!(response.answers["severity"], body["answers"][2]);
        assert_eq!(response.usage.input_tokens, 50);
        assert_eq!(provider.default_model(), "gpt-6-luna");
    }

    #[tokio::test]
    async fn gemini_uses_input_and_api_key_header_and_maps_usage() {
        let body = json!({"answers": {"urgent": {"type": "noul", "noul": 0.9, "confidence": 0.9}}, "usage": {"total_input_tokens": 50, "total_output_tokens": 0, "total_tokens": 50}});
        let (endpoint, task) = server("200 OK", body.clone()).await;
        let provider = NativeDecisionProvider::new(DecisionApi::Gemini, "test-key".into())
            .unwrap()
            .with_endpoint(endpoint);
        let mut input = request();
        input.questions.retain(|name, _| name == "urgent");
        let response = provider.decide(input).await.unwrap();
        let wire = task.await.unwrap();
        assert!(
            wire.to_ascii_lowercase()
                .contains("x-goog-api-key: test-key")
        );
        assert!(!wire.to_ascii_lowercase().contains("authorization:"));
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(payload["input"], r#"{"ticket":"Payment failed"}"#);
        assert!(payload.get("state").is_none());
        assert_eq!(payload["questions"]["urgent"]["type"], "noul");
        assert!(payload["questions"]["urgent"].get("criteria").is_none());
        assert!(
            payload["questions"]["urgent"]["instructions"]
                .as_str()
                .unwrap()
                .contains("Blocking")
        );
        assert_eq!(
            response.answers,
            serde_json::from_value(body["answers"].clone()).unwrap()
        );
        assert_eq!(response.usage.input_tokens, 50);
        assert_eq!(response.usage.output_tokens, 0);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.provider.as_deref(), Some("gemini"));
        assert_eq!(provider.default_model(), "gemini-decision-1");
    }

    #[tokio::test]
    async fn errors_do_not_expose_upstream_body() {
        for api in [DecisionApi::OpenAi, DecisionApi::Gemini] {
            let (endpoint, task) = server(
                "401 Unauthorized",
                json!({"error": "secret-key and private state"}),
            )
            .await;
            let provider = NativeDecisionProvider::new(api, "test-key".into())
                .unwrap()
                .with_endpoint(endpoint);
            let error = provider.decide(request()).await.unwrap_err();
            task.await.unwrap();
            assert!(error.to_string().contains("401"));
            assert!(!format!("{error:?}").contains("secret-key"));
            assert!(!format!("{error:?}").contains("private state"));
        }
    }

    #[test]
    fn rejects_mismatched_native_response_question_names() {
        let provider = NativeDecisionProvider::new(DecisionApi::OpenAi, "test-key".into()).unwrap();
        for answers in [
            json!([]),
            json!([{"name": "unknown"}]),
            json!([{"name": "urgent"}, {"name": "urgent"}]),
            json!([{}]),
        ] {
            assert!(
                provider
                    .parse_response(
                        json!({"model": "test-model", "answers": answers, "usage": {}}),
                        &request()
                    )
                    .is_err()
            );
        }
        let provider = NativeDecisionProvider::new(DecisionApi::Gemini, "test-key".into()).unwrap();
        assert!(provider.parse_response(json!({"answers": {}, "usage": {"total_input_tokens": 1, "total_output_tokens": 0}}), &request()).is_err());
    }
}
