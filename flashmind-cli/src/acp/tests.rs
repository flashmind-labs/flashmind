//! `flsh acp` driven by an ACP client over a scripted model.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, Content, ContentBlock, ContentChunk, ImageContent, InitializeRequest,
    NewSessionRequest, PromptRequest, ResourceLink, SessionNotification, SessionUpdate, StopReason,
    TextContent, ToolCallContent, ToolCallStatus, ToolCallUpdate, UsageUpdate,
};
use agent_client_protocol::{Agent as AcpAgent, Client, ConnectionTo, on_receive_notification};
use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_core::{Agent, Conversation};
use flashmind_tools::attach::AttachTool;
use flashmind_tools::background::BackgroundExecTool;
use flashmind_tools::bash::BashTool;
use flashmind_tools::image_read::ImageReadTool;
use flashmind_tools::protected::ProtectedPaths;
use flashmind_types::tool::ToolRegistry;
use flashmind_types::{
    AgentLlmConfig, CompletionRequest, CompletionStream, ContentPart, CreditBalance, FinishReason,
    LlmProvider, ModelCapabilities, ModelPricing, Provider, Role, StreamEvent, TokenUsage,
};
use futures_util::{StreamExt, stream};
use rust_decimal::Decimal;
use serde_json::json;
use tokio::sync::mpsc;

use super::background;
use super::media::MAX_INLINE_IMAGE;
use super::open::{Opened, Opener};
use super::server::Server;
use super::steering::{SteeringRequest, SteeringResponse};
use super::system::SystemInstructions;
use super::usage::Meter;
use crate::session::new_session_key;

/// What the scripted model answers to one request.
enum Reply {
    Text(&'static str),
    /// Calls a tool that does not exist, so the call fails.
    MissingTool,
    /// Sends the text, then nothing until the turn is cancelled.
    Hang(&'static str),
    /// Signals that a model call is running, then waits before completing it.
    Paused {
        release: Arc<tokio::sync::Notify>,
    },
    Fail,
    /// Calls `attach` on the file at this path.
    Attach(&'static str),
    /// Calls `exec` with this command.
    Exec {
        command: &'static str,
        background: bool,
    },
    /// Replies with a PNG of these bytes.
    Picture(&'static [u8]),
    /// Sends the text and reports the tokens the call took.
    Counted {
        text: &'static str,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}

/// A model that answers each request with the next reply of its script,
/// and keeps the requests it got.
struct Script {
    provider: Provider,
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<CompletionRequest>>,
    initial_reasoning: flashmind_types::ReasoningLevel,
    images: bool,
    called: tokio::sync::Notify,
    balance: Option<CreditBalance>,
    balance_fails: bool,
}

impl Script {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Arc<Self> {
        Arc::new(Self {
            provider: Provider::Ollama,
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            initial_reasoning: flashmind_types::ReasoningLevel::Off,
            images: true,
            called: tokio::sync::Notify::new(),
            balance: None,
            balance_fails: false,
        })
    }
}

#[async_trait]
impl LlmProvider for Script {
    fn name(&self) -> &str {
        "script"
    }

    fn provider(&self) -> Provider {
        self.provider.clone()
    }

    async fn credit_balance(&self) -> Result<Option<CreditBalance>> {
        anyhow::ensure!(!self.balance_fails, "balance unavailable");
        Ok(self.balance.clone())
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        self.requests.lock().unwrap().push(request);
        self.called.notify_one();
        let reply = self.replies.lock().unwrap().pop_front();
        let finished = || Ok(StreamEvent::Finished(FinishReason::Stop));
        match reply {
            Some(Reply::Text(text)) => Box::pin(stream::iter([
                Ok(StreamEvent::ContentDelta(text.into())),
                finished(),
            ])),
            Some(Reply::MissingTool) => Box::pin(stream::iter([
                Ok(StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call-1".into(),
                    name: "no_such_tool".into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: "{}".into(),
                }),
                Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
            ])),
            Some(Reply::Attach(path)) => Box::pin(stream::iter([
                Ok(StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call-1".into(),
                    name: "attach".into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: json!({ "path": path }).to_string(),
                }),
                Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
            ])),
            Some(Reply::Exec {
                command,
                background,
            }) => Box::pin(stream::iter([
                Ok(StreamEvent::ToolCallStart {
                    index: 0,
                    id: "call-1".into(),
                    name: "exec".into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: json!({ "command": command, "run_in_background": background })
                        .to_string(),
                }),
                Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
            ])),
            Some(Reply::Picture(bytes)) => Box::pin(stream::iter([
                Ok(StreamEvent::FileAttachment {
                    filename: "picture.png".into(),
                    media_type: "image/png".into(),
                    data: bytes.to_vec(),
                }),
                finished(),
            ])),
            Some(Reply::Counted {
                text,
                prompt_tokens,
                completion_tokens,
            }) => Box::pin(stream::iter([
                Ok(StreamEvent::ContentDelta(text.into())),
                Ok(StreamEvent::Usage(TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    ..TokenUsage::default()
                })),
                finished(),
            ])),
            Some(Reply::Hang(text)) => Box::pin(
                stream::iter([Ok(StreamEvent::ContentDelta(text.into()))]).chain(stream::pending()),
            ),
            Some(Reply::Paused { release }) => Box::pin(
                stream::iter([Ok(StreamEvent::ContentDelta("Working.".into()))]).chain(
                    stream::once(async move {
                        release.notified().await;
                        Ok(StreamEvent::Finished(FinishReason::Stop))
                    }),
                ),
            ),
            Some(Reply::Fail) => Box::pin(stream::iter([Err(flashmind_types::LlmError::new(
                flashmind_types::LlmErrorKind::Configuration,
                "Example model failure.",
            )
            .into())])),
            None => Box::pin(stream::iter([
                Ok(StreamEvent::ContentDelta("The script ran out.".into())),
                finished(),
            ])),
        }
    }
}

/// The context window of the scripted model, in tokens.
const WINDOW: u32 = 8_000;

/// Opens sessions over the scripted model, priced at `pricing`, with the
/// `attach` tool and without saving.
struct Scripted {
    script: Arc<Script>,
    pricing: ModelPricing,
}

#[async_trait]
impl Opener for Scripted {
    async fn open(&self, cwd: &Path, _resume: Option<&str>) -> Result<Opened> {
        let model: flashmind_types::Model = "ollama:script".parse()?;
        let mut tools = ToolRegistry::default();
        tools.register(Arc::new(AttachTool {
            protected: Arc::new(ProtectedPaths::new(cwd)),
            image_reader: ImageReadTool {
                providers: Arc::default(),
                ocr_model: None,
            },
        }));
        let protected = Arc::new(ProtectedPaths::new(cwd));
        let exec = BackgroundExecTool::new(Arc::new(BashTool {
            protected,
            secrets: Vec::new(),
            forbidden_cmds: Vec::new(),
            allowlist: None,
        }));
        let jobs = exec.jobs();
        tools.register(Arc::new(exec.process_tool()));
        tools.register(Arc::new(exec));
        let mut agent = Agent::builder(self.script.clone())
            .llm(AgentLlmConfig::new(model.clone()))
            .tools(tools)
            .working_dir(cwd.to_path_buf())
            .keep_recent_tokens(0)
            .build_sync();
        agent.apply_features(
            ModelCapabilities {
                images: self.script.images,
                ..Default::default()
            },
            Some(WINDOW),
        );
        agent.llm_mut().reasoning = self.script.initial_reasoning;
        Ok(Opened {
            key: new_session_key(),
            agent,
            conversation: Conversation::with_system("You are a test."),
            system_instructions: SystemInstructions::default(),
            saved: None,
            subagents: None,
            tool_sync: None,
            jobs: Some(jobs),
            meter: Meter::new(model, self.pricing.clone()),
        })
    }
}

fn text(prompt: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text(TextContent::new(prompt))]
}

fn steering(
    id: &agent_client_protocol::schema::v1::SessionId,
    prompt: Vec<ContentBlock>,
) -> SteeringRequest {
    SteeringRequest {
        session_id: id.clone(),
        prompt,
        meta: Some(
            serde_json::from_value(json!({"steering": {"idleBehavior": "promptRequired"}}))
                .unwrap(),
        ),
    }
}

/// A synchronous rejected request is a dispatch barrier for earlier steers.
async fn steering_barrier(
    cx: &ConnectionTo<AcpAgent>,
    id: &agent_client_protocol::schema::v1::SessionId,
) {
    let mut request = steering(id, text("Invalid metadata."));
    request.meta =
        Some(serde_json::from_value(json!({"steering": {"idleBehavior": "unknown"}})).unwrap());
    let error = cx.send_request(request).block_task().await.unwrap_err();
    assert_eq!(
        error.code,
        agent_client_protocol::Error::invalid_params().code
    );
}

#[tokio::test]
async fn steering_waits_for_tools_and_preserves_order_images_and_prompt_ownership() {
    use tokio::io::AsyncWriteExt;

    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let mut gate = gate(&cwd);
    let script = Script::new([
        Reply::Exec {
            command: "echo ready; read -r line < gate; echo released",
            background: false,
        },
        Reply::Counted {
            text: "Steered.",
            prompt_tokens: 100,
            completion_tokens: 20,
        },
        Reply::Text("Next turn."),
    ]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx.send_request(NewSessionRequest::new(cwd)).block_task().await?;
        let id = session.session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Run the command.")));
        loop {
            if let Some(SessionUpdate::ToolCallUpdate(update)) = updates.recv().await
                && update.fields.status.is_none()
                && update.fields.content.as_ref().is_some_and(|content| !content.is_empty())
            { break; }
        }
        let mixed = vec![
            ContentBlock::Text(TextContent::new("Before.")),
            ContentBlock::Image(ImageContent::new("aW1hZ2Ux", "image/png")),
            ContentBlock::Text(TextContent::new("Between.")),
            ContentBlock::Image(ImageContent::new("aW1hZ2Uy", "image/jpeg")),
            ContentBlock::Text(TextContent::new("After.")),
        ];
        let mut pending = vec![cx.send_request(steering(&id, mixed))];
        for index in 1..8 {
            pending.push(cx.send_request(steering(&id, text(&format!("Steer {index}.")))));
        }
        let full = cx.send_request(steering(&id, text("Overflow."))).block_task().await.unwrap_err();
        assert_eq!(full.code, agent_client_protocol::Error::invalid_request().code);
        assert_eq!(script.requests.lock().unwrap().len(), 1, "the blocked tool was not cancelled or restarted");
        gate.write_all(b"go\n").await.unwrap();
        for response in pending {
            assert_eq!(response.block_task().await?, SteeringResponse::Injected);
        }
        assert_eq!(answer.block_task().await?.stop_reason, StopReason::EndTurn);
        let shown = drain(&mut updates);
        assert_eq!(message(&shown), "Steered.");
        assert_eq!(usage(&shown).last().unwrap().used, 120);
        assert!(shown.iter().any(|update| matches!(update,
            SessionUpdate::ToolCallUpdate(update) if update.fields.status == Some(ToolCallStatus::Completed)
        )));
        cx.send_request(PromptRequest::new(id, text("Next."))).block_task().await?;
        Ok(())
    }).await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let messages = &requests[1].messages;
    let mixed = messages
        .iter()
        .position(|message| message.parts.is_some())
        .unwrap();
    assert!(
        messages[..mixed]
            .iter()
            .any(|message| message.role == Role::Tool && message.content.contains("released"))
    );
    assert!(
        messages[..mixed]
            .iter()
            .any(|message| message.content == "Run the command.")
    );
    assert_eq!(messages[mixed].content, "");
    assert_eq!(
        serde_json::to_value(&messages[mixed].parts).unwrap(),
        json!([
            {"type":"text","text":"Before."},
            {"type":"image","media_type":"image/png","data":"aW1hZ2Ux"},
            {"type":"text","text":"Between."},
            {"type":"image","media_type":"image/jpeg","data":"aW1hZ2Uy"},
            {"type":"text","text":"After."},
        ])
    );
    for index in 1..8 {
        assert_eq!(messages[mixed + index].content, format!("Steer {index}."));
    }
    assert!(requests.iter().all(|request| {
        request
            .messages
            .iter()
            .all(|message| message.content != "Overflow.")
    }));
    assert_eq!(
        requests[2]
            .messages
            .iter()
            .filter(|message| message.content == "Steer 1.")
            .count(),
        1
    );
}

#[tokio::test]
async fn steering_at_model_completion_continues_the_owning_prompt_then_becomes_idle() {
    let dir = tempfile::tempdir().unwrap();
    let release = Arc::new(tokio::sync::Notify::new());
    let script = Script::new([
        Reply::Paused {
            release: release.clone(),
        },
        Reply::Text("Adjusted."),
        Reply::Text("Next."),
    ]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
        while !matches!(
            updates.recv().await,
            Some(SessionUpdate::AgentMessageChunk(_))
        ) {}
        let steer = cx.send_request(steering(&id, text("Adjust.")));
        steering_barrier(&cx, &id).await;
        release.notify_one();
        assert_eq!(steer.block_task().await?, SteeringResponse::Injected);
        assert_eq!(answer.block_task().await?.stop_reason, StopReason::EndTurn);
        assert_eq!(message(&drain(&mut updates)), "Adjusted.");
        assert_eq!(
            cx.send_request(steering(&id, text("Keep with client.")))
                .block_task()
                .await?,
            SteeringResponse::idle()
        );
        assert_eq!(script.requests.lock().unwrap().len(), 2);
        cx.send_request(PromptRequest::new(id, text("Next.")))
            .block_task()
            .await?;
        Ok(())
    })
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[1]
            .messages
            .iter()
            .any(|message| message.content == "Adjust.")
    );
    assert!(
        requests[2]
            .messages
            .iter()
            .all(|message| message.content != "Keep with client.")
    );
}

#[tokio::test]
async fn steering_racing_completion_is_either_injected_once_or_left_with_the_client() {
    for _ in 0..16 {
        let dir = tempfile::tempdir().unwrap();
        let release = Arc::new(tokio::sync::Notify::new());
        let script = Script::new([
            Reply::Paused {
                release: release.clone(),
            },
            Reply::Text("Adjusted."),
        ]);
        connect(script.clone(), async |cx, mut updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let id = cx
                .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
                .block_task()
                .await?
                .session_id;
            let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
            while !matches!(
                updates.recv().await,
                Some(SessionUpdate::AgentMessageChunk(_))
            ) {}
            release.notify_one();
            let response = cx
                .send_request(steering(&id, text("Race.")))
                .block_task()
                .await?;
            assert_eq!(answer.block_task().await?.stop_reason, StopReason::EndTurn);
            match response {
                SteeringResponse::Injected => assert_eq!(script.requests.lock().unwrap().len(), 2),
                SteeringResponse::PromptRequired { .. } => {
                    assert_eq!(script.requests.lock().unwrap().len(), 1);
                    cx.send_request(PromptRequest::new(id, text("Race.")))
                        .block_task()
                        .await?;
                }
                SteeringResponse::StartedNewTurn => panic!("idle opt-in must retain input"),
            }
            Ok(())
        })
        .await;
        let requests = script.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1]
                .messages
                .iter()
                .filter(|message| message.content == "Race.")
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn cancellation_returns_undelivered_steering_without_leaking_into_the_next_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let _gate = gate(dir.path());
    let script = Script::new([
        Reply::Exec {
            command: "echo ready; read -r line < gate",
            background: false,
        },
        Reply::Text("Next."),
    ]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
        loop {
            if let Some(SessionUpdate::ToolCallUpdate(update)) = updates.recv().await
                && update.fields.status.is_none()
                && update
                    .fields
                    .content
                    .as_ref()
                    .is_some_and(|content| !content.is_empty())
            {
                break;
            }
        }
        let steer = cx.send_request(steering(&id, text("Undelivered.")));
        let mut default = steering(&id, text("Also undelivered."));
        default.meta = None;
        let default = cx.send_request(default);
        steering_barrier(&cx, &id).await;
        cx.send_notification(CancelNotification::new(id.clone()))?;
        assert_eq!(steer.block_task().await?, SteeringResponse::idle());
        assert!(default.block_task().await.is_err());
        assert_eq!(
            answer.block_task().await?.stop_reason,
            StopReason::Cancelled
        );
        cx.send_request(PromptRequest::new(id, text("Next.")))
            .block_task()
            .await?;
        Ok(())
    })
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1]
            .messages
            .iter()
            .all(|message| !message.content.contains("undelivered.")
                && !message.content.contains("Undelivered."))
    );
    for call in requests[1]
        .messages
        .iter()
        .filter_map(|message| message.tool_calls.as_ref())
        .flatten()
    {
        assert_eq!(
            requests[1]
                .messages
                .iter()
                .filter(|message| message.tool_call_id.as_deref() == Some(call.id.as_str()))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn idle_steering_defaults_to_a_session_owned_turn_and_treats_commands_as_input() {
    let dir = tempfile::tempdir().unwrap();
    let script = Script::new([Reply::Hang("Started."), Reply::Text("Next.")]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        assert_eq!(
            cx.send_request(steering(&id, text("Not consumed.")))
                .block_task()
                .await?,
            SteeringResponse::idle()
        );
        let mut request = steering(&id, text("/system literal prompt"));
        request.meta = None;
        assert_eq!(
            cx.send_request(request).block_task().await?,
            SteeringResponse::StartedNewTurn
        );
        while !matches!(
            updates.recv().await,
            Some(SessionUpdate::AgentMessageChunk(_))
        ) {}
        cx.send_notification(CancelNotification::new(id.clone()))?;
        cx.send_request(PromptRequest::new(id, text("Next.")))
            .block_task()
            .await?;
        Ok(())
    })
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0].messages.iter().any(
            |message| message.role == Role::User && message.content == "/system literal prompt"
        )
    );
    assert!(
        requests[1]
            .messages
            .iter()
            .all(|message| message.content != "Not consumed.")
    );
}

#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, agent_client_protocol::JsonRpcRequest,
)]
#[request(method = "_session/steering", response = SteeringResponse)]
#[serde(rename_all = "camelCase")]
struct WireSteering {
    session_id: serde_json::Value,
    prompt: serde_json::Value,
    #[serde(rename = "_meta")]
    meta: serde_json::Value,
}

#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, agent_client_protocol::JsonRpcRequest,
)]
#[request(method = "_session/unknown_steering", response = SteeringResponse)]
struct UnknownSteering {}

#[tokio::test]
async fn steering_rejects_malformed_metadata_content_unknown_sessions_and_methods() {
    let dir = tempfile::tempdir().unwrap();
    let script = Script::new([]);
    connect(script.clone(), async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        for (prompt, meta) in [
            (json!([]), json!(null)),
            (json!("invalid"), json!(null)),
            (json!([{"type":"unknown","text":"Invalid."}]), json!(null)),
            (json!([{"type":"text","text":""}]), json!(null)),
            (json!([{"type":"text","text":"Valid."}]), json!([])),
            (
                json!([{"type":"text","text":"Valid."}]),
                json!({"steering": null}),
            ),
            (
                json!([{"type":"text","text":"Valid."}]),
                json!({"steering": {"idleBehavior": null}}),
            ),
            (
                json!([{"type":"text","text":"Valid."}]),
                json!({"steering": {"idleBehavior": "unknown"}}),
            ),
        ] {
            let error = cx
                .send_request(WireSteering {
                    session_id: json!(id),
                    prompt,
                    meta,
                })
                .block_task()
                .await
                .unwrap_err();
            assert_eq!(
                error.code,
                agent_client_protocol::Error::invalid_params().code
            );
        }
        let unknown = cx
            .send_request(steering(
                &agent_client_protocol::schema::v1::SessionId::new("session_unknown"),
                text("Hi."),
            ))
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(
            unknown.code,
            agent_client_protocol::Error::session_not_found("session_unknown").code
        );
        let unknown = cx
            .send_request(UnknownSteering {})
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(
            unknown.code,
            agent_client_protocol::Error::method_not_found().code
        );
        let mut request = steering(&id, text("Unrelated metadata."));
        request
            .meta
            .as_mut()
            .unwrap()
            .insert("otherExtension".into(), json!({"version": 1}));
        let response = cx.send_request(request).block_task().await?;
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({"outcome":"promptRequired","reason":"noRunningTurn"})
        );
        assert!(script.requests.lock().unwrap().is_empty());
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn steering_rejects_images_for_a_text_model_and_bounds_payload_size() {
    let dir = tempfile::tempdir().unwrap();
    let mut script = Script::new([]);
    Arc::get_mut(&mut script).unwrap().images = false;
    connect(script.clone(), async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let image = vec![ContentBlock::Image(ImageContent::new(
            "aW1hZ2U=",
            "image/png",
        ))];
        let error = cx
            .send_request(steering(&id, image))
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(
            error.code,
            agent_client_protocol::Error::invalid_request().code
        );
        let error = cx
            .send_request(steering(&id, text(&"x".repeat(16 * 1024 * 1024 + 1))))
            .block_task()
            .await
            .unwrap_err();
        assert_eq!(
            error.code,
            agent_client_protocol::Error::invalid_request().code
        );
        assert!(script.requests.lock().unwrap().is_empty());
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn null_steering_metadata_uses_the_default_idle_behavior() {
    let dir = tempfile::tempdir().unwrap();
    let script = Script::new([Reply::Hang("Started.")]);
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let response = cx
            .send_request(WireSteering {
                session_id: json!(id),
                prompt: json!([{"type":"text","text":"Work."}]),
                meta: json!(null),
            })
            .block_task()
            .await?;
        assert_eq!(response, SteeringResponse::StartedNewTurn);
        while !matches!(
            updates.recv().await,
            Some(SessionUpdate::AgentMessageChunk(_))
        ) {}
        cx.send_notification(CancelNotification::new(id))?;
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_model_failure_after_steering_belongs_to_the_original_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let release = Arc::new(tokio::sync::Notify::new());
    let script = Script::new([
        Reply::Paused {
            release: release.clone(),
        },
        Reply::Fail,
        Reply::Text("Next."),
    ]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
        while !matches!(
            updates.recv().await,
            Some(SessionUpdate::AgentMessageChunk(_))
        ) {}
        let steer = cx.send_request(steering(&id, text("Adjust.")));
        steering_barrier(&cx, &id).await;
        release.notify_one();
        assert_eq!(steer.block_task().await?, SteeringResponse::Injected);
        let failure = answer.block_task().await.unwrap_err();
        assert_eq!(
            failure.code,
            agent_client_protocol::Error::internal_error().code
        );
        assert!(
            failure
                .data
                .unwrap()
                .as_str()
                .unwrap()
                .contains("Example model failure.")
        );
        assert_eq!(
            cx.send_request(steering(&id, text("Not consumed.")))
                .block_task()
                .await?,
            SteeringResponse::idle()
        );
        cx.send_request(PromptRequest::new(id, text("Next.")))
            .block_task()
            .await?;
        Ok(())
    })
    .await;
    assert_eq!(script.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn steering_waits_out_compaction_and_is_delivered_without_being_summarized() {
    let dir = tempfile::tempdir().unwrap();
    let release = Arc::new(tokio::sync::Notify::new());
    let script = Script::new([
        Reply::Counted {
            text: "Initial answer.",
            prompt_tokens: 7_000,
            completion_tokens: 10,
        },
        Reply::Paused {
            release: release.clone(),
        },
        Reply::Text("Adjusted."),
    ]);
    connect(script.clone(), async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let id = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?
            .session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
        loop {
            let called = script.called.notified();
            if script.requests.lock().unwrap().len() >= 2 {
                break;
            }
            called.await;
        }
        let steer = cx.send_request(steering(
            &id,
            vec![
                ContentBlock::Text(TextContent::new("After compaction.")),
                ContentBlock::Image(ImageContent::new("aW1hZ2U=", "image/png")),
            ],
        ));
        steering_barrier(&cx, &id).await;
        release.notify_one();
        assert_eq!(steer.block_task().await?, SteeringResponse::Injected);
        assert_eq!(answer.block_task().await?.stop_reason, StopReason::EndTurn);
        Ok(())
    })
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[1]
            .messages
            .iter()
            .all(|message| message.parts.is_none())
    );
    let parts = requests[2]
        .messages
        .iter()
        .filter_map(|message| message.parts.as_ref())
        .flatten()
        .collect::<Vec<_>>();
    assert!(
        matches!(parts.as_slice(), [ContentPart::Text { text }, ContentPart::Image { data, .. }] if text == "After compaction." && data == "aW1hZ2U=")
    );
}

#[tokio::test]
async fn automatic_compaction_errors_fail_the_owning_prompt_without_losing_history() {
    for summary in [Reply::Fail, Reply::Text("")] {
        let dir = tempfile::tempdir().unwrap();
        let script = Script::new([
            Reply::Counted {
                text: "Original answer.",
                prompt_tokens: 7_000,
                completion_tokens: 10,
            },
            summary,
            Reply::Text("Next answer."),
        ]);
        connect(script.clone(), async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let id = cx
                .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
                .block_task()
                .await?
                .session_id;
            let failure = cx
                .send_request(PromptRequest::new(id.clone(), text("Original question.")))
                .block_task()
                .await
                .unwrap_err();
            assert_eq!(
                failure.code,
                agent_client_protocol::Error::internal_error().code
            );
            assert!(
                failure
                    .data
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .contains("Compaction")
            );
            assert_eq!(
                cx.send_request(steering(&id, text("Not consumed.")))
                    .block_task()
                    .await?,
                SteeringResponse::idle()
            );
            cx.send_request(PromptRequest::new(id, text("Next question.")))
                .block_task()
                .await?;
            Ok(())
        })
        .await;
        let requests = script.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        for content in ["Original question.", "Original answer."] {
            assert!(
                requests[2]
                    .messages
                    .iter()
                    .any(|message| message.content == content)
            );
        }
        assert!(
            requests[2]
                .messages
                .iter()
                .all(|message| message.content != "Not consumed.")
        );
    }
}

/// The text of the message chunks among `updates`.
fn message(updates: &[SessionUpdate]) -> String {
    updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn drain(updates: &mut mpsc::UnboundedReceiver<SessionUpdate>) -> Vec<SessionUpdate> {
    let mut drained = Vec::new();
    while let Ok(update) = updates.try_recv() {
        drained.push(update);
    }
    drained
}

/// The usage updates among `updates`.
fn usage(updates: &[SessionUpdate]) -> Vec<&UsageUpdate> {
    updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::UsageUpdate(usage) => Some(usage),
            _ => None,
        })
        .collect()
}

/// Connects a client to a server over `script` and runs `client` with the
/// connection and the updates it receives.
async fn connect(
    script: Arc<Script>,
    client: impl AsyncFnOnce(
        ConnectionTo<AcpAgent>,
        mpsc::UnboundedReceiver<SessionUpdate>,
    ) -> Result<(), agent_client_protocol::Error>,
) {
    connect_priced(script, ModelPricing::default(), client).await;
}

/// [`connect`], with the model priced at `pricing`.
async fn connect_priced(
    script: Arc<Script>,
    pricing: ModelPricing,
    client: impl AsyncFnOnce(
        ConnectionTo<AcpAgent>,
        mpsc::UnboundedReceiver<SessionUpdate>,
    ) -> Result<(), agent_client_protocol::Error>,
) {
    connect_opener(Scripted { script, pricing }, client).await;
}

async fn connect_opener(
    opener: impl Opener,
    client: impl AsyncFnOnce(
        ConnectionTo<AcpAgent>,
        mpsc::UnboundedReceiver<SessionUpdate>,
    ) -> Result<(), agent_client_protocol::Error>,
) {
    connect_server(Server::new(opener), client).await;
}

async fn connect_server(
    mut server: Server,
    client: impl AsyncFnOnce(
        ConnectionTo<AcpAgent>,
        mpsc::UnboundedReceiver<SessionUpdate>,
    ) -> Result<(), agent_client_protocol::Error>,
) {
    let isolated = tempfile::tempdir().unwrap();
    if server.presence_file.is_none() {
        server.presence_file = Some(isolated.path().join("absent-presence.json"));
    }
    let (updates_tx, updates) = mpsc::unbounded_channel();
    Client
        .builder()
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                updates_tx
                    .send(notification.update)
                    .expect("The test holds the receiver until the connection ends.");
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(server.agent(), async |cx| client(cx, updates).await)
        .await
        .expect("The ACP exchange should succeed.");
    server.shutdown().await;
}

#[cfg(unix)]
#[tokio::test]
async fn idle_acp_sessions_keep_separate_presence_until_shutdown() {
    use crate::presence::tests::{Receiver, eof};
    use agent_client_protocol::schema::v1::ResumeSessionRequest;

    let receiver = Receiver::new();
    let first_directory = tempfile::tempdir().unwrap();
    let second_directory = tempfile::tempdir().unwrap();
    let mut server = Server::new(Scripted {
        script: Script::new([]),
        pricing: ModelPricing::default(),
    });
    server.presence_file = Some(receiver.locator());
    let shutdown = server.clone();
    connect_server(server, async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let first = cx
            .send_request(NewSessionRequest::new(first_directory.path().to_path_buf()))
            .block_task()
            .await?;
        let (mut first_connection, first_registration) = receiver.accept().await;
        assert_eq!(
            first_registration["session_id"],
            first.session_id.0.as_ref()
        );
        assert_eq!(
            first_registration["cwd"],
            first_directory.path().to_str().unwrap()
        );

        let second = cx
            .send_request(NewSessionRequest::new(
                second_directory.path().to_path_buf(),
            ))
            .block_task()
            .await?;
        let (mut second_connection, second_registration) = receiver.accept().await;
        assert_eq!(
            second_registration["session_id"],
            second.session_id.0.as_ref()
        );
        assert_eq!(
            second_registration["cwd"],
            second_directory.path().to_str().unwrap()
        );
        assert_ne!(
            first_registration["session_id"],
            second_registration["session_id"]
        );

        cx.send_request(ResumeSessionRequest::new(
            first.session_id,
            first_directory.path().to_path_buf(),
        ))
        .block_task()
        .await?;
        shutdown.shutdown().await;
        eof(&mut first_connection).await;
        eof(&mut second_connection).await;
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_prompt_streams_the_reply_and_ends_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([Reply::MissingTool, Reply::Text("Hello from flsh.")]);
    connect(script, async |cx, mut updates| {
        let initialized = cx
            .send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        assert_eq!(initialized.protocol_version, ProtocolVersion::V1);
        assert!(
            initialized
                .agent_capabilities
                .session_capabilities
                .resume
                .is_some()
        );
        assert!(initialized.agent_capabilities.prompt_capabilities.image);
        assert_eq!(initialized.agent_info.unwrap().name, "flsh");
        assert_eq!(
            initialized.meta.unwrap()["steering"],
            json!({"supported": true})
        );
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        assert!(session.session_id.to_string().starts_with("session_"));
        let done = cx
            .send_request(PromptRequest::new(session.session_id, text("Say hello.")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);

        // Every update was handled before the answer to the prompt.
        let updates = drain(&mut updates);
        let Some(SessionUpdate::ToolCall(call)) = updates
            .iter()
            .find(|update| matches!(update, SessionUpdate::ToolCall(_)))
        else {
            panic!("expected a tool call among {updates:?}");
        };
        assert_eq!(call.title, "no_such_tool");
        assert_eq!(call.status, ToolCallStatus::InProgress);
        assert!(updates.iter().any(|update| matches!(
            update,
            SessionUpdate::ToolCallUpdate(update)
                if update.tool_call_id == call.tool_call_id
                    && update.fields.status == Some(ToolCallStatus::Failed)
        )));
        assert_eq!(message(&updates), "Hello from flsh.");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn cancel_stops_the_turn_and_the_session_goes_on() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([Reply::Hang("Working"), Reply::Text("Still here.")]);
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let id = session.session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work.")));
        // The turn is under way once its first chunk arrives.
        let first = loop {
            let update = updates.recv().await.expect("the turn sends a chunk");
            if matches!(update, SessionUpdate::AgentMessageChunk(_)) {
                break update;
            }
        };
        assert_eq!(message(&[first]), "Working");
        cx.send_notification(CancelNotification::new(id.clone()))?;
        let done = answer.block_task().await?;
        assert_eq!(done.stop_reason, StopReason::Cancelled);

        drain(&mut updates);
        let done = cx
            .send_request(PromptRequest::new(id, text("Are you there?")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        assert_eq!(message(&drain(&mut updates)), "Still here.");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_prompt_for_an_unknown_session_fails() {
    connect(Script::new([]), async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let answer = cx
            .send_request(PromptRequest::new("session_unknown", text("Hi.")))
            .block_task()
            .await;
        assert!(answer.is_err(), "expected an error, got {answer:?}");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_prompt_passes_its_images_to_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([Reply::Text("A red pixel.")]);
    connect(script.clone(), async |cx, _updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let mut prompt = text("What is in this picture?");
        prompt.push(ContentBlock::Image(ImageContent::new(
            "iVBORw0KGgo=",
            "image/png",
        )));
        let done = cx
            .send_request(PromptRequest::new(session.session_id, prompt))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        Ok(())
    })
    .await;

    let requests = script.requests.lock().unwrap();
    let user = requests[0]
        .messages
        .iter()
        .rfind(|message| message.role == Role::User)
        .expect("the request carries the prompt");
    assert!(user.content.contains("What is in this picture?"));
    let parts = user.parts.as_deref().expect("the prompt has parts");
    assert!(
        parts.iter().any(|part| matches!(
            part,
            ContentPart::Image { media_type, data }
                if media_type == "image/png" && data == "iVBORw0KGgo="
        )),
        "expected the image among {parts:?}"
    );
}

/// Runs one turn in which the model attaches `name`, written to the session
/// directory with `bytes`, and returns the update that finished the call.
async fn attach_file(name: &'static str, bytes: &[u8]) -> ToolCallUpdate {
    let dir = tempfile::tempdir().unwrap();
    // A repository root, so a .gitignore above the temporary directory
    // cannot make the file protected.
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join(name), bytes).unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([Reply::Attach(name), Reply::Text("Seen.")]);
    let mut finished = None;
    let mut all = Vec::new();
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        cx.send_request(PromptRequest::new(session.session_id, text("Look.")))
            .block_task()
            .await?;
        all = drain(&mut updates);
        finished = all.clone().into_iter().find_map(|update| match update {
            SessionUpdate::ToolCallUpdate(update)
                if update.fields.status == Some(ToolCallStatus::Completed) =>
            {
                Some(update)
            }
            _ => None,
        });
        Ok(())
    })
    .await;
    let update = finished.unwrap_or_else(|| panic!("the attach call completes: {all:?}"));
    let locations = update.fields.locations.as_deref().unwrap_or_default();
    assert_eq!(locations.len(), 1, "{locations:?}");
    assert_eq!(locations[0].path, dir.path().join(name));
    update
}

/// The images and the links among a finished call's content.
fn shown(update: &ToolCallUpdate) -> (Vec<&ImageContent>, Vec<&ResourceLink>) {
    let mut images = Vec::new();
    let mut links = Vec::new();
    for content in update.fields.content.iter().flatten() {
        match content {
            ToolCallContent::Content(Content {
                content: ContentBlock::Image(image),
                ..
            }) => images.push(image),
            ToolCallContent::Content(Content {
                content: ContentBlock::ResourceLink(link),
                ..
            }) => links.push(link),
            _ => {}
        }
    }
    (images, links)
}

/// The first bytes of a PNG, enough for a file the tools take as one.
const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";

#[tokio::test]
async fn attaching_a_png_shows_the_image_and_links_its_file() {
    let update = attach_file("picture.png", PNG).await;
    let (images, links) = shown(&update);
    assert_eq!(images.len(), 1, "{update:?}");
    assert_eq!(images[0].mime_type, "image/png");
    assert_eq!(images[0].data, STANDARD.encode(PNG));
    assert_eq!(links.len(), 1, "{update:?}");
    assert!(links[0].uri.starts_with("file:///"), "{}", links[0].uri);
    assert!(links[0].uri.ends_with("/picture.png"), "{}", links[0].uri);
    assert_eq!(links[0].mime_type.as_deref(), Some("image/png"));
}

#[tokio::test]
async fn a_large_image_is_only_linked() {
    let mut bytes = PNG.to_vec();
    bytes.resize(MAX_INLINE_IMAGE + 1024, 0);
    let update = attach_file("large.png", &bytes).await;
    let (images, links) = shown(&update);
    assert!(images.is_empty(), "{update:?}");
    assert_eq!(links.len(), 1);
    assert!(links[0].uri.ends_with("/large.png"));
}

#[tokio::test]
async fn a_file_that_is_no_image_is_neither_shown_nor_linked() {
    let update = attach_file("notes.pdf", b"%PDF-1.7").await;
    let (images, links) = shown(&update);
    assert!(images.is_empty(), "{update:?}");
    assert!(links.is_empty(), "{update:?}");
}

#[tokio::test]
async fn an_image_in_the_reply_is_sent_as_a_message_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    connect(
        Script::new([Reply::Picture(PNG)]),
        async |cx, mut updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(session.session_id, text("Draw.")))
                .block_task()
                .await?;
            let updates = drain(&mut updates);
            let image = updates.iter().find_map(|update| match update {
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Image(image),
                    ..
                }) => Some(image),
                _ => None,
            });
            let image = image.expect("the reply's image is sent");
            assert_eq!(image.mime_type, "image/png");
            assert_eq!(image.data, STANDARD.encode(PNG));
            Ok(())
        },
    )
    .await;
}

/// Runs two counted turns in a new session over a model priced at
/// `pricing`, and returns the usage updates of each turn.
async fn two_counted_turns(pricing: ModelPricing) -> [Vec<UsageUpdate>; 2] {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([
        Reply::Counted {
            text: "One.",
            prompt_tokens: 1_000,
            completion_tokens: 100,
        },
        Reply::Counted {
            text: "Two.",
            prompt_tokens: 1_200,
            completion_tokens: 50,
        },
    ]);
    let mut turns = [Vec::new(), Vec::new()];
    connect_priced(script, pricing, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        for turn in &mut turns {
            let done = cx
                .send_request(PromptRequest::new(session.session_id.clone(), text("Go.")))
                .block_task()
                .await?;
            assert_eq!(done.stop_reason, StopReason::EndTurn);
            *turn = usage(&drain(&mut updates)).into_iter().cloned().collect();
        }
        Ok(())
    })
    .await;
    turns
}

#[tokio::test]
async fn each_model_call_reports_the_context_and_the_session_cost() {
    let pricing = ModelPricing {
        prompt: Some(Decimal::new(2, 6)),
        completion: Some(Decimal::new(10, 6)),
        ..Default::default()
    };
    let [first, second] = two_counted_turns(pricing).await;
    let [first] = first.as_slice() else {
        panic!("expected one usage update, got {first:?}");
    };
    assert_eq!(first.used, 1_100);
    assert_eq!(first.size, u64::from(WINDOW));
    let cost = first.cost.as_ref().expect("the model has a price");
    assert_eq!(cost.currency, "USD");
    // 1000 * $2/m + 100 * $10/m
    assert!((cost.amount - 0.003).abs() < 1e-12, "{}", cost.amount);

    let [second] = second.as_slice() else {
        panic!("expected one usage update, got {second:?}");
    };
    assert_eq!(second.used, 1_250);
    // The cost adds up over the session: + 1200 * $2/m + 50 * $10/m
    let cost = second.cost.as_ref().expect("the model has a price");
    assert!((cost.amount - 0.0059).abs() < 1e-12, "{}", cost.amount);
}

#[tokio::test]
async fn a_model_without_a_price_reports_no_cost() {
    let [first, second] = two_counted_turns(ModelPricing::default()).await;
    for update in first.iter().chain(&second) {
        assert_eq!(update.size, u64::from(WINDOW));
        assert!(update.cost.is_none(), "expected no cost in {update:?}");
    }
    assert_eq!(first.len(), 1);
    assert_eq!(second[0].used, 1_250);
}

#[tokio::test]
async fn compact_is_offered_and_summarizes_the_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([Reply::Text("Hello."), Reply::Text("The user said hi.")]);
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let id = session.session_id;
        cx.send_request(PromptRequest::new(id.clone(), text("Hi.")))
            .block_task()
            .await?;
        let offered = drain(&mut updates);
        assert!(
            offered.iter().any(|update| matches!(
                update,
                SessionUpdate::AvailableCommandsUpdate(commands)
                    if commands.available_commands.iter().any(|command| command.name == "compact")
            )),
            "expected /compact among {offered:?}"
        );

        let done = cx
            .send_request(PromptRequest::new(id, text("/compact")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        assert_eq!(message(&drain(&mut updates)), "Compacted the conversation.");
        Ok(())
    })
    .await;
    // The summary came from a second request, not from a turn on "/compact".
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1]
            .messages
            .iter()
            .all(|message| !message.content.contains("/compact"))
    );
}

/// Every `session/update` a client gets, as the JSON on the wire: the
/// `async_task_*` updates are not in the ACP crate's `SessionUpdate`.
#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, agent_client_protocol::JsonRpcNotification,
)]
#[notification(method = "session/update")]
#[serde(rename_all = "camelCase")]
struct RawUpdate {
    session_id: String,
    update: serde_json::Value,
}

type Raw = mpsc::UnboundedReceiver<serde_json::Value>;

/// [`connect`] for a client that sees the updates as JSON, and declares the
/// `asyncTasks` extension when `air` is set.
async fn connect_raw(
    script: Arc<Script>,
    air: bool,
    client: impl AsyncFnOnce(ConnectionTo<AcpAgent>, Raw) -> Result<(), agent_client_protocol::Error>,
) {
    let mut server = Server::new(Scripted {
        script,
        pricing: ModelPricing::default(),
    });
    let isolated = tempfile::tempdir().unwrap();
    server.presence_file = Some(isolated.path().join("absent-presence.json"));
    let (updates_tx, updates) = mpsc::unbounded_channel();
    Client
        .builder()
        .on_receive_notification(
            async move |notification: RawUpdate, _cx| {
                updates_tx
                    .send(notification.update)
                    .expect("The test holds the receiver until the connection ends.");
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(server.agent(), async |cx| {
            let mut initialize = InitializeRequest::new(ProtocolVersion::V1);
            if air {
                let meta = json!({
                    "jetbrains": { "air": { "version": 1, "capabilities": ["asyncTasks"] } }
                });
                let serde_json::Value::Object(meta) = meta else {
                    return Err(agent_client_protocol::Error::internal_error());
                };
                initialize = initialize.client_capabilities(
                    agent_client_protocol::schema::v1::ClientCapabilities::new().meta(meta),
                );
            }
            let initialized = cx.send_request(initialize).block_task().await?;
            let meta = initialized.meta.unwrap();
            assert_eq!(meta["steering"], json!({"supported": true}));
            assert_eq!(meta.contains_key("jetbrains"), air);
            client(cx, updates).await
        })
        .await
        .expect("The ACP exchange should succeed.");
    server.shutdown().await;
}

/// The next update with `kind` as its `sessionUpdate`, skipping the others.
async fn next_update(updates: &mut Raw, kind: &str) -> serde_json::Value {
    loop {
        let update = updates.recv().await.expect("the connection is open");
        if update["sessionUpdate"] == kind {
            return update;
        }
    }
}

/// A fifo named `gate` in `dir`, opened for reading and writing so that a
/// command blocked on it wakes when the test writes a line.
fn gate(dir: &Path) -> tokio::fs::File {
    let made = std::process::Command::new("mkfifo")
        .arg(dir.join("gate"))
        .status()
        .expect("mkfifo runs");
    assert!(made.success());
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(dir.join("gate"))
        .expect("the fifo opens");
    tokio::fs::File::from_std(file)
}

fn meta_marks_backgrounded(update: &serde_json::Value) -> bool {
    update["_meta"]["jetbrains"]["air"]["asyncTasks"]["backgrounded"] == true
}

#[tokio::test]
async fn a_background_command_is_reported_and_its_end_arrives_after_the_turn() {
    use tokio::io::AsyncWriteExt;

    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let mut gate = gate(&cwd);
    let script = Script::new([
        Reply::Exec {
            command: "read -r line < gate; echo released",
            background: true,
        },
        Reply::Text("Started it."),
    ]);
    connect_raw(script, true, async |cx, mut updates| {
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let done = cx
            .send_request(PromptRequest::new(session.session_id, text("Serve.")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);

        let call = next_update(&mut updates, "tool_call").await["toolCallId"].clone();
        let spawned = next_update(&mut updates, "async_task_spawned").await;
        assert_eq!(spawned["name"], "read -r line < gate; echo released");
        assert_eq!(spawned["taskType"], "shell");
        assert_eq!(spawned["canStop"], true);
        assert_eq!(spawned["showInTranscript"], true);
        assert_eq!(spawned["toolCallId"], call);
        let task = spawned["asyncTaskId"]
            .as_str()
            .expect("a task id")
            .to_owned();
        let finished = next_update(&mut updates, "tool_call_update").await;
        assert_eq!(finished["toolCallId"], call);
        assert!(meta_marks_backgrounded(&finished), "{finished}");

        // The turn is over; the command ends on its own and says so.
        gate.write_all(b"go\n")
            .await
            .expect("the fifo takes a line");
        let ended = next_update(&mut updates, "async_task_state_update").await;
        assert_eq!(ended["asyncTaskId"], task);
        assert_eq!(ended["state"], "completed");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn stopping_a_task_ends_it_and_leaves_the_turn_alone() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([
        Reply::Exec {
            command: "sleep 300",
            background: true,
        },
        Reply::Hang("Waiting"),
        Reply::Text("Still here."),
    ]);
    connect_raw(script, true, async |cx, mut updates| {
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let id = session.session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Serve.")));
        let spawned = next_update(&mut updates, "async_task_spawned").await;
        let task = spawned["asyncTaskId"]
            .as_str()
            .expect("a task id")
            .to_owned();
        // The turn is under way: the model is waiting.
        next_update(&mut updates, "agent_message_chunk").await;

        let unknown = cx
            .send_request(background::StopRequest {
                session_id: id.clone(),
                async_task_id: "no-such-task".into(),
            })
            .block_task()
            .await?;
        assert!(!unknown.stopped);
        let elsewhere = cx
            .send_request(background::StopRequest {
                session_id: "session_unknown".into(),
                async_task_id: task.clone(),
            })
            .block_task()
            .await?;
        assert!(!elsewhere.stopped);
        let stopped = cx
            .send_request(background::StopRequest {
                session_id: id.clone(),
                async_task_id: task.clone(),
            })
            .block_task()
            .await?;
        assert!(stopped.stopped);
        let ended = next_update(&mut updates, "async_task_state_update").await;
        assert_eq!(ended["asyncTaskId"], task);
        assert_eq!(ended["state"], "stopped");

        // The turn was not cancelled: it ends when the client says so.
        cx.send_notification(CancelNotification::new(id.clone()))?;
        let done = answer.block_task().await?;
        assert_eq!(done.stop_reason, StopReason::Cancelled);
        let again = cx
            .send_request(background::StopRequest {
                session_id: id,
                async_task_id: task,
            })
            .block_task()
            .await?;
        assert!(!again.stopped);
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_foreground_command_moves_to_the_background_and_the_turn_ends() {
    use tokio::io::AsyncWriteExt;

    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let mut gate = gate(&cwd);
    let script = Script::new([
        Reply::Exec {
            command: "echo ready; read -r line < gate; echo released",
            background: false,
        },
        Reply::Text("Moved on."),
    ]);
    connect_raw(script, true, async |cx, mut updates| {
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let id = session.session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Run.")));
        let call = next_update(&mut updates, "tool_call").await["toolCallId"]
            .as_str()
            .expect("a tool call id")
            .to_owned();
        // The command prints a line once it runs.
        let running = next_update(&mut updates, "tool_call_update").await;
        assert_eq!(running["content"][0]["content"]["text"], "ready");

        let missed = cx
            .send_request(background::BackgroundTaskRequest {
                session_id: id.clone(),
                tool_call_id: Some("call-2".into()),
            })
            .block_task()
            .await?;
        assert!(!missed.backgrounded);
        let moved = cx
            .send_request(background::BackgroundTaskRequest {
                session_id: id.clone(),
                tool_call_id: Some(call.clone()),
            })
            .block_task()
            .await?;
        assert!(moved.backgrounded);

        // The turn ends while the command still waits on the fifo.
        let done = answer.block_task().await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        let spawned = next_update(&mut updates, "async_task_spawned").await;
        assert_eq!(spawned["toolCallId"], call);
        let finished = next_update(&mut updates, "tool_call_update").await;
        assert_eq!(finished["status"], "completed");
        assert!(meta_marks_backgrounded(&finished), "{finished}");

        gate.write_all(b"go\n")
            .await
            .expect("the fifo takes a line");
        let ended = next_update(&mut updates, "async_task_state_update").await;
        assert_eq!(ended["asyncTaskId"], spawned["asyncTaskId"]);
        assert_eq!(ended["state"], "completed");
        let nothing = cx
            .send_request(background::BackgroundTaskRequest {
                session_id: id,
                tool_call_id: None,
            })
            .block_task()
            .await?;
        assert!(!nothing.backgrounded);
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn a_client_without_the_extension_hears_nothing_of_background_commands() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let script = Script::new([
        Reply::Exec {
            command: "echo hi",
            background: true,
        },
        Reply::Text("Started it."),
    ]);
    connect_raw(script, false, async |cx, mut updates| {
        let session = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        let done = cx
            .send_request(PromptRequest::new(session.session_id, text("Serve.")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        let mut seen = Vec::new();
        while let Ok(update) = updates.try_recv() {
            seen.push(update);
        }
        assert!(
            seen.iter()
                .any(|update| update["sessionUpdate"] == "tool_call_update"),
            "the command still reports as a tool call: {seen:?}"
        );
        for update in &seen {
            let kind = update["sessionUpdate"].as_str().unwrap_or_default();
            assert!(!kind.starts_with("async_task"), "{update}");
            assert!(update.get("_meta").is_none(), "{update}");
        }
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn thinking_config_tracks_commands_and_controls_the_next_prompt() {
    use agent_client_protocol::schema::v1::{
        LoadSessionRequest, ResumeSessionRequest, SetSessionConfigOptionRequest,
    };
    use flashmind_types::ReasoningLevel;

    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    let mut script = Script::new([Reply::Text("first"), Reply::Text("second")]);
    Arc::get_mut(&mut script).unwrap().initial_reasoning = ReasoningLevel::Medium;
    let recorded = script.clone();
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(cwd.clone()))
            .block_task()
            .await?;
        let options = serde_json::to_value(session.config_options.unwrap()).unwrap();
        assert_eq!(
            options[0],
            json!({
                "id": "thought_level", "name": "Thinking level", "category": "thought_level",
                "type": "select", "currentValue": "medium",
                "options": [{"value":"off","name":"Off"}, {"value":"low","name":"Low"},
                            {"value":"medium","name":"Medium"}, {"value":"high","name":"High"}]
            })
        );
        let id = session.session_id;
        for (config, value) in [("thought_level", "bad"), ("unknown", "high")] {
            let error = cx
                .send_request(SetSessionConfigOptionRequest::new(
                    id.clone(),
                    config,
                    value,
                ))
                .block_task()
                .await
                .unwrap_err();
            assert_eq!(serde_json::to_value(error).unwrap()["code"], -32602);
        }
        let changed = cx
            .send_request(SetSessionConfigOptionRequest::new(
                id.clone(),
                "thought_level",
                "high",
            ))
            .block_task()
            .await?;
        assert_eq!(
            serde_json::to_value(changed).unwrap()["configOptions"][0]["currentValue"],
            "high"
        );
        cx.send_request(PromptRequest::new(id.clone(), text("first")))
            .block_task()
            .await?;
        assert_eq!(
            recorded.requests.lock().unwrap()[0].reasoning,
            ReasoningLevel::High
        );
        cx.send_request(PromptRequest::new(id.clone(), text("/thinking low")))
            .block_task()
            .await?;
        assert!(drain(&mut updates).iter().any(|update| {
            let value = serde_json::to_value(update).unwrap();
            value["sessionUpdate"] == "config_option_update"
                && value["configOptions"][0]["currentValue"] == "low"
        }));
        drain(&mut updates);
        let loaded = cx
            .send_request(LoadSessionRequest::new(id.clone(), cwd.clone()))
            .block_task()
            .await?;
        assert_eq!(
            serde_json::to_value(loaded).unwrap()["configOptions"][0]["currentValue"],
            "low"
        );
        let replayed = drain(&mut updates);
        assert_eq!(message(&replayed), "first");
        assert!(
            replayed
                .iter()
                .any(|update| matches!(update, SessionUpdate::UserMessageChunk(_)))
        );
        let resumed = cx
            .send_request(ResumeSessionRequest::new(id.clone(), cwd))
            .block_task()
            .await?;
        assert_eq!(
            serde_json::to_value(resumed).unwrap()["configOptions"][0]["currentValue"],
            "low"
        );
        cx.send_request(PromptRequest::new(id, text("second")))
            .block_task()
            .await?;
        assert_eq!(
            recorded.requests.lock().unwrap()[1].reasoning,
            ReasoningLevel::Low
        );
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn changing_thinking_during_a_turn_only_affects_the_next_prompt() {
    use agent_client_protocol::schema::v1::SetSessionConfigOptionRequest;
    use flashmind_types::ReasoningLevel;

    let dir = tempfile::tempdir().unwrap();
    let script = Script::new([Reply::Hang("Working"), Reply::Text("Done")]);
    let recorded = script.clone();
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
            .block_task()
            .await?;
        let id = session.session_id;
        let answer = cx.send_request(PromptRequest::new(id.clone(), text("Work")));
        while !matches!(
            updates.recv().await.unwrap(),
            SessionUpdate::AgentMessageChunk(_)
        ) {}
        cx.send_request(SetSessionConfigOptionRequest::new(
            id.clone(),
            "thought_level",
            "high",
        ))
        .block_task()
        .await?;
        assert_eq!(
            recorded.requests.lock().unwrap()[0].reasoning,
            ReasoningLevel::Off
        );
        cx.send_notification(CancelNotification::new(id.clone()))?;
        assert_eq!(
            answer.block_task().await?.stop_reason,
            StopReason::Cancelled
        );
        cx.send_request(PromptRequest::new(id, text("Continue")))
            .block_task()
            .await?;
        assert_eq!(
            recorded.requests.lock().unwrap()[1].reasoning,
            ReasoningLevel::High
        );
        Ok(())
    })
    .await;
}

/// Reopens the JSONL store on each call, like a fresh server process.
struct PersistentScript {
    database: std::path::PathBuf,
    script: Arc<Script>,
}

struct SwitchingScript {
    first: Arc<Script>,
    second: Arc<Script>,
}

#[async_trait]
impl Opener for SwitchingScript {
    async fn open(&self, cwd: &Path, resume: Option<&str>) -> Result<Opened> {
        Scripted {
            script: self.first.clone(),
            pricing: ModelPricing::default(),
        }
        .open(cwd, resume)
        .await
    }

    fn provider(&self, model: &flashmind_types::Model) -> Result<Arc<dyn LlmProvider>> {
        anyhow::ensure!(
            model.provider == self.second.provider,
            "provider is not configured"
        );
        Ok(self.second.clone())
    }
}

#[tokio::test]
async fn switching_providers_routes_the_next_prompt_and_keeps_the_conversation() {
    use agent_client_protocol::schema::v1::SetSessionConfigOptionRequest;

    let dir = tempfile::tempdir().unwrap();
    let first = Script::new([Reply::Text("First answer")]);
    let mut second = Script::new([Reply::Text("Second answer")]);
    Arc::get_mut(&mut second).unwrap().provider = Provider::OpenRouter;
    connect_opener(
        SwitchingScript {
            first: first.clone(),
            second: second.clone(),
        },
        async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let id = cx
                .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
                .block_task()
                .await?
                .session_id;
            cx.send_request(PromptRequest::new(id.clone(), text("First question")))
                .block_task()
                .await?;
            cx.send_request(SetSessionConfigOptionRequest::new(
                id.clone(),
                "model",
                "openrouter:vendor/second",
            ))
            .block_task()
            .await?;
            cx.send_request(PromptRequest::new(id, text("Second question")))
                .block_task()
                .await?;
            assert_eq!(first.requests.lock().unwrap().len(), 1);
            let requests = second.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].model.to_string(), "openrouter:vendor/second");
            assert!(
                requests[0]
                    .messages
                    .iter()
                    .any(|message| message.content == "First question")
            );
            assert!(
                requests[0]
                    .messages
                    .iter()
                    .any(|message| message.content == "First answer")
            );
            Ok(())
        },
    )
    .await;
}

#[tokio::test]
async fn switching_models_during_a_turn_preserves_history_and_survives_reopening() {
    use agent_client_protocol::schema::v1::{LoadSessionRequest, SetSessionConfigOptionRequest};

    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("sessions");
    let release = Arc::new(tokio::sync::Notify::new());
    let script = Script::new([
        Reply::Paused {
            release: release.clone(),
        },
        Reply::Text("Second answer"),
    ]);
    let mut saved_id = None;
    connect_opener(
        PersistentScript {
            database: database.clone(),
            script: script.clone(),
        },
        async |cx, mut updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(dir.path().to_path_buf()))
                .block_task()
                .await?;
            let options = serde_json::to_value(session.config_options).unwrap();
            assert_eq!(options[1]["category"], "model");
            assert_eq!(options[1]["currentValue"], "ollama:script");
            let id = session.session_id;
            let first = cx.send_request(PromptRequest::new(id.clone(), text("First question")));
            while !matches!(
                updates.recv().await.unwrap(),
                SessionUpdate::AgentMessageChunk(_)
            ) {}
            cx.send_request(SetSessionConfigOptionRequest::new(
                id.clone(),
                "model",
                "ollama:second",
            ))
            .block_task()
            .await?;
            assert_eq!(
                script.requests.lock().unwrap()[0].model.to_string(),
                "ollama:script"
            );
            release.notify_one();
            first.block_task().await?;
            cx.send_request(PromptRequest::new(id.clone(), text("Second question")))
                .block_task()
                .await?;
            {
                let requests = script.requests.lock().unwrap();
                assert_eq!(requests[1].model.to_string(), "ollama:second");
                assert!(
                    requests[1]
                        .messages
                        .iter()
                        .any(|m| m.content == "First question")
                );
            }
            // Save the selection even when no further prompt follows it.
            cx.send_request(SetSessionConfigOptionRequest::new(
                id.clone(),
                "model",
                "ollama:third",
            ))
            .block_task()
            .await?;
            for value in ["missing-provider", "ollama:", "openai:unconfigured"] {
                assert!(
                    cx.send_request(SetSessionConfigOptionRequest::new(
                        id.clone(),
                        "model",
                        value
                    ))
                    .block_task()
                    .await
                    .is_err()
                );
            }
            let stored = flashmind_memory::session::SessionStore::open(&database)?;
            let summary = stored
                .list_sessions()
                .await?
                .into_iter()
                .find(|summary| summary.chat_key == id.0.as_ref())
                .unwrap();
            assert_eq!(summary.working_dir.as_deref(), dir.path().to_str());
            assert_eq!(summary.model.as_deref(), Some("ollama:third"));
            saved_id = Some(id);
            Ok(())
        },
    )
    .await;
    connect_opener(
        PersistentScript { database, script },
        async |cx, mut updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let loaded = cx
                .send_request(LoadSessionRequest::new(
                    saved_id.unwrap(),
                    dir.path().to_path_buf(),
                ))
                .block_task()
                .await?;
            assert_eq!(
                serde_json::to_value(loaded.config_options).unwrap()[1]["currentValue"],
                "ollama:third"
            );
            assert!(message(&drain(&mut updates)).contains("Second answer"));
            Ok(())
        },
    )
    .await;
}

#[async_trait]
impl Opener for PersistentScript {
    async fn open(&self, cwd: &Path, resume: Option<&str>) -> Result<Opened> {
        use super::open::Saved;
        let store = flashmind_memory::session::SessionStore::open(&self.database)?;
        let mut opened = Scripted {
            script: self.script.clone(),
            pricing: ModelPricing::default(),
        }
        .open(cwd, None)
        .await?;
        if let Some(key) = resume {
            opened.key = key.to_owned();
            opened.conversation =
                crate::session::load_conversation_from(&store, key, "test").await?;
            opened.system_instructions =
                SystemInstructions::restore(&store, key, &mut opened.conversation).await?;
            if let Some(model) = store
                .list_sessions()
                .await?
                .into_iter()
                .find(|s| s.chat_key == key)
                .and_then(|s| s.model)
            {
                opened.agent.llm_mut().model = model.parse()?;
            }
            if let Some(level) = store.reasoning(key).await? {
                opened.agent.llm_mut().reasoning = level;
            }
        }
        opened.saved = Some(Saved { store });
        Ok(opened)
    }
}

#[tokio::test]
async fn new_sessions_save_their_directory_before_the_first_prompt() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("sessions");
    let cwd = directory.path().to_path_buf();
    connect_opener(
        PersistentScript {
            database: database.clone(),
            script: Script::new([]),
        },
        async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            use sha2::{Digest, Sha256};
            let name: String = Sha256::digest(session.session_id.to_string())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let file = database.join(format!("{name}.jsonl"));
            let text = std::fs::read_to_string(file).unwrap();
            let header: serde_json::Value = serde_json::from_str(text.lines().next().unwrap())?;
            assert_eq!(header["metadata"]["working_dir"], cwd.to_str().unwrap());
            assert!(header["metadata"]["total_usage"].is_null());
            Ok(())
        },
    )
    .await;
}

#[tokio::test]
async fn thinking_selection_and_history_survive_server_restart() {
    use agent_client_protocol::schema::v1::{
        LoadSessionRequest, ResumeSessionRequest, SetSessionConfigOptionRequest,
    };
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("sessions");
    let cwd = directory.path().to_path_buf();
    let script = Script::new([Reply::Text("Saved answer")]);
    let mut id = None;
    connect_opener(
        PersistentScript {
            database: database.clone(),
            script: script.clone(),
        },
        async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            cx.send_request(PromptRequest::new(
                session.session_id.clone(),
                text("Saved question"),
            ))
            .block_task()
            .await?;
            // No prompt follows the picker change, so persistence cannot depend on saving a turn.
            cx.send_request(SetSessionConfigOptionRequest::new(
                session.session_id.clone(),
                "thought_level",
                "high",
            ))
            .block_task()
            .await?;
            id = Some(session.session_id);
            Ok(())
        },
    )
    .await;
    let id = id.unwrap();
    connect_opener(PersistentScript { database: database.clone(), script: script.clone() }, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1)).block_task().await?;
        let loaded = cx.send_request(LoadSessionRequest::new(id.clone(), cwd.clone())).block_task().await?;
        assert_eq!(serde_json::to_value(loaded).unwrap()["configOptions"][0]["currentValue"], "high");
        let history = drain(&mut updates);
        assert_eq!(message(&history), "Saved answer");
        assert!(matches!(&history[0], SessionUpdate::UserMessageChunk(chunk) if matches!(&chunk.content, ContentBlock::Text(text) if text.text == "Saved question")));
        cx.send_request(PromptRequest::new(id.clone(), text("/thinking low"))).block_task().await?;
        Ok(())
    }).await;
    connect_opener(
        PersistentScript { database, script },
        async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let resumed = cx
                .send_request(ResumeSessionRequest::new(id, cwd))
                .block_task()
                .await?;
            assert_eq!(
                serde_json::to_value(resumed).unwrap()["configOptions"][0]["currentValue"],
                "low"
            );
            Ok(())
        },
    )
    .await;
}

#[tokio::test]
async fn system_command_appends_instructions_without_a_model_call() {
    let directory = tempfile::tempdir().unwrap();
    let script = Script::new([
        Reply::Text("First answer"),
        Reply::Text("Summary"),
        Reply::Text("After compaction"),
        Reply::Text("Other session"),
    ]);
    let expected = "You are a test.\n\nKeep replies short.\nUse plain text.\n\nPrefer examples.";
    connect(script.clone(), async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let cwd = directory.path().to_path_buf();
        let session = cx
            .send_request(NewSessionRequest::new(cwd.clone()))
            .block_task()
            .await?;
        let offered = drain(&mut updates);
        assert!(offered.iter().any(|update| matches!(
            update,
            SessionUpdate::AvailableCommandsUpdate(commands)
                if commands.available_commands.iter().any(|command| command.name == "system")
        )));
        let id = session.session_id;
        for (command, response) in [
            (" /system \n", "You are a test."),
            (
                "/system Keep replies short.\nUse plain text.",
                "Added system instructions.",
            ),
            ("/system Prefer examples.", "Added system instructions."),
            ("/system", expected),
        ] {
            let done = cx
                .send_request(PromptRequest::new(id.clone(), text(command)))
                .block_task()
                .await?;
            assert_eq!(done.stop_reason, StopReason::EndTurn);
            assert_eq!(message(&drain(&mut updates)), response);
            assert!(script.requests.lock().unwrap().is_empty());
        }
        for prompt in ["First question", "/compact", "Second question"] {
            cx.send_request(PromptRequest::new(id.clone(), text(prompt)))
                .block_task()
                .await?;
        }
        let other = cx
            .send_request(NewSessionRequest::new(cwd))
            .block_task()
            .await?;
        cx.send_request(PromptRequest::new(other.session_id, text("Other question")))
            .block_task()
            .await?;
        Ok(())
    })
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    for index in [0, 2] {
        assert_eq!(requests[index].messages[0].role, Role::System);
        assert_eq!(requests[index].messages[0].content, expected);
    }
    assert_eq!(requests[3].messages[0].content, "You are a test.");
}

#[tokio::test]
async fn system_additions_survive_server_restart_with_a_refreshed_base_prompt() {
    use agent_client_protocol::schema::v1::ResumeSessionRequest;

    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("sessions");
    let cwd = directory.path().to_path_buf();
    let script = Script::new([Reply::Text("Resumed answer")]);
    let mut id = None;
    connect_opener(
        PersistentScript {
            database: database.clone(),
            script: script.clone(),
        },
        async |cx, _updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            for command in ["/system Keep replies short.", "/system Prefer examples."] {
                cx.send_request(PromptRequest::new(
                    session.session_id.clone(),
                    text(command),
                ))
                .block_task()
                .await?;
            }
            id = Some(session.session_id);
            Ok(())
        },
    )
    .await;
    let id = id.unwrap();
    connect_opener(
        PersistentScript {
            database,
            script: script.clone(),
        },
        async |cx, mut updates| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            cx.send_request(ResumeSessionRequest::new(id.clone(), cwd))
                .block_task()
                .await?;
            drain(&mut updates);
            cx.send_request(PromptRequest::new(id.clone(), text("/system")))
                .block_task()
                .await?;
            assert_eq!(
                message(&drain(&mut updates)),
                "test\n\nKeep replies short.\n\nPrefer examples."
            );
            cx.send_request(PromptRequest::new(id, text("Resumed question")))
                .block_task()
                .await?;
            Ok(())
        },
    )
    .await;
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].messages[0].content,
        "test\n\nKeep replies short.\n\nPrefer examples."
    );
}

#[tokio::test]
async fn credits_are_structured_separately_from_reply_text_and_context_usage() {
    use flashmind_types::{CreditBalance, CreditScope, CreditUnit};
    let dir = tempfile::tempdir().unwrap();
    let mut script = Script::new([Reply::Counted {
        text: "Done.",
        prompt_tokens: 100,
        completion_tokens: 20,
    }]);
    Arc::get_mut(&mut script).unwrap().balance = Some(CreditBalance {
        scope: CreditScope::ApiKey,
        unit: CreditUnit::Usd,
        remaining: Some(Decimal::new(725, 2)),
        total: Some(Decimal::TEN),
        used: Some(Decimal::new(275, 2)),
        resets: Some("monthly".into()),
    });
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(dir.path()))
            .block_task()
            .await?;
        let done = cx
            .send_request(PromptRequest::new(session.session_id, text("Go.")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        let shown = drain(&mut updates);
        assert_eq!(message(&shown), "Done.");
        let usages = usage(&shown);
        let last = usages.last().unwrap();
        assert_eq!(last.used, 120);
        assert_eq!(last.size, usages[0].size);
        let meta = last.meta.as_ref().unwrap();
        assert_eq!(meta["flsh/credits"]["balance"]["remaining"], "7.25");
        assert_eq!(meta["flsh/credits"]["balance"]["scope"], "api_key");
        assert_eq!(meta["flsh/sessionUsage"]["input_tokens"], 100);
        assert_eq!(meta["flsh/sessionUsage"]["output_tokens"], 20);
        assert!(last.cost.is_none());
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn failed_credit_lookup_preserves_the_completed_acp_reply() {
    let dir = tempfile::tempdir().unwrap();
    let mut script = Script::new([Reply::Text("Done.")]);
    Arc::get_mut(&mut script).unwrap().balance_fails = true;
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(dir.path()))
            .block_task()
            .await?;
        let done = cx
            .send_request(PromptRequest::new(session.session_id, text("Go.")))
            .block_task()
            .await?;
        assert_eq!(done.stop_reason, StopReason::EndTurn);
        assert_eq!(message(&drain(&mut updates)), "Done.");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn credits_are_structured_even_without_reported_tokens() {
    use flashmind_types::{CreditScope, CreditUnit};
    let dir = tempfile::tempdir().unwrap();
    let mut script = Script::new([Reply::Text("Done.")]);
    Arc::get_mut(&mut script).unwrap().balance = Some(CreditBalance {
        scope: CreditScope::ApiKey,
        unit: CreditUnit::Usd,
        remaining: None,
        total: None,
        used: Some(Decimal::from(42)),
        resets: None,
    });
    connect(script, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(dir.path()))
            .block_task()
            .await?;
        cx.send_request(PromptRequest::new(session.session_id, text("Go.")))
            .block_task()
            .await?;
        let shown = drain(&mut updates);
        let usages = usage(&shown);
        let meta = usages.last().unwrap().meta.as_ref().unwrap();
        assert_eq!(meta["flsh/credits"]["balance"]["used"], "42");
        assert_eq!(
            meta["flsh/credits"]["balance"]["remaining"],
            serde_json::Value::Null
        );
        assert_eq!(meta["flsh/contextUsageAvailable"], false);
        assert_eq!(message(&shown), "Done.");
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn credit_consumption_is_per_turn_and_does_not_use_account_spend() {
    use flashmind_types::{CreditScope, CreditUnit};
    let dir = tempfile::tempdir().unwrap();
    let mut script = Script::new((0..2).map(|_| Reply::Counted {
        text: "Done.",
        prompt_tokens: 1000,
        completion_tokens: 100,
    }));
    Arc::get_mut(&mut script).unwrap().balance = Some(CreditBalance {
        scope: CreditScope::Account,
        unit: CreditUnit::Usd,
        remaining: Some(Decimal::from(70)),
        total: Some(Decimal::from(100)),
        used: Some(Decimal::from(30)),
        resets: None,
    });
    let pricing = ModelPricing {
        prompt: Some(Decimal::new(2, 6)),
        completion: Some(Decimal::new(10, 6)),
        ..Default::default()
    };
    connect_priced(script, pricing, async |cx, mut updates| {
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let session = cx
            .send_request(NewSessionRequest::new(dir.path()))
            .block_task()
            .await?;
        for turn in 1..=2 {
            cx.send_request(PromptRequest::new(session.session_id.clone(), text("Go.")))
                .block_task()
                .await?;
            let shown = drain(&mut updates);
            let usages = usage(&shown);
            let meta = usages.last().unwrap().meta.as_ref().unwrap();
            assert_eq!(meta["flsh/credits"]["consumed"], "0.003");
            assert_eq!(meta["flsh/sessionUsage"]["input_tokens"], turn * 1000);
            assert_eq!(meta["flsh/sessionUsage"]["output_tokens"], turn * 100);
            assert_eq!(meta["flsh/credits"]["balance"]["used"], "30");
            assert_eq!(message(&shown), "Done.");
        }
        Ok(())
    })
    .await;
}

#[test]
fn project_cache_routing_is_stable_and_scoped_to_the_directory() {
    let root = tempfile::tempdir().unwrap();
    let first = root.path().join("first");
    let second = root.path().join("second");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();

    let first_cache = super::open::project_cache_key(&first).unwrap();
    let second_cache = super::open::project_cache_key(&first.join(".")).unwrap();

    assert_eq!(first_cache, second_cache);
    assert_eq!(first_cache.as_str().len(), 64);
    assert_ne!(
        first_cache,
        super::open::project_cache_key(&second).unwrap()
    );
    assert!(super::open::project_cache_key(&root.path().join("missing")).is_err());
}

#[cfg(unix)]
#[test]
fn project_cache_routing_resolves_directory_symlinks() {
    let root = tempfile::tempdir().unwrap();
    let alias = root.path().join("alias");
    let project = root.path().join("project");
    std::fs::create_dir(&project).unwrap();
    std::os::unix::fs::symlink(&project, &alias).unwrap();

    assert_eq!(
        super::open::project_cache_key(&project).unwrap(),
        super::open::project_cache_key(&alias).unwrap(),
    );
}
