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
    AgentLlmConfig, CompletionRequest, CompletionStream, ContentPart, FinishReason, LlmProvider,
    ModelCapabilities, ModelPricing, Provider, Role, StreamEvent, TokenUsage,
};
use futures_util::{StreamExt, stream};
use rust_decimal::Decimal;
use serde_json::json;
use tokio::sync::mpsc;

use super::background;
use super::media::MAX_INLINE_IMAGE;
use super::open::{Opened, Opener};
use super::server::Server;
use super::usage::Meter;
use crate::session::new_session_key;

/// What the scripted model answers to one request.
enum Reply {
    Text(&'static str),
    /// Calls a tool that does not exist, so the call fails.
    MissingTool,
    /// Sends the text, then nothing until the turn is cancelled.
    Hang(&'static str),
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
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<CompletionRequest>>,
    initial_reasoning: flashmind_types::ReasoningLevel,
}

impl Script {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            initial_reasoning: flashmind_types::ReasoningLevel::Off,
        })
    }
}

#[async_trait]
impl LlmProvider for Script {
    fn name(&self) -> &str {
        "script"
    }

    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        self.requests.lock().unwrap().push(request);
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
            .build_sync();
        agent.apply_features(ModelCapabilities::default(), Some(WINDOW));
        agent.llm_mut().reasoning = self.script.initial_reasoning;
        Ok(Opened {
            key: new_session_key(),
            agent,
            conversation: Conversation::with_system("You are a test."),
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
    let server = Server::new(opener);
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
    let server = Server::new(Scripted {
        script,
        pricing: ModelPricing::default(),
    });
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
            assert_eq!(initialized.meta.is_some(), air);
            client(cx, updates).await
        })
        .await
        .expect("The ACP exchange should succeed.");
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

/// Reopens the SQLite store on each call, like a fresh server process.
struct PersistentScript {
    database: std::path::PathBuf,
    script: Arc<Script>,
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
            if let Some(level) = store.reasoning(key).await? {
                opened.agent.llm_mut().reasoning = level;
            }
        }
        opened.saved = Some(Saved {
            store,
            model: opened.agent.llm().model.to_string(),
        });
        Ok(opened)
    }
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
