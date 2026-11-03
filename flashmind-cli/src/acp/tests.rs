//! `flsh acp` driven by an ACP client over a scripted model.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, ImageContent, InitializeRequest, NewSessionRequest,
    PromptRequest, SessionNotification, SessionUpdate, StopReason, TextContent, ToolCallStatus,
    UsageUpdate,
};
use agent_client_protocol::{Agent as AcpAgent, Client, ConnectionTo, on_receive_notification};
use anyhow::Result;
use async_trait::async_trait;
use flashmind_core::{Agent, Conversation};
use flashmind_types::{
    AgentLlmConfig, CompletionRequest, CompletionStream, ContentPart, FinishReason, LlmProvider,
    ModelCapabilities, ModelPricing, Provider, Role, StreamEvent, TokenUsage,
};
use futures_util::{StreamExt, stream};
use rust_decimal::Decimal;
use tokio::sync::mpsc;

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
}

impl Script {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
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

/// Opens sessions over the scripted model, priced at `pricing`, without
/// tools or saving.
struct Scripted {
    script: Arc<Script>,
    pricing: ModelPricing,
}

#[async_trait]
impl Opener for Scripted {
    async fn open(&self, cwd: &Path, _resume: Option<&str>) -> Result<Opened> {
        let model: flashmind_types::Model = "ollama:script".parse()?;
        let mut agent = Agent::builder(self.script.clone())
            .llm(AgentLlmConfig::new(model.clone()))
            .working_dir(cwd.to_path_buf())
            .build_sync();
        agent.apply_features(ModelCapabilities::default(), Some(WINDOW));
        Ok(Opened {
            key: new_session_key(),
            agent,
            conversation: Conversation::with_system("You are a test."),
            saved: None,
            subagents: None,
            tool_sync: None,
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
    let server = Server::new(Scripted { script, pricing });
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
