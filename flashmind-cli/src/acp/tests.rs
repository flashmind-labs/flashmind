//! `flsh acp` driven by an ACP client over a scripted model.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest,
    SessionNotification, SessionUpdate, StopReason, TextContent, ToolCallStatus,
};
use agent_client_protocol::{Agent as AcpAgent, Client, ConnectionTo, on_receive_notification};
use anyhow::Result;
use async_trait::async_trait;
use flashmind_core::{Agent, Conversation};
use flashmind_types::{
    AgentLlmConfig, CompletionRequest, CompletionStream, FinishReason, LlmProvider, Provider,
    StreamEvent,
};
use futures_util::{StreamExt, stream};
use tokio::sync::mpsc;

use super::open::{Opened, Opener};
use super::server::Server;
use crate::session::new_session_key;

/// What the scripted model answers to one request.
enum Reply {
    Text(&'static str),
    /// Calls a tool that does not exist, so the call fails.
    MissingTool,
    /// Sends the text, then nothing until the turn is cancelled.
    Hang(&'static str),
}

/// A model that answers each request with the next reply of its script.
struct Script(Mutex<VecDeque<Reply>>);

impl Script {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(replies.into_iter().collect())))
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

    fn complete(&self, _: CompletionRequest) -> CompletionStream {
        let reply = self.0.lock().unwrap().pop_front();
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

/// Opens sessions over the scripted model, without tools or saving.
struct Scripted(Arc<Script>);

#[async_trait]
impl Opener for Scripted {
    async fn open(&self, cwd: &Path, _resume: Option<&str>) -> Result<Opened> {
        let agent = Agent::builder(self.0.clone())
            .llm(AgentLlmConfig::new("ollama:script".parse()?))
            .working_dir(cwd.to_path_buf())
            .build_sync();
        Ok(Opened {
            key: new_session_key(),
            agent,
            conversation: Conversation::with_system("You are a test."),
            saved: None,
            subagents: None,
            tool_sync: None,
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

/// Connects a client to a server over `script` and runs `client` with the
/// connection and the updates it receives.
async fn connect(
    script: Arc<Script>,
    client: impl AsyncFnOnce(
        ConnectionTo<AcpAgent>,
        mpsc::UnboundedReceiver<SessionUpdate>,
    ) -> Result<(), agent_client_protocol::Error>,
) {
    let server = Server::new(Scripted(script));
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
        let first = updates.recv().await.expect("the turn sends a chunk");
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
