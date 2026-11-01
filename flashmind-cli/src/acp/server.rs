//! The ACP agent: answers the client's requests, and runs each session's
//! prompts one at a time in a task of its own.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ContentBlock, Implementation, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities, SessionId,
    SessionNotification, SessionResumeCapabilities, StopReason,
};
use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Error, Responder, on_receive_notification,
    on_receive_request,
};
use flashmind_core::subagent::Notifications;
use flashmind_core::{CancellationToken, ConversationEntry, EntryKind};
use flashmind_tools::ToolSync;
use flashmind_types::AgentInput;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::open::{Opened, Opener};
use super::updates::{Ending, Updates};
use crate::interactive::prompt_input;
use crate::session::save_turn;

/// The most prompts waiting behind a session's running turn.
const MAX_QUEUED: usize = 8;

/// A prompt waiting for its session's task.
struct Prompt {
    request: PromptRequest,
    responder: Responder<PromptResponse>,
    /// Cancelled by `session/cancel`, also while the prompt still waits.
    cancel: CancellationToken,
}

/// What the handlers keep of an open session.
struct Handle {
    prompts: mpsc::Sender<Prompt>,
    /// The tokens of the prompts sent and not yet answered.
    pending: Arc<Mutex<VecDeque<CancellationToken>>>,
    tool_sync: Option<Arc<ToolSync>>,
}

/// The state the handlers share.
#[derive(Clone)]
pub(crate) struct Server {
    opener: Arc<dyn Opener>,
    sessions: Arc<Mutex<HashMap<SessionId, Handle>>>,
}

impl Server {
    pub(crate) fn new(opener: impl Opener) -> Self {
        Self {
            opener: Arc::new(opener),
            sessions: Arc::default(),
        }
    }

    /// The agent side of the connection, ready to connect to a client.
    pub(crate) fn agent(&self) -> impl ConnectTo<Client> + 'static {
        let sessions = self.clone();
        let resumes = self.clone();
        let prompts = self.clone();
        let cancels = self.clone();
        Agent
            .builder()
            .name("flsh")
            .on_receive_request(
                async |request: InitializeRequest, responder, _cx| {
                    tracing::info!(client = ?request.client_info, "ACP client connected");
                    responder.respond(initialized())
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: NewSessionRequest,
                            responder: Responder<NewSessionResponse>,
                            cx: ConnectionTo<Client>| {
                    let server = sessions.clone();
                    let opening = cx.clone();
                    cx.spawn(async move {
                        match server.open(&opening, &request.cwd, None).await {
                            Ok(id) => responder.respond(NewSessionResponse::new(id)),
                            Err(error) => responder.respond_with_error(error),
                        }
                    })
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: ResumeSessionRequest,
                            responder: Responder<ResumeSessionResponse>,
                            cx: ConnectionTo<Client>| {
                    let server = resumes.clone();
                    let opening = cx.clone();
                    cx.spawn(async move {
                        let resume = Some(request.session_id.0.as_ref());
                        match server.open(&opening, &request.cwd, resume).await {
                            Ok(_) => responder.respond(ResumeSessionResponse::new()),
                            Err(error) => responder.respond_with_error(error),
                        }
                    })
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: PromptRequest, responder, _cx| {
                    prompts.prompt(request, responder)
                },
                on_receive_request!(),
            )
            .on_receive_notification(
                async move |cancel: CancelNotification, _cx| {
                    cancels.cancel(&cancel.session_id);
                    Ok(())
                },
                on_receive_notification!(),
            )
    }

    /// Opens a session working in `cwd`, or the saved session `resume`,
    /// and starts its task. A session open already is used as it is.
    async fn open(
        &self,
        cx: &ConnectionTo<Client>,
        cwd: &Path,
        resume: Option<&str>,
    ) -> Result<SessionId, Error> {
        if !cwd.is_absolute() {
            return Err(Error::invalid_params().data(format!(
                "the session directory must be absolute: {}",
                cwd.display()
            )));
        }
        if let Some(key) = resume
            && self.lock().contains_key(&SessionId::new(key))
        {
            return Ok(SessionId::new(key));
        }
        let opened = self
            .opener
            .open(cwd, resume)
            .await
            .map_err(|error| failed("could not open the session", &error))?;
        let id = SessionId::new(opened.key.as_str());
        let (prompts, queue) = mpsc::channel(MAX_QUEUED);
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        let task = Session {
            cx: cx.clone(),
            id: id.clone(),
            cwd: cwd.to_path_buf(),
            notifications: opened.subagents.clone().map(Notifications::new),
            pending: pending.clone(),
            opened,
        };
        let tool_sync = task.opened.tool_sync.clone();
        cx.spawn(task.run(queue))?;
        self.lock().insert(
            id.clone(),
            Handle {
                prompts,
                pending,
                tool_sync,
            },
        );
        tracing::info!(session = %id, cwd = %cwd.display(), "ACP session opened");
        Ok(id)
    }

    /// Hands `request` to its session's task, which answers it.
    fn prompt(
        &self,
        request: PromptRequest,
        responder: Responder<PromptResponse>,
    ) -> Result<(), Error> {
        let sessions = self.lock();
        let Some(handle) = sessions.get(&request.session_id) else {
            return responder.respond_with_error(Error::session_not_found(request.session_id));
        };
        let cancel = CancellationToken::new();
        let prompt = Prompt {
            request,
            responder,
            cancel: cancel.clone(),
        };
        // Pushed first, so the task always finds it: the task takes tokens
        // from the front, in the order the prompts were queued.
        lock(&handle.pending).push_back(cancel);
        let refused = match handle.prompts.try_send(prompt) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(prompt)) => (
                prompt,
                Error::invalid_request()
                    .data(format!("{MAX_QUEUED} prompts already wait in this session")),
            ),
            Err(mpsc::error::TrySendError::Closed(prompt)) => (
                prompt,
                Error::internal_error().data("the session has ended"),
            ),
        };
        lock(&handle.pending).pop_back();
        let (prompt, error) = refused;
        prompt.responder.respond_with_error(error)
    }

    /// Stops the running turn of `session` and every prompt waiting.
    fn cancel(&self, session: &SessionId) {
        let sessions = self.lock();
        let Some(handle) = sessions.get(session) else {
            tracing::debug!(%session, "cancel for an unknown session");
            return;
        };
        for token in lock(&handle.pending).iter() {
            token.cancel();
        }
    }

    /// Closes what the sessions hold open, such as MCP servers.
    pub(crate) async fn shutdown(&self) {
        let syncs: Vec<Arc<ToolSync>> = self
            .lock()
            .drain()
            .filter_map(|(_, handle)| handle.tool_sync)
            .collect();
        for sync in syncs {
            sync.shutdown().await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<SessionId, Handle>> {
        lock(&self.sessions)
    }
}

/// Locks `mutex`, also after a panic elsewhere: its data is a plain list
/// that no half-done change can leave inconsistent.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The answer to `initialize`: protocol 1, and sessions that resume. flsh
/// reads and writes files and runs commands itself, so it asks the client
/// for none of that.
fn initialized() -> InitializeResponse {
    let sessions = SessionCapabilities::new().resume(SessionResumeCapabilities::new());
    InitializeResponse::new(ProtocolVersion::V1)
        .agent_capabilities(AgentCapabilities::new().session_capabilities(sessions))
        .agent_info(Implementation::new("flsh", env!("CARGO_PKG_VERSION")))
}

fn failed(what: &str, error: &anyhow::Error) -> Error {
    Error::internal_error().data(format!("{what}: {error:#}"))
}

/// One session's task: it owns the flsh session and runs its prompts.
struct Session {
    cx: ConnectionTo<Client>,
    id: SessionId,
    cwd: PathBuf,
    opened: Opened,
    notifications: Option<Notifications>,
    pending: Arc<Mutex<VecDeque<CancellationToken>>>,
}

impl Session {
    async fn run(mut self, mut queue: mpsc::Receiver<Prompt>) -> Result<(), Error> {
        while let Some(prompt) = queue.recv().await {
            let answer = if prompt.cancel.is_cancelled() {
                Ok(StopReason::Cancelled)
            } else {
                self.turn(&prompt.request, &prompt.cancel).await
            };
            self.save().await;
            lock(&self.pending).pop_front();
            match answer {
                Ok(stop) => prompt.responder.respond(PromptResponse::new(stop))?,
                Err(error) => prompt.responder.respond_with_error(error)?,
            }
        }
        Ok(())
    }

    /// Runs one prompt to the end of its turn, streaming its updates.
    async fn turn(
        &mut self,
        request: &PromptRequest,
        cancel: &CancellationToken,
    ) -> Result<StopReason, Error> {
        let Opened {
            agent,
            conversation,
            tool_sync,
            ..
        } = &mut self.opened;
        if let Some(sync) = tool_sync {
            sync.sync(agent.tools_mut()).await;
        }
        if let Some(notifications) = &mut self.notifications {
            notifications.deliver(conversation);
        }
        let mut input = prompt_input(&self.cwd, prompt_text(&request.prompt));
        loop {
            let mut updates = Updates::default();
            let mut ending = None;
            {
                let stream = agent.start(conversation, cancel.clone(), input, None);
                futures_util::pin_mut!(stream);
                while let Some(event) = stream.next().await {
                    for update in updates.updates(&event) {
                        self.cx
                            .send_notification(SessionNotification::new(self.id.clone(), update))?;
                    }
                    ending = ending.or(Ending::of(&event));
                }
            }
            if cancel.is_cancelled() || ending == Some(Ending::Cancelled) {
                answer_open_tool_calls(conversation);
                conversation.add(ConversationEntry::system_message(
                    "The user interrupted the previous turn.",
                ));
                return Ok(StopReason::Cancelled);
            }
            if let Some(Ending::Failed(error)) = ending {
                return Err(Error::internal_error().data(error));
            }
            // Results of delegated agents that finished during the turn
            // go to the agent before the turn ends.
            let delivered = self
                .notifications
                .as_mut()
                .is_some_and(|notifications| notifications.deliver(conversation));
            if !delivered {
                return Ok(StopReason::EndTurn);
            }
            input = AgentInput::Resume;
        }
    }

    /// Saves the conversation for `flsh resume`. A failure is logged rather
    /// than returned: the turn itself went through, and the next one saves
    /// the whole conversation again.
    async fn save(&self) {
        let Some(saved) = &self.opened.saved else {
            return;
        };
        let key = &self.opened.key;
        let result = async {
            save_turn(&saved.store, key, &self.opened.conversation).await?;
            saved
                .store
                .save_meta(key, None, Some(&saved.model), self.cwd.to_str())
                .await
        }
        .await;
        if let Err(error) = result {
            tracing::warn!(session = %key, "could not save the session: {error:#}");
        }
    }
}

/// The text of a prompt: its text blocks, and the URI of each resource
/// it links.
fn prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.clone()),
            ContentBlock::ResourceLink(link) => Some(link.uri.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Answers the tool calls a cancelled turn left without a result, so the
/// next request to the model has a result for each call.
fn answer_open_tool_calls(conversation: &mut flashmind_core::Conversation) {
    let entries = conversation.entries();
    let Some(last_call) = entries.iter().rposition(|entry| {
        matches!(&entry.kind, EntryKind::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty())
    }) else {
        return;
    };
    let EntryKind::Assistant {
        tool_calls: Some(calls),
        ..
    } = &entries[last_call].kind
    else {
        return;
    };
    let answered: Vec<&str> = entries[last_call..]
        .iter()
        .filter_map(|entry| match &entry.kind {
            EntryKind::Tool { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    let open: Vec<String> = calls
        .iter()
        .filter(|call| !answered.contains(&call.id.as_str()))
        .map(|call| call.id.clone())
        .collect();
    for id in open {
        conversation.add(ConversationEntry::tool(&id, "Cancelled by user."));
    }
}
