//! The ACP agent: answers the client's requests, and runs each session's
//! prompts one at a time in a task of its own.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification, ContentBlock,
    ContentChunk, Implementation, InitializeRequest, InitializeResponse, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ResumeSessionRequest,
    ResumeSessionResponse, SessionCapabilities, SessionId, SessionNotification,
    SessionResumeCapabilities, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Error, Responder, on_receive_notification,
    on_receive_request,
};
use flashmind_core::compaction::compact_manually;
use flashmind_core::subagent::Notifications;
use flashmind_core::{CancellationToken, ConversationEntry, EntryKind};
use flashmind_tools::ToolSync;
use flashmind_tools::background::Jobs;
use flashmind_types::{AgentEvent, AgentInput, ContentPart};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::background::{
    self, BackgroundResponse, BackgroundTaskRequest, Foreground, StopRequest, StopResponse, Tasks,
};
use super::open::{Opened, Opener};
use super::updates::{Ending, Updates};
use crate::interactive::prompt_input;
use crate::session::save_turn;

/// The most prompts waiting behind a session's running turn.
const MAX_QUEUED: usize = 8;

/// The prompt that compacts the conversation instead of starting a turn.
const COMPACT: &str = "/compact";

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
    /// The session's background jobs, to stop one on request.
    jobs: Option<Jobs>,
    /// The running foreground commands that can move to the background.
    foreground: Foreground,
}

/// The state the handlers share.
#[derive(Clone)]
pub(crate) struct Server {
    opener: Arc<dyn Opener>,
    sessions: Arc<Mutex<HashMap<SessionId, Handle>>>,
    /// The client declared the `asyncTasks` extension, so sessions report
    /// background commands to it.
    async_tasks: Arc<AtomicBool>,
}

impl Server {
    pub(crate) fn new(opener: impl Opener) -> Self {
        Self {
            opener: Arc::new(opener),
            sessions: Arc::default(),
            async_tasks: Arc::default(),
        }
    }

    /// The agent side of the connection, ready to connect to a client.
    pub(crate) fn agent(&self) -> impl ConnectTo<Client> + 'static {
        let sessions = self.clone();
        let resumes = self.clone();
        let prompts = self.clone();
        let cancels = self.clone();
        let initializes = self.clone();
        let stops = self.clone();
        let backgrounds = self.clone();
        Agent
            .builder()
            .name("flsh")
            .on_receive_request(
                async move |request: InitializeRequest, responder, _cx| {
                    tracing::info!(client = ?request.client_info, "ACP client connected");
                    let tasks = background::client_supports(&request.client_capabilities);
                    initializes.async_tasks.store(tasks, Ordering::Relaxed);
                    responder.respond(initialized(tasks))
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
                            Ok(id) => {
                                responder.respond(NewSessionResponse::new(id.clone()))?;
                                advertise_commands(&opening, id)
                            }
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
                            Ok(id) => {
                                responder.respond(ResumeSessionResponse::new())?;
                                advertise_commands(&opening, id)
                            }
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
            .on_receive_request(
                async move |request: StopRequest, responder, _cx| {
                    responder.respond(stops.stop(&request)?)
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: BackgroundTaskRequest, responder, _cx| {
                    responder.respond(backgrounds.background(&request))
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
        let foreground = Foreground::default();
        let jobs = opened.jobs.clone();
        let events = jobs
            .as_ref()
            .filter(|_| self.async_tasks.load(Ordering::Relaxed))
            .map(Jobs::subscribe);
        let task = Session {
            cx: cx.clone(),
            id: id.clone(),
            cwd: cwd.to_path_buf(),
            notifications: opened.subagents.clone().map(Notifications::new),
            pending: pending.clone(),
            tasks: Tasks::new(events),
            foreground: foreground.clone(),
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
                jobs,
                foreground,
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

    /// Stops a background job without touching the session's turn.
    fn stop(&self, request: &StopRequest) -> Result<StopResponse, Error> {
        if request.async_task_id.trim().is_empty() {
            return Err(Error::invalid_params().data("asyncTaskId must not be empty"));
        }
        let sessions = self.lock();
        let stopped = sessions
            .get(&request.session_id)
            .and_then(|handle| handle.jobs.as_ref())
            .zip(background::job_id(&request.async_task_id))
            .is_some_and(|(jobs, id)| jobs.stop(id));
        Ok(StopResponse { stopped })
    }

    /// Moves running foreground commands to the background.
    fn background(&self, request: &BackgroundTaskRequest) -> BackgroundResponse {
        let sessions = self.lock();
        let backgrounded = sessions.get(&request.session_id).is_some_and(|handle| {
            background::background(&handle.foreground, request.tool_call_id.as_deref())
        });
        BackgroundResponse { backgrounded }
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

/// The answer to `initialize`: protocol 1, sessions that resume, and
/// prompts that carry images. flsh reads and writes files and runs commands
/// itself, so it asks the client for none of that.
fn initialized(async_tasks: bool) -> InitializeResponse {
    let sessions = SessionCapabilities::new().resume(SessionResumeCapabilities::new());
    let capabilities = AgentCapabilities::new()
        .session_capabilities(sessions)
        .prompt_capabilities(PromptCapabilities::new().image(true));
    let response = InitializeResponse::new(ProtocolVersion::V1)
        .agent_capabilities(capabilities)
        .agent_info(Implementation::new("flsh", env!("CARGO_PKG_VERSION")));
    if async_tasks {
        response.meta(background::agent_meta())
    } else {
        response
    }
}

/// Tells the client about the commands a prompt can start with.
fn advertise_commands(cx: &ConnectionTo<Client>, session: SessionId) -> Result<(), Error> {
    let compact = AvailableCommand::new(
        COMPACT.trim_start_matches('/'),
        "Summarize the conversation so far to free up context",
    );
    let update =
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![compact]));
    cx.send_notification(SessionNotification::new(session, update))
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
    tasks: Tasks,
    foreground: Foreground,
}

impl Session {
    async fn run(mut self, mut queue: mpsc::Receiver<Prompt>) -> Result<(), Error> {
        loop {
            // A job that ends between turns is reported as it ends.
            let prompt = tokio::select! {
                prompt = queue.recv() => prompt,
                Some(event) = self.tasks.next() => {
                    self.tasks.publish(&self.cx, &self.id, event)?;
                    continue;
                }
            };
            let Some(prompt) = prompt else {
                break;
            };
            let answer = if prompt.cancel.is_cancelled() {
                Ok(StopReason::Cancelled)
            } else if prompt_text(&prompt.request.prompt).trim() == COMPACT {
                self.compact(&prompt.cancel).await
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
            meter,
            ..
        } = &mut self.opened;
        let tasks = &mut self.tasks;
        if let Some(sync) = tool_sync {
            sync.sync(agent.tools_mut()).await;
        }
        if let Some(notifications) = &mut self.notifications {
            notifications.deliver(conversation);
        }
        let mut input = prompt_input(&self.cwd, prompt_text(&request.prompt));
        if let AgentInput::User { parts, .. } = &mut input {
            *parts = prompt_images(&request.prompt);
        }
        loop {
            let mut updates = Updates::default();
            let mut ending = None;
            let size = agent.context_window();
            {
                let stream = agent.start(conversation, cancel.clone(), input, None);
                futures_util::pin_mut!(stream);
                loop {
                    let event = tokio::select! {
                        biased;
                        Some(job) = tasks.next() => {
                            tasks.publish(&self.cx, &self.id, job)?;
                            continue;
                        }
                        event = stream.next() => match event {
                            Some(event) => event,
                            None => break,
                        },
                    };
                    // A job is announced before the result of its tool call.
                    tasks.drain(&self.cx, &self.id)?;
                    track_foreground(&self.foreground, &event);
                    for mut update in updates.updates(&event) {
                        tasks.mark(&mut update);
                        self.cx
                            .send_notification(SessionNotification::new(self.id.clone(), update))?;
                    }
                    if meter.record(&event)
                        && let Some(update) = meter.update(size)
                    {
                        self.cx
                            .send_notification(SessionNotification::new(self.id.clone(), update))?;
                    }
                    ending = ending.or(Ending::of(&event));
                }
            }
            lock(&self.foreground).clear();
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

    /// Summarizes the conversation in place of a turn, for `/compact`.
    async fn compact(&mut self, cancel: &CancellationToken) -> Result<StopReason, Error> {
        let Opened {
            agent,
            conversation,
            ..
        } = &mut self.opened;
        let model = agent.llm().model.clone();
        let stream =
            compact_manually(conversation, agent.provider(), &model).take_until(cancel.cancelled());
        futures_util::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            let said = match event {
                AgentEvent::Compacted(_) => "Compacted the conversation.".to_owned(),
                AgentEvent::Status(status) => status,
                AgentEvent::Error(error) => return Err(Error::internal_error().data(error)),
                _ => continue,
            };
            let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(said)));
            self.cx.send_notification(SessionNotification::new(
                self.id.clone(),
                SessionUpdate::AgentMessageChunk(chunk),
            ))?;
        }
        if cancel.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }
        Ok(StopReason::EndTurn)
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
                .await?;
            let meter = &self.opened.meter;
            saved
                .store
                .update_session_usage(key, meter.total(), meter.last())
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

/// The images of a prompt, as attachments for the model, or `None` when it
/// has none.
fn prompt_images(blocks: &[ContentBlock]) -> Option<Vec<ContentPart>> {
    let images: Vec<ContentPart> = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Image(image) => Some(ContentPart::Image {
                media_type: image.mime_type.clone(),
                data: image.data.clone(),
            }),
            _ => None,
        })
        .collect();
    (!images.is_empty()).then_some(images)
}

/// Keeps `foreground` to the commands running now that can move to the
/// background.
fn track_foreground(foreground: &Foreground, event: &AgentEvent) {
    match event {
        AgentEvent::ToolBackgroundReady {
            id,
            request: Some(request),
        } => {
            lock(foreground).insert(id.clone(), request.clone());
        }
        AgentEvent::ToolResult { id, .. } => {
            lock(foreground).remove(id);
        }
        AgentEvent::Interrupted { tool_call_id, .. } => {
            lock(foreground).remove(tool_call_id);
        }
        _ => {}
    }
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
