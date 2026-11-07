//! The ACP agent: answers the client's requests, and runs each session's
//! prompts one at a time in a task of its own.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    ConfigOptionUpdate, ContentBlock, ContentChunk, Implementation, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ResumeSessionRequest,
    ResumeSessionResponse, SessionCapabilities, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigOptionValue, SessionConfigSelectOption, SessionId, SessionNotification,
    SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, StopReason, TextContent,
};
use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Error, Responder, on_receive_notification,
    on_receive_request,
};
use flashmind_core::compaction::compact_manually;
use flashmind_core::subagent::Notifications;
use flashmind_core::{CancellationToken, Conversation, ConversationEntry, EntryKind};
use flashmind_tools::ToolSync;
use flashmind_tools::background::Jobs;
use flashmind_types::{AgentEvent, AgentInput, ContentPart, ReasoningLevel};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::background::{
    self, BackgroundResponse, BackgroundTaskRequest, Foreground, StopRequest, StopResponse, Tasks,
};
use super::open::{Opened, Opener, Saved};
use super::system::SystemInstructions;
use super::updates::{Ending, Updates};
use crate::interactive::prompt_input;
use crate::presence::{Presence, Updates as PresenceUpdates};

type AsyncMutex<T> = tokio::sync::Mutex<T>;

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
    presence: Presence,
    settings: Arc<Settings>,
    history: Arc<Mutex<Vec<ConversationEntry>>>,
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
    pub(super) presence_file: Option<PathBuf>,
    opener: Arc<dyn Opener>,
    sessions: Arc<Mutex<HashMap<SessionId, Handle>>>,
    /// The client declared the `asyncTasks` extension, so sessions report
    /// background commands to it.
    async_tasks: Arc<AtomicBool>,
}

impl Server {
    pub(crate) fn new(opener: impl Opener) -> Self {
        Self {
            presence_file: None,
            opener: Arc::new(opener),
            sessions: Arc::default(),
            async_tasks: Arc::default(),
        }
    }

    /// The agent side of the connection, ready to connect to a client.
    pub(crate) fn agent(&self) -> impl ConnectTo<Client> + 'static {
        let sessions = self.clone();
        let resumes = self.clone();
        let loads = self.clone();
        let configs = self.clone();
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
                async move |request: NewSessionRequest, responder, cx| {
                    sessions.new_session(request, responder, cx)
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: ResumeSessionRequest, responder, cx| {
                    resumes.resume_session(request, responder, cx)
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: LoadSessionRequest, responder, cx| {
                    loads.load_session(request, responder, cx)
                },
                on_receive_request!(),
            )
            .on_receive_request(
                async move |request: SetSessionConfigOptionRequest, responder, cx| {
                    configs.configure(request, responder, cx)
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

    fn new_session(
        &self,
        request: NewSessionRequest,
        responder: Responder<NewSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), Error> {
        let server = self.clone();
        let opening = cx.clone();
        cx.spawn(async move {
            match server.open(&opening, &request.cwd, None).await {
                Ok(id) => {
                    responder.respond(
                        NewSessionResponse::new(id.clone())
                            .config_options(server.config_options(&id).await?),
                    )?;
                    advertise_commands(&opening, id)
                }
                Err(error) => responder.respond_with_error(error),
            }
        })
    }

    fn resume_session(
        &self,
        request: ResumeSessionRequest,
        responder: Responder<ResumeSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), Error> {
        let server = self.clone();
        let opening = cx.clone();
        cx.spawn(async move {
            match server
                .open(&opening, &request.cwd, Some(request.session_id.0.as_ref()))
                .await
            {
                Ok(id) => {
                    responder.respond(
                        ResumeSessionResponse::new()
                            .config_options(server.config_options(&id).await?),
                    )?;
                    advertise_commands(&opening, id)
                }
                Err(error) => responder.respond_with_error(error),
            }
        })
    }

    fn load_session(
        &self,
        request: LoadSessionRequest,
        responder: Responder<LoadSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), Error> {
        let server = self.clone();
        let opening = cx.clone();
        cx.spawn(async move {
            let loaded = async {
                let id = server
                    .open(&opening, &request.cwd, Some(request.session_id.0.as_ref()))
                    .await?;
                let history = {
                    let sessions = server.lock();
                    let handle = sessions
                        .get(&id)
                        .ok_or_else(|| Error::session_not_found(id.clone()))?;
                    lock(&handle.history).clone()
                };
                super::history::replay(&opening, &id, &history)?;
                let options = server.config_options(&id).await?;
                Ok::<_, Error>((id, options))
            }
            .await;
            match loaded {
                Ok((id, options)) => {
                    responder.respond(LoadSessionResponse::new().config_options(options))?;
                    advertise_commands(&opening, id)
                }
                Err(error) => responder.respond_with_error(error),
            }
        })
    }

    fn configure(
        &self,
        request: SetSessionConfigOptionRequest,
        responder: Responder<SetSessionConfigOptionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), Error> {
        let server = self.clone();
        cx.spawn(async move {
            match server.set_config(&request).await {
                Ok(options) => responder.respond(SetSessionConfigOptionResponse::new(options)),
                Err(error) => responder.respond_with_error(error),
            }
        })
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
        let settings = Arc::new(Settings {
            level: AsyncMutex::new(opened.agent.llm().reasoning),
            saved: opened.saved.clone(),
            key: opened.key.clone(),
        });
        settings.set(opened.agent.llm().reasoning).await?;
        let history = Arc::new(Mutex::new(opened.conversation.entries().to_vec()));
        let presence = Presence::with_locator(&opened.key, cwd, self.presence_file.clone());
        let task = Session {
            presence: presence.updates(),
            settings: settings.clone(),
            history: history.clone(),
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
                presence,
                settings,
                history,
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

    fn settings(&self, id: &SessionId) -> Result<Arc<Settings>, Error> {
        self.lock()
            .get(id)
            .map(|handle| handle.settings.clone())
            .ok_or_else(|| Error::session_not_found(id.clone()))
    }

    async fn config_options(&self, id: &SessionId) -> Result<Vec<SessionConfigOption>, Error> {
        Ok(thought_options(*self.settings(id)?.level.lock().await))
    }

    async fn set_config(
        &self,
        request: &SetSessionConfigOptionRequest,
    ) -> Result<Vec<SessionConfigOption>, Error> {
        if request.config_id.0.as_ref() != "thought_level" {
            return Err(Error::invalid_params().data("unknown configId"));
        }
        let SessionConfigOptionValue::ValueId { value } = &request.value else {
            return Err(Error::invalid_params().data("expected a thinking level"));
        };
        let level = parse_level(value.0.as_ref())?;
        self.settings(&request.session_id)?.set(level).await?;
        Ok(thought_options(level))
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
        let (prompt, error) = match handle.prompts.try_send(prompt) {
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
        let handles: Vec<Handle> = self.lock().drain().map(|(_, handle)| handle).collect();
        for handle in handles {
            handle.presence.shutdown().await;
            if let Some(sync) = handle.tool_sync {
                sync.shutdown().await;
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<SessionId, Handle>> {
        lock(&self.sessions)
    }
}

/// Locks `mutex`, also after a panic elsewhere: its data is a plain list
/// that no half-done change can leave inconsistent.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
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
        .load_session(true)
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
    let update = SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![
        compact,
        AvailableCommand::new("thinking", "Set reasoning level: off, low, medium, high"),
        AvailableCommand::new(
            "system",
            "View the system prompt or append instructions: /system [text]",
        ),
    ]));
    cx.send_notification(SessionNotification::new(session, update))
}

fn failed(what: &str, error: &anyhow::Error) -> Error {
    Error::internal_error().data(format!("{what}: {error:#}"))
}

/// One session's task: it owns the flsh session and runs its prompts.
struct Session {
    presence: PresenceUpdates,
    settings: Arc<Settings>,
    history: Arc<Mutex<Vec<ConversationEntry>>>,
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
            let text = prompt_text(&prompt.request.prompt);
            let mut words = text.split_whitespace();
            let command = words.next();
            let answer = if prompt.cancel.is_cancelled() {
                Ok(StopReason::Cancelled)
            } else if command == Some("/system") {
                self.system(
                    text.trim()
                        .strip_prefix("/system")
                        .unwrap_or_default()
                        .trim(),
                )
            } else if command == Some("/thinking") {
                match words.next().map(parse_level).transpose() {
                    Ok(level) if words.next().is_none() => {
                        if let Some(level) = level {
                            self.settings.set(level).await?;
                            self.cx.send_notification(SessionNotification::new(
                                self.id.clone(),
                                SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                                    thought_options(level),
                                )),
                            ))?;
                        }
                        Ok(StopReason::EndTurn)
                    }
                    _ => {
                        Err(Error::invalid_params().data("usage: /thinking [off|low|medium|high]"))
                    }
                }
            } else if text.trim() == COMPACT {
                self.compact(&prompt.cancel).await
            } else {
                self.turn(&prompt.request, &text, &prompt.cancel).await
            };
            *lock(&self.history) = self.opened.conversation.entries().to_vec();
            self.save().await;
            lock(&self.pending).pop_front();
            match answer {
                Ok(stop) => prompt.responder.respond(PromptResponse::new(stop))?,
                Err(error) => prompt.responder.respond_with_error(error)?,
            }
        }
        Ok(())
    }

    fn system(&mut self, instructions: &str) -> Result<StopReason, Error> {
        let said = if instructions.is_empty() {
            SystemInstructions::current(&self.opened.conversation)
        } else {
            self.opened
                .system_instructions
                .append(&mut self.opened.conversation, instructions);
            "Added system instructions."
        };
        self.cx.send_notification(SessionNotification::new(
            self.id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                TextContent::new(said),
            ))),
        ))?;
        Ok(StopReason::EndTurn)
    }

    /// Runs one prompt to the end of its turn, streaming its updates.
    async fn turn(
        &mut self,
        request: &PromptRequest,
        text: &str,
        cancel: &CancellationToken,
    ) -> Result<StopReason, Error> {
        self.opened.agent.llm_mut().reasoning = *self.settings.level.lock().await;
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
        let mut input = prompt_input(&self.cwd, text.to_owned());
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
            if let Some(cwd) = agent.working_dir() {
                self.presence.cwd(cwd);
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
            self.opened
                .system_instructions
                .save(&saved.store, key, &self.opened.conversation)
                .await?;
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
            ContentBlock::Text(text) => Some(text.text.as_str()),
            ContentBlock::ResourceLink(link) => Some(link.uri.as_str()),
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
        AgentEvent::ToolResult { id, .. }
        | AgentEvent::Interrupted {
            tool_call_id: id, ..
        } => {
            lock(foreground).remove(id);
        }
        _ => {}
    }
}

/// Answers the tool calls a cancelled turn left without a result, so the
/// next request to the model has a result for each call.
fn answer_open_tool_calls(conversation: &mut Conversation) {
    let entries = conversation.entries();
    let Some((last_call, calls)) =
        entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| match &entry.kind {
                EntryKind::Assistant {
                    tool_calls: Some(calls),
                    ..
                } if !calls.is_empty() => Some((index, calls)),
                _ => None,
            })
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

fn parse_level(value: &str) -> Result<ReasoningLevel, Error> {
    value
        .parse()
        .map_err(|_| Error::invalid_params().data("expected off, low, medium, or high"))
}

fn thought_options(level: ReasoningLevel) -> Vec<SessionConfigOption> {
    let options = [
        ("off", "Off"),
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
    ]
    .into_iter()
    .map(|(id, name)| SessionConfigSelectOption::new(id, name))
    .collect::<Vec<_>>();
    vec![
        SessionConfigOption::select(
            "thought_level",
            "Thinking level",
            level.to_string(),
            options,
        )
        .category(SessionConfigOptionCategory::ThoughtLevel),
    ]
}

/// Serializes durable setting changes so an older write cannot replace a newer selection.
struct Settings {
    level: AsyncMutex<ReasoningLevel>,
    saved: Option<Saved>,
    key: String,
}

impl Settings {
    async fn set(&self, level: ReasoningLevel) -> Result<(), Error> {
        let mut current = self.level.lock().await;
        if let Some(saved) = &self.saved {
            saved
                .store
                .update_reasoning(&self.key, level)
                .await
                .map_err(|error| failed("could not save the thinking level", &error))?;
        }
        *current = level;
        Ok(())
    }
}
