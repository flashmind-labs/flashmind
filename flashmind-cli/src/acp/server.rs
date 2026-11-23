//! The ACP agent: answers the client's requests, and runs each session's
//! prompts one at a time in a task of its own.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::v1::schema::Meta;
use agent_client_protocol::v1::schema::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    ConfigOptionUpdate, ContentBlock, ContentChunk, Error as SchemaError, Implementation,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse, McpServer,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities, SessionConfigOption,
    SessionConfigOptionValue, SessionId, SessionInfoUpdate, SessionNotification,
    SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, StopReason, TextContent, UsageUpdate,
};
use agent_client_protocol::v1::{ConnectionTo, Responder};
use agent_client_protocol::{
    Agent, Client, ConnectTo, Error, on_receive_notification, on_receive_request,
};
use flashmind_core::compaction::compact_manually;
use flashmind_core::subagent::Notifications;
use flashmind_core::{
    AgentRun, CancellationToken, Conversation, ConversationEntry, RunBoundary, ToolBatchEvent,
};
use flashmind_tools::ToolSync;
use flashmind_tools::background::Jobs;
use flashmind_tools::mcp::session::{SessionMcp, SessionServerConfig};
use flashmind_types::{AgentEvent, AgentInput, ContentPart};
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use super::background::{
    self, BackgroundResponse, BackgroundTaskRequest, Foreground, StopRequest, StopResponse, Tasks,
};
use super::open::{Opened, Opener, Saved};
use super::settings::{Settings, parse_level};
use super::steering::{self, SteeringRequest, SteeringResponse};
use super::system::SystemInstructions;
use super::updates::Updates;
use super::usage::Meter;
use crate::interactive::prompt_input;
use crate::presence::{Presence, Updates as PresenceUpdates};
use crate::session::{answer_open_tool_calls, record_tool};

/// The most prompts waiting behind a session's running turn.
const MAX_QUEUED: usize = 8;

/// The prompt that compacts the conversation instead of starting a turn.
const COMPACT: &str = "/compact";

/// A prompt waiting for its session's task.
struct Prompt {
    request: PromptRequest,
    owner: Owner,
    /// Cancelled by `session/cancel`, also while the prompt still waits.
    cancel: CancellationToken,
}

enum Owner {
    Prompt(Responder<PromptResponse>),
    /// Idle steering starts a turn owned by the session task. Its request
    /// answers at startup; completion and errors are sent as session updates.
    Steering(Option<Responder<SteeringResponse>>),
}

struct CancelTurnOnDrop(CancellationToken);

impl Drop for CancelTurnOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// What the handlers keep of an open session.
struct Handle {
    mcp: SessionMcp,
    mcp_servers: Vec<McpServer>,
    /// The model may start flsh subagents.
    subagents: bool,
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
    steering: Arc<Mutex<steering::State>>,
    /// The session's title, once it has one.
    title: Arc<Mutex<Option<String>>>,
    stop: CancellationToken,
    closed: oneshot::Receiver<()>,
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
        let steers = self.clone();
        Agent
            .v1()
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
                async move |request: SteeringRequest, responder, _cx| {
                    steers.steer(request, responder)
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
            match server
                .open(
                    &opening,
                    &request.cwd,
                    None,
                    &request.mcp_servers,
                    subagents(request.meta.as_ref()),
                )
                .await
            {
                Ok(id) => {
                    responder.respond(
                        NewSessionResponse::new(id.clone())
                            .config_options(server.config_options(&id).await?),
                    )?;
                    server.advertise(&opening, id)
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
                .open(
                    &opening,
                    &request.cwd,
                    Some(request.session_id.0.as_ref()),
                    &request.mcp_servers,
                    subagents(request.meta.as_ref()),
                )
                .await
            {
                Ok(id) => {
                    responder.respond(
                        ResumeSessionResponse::new()
                            .config_options(server.config_options(&id).await?),
                    )?;
                    server.advertise(&opening, id)
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
                    .open(
                        &opening,
                        &request.cwd,
                        Some(request.session_id.0.as_ref()),
                        &request.mcp_servers,
                        subagents(request.meta.as_ref()),
                    )
                    .await?;
                let history = {
                    let sessions = server.lock();
                    let handle = sessions
                        .get(&id)
                        .ok_or_else(|| Error::from(SchemaError::session_not_found(id.clone())))?;
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
                    server.advertise(&opening, id)
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
        mcp_servers: &[McpServer],
        subagents: bool,
    ) -> Result<SessionId, Error> {
        if !cwd.is_absolute() {
            return Err(Error::invalid_params().data(format!(
                "the session directory must be absolute: {}",
                cwd.display()
            )));
        }
        let servers = stdio_servers(mcp_servers)?;
        if let Some(key) = resume
            && let Some(handle) = self.lock().get(&SessionId::new(key))
        {
            if handle.mcp_servers != mcp_servers {
                return Err(
                    Error::invalid_params().data("cannot change MCP servers of an active session")
                );
            }
            if handle.subagents != subagents {
                return Err(
                    Error::invalid_params().data("cannot change subagents of an active session")
                );
            }
            return Ok(SessionId::new(key));
        }
        let mut opened = self
            .opener
            .open(cwd, resume, subagents)
            .await
            .map_err(|error| failed("could not open the session", &error))?;
        let mcp = SessionMcp::connect(&servers, cwd, None, opened.agent.tools_mut())
            .await
            .map_err(|error| failed("could not connect session MCP servers", &error))?;
        if let Some(sync) = &opened.tool_sync {
            sync.sync(opened.agent.tools_mut()).await;
        }
        if let Some(saved) = &opened.saved {
            saved
                .store
                .update_working_dir(&opened.key, cwd)
                .await
                .map_err(|error| failed("could not save the session directory", &error))?;
        }
        let id = SessionId::new(opened.key.as_str());
        let (prompts, queue) = mpsc::channel(MAX_QUEUED);
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        let foreground = Foreground::default();
        let steering = Arc::new(Mutex::new(steering::State::default()));
        let jobs = opened.jobs.clone();
        let events = jobs
            .as_ref()
            .filter(|_| self.async_tasks.load(Ordering::Relaxed))
            .map(Jobs::subscribe);
        let settings = Arc::new(Settings::new(&opened, self.opener.clone()).await);
        settings.set(opened.agent.llm().reasoning).await?;
        settings
            .save_model()
            .await
            .map_err(|error| failed("could not save the model", &error))?;
        let history = Arc::new(Mutex::new(opened.conversation.entries().to_vec()));
        let presence = Presence::with_locator(&opened.key, cwd, self.presence_file.clone());
        if let Some(saved) = &opened.title {
            presence.updates().title(&opened.key, saved);
        }
        let title = Arc::new(Mutex::new(opened.title.clone()));
        let stop = CancellationToken::new();
        let (closed, wait_for_close) = oneshot::channel();
        let task = Session {
            title: title.clone(),
            title_attempted: false,
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
            steering: steering.clone(),
            opened,
            _closed: closed,
        };
        let stop_task = stop.clone();
        let tool_sync = task.opened.tool_sync.clone();
        cx.spawn(async move {
            tokio::select! {
                biased;
                () = stop_task.cancelled() => Ok(()),
                result = task.run(queue) => result,
            }
        })?;
        self.lock().insert(
            id.clone(),
            Handle {
                mcp,
                mcp_servers: mcp_servers.to_vec(),
                subagents,
                presence,
                settings,
                history,
                prompts,
                pending,
                tool_sync,
                jobs,
                foreground,
                steering,
                title,
                stop,
                closed: wait_for_close,
            },
        );
        tracing::info!(session = %id, cwd = %cwd.display(), "ACP session opened");
        Ok(id)
    }

    fn settings(&self, id: &SessionId) -> Result<Arc<Settings>, Error> {
        self.lock()
            .get(id)
            .map(|handle| handle.settings.clone())
            .ok_or_else(|| Error::from(SchemaError::session_not_found(id.clone())))
    }

    async fn config_options(&self, id: &SessionId) -> Result<Vec<SessionConfigOption>, Error> {
        Ok(self.settings(id)?.options().await)
    }

    async fn set_config(
        &self,
        request: &SetSessionConfigOptionRequest,
    ) -> Result<Vec<SessionConfigOption>, Error> {
        let SessionConfigOptionValue::ValueId { value } = &request.value else {
            return Err(Error::invalid_params().data("expected a selection"));
        };
        let settings = self.settings(&request.session_id)?;
        match request.config_id.0.as_ref() {
            "thought_level" => settings.set(parse_level(value.0.as_ref())?).await?,
            "model" => settings.select_model(value.0.as_ref()).await?,
            _ => return Err(Error::invalid_params().data("unknown configId")),
        }
        Ok(settings.options().await)
    }

    /// Hands `request` to its session's task, which answers it.
    fn prompt(
        &self,
        request: PromptRequest,
        responder: Responder<PromptResponse>,
    ) -> Result<(), Error> {
        let sessions = self.lock();
        let Some(handle) = sessions.get(&request.session_id) else {
            return responder.respond_with_error(Error::from(SchemaError::session_not_found(
                request.session_id,
            )));
        };
        let cancel = CancellationToken::new();
        let prompt = Prompt {
            request,
            owner: Owner::Prompt(responder),
            cancel: cancel.clone(),
        };
        Self::enqueue(handle, prompt)
    }

    fn steer(
        &self,
        request: SteeringRequest,
        responder: Responder<SteeringResponse>,
    ) -> Result<(), Error> {
        let prompt_required = request.prompt_required()?;
        let sessions = self.lock();
        let handle = sessions.get(&request.session_id).ok_or_else(|| {
            Error::from(SchemaError::session_not_found(request.session_id.clone()))
        })?;
        let bytes = request.validate(handle.settings.images())?;
        let pending = steering::Pending {
            prompt: request.prompt,
            responder,
            prompt_required,
            bytes,
        };
        let Some(pending) = lock(&handle.steering).enqueue(pending)? else {
            return Ok(());
        };
        if prompt_required {
            return pending.responder.respond(SteeringResponse::idle());
        }
        Self::enqueue(
            handle,
            Prompt {
                request: PromptRequest::new(request.session_id, pending.prompt),
                owner: Owner::Steering(Some(pending.responder)),
                cancel: CancellationToken::new(),
            },
        )
    }

    fn enqueue(handle: &Handle, prompt: Prompt) -> Result<(), Error> {
        // Pushed first, so the task always finds it: the task takes tokens
        // from the front, in the order the prompts were queued.
        lock(&handle.pending).push_back(prompt.cancel.clone());
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
        match prompt.owner {
            Owner::Prompt(responder) => responder.respond_with_error(error),
            Owner::Steering(Some(responder)) => responder.respond_with_error(error),
            Owner::Steering(None) => Ok(()),
        }
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
        for handle in &handles {
            handle.stop.cancel();
        }
        for mut handle in handles {
            let _ = handle.closed.await;
            handle.mcp.shutdown().await;
            handle.presence.shutdown().await;
            if let Some(sync) = handle.tool_sync {
                sync.shutdown().await;
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<SessionId, Handle>> {
        lock(&self.sessions)
    }

    /// Tells the client an opened session's title, if it has one, and its
    /// commands.
    fn advertise(&self, cx: &ConnectionTo<Client>, id: SessionId) -> Result<(), Error> {
        let title = self
            .lock()
            .get(&id)
            .and_then(|handle| lock(&handle.title).clone());
        if let Some(title) = title {
            send_title(cx, id.clone(), title)?;
        }
        advertise_commands(cx, id)
    }
}

/// Whether the session's model may start flsh subagents. A client turns
/// them off with `_meta: {"flsh": {"subagents": false}}`; any other value
/// keeps them.
fn subagents(meta: Option<&Meta>) -> bool {
    meta.and_then(|meta| meta.get("flsh"))
        .and_then(|flsh| flsh.get("subagents"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn send_title(cx: &ConnectionTo<Client>, id: SessionId, title: String) -> Result<(), Error> {
    let update = SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title));
    cx.send_notification(SessionNotification::new(id, update))
}

fn stdio_servers(servers: &[McpServer]) -> Result<Vec<SessionServerConfig>, Error> {
    servers
        .iter()
        .map(|server| match server {
            McpServer::Stdio(server) => Ok(SessionServerConfig {
                name: server.name.clone(),
                command: Some(server.command.clone()),
                args: server.args.clone(),
                env: server
                    .env
                    .iter()
                    .map(|entry| (entry.name.clone(), entry.value.clone()))
                    .collect(),
                ..Default::default()
            }),
            _ => Err(Error::invalid_params().data("flsh supports only stdio MCP servers")),
        })
        .collect()
}

/// Locks `mutex`, also after a panic elsewhere: its data is a plain list
/// that no half-done change can leave inconsistent.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
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
    let mut meta = if async_tasks {
        background::agent_meta()
    } else {
        Default::default()
    };
    meta.insert(
        "steering".to_owned(),
        serde_json::json!({"supported": true}),
    );
    response.meta(meta)
}

/// Tells the client about the commands a prompt can start with.
fn advertise_commands(cx: &ConnectionTo<Client>, session: SessionId) -> Result<(), Error> {
    let compact = AvailableCommand::new(
        COMPACT.trim_start_matches('/'),
        "Summarize the conversation so far to free up context",
    );
    let update = SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![
        compact,
        AvailableCommand::new(
            "thinking",
            "Set reasoning level: off, minimal, low, medium, high, xhigh or max",
        ),
        AvailableCommand::new(
            "system",
            "View the system prompt or append instructions: /system [text]",
        ),
    ]));
    cx.send_notification(SessionNotification::new(session, update))
}

pub(super) fn failed(what: &str, error: &anyhow::Error) -> Error {
    Error::internal_error().data(format!("{what}: {error:#}"))
}

/// One session's task: it owns the flsh session and runs its prompts.
struct Session {
    /// Shared with the handle, which tells clients that open the session.
    title: Arc<Mutex<Option<String>>>,
    /// The title was asked for once; a model that cannot title is not asked again.
    title_attempted: bool,
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
    steering: Arc<Mutex<steering::State>>,
    /// Dropped after session state and its writer reservation.
    _closed: oneshot::Sender<()>,
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
            let Some(mut prompt) = prompt else {
                break;
            };
            let text = prompt_text(&prompt.request.prompt);
            let mut words = text.split_whitespace();
            let command = words.next();
            let command_prompt =
                matches!(command, Some("/system" | "/thinking")) || text.trim() == COMPACT;
            let steering = matches!(prompt.owner, Owner::Steering(_));
            let answer = if prompt.cancel.is_cancelled() && (command_prompt || steering) {
                Ok(StopReason::Cancelled)
            } else if steering {
                self.turn(&prompt.request, &text, &prompt.cancel, &mut prompt.owner)
                    .await
            } else if command == Some("/system") {
                self.system(
                    text.trim()
                        .strip_prefix("/system")
                        .unwrap_or_default()
                        .trim(),
                )
                .await
            } else if command == Some("/thinking") {
                match words.next().map(parse_level).transpose() {
                    Ok(level) if words.next().is_none() => {
                        if let Some(level) = level {
                            self.settings.set(level).await?;
                            self.cx.send_notification(SessionNotification::new(
                                self.id.clone(),
                                SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                                    self.settings.options().await,
                                )),
                            ))?;
                        }
                        Ok(StopReason::EndTurn)
                    }
                    _ => Err(Error::invalid_params()
                        .data("usage: /thinking [off|minimal|low|medium|high|xhigh|max]")),
                }
            } else if text.trim() == COMPACT {
                self.compact(&prompt.cancel).await
            } else {
                self.turn(&prompt.request, &text, &prompt.cancel, &mut prompt.owner)
                    .await
            };
            lock(&self.pending).pop_front();
            match prompt.owner {
                Owner::Prompt(responder) => match answer {
                    Ok(stop) => responder.respond(PromptResponse::new(stop))?,
                    Err(error) => responder.respond_with_error(error)?,
                },
                Owner::Steering(Some(responder)) => {
                    responder.respond_with_error(answer.err().unwrap_or_else(|| {
                        Error::invalid_request().data("the turn was cancelled before startup")
                    }))?
                }
                Owner::Steering(None) => {
                    if let Err(error) = answer {
                        self.cx.send_notification(SessionNotification::new(
                            self.id.clone(),
                            SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                ContentBlock::Text(TextContent::new(format!(
                                    "Turn failed: {error}"
                                ))),
                            )),
                        ))?;
                    }
                }
            }
        }
        Ok(())
    }

    async fn system(&mut self, instructions: &str) -> Result<StopReason, Error> {
        let said = if instructions.is_empty() {
            SystemInstructions::current(&self.opened.conversation)
        } else {
            self.opened
                .system_instructions
                .append(&mut self.opened.conversation, instructions);
            self.checkpoint().save(&self.opened.conversation).await?;
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

    /// Drives a single run through safe boundaries, keeping ownership of every
    /// model call, tool, steer, and usage update until this prompt completes.
    async fn turn(
        &mut self,
        request: &PromptRequest,
        text: &str,
        cancel: &CancellationToken,
        owner: &mut Owner,
    ) -> Result<StopReason, Error> {
        let cancel = cancel.child_token();
        let _cancel_guard = CancelTurnOnDrop(cancel.clone());
        let cancel = &cancel;
        self.settings.apply(&mut self.opened).await;
        let conversation = &mut self.opened.conversation;
        let steering = matches!(owner, Owner::Steering(_));
        if let Some(notifications) = &mut self.notifications {
            notifications.deliver(conversation);
        }
        let entries_before = conversation.entries().len();
        let input = if steering {
            conversation.add(steering::entry(&request.prompt));
            AgentInput::Resume
        } else {
            let mut input = prompt_input(&self.cwd, text.to_owned());
            if let AgentInput::User { parts, .. } = &mut input {
                *parts = prompt_images(&request.prompt);
            }
            input
        };
        let mut run = AgentRun::new(conversation, input, None);
        if let Err(error) = self.checkpoint().save(&self.opened.conversation).await {
            let conversation = &mut self.opened.conversation;
            conversation.entries_mut().truncate(entries_before);
            run.finish(conversation);
            return Err(error);
        }
        if !steering && !cancel.is_cancelled() {
            self.start_title(text)?;
        }
        let Opened {
            agent,
            conversation,
            tool_sync,
            meter,
            key,
            saved,
            system_instructions,
            ..
        } = &mut self.opened;
        let checkpoint = Transcript {
            key,
            saved: saved.as_ref(),
            system_instructions,
            history: &self.history,
        };
        if let Some(sync) = tool_sync
            && !cancel.is_cancelled()
        {
            sync.sync(agent.tools_mut()).await;
        }
        let cost_before = meter.total().clone();
        let turn = steering::Turn::begin(self.steering.clone(), cancel.clone());
        let mut output = TurnUpdates {
            cx: &self.cx,
            id: &self.id,
            tasks: &mut self.tasks,
            foreground: &self.foreground,
            meter,
            size: agent.context_window(),
            updates: Updates::default(),
        };
        let result = async {
            if cancel.is_cancelled() {
                return Ok(StopReason::Cancelled);
            }
            if let Owner::Steering(responder) = owner
                && let Some(responder) = responder.take()
            {
                responder.respond(SteeringResponse::StartedNewTurn)?;
            }
            loop {
                let (boundary, streamed) = {
                    let mut step = run.advance(agent, conversation, cancel);
                    let streamed = output.stream(&mut step).await;
                    let boundary = step.take_result().unwrap_or_else(|| {
                        Err(anyhow::anyhow!("Agent step ended without a result"))
                    });
                    (boundary, streamed)
                };
                if matches!(boundary, Ok(RunBoundary::Checkpoint(_))) {
                    checkpoint.save(conversation).await?;
                }
                streamed?;
                if cancel.is_cancelled() {
                    return Ok(StopReason::Cancelled);
                }
                match boundary {
                    Ok(RunBoundary::BeforeModel) => {
                        if let Some(notifications) = &mut self.notifications
                            && notifications.deliver(conversation)
                        {
                            checkpoint.save(conversation).await?;
                        }
                        checkpoint.deliver(&turn, conversation, false).await?;
                    }
                    Ok(RunBoundary::Checkpoint(_)) => {}
                    Ok(RunBoundary::Tools(calls)) => {
                        let mut interrupted = false;
                        for batch in agent.tool_batches(&calls) {
                            let mut execution = agent.stream_tool_batch(&batch, cancel);
                            while let Some(event) = output.next(&mut execution).await? {
                                match event {
                                    ToolBatchEvent::Event(event) => output.publish(&event)?,
                                    ToolBatchEvent::Completed { tool, .. } => {
                                        interrupted |= tool.result.is_interrupt();
                                        let events = record_tool(tool, conversation);
                                        checkpoint.save(conversation).await?;
                                        for event in events {
                                            output.publish(&event)?;
                                        }
                                    }
                                }
                            }
                            if interrupted || cancel.is_cancelled() {
                                break;
                            }
                        }
                        if cancel.is_cancelled() {
                            return Ok(StopReason::Cancelled);
                        }
                        if interrupted {
                            answer_open_tool_calls(conversation, "Tool execution was interrupted.");
                            return Ok(StopReason::EndTurn);
                        }
                    }
                    Ok(RunBoundary::Settling) => {
                        let delivered = self
                            .notifications
                            .as_mut()
                            .is_some_and(|notifications| notifications.deliver(conversation));
                        if delivered {
                            checkpoint.save(conversation).await?;
                        }
                        // Keep acceptance open when delegated results require another
                        // model request. Otherwise delivery and closure share a lock.
                        if checkpoint.deliver(&turn, conversation, !delivered).await? || delivered {
                            run.continue_turn();
                        }
                    }
                    Ok(RunBoundary::Done(_)) => return Ok(StopReason::EndTurn),
                    Err(error) => return Err(Error::internal_error().data(error.to_string())),
                }
            }
        }
        .await;
        run.finish(conversation);
        drop(turn);
        if let Some(cwd) = agent.working_dir() {
            self.presence.cwd(cwd);
        }
        lock(&self.foreground).clear();
        let result = if cancel.is_cancelled() {
            answer_open_tool_calls(conversation, "Cancelled by user.");
            conversation.add(ConversationEntry::system_message(
                "The user interrupted the previous turn.",
            ));
            result.map(|_| StopReason::Cancelled)
        } else {
            if result.is_err() {
                answer_open_tool_calls(conversation, "The turn ended before this tool completed.");
            }
            result
        };
        checkpoint.save(conversation).await?;
        if let Some(saved) = saved {
            saved
                .store
                .update_session_usage_by_model(key, meter.total(), meter.last(), &meter.unsaved)
                .await
                .map_err(|error| failed("could not save session usage", &error))?;
            meter.unsaved.clear();
        }
        if matches!(result, Ok(StopReason::EndTurn))
            && let Some(balance) = crate::credits::fetch(agent.provider()).await
        {
            let reported_tokens = meter.total().input_tokens > cost_before.input_tokens
                || meter.total().output_tokens > cost_before.output_tokens;
            let consumed =
                (reported_tokens && !cost_before.cost_unknown && !meter.total().cost_unknown)
                    .then(|| meter.total().total_cost.checked_sub(cost_before.total_cost))
                    .flatten();
            let (mut update, context_available) = match meter.update(agent.context_window()) {
                Some(SessionUpdate::UsageUpdate(update)) => (update, true),
                _ => (
                    UsageUpdate::new(0, u64::from(agent.context_window())),
                    false,
                ),
            };
            update.meta.get_or_insert_default().extend([
                (
                    "flsh/credits".to_owned(),
                    serde_json::json!({
                        "provider": agent.provider().name(), "balance": balance,
                        "consumed": consumed.map(|amount| amount.normalize().to_string()),
                    }),
                ),
                (
                    "flsh/contextUsageAvailable".to_owned(),
                    context_available.into(),
                ),
            ]);
            self.cx.send_notification(SessionNotification::new(
                self.id.clone(),
                SessionUpdate::UsageUpdate(update),
            ))?;
        }
        result
    }

    /// Writes the session's title from its first prompt while the turn runs.
    fn start_title(&mut self, prompt: &str) -> Result<(), Error> {
        if self.title_attempted || lock(&self.title).is_some() || prompt.trim().is_empty() {
            return Ok(());
        }
        let Some(titler) = self.opened.titles.titler(&self.opened.agent) else {
            return Ok(());
        };
        self.title_attempted = true;
        let prompt = prompt.to_owned();
        let key = self.opened.key.clone();
        let store = self.opened.saved.as_ref().map(|saved| saved.store.clone());
        let (presence, shared, cx, id) = (
            self.presence.clone(),
            self.title.clone(),
            self.cx.clone(),
            self.id.clone(),
        );
        self.cx.spawn(async move {
            let to = crate::session_title::Destination {
                key: &key,
                store: store.as_ref(),
                presence: &presence,
            };
            if let Some(title) = crate::session_title::write(&titler, &prompt, to).await {
                *lock(&shared) = Some(title.clone());
                send_title(&cx, id, title)?;
            }
            Ok(())
        })
    }

    /// Summarizes the conversation in place of a turn, for `/compact`.
    async fn compact(&mut self, cancel: &CancellationToken) -> Result<StopReason, Error> {
        let result = self.compact_stream(cancel).await;
        self.checkpoint().save(&self.opened.conversation).await?;
        result
    }

    async fn compact_stream(&mut self, cancel: &CancellationToken) -> Result<StopReason, Error> {
        let Opened {
            agent,
            conversation,
            ..
        } = &mut self.opened;
        let model = agent.llm().model.clone();
        conversation.set_compaction_context(agent.llm(), agent.tools().definitions());
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

    fn checkpoint(&self) -> Transcript<'_> {
        Transcript {
            key: &self.opened.key,
            saved: self.opened.saved.as_ref(),
            system_instructions: &self.opened.system_instructions,
            history: &self.history,
        }
    }
}

struct Transcript<'a> {
    key: &'a str,
    saved: Option<&'a Saved>,
    system_instructions: &'a SystemInstructions,
    history: &'a Mutex<Vec<ConversationEntry>>,
}

impl Transcript<'_> {
    async fn save(&self, conversation: &Conversation) -> Result<(), Error> {
        if let Some(saved) = self.saved {
            self.system_instructions
                .save(saved, self.key, conversation)
                .await
                .map_err(|error| failed("could not save the transcript", &error))?;
        }
        *lock(self.history) = conversation.entries().to_vec();
        Ok(())
    }

    async fn deliver(
        &self,
        turn: &steering::Turn,
        conversation: &mut Conversation,
        settling: bool,
    ) -> Result<bool, Error> {
        let pending = turn.take_pending(settling);
        if pending.is_empty() {
            return Ok(false);
        }
        let entries_before = conversation.entries().len();
        for input in &pending {
            conversation.add(steering::entry(&input.prompt));
        }
        if let Err(error) = self.save(conversation).await {
            conversation.entries_mut().truncate(entries_before);
            for input in pending {
                let _ = input.responder.respond_with_error(error.clone());
            }
            return Err(error);
        }
        for input in pending {
            input.responder.respond(SteeringResponse::Injected)?;
        }
        Ok(true)
    }
}

/// One prompt's update translator and accounting, shared by its model and tool streams.
struct TurnUpdates<'a> {
    cx: &'a ConnectionTo<Client>,
    id: &'a SessionId,
    tasks: &'a mut Tasks,
    foreground: &'a Foreground,
    meter: &'a mut Meter,
    size: u32,
    updates: Updates,
}

impl TurnUpdates<'_> {
    async fn next<T>(
        &mut self,
        stream: &mut (impl Stream<Item = T> + Unpin),
    ) -> Result<Option<T>, Error> {
        loop {
            tokio::select! {
                biased;
                Some(job) = self.tasks.next() => self.tasks.publish(self.cx, self.id, job)?,
                event = stream.next() => return Ok(event),
            }
        }
    }

    async fn stream(&mut self, stream: impl Stream<Item = AgentEvent>) -> Result<(), Error> {
        futures_util::pin_mut!(stream);
        let mut failure = None;
        while let Some(event) = self.next(&mut stream).await? {
            if let AgentEvent::Error(error) = &event {
                failure.get_or_insert_with(|| error.clone());
            }
            self.publish(&event)?;
        }
        // Finish the step before reporting errors so compaction releases history.
        match failure {
            Some(error) => Err(Error::internal_error().data(error)),
            None => Ok(()),
        }
    }

    fn publish(&mut self, event: &AgentEvent) -> Result<(), Error> {
        self.tasks.drain(self.cx, self.id)?;
        track_foreground(self.foreground, event);
        for mut update in self.updates.updates(event) {
            self.tasks.mark(&mut update);
            self.cx
                .send_notification(SessionNotification::new(self.id.clone(), update))?;
        }
        if self.meter.record(event)
            && let Some(update) = self.meter.update(self.size)
        {
            self.cx
                .send_notification(SessionNotification::new(self.id.clone(), update))?;
        }
        Ok(())
    }
}

/// The text of a prompt: its text blocks, and the URI of each resource
/// it links.
pub(super) fn prompt_text(blocks: &[ContentBlock]) -> String {
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
