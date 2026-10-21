//! Spawning, retained conversations, and control of child agents.

use std::sync::{Arc, Mutex};

use futures::{FutureExt, StreamExt};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use flashmind_types::{AgentEvent, AgentInput, AgentLlmConfig};

use crate::agent::Agent;
use crate::conversation::{Conversation, ConversationEntry};

use super::builder::SpawnBuilder;
use super::handle::{AgentHandle, AgentStatus, SessionState, SubagentEvent};

/// Orchestrates spawned agents with concurrency and depth limits.
pub struct AgentManager {
    handles: Mutex<Vec<Arc<AgentHandle>>>,
    max_concurrent: usize,
    max_depth: usize,
    current_depth: usize,
    event_sender: Option<mpsc::UnboundedSender<AgentEvent>>,
}

impl AgentManager {
    pub fn new(max_concurrent: usize, max_depth: usize) -> Self {
        Self {
            handles: Mutex::new(Vec::new()),
            max_concurrent,
            max_depth,
            current_depth: 0,
            event_sender: None,
        }
    }

    pub fn with_event_sender(mut self, sender: mpsc::UnboundedSender<AgentEvent>) -> Self {
        self.event_sender = Some(sender);
        self
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.current_depth = depth;
        self
    }

    /// All conversations in spawn order, including completed agents.
    pub fn conversations(&self) -> Vec<Arc<AgentHandle>> {
        self.handles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub async fn active_count(&self) -> usize {
        self.conversations()
            .iter()
            .filter(|h| !h.is_finished())
            .count()
    }

    pub async fn spawn(&self, builder: SpawnBuilder) -> anyhow::Result<Uuid> {
        if self.current_depth >= self.max_depth {
            anyhow::bail!(
                "maximum agent depth reached ({}/{})",
                self.current_depth,
                self.max_depth
            );
        }

        let mut tools = builder.tools.unwrap_or_default();
        let prefixes: Vec<&str> = builder.strip_prefixes.iter().map(String::as_str).collect();
        tools.strip_prefixes(&prefixes);
        let mut llm = builder.llm;
        if let Some(model) = builder.model {
            let provider = builder.provider.provider();
            if model.provider != provider {
                anyhow::bail!("subagent model must use the current provider ({provider})");
            }
            if let Some(config) = llm.as_mut() {
                config.model = model;
            } else {
                llm = Some(AgentLlmConfig::new(model));
            }
        }
        let mut agent_builder = Agent::builder(builder.provider).tools(tools);
        if let Some(llm) = llm {
            agent_builder = agent_builder.llm(llm);
        }
        let agent = agent_builder.build().await;
        let mut conversation = Conversation::new();
        let mut system = builder.system_prompt.unwrap_or_else(|| {
            String::from(
                "You are a background agent running autonomously. Use your tools to complete the task, then \
                 respond with a concise summary. Work independently with the information \
                 available. The user may open your conversation and send follow-up messages.",
            )
        });
        if !builder.instructions.is_empty() {
            system.push_str("\n\n");
            system.push_str(&builder.instructions);
        }
        conversation.set_system(&system);

        let id = Uuid::new_v4();
        let (sender, receiver) = mpsc::unbounded_channel();
        let state = Arc::new(Mutex::new(SessionState {
            events: Vec::new(),
            pending: 0,
            cancel: CancellationToken::new(),
        }));
        let status = Arc::new(Mutex::new(AgentStatus::Running { turn: 0 }));
        let changed = Arc::new(Notify::new());
        let handle = Arc::new(AgentHandle {
            id,
            name: builder.name,
            task: builder.task,
            sender,
            state: state.clone(),
            status: status.clone(),
            changed: changed.clone(),
        });
        {
            let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
            let active = handles.iter().filter(|h| !h.is_finished()).count();
            if active >= self.max_concurrent {
                anyhow::bail!(
                    "maximum concurrent agents reached ({}/{})",
                    active,
                    self.max_concurrent
                );
            }
            if let Some(name) = &handle.name
                && handles
                    .iter()
                    .any(|h| h.name.as_ref() == Some(name) && !h.is_finished())
            {
                anyhow::bail!("agent name already in use: {name}");
            }
            handle.send(ConversationEntry::user(&handle.task))?;
            handles.push(handle.clone());
        }

        let ctx = SpawnContext {
            id,
            task: handle.task.clone(),
            event_sender: self.event_sender.clone(),
            agent,
            conversation,
            max_iterations: builder.max_iterations,
            state,
            status: status.clone(),
            changed: changed.clone(),
        };
        tokio::spawn(async move {
            if std::panic::AssertUnwindSafe(run_agent(ctx, receiver))
                .catch_unwind()
                .await
                .is_err()
            {
                *status.lock().unwrap_or_else(|e| e.into_inner()) = AgentStatus::Failed {
                    error: "agent task panicked".into(),
                };
                changed.notify_waiters();
            }
        });
        Ok(id)
    }

    /// Resolve the most recently spawned agent with a matching name or ID.
    pub async fn resolve(&self, agent: &str) -> anyhow::Result<Uuid> {
        let handles = self.conversations();
        if let Some(handle) = handles
            .iter()
            .rev()
            .find(|h| h.name.as_deref() == Some(agent) && !h.is_finished())
        {
            return Ok(handle.id);
        }
        if let Some(handle) = handles
            .iter()
            .rev()
            .find(|h| h.name.as_deref() == Some(agent))
        {
            return Ok(handle.id);
        }
        if let Ok(id) = Uuid::parse_str(agent)
            && handles.iter().any(|h| h.id == id)
        {
            return Ok(id);
        }
        if !agent.is_empty() && agent.len() <= 32 && agent.chars().all(|c| c.is_ascii_hexdigit()) {
            let mut matches = handles
                .iter()
                .filter(|h| h.id.simple().to_string().starts_with(agent));
            if let Some(handle) = matches.next() {
                if matches.next().is_some() {
                    anyhow::bail!("ambiguous agent ID prefix: {agent}");
                }
                return Ok(handle.id);
            }
        }
        anyhow::bail!("agent not found: {agent}")
    }

    pub fn conversation(&self, id: Uuid) -> anyhow::Result<Arc<AgentHandle>> {
        self.conversations()
            .into_iter()
            .find(|h| h.id == id)
            .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))
    }

    /// Queue a user message using the subagent's existing runtime and context.
    pub fn follow_up(&self, id: Uuid, entry: ConversationEntry) -> anyhow::Result<()> {
        if !entry.is_user() {
            anyhow::bail!("a follow-up must be a user message");
        }
        let handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        let handle = handles
            .iter()
            .find(|h| h.id == id)
            .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))?;
        if let Some(name) = &handle.name
            && handles.iter().any(|other| {
                other.id != id && other.name.as_ref() == Some(name) && !other.is_finished()
            })
        {
            anyhow::bail!("agent name already in use: {name}");
        }
        if handle.is_finished()
            && handles.iter().filter(|h| !h.is_finished()).count() >= self.max_concurrent
        {
            anyhow::bail!(
                "maximum concurrent agents reached ({})",
                self.max_concurrent
            );
        }
        handle.send(entry)
    }

    pub async fn status(&self, id: Uuid) -> anyhow::Result<AgentStatus> {
        Ok(self.conversation(id)?.status())
    }

    pub async fn all_statuses(&self) -> Vec<(Uuid, Option<String>, String, AgentStatus)> {
        self.conversations()
            .iter()
            .map(|h| (h.id, h.name.clone(), h.task.clone(), h.status()))
            .collect()
    }

    pub async fn terminate(&self, id: Uuid) -> anyhow::Result<()> {
        self.conversation(id)?.cancel();
        Ok(())
    }

    /// Wait without removing the conversation, including if the wait is cancelled.
    pub async fn wait(&self, id: Uuid) -> anyhow::Result<String> {
        self.conversation(id)?.wait().await
    }

    pub async fn cancel_all(&self) {
        for handle in self.conversations() {
            handle.cancel();
        }
    }
}

struct SpawnContext {
    id: Uuid,
    task: String,
    event_sender: Option<mpsc::UnboundedSender<AgentEvent>>,
    agent: Agent,
    conversation: Conversation,
    max_iterations: Option<usize>,
    state: Arc<Mutex<SessionState>>,
    status: Arc<Mutex<AgentStatus>>,
    changed: Arc<Notify>,
}

async fn run_agent(
    mut ctx: SpawnContext,
    mut messages: mpsc::UnboundedReceiver<ConversationEntry>,
) {
    while let Some(entry) = messages.recv().await {
        ctx.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .events
            .push(SubagentEvent::User(entry.clone()));
        ctx.conversation.add(entry);
        let cancel = ctx
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancel
            .clone();
        let model = ctx.agent.llm().model.clone();
        let mut outcome = AgentStatus::Completed {
            result: String::new(),
        };
        let mut turn_count = 0;
        {
            let stream = ctx.agent.start(
                &mut ctx.conversation,
                cancel.clone(),
                AgentInput::Resume,
                ctx.max_iterations,
            );
            tokio::pin!(stream);
            while let Some(event) = stream.next().await {
                match &event {
                    AgentEvent::ToolResult { .. } => {
                        turn_count += 1;
                        *ctx.status.lock().unwrap_or_else(|e| e.into_inner()) =
                            AgentStatus::Running { turn: turn_count };
                    }
                    AgentEvent::Done(text) => {
                        outcome = AgentStatus::Completed {
                            result: text.clone(),
                        }
                    }
                    AgentEvent::Error(error) => {
                        outcome = AgentStatus::Failed {
                            error: error.clone(),
                        }
                    }
                    AgentEvent::Cancelled => outcome = AgentStatus::Cancelled,
                    _ => {}
                }
                ctx.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .events
                    .push(SubagentEvent::Agent(event.clone()));
                if let Some(sender) = &ctx.event_sender {
                    let _ = sender.send(AgentEvent::SpawnedEvent {
                        id: ctx.id.to_string(),
                        task: ctx.task.clone(),
                        model: Some(model.clone()),
                        event: Box::new(event),
                    });
                }
            }
        }
        {
            let mut state = ctx.state.lock().unwrap_or_else(|e| e.into_inner());
            state.pending -= 1;
            if cancel.is_cancelled() {
                outcome = AgentStatus::Cancelled;
            }
            state.cancel = CancellationToken::new();
            *ctx.status.lock().unwrap_or_else(|e| e.into_inner()) = if state.pending == 0 {
                outcome
            } else {
                AgentStatus::Running { turn: 0 }
            };
        }
        ctx.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use flashmind_types::{
        AgentLlmConfig, CompletionRequest, CompletionStream, FinishReason, LlmProvider, Model,
        ModelCapabilities, Provider, Role, StreamEvent,
    };

    use super::*;

    #[derive(Default)]
    struct RecordingProvider {
        requests: Mutex<Vec<CompletionRequest>>,
        gate: Option<Arc<Notify>>,
    }

    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        async fn capabilities(&self, _: &Model) -> ModelCapabilities {
            ModelCapabilities {
                reasoning: true,
                ..Default::default()
            }
        }

        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            let text = request
                .messages
                .iter()
                .rev()
                .find(|message| message.role == Role::User)
                .expect("a response has a user message")
                .content
                .clone();
            let mut requests = self.requests.lock().unwrap();
            let gate = if requests.is_empty() {
                self.gate.clone()
            } else {
                None
            };
            requests.push(request);
            Box::pin(async_stream::try_stream! {
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                yield StreamEvent::ContentDelta(format!("Reply: {text}"));
                yield StreamEvent::Finished(FinishReason::Stop);
            })
        }
    }

    #[tokio::test]
    async fn model_override_changes_requests_and_preserves_inherited_settings() {
        use flashmind_types::{ProviderPreferences, ReasoningLevel, SessionId};

        let manager = AgentManager::new(1, 1);
        let provider = Arc::new(RecordingProvider::default());
        let llm = AgentLlmConfig::new("ollama:parent".parse().unwrap())
            .with_session_id(SessionId::new())
            .with_provider_preferences(ProviderPreferences {
                zdr: Some(true),
                ..Default::default()
            });
        let mut llm = llm;
        llm.reasoning = ReasoningLevel::High;
        for inherited in [None, Some(llm.clone())] {
            let mut builder = SpawnBuilder::new("Inspect the code.", provider.clone())
                .model("ollama:worker".parse().unwrap());
            if let Some(config) = inherited {
                builder = builder.llm(config);
            }
            let id = manager.spawn(builder).await.unwrap();
            manager.wait(id).await.unwrap();
        }
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert_eq!(request.model.to_string(), "ollama:worker");
        }
        assert_eq!(requests[1].session_id, llm.session_id);
        assert_eq!(requests[1].reasoning, ReasoningLevel::High);
        assert_eq!(
            requests[1].provider_preferences.as_ref().unwrap().zdr,
            Some(true)
        );
        assert_eq!(llm.model.to_string(), "ollama:parent");
    }

    #[tokio::test]
    async fn model_override_rejects_another_provider_before_sending_requests() {
        let manager = AgentManager::new(1, 1);
        let provider = Arc::new(RecordingProvider::default());
        let error = manager
            .spawn(
                SpawnBuilder::new("Inspect the code.", provider.clone())
                    .model("openrouter:example/worker".parse().unwrap()),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("current provider (ollama)"));
        assert!(provider.requests.lock().unwrap().is_empty());
        assert!(manager.conversations().is_empty());
    }

    #[tokio::test]
    async fn follow_up_retains_history_and_wait_keeps_the_conversation() {
        let manager = AgentManager::new(1, 1);
        let provider = Arc::new(RecordingProvider::default());
        let id = manager
            .spawn(
                SpawnBuilder::new("Inspect the code.", provider.clone())
                    .name("worker")
                    .system_prompt("Review carefully.")
                    .llm(AgentLlmConfig::new("ollama:example".parse().unwrap())),
            )
            .await
            .unwrap();
        assert_eq!(manager.wait(id).await.unwrap(), "Reply: Inspect the code.");
        assert_eq!(manager.active_count().await, 0);
        assert_eq!(manager.resolve("worker").await.unwrap(), id);
        assert_eq!(manager.wait(id).await.unwrap(), "Reply: Inspect the code.");
        let handle = manager.conversation(id).unwrap();
        let before = handle.events_since(0).len();
        assert!(before > 1);

        manager
            .follow_up(id, ConversationEntry::user("Explain the result."))
            .unwrap();
        assert_eq!(
            manager.wait(id).await.unwrap(),
            "Reply: Explain the result."
        );
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].model, requests[0].model);
        let messages = &requests[1].messages;
        assert!(
            messages
                .iter()
                .any(|m| m.role == Role::System && m.content == "Review carefully.")
        );
        assert!(
            messages
                .iter()
                .any(|m| m.role == Role::User && m.content == "Inspect the code.")
        );
        assert!(
            messages
                .iter()
                .any(|m| m.role == Role::Assistant && m.content == "Reply: Inspect the code.")
        );
        assert_eq!(messages.last().unwrap().content, "Explain the result.");
        let following = handle.events_since(before);
        assert!(
            matches!(&following[0], SubagentEvent::User(entry) if entry.content() == "Explain the result.")
        );
        assert!(
            following
                .iter()
                .any(|event| matches!(event, SubagentEvent::Agent(AgentEvent::Done(_))))
        );
        assert!(handle.events_since(usize::MAX).is_empty());
    }

    #[tokio::test]
    async fn queued_messages_survive_a_timed_out_wait_and_respect_the_limit() {
        let gate = Arc::new(Notify::new());
        let provider = Arc::new(RecordingProvider {
            gate: Some(gate.clone()),
            ..Default::default()
        });
        let manager = AgentManager::new(1, 1);
        let id = manager
            .spawn(SpawnBuilder::new("First.", provider.clone()))
            .await
            .unwrap();
        manager
            .follow_up(id, ConversationEntry::user("Second."))
            .unwrap();
        manager
            .follow_up(id, ConversationEntry::user("Third."))
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), manager.wait(id))
                .await
                .is_err()
        );
        assert_eq!(manager.active_count().await, 1);
        assert!(
            manager
                .spawn(SpawnBuilder::new("Other.", provider.clone()))
                .await
                .is_err()
        );
        assert_eq!(
            manager
                .resolve(&id.simple().to_string()[..8])
                .await
                .unwrap(),
            id
        );
        gate.notify_one();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), manager.wait(id))
                .await
                .unwrap()
                .unwrap(),
            "Reply: Third."
        );
        assert_eq!(provider.requests.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn cancelled_conversations_can_be_reopened_and_address_the_latest_name() {
        let provider = Arc::new(RecordingProvider {
            gate: Some(Arc::new(Notify::new())),
            ..Default::default()
        });
        let manager = AgentManager::new(1, 1);
        let first = manager
            .spawn(SpawnBuilder::new("First.", provider.clone()).name("worker"))
            .await
            .unwrap();
        // Wait until the first request is in flight before cancelling it.
        tokio::time::timeout(Duration::from_secs(5), async {
            while provider.requests.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        manager.terminate(first).await.unwrap();
        assert!(manager.wait(first).await.is_err());
        manager
            .follow_up(first, ConversationEntry::user("Try again."))
            .unwrap();
        assert_eq!(manager.wait(first).await.unwrap(), "Reply: Try again.");
        let second = manager
            .spawn(SpawnBuilder::new("Second.", provider).name("worker"))
            .await
            .unwrap();
        assert_eq!(manager.resolve("worker").await.unwrap(), second);
        assert_eq!(manager.conversations().len(), 2);
        assert_eq!(manager.wait(second).await.unwrap(), "Reply: Second.");
    }
}
