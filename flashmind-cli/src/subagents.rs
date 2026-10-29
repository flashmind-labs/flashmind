//! Show active agent conversations in the terminal browser.

use std::sync::Arc;

use flashmind_core::AgentManager;
use flashmind_core::subagent::SubagentEvent;
use flashmind_tui::widgets::PastedImage;
use flashmind_tui::widgets::repl::subagents::{ConversationEvent, SubagentInfo, SubagentSource};
use flashmind_types::Model;
use flashmind_types::utils::truncate_utf8;
use uuid::Uuid;

use crate::interactive::user_entry;

pub struct SubagentConversations {
    manager: Arc<AgentManager>,
}

impl SubagentConversations {
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self { manager }
    }
}

impl SubagentSource for SubagentConversations {
    fn list(&self) -> Vec<SubagentInfo> {
        self.manager
            .conversations()
            .iter()
            .filter_map(|handle| {
                let status = handle.status();
                if !status.is_running() {
                    return None;
                }
                let short_id = handle.id.simple().to_string()[..8].to_string();
                let name = handle.name.as_deref().unwrap_or(&short_id);
                let title = format!("{name} ({short_id})");
                let task = handle.task.lines().next().unwrap_or_default();
                Some(SubagentInfo {
                    id: handle.id,
                    label: format!("{title}: {}", truncate_utf8(task, 100)),
                    title,
                    running: status.is_running(),
                    status: status.to_string(),
                })
            })
            .collect()
    }

    fn events(&self, id: Uuid, offset: usize) -> Result<Vec<ConversationEvent>, String> {
        let handle = self.manager.conversation(id).map_err(|e| e.to_string())?;
        Ok(handle
            .events_since(offset)
            .into_iter()
            .map(|event| match event {
                SubagentEvent::User(entry) => ConversationEvent::User(entry.content().to_string()),
                SubagentEvent::Agent(event) => ConversationEvent::Agent(event),
            })
            .collect())
    }

    fn send(&self, id: Uuid, text: &str, images: Vec<PastedImage>) -> Result<(), String> {
        self.manager
            .follow_up(id, user_entry(text, images))
            .map_err(|e| e.to_string())
    }

    fn set_model(&self, id: Uuid, model: Model) -> Result<(), String> {
        self.manager.set_model(id, model).map_err(|e| e.to_string())
    }

    fn cancel(&self, id: Uuid) {
        if let Ok(handle) = self.manager.conversation(id) {
            handle.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use flashmind_core::subagent::Notifications;
    use flashmind_core::{Agent, CancellationToken, Conversation, ConversationEntry, SpawnBuilder};
    use flashmind_tui::{Repl, ReplConfig};
    use flashmind_types::{
        AgentLlmConfig, CompletionRequest, CompletionStream, FinishReason, LlmProvider, Provider,
        Role, StreamEvent,
    };
    use futures_util::{StreamExt, stream};
    use std::sync::Mutex;
    use tokio::sync::Notify;

    struct GatedProvider(Arc<Notify>);

    #[async_trait]
    impl LlmProvider for GatedProvider {
        fn name(&self) -> &str {
            "test"
        }

        fn provider(&self) -> Provider {
            Provider::Ollama
        }

        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            let gate = self.0.clone();
            Box::pin(
                stream::once(async move {
                    gate.notified().await;
                    Ok(StreamEvent::ContentDelta("Result".into()))
                })
                .chain(stream::once(async {
                    Ok(StreamEvent::Finished(FinishReason::Stop))
                })),
            )
        }
    }

    struct RecordingProvider(Mutex<Vec<CompletionRequest>>);

    #[async_trait]
    impl LlmProvider for RecordingProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            self.0.lock().unwrap().push(request);
            Box::pin(stream::iter([
                Ok(StreamEvent::ContentDelta("Parent response.".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    #[tokio::test]
    async fn child_completion_wakes_idle_input_and_reaches_parent_without_a_user_message() {
        let manager = Arc::new(AgentManager::new(1, 1));
        let mut notifications = Notifications::new(manager.clone());
        let gate = Arc::new(Notify::new());
        let id = manager
            .spawn(
                SpawnBuilder::new("Inspect the code.", Arc::new(GatedProvider(gate.clone())))
                    .name("reviewer"),
            )
            .await
            .unwrap();
        assert!(notifications.take_completed().is_empty());
        gate.notify_one();
        manager.wait(id).await.unwrap();

        let mut parent = Conversation::with_system("Finish the task.");
        parent.add(ConversationEntry::user("Review the code."));
        let mut repl = Repl::new(ReplConfig::default());
        assert!(
            repl.read_input_until(|_| Ok(notifications.deliver(&mut parent)))
                .unwrap()
                .is_none()
        );
        assert!(!notifications.deliver(&mut parent));

        let provider = Arc::new(RecordingProvider(Mutex::new(Vec::new())));
        let mut agent = Agent::builder(provider.clone())
            .llm(AgentLlmConfig::new("ollama:parent".parse().unwrap()))
            .build_sync();
        let cancel = CancellationToken::new();
        let mut turn = agent.run_turn(&mut parent, &cancel);
        while turn.next().await.is_some() {}
        assert!(turn.take_result().unwrap().is_ok());
        let requests = provider.0.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0]
                .messages
                .iter()
                .filter(|message| message.role == Role::User)
                .count(),
            1
        );
        assert!(requests[0].messages.iter().any(|message| {
            message.content.contains("Subagent reviewer")
                && message.content.contains("Result")
                && message.content.contains("Inspect the code.")
        }));
    }

    #[tokio::test]
    async fn completed_agents_leave_the_view_but_keep_their_results() {
        let manager = Arc::new(AgentManager::new(1, 1));
        let gate = Arc::new(Notify::new());
        let provider = Arc::new(GatedProvider(gate.clone()));
        let id = manager
            .spawn(SpawnBuilder::new("Example task", provider))
            .await
            .unwrap();
        let source = SubagentConversations::new(manager.clone());
        assert_eq!(
            source.list().iter().map(|info| info.id).collect::<Vec<_>>(),
            [id]
        );

        gate.notify_one();
        assert_eq!(manager.wait(id).await.unwrap(), "Result");
        assert!(source.list().is_empty());
        assert!(!source.events(id, 0).unwrap().is_empty());
        assert_eq!(manager.wait(id).await.unwrap(), "Result");
    }
}
