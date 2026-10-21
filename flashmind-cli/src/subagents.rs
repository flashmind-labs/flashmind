//! Show active agent conversations in the terminal browser.

use std::sync::Arc;

use flashmind_core::AgentManager;
use flashmind_core::subagent::SubagentEvent;
use flashmind_tui::widgets::PastedImage;
use flashmind_tui::widgets::repl::subagents::{ConversationEvent, SubagentInfo, SubagentSource};
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
    use flashmind_core::SpawnBuilder;
    use flashmind_types::{
        CompletionRequest, CompletionStream, FinishReason, LlmProvider, Provider, StreamEvent,
    };
    use futures::{StreamExt, stream};
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
