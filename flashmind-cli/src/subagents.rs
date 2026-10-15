//! Connect retained agent conversations to the terminal browser.

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
            .map(|handle| {
                let status = handle.status();
                let short_id = handle.id.simple().to_string()[..8].to_string();
                let name = handle.name.as_deref().unwrap_or(&short_id);
                let title = format!("{name} ({short_id})");
                let task = handle.task.lines().next().unwrap_or_default();
                SubagentInfo {
                    id: handle.id,
                    label: format!("{title}: {}", truncate_utf8(task, 100)),
                    title,
                    running: status.is_running(),
                    status: status.to_string(),
                }
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
