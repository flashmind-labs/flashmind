//! Additional system instructions entered during an ACP session.

use anyhow::Result;
use flashmind_core::Conversation;
use flashmind_memory::session::{SessionEntryKind, SessionStore};

use crate::session::conv_to_session;

const METADATA_KEY: &str = "acp_system_instructions";

#[derive(Default)]
pub(crate) struct SystemInstructions {
    additions: Vec<String>,
}

impl SystemInstructions {
    pub(super) fn append(&mut self, conversation: &mut Conversation, instructions: &str) {
        let mut system = Self::current(conversation).to_owned();
        system.push_str("\n\n");
        system.push_str(instructions);
        conversation.set_system(system);
        self.additions.push(instructions.to_owned());
    }

    pub(super) fn current(conversation: &Conversation) -> &str {
        conversation
            .entries()
            .first()
            .filter(|entry| entry.is_system())
            .map_or("", |entry| entry.content())
    }

    /// Reapply session additions after refreshing the launch prompt and files.
    pub(super) async fn restore(
        store: &SessionStore,
        key: &str,
        conversation: &mut Conversation,
    ) -> Result<Self> {
        let entries = store.load(key).await?;
        let additions = entries
            .iter()
            .find(|entry| matches!(entry.entry_kind, SessionEntryKind::SystemPrompt))
            .and_then(|entry| entry.metadata.as_ref())
            .and_then(|metadata| metadata.get(METADATA_KEY))
            .cloned()
            .map(serde_json::from_value::<Vec<String>>)
            .transpose()?
            .unwrap_or_default();
        let mut instructions = Self::default();
        for addition in additions {
            instructions.append(conversation, &addition);
        }
        Ok(instructions)
    }

    pub(super) async fn save(
        &self,
        store: &SessionStore,
        key: &str,
        conversation: &Conversation,
    ) -> Result<()> {
        let mut entries: Vec<_> = conversation
            .entries()
            .iter()
            .enumerate()
            .map(|(index, entry)| conv_to_session(entry, key, index as i64))
            .collect();
        if let Some(system) = entries
            .iter_mut()
            .find(|entry| matches!(entry.entry_kind, SessionEntryKind::SystemPrompt))
        {
            system.metadata = Some(serde_json::json!({ METADATA_KEY: self.additions }));
        }
        store.rewrite(key, &entries).await
    }
}
