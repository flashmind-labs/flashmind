//! Opening the flsh session behind an ACP session.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use flashmind_core::{Agent, AgentManager, Conversation};
use flashmind_memory::session::SessionStore;
use flashmind_tools::ToolSync;
use flashmind_tools::background::Jobs;
use flashmind_types::ModelPricing;

use super::usage::Meter;

use crate::config::Config;
use crate::provider::fetch_pricing;
use crate::session::{load_conversation_from, new_session_key, open_session_store_for};
use crate::{Cli, Startup, start_session};

/// Where a session's turns are saved, so `flsh resume` can pick it up.
pub(crate) struct Saved {
    pub store: SessionStore,
    pub model: String,
}

/// A flsh session ready for prompts.
pub(crate) struct Opened {
    /// The flsh session key, which is also the ACP session id.
    pub key: String,
    pub agent: Agent,
    pub conversation: Conversation,
    pub saved: Option<Saved>,
    /// Delegated agents, whose results go to the conversation.
    pub subagents: Option<Arc<AgentManager>>,
    pub tool_sync: Option<Arc<ToolSync>>,
    /// The background jobs of the session's `exec` tool.
    pub jobs: Option<Jobs>,
    /// The session's context use and cost.
    pub meter: Meter,
}

/// How long the model's price may take to arrive before the session opens
/// without it.
const PRICING_TIMEOUT: Duration = Duration::from_secs(15);

/// Opens flsh sessions; tests open them over a stub model.
#[async_trait]
pub(crate) trait Opener: Send + Sync + 'static {
    /// A new session working in `cwd`, or the saved session `resume`.
    async fn open(&self, cwd: &Path, resume: Option<&str>) -> Result<Opened>;
}

/// Opens sessions as `flsh` does: the configured model, every tool, and
/// the default system prompt with the project's instructions.
pub(crate) struct FlshOpener {
    pub cli: Cli,
    pub config: Config,
}

#[async_trait]
impl Opener for FlshOpener {
    async fn open(&self, cwd: &Path, resume: Option<&str>) -> Result<Opened> {
        anyhow::ensure!(
            cwd.is_dir(),
            "the session directory {} does not exist",
            cwd.display()
        );
        let store = open_session_store_for(cwd).await?;
        let mut saved_usage = None;
        let (key, model) = match resume {
            Some(key) => {
                let sessions = store.list_sessions().await?;
                let summary = sessions
                    .iter()
                    .find(|summary| summary.chat_key == key)
                    .with_context(|| format!("no saved session {key} in {}", cwd.display()))?;
                let model = crate::interactive::resume_model(&self.cli, &self.config, summary)?;
                saved_usage = Some((
                    crate::interactive::summary_totals(summary),
                    crate::interactive::summary_usage(summary),
                ));
                (key.to_owned(), model)
            }
            None => (
                new_session_key(),
                crate::provider::resolve_model(&self.cli, &self.config)?,
            ),
        };
        let Startup {
            mut agent,
            subagents,
            tool_sync,
            system_prompt,
            jobs,
            ..
        } = start_session(&self.cli, &self.config, &model, &key, cwd).await?;
        let provider = agent.provider_arc().clone();
        let pricing = async {
            tokio::time::timeout(PRICING_TIMEOUT, fetch_pricing(&provider, &model))
                .await
                .unwrap_or_else(|_| {
                    tracing::warn!(%model, "the model's price timed out; cost is not reported");
                    ModelPricing::default()
                })
        };
        let ((), pricing) = tokio::join!(agent.refresh_features(), pricing);
        let mut meter = Meter::new(model.clone(), pricing);
        if let Some((total, last)) = saved_usage {
            meter = meter.resumed(total, last);
        }
        let conversation = match resume {
            Some(_) => load_conversation_from(&store, &key, &system_prompt).await?,
            None => Conversation::with_system(&system_prompt),
        };
        Ok(Opened {
            key,
            agent,
            conversation,
            saved: Some(Saved {
                store,
                model: model.to_string(),
            }),
            subagents: Some(subagents),
            tool_sync: Some(Arc::new(tool_sync)),
            jobs: Some(jobs),
            meter,
        })
    }
}
