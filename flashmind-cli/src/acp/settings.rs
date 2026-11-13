//! Session selections, saved before they are acknowledged and applied between turns.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use agent_client_protocol::{
    Error,
    schema::v1::{SessionConfigOption, SessionConfigOptionCategory, SessionConfigSelectOption},
};
use flashmind_types::{LlmProvider, Model, ModelCapabilities, ModelPricing, ReasoningLevel};
use tokio::time::timeout;

use super::{
    open::{Opened, Opener, Saved},
    server::failed,
};
use crate::provider::fetch_pricing;

type AsyncMutex<T> = tokio::sync::Mutex<T>;
const CATALOG_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_MODELS: usize = 500;

struct Selection {
    model: Model,
    provider: Arc<dyn LlmProvider>,
    pricing: ModelPricing,
    capabilities: ModelCapabilities,
    context_window: Option<u32>,
    options: Vec<SessionConfigSelectOption>,
}

/// Locks serialize durable selections so an older save cannot replace a newer one.
pub(super) struct Settings {
    level: AsyncMutex<ReasoningLevel>,
    images: AtomicBool,
    selected: AsyncMutex<Selection>,
    saved: Option<Saved>,
    key: String,
    opener: Arc<dyn Opener>,
}

impl Settings {
    pub(super) async fn new(opened: &Opened, opener: Arc<dyn Opener>) -> Self {
        let model = opened.agent.llm().model.clone();
        let provider = opened.agent.provider_arc().clone();
        let options = model_options(&model, &provider).await;
        Self {
            level: AsyncMutex::new(opened.agent.llm().reasoning),
            images: AtomicBool::new(opened.agent.capabilities().images),
            selected: AsyncMutex::new(Selection {
                model,
                provider,
                options,
                pricing: ModelPricing::default(),
                capabilities: opened.agent.capabilities(),
                context_window: Some(opened.agent.context_window()),
            }),
            saved: opened.saved.clone(),
            key: opened.key.clone(),
            opener,
        }
    }

    pub(super) async fn options(&self) -> Vec<SessionConfigOption> {
        let mut options = thought_options(*self.level.lock().await);
        let selected = self.selected.lock().await;
        options.push(
            SessionConfigOption::select(
                "model",
                "Model",
                selected.model.to_string(),
                selected.options.clone(),
            )
            .category(SessionConfigOptionCategory::Model),
        );
        options
    }

    pub(super) async fn set(&self, level: ReasoningLevel) -> Result<(), Error> {
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

    pub(super) async fn select_model(&self, value: &str) -> Result<(), Error> {
        let model: Model = value
            .parse()
            .map_err(|_| Error::invalid_params().data("expected provider:model"))?;
        if model.name().is_empty() || value.chars().any(char::is_control) {
            return Err(Error::invalid_params().data("expected provider:model"));
        }
        let mut selected = self.selected.lock().await;
        if selected.model == model {
            return Ok(());
        }
        let provider = if model.provider == selected.model.provider {
            selected.provider.clone()
        } else {
            self.opener
                .provider(&model)
                .map_err(|error| failed("could not switch provider", &error))?
        };
        let (pricing, capabilities, context_window) = timeout(CATALOG_TIMEOUT, async {
            tokio::join!(
                fetch_pricing(&provider, &model),
                provider.capabilities(&model),
                provider.context_window(&model)
            )
        })
        .await
        .map_err(|_| Error::internal_error().data("the model's metadata timed out"))?;
        let options = model_options(&model, &provider).await;
        self.persist_model(&model)
            .await
            .map_err(|error| failed("could not save the model", &error))?;
        *selected = Selection {
            model,
            provider,
            pricing,
            options,
            capabilities,
            context_window,
        };
        Ok(())
    }

    /// The running call keeps its model; the next turn uses the latest selection.
    pub(super) async fn apply(&self, opened: &mut Opened) {
        let selected = self.selected.lock().await;
        if opened.agent.llm().model != selected.model {
            opened.agent.set_provider(selected.provider.clone());
            opened.agent.llm_mut().model = selected.model.clone();
            opened
                .agent
                .apply_features(selected.capabilities, selected.context_window);
            self.images
                .store(selected.capabilities.images, Ordering::Relaxed);
            self.opener.configure_model(&mut opened.agent);
            opened
                .meter
                .switch_model(selected.model.clone(), selected.pricing.clone());
        }
        opened.agent.llm_mut().reasoning = *self.level.lock().await;
    }

    pub(super) fn images(&self) -> bool {
        self.images.load(Ordering::Relaxed)
    }

    pub(super) async fn save_model(&self) -> anyhow::Result<()> {
        let selected = self.selected.lock().await;
        self.persist_model(&selected.model).await
    }

    async fn persist_model(&self, model: &Model) -> anyhow::Result<()> {
        if let Some(saved) = &self.saved {
            saved.store.update_model(&self.key, model).await?;
        }
        Ok(())
    }
}

async fn model_options(
    model: &Model,
    provider: &Arc<dyn LlmProvider>,
) -> Vec<SessionConfigSelectOption> {
    let mut options = vec![SessionConfigSelectOption::new(
        model.to_string(),
        model.to_string(),
    )];
    let providers = Arc::new([(model.provider.clone(), provider.clone())].into());
    if let Ok(models) = flashmind_tools::list_models::list_models(
        &providers,
        Some(flashmind_types::ModelCategory::Chat),
        None,
    )
    .await
    {
        for item in models.into_iter().take(MAX_MODELS - 1) {
            let id = item.id;
            if id != model.to_string() {
                options.push(SessionConfigSelectOption::new(
                    id.clone(),
                    item.name.unwrap_or(id),
                ));
            }
        }
    }
    options
}

pub(super) fn parse_level(value: &str) -> Result<ReasoningLevel, Error> {
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
