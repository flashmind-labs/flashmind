//! Model cards from Mistral's `/v1/models` and the capabilities they imply.

use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, Instant};

use chrono::DateTime;
use serde::Deserialize;

use flashmind_types::{ModelCapabilities, ModelInfo, ModelPricing};

const CACHE_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Deserialize)]
pub(super) struct ModelList {
    pub(super) data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct ModelCard {
    pub(super) id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    max_context_length: Option<u32>,
    #[serde(default)]
    aliases: Option<Vec<String>>,
    #[serde(default)]
    capabilities: CardCapabilities,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default)]
struct CardCapabilities {
    function_calling: bool,
    vision: bool,
    reasoning: bool,
    audio: bool,
    audio_transcription: bool,
}

impl ModelCard {
    fn answers_to(&self, model: &str) -> bool {
        self.id == model || self.aliases.iter().flatten().any(|alias| alias == model)
    }

    pub(super) fn context_length(&self) -> Option<u32> {
        self.max_context_length
    }

    pub(super) fn capabilities(&self) -> ModelCapabilities {
        let caps = self.capabilities;
        ModelCapabilities {
            tool_calling: caps.function_calling,
            images: caps.vision,
            reasoning: caps.reasoning,
            audio: caps.audio || caps.audio_transcription,
            ..ModelCapabilities::default()
        }
    }

    pub(super) fn to_model_info(&self) -> ModelInfo {
        let capabilities = self.capabilities();
        ModelInfo {
            id: self.id.clone(),
            name: self.name.clone(),
            created_at: self
                .created
                .and_then(|created| DateTime::from_timestamp(created, 0)),
            context_length: self.max_context_length,
            max_completion_tokens: None,
            categories: capabilities.categories(),
            capabilities,
            pricing: ModelPricing::default(),
        }
    }
}

/// Capabilities for a model `/v1/models` does not list: assume a text model
/// with tools, which every current Mistral chat model has.
pub(super) fn fallback_capabilities() -> ModelCapabilities {
    ModelCapabilities {
        tool_calling: true,
        ..ModelCapabilities::default()
    }
}

struct Snapshot {
    fetched: Instant,
    cards: Arc<[ModelCard]>,
}

/// The last model list, kept for an hour.
#[derive(Default)]
pub(super) struct ModelCache {
    snapshot: RwLock<Option<Snapshot>>,
}

impl ModelCache {
    /// Cards from a fresh snapshot, or `None` when it is missing or stale.
    pub(super) fn fresh(&self) -> Option<Arc<[ModelCard]>> {
        let snapshot = self.snapshot.read().unwrap_or_else(PoisonError::into_inner);
        snapshot
            .as_ref()
            .filter(|snapshot| snapshot.fetched.elapsed() < CACHE_TTL)
            .map(|snapshot| snapshot.cards.clone())
    }

    pub(super) fn store(&self, cards: Vec<ModelCard>) -> Arc<[ModelCard]> {
        let cards: Arc<[ModelCard]> = cards.into();
        *self
            .snapshot
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Some(Snapshot {
            fetched: Instant::now(),
            cards: cards.clone(),
        });
        cards
    }

    /// Whether the cached card for `model` says it reasons. False when unknown.
    pub(super) fn reasons(&self, model: &str) -> bool {
        self.fresh()
            .and_then(|cards| find(&cards, model).map(|card| card.capabilities.reasoning))
            .unwrap_or(false)
    }
}

pub(super) fn find<'a>(cards: &'a [ModelCard], model: &str) -> Option<&'a ModelCard> {
    cards.iter().find(|card| card.answers_to(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Follows the model card schema of Mistral's `/v1/models` (`BaseModelCard`).
    const MODELS: &str = r#"{
        "object": "list",
        "data": [
            {
                "id": "mistral-medium-3-5",
                "object": "model",
                "created": 1777334400,
                "owned_by": "mistralai",
                "capabilities": {
                    "completion_chat": true,
                    "function_calling": true,
                    "reasoning": true,
                    "completion_fim": false,
                    "fine_tuning": false,
                    "vision": true,
                    "classification": false,
                    "audio": false
                },
                "name": "mistral-medium-3-5",
                "description": "Frontier-class multimodal model",
                "max_context_length": 262144,
                "aliases": ["mistral-medium-latest", "mistral-medium-3"],
                "deprecation": null,
                "default_model_temperature": 0.7,
                "type": "base"
            },
            {
                "id": "voxtral-mini-transcribe-realtime-3-0",
                "object": "model",
                "capabilities": {
                    "completion_chat": false,
                    "audio_transcription": true,
                    "audio_transcription_realtime": true
                },
                "aliases": null,
                "type": "base"
            }
        ]
    }"#;

    fn cards() -> Vec<ModelCard> {
        serde_json::from_str::<ModelList>(MODELS).unwrap().data
    }

    #[test]
    fn chat_card_maps_to_capabilities() {
        let cards = cards();
        let card = find(&cards, "mistral-medium-latest").unwrap();
        assert_eq!(card.id, "mistral-medium-3-5");
        let caps = card.capabilities();
        assert!(caps.tool_calling && caps.images && caps.reasoning);
        assert!(!caps.audio && !caps.documents && !caps.video);
        let info = card.to_model_info();
        assert_eq!(info.context_length, Some(262_144));
        assert!(info.created_at.is_some());
    }

    #[test]
    fn transcription_card_counts_as_audio_input() {
        let cards = cards();
        let card = find(&cards, "voxtral-mini-transcribe-realtime-3-0").unwrap();
        let caps = card.capabilities();
        assert!(caps.audio && !caps.tool_calling);
        assert_eq!(card.context_length(), None);
    }

    #[test]
    fn cache_answers_reasoning_by_alias() {
        let cache = ModelCache::default();
        assert!(!cache.reasons("mistral-medium-latest"));
        cache.store(cards());
        assert!(cache.reasons("mistral-medium-latest"));
        assert!(!cache.reasons("voxtral-mini-transcribe-realtime-3-0"));
    }
}
