//! `/model` argument completion for the REPL.
//!
//! [`ModelCompletion`] fetches the model lists of every configured provider in
//! the background and offers the `provider:id` strings that fuzzy match what
//! the user typed after `/model `, best match first.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use flashmind_tui::ArgumentProvider;
use flashmind_types::{Model, ModelPricing, Provider};
use nucleo_matcher::Matcher;
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};

use crate::config::Config;
use crate::provider::build_provider;
use crate::usage::ModelPriceSummary;

/// How long one provider may take to list its models.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// Model ids offered after `/model `, filled in once the providers answer.
#[derive(Debug, Default)]
pub struct ModelCompletion {
    models: Mutex<BTreeMap<String, ModelPricing>>,
}

impl ModelCompletion {
    pub(crate) fn pricing(&self, model: &Model) -> Option<ModelPricing> {
        self.models
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&format!("{}:{}", model.provider, model.capability_name()))
            .cloned()
    }

    /// Start fetching models from `providers` and return the completion
    /// source right away. Candidates appear as each provider answers.
    pub fn spawn(providers: impl IntoIterator<Item = Provider>, config: &Config) -> Arc<Self> {
        let completion = Arc::new(Self::default());
        for provider in providers {
            let Ok(llm) = build_provider(&provider, config) else {
                continue;
            };
            let completion = Arc::clone(&completion);
            tokio::spawn(async move {
                if let Ok(Some(models)) =
                    tokio::time::timeout(FETCH_TIMEOUT, llm.list_models()).await
                {
                    completion.extend(
                        models
                            .into_iter()
                            .map(|m| (format!("{provider}:{}", m.id), m.pricing)),
                    );
                }
            });
        }
        completion
    }

    fn extend(&self, entries: impl IntoIterator<Item = (String, ModelPricing)>) {
        if let Ok(mut models) = self.models.lock() {
            models.extend(entries);
        }
    }
}

impl ArgumentProvider for ModelCompletion {
    fn complete(&self, command: &str, query: &str) -> Vec<String> {
        if command != "/model" {
            return Vec::new();
        }
        let Ok(models) = self.models.lock() else {
            return Vec::new();
        };
        let pattern = Pattern::parse(query, CaseMatching::Ignore, Normalization::Smart);
        let mut matcher = Matcher::new(nucleo_matcher::Config::DEFAULT);
        pattern
            .match_list(models.keys(), &mut matcher)
            .into_iter()
            .map(|(model, _score)| model.clone())
            .collect()
    }

    fn description(&self, command: &str, candidate: &str) -> Option<String> {
        if command != "/model" {
            return None;
        }
        let Ok(models) = self.models.lock() else {
            return None;
        };
        models
            .get(candidate)
            .map(|pricing| ModelPriceSummary(pricing).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    #[test]
    fn pricing_uses_the_resolved_model_id() {
        let completion = ModelCompletion::default();
        completion.extend([(
            "openrouter:example/worker".into(),
            ModelPricing {
                prompt: Some(Decimal::new(1, 7)),
                ..Default::default()
            },
        )]);
        let model = "openrouter:alias,example/worker".parse().unwrap();
        assert_eq!(
            completion.pricing(&model).unwrap().prompt,
            Some(Decimal::new(1, 7))
        );
        assert!(
            completion
                .pricing(&"ollama:worker".parse().unwrap())
                .is_none()
        );
    }

    #[test]
    fn prices_are_displayed_separately_from_model_ids() {
        let completion = ModelCompletion::default();
        let id = "openrouter:example/model";
        completion.extend([(
            id.to_string(),
            ModelPricing {
                prompt: Some(Decimal::new(25, 8)),
                completion: Some(Decimal::new(15, 7)),
                ..Default::default()
            },
        )]);
        assert_eq!(completion.complete("/model", "example"), [id]);
        assert_eq!(
            completion.description("/model", id).as_deref(),
            Some("$0.25/m in, $1.5/m out")
        );
        assert!(completion.description("/other", id).is_none());
        assert!(completion.description("/model", "missing").is_none());
        assert!(completion.complete("/model", "$0.25").is_empty());
    }

    #[test]
    fn model_command_lists_models_matching_the_query() {
        let completion = ModelCompletion::default();
        completion.extend(
            [
                "openrouter:z-ai/GLM-4.6",
                "ollama:qwen3",
                "openrouter:z-ai/glm-4.5",
            ]
            .map(|id| (id.to_string(), ModelPricing::default())),
        );
        assert_eq!(
            completion.complete("/model", "glm"),
            ["openrouter:z-ai/GLM-4.6", "openrouter:z-ai/glm-4.5"]
        );
        assert_eq!(completion.complete("/model", "").len(), 3);
        assert!(completion.complete("/reasoning", "glm").is_empty());
    }

    #[test]
    fn query_matches_without_separators_and_ranks_closest_first() {
        let completion = ModelCompletion::default();
        completion.extend(
            [
                "openrouter:z-ai/glm-4.5",
                "openrouter:z-ai/glm-5.3",
                "openrouter:z-ai/glm-5.3-air",
            ]
            .map(|id| (id.to_string(), ModelPricing::default())),
        );
        assert_eq!(
            completion.complete("/model", "glm5.3"),
            ["openrouter:z-ai/glm-5.3", "openrouter:z-ai/glm-5.3-air"]
        );
    }
}
