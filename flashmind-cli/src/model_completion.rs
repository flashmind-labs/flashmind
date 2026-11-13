//! `/model` argument completion for the REPL.
//!
//! [`ModelCompletion`] fetches the model lists of every configured provider in
//! the background and offers the `provider:id` strings containing every word
//! typed after `/model `, newest first, then best match. It also
//! offers the reasoning levels after `/thinking `.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use flashmind_tui::ArgumentProvider;
use flashmind_types::{Model, ModelCategory, ModelInfo, ModelPricing, Provider, ReasoningLevel};
#[cfg(feature = "fuzzy-completion")]
use nucleo_matcher::Matcher;
#[cfg(feature = "fuzzy-completion")]
use nucleo_matcher::pattern::{AtomKind, CaseMatching, Normalization, Pattern};
use strum::VariantArray;
use tokio::task::JoinSet;

use crate::config::Config;
use crate::provider::build_provider;
use crate::usage::ModelPriceSummary;

/// Model ids offered after `/model `, filled in once the providers answer.
#[derive(Debug, Default)]
pub struct ModelCompletion {
    models: Mutex<BTreeMap<String, ModelInfo>>,
}

impl ModelCompletion {
    pub(crate) fn pricing(&self, model: &Model) -> Option<ModelPricing> {
        self.models
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&format!("{}:{}", model.provider, model.capability_name()))
            .map(|model| model.pricing.clone())
    }

    /// Fetch model catalogs in owned tasks. Dropping the set cancels the fetches.
    pub fn start(
        self: &Arc<Self>,
        providers: impl IntoIterator<Item = Provider>,
        config: &Config,
    ) -> JoinSet<()> {
        let mut tasks = JoinSet::new();
        for provider in providers {
            let Ok(llm) = build_provider(&provider, config) else {
                continue;
            };
            let completion = Arc::clone(self);
            tasks.spawn(async move {
                let registry = Arc::new([(provider.clone(), llm)].into());
                match flashmind_tools::list_models::list_models(
                    &registry,
                    Some(ModelCategory::Chat),
                    None,
                )
                .await
                {
                    Ok(mut models) => {
                        crate::reference_prices::enrich(&provider, &mut models).await;
                        completion.extend(models.into_iter().map(|m| (m.id.clone(), m)));
                    }
                    Err(error) => tracing::warn!(%provider, %error, "model catalog unavailable"),
                }
            });
        }
        tasks
    }

    fn extend(&self, entries: impl IntoIterator<Item = (String, ModelInfo)>) {
        if let Ok(mut models) = self.models.lock() {
            models.extend(
                entries
                    .into_iter()
                    .filter(|(_, model)| model.categories.contains(&ModelCategory::Chat)),
            );
        }
    }
}

impl ArgumentProvider for ModelCompletion {
    fn complete(&self, command: &str, query: &str) -> Vec<String> {
        if command == "/autocompact" {
            let query = query.to_lowercase();
            return ["global", "model", "session"]
                .into_iter()
                .filter(|scope| scope.starts_with(&query))
                .map(str::to_owned)
                .collect();
        }
        if command == "/thinking" {
            let query = query.to_lowercase();
            return ReasoningLevel::VARIANTS
                .iter()
                .map(ReasoningLevel::to_string)
                .filter(|level| level.starts_with(&query))
                .collect();
        }
        if !matches!(command, "/model" | "/models") {
            return Vec::new();
        }
        let Ok(models) = self.models.lock() else {
            return Vec::new();
        };
        #[cfg(feature = "fuzzy-completion")]
        let pattern = Pattern::new(
            query,
            CaseMatching::Ignore,
            Normalization::Smart,
            AtomKind::Substring,
        );
        #[cfg(feature = "fuzzy-completion")]
        let mut matcher = Matcher::new(nucleo_matcher::Config::DEFAULT);
        #[cfg(feature = "fuzzy-completion")]
        let mut matching = pattern.match_list(models.keys(), &mut matcher);
        #[cfg(not(feature = "fuzzy-completion"))]
        let mut matching: Vec<_> = {
            let query = query.to_lowercase();
            let words: Vec<_> = query.split_whitespace().collect();
            models
                .keys()
                .filter_map(|name| {
                    let normalized = name.to_lowercase();
                    words
                        .iter()
                        .all(|word| normalized.contains(word))
                        .then_some((name, 0_u32))
                })
                .collect()
        };
        matching.sort_by(|(a, score_a), (b, score_b)| {
            models[*b]
                .created_at
                .cmp(&models[*a].created_at)
                .then_with(|| score_b.cmp(score_a))
                .then_with(|| a.cmp(b))
        });
        matching
            .into_iter()
            .map(|(model, _)| model.clone())
            .collect()
    }

    fn description(&self, command: &str, candidate: &str) -> Option<String> {
        if !matches!(command, "/model" | "/models") {
            return None;
        }
        let Ok(models) = self.models.lock() else {
            return None;
        };
        models
            .get(candidate)
            .map(|pricing| ModelPriceSummary(&pricing.pricing).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    #[test]
    fn autocompact_completes_session_scope() {
        let completion = ModelCompletion::default();
        assert_eq!(
            completion.complete("/autocompact", ""),
            ["global", "model", "session"]
        );
        assert_eq!(completion.complete("/autocompact", "se"), ["session"]);
    }

    fn model(pricing: ModelPricing) -> ModelInfo {
        ModelInfo {
            image: Default::default(),
            id: String::new(),
            name: None,
            created_at: None,
            context_length: None,
            max_completion_tokens: None,
            capabilities: Default::default(),
            categories: vec![ModelCategory::Chat],
            pricing,
        }
    }

    #[test]
    fn completion_excludes_media_only_models() {
        let completion = ModelCompletion::default();
        completion.extend([
            (
                "openrouter:example/chat".into(),
                model(ModelPricing::default()),
            ),
            (
                "openrouter:example/image".into(),
                ModelInfo {
                    categories: vec![ModelCategory::ImageGeneration],
                    ..model(ModelPricing::default())
                },
            ),
        ]);
        for command in ["/model", "/models"] {
            assert_eq!(
                completion.complete(command, ""),
                ["openrouter:example/chat"]
            );
            assert!(completion.complete(command, "image").is_empty());
        }
    }

    #[test]
    fn pricing_uses_the_resolved_model_id() {
        let completion = ModelCompletion::default();
        completion.extend([(
            "openrouter:example/worker".into(),
            model(ModelPricing {
                prompt: Some(Decimal::new(1, 7)),
                ..Default::default()
            }),
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
            model(ModelPricing {
                prompt: Some(Decimal::new(25, 8)),
                completion: Some(Decimal::new(15, 7)),
                ..Default::default()
            }),
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
            .map(|id| (id.to_string(), model(ModelPricing::default()))),
        );
        assert_eq!(
            completion.complete("/model", "glm"),
            ["openrouter:z-ai/GLM-4.6", "openrouter:z-ai/glm-4.5"]
        );
        assert_eq!(completion.complete("/model", "").len(), 3);
        assert!(completion.complete("/reasoning", "glm").is_empty());
    }

    #[test]
    fn thinking_command_lists_reasoning_levels() {
        let completion = ModelCompletion::default();
        assert_eq!(
            completion.complete("/thinking", ""),
            ["off", "low", "medium", "high"]
        );
        assert_eq!(completion.complete("/thinking", "Me"), ["medium"]);
        assert!(completion.complete("/thinking", "x").is_empty());
    }

    #[cfg(feature = "fuzzy-completion")]
    #[test]
    fn model_query_requires_contiguous_words() {
        let completion = ModelCompletion::default();
        completion.extend(
            [
                "openrouter:anthropic/claude-opus-5.5",
                "openrouter:anthropic/claude-opus-5.4",
                "openrouter:example/optimum-super-5.5",
                "openrouter:example/opus-5-extra-5",
            ]
            .map(|id| (id.to_string(), model(ModelPricing::default()))),
        );
        for command in ["/model", "/models"] {
            assert_eq!(
                completion.complete(command, "opus"),
                [
                    "openrouter:example/opus-5-extra-5",
                    "openrouter:anthropic/claude-opus-5.4",
                    "openrouter:anthropic/claude-opus-5.5",
                ]
            );
            for query in ["opus 5.5", "OPUS 5.5", "5.5 opus", "  opus  5.5  "] {
                assert_eq!(
                    completion.complete(command, query),
                    ["openrouter:anthropic/claude-opus-5.5"]
                );
            }
        }
    }

    #[cfg(not(feature = "fuzzy-completion"))]
    #[test]
    fn simple_completion_matches_all_words_case_insensitively() {
        let completion = ModelCompletion::default();
        completion.extend([
            (
                "provider:example/opus-5.5".into(),
                model(ModelPricing::default()),
            ),
            (
                "provider:example/opus-4".into(),
                model(ModelPricing::default()),
            ),
        ]);
        assert_eq!(
            completion.complete("/model", "5.5 OPUS"),
            ["provider:example/opus-5.5"]
        );
    }

    #[test]
    fn query_matches_words_across_separators_and_ranks_closest_first() {
        let completion = ModelCompletion::default();
        completion.extend(
            [
                "openrouter:z-ai/glm-4.5",
                "openrouter:z-ai/glm-5.3",
                "openrouter:z-ai/glm-5.3-air",
            ]
            .map(|id| (id.to_string(), model(ModelPricing::default()))),
        );
        assert_eq!(
            completion.complete("/model", "glm 5.3"),
            ["openrouter:z-ai/glm-5.3", "openrouter:z-ai/glm-5.3-air"]
        );
    }
    #[test]
    fn completions_sort_newest_first_even_when_filtering_and_put_unknown_dates_last() {
        use chrono::DateTime;
        let completion = ModelCompletion::default();
        completion.extend(
            [
                ("provider:example/model", Some(100)),
                ("provider:example/model-new", Some(300)),
                ("provider:example/unknown", None),
                ("provider:example/middle", Some(200)),
            ]
            .map(|(id, date)| {
                let mut info = model(ModelPricing::default());
                info.created_at = date.and_then(|date| DateTime::from_timestamp(date, 0));
                (id.to_owned(), info)
            }),
        );
        for command in ["/model", "/models"] {
            for query in ["", "example"] {
                assert_eq!(
                    completion.complete(command, query),
                    [
                        "provider:example/model-new",
                        "provider:example/middle",
                        "provider:example/model",
                        "provider:example/unknown",
                    ]
                );
            }
        }
    }
}
