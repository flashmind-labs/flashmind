//! `/model` argument completion for the REPL.
//!
//! [`ModelCompletion`] fetches the model lists of every configured provider in
//! the background and offers the `provider:id` strings that fuzzy match what
//! the user typed after `/model `, best match first.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use flashmind_tui::ArgumentProvider;
use flashmind_types::Provider;
use nucleo_matcher::Matcher;
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};

use crate::config::Config;
use crate::provider::build_provider;

/// How long one provider may take to list its models.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// Model ids offered after `/model `, filled in once the providers answer.
#[derive(Debug, Default)]
pub struct ModelCompletion {
    models: Mutex<Vec<String>>,
}

impl ModelCompletion {
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
                    completion.extend(models.into_iter().map(|m| format!("{provider}:{}", m.id)));
                }
            });
        }
        completion
    }

    fn extend(&self, ids: impl IntoIterator<Item = String>) {
        if let Ok(mut models) = self.models.lock() {
            models.extend(ids);
            models.sort();
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
            .match_list(models.iter(), &mut matcher)
            .into_iter()
            .map(|(model, _score)| model.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_command_lists_models_matching_the_query() {
        let completion = ModelCompletion::default();
        completion.extend(
            [
                "openrouter:z-ai/GLM-4.6",
                "ollama:qwen3",
                "openrouter:z-ai/glm-4.5",
            ]
            .map(String::from),
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
            .map(String::from),
        );
        assert_eq!(
            completion.complete("/model", "glm5.3"),
            ["openrouter:z-ai/glm-5.3", "openrouter:z-ai/glm-5.3-air"]
        );
    }
}
