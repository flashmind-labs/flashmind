//! `/model` argument completion for the REPL.
//!
//! [`ModelCompletion`] fetches the model lists of every configured provider in
//! the background and offers `provider:id` strings matching what the user has
//! typed after `/model `.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use flashmind_tui::ArgumentProvider;
use flashmind_types::Provider;

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
    pub fn spawn(providers: &[Provider], config: &Config) -> Arc<Self> {
        let completion = Arc::new(Self::default());
        for provider in providers {
            let Ok(llm) = build_provider(provider, config) else {
                continue;
            };
            let provider = provider.clone();
            let completion = Arc::clone(&completion);
            tokio::spawn(async move {
                let Ok(Some(models)) = tokio::time::timeout(FETCH_TIMEOUT, llm.list_models()).await
                else {
                    tracing::debug!(%provider, "could not list models for completion");
                    return;
                };
                let ids = models.into_iter().map(|m| format!("{provider}:{}", m.id));
                completion.extend(ids);
            });
        }
        completion
    }

    fn extend(&self, ids: impl IntoIterator<Item = String>) {
        let Ok(mut models) = self.models.lock() else {
            return;
        };
        models.extend(ids);
        models.sort();
        models.dedup();
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
        matching_models(&models, query)
    }
}

/// Models containing `query`, ignoring case. Matches at the start of a name
/// segment (after `:`, `/`, `-`, `.` or `_`) come before matches inside one.
fn matching_models(models: &[String], query: &str) -> Vec<String> {
    let query = query.trim().to_lowercase();
    let mut at_segment = Vec::new();
    let mut inside = Vec::new();
    for model in models {
        let lower = model.to_lowercase();
        let Some(index) = lower.find(&query) else {
            continue;
        };
        let starts_segment = index == 0 || lower[..index].ends_with([':', '/', '-', '.', '_']);
        if starts_segment {
            at_segment.push(model.clone());
        } else {
            inside.push(model.clone());
        }
    }
    at_segment.extend(inside);
    at_segment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<String> {
        [
            "ollama:qwen3.5:2b",
            "openrouter:anthropic/claude-sonnet",
            "openrouter:thudm/xglm-test",
            "openrouter:z-ai/glm-4.6",
            "openrouter:z-ai/GLM-4.5-air",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn query_matches_anywhere_ignoring_case() {
        let found = matching_models(&models(), "glm");
        assert_eq!(
            found,
            [
                "openrouter:z-ai/glm-4.6",
                "openrouter:z-ai/GLM-4.5-air",
                "openrouter:thudm/xglm-test",
            ]
        );
    }

    #[test]
    fn empty_query_lists_every_model() {
        assert_eq!(matching_models(&models(), "").len(), models().len());
    }

    #[test]
    fn only_the_model_command_is_completed() {
        let completion = ModelCompletion::default();
        completion.extend(models());
        assert!(completion.complete("/reasoning", "glm").is_empty());
        assert_eq!(completion.complete("/model", "qwen"), ["ollama:qwen3.5:2b"]);
    }
}
