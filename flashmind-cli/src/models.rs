use std::time::Duration;

use anyhow::{Context, Result, bail};
use flashmind_llm::OpenRouterProvider;
use flashmind_types::{ModelCategory, ModelInfo, Provider};
use tokio::task::JoinSet;

use crate::config::Config;
use crate::interactive::configured_providers;
use crate::provider::build_provider;
use crate::setup::{CATALOG_FEATURES, model_features};

const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

pub(crate) async fn run(
    filter: Option<&Provider>,
    feature: Option<&str>,
    config: &Config,
) -> Result<()> {
    if let Some(feature) = feature
        && !CATALOG_FEATURES.contains(&feature)
    {
        bail!(
            "unknown model feature '{feature}'; choose from: {}",
            CATALOG_FEATURES.join(", ")
        );
    }
    let auto_ollama = filter.is_none() && config.ollama_url.is_none();
    let providers: Vec<_> = match filter {
        Some(provider) => vec![provider.clone()],
        None => configured_providers(config)
            .into_iter()
            .map(|(_, provider)| provider)
            .filter(supports_model_catalog)
            .collect(),
    };
    let mut tasks = JoinSet::new();
    for provider in providers {
        let llm = match build_provider(&provider, config) {
            Ok(llm) => llm,
            Err(error) if filter.is_none() => {
                eprintln!("{provider}: {error:#}");
                continue;
            }
            Err(error) => return Err(error),
        };
        let audio_provider = if provider == Provider::OpenRouter {
            config
                .api_key_for(&provider)
                .map(|key| OpenRouterProvider::new(key.to_owned()))
        } else {
            None
        };
        tasks.spawn(async move {
            let (catalog, audio) = tokio::join!(
                tokio::time::timeout(FETCH_TIMEOUT, llm.list_models()),
                async {
                    match audio_provider {
                        Some(provider) => {
                            let (speech, transcription) = tokio::join!(
                                tokio::time::timeout(FETCH_TIMEOUT, provider.list_speech_models()),
                                tokio::time::timeout(
                                    FETCH_TIMEOUT,
                                    provider.list_transcription_models()
                                )
                            );
                            Some((speech, transcription))
                        }
                        None => None,
                    }
                }
            );
            let result = catalog.map(|catalog| {
                catalog.map(|mut models| {
                    if let Some((speech, transcription)) = audio {
                        for (result, category) in [
                            (speech, ModelCategory::Tts),
                            (transcription, ModelCategory::Stt),
                        ] {
                            match result {
                                Ok(Ok(entries)) => {
                                    for entry in entries {
                                        if let Some(model) =
                                            models.iter_mut().find(|model| model.id == entry.id)
                                        {
                                            if !model.categories.contains(&category) {
                                                model.categories.push(category);
                                            }
                                        } else {
                                            models.push(ModelInfo {
                                                image: Default::default(),
                                                id: entry.id,
                                                name: entry.name,
                                                created_at: None,
                                                context_length: None,
                                                max_completion_tokens: None,
                                                capabilities: Default::default(),
                                                categories: vec![category],
                                                pricing: Default::default(),
                                            });
                                        }
                                    }
                                }
                                Ok(Err(error)) => eprintln!(
                                    "openrouter: could not list {category} models: {error}"
                                ),
                                Err(_) => {
                                    eprintln!("openrouter: {category} model listing timed out")
                                }
                            }
                        }
                    }
                    models
                })
            });
            (provider, result)
        });
    }

    let mut models = Vec::new();
    let mut successful = false;
    while let Some(result) = tasks.join_next().await {
        let (provider, result) = result.context("model catalog task failed")?;
        match result {
            Ok(Some(catalog)) => {
                successful = true;
                models.extend(prefix_models(&provider, catalog));
            }
            Ok(None) if !(auto_ollama && provider == Provider::Ollama) => {
                eprintln!("{provider}: could not list models")
            }
            Err(_) if !(auto_ollama && provider == Provider::Ollama) => {
                eprintln!("{provider}: model listing timed out")
            }
            _ => {}
        }
    }
    if !successful {
        bail!("could not load any model catalogs");
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    for model in models
        .iter()
        .filter(|model| feature.is_none_or(|feature| model_features(model).contains(&feature)))
    {
        println!("{}", model.id);
    }
    Ok(())
}

fn supports_model_catalog(provider: &Provider) -> bool {
    !matches!(provider, Provider::Custom(name) if matches!(name.as_str(), "google" | "fal" | "meshy" | "tripo" | "elevenlabs"))
}

fn prefix_models(provider: &Provider, models: Vec<ModelInfo>) -> Vec<ModelInfo> {
    models
        .into_iter()
        .map(|mut model| {
            model.id = format!("{provider}:{}", model.id);
            model
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_skips_media_only_providers() {
        for name in ["google", "fal", "meshy", "tripo", "elevenlabs"] {
            assert!(!supports_model_catalog(&Provider::Custom(name.into())));
        }
        assert!(supports_model_catalog(&Provider::OpenAi));
        assert!(supports_model_catalog(&Provider::Custom("example".into())));
    }

    #[test]
    fn image_to_video_filter_uses_catalog_capabilities() {
        let model = ModelInfo {
            id: "example".into(),
            name: None,
            created_at: None,
            context_length: None,
            max_completion_tokens: None,
            capabilities: Default::default(),
            categories: vec![ModelCategory::VideoGeneration, ModelCategory::ImageToVideo],
            pricing: Default::default(),
            image: Default::default(),
        };
        assert!(model_features(&model).contains(&"image-to-video"));
        assert!(model_features(&model).contains(&"video"));
        assert!(!model_features(&model).contains(&"image"));
    }
}
