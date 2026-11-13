//! Explicit-key media registration. Provider defaults follow the tool examples.

use std::collections::HashMap;
use std::path::Path;

use flashmind_llm::OpenAiProvider;
use flashmind_llm::elevenlabs::ElevenLabsProvider;
use flashmind_llm::image_edit::{ImageEditBackend, ImageEditProvider};
use flashmind_types::model::AliasedModel;
use flashmind_types::{ApiKey, LlmProvider, Provider};

use super::*;
use crate::three_d::{GenerateThreeDTool, ThreeDModelCatalog};

fn model(provider: Provider, name: &str) -> Model {
    Model {
        provider,
        model: AliasedModel {
            name: name.into(),
            real_name: None,
        },
    }
}

pub(super) fn register(
    mut builder: ToolBuilder,
    keys: impl Fn(ApiKey) -> Option<String>,
    output_dir: PathBuf,
) -> ToolBuilder {
    if builder.offline {
        return builder;
    }
    // Snapshot once so every tool sees the same keys, including decision and 3D.
    let keys: HashMap<_, _> = [
        ApiKey::OpenRouter,
        ApiKey::OpenAi,
        ApiKey::Mistral,
        ApiKey::Zai,
        ApiKey::Gemini,
        ApiKey::Fal,
        ApiKey::ElevenLabs,
        ApiKey::Meshy,
        ApiKey::Tripo,
    ]
    .into_iter()
    .filter_map(|kind| {
        keys(kind)
            .filter(|key| !key.trim().is_empty())
            .map(|key| (kind, key))
    })
    .collect();
    let mut providers = HashMap::new();
    for (kind, provider) in [
        (ApiKey::OpenRouter, Provider::OpenRouter),
        (ApiKey::OpenAi, Provider::OpenAi),
        (ApiKey::Mistral, Provider::Mistral),
        (ApiKey::Zai, Provider::Custom("zai".into())),
        (ApiKey::Gemini, Provider::Custom("google".into())),
        (ApiKey::Fal, Provider::Custom("fal".into())),
        (ApiKey::ElevenLabs, Provider::Custom("elevenlabs".into())),
    ] {
        let Some(key) = keys.get(&kind) else { continue };
        let created: anyhow::Result<Arc<dyn LlmProvider>> = match kind {
            ApiKey::Gemini | ApiKey::Fal => ImageEditProvider::new(
                if kind == ApiKey::Gemini {
                    ImageEditBackend::Google
                } else {
                    ImageEditBackend::Fal
                },
                key.clone(),
                None,
                Default::default(),
            )
            .map(|p| Arc::new(p) as Arc<dyn LlmProvider>),
            ApiKey::Zai => OpenAiProvider::builder("https://api.z.ai/api/paas/v4")
                .provider(provider.clone())
                .api_key(key.clone())
                .build()
                .map(|p| Arc::new(p) as Arc<dyn LlmProvider>),
            ApiKey::ElevenLabs => {
                ElevenLabsProvider::new(key.clone()).map(|p| Arc::new(p) as Arc<dyn LlmProvider>)
            }
            // Explicit endpoint avoids application routing or environment configuration.
            ApiKey::OpenAi => OpenAiProvider::builder("https://api.openai.com/v1")
                .api_key(key.clone())
                .build()
                .map(|p| Arc::new(p) as Arc<dyn LlmProvider>),
            _ => flashmind_llm::create_provider(provider.clone(), Some(key)),
        };
        match created {
            Ok(client) => {
                providers.insert(provider, client);
            }
            Err(error) => tracing::warn!(%error, %kind, "Could not initialize media provider"),
        }
    }
    for (kind, provider) in [
        (ApiKey::Meshy, flashmind_llm::three_d::ThreeDProvider::Meshy),
        (ApiKey::Tripo, flashmind_llm::three_d::ThreeDProvider::Tripo),
    ] {
        if keys.contains_key(&kind) {
            let catalog = ThreeDModelCatalog(provider);
            providers.insert(catalog.provider(), Arc::new(catalog));
        }
    }
    builder.providers = Arc::new(providers);
    let providers = Arc::clone(&builder.providers);
    register_provider_tools(&mut builder, providers, &output_dir);
    if !builder.providers.is_empty() {
        builder = builder.models();
    }
    register_extra(builder, |kind| keys.get(&kind).cloned(), output_dir)
}

pub(super) fn register_extra(
    mut builder: ToolBuilder,
    keys: impl Fn(ApiKey) -> Option<String>,
    output_dir: PathBuf,
) -> ToolBuilder {
    if builder.offline {
        return builder;
    }
    match GenerateThreeDTool::from_keys(&keys, output_dir) {
        Ok(Some(tool)) => builder.registry.register(Arc::new(tool)),
        Ok(None) => {}
        Err(error) => tracing::warn!(%error, "Could not initialize 3D tool"),
    }
    builder.decision_with_keys(keys)
}

fn register_provider_tools(
    builder: &mut ToolBuilder,
    providers: ProviderRegistry,
    output_dir: &Path,
) {
    let choose = |candidates: &[(Provider, &str)]| {
        candidates
            .iter()
            .find(|(p, _)| providers.contains_key(p))
            .map(|(p, name)| model(p.clone(), name))
    };
    if let Some(default) = choose(&[
        (Provider::OpenAi, "dall-e-3"),
        (
            Provider::OpenRouter,
            "google/gemini-3.1-flash-image-preview",
        ),
        (Provider::Custom("zai".into()), "glm-image"),
    ]) {
        builder.registry.register(Arc::new(GenerateImageTool::new(
            Some(default),
            output_dir.to_path_buf(),
            providers.clone(),
        )));
    }
    if let Some(default) = choose(&[
        (
            Provider::OpenRouter,
            "google/gemini-3.1-flash-image-preview",
        ),
        (Provider::OpenAi, "gpt-image-2"),
        (Provider::Custom("google".into()), "gemini-nano-banana-2.1"),
        (Provider::Custom("fal".into()), "fal-ai/flux-2/edit"),
    ]) {
        builder.registry.register(Arc::new(ImageEditTool::new(
            Some(default),
            output_dir.to_path_buf(),
            providers.clone(),
        )));
    }
    if let Some(default) = choose(&[(Provider::OpenRouter, "google/veo-2.0-generate-001")]) {
        builder.registry.register(Arc::new(GenerateVideoTool::new(
            Some(default),
            output_dir.to_path_buf(),
            providers.clone(),
        )));
    }
    if let Some(default) = choose(&[
        (Provider::OpenAi, "tts-1"),
        (Provider::OpenRouter, "openai/tts-1"),
        (Provider::Mistral, "voxtral-mini-tts-3-0"),
        (
            Provider::Custom("elevenlabs".into()),
            "eleven_multilingual_v2",
        ),
    ]) {
        let voice = matches!(default.provider, Provider::OpenAi | Provider::OpenRouter)
            .then(|| "alloy".into());
        let config = AudioConfig {
            model: Some(default),
            voice,
        };
        builder.registry.register(Arc::new(TtsTool::new(
            config.clone(),
            output_dir.to_path_buf(),
            providers.clone(),
        )));
        builder
            .registry
            .register(Arc::new(ListVoicesTool::new(config, providers.clone())));
    }
    if let Some(default) = choose(&[
        (Provider::OpenAi, "whisper-1"),
        (Provider::OpenRouter, "openai/whisper-1"),
        (Provider::Mistral, "voxtral-mini-latest"),
    ]) {
        builder.registry.register(Arc::new(TranscribeTool::new(
            Some(default),
            providers.clone(),
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolContext;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn registers_only_tools_supported_by_supplied_keys() {
        let image = [
            "decision",
            "generate_image",
            "image_edit",
            "list_voices",
            "transcribe",
            "tts",
        ];
        for (key, expected) in [
            (
                ApiKey::OpenRouter,
                [image.as_slice(), &["generate_video"]].concat(),
            ),
            (ApiKey::OpenAi, image.to_vec()),
            (ApiKey::Mistral, vec!["list_voices", "transcribe", "tts"]),
            (ApiKey::Gemini, vec!["decision", "image_edit"]),
            (ApiKey::Fal, vec!["image_edit"]),
            (ApiKey::ElevenLabs, vec!["list_voices", "tts"]),
            (ApiKey::Meshy, vec!["generate_3d"]),
            (ApiKey::Tripo, vec!["generate_3d"]),
            (ApiKey::Anthropic, vec![]),
            (ApiKey::Moonshot, vec![]),
            (ApiKey::Zai, vec!["generate_image"]),
            (ApiKey::KimiCode, vec![]),
            (ApiKey::Firecrawl, vec![]),
        ] {
            let registry = ToolBuilder::new()
                .media(
                    |candidate| (candidate == key).then(|| "fake-key".into()),
                    PathBuf::new(),
                )
                .build();
            let mut names = registry.list();
            names.sort();
            let mut expected = expected;
            if !expected.is_empty() {
                expected.push("list_models");
            }
            expected.sort();
            assert_eq!(names, expected, "{key}");
        }
        for key in [None, Some(""), Some("  ")] {
            assert!(
                ToolBuilder::new()
                    .media(|_| key.map(str::to_owned), PathBuf::new())
                    .build()
                    .list()
                    .is_empty()
            );
        }
        assert!(
            ToolBuilder::new()
                .with_offline(true)
                .media(|_| panic!("offline must not read keys"), PathBuf::new())
                .build()
                .list()
                .is_empty()
        );
        let registry = ToolBuilder::new()
            .media(|_| Some("fake-key".into()), PathBuf::new())
            .build();
        assert_eq!(registry.list().len(), 9);
    }

    #[tokio::test]
    async fn media_retains_catalogs_for_helper_and_models_registration() {
        for (key, provider) in [(ApiKey::Meshy, "meshy"), (ApiKey::Tripo, "tripo")] {
            let builder = ToolBuilder::new().media(
                |candidate| (candidate == key).then(|| "fake-key".into()),
                PathBuf::new(),
            );
            let providers = builder.providers().clone();
            let registry = builder.models().build();
            let models = crate::list_models::list_models(
                &providers,
                Some(flashmind_types::ModelCategory::ThreeD),
                None,
            )
            .await
            .unwrap();
            assert_eq!(models.len(), 4);
            assert!(
                models
                    .iter()
                    .all(|model| model.id.starts_with(&format!("{provider}:")))
            );
            assert!(
                models
                    .iter()
                    .any(|model| model.id == format!("{provider}:latest"))
            );
            let cancel = CancellationToken::new();
            let result = registry
                .get("list_models")
                .unwrap()
                .execute(ToolContext::new(
                    "call",
                    json!({"category": "3d"}),
                    None,
                    &cancel,
                ))
                .await
                .unwrap();
            assert!(result.is_success());
            for model in models {
                assert!(result.output().contains(&model.id));
            }
            assert!(!result.output().contains(&format!("{provider}:{provider}:")));
        }
    }

    struct FakeProvider;

    fn stream(event: flashmind_types::StreamEvent) -> flashmind_types::CompletionStream {
        Box::pin(futures_util::stream::iter([Ok(event)]))
    }

    fn image() -> flashmind_types::CompletionStream {
        stream(flashmind_types::StreamEvent::FileAttachment {
            filename: "test.png".into(),
            media_type: "image/png".into(),
            data: vec![1, 2, 3],
        })
    }

    #[async_trait::async_trait]
    impl LlmProvider for FakeProvider {
        fn name(&self) -> &str {
            "fake"
        }
        fn provider(&self) -> Provider {
            Provider::OpenRouter
        }
        fn complete(
            &self,
            request: flashmind_types::CompletionRequest,
        ) -> flashmind_types::CompletionStream {
            assert!(request.model.name().contains("gemini"));
            image()
        }
        fn edit_image(
            &self,
            request: flashmind_types::ImageEditRequest,
        ) -> flashmind_types::CompletionStream {
            assert_eq!(request.images, ["https://example.com/reference.png"]);
            image()
        }
        fn generate_image(
            &self,
            request: flashmind_types::ImageGenRequest,
        ) -> flashmind_types::CompletionStream {
            assert!(!request.model.is_empty());
            image()
        }
        fn generate_video(
            &self,
            request: flashmind_types::VideoGenRequest,
        ) -> flashmind_types::CompletionStream {
            assert!(request.model.contains("veo"));
            stream(flashmind_types::StreamEvent::FileAttachment {
                filename: "test.mp4".into(),
                media_type: "video/mp4".into(),
                data: vec![4, 5, 6],
            })
        }
        fn text_to_speech(
            &self,
            request: flashmind_types::TtsRequest,
        ) -> flashmind_types::CompletionStream {
            assert_eq!(request.voice, "alloy");
            stream(flashmind_types::StreamEvent::AudioDelta {
                data: vec![7, 8, 9],
                format: "mp3".into(),
            })
        }
        fn transcribe(
            &self,
            request: flashmind_types::SttRequest,
        ) -> flashmind_types::CompletionStream {
            assert_eq!(request.audio, [10, 11, 12]);
            stream(flashmind_types::StreamEvent::ContentDelta(
                "transcribed".into(),
            ))
        }
        async fn list_voices(&self, _: &str) -> Option<Vec<flashmind_types::Voice>> {
            Some(vec![flashmind_types::Voice {
                id: "alloy".into(),
                name: "alloy".into(),
            }])
        }
    }

    #[tokio::test]
    async fn media_tools_execute_without_context_llm_or_provider() {
        let dir = tempfile::tempdir().unwrap();
        let working_dir = dir.path().to_path_buf();
        std::fs::write(dir.path().join("input.mp3"), [10, 11, 12]).unwrap();
        let cancel = CancellationToken::new();
        for provider in [
            Provider::OpenRouter,
            Provider::OpenAi,
            Provider::Custom("google".into()),
            Provider::Custom("fal".into()),
        ] {
            let mut builder = ToolBuilder::new();
            let providers: ProviderRegistry =
                Arc::new([(provider, Arc::new(FakeProvider) as Arc<dyn LlmProvider>)].into());
            register_provider_tools(&mut builder, providers, dir.path());
            let registry = builder.build();
            for (name, args) in [
                ("generate_image", json!({"prompt": "chair"})),
                (
                    "image_edit",
                    json!({"prompt": "chair", "reference_images": ["https://example.com/reference.png"]}),
                ),
                ("generate_video", json!({"description": "chair"})),
                ("tts", json!({"text": "hello"})),
                ("transcribe", json!({"file_path": "input.mp3"})),
                ("list_voices", json!({})),
            ] {
                let Some(tool) = registry.get(name) else {
                    continue;
                };
                let result = tool
                    .execute(ToolContext::new("call", args, Some(&working_dir), &cancel))
                    .await
                    .unwrap();
                assert!(result.is_success(), "{name}: {}", result.output());
                assert_eq!(result.tool_call_id(), "call");
                if name == "transcribe" {
                    assert_eq!(result.output(), "transcribed");
                }
            }
        }
    }
}
