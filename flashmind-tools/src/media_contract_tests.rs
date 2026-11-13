//! Parameter contracts at the media tools' provider boundary.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use flashmind_types::{
    CompletionRequest, CompletionStream, ImageEditRequest, ImageGenRequest, LlmProvider, Provider,
    SttRequest, TtsRequest, VideoGenRequest, Voice,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::audio::{AudioConfig, ListVoicesTool, TranscribeTool, TtsTool};
use crate::image_edit::ImageEditTool;
use crate::image_gen::GenerateImageTool;
use crate::video_gen::GenerateVideoTool;
use crate::{Tool, ToolContext};

#[derive(Default)]
struct Requests(Mutex<Vec<Value>>);

struct CaptureProvider {
    provider: Provider,
    requests: Arc<Requests>,
}

impl CaptureProvider {
    fn record(&self, operation: &str, body: Value) -> CompletionStream {
        self.requests.0.lock().unwrap().push(json!({
            "provider": self.provider.to_string(), "operation": operation, "body": body,
        }));
        Box::pin(futures_util::stream::empty())
    }
}

#[async_trait]
impl LlmProvider for CaptureProvider {
    fn name(&self) -> &str {
        "capture"
    }

    fn provider(&self) -> Provider {
        self.provider.clone()
    }

    fn complete(&self, request: CompletionRequest) -> CompletionStream {
        self.record(
            "complete",
            json!({
                "model": request.model.name(),
                "prompt": request.messages[0].content,
                "reference_images": request.messages[0].parts,
                "aspect_ratio": request.image_config.as_ref().unwrap().aspect_ratio,
                "size": request.image_config.as_ref().unwrap().size,
            }),
        )
    }

    fn generate_image(&self, request: ImageGenRequest) -> CompletionStream {
        self.record(
            "generate_image",
            json!({
                "model": request.model, "prompt": request.prompt, "size": request.size,
                "quality": request.quality, "style": request.style, "n": request.n,
            }),
        )
    }

    fn edit_image(&self, request: ImageEditRequest) -> CompletionStream {
        self.record(
            "edit_image",
            json!({
                "model": request.model, "prompt": request.prompt, "size": request.resolution,
                "aspect_ratio": request.aspect_ratio, "reference_images": request.images,
            }),
        )
    }

    fn generate_video(&self, request: VideoGenRequest) -> CompletionStream {
        self.record(
            "generate_video",
            json!({
                "model": request.model, "description": request.description,
                "resolution": request.resolution, "aspect_ratio": request.aspect_ratio,
                "duration": request.duration, "generate_audio": request.generate_audio,
                "frame_images": request.frame_images.into_iter().map(|frame| json!({
                    "url": frame.url, "frame_type": frame.frame_type,
                })).collect::<Vec<_>>(),
                "input_references": request.input_references,
            }),
        )
    }

    fn text_to_speech(&self, request: TtsRequest) -> CompletionStream {
        self.record(
            "tts",
            json!({
                "model": request.model, "text": request.input, "voice": request.voice,
                "output_format": request.response_format,
            }),
        )
    }

    fn transcribe(&self, request: SttRequest) -> CompletionStream {
        self.record(
            "transcribe",
            json!({
                "model": request.model, "audio": request.audio, "media_type": request.media_type,
                "language": request.language,
            }),
        )
    }

    async fn list_voices(&self, model: &str) -> Option<Vec<Voice>> {
        drop(self.record("list_voices", json!({"model": model})));
        Some(vec![Voice {
            id: "example".into(),
            name: "Example".into(),
        }])
    }
}

fn providers(requests: &Arc<Requests>) -> flashmind_types::llm::ProviderRegistry {
    Arc::new(
        [
            Provider::OpenAi,
            Provider::OpenRouter,
            Provider::Custom("google".into()),
            Provider::Custom("fal".into()),
        ]
        .into_iter()
        .map(|provider| {
            (
                provider.clone(),
                Arc::new(CaptureProvider {
                    provider,
                    requests: requests.clone(),
                }) as Arc<dyn LlmProvider>,
            )
        })
        .collect::<HashMap<_, _>>(),
    )
}

async fn assert_contract(tool: &dyn Tool, args: Value, requests: &Requests, expected: Value) {
    let schema = tool.parameters();
    assert_eq!(schema["additionalProperties"], false, "{}", tool.name());
    assert_eq!(
        schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        args.as_object().unwrap().keys().collect::<Vec<_>>(),
        "provide every schema property for {}",
        tool.name(),
    );
    let cancel = CancellationToken::new();
    tool.execute(ToolContext::new("contract", args.clone(), None, &cancel))
        .await
        .unwrap();
    assert_eq!(*requests.0.lock().unwrap(), vec![expected]);
    requests.0.lock().unwrap().clear();
    let mut unknown = args;
    unknown["unsupported_option"] = json!(true);
    let error = tool
        .execute(ToolContext::new("contract", unknown, None, &cancel))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("unsupported_option"),
        "{error:#}"
    );
    assert!(requests.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn generate_image_forwards_every_declared_parameter_and_rejects_extras() {
    let requests = Arc::new(Requests::default());
    let dir = tempfile::tempdir().unwrap();
    let tool = GenerateImageTool::new(
        Some("openai:default".parse().unwrap()),
        dir.path().into(),
        providers(&requests),
    );
    let args = json!({
        "model": "openrouter:selected", "prompt": "Paint a chair", "size": "1024x1024",
        "quality": "hd", "style": "vivid", "n": 2,
    });
    let mut body = args.clone();
    body["model"] = json!("selected");
    assert_contract(
        &tool,
        args,
        &requests,
        json!({
            "provider": "openrouter", "operation": "generate_image", "body": body,
        }),
    )
    .await;
}

#[tokio::test]
async fn image_edit_forwards_every_declared_parameter_on_native_and_chat_paths() {
    let requests = Arc::new(Requests::default());
    let dir = tempfile::tempdir().unwrap();
    let tool = ImageEditTool::new(
        Some("openai:default".parse().unwrap()),
        dir.path().into(),
        providers(&requests),
    );
    for provider in ["openai", "google", "fal", "openrouter"] {
        let args = json!({
            "model": format!("{provider}:selected"), "prompt": "Paint it blue",
            "size": "2K", "aspect_ratio": "16:9",
            "reference_images": ["https://example.com/reference.png"],
        });
        let mut body = args.clone();
        body["model"] = json!("selected");
        if provider == "openrouter" {
            body["reference_images"] =
                json!([{"type": "image_url", "url": "https://example.com/reference.png"}]);
        }
        assert_contract(&tool, args, &requests, json!({
            "provider": provider, "operation": if provider == "openrouter" { "complete" } else { "edit_image" },
            "body": body,
        })).await;
        requests.0.lock().unwrap().clear();
    }
}

#[tokio::test]
async fn generate_video_forwards_every_declared_parameter_and_rejects_extras() {
    let requests = Arc::new(Requests::default());
    let dir = tempfile::tempdir().unwrap();
    let tool = GenerateVideoTool::new(
        Some("openai:default".parse().unwrap()),
        dir.path().into(),
        providers(&requests),
    );
    let args = json!({
        "model": "openrouter:selected", "description": "A chair rotates",
        "resolution": "1080p", "aspect_ratio": "16:9", "duration": 7,
        "generate_audio": false,
        "frame_images": [{"url": "https://example.com/first.png", "frame_type": "first_frame"},
            {"url": "https://example.com/last.png", "frame_type": "last_frame"}],
        "input_references": ["https://example.com/style.png"],
    });
    let mut body = args.clone();
    body["model"] = json!("selected");
    assert_contract(
        &tool,
        args.clone(),
        &requests,
        json!({
            "provider": "openrouter", "operation": "generate_video", "body": body,
        }),
    )
    .await;
    let mut unknown = args;
    unknown["frame_images"][0]["unsupported_option"] = json!(true);
    let cancel = CancellationToken::new();
    let error = tool
        .execute(ToolContext::new("contract", unknown, None, &cancel))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("unsupported_option"),
        "{error:#}"
    );
    assert!(requests.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn tts_forwards_every_declared_parameter_and_rejects_extras() {
    let requests = Arc::new(Requests::default());
    let dir = tempfile::tempdir().unwrap();
    let tool = TtsTool::new(
        AudioConfig {
            model: Some("openai:default".parse().unwrap()),
            voice: Some("default-voice".into()),
        },
        dir.path().into(),
        providers(&requests),
    );
    for format in ["mp3", "wav", "opus", "aac", "flac"] {
        let args = json!({"model": "openrouter:selected", "text": "Hello", "voice": "chosen-voice", "output_format": format});
        let mut body = args.clone();
        body["model"] = json!("selected");
        assert_contract(
            &tool,
            args,
            &requests,
            json!({
                "provider": "openrouter", "operation": "tts", "body": body,
            }),
        )
        .await;
        requests.0.lock().unwrap().clear();
    }
}

#[tokio::test]
async fn transcribe_forwards_every_declared_parameter_and_rejects_extras() {
    let requests = Arc::new(Requests::default());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.flac");
    std::fs::write(&path, [10, 20, 30]).unwrap();
    let tool = TranscribeTool::new(
        Some("openai:default".parse().unwrap()),
        providers(&requests),
    );
    assert_contract(&tool, json!({"model": "openrouter:selected", "file_path": path, "language": "fr"}), &requests, json!({
        "provider": "openrouter", "operation": "transcribe", "body": {
            "model": "selected", "audio": [10, 20, 30], "media_type": "audio/flac", "language": "fr",
        },
    })).await;
}

#[tokio::test]
async fn list_voices_forwards_model_and_rejects_extras() {
    let requests = Arc::new(Requests::default());
    let tool = ListVoicesTool::new(
        AudioConfig {
            model: Some("openai:default".parse().unwrap()),
            voice: None,
        },
        providers(&requests),
    );
    assert_contract(
        &tool,
        json!({"model": "openrouter:selected"}),
        &requests,
        json!({
            "provider": "openrouter", "operation": "list_voices", "body": {"model": "selected"},
        }),
    )
    .await;
}
