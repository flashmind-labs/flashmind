//! Attach local files to the current model's conversation.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use flashmind_types::ContentPart;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::file_ops::{expand_tilde, resolve_path};
use crate::image_read::ImageReadTool;
use crate::protected::ProtectedPaths;

#[derive(Deserialize)]
struct AttachArgs {
    path: String,
    media_type: Option<String>,
}

/// Attach a local file as multimodal input to the active model.
pub struct AttachTool {
    pub protected: Arc<ProtectedPaths>,
    /// Vision provider used when the active model cannot accept images.
    pub image_reader: ImageReadTool,
}

fn media_type(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        "mov" => Some("video/quicktime"),
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "flac" => Some("audio/flac"),
        "ogg" => Some("audio/ogg"),
        "m4a" => Some("audio/mp4"),
        _ => None,
    }
}

#[async_trait]
impl Tool for AttachTool {
    fn name(&self) -> &str {
        "attach"
    }

    fn description(&self) -> &str {
        "Attach a local PDF, image, audio, video, or other document for the current model to examine. If the model cannot accept images, use a vision model to describe them. Other formats depend on model and provider support."
    }

    fn parameters(&self) -> Value {
        json!({"type":"object", "properties": {
            "path": {"type":"string", "description":"Path to the local file"},
            "media_type": {"type":"string", "description":"MIME type, required for unrecognized file extensions"},
            "model": {"type":"string", "description":"Optional vision model for image OCR fallback, in provider:model format"}
        }, "required":["path"]})
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: AttachArgs = ctx.parse_args(self.name())?;
        let path = resolve_path(&expand_tilde(&args.path), ctx.working_dir);
        if self.protected.is_read_protected(&path) {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "File is protected from reading.",
            ));
        }
        let mime = match args.media_type.as_deref().or_else(|| media_type(&path)) {
            Some(mime) if mime.contains('/') => mime,
            _ => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    "Unknown file type. Specify its MIME type using media_type.",
                ));
            }
        };
        let (Some(llm), Some(provider)) = (ctx.llm, &ctx.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Current model capabilities are unavailable.",
            ));
        };
        let caps = provider.capabilities(&llm.model).await;
        let supported = if mime.starts_with("image/") {
            caps.images
        } else if mime.starts_with("video/") {
            caps.video
        } else if mime.starts_with("audio/") {
            caps.audio
        } else {
            caps.documents
        };
        if !supported && mime.starts_with("image/") {
            return self.image_reader.execute(ctx).await;
        }
        if !supported {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!(
                    "Model {} does not report support for {mime} attachments.",
                    llm.model
                ),
            ));
        }
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) => {
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error reading file: {error}"),
                ));
            }
        };
        let data = STANDARD.encode(bytes);
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let media_type = mime.to_owned();
        let part = if mime.starts_with("image/") {
            ContentPart::Image { media_type, data }
        } else if mime.starts_with("video/") {
            ContentPart::Video {
                media_type,
                filename,
                data,
            }
        } else if mime.starts_with("audio/") {
            ContentPart::Audio {
                media_type,
                filename,
                data,
            }
        } else {
            ContentPart::Document {
                media_type,
                filename,
                data,
            }
        };
        Ok(
            ToolResult::success(ctx.tool_call_id, format!("Attached file: {}", args.path))
                .with_parts(vec![part]),
        )
    }

    fn max_output_bytes(&self) -> usize {
        usize::MAX
    }
    fn max_output_lines(&self) -> usize {
        usize::MAX
    }
    fn humanize(&self, args: &Value) -> String {
        format!(
            "Attaching {}",
            args.get("path").and_then(Value::as_str).unwrap_or("?")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::{
        AgentLlmConfig, CompletionRequest, CompletionStream, LlmProvider, Model, ModelCapabilities,
        Provider,
    };
    use tokio_util::sync::CancellationToken;

    struct TestProvider(ModelCapabilities);

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        async fn capabilities(&self, _: &Model) -> ModelCapabilities {
            self.0
        }
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::empty())
        }
    }

    struct VisionProvider;

    #[async_trait]
    impl LlmProvider for VisionProvider {
        fn name(&self) -> &str {
            "vision"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        async fn capabilities(&self, _: &Model) -> ModelCapabilities {
            ModelCapabilities {
                images: true,
                ..Default::default()
            }
        }
        fn complete(&self, request: CompletionRequest) -> CompletionStream {
            assert!(request.messages.iter().flat_map(|message| message.parts.iter().flatten()).any(|part| matches!(part, ContentPart::Image { media_type, data } if media_type == "image/png" && data == "aW1hZ2U=")));
            Box::pin(futures_util::stream::iter([Ok(
                flashmind_types::StreamEvent::ContentDelta("Image text".into()),
            )]))
        }
    }

    #[tokio::test]
    async fn image_falls_back_to_vision_model_for_text_only_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        tokio::fs::write(dir.path().join("picture.unknown"), b"image")
            .await
            .unwrap();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            Provider::Ollama,
            Arc::new(VisionProvider) as Arc<dyn LlmProvider>,
        );
        let tool = AttachTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            image_reader: ImageReadTool {
                providers: Arc::new(providers),
                ocr_model: Some("ollama:vision".parse().unwrap()),
            },
        };
        let cwd = dir.path().to_path_buf();
        let token = CancellationToken::new();
        let llm = AgentLlmConfig::new("ollama:text".parse().unwrap());
        let context = ToolContext::new(
            "call",
            json!({"path":"picture.unknown", "media_type":"image/png"}),
            Some(&cwd),
            &token,
        )
        .with_llm(&llm)
        .with_provider(Arc::new(TestProvider(ModelCapabilities::default())));
        let result = tool.execute(context).await.unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert_eq!(result.output(), "Image text");
        assert!(
            result
                .content()
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
        );
    }

    #[tokio::test]
    async fn attaches_supported_files_and_rejects_unsupported_media() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        tokio::fs::write(dir.path().join("report.pdf"), b"pdf")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("clip.mp4"), b"video")
            .await
            .unwrap();
        let tool = AttachTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            image_reader: ImageReadTool {
                providers: Arc::new(Default::default()),
                ocr_model: None,
            },
        };
        let cwd = dir.path().to_path_buf();
        let token = CancellationToken::new();
        let llm = AgentLlmConfig::new("ollama:test".parse().unwrap());
        let provider = Arc::new(TestProvider(ModelCapabilities {
            documents: true,
            ..Default::default()
        }));
        let context = ToolContext::new("call", json!({"path":"report.pdf"}), Some(&cwd), &token)
            .with_llm(&llm)
            .with_provider(provider.clone());
        let result = tool.execute(context).await.unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert!(result.content().iter().any(|part| matches!(part, ContentPart::Document { media_type, filename, data } if media_type == "application/pdf" && filename == "report.pdf" && data == "cGRm")));
        let context = ToolContext::new("call", json!({"path":"clip.mp4"}), Some(&cwd), &token)
            .with_llm(&llm)
            .with_provider(provider);
        let result = tool.execute(context).await.unwrap();
        assert!(!result.is_success());
        assert!(result.output().contains("does not report support"));
        assert!(
            result
                .content()
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
        );
    }

    #[tokio::test]
    async fn video_is_attached_for_capable_models() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        tokio::fs::write(dir.path().join("clip.mp4"), b"video")
            .await
            .unwrap();
        let tool = AttachTool {
            protected: Arc::new(ProtectedPaths::new(dir.path())),
            image_reader: ImageReadTool {
                providers: Arc::new(Default::default()),
                ocr_model: None,
            },
        };
        let cwd = dir.path().to_path_buf();
        let token = CancellationToken::new();
        let llm = AgentLlmConfig::new("ollama:test".parse().unwrap());
        let provider = Arc::new(TestProvider(ModelCapabilities {
            video: true,
            ..Default::default()
        }));
        let context = ToolContext::new("call", json!({"path":"clip.mp4"}), Some(&cwd), &token)
            .with_llm(&llm)
            .with_provider(provider);
        let result = tool.execute(context).await.unwrap();
        assert!(result.is_success(), "{}", result.output());
        assert!(result.content().iter().any(|part| matches!(part, ContentPart::Video { media_type, data, .. } if media_type == "video/mp4" && data == "dmlkZW8=")));
    }
}
