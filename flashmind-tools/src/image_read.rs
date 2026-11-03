//! Image reading tool for vision models and text-only models.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use futures_util::StreamExt;
use rust_decimal::Decimal;
use serde_json::{Value, json};
use tracing::debug;

use flashmind_types::llm::{CompletionRequest, LlmProvider, ProviderRegistry, StreamEvent};
use flashmind_types::message::{ContentPart, Message};
use flashmind_types::model::{Model, ReasoningLevel, SamplingParams};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};

use crate::file_ops::{expand_tilde, resolve_path};

const OCR_SYSTEM_PROMPT: &str = "You are an image description assistant. Describe the image in detail, including any text, objects, colors, layout, and notable features. Be thorough but concise.";

#[derive(Debug, serde::Deserialize)]
struct ImageArgs {
    path: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    media_type: Option<String>,
}

fn mime_type(path: &str) -> &'static str {
    flashmind_types::llm::mime_from_extension(std::path::Path::new(path))
}

async fn find_vision_model(providers: &ProviderRegistry) -> Option<(Model, Arc<dyn LlmProvider>)> {
    for (provider_type, provider) in providers.iter() {
        let models = provider.list_models().await.unwrap_or_default();
        for m in &models {
            let model: Model = match format!("{provider_type}:{}", m.id).parse() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let caps = provider.capabilities(&model).await;
            if caps.images {
                return Some((model, Arc::clone(provider)));
            }
        }
    }
    None
}

async fn describe_image(
    provider: &dyn LlmProvider,
    model: &Model,
    media_type: &str,
    data: &str,
) -> anyhow::Result<String> {
    let image_part = ContentPart::Image {
        media_type: media_type.to_string(),
        data: data.to_string(),
    };

    let messages = vec![
        Message::system(OCR_SYSTEM_PROMPT),
        Message::user_with_parts("Please describe this image in detail.", vec![image_part]),
    ];

    let request = CompletionRequest {
        model: model.clone(),
        messages,
        tools: vec![],
        max_tokens: Some(4096),
        reasoning: ReasoningLevel::Off,
        sampling: SamplingParams {
            temperature: Some(Decimal::new(7, 1)),
            ..Default::default()
        },
        modalities: vec![],
        audio_config: None,
        image_config: None,
        user: None,
        provider_preferences: None,
        session_id: None,
    };

    let mut stream = provider.complete(request);
    let mut description = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(StreamEvent::ContentDelta(text)) => description.push_str(&text),
            Ok(StreamEvent::Finished(_)) => break,
            Err(e) => return Err(e),
            _ => {}
        }
    }

    Ok(description)
}

/// Image reading and OCR tool.
pub struct ImageReadTool {
    pub providers: ProviderRegistry,
    pub ocr_model: Option<Model>,
}

#[async_trait]
impl Tool for ImageReadTool {
    fn name(&self) -> &str {
        "image_read"
    }

    fn description(&self) -> &str {
        "Read an image file and return its content for analysis. Vision-capable models receive the image directly; other models get an OCR text description."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file to read"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model to use (provider/model format, e.g. 'ollama/llava'). Defaults to configured OCR model."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ImageArgs = ctx.parse_args(self.name())?;

        let path_str = expand_tilde(&args.path);
        let resolved_path = resolve_path(&path_str, ctx.working_dir);

        debug!(path = %resolved_path.display(), "image_read requested");

        let bytes = match tokio::fs::read(&resolved_path).await {
            Ok(b) => b,
            Err(e) => {
                debug!(path = %resolved_path.display(), error = %e, "image_read failed");
                return Ok(ToolResult::failure(
                    ctx.tool_call_id,
                    format!("Error reading image: {}", e),
                ));
            }
        };

        let mime = args
            .media_type
            .as_deref()
            .unwrap_or_else(|| mime_type(args.path.as_str()));
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);

        if let (Some(llm), Some(provider)) = (ctx.llm, &ctx.provider)
            && provider.capabilities(&llm.model).await.images
        {
            return Ok(ToolResult::success(
                ctx.tool_call_id,
                format!("Image loaded: {}", args.path),
            )
            .with_parts(vec![ContentPart::Image {
                media_type: mime.to_string(),
                data: base64_data,
            }])
            .with_files(vec![resolved_path]));
        }

        let (model, provider) = if let Some(model_str) = &args.model {
            let model = model_str.parse::<Model>()?;
            let provider = self
                .providers
                .get(&model.provider)
                .ok_or_else(|| anyhow::anyhow!("No provider available for model: {model_str}"))?;
            (model, Arc::clone(provider))
        } else if let Some(model) = &self.ocr_model {
            let provider = self
                .providers
                .get(&model.provider)
                .ok_or_else(|| anyhow::anyhow!("No provider available for model: {model}"))?;
            (model.clone(), Arc::clone(provider))
        } else {
            match find_vision_model(&self.providers).await {
                Some(found) => found,
                None => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        "No vision-capable model found. Configure a vision model or specify one with the model parameter.",
                    ));
                }
            }
        };
        if !provider.capabilities(&model).await.images {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Model {model} does not support images."),
            ));
        }
        let description = describe_image(provider.as_ref(), &model, mime, &base64_data).await?;
        if description.trim().is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "Vision model returned an empty description.",
            ));
        }
        Ok(ToolResult::success(ctx.tool_call_id, description).with_files(vec![resolved_path]))
    }

    fn max_output_bytes(&self) -> usize {
        usize::MAX
    }

    fn max_output_lines(&self) -> usize {
        usize::MAX
    }

    fn humanize(&self, args: &serde_json::Value) -> String {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Reading image {}", path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use flashmind_types::{
        AgentLlmConfig, CompletionStream, FinishReason, ModelCapabilities, Provider,
    };
    use tokio_util::sync::CancellationToken;

    struct VisionProvider;
    struct TextProvider;

    #[async_trait]
    impl LlmProvider for VisionProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::iter([
                Ok(StreamEvent::ContentDelta("A test image".into())),
                Ok(StreamEvent::Finished(FinishReason::Stop)),
            ]))
        }
    }

    #[async_trait]
    impl LlmProvider for TextProvider {
        fn name(&self) -> &str {
            "text"
        }
        fn provider(&self) -> Provider {
            Provider::Ollama
        }
        async fn capabilities(&self, _: &Model) -> ModelCapabilities {
            ModelCapabilities {
                images: false,
                ..Default::default()
            }
        }
        fn complete(&self, _: CompletionRequest) -> CompletionStream {
            Box::pin(futures_util::stream::empty())
        }
    }

    #[tokio::test]
    async fn text_model_receives_description_from_vision_model() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("sample.png"), b"image")
            .await
            .unwrap();
        let cwd = dir.path().to_path_buf();
        let token = CancellationToken::new();
        let llm = AgentLlmConfig::new("ollama:text".parse().unwrap());
        let mut providers = HashMap::new();
        providers.insert(
            Provider::Ollama,
            Arc::new(VisionProvider) as Arc<dyn LlmProvider>,
        );
        let tool = ImageReadTool {
            providers: Arc::new(providers),
            ocr_model: Some("ollama:vision".parse().unwrap()),
        };
        let context = ToolContext::new("call", json!({"path": "sample.png"}), Some(&cwd), &token)
            .with_llm(&llm)
            .with_provider(Arc::new(TextProvider));
        let result = tool.execute(context).await.unwrap();
        assert_eq!(result.output(), "A test image");
        assert!(
            result
                .content()
                .iter()
                .all(|part| matches!(part, ContentPart::Text { .. }))
        );
    }

    #[tokio::test]
    async fn vision_model_receives_image_content_instead_of_base64_text() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("sample.png"), b"image")
            .await
            .unwrap();
        let cwd = dir.path().to_path_buf();
        let token = CancellationToken::new();
        let llm = AgentLlmConfig::new("ollama:test".parse().unwrap());
        let tool = ImageReadTool {
            providers: Arc::new(Default::default()),
            ocr_model: None,
        };
        let context = ToolContext::new("call", json!({"path": "sample.png"}), Some(&cwd), &token)
            .with_llm(&llm)
            .with_provider(Arc::new(VisionProvider));
        let result = tool.execute(context).await.unwrap();
        assert!(result.is_success());
        assert!(!result.output().contains("aW1hZ2U="));
        assert!(
            result.content().iter().any(|part| matches!(part, ContentPart::Image { media_type, data } if media_type == "image/png" && data == "aW1hZ2U="))
        );
    }
}
