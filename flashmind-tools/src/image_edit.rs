//! Image edit tool — generates or edits images via multimodal chat completion models.

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::image_gen::{describe_saved_images, save_output};
use crate::utils::truncate_utf8;
use flashmind_types::llm::{ImageGenConfig, Modality};
use flashmind_types::model::Model;
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::{CompletionRequest, StreamEvent};

pub use flashmind_types::llm::ProviderRegistry;

#[derive(serde::Deserialize)]
struct ImageEditArgs {
    prompt: String,
    model: Option<Model>,
    aspect_ratio: Option<String>,
    size: Option<String>,
    #[serde(default)]
    reference_images: Vec<String>, // URLs or data URIs → injected as ContentPart::ImageUrl
}

/// Multimodal image editing tool.
pub struct ImageEditTool {
    default_model: Option<Model>,
    output_dir: PathBuf,
    providers: ProviderRegistry,
}

impl ImageEditTool {
    pub fn new(
        default_model: Option<Model>,
        output_dir: PathBuf,
        providers: ProviderRegistry,
    ) -> Self {
        Self {
            default_model,
            output_dir,
            providers,
        }
    }
}

#[async_trait]
impl Tool for ImageEditTool {
    fn name(&self) -> &str {
        "image_edit"
    }

    fn description(&self) -> &str {
        "Edit or generate images using a multimodal chat model. Supports reference images for style guidance. After generating, ALWAYS include the file marker from the result in your response so the image is sent to the user."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate"
                },
                "model": {
                    "type": "string",
                    "description": "Image generation model (e.g., 'openrouter:google/gemini-3.1-flash-image-preview'). Uses config default if not specified."
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio (e.g., '1:1', '16:9', '9:16')"
                },
                "size": {
                    "type": "string",
                    "description": "Image size (e.g., '1K', '2K', '4K')"
                },
                "reference_images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "URLs of reference images for style/quality guidance (max 4, $0.20 each)"
                }
            },
            "required": ["prompt"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(120)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: ImageEditArgs = ctx.parse_args(self.name())?;

        let model = args.model.or_else(|| self.default_model.clone());
        let Some(model) = model else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No image generation model configured. Pass a model argument (e.g., 'openrouter:google/gemini-3.1-flash-image-preview').",
            ));
        };

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        let image_config = ImageGenConfig {
            aspect_ratio: args.aspect_ratio,
            size: args.size,
        };

        let message = if args.reference_images.is_empty() {
            flashmind_types::Message::user(&args.prompt)
        } else {
            use flashmind_types::message::ContentPart;
            let parts = args
                .reference_images
                .into_iter()
                .map(|url| ContentPart::ImageUrl { url })
                .collect();
            flashmind_types::Message::user_with_parts(&args.prompt, parts)
        };

        let request = CompletionRequest {
            model: model.clone(),
            messages: vec![message],
            tools: vec![],
            max_tokens: Some(4096),
            reasoning: flashmind_types::model::ReasoningLevel::Off,
            sampling: Default::default(),
            modalities: vec![Modality::Image, Modality::Text],
            audio_config: None,
            image_config: Some(image_config),
            user: None,
            provider_preferences: None,
            session_id: None,
        };

        let mut stream = provider.complete(request);

        use futures::StreamExt;
        let mut saved_files = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::FileAttachment {
                    filename,
                    media_type,
                    data,
                }) => {
                    let ext = media_type.split('/').next_back().unwrap_or("png");
                    let output_path = save_output(&self.output_dir, "image", ext, &data)?;

                    let size_kb = data.len() / 1024;
                    tracing::debug!(
                        path = %output_path.display(),
                        size_kb,
                        media_type = %media_type,
                        orig_filename = %filename,
                        "Image saved"
                    );
                    saved_files.push((output_path, size_kb));
                }
                Ok(StreamEvent::Finished(_)) => break,
                Err(e) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Image generation failed: {e}"),
                    ));
                }
                _ => {}
            }
        }

        if saved_files.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No images were generated. The model may not support image generation.",
            ));
        }

        Ok(ToolResult::success(
            ctx.tool_call_id,
            describe_saved_images(&saved_files),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        format!("Generating image: {}", truncate_utf8(prompt, 60))
    }
}
