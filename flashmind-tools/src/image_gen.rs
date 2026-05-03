//! Image generation tool — generates images via LLM provider completions with image modality.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::utils::truncate_utf8;
use flashmind_types::llm::{ImageGenConfig, LlmProvider, Modality};
use flashmind_types::model::{Model, Provider};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::{CompletionRequest, StreamEvent};

pub type ProviderRegistry = Arc<std::collections::HashMap<Provider, Arc<dyn LlmProvider>>>;

#[derive(serde::Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    model: Option<Model>,
    aspect_ratio: Option<String>,
    size: Option<String>,
    #[serde(default)]
    reference_images: Vec<String>,
}

pub struct GenerateImageTool {
    default_model: Option<Model>,
    output_dir: PathBuf,
    providers: ProviderRegistry,
}

impl GenerateImageTool {
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
impl Tool for GenerateImageTool {
    fn name(&self) -> &str {
        "generate_image"
    }

    fn description(&self) -> &str {
        "Generate an image from a text prompt. After generating, ALWAYS include the file marker from the result in your response so the image is sent to the user."
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
        let args: GenerateImageArgs = ctx.parse_args(self.name())?;

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
            reference_images: args.reference_images,
        };

        let request = CompletionRequest {
            model: model.clone(),
            messages: vec![flashmind_types::Message::user(&args.prompt)],
            tools: vec![],
            temperature: "0.7".parse().unwrap(),
            max_tokens: Some(4096),
            reasoning: flashmind_types::model::ReasoningLevel::Off,
            sampling: Default::default(),
            modalities: Some(vec![Modality::Image, Modality::Text]),
            audio_config: None,
            image_config: Some(image_config),
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
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis();
                    let ext = media_type.split('/').next_back().unwrap_or("png");
                    let out_filename = format!("image_{timestamp}.{ext}");
                    let output_path = self.output_dir.join(&out_filename);

                    if let Some(parent) = output_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }

                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&data)
                        .map_err(|e| anyhow::anyhow!("Failed to decode image data: {e}"))?;

                    std::fs::write(&output_path, &bytes)?;

                    let size_kb = bytes.len() / 1024;
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

        let descriptions: Vec<String> = saved_files
            .iter()
            .map(|(path, size_kb)| {
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
                format!(
                    "{} ({}KB) — include [{}]({}) in your response to send it",
                    path.display(),
                    size_kb,
                    fname,
                    path.display(),
                )
            })
            .collect();

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Generated {} image(s):\n{}",
                saved_files.len(),
                descriptions.join("\n")
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        format!("Generating image: {}", truncate_utf8(prompt, 60))
    }
}
