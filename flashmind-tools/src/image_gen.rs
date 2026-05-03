//! Image generation tool — generates images via dedicated image APIs (e.g. DALL-E).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::utils::truncate_utf8;
use flashmind_types::llm::LlmProvider;
use flashmind_types::model::{Model, Provider};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::{ImageGenRequest, StreamEvent};

pub type ProviderRegistry = Arc<std::collections::HashMap<Provider, Arc<dyn LlmProvider>>>;

#[derive(serde::Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    model: Option<Model>,
    size: Option<String>,
    quality: Option<String>,
    style: Option<String>,
    n: Option<u32>,
}

/// Dedicated image generation tool (DALL-E style).
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
        "Generate an image from a text prompt using a dedicated image generation API (e.g. DALL-E). After generating, ALWAYS include the file marker from the result in your response so the image is sent to the user."
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
                    "description": "Image generation model (e.g., 'openai:dall-e-3')"
                },
                "size": {
                    "type": "string",
                    "description": "Image size (e.g., '1024x1024', '1792x1024', '1024x1792')"
                },
                "quality": {
                    "type": "string",
                    "description": "Image quality ('standard' or 'hd')"
                },
                "style": {
                    "type": "string",
                    "description": "Image style ('natural' or 'vivid')"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of images to generate (default 1)"
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
                "No image generation model configured. Pass a model argument (e.g., 'openai:dall-e-3').",
            ));
        };

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        let request = ImageGenRequest {
            model: model.name().to_string(),
            prompt: args.prompt,
            size: args.size,
            aspect_ratio: None,
            quality: args.quality,
            style: args.style,
            n: args.n,
        };

        let mut stream = provider.generate_image(request);

        use futures::StreamExt;
        let mut saved_files = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::FileAttachment {
                    media_type, data, ..
                }) => {
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis();
                    let ext = media_type.split('/').next_back().unwrap_or("png");
                    let filename = format!("image_{timestamp}.{ext}");
                    let output_path = self.output_dir.join(&filename);

                    if let Some(parent) = output_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }

                    std::fs::write(&output_path, &data)?;

                    let size_kb = data.len() / 1024;
                    tracing::debug!(
                        path = %output_path.display(),
                        size_kb,
                        media_type = %media_type,
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
                "No images were generated.",
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
