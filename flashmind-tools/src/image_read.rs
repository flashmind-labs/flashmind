//! Image reading tool — returns base64 data URL or OCR text description.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use serde_json::{Value, json};
use tracing::{debug, warn};

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
}

fn mime_type(path: &str) -> &'static str {
    flashmind_types::llm::mime_from_extension(std::path::Path::new(path))
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
            temperature: Some(rust_decimal_macros::dec!(0.7)),
            ..Default::default()
        },
        modalities: vec![],
        audio_config: None,
        image_config: None,
    };

    let mut stream = provider.complete(request);
    let mut description = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(StreamEvent::ContentDelta(text)) => description.push_str(&text),
            Ok(StreamEvent::Finished(_)) => break,
            Err(e) => {
                warn!(error = %e, "OCR stream error");
                break;
            }
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

        if let Err(r) = ctx.check_absolute_path(&args.path) {
            return Ok(r);
        }

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

        let mime = mime_type(args.path.as_str());
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let (model, provider) = match &args.model {
            Some(model_str) => {
                let model = match model_str.parse::<Model>() {
                    Ok(m) => m,
                    Err(_) => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!(
                                "Invalid model format: {}. Use provider/model (e.g. ollama/llava)",
                                model_str
                            ),
                        ));
                    }
                };
                let provider = match self.providers.get(&model.provider) {
                    Some(p) => Arc::clone(p),
                    None => {
                        return Ok(ToolResult::failure(
                            ctx.tool_call_id,
                            format!("No provider available for model: {}", model_str),
                        ));
                    }
                };
                (model, provider)
            }
            None => match &self.ocr_model {
                Some(ocr_model) => {
                    let provider = match self.providers.get(&ocr_model.provider) {
                        Some(p) => Arc::clone(p),
                        None => {
                            return Ok(ToolResult::success(
                                ctx.tool_call_id,
                                format!("data:{};base64,{}", mime, base64_data),
                            ));
                        }
                    };
                    (ocr_model.clone(), provider)
                }
                None => {
                    return Ok(ToolResult::success(
                        ctx.tool_call_id,
                        format!("data:{};base64,{}", mime, base64_data),
                    ));
                }
            },
        };

        let has_vision = provider.capabilities(&model).await.images;
        if has_vision {
            let data_url = format!("data:{};base64,{}", mime, base64_data);
            debug!(path = %resolved_path.display(), model = %model, "image_read: returning base64 for vision model");
            return Ok(ToolResult::success(ctx.tool_call_id, data_url));
        }

        debug!(path = %resolved_path.display(), model = %model, "image_read: running OCR");
        match describe_image(provider.as_ref(), &model, mime, &base64_data).await {
            Ok(description) => {
                debug!(path = %resolved_path.display(), desc_len = description.len(), "OCR complete");
                Ok(ToolResult::success(ctx.tool_call_id, description))
            }
            Err(e) => {
                warn!(error = %e, "OCR failed, falling back to base64");
                Ok(ToolResult::success(
                    ctx.tool_call_id,
                    format!("data:{};base64,{}", mime, base64_data),
                ))
            }
        }
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
