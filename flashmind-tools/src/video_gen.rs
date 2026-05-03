//! Video generation tool — generates videos via LLM provider video generation APIs.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::utils::truncate_utf8;
use flashmind_types::llm::LlmProvider;
use flashmind_types::model::{Model, Provider};
use flashmind_types::tool::{Tool, ToolContext, ToolResult};
use flashmind_types::{FrameImage, StreamEvent, VideoGenRequest};

pub type ProviderRegistry = Arc<std::collections::HashMap<Provider, Arc<dyn LlmProvider>>>;

#[derive(serde::Deserialize)]
struct GenerateVideoArgs {
    description: String,
    model: Option<Model>,
    resolution: Option<String>,
    aspect_ratio: Option<String>,
    duration: Option<u32>,
    generate_audio: Option<bool>,
    #[serde(default)]
    frame_images: Vec<FrameImageArg>,
    #[serde(default)]
    input_references: Vec<String>,
}

#[derive(serde::Deserialize)]
struct FrameImageArg {
    url: String,
    frame_type: String,
}

pub struct GenerateVideoTool {
    default_model: Option<Model>,
    output_dir: PathBuf,
    providers: ProviderRegistry,
}

impl GenerateVideoTool {
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
impl Tool for GenerateVideoTool {
    fn name(&self) -> &str {
        "generate_video"
    }

    fn description(&self) -> &str {
        "Generate a video from a text description. This may take several minutes. You can provide frame_images for image-to-video (first/last frame) or input_references for style guidance. After generating, ALWAYS include the file marker from the result in your response so the video is sent to the user."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Text description of the video to generate"
                },
                "model": {
                    "type": "string",
                    "description": "Video generation model (e.g., 'openrouter:google/veo-2.0-generate-001')"
                },
                "resolution": {
                    "type": "string",
                    "description": "Video resolution (e.g., '720p', '1080p')"
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio (e.g., '16:9', '9:16', '1:1')"
                },
                "duration": {
                    "type": "integer",
                    "description": "Video duration in seconds"
                },
                "generate_audio": {
                    "type": "boolean",
                    "description": "Whether to generate audio (defaults to true)"
                },
                "frame_images": {
                    "type": "array",
                    "description": "Images for image-to-video (first/last frame)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": { "type": "string", "description": "Image URL (https)" },
                            "frame_type": { "type": "string", "enum": ["first_frame", "last_frame"] }
                        },
                        "required": ["url", "frame_type"]
                    }
                },
                "input_references": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Image URLs for style/visual guidance"
                }
            },
            "required": ["description"]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(600)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let args: GenerateVideoArgs = ctx.parse_args(self.name())?;

        let model = args.model.or_else(|| self.default_model.clone());
        let Some(model) = model else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "No video generation model configured. Pass a model argument (e.g., 'openrouter:google/veo-2.0-generate-001').",
            ));
        };

        let Some(provider) = self.providers.get(&model.provider) else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("Provider '{}' not configured", model.provider),
            ));
        };

        let request = VideoGenRequest {
            model: model.name().to_string(),
            description: args.description,
            resolution: args.resolution,
            aspect_ratio: args.aspect_ratio,
            duration: args.duration,
            generate_audio: args.generate_audio,
            frame_images: args
                .frame_images
                .into_iter()
                .map(|f| FrameImage {
                    url: f.url,
                    frame_type: f.frame_type,
                })
                .collect(),
            input_references: args.input_references,
        };

        let mut stream = provider.generate_video(request);

        use futures::StreamExt;
        let mut saved_path: Option<PathBuf> = None;
        let mut progress = String::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ContentDelta(text)) => {
                    progress.push_str(&text);
                    tracing::debug!("{}", text.trim());
                }
                Ok(StreamEvent::FileAttachment {
                    media_type, data, ..
                }) => {
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis();
                    let ext = media_type.split('/').next_back().unwrap_or("mp4");
                    let filename = format!("video_{timestamp}.{ext}");
                    let output_path = self.output_dir.join(&filename);

                    if let Some(parent) = output_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }

                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&data)
                        .map_err(|e| anyhow::anyhow!("Failed to decode video data: {e}"))?;

                    std::fs::write(&output_path, &bytes)?;

                    let size_mb = bytes.len() / (1024 * 1024);
                    tracing::debug!(path = %output_path.display(), size_mb, "Video saved");
                    saved_path = Some(output_path);
                }
                Ok(StreamEvent::Finished(_)) => break,
                Err(e) => {
                    return Ok(ToolResult::failure(
                        ctx.tool_call_id,
                        format!("Video generation failed: {e}"),
                    ));
                }
                _ => {}
            }
        }

        let Some(path) = saved_path else {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                format!("No video was generated.\n{progress}"),
            ));
        };

        let size_mb = std::fs::metadata(&path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("video");

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Video saved to {} ({size_mb}MB). Include [{fname}]({}) in your response to send it.",
                path.display(),
                path.display(),
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let desc = args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        format!("Generating video: {}", truncate_utf8(desc, 60))
    }
}
