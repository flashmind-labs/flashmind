use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_types::{
    CompletionStream, FinishReason, ImageGenRequest, ImageOutputFormat, StreamEvent,
};
use ratelimit::Ratelimiter;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};

use crate::http::wait_for_rate_limit;

#[derive(Serialize)]
struct VeniceImageRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style_preset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variants: Option<u32>,
    format: ImageOutputFormat,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    style_references: Vec<StyleReference>,
}

#[derive(Serialize)]
struct StyleReference {
    image: String,
    strength: f32,
}

impl TryFrom<ImageGenRequest> for VeniceImageRequest {
    type Error = anyhow::Error;

    fn try_from(request: ImageGenRequest) -> Result<Self> {
        let mut resolution = None;
        let mut width = None;
        let mut height = None;
        if let Some(size) = &request.size {
            if let Some((w, h)) = size.split_once(['x', 'X']) {
                width = Some(w.parse().context("invalid image width")?);
                height = Some(h.parse().context("invalid image height")?);
                if request.aspect_ratio.is_some() {
                    bail!("Venice pixel sizes cannot be combined with --aspect-ratio");
                }
            } else {
                resolution = Some(size.to_uppercase());
            }
        }
        Ok(Self {
            model: request.model,
            prompt: request.prompt,
            aspect_ratio: request.aspect_ratio,
            resolution,
            width,
            height,
            quality: request.quality,
            style_preset: request.style,
            variants: request.n,
            format: request.output_format.unwrap_or(ImageOutputFormat::Png),
            style_references: request
                .input_references
                .into_iter()
                .map(|image| StyleReference {
                    image,
                    strength: 0.5,
                })
                .collect(),
        })
    }
}

#[derive(Deserialize)]
struct VeniceImageResponse {
    images: Vec<String>,
}

pub(crate) fn generate(
    builder: RequestBuilder,
    request: ImageGenRequest,
    limiter: Option<Arc<Ratelimiter>>,
    extra: serde_json::Map<String, serde_json::Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        let body = VeniceImageRequest::try_from(request)?;
        let media_type = body.format.media_type();
        let body = crate::http::add_extra_args(serde_json::to_value(body)?, &extra)?;
        let builder = builder.json(&body);
        if let Some(limiter) = &limiter {
            wait_for_rate_limit(limiter).await;
        }
        let response = builder.send().await.context("image generation request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read image generation response")?;
        if !status.is_success() {
            Err(anyhow::anyhow!("image generation failed ({status}): {}", String::from_utf8_lossy(&bytes)))?;
        }
        let response: VeniceImageResponse = serde_json::from_slice(&bytes).context("invalid image response")?;
        for image in response.images {
            let data = STANDARD.decode(image).context("invalid image data")?;
            yield StreamEvent::FileAttachment { filename: String::new(), media_type: media_type.into(), data };
        }
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}
