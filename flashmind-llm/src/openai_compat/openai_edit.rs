use anyhow::Context;
use base64::{Engine, engine::general_purpose::STANDARD};
use flashmind_types::{
    CompletionStream, FinishReason, ImageEditRequest, ImageOutputFormat, StreamEvent,
};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{Map, Value, json};

#[derive(Deserialize)]
struct Response {
    data: Vec<Image>,
}
#[derive(Deserialize)]
struct Image {
    b64_json: String,
}

pub(crate) fn edit(
    builder: RequestBuilder,
    request: ImageEditRequest,
    extra: Map<String, Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        if request.images.is_empty() || request.images.len() > 16 { Err(anyhow::anyhow!("OpenAI image edits require 1 to 16 images"))?; }
        if request.aspect_ratio.is_some() { Err(anyhow::anyhow!("OpenAI image edits use --size WIDTHxHEIGHT instead of --aspect-ratio"))?; }
        for key in ["model", "prompt", "images", "image", "mask", "size", "quality", "output_format", "stream"] {
            if extra.contains_key(key) { Err(anyhow::anyhow!("{key} is controlled by the image edit command"))?; }
        }
        let images: Vec<_> = request.images.iter().map(|image| json!({"image_url": image})).collect();
        let mut body = Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("prompt".into(), json!(request.prompt));
        body.insert("images".into(), json!(images));
        let format = request.output_format.unwrap_or(ImageOutputFormat::Png);
        body.insert("output_format".into(), json!(format));
        if let Some(mask) = request.mask { body.insert("mask".into(), json!({"image_url": mask})); }
        if let Some(size) = request.resolution { body.insert("size".into(), json!(size.to_lowercase())); }
        if let Some(quality) = request.quality { body.insert("quality".into(), json!(quality)); }
        body.extend(extra);
        let response = builder.json(&body).send().await.context("image edit request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read image edit response")?;
        if !status.is_success() { Err(anyhow::anyhow!("image edit failed ({status}): {}", String::from_utf8_lossy(&bytes)))?; }
        let response: Response = serde_json::from_slice(&bytes).context("invalid image edit response")?;
        if response.data.is_empty() { Err(anyhow::anyhow!("image edit returned no images"))?; }
        for image in response.data {
            let data = STANDARD.decode(image.b64_json).context("invalid edited image data")?;
            yield StreamEvent::FileAttachment { filename: String::new(), media_type: format.media_type().into(), data };
        }
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}
