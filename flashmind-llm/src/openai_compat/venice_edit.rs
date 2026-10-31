use anyhow::Context;
use flashmind_types::{
    CompletionStream, FinishReason, ImageEditRequest, ImageOutputFormat, StreamEvent,
};
use reqwest::RequestBuilder;
use serde_json::{Map, Value, json};

pub(crate) fn edit(
    builder: RequestBuilder,
    request: ImageEditRequest,
    extra: Map<String, Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        if request.images.is_empty() { Err(anyhow::anyhow!("at least one input image is required"))?; }
        if request.mask.is_some() { Err(anyhow::anyhow!("Venice image editing does not support masks"))?; }
        for key in ["model", "modelId", "prompt", "image", "images", "resolution", "aspect_ratio", "quality", "output_format"] {
            if extra.contains_key(key) { Err(anyhow::anyhow!("{key} is controlled by the image edit command"))?; }
        }
        let mut body = Map::new();
        let model_field = if request.images.len() == 1 { "model" } else { "modelId" };
        body.insert(model_field.into(), json!(request.model));
        body.insert("prompt".into(), json!(request.prompt));
        if request.images.len() == 1 {
            body.insert("image".into(), json!(request.images[0]));
        } else {
            body.insert("images".into(), json!(request.images));
        }
        for (key, value) in [("resolution", request.resolution), ("aspect_ratio", request.aspect_ratio), ("quality", request.quality)] {
            if let Some(value) = value { body.insert(key.into(), json!(value)); }
        }
        let format = request.output_format.unwrap_or(ImageOutputFormat::Png);
        body.insert("output_format".into(), json!(format));
        body.extend(extra);
        let response = builder.json(&body).send().await.context("image edit request failed")?;
        let status = response.status();
        let media_type = response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|value| value.to_str().ok()).unwrap_or("").split(';').next().unwrap_or("").to_owned();
        let bytes = response.bytes().await.context("cannot read image edit response")?;
        if !status.is_success() { Err(anyhow::anyhow!("image edit failed ({status}): {}", String::from_utf8_lossy(&bytes)))?; }
        if media_type != format.media_type() { Err(anyhow::anyhow!("unexpected image edit response type: {media_type}"))?; }
        yield StreamEvent::FileAttachment { filename: String::new(), media_type, data: bytes.to_vec() };
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}
