use std::time::Duration;

use anyhow::Context;
use flashmind_types::{CompletionStream, FinishReason, ImageGenRequest, StreamEvent};
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use url::Url;

#[derive(Deserialize)]
struct Response {
    data: Vec<Image>,
}
#[derive(Deserialize)]
struct Image {
    url: Url,
}

pub(crate) fn generate(
    builder: RequestBuilder,
    request: ImageGenRequest,
    extra: Map<String, Value>,
) -> CompletionStream {
    Box::pin(async_stream::try_stream! {
        if !request.input_references.is_empty() || request.aspect_ratio.is_some() || request.style.is_some() || request.output_format.is_some() || request.n.is_some_and(|n| n != 1) {
            Err(anyhow::anyhow!("Z.ai image generation supports one image with --size and --quality; references, aspect ratio, style and output format are unsupported"))?;
        }
        let mut body = json!({"model":request.model,"prompt":request.prompt});
        if let Some(size) = request.size { body["size"] = json!(size); }
        if let Some(quality) = request.quality { body["quality"] = json!(quality); }
        let body = crate::http::add_extra_args(body, &extra)?;
        let response = builder.json(&body).send().await.context("Z.ai image request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("cannot read Z.ai image response")?;
        if !status.is_success() { Err(anyhow::anyhow!("Z.ai image generation failed ({status}): {}", String::from_utf8_lossy(&bytes)))?; }
        let response: Response = serde_json::from_slice(&bytes).context("invalid Z.ai image response")?;
        if response.data.is_empty() { Err(anyhow::anyhow!("Z.ai returned no image"))?; }
        let client = Client::builder().timeout(Duration::from_secs(120)).build()?;
        for image in response.data {
            if image.url.scheme() != "https" { Err(anyhow::anyhow!("Z.ai image URL must use HTTPS"))?; }
            let response = client.get(image.url).send().await.map_err(|error| error.without_url()).context("cannot download Z.ai image")?.error_for_status().map_err(|error| error.without_url()).context("cannot download Z.ai image")?;
            let media_type = response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|value| value.to_str().ok()).unwrap_or("").split(';').next().unwrap_or("").to_owned();
            if !matches!(media_type.as_str(), "image/png" | "image/jpeg" | "image/webp") { Err(anyhow::anyhow!("unexpected Z.ai image type: {media_type}"))?; }
            let mut data = Vec::new();
            let mut chunks = response.bytes_stream();
            while let Some(chunk) = chunks.next().await {
                let chunk = chunk.map_err(|error| error.without_url()).context("cannot read Z.ai image")?;
                if chunk.len() > (50 * 1024 * 1024usize).saturating_sub(data.len()) { Err(anyhow::anyhow!("Z.ai image exceeds 50 MiB"))?; }
                data.extend_from_slice(&chunk);
            }
            yield StreamEvent::FileAttachment { filename: String::new(), media_type, data };
        }
        yield StreamEvent::Finished(FinishReason::Stop);
    })
}
