//! Image editing through Google Gemini and fal.

use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use flashmind_types::{
    CompletionRequest, CompletionStream, FinishReason, ImageEditRequest, LlmProvider, Provider,
    StreamEvent,
};
use futures_util::StreamExt;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Map, Value, json};
use url::Url;

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Image editing backend.
#[derive(Clone, Copy)]
pub enum ImageEditBackend {
    Google,
    Fal,
}

/// Native image editing provider for Google Gemini and fal.
pub struct ImageEditProvider {
    backend: ImageEditBackend,
    key: SecretString,
    base_url: Url,
    client: Client,
    extra: Map<String, Value>,
}

impl ImageEditProvider {
    /// Create an image editing provider, optionally using a custom API base URL.
    pub fn new(
        backend: ImageEditBackend,
        key: impl Into<SecretString>,
        base_url: Option<Url>,
        extra: Map<String, Value>,
    ) -> Result<Self> {
        let default = match backend {
            ImageEditBackend::Google => "https://generativelanguage.googleapis.com/",
            ImageEditBackend::Fal => "https://fal.run/",
        };
        Ok(Self {
            backend,
            key: key.into(),
            base_url: match base_url {
                Some(url) => url,
                None => Url::parse(default)?,
            },
            client: Client::builder()
                .timeout(Duration::from_secs(180))
                .build()?,
            extra,
        })
    }
}

#[async_trait]
impl LlmProvider for ImageEditProvider {
    fn name(&self) -> &str {
        match self.backend {
            ImageEditBackend::Google => "google",
            ImageEditBackend::Fal => "fal",
        }
    }
    fn provider(&self) -> Provider {
        Provider::Custom(self.name().into())
    }
    fn complete(&self, _request: CompletionRequest) -> CompletionStream {
        Box::pin(futures_util::stream::once(async {
            Err(anyhow::anyhow!("This provider only supports image editing"))
        }))
    }
    fn edit_image(&self, request: ImageEditRequest) -> CompletionStream {
        let backend = self.backend;
        let key = self.key.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let extra = self.extra.clone();
        Box::pin(async_stream::try_stream! {
            let image = match backend {
                ImageEditBackend::Google => google(&client, &base_url, &key, request, extra).await?,
                ImageEditBackend::Fal => fal(&client, &base_url, &key, request, extra).await?,
            };
            yield StreamEvent::FileAttachment { filename: String::new(), media_type: image.media_type, data: image.data };
            yield StreamEvent::Finished(FinishReason::Stop);
        })
    }
}

struct Image {
    media_type: String,
    data: Vec<u8>,
}

fn data_image(value: &str) -> Result<Image> {
    let (metadata, encoded) = value.split_once(',').context("invalid image data URL")?;
    let mime = metadata
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .context("expected base64 image data URL")?;
    ensure!(
        matches!(mime, "image/png" | "image/jpeg" | "image/webp"),
        "unsupported image type {mime}"
    );
    ensure!(
        encoded.len() <= (MAX_IMAGE_BYTES / 3 + 1) * 4,
        "image exceeds 20 MiB"
    );
    let data = STANDARD.decode(encoded).context("invalid image data")?;
    ensure!(data.len() <= MAX_IMAGE_BYTES, "image exceeds 20 MiB");
    Ok(Image {
        media_type: mime.into(),
        data,
    })
}

async fn load_image(client: &Client, value: &str) -> Result<Image> {
    if value.starts_with("data:") {
        return data_image(value);
    }
    let url = Url::parse(value).context("invalid image URL")?;
    ensure!(url.scheme() == "https", "image URL must use HTTPS");
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| error.without_url())
        .context("image download failed")?
        .error_for_status()
        .map_err(|error| error.without_url())
        .context("image download failed")?;
    let media_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .to_owned();
    ensure!(
        matches!(
            media_type.as_str(),
            "image/png" | "image/jpeg" | "image/webp"
        ),
        "unsupported image type {media_type}"
    );
    let mut data = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk
            .map_err(|error| error.without_url())
            .context("cannot read image")?;
        ensure!(
            chunk.len() <= MAX_IMAGE_BYTES.saturating_sub(data.len()),
            "image exceeds 20 MiB"
        );
        data.extend_from_slice(&chunk);
    }
    Ok(Image { media_type, data })
}

async fn response_json(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let mut data = Vec::new();
    let mut chunks = response.bytes_stream();
    const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.context("cannot read image edit response")?;
        ensure!(
            chunk.len() <= MAX_RESPONSE_BYTES.saturating_sub(data.len()),
            "image edit response exceeds 32 MiB"
        );
        data.extend_from_slice(&chunk);
    }
    ensure!(
        status.is_success(),
        "image edit failed ({status}): {}",
        String::from_utf8_lossy(&data)
    );
    serde_json::from_slice(&data).context("invalid image edit response")
}

fn endpoint(base: &Url, segments: &[&str]) -> Result<Url> {
    let mut url = base.clone();
    let mut path = url
        .path_segments_mut()
        .map_err(|_| anyhow::anyhow!("invalid API base URL"))?;
    path.pop_if_empty();
    for segment in segments {
        path.push(segment);
    }
    drop(path);
    Ok(url)
}

fn check_options(
    request: &ImageEditRequest,
    extra: &Map<String, Value>,
    protected: &[&str],
) -> Result<()> {
    ensure!(
        !request.images.is_empty(),
        "at least one input image is required"
    );
    for key in protected {
        ensure!(
            !extra.contains_key(*key),
            "{key} is controlled by the image edit command"
        );
    }
    Ok(())
}

async fn google(
    client: &Client,
    base: &Url,
    key: &SecretString,
    request: ImageEditRequest,
    mut extra: Map<String, Value>,
) -> Result<Image> {
    check_options(&request, &extra, &["contents", "generationConfig"])?;
    ensure!(
        request.mask.is_none(),
        "Google image editing does not support --mask"
    );
    ensure!(
        request.output_format.is_none(),
        "Google image editing does not support --format"
    );
    ensure!(
        request.quality.is_none(),
        "Google image edits do not support --quality"
    );
    ensure!(
        !request.model.contains('/'),
        "Google model must be a model ID"
    );
    let mut parts = vec![json!({"text":request.prompt})];
    for image in &request.images {
        let image = load_image(client, image).await?;
        parts.push(
            json!({"inlineData":{"mimeType":image.media_type,"data":STANDARD.encode(image.data)}}),
        );
    }
    let mut image_config = Map::new();
    if let Some(size) = request.resolution {
        image_config.insert("imageSize".into(), json!(size));
    }
    if let Some(ratio) = request.aspect_ratio {
        image_config.insert("aspectRatio".into(), json!(ratio));
    }
    let mut config = json!({"responseModalities":["IMAGE"]});
    if !image_config.is_empty() {
        config["imageConfig"] = Value::Object(image_config);
    }
    extra.insert("contents".into(), json!([{"parts":parts}]));
    extra.insert("generationConfig".into(), config);
    let url = endpoint(
        base,
        &[
            "v1beta",
            "models",
            &format!("{}:generateContent", request.model),
        ],
    )?;
    let response = client
        .post(url)
        .header("x-goog-api-key", key.expose_secret())
        .json(&extra)
        .send()
        .await
        .context("Google image edit request failed")?;
    let body = response_json(response).await?;
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .context("Google returned no image")?;
    for part in parts {
        if let Some(data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
            let mime = data
                .get("mimeType")
                .or_else(|| data.get("mime_type"))
                .and_then(Value::as_str)
                .context("Google image has no MIME type")?;
            let encoded = data
                .get("data")
                .and_then(Value::as_str)
                .context("Google image has no data")?;
            return data_image(&format!("data:{mime};base64,{encoded}"));
        }
    }
    bail!("Google returned no image")
}

async fn fal(
    client: &Client,
    base: &Url,
    key: &SecretString,
    request: ImageEditRequest,
    mut extra: Map<String, Value>,
) -> Result<Image> {
    check_options(
        &request,
        &extra,
        &["prompt", "image_urls", "sync_mode", "output_format"],
    )?;
    ensure!(
        request.mask.is_none(),
        "fal FLUX.2 image editing does not support --mask"
    );
    ensure!(
        request.images.len() <= 4,
        "fal FLUX.2 edit supports at most 4 images"
    );
    ensure!(
        request.aspect_ratio.is_none() && request.quality.is_none() && request.resolution.is_none(),
        "fal uses -e 'image_size=...' for sizing; --resolution, --aspect-ratio and --quality are not supported"
    );
    let segments: Vec<_> = request.model.split('/').collect();
    ensure!(
        segments.len() >= 2
            && segments.iter().all(|segment| !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))),
        "invalid fal model ID"
    );
    extra.insert("prompt".into(), json!(request.prompt));
    extra.insert("image_urls".into(), json!(request.images));
    extra.insert("sync_mode".into(), json!(true));
    if let Some(format) = request.output_format {
        extra.insert("output_format".into(), json!(format));
    }
    let url = endpoint(base, &segments)?;
    let response = client
        .post(url)
        .header("Authorization", format!("Key {}", key.expose_secret()))
        .json(&extra)
        .send()
        .await
        .context("fal image edit request failed")?;
    let body = response_json(response).await?;
    let url = body
        .pointer("/images/0/url")
        .and_then(Value::as_str)
        .context("fal returned no image")?;
    load_image(client, url).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn google_and_fal_edit_use_native_requests() {
        for backend in [ImageEditBackend::Google, ImageEditBackend::Fal] {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let provider =
                ImageEditProvider::new(backend, "example-key", Some(base), Map::new()).unwrap();
            let request = ImageEditRequest {
                mask: None,
                output_format: None,
                model: match backend {
                    ImageEditBackend::Google => "example-model",
                    ImageEditBackend::Fal => "fal-ai/flux-2/edit",
                }
                .into(),
                prompt: "change the sky".into(),
                images: vec!["data:image/webp;base64,YQ==".into()],
                resolution: None,
                aspect_ratio: None,
                quality: None,
            };
            let serve = async {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    headers.push(socket.read_u8().await.unwrap());
                }
                let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap()
                    .parse()
                    .unwrap();
                let mut body = vec![0; length];
                socket.read_exact(&mut body).await.unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                let response = match backend {
                    ImageEditBackend::Google => {
                        assert!(headers.starts_with("post /v1beta/models/example-model:generatecontent "));
                        assert!(headers.contains("x-goog-api-key: example-key"));
                        assert_eq!(body["contents"][0]["parts"][1]["inlineData"]["mimeType"], "image/webp");
                        json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"YQ=="}}]}}]})
                    }
                    ImageEditBackend::Fal => {
                        assert!(headers.starts_with("post /fal-ai/flux-2/edit "));
                        assert!(headers.contains("authorization: key example-key"));
                        assert_eq!(body["image_urls"][0], "data:image/webp;base64,YQ==");
                        assert_eq!(body["sync_mode"], true);
                        json!({"images":[{"url":"data:image/png;base64,YQ=="}]})
                    }
                }.to_string();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
            };
            let (events, ()) =
                tokio::join!(provider.edit_image(request).collect::<Vec<_>>(), serve);
            let events = events.into_iter().collect::<Result<Vec<_>>>().unwrap();
            assert!(
                matches!(&events[0], StreamEvent::FileAttachment { media_type, data, .. } if media_type == "image/png" && data == b"a")
            );
        }
    }
}
