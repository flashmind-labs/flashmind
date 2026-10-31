//! Text and image to 3D generation through Meshy and Tripo.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_types::llm::mime_from_extension;
use flashmind_types::{Model, Provider};
use futures_util::StreamExt;
use reqwest::{Client, Method, Response};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{Instant, sleep, timeout_at};
use url::Url;

use crate::http::{add_extra_args, http_client_builder};

const POLL_INTERVAL: Duration = Duration::from_secs(5);
const JOB_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 500 * 1024 * 1024;
const MAX_RIG_INPUT_BYTES: u64 = 100 * 1024 * 1024;

/// 3D generation service.
#[derive(Clone, Copy, Debug)]
pub enum ThreeDProvider {
    Meshy,
    Tripo,
}

/// Text or image to turn into a 3D asset.
pub enum ThreeDInput {
    Text(String),
    Images(Vec<ThreeDImage>),
}

/// An image reference as a URL or local file.
pub enum ThreeDImage {
    Url(Url),
    File(PathBuf),
}

/// A GLB to rig, supplied as a public URL or local file.
pub enum RigInput {
    Url(Url),
    File(PathBuf),
}

/// One 3D generation request.
pub struct ThreeDRequest {
    pub model: Model,
    pub input: ThreeDInput,
    /// Generate a texture. Meshy text requests create an additional refine task.
    pub texture: bool,
    pub extra_args: Map<String, Value>,
}

/// Client for asynchronous 3D generation tasks.
pub struct ThreeDClient {
    provider: ThreeDProvider,
    api_key: SecretString,
    base_url: Url,
    http: Client,
}

impl ThreeDClient {
    /// Create a client for the selected service.
    pub fn new(provider: ThreeDProvider, api_key: impl Into<SecretString>) -> Result<Self> {
        let base = match provider {
            ThreeDProvider::Meshy => "https://api.meshy.ai/",
            ThreeDProvider::Tripo => "https://openapi.tripo3d.ai/",
        };
        Ok(Self {
            provider,
            api_key: api_key.into(),
            base_url: Url::parse(base)?,
            http: http_client_builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(120))
                .build()?,
        })
    }

    /// Override the API root, for a proxy or local test server.
    pub fn with_base_url(mut self, base_url: Url) -> Self {
        self.base_url = base_url;
        self
    }

    fn endpoint(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        let prefix = url.path().trim_end_matches('/');
        url.set_path(&format!("{prefix}/{}", path.trim_start_matches('/')));
        url
    }

    fn authed(&self, method: Method, url: Url) -> reqwest::RequestBuilder {
        self.http
            .request(method, url)
            .bearer_auth(self.api_key.expose_secret())
    }

    /// Submit and wait for a model. `on_task` receives each submitted task ID.
    pub async fn generate(
        &self,
        request: ThreeDRequest,
        mut on_task: impl FnMut(&str),
    ) -> Result<Url> {
        let expected = match self.provider {
            ThreeDProvider::Meshy => "meshy",
            ThreeDProvider::Tripo => "tripo",
        };
        if !matches!(&request.model.provider, Provider::Custom(name) if name == expected) {
            bail!("3D model provider must be {expected}");
        }
        for key in [
            "mode",
            "prompt",
            "input",
            "image_url",
            "image_urls",
            "inputs",
            "model",
            "ai_model",
            "texture",
            "should_texture",
            "preview_task_id",
            "target_formats",
        ] {
            if request.extra_args.contains_key(key) {
                bail!("extra argument '{key}' conflicts with a 3D command field");
            }
        }
        if matches!(self.provider, ThreeDProvider::Tripo)
            && !request.texture
            && request.extra_args.contains_key("pbr")
        {
            bail!("pbr requires --texture");
        }
        if let ThreeDInput::Text(prompt) = &request.input
            && prompt.trim().is_empty()
        {
            bail!("3D prompt is empty");
        }
        if let ThreeDInput::Images(images) = &request.input
            && !(1..=4).contains(&images.len())
        {
            bail!("provide between one and four input images");
        }
        let deadline = Instant::now() + JOB_TIMEOUT;
        match self.provider {
            ThreeDProvider::Meshy => self.meshy(request, deadline, &mut on_task).await,
            ThreeDProvider::Tripo => self.tripo(request, deadline, &mut on_task).await,
        }
    }

    /// Optionally remesh, then rig a textured humanoid model with Meshy.
    pub async fn rig(
        &self,
        input: RigInput,
        remesh: bool,
        height_meters: f64,
        mut on_task: impl FnMut(&str),
    ) -> Result<Url> {
        if !matches!(self.provider, ThreeDProvider::Meshy) {
            bail!("rigging is only supported for Meshy models");
        }
        if !height_meters.is_finite() || height_meters <= 0.0 {
            bail!("character height must be a positive number of meters");
        }
        let mut model_url = match input {
            RigInput::Url(url) => {
                if !matches!(url.scheme(), "http" | "https") {
                    bail!("model URL must use HTTP or HTTPS");
                }
                url.to_string()
            }
            RigInput::File(path) => {
                if !path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"))
                {
                    bail!("rigging input must be a GLB file");
                }
                let file = File::open(&path)
                    .await
                    .with_context(|| format!("cannot read {}", path.display()))?;
                let mut bytes = Vec::new();
                file.take(MAX_RIG_INPUT_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .await
                    .with_context(|| format!("cannot read {}", path.display()))?;
                if u64::try_from(bytes.len())? > MAX_RIG_INPUT_BYTES {
                    bail!("local GLB exceeds the 100 MB data URI limit");
                }
                if bytes.is_empty() {
                    bail!("rigging input is empty");
                }
                format!(
                    "data:application/octet-stream;base64,{}",
                    STANDARD.encode(bytes)
                )
            }
        };
        let deadline = Instant::now() + JOB_TIMEOUT;
        if remesh {
            let path = "openapi/v1/remesh";
            let id = self.meshy_submit(path, json!({"model_url": model_url, "topology": "triangle", "target_polycount": 300000, "target_formats": ["glb"]})).await?;
            on_task(&id);
            model_url = self.meshy_poll(path, &id, deadline).await?.to_string();
        }
        let path = "openapi/v1/rigging";
        let id = self
            .meshy_submit(
                path,
                json!({"model_url": model_url, "height_meters": height_meters}),
            )
            .await?;
        on_task(&id);
        self.meshy_poll(path, &id, deadline).await
    }

    async fn meshy(
        &self,
        request: ThreeDRequest,
        deadline: Instant,
        on_task: &mut impl FnMut(&str),
    ) -> Result<Url> {
        let is_text = matches!(&request.input, ThreeDInput::Text(_));
        let (path, body) = match request.input {
            ThreeDInput::Text(prompt) => (
                "openapi/v2/text-to-3d",
                json!({"mode":"preview", "prompt":prompt, "ai_model":request.model.capability_name(), "target_formats":["glb"]}),
            ),
            ThreeDInput::Images(images) => {
                let mut inputs = Vec::with_capacity(images.len());
                for image in images {
                    inputs.push(match image {
                        ThreeDImage::Url(url) => url.to_string(),
                        ThreeDImage::File(path) => {
                            let image = read_image(&path, false).await?;
                            format!(
                                "data:{};base64,{}",
                                image.mime,
                                STANDARD.encode(image.bytes)
                            )
                        }
                    });
                }
                if inputs.len() == 1 {
                    (
                        "openapi/v1/image-to-3d",
                        json!({"image_url":inputs[0], "ai_model":request.model.capability_name(), "should_texture":request.texture, "target_formats":["glb"]}),
                    )
                } else {
                    (
                        "openapi/v1/multi-image-to-3d",
                        json!({"image_urls":inputs, "ai_model":request.model.capability_name(), "should_texture":request.texture, "target_formats":["glb"]}),
                    )
                }
            }
        };
        let body = add_extra_args(body, &request.extra_args)?;
        let id = self.meshy_submit(path, body).await?;
        on_task(&id);
        let mut asset = self.meshy_poll(path, &id, deadline).await?;
        if is_text && request.texture {
            let body = json!({"mode":"refine", "preview_task_id":id, "target_formats":["glb"]});
            let id = self.meshy_submit(path, body).await?;
            on_task(&id);
            asset = self.meshy_poll(path, &id, deadline).await?;
        }
        Ok(asset)
    }

    async fn meshy_submit(&self, path: &str, body: Value) -> Result<String> {
        let response = self
            .authed(Method::POST, self.endpoint(path))
            .json(&body)
            .send()
            .await
            .context("Meshy task submission failed")?;
        let created: MeshyCreated = response_json(response, "Meshy task submission").await?;
        Ok(created.result)
    }

    async fn meshy_poll(&self, path: &str, id: &str, deadline: Instant) -> Result<Url> {
        let mut url = self.endpoint(path);
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid Meshy task URL"))?
            .push(id);
        loop {
            let response = timeout_at(deadline, self.authed(Method::GET, url.clone()).send())
                .await
                .context("Meshy task timed out")?
                .context("Meshy task request failed")?;
            let task: MeshyTask =
                timeout_at(deadline, response_json(response, "Meshy task status"))
                    .await
                    .context("Meshy task timed out")??;
            match task.status.as_str() {
                "SUCCEEDED" => {
                    return task
                        .model_urls
                        .and_then(|urls| urls.glb)
                        .or_else(|| {
                            task.result
                                .and_then(|result| result.rigged_character_glb_url)
                        })
                        .context("Meshy task did not provide a GLB model");
                }
                "FAILED" | "CANCELED" | "EXPIRED" => bail!(
                    "Meshy task {id} ended with status {}{}",
                    task.status,
                    task.task_error
                        .map(|error| format!(": {}", error.message))
                        .unwrap_or_default()
                ),
                "PENDING" | "IN_PROGRESS" => {}
                other => bail!("unknown Meshy task status: {other}"),
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .context("Meshy task timed out")?;
        }
    }

    async fn tripo(
        &self,
        request: ThreeDRequest,
        deadline: Instant,
        on_task: &mut impl FnMut(&str),
    ) -> Result<Url> {
        let mut body = json!({"model":request.model.capability_name(), "texture":request.texture});
        if !request.texture {
            body["pbr"] = json!(false);
        }
        let path = match request.input {
            ThreeDInput::Text(prompt) => {
                body["prompt"] = json!(prompt);
                "v3/generation/text-to-model"
            }
            ThreeDInput::Images(images) => {
                let mut prepared = Vec::with_capacity(images.len());
                for image in images {
                    prepared.push(match image {
                        ThreeDImage::Url(url) => TripoImage::Url(url),
                        ThreeDImage::File(path) => TripoImage::File(read_image(&path, true).await?),
                    });
                }
                let mut inputs = Vec::with_capacity(prepared.len());
                for image in prepared {
                    inputs.push(self.tripo_image(image).await?);
                }
                if inputs.len() == 1 {
                    body["input"] = json!(inputs[0]);
                    "v3/generation/image-to-model"
                } else {
                    let views = ["front", "left", "back", "right"];
                    body["inputs"] = Value::Array(
                        inputs
                            .into_iter()
                            .enumerate()
                            .map(|(index, input)| {
                                let mut view = Map::new();
                                view.insert(views[index].to_owned(), json!(input));
                                Value::Object(view)
                            })
                            .collect(),
                    );
                    "v3/generation/multiview-to-model"
                }
            }
        };
        let body = add_extra_args(body, &request.extra_args)?;
        let response = self
            .authed(Method::POST, self.endpoint(path))
            .json(&body)
            .send()
            .await
            .context("Tripo task submission failed")?;
        let created: TripoResponse<TripoCreated> =
            response_json(response, "Tripo task submission").await?;
        let id = created.into_data()?.task_id;
        on_task(&id);
        let mut url = self.endpoint("v3/tasks");
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid Tripo task URL"))?
            .push(&id);
        loop {
            let response = timeout_at(deadline, self.authed(Method::GET, url.clone()).send())
                .await
                .context("Tripo task timed out")?
                .context("Tripo task request failed")?;
            let result: TripoResponse<TripoTask> =
                timeout_at(deadline, response_json(response, "Tripo task status"))
                    .await
                    .context("Tripo task timed out")??;
            let task = result.into_data()?;
            match task.status.as_str() {
                "success" => {
                    return task
                        .output
                        .and_then(|output| output.model_url)
                        .context("Tripo task did not provide a GLB model");
                }
                "failed" | "cancelled" | "banned" | "expired" => bail!(
                    "Tripo task {id} ended with status {}{}",
                    task.status,
                    task.error_message
                        .map(|message| format!(": {message}"))
                        .unwrap_or_default()
                ),
                "queued" | "running" | "unknown" => {}
                other => bail!("unknown Tripo task status: {other}"),
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .context("Tripo task timed out")?;
        }
    }

    async fn tripo_image(&self, source: TripoImage) -> Result<String> {
        match source {
            TripoImage::Url(url) => Ok(url.to_string()),
            TripoImage::File(image) => {
                let part = reqwest::multipart::Part::bytes(image.bytes)
                    .file_name(image.filename)
                    .mime_str(image.mime)?;
                let response = self
                    .authed(Method::POST, self.endpoint("v3/files"))
                    .multipart(reqwest::multipart::Form::new().part("file", part))
                    .send()
                    .await
                    .context("Tripo image upload failed")?;
                let uploaded: TripoResponse<TripoUpload> =
                    response_json(response, "Tripo image upload").await?;
                Ok(uploaded.into_data()?.file_token)
            }
        }
    }

    /// Download the generated GLB without sending provider credentials.
    pub async fn save(&self, url: &Url, file: &mut File) -> Result<()> {
        if url.scheme() != "https" {
            bail!("3D asset URL must use HTTPS");
        }
        let response = self
            .http
            .get(url.clone())
            .send()
            .await
            .map_err(|error| error.without_url())
            .context("3D model download failed")?
            .error_for_status()
            .map_err(|error| error.without_url())
            .context("3D model download failed")?;
        write_model(response, file).await
    }
}

async fn write_model(response: Response, file: &mut File) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut total = 0u64;
    while let Some(bytes) = stream.next().await {
        let bytes = bytes
            .map_err(|error| error.without_url())
            .context("3D model download failed")?;
        total = total
            .checked_add(u64::try_from(bytes.len())?)
            .context("3D model size overflow")?;
        if total > MAX_MODEL_BYTES {
            bail!("3D model exceeds the download size limit");
        }
        file.write_all(&bytes)
            .await
            .context("cannot write 3D model")?;
    }
    if total == 0 {
        bail!("3D model download was empty");
    }
    file.flush().await.context("cannot finish 3D model")
}

enum TripoImage {
    Url(Url),
    File(InputImage),
}

struct InputImage {
    bytes: Vec<u8>,
    mime: &'static str,
    filename: &'static str,
}

async fn read_image(path: &Path, tripo: bool) -> Result<InputImage> {
    let mime = mime_from_extension(path);
    let filename = match mime {
        "image/png" => "image.png",
        "image/jpeg" => "image.jpg",
        "image/webp" if !tripo => "image.webp",
        _ => bail!(
            "image must be PNG or JPEG{}",
            if tripo { "" } else { ", or WebP" }
        ),
    };
    let size = fs::metadata(path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?
        .len();
    if size > MAX_IMAGE_BYTES {
        bail!("image exceeds the 20 MB limit");
    }
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    if u64::try_from(bytes.len())? > MAX_IMAGE_BYTES {
        bail!("image exceeds the 20 MB limit");
    }
    Ok(InputImage {
        bytes,
        mime,
        filename,
    })
}

async fn response_json<T: serde::de::DeserializeOwned>(
    response: Response,
    operation: &str,
) -> Result<T> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .context("cannot read provider response")?;
    if !status.is_success() {
        bail!(
            "{operation} failed ({status}): {}",
            String::from_utf8_lossy(&bytes)
        );
    }
    serde_json::from_slice(&bytes).with_context(|| format!("invalid {operation} response"))
}

#[derive(Deserialize)]
struct MeshyCreated {
    result: String,
}
#[derive(Deserialize)]
struct MeshyTask {
    status: String,
    model_urls: Option<MeshyUrls>,
    result: Option<MeshyRigResult>,
    task_error: Option<MeshyError>,
}
#[derive(Deserialize)]
struct MeshyRigResult {
    rigged_character_glb_url: Option<Url>,
}
#[derive(Deserialize)]
struct MeshyUrls {
    glb: Option<Url>,
}
#[derive(Deserialize)]
struct MeshyError {
    message: String,
}
#[derive(Deserialize)]
struct TripoResponse<T> {
    code: i64,
    data: Option<T>,
    message: Option<String>,
}
impl<T> TripoResponse<T> {
    fn into_data(self) -> Result<T> {
        if self.code != 0 {
            bail!(
                "Tripo error {}: {}",
                self.code,
                self.message.unwrap_or_default()
            );
        }
        self.data.context("Tripo response missing data")
    }
}
#[derive(Deserialize)]
struct TripoCreated {
    task_id: String,
}
#[derive(Deserialize)]
struct TripoUpload {
    file_token: String,
}
#[derive(Deserialize)]
struct TripoTask {
    status: String,
    output: Option<TripoOutput>,
    error_message: Option<String>,
}
#[derive(Deserialize)]
struct TripoOutput {
    model_url: Option<Url>,
}

#[cfg(test)]
mod tests;
