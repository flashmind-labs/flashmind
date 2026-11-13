//! Generate GLB assets from text or reference images.

use std::path::PathBuf;

use anyhow::{Context, bail};
use async_trait::async_trait;
use flashmind_llm::three_d::{
    ThreeDClient, ThreeDImage, ThreeDInput, ThreeDProvider, ThreeDRequest,
};
use flashmind_types::{
    ApiKey, CompletionRequest, CompletionStream, LlmProvider, Model, ModelCategory, ModelInfo,
    Provider,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use url::Url;

use crate::{Tool, ToolContext, ToolResult};

// Tripo documents version IDs, not a native `latest` alias:
// https://developers.tripo3d.ai/en/docs/models-and-versions
pub(crate) const TRIPO_LATEST: &str = "v3.1-20260211";

/// Listing adapter for 3D services that do not expose a model catalog endpoint.
pub(crate) struct ThreeDModelCatalog(pub ThreeDProvider);

#[async_trait]
impl LlmProvider for ThreeDModelCatalog {
    fn name(&self) -> &str {
        match self.0 {
            ThreeDProvider::Meshy => "meshy",
            ThreeDProvider::Tripo => "tripo",
        }
    }

    fn provider(&self) -> Provider {
        Provider::Custom(self.name().into())
    }

    fn complete(&self, _: CompletionRequest) -> CompletionStream {
        Box::pin(futures_util::stream::once(async {
            Err(anyhow::anyhow!("Use generate_3d for 3D generation"))
        }))
    }

    async fn list_models(&self) -> Option<Vec<ModelInfo>> {
        let ids: &[&str] = match self.0 {
            // Standard geometry models accepted by the text and image endpoints:
            // https://docs.meshy.ai/en/api/models
            ThreeDProvider::Meshy => &["latest", "meshy-7.1", "meshy-6", "meshy-6-lite"],
            ThreeDProvider::Tripo => &["latest", TRIPO_LATEST, "v3.0-20250812", "v2.5-20250123"],
        };
        Some(
            ids.iter()
                .map(|id| ModelInfo {
                    id: (*id).into(),
                    name: Some(
                        if matches!(self.0, ThreeDProvider::Tripo) && *id == "latest" {
                            format!("Latest H series ({TRIPO_LATEST})")
                        } else {
                            (*id).into()
                        },
                    ),
                    categories: vec![ModelCategory::ThreeD],
                    created_at: None,
                    context_length: None,
                    max_completion_tokens: None,
                    capabilities: Default::default(),
                    pricing: Default::default(),
                    image: Default::default(),
                })
                .collect(),
        )
    }
}

#[derive(Deserialize)]
struct GenerateThreeDArgs {
    prompt: Option<String>,
    #[serde(default)]
    images: Vec<String>,
    model: Option<Model>,
    #[serde(default)]
    texture: bool,
    polycount: Option<u64>,
    #[serde(flatten)]
    extra_args: Map<String, Value>,
}

/// Text or image to GLB generation with explicitly configured clients.
pub struct GenerateThreeDTool {
    clients: Vec<(Model, ThreeDClient)>,
    output_dir: PathBuf,
}

impl GenerateThreeDTool {
    /// Configure Meshy and Tripo from supplied keys, preferring Meshy by default.
    pub fn from_keys(
        keys: impl Fn(ApiKey) -> Option<String>,
        output_dir: PathBuf,
    ) -> anyhow::Result<Option<Self>> {
        let mut clients = Vec::new();
        for (kind, provider, default) in [
            (ApiKey::Meshy, ThreeDProvider::Meshy, "meshy:latest"),
            (ApiKey::Tripo, ThreeDProvider::Tripo, "tripo:v3.1-20260211"),
        ] {
            if let Some(key) = keys(kind).filter(|key| !key.trim().is_empty()) {
                clients.push((default.parse()?, ThreeDClient::new(provider, key)?));
            }
        }
        Ok((!clients.is_empty()).then_some(Self {
            clients,
            output_dir,
        }))
    }

    fn request(&self, ctx: &ToolContext<'_>) -> anyhow::Result<(&ThreeDClient, ThreeDRequest)> {
        let args: GenerateThreeDArgs = ctx.parse_args(self.name())?;
        let model = args
            .model
            .as_ref()
            .or_else(|| self.clients.first().map(|(model, _)| model))
            .context("No 3D provider configured")?;
        let client = self
            .clients
            .iter()
            .find(|(default, _)| default.provider == model.provider)
            .map(|(_, client)| client)
            .context("3D provider is unsupported or has no configured API key")?;
        let input = match (args.prompt, args.images.is_empty()) {
            (Some(prompt), true) if !prompt.trim().is_empty() => ThreeDInput::Text(prompt),
            (None, false) if args.images.len() <= 4 => ThreeDInput::Images(
                args.images
                    .into_iter()
                    .map(|image| match Url::parse(&image) {
                        Ok(url) if matches!(url.scheme(), "http" | "https") => {
                            Ok(ThreeDImage::Url(url))
                        }
                        Ok(_) => bail!("image URL must use HTTP or HTTPS"),
                        Err(url::ParseError::RelativeUrlWithoutBase) => Ok(ThreeDImage::File(
                            ctx.working_dir
                                .map_or_else(|| PathBuf::from(&image), |dir| dir.join(&image)),
                        )),
                        Err(error) => Err(error).context("invalid image URL"),
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?,
            ),
            _ => bail!("provide either a nonempty prompt or between one and four images"),
        };
        let mut model = model.clone();
        if model.provider == Provider::Custom("tripo".into()) && model.name() == "latest" {
            model.model.name = TRIPO_LATEST.into();
            model.model.real_name = None;
        }
        let mut request = ThreeDRequest {
            model,
            input,
            texture: args.texture,
            extra_args: args.extra_args,
        };
        if let Some(polycount) = args.polycount {
            request = request.with_polycount(polycount)?;
        }
        Ok((client, request))
    }
}

#[async_trait]
impl Tool for GenerateThreeDTool {
    fn name(&self) -> &str {
        "generate_3d"
    }

    fn description(&self) -> &str {
        "Generate a GLB 3D model from text or one to four reference images with Meshy or Tripo. Returns a saved file path and file marker. Tripo images use front, left, back, right order. Set polycount to request fewer faces for smaller files and faster loading (Meshy uses an approximate target). Other arguments pass through to the provider, such as topology or symmetry_mode; command fields cannot be overridden."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "description": "Text description, mutually exclusive with images"},
                "images": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}, "description": "HTTP(S) URLs or local paths relative to the working directory"},
                "model": {"type": "string", "description": "meshy:model or tripo:model. Defaults to meshy:latest, then tripo:v3.1-20260211, according to configured keys. Use list_models with category 3d to discover models. tripo:latest maps to the latest known H series version"},
                "texture": {"type": "boolean", "default": false, "description": "Generate textures (Meshy text generation adds a refine task)"},
                "polycount": {"type": "integer", "minimum": 1, "description": "Maximum faces (triangles, or quads with quad topology). A lower count gives a smaller file that loads faster; it does not lower the credits charged. Ranges: Meshy 100 to 300000, an approximate target (meshy-t2 up to 15000, meshy-t2.5 up to 30000); Tripo v3.1 up to 1500000, v3.0 up to 1000000, v2.5 up to 500000, 500 to 20000 with smart_low_poly, P1 50 to 20000, P2 48 to 50000. Out-of-range values and conflicting options are rejected with the valid range"}
            },
            "additionalProperties": true,
            "oneOf": [
                {"required": ["prompt"], "not": {"required": ["images"]}},
                {"required": ["images"], "not": {"required": ["prompt"]}}
            ]
        })
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(1920)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let (client, request) = self.request(&ctx)?;
        let output_dir = ctx
            .working_dir
            .map_or_else(|| self.output_dir.clone(), |dir| dir.join(&self.output_dir));
        // The caller may cancel polling, but providers can still charge for submitted jobs.
        let url = tokio::select! {
            biased;
            _ = ctx.cancel_token().cancelled() => bail!("3D generation cancelled"),
            result = client.generate(request, |id| ctx.progress(&format!("3D task: {id}"))) => result?,
        };
        tokio::fs::create_dir_all(&output_dir).await?;
        let stem = format!("model_{}", uuid::Uuid::new_v4());
        let partial = output_dir.join(format!("{stem}.partial"));
        let path = output_dir.join(format!("{stem}.glb"));
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .await?;
        let result = tokio::select! {
            biased;
            _ = ctx.cancel_token().cancelled() => Err(anyhow::anyhow!("3D download cancelled")),
            result = client.save(&url, &mut file) => result,
        };
        drop(file);
        if let Err(error) = result {
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(error);
        }
        if let Err(error) = tokio::fs::rename(&partial, &path).await {
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(error.into());
        }
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.glb");
        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "3D model saved to {}. Include [{filename}]({}) in your response to send it.",
                path.display(),
                path.display()
            ),
        )
        .with_files(vec![path]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::Provider;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn polycount_and_unknown_arguments_reach_the_request() {
        let tool = GenerateThreeDTool::from_keys(|_| Some("example-key".into()), PathBuf::new())
            .unwrap()
            .unwrap();
        let cancel = CancellationToken::new();
        for (model, field) in [
            ("meshy:latest", "target_polycount"),
            ("tripo:latest", "face_limit"),
        ] {
            let ctx = ToolContext::new(
                "call",
                json!({
                    "prompt": "chair", "model": model, "texture": false,
                    "polycount": 5000, "symmetry_mode": "auto",
                    "future_option": {"enabled": true}
                }),
                None,
                &cancel,
            );
            let (_, request) = tool.request(&ctx).unwrap();
            assert_eq!(request.extra_args[field], 5000);
            assert_eq!(request.extra_args["symmetry_mode"], "auto");
            assert_eq!(
                request.extra_args["future_option"],
                json!({"enabled": true})
            );
            for key in ["prompt", "images", "model", "texture", "polycount"] {
                assert!(!request.extra_args.contains_key(key));
            }
        }
        let ctx = ToolContext::new(
            "call",
            json!({"prompt": "chair", "should_remesh": true, "target_polycount": 5000, "topology": "quad"}),
            None,
            &cancel,
        );
        let (_, request) = tool.request(&ctx).unwrap();
        assert_eq!(
            request.extra_args,
            serde_json::from_value::<Map<String, Value>>(
                json!({"should_remesh": true, "target_polycount": 5000, "topology": "quad"})
            )
            .unwrap()
        );
        for polycount in [
            json!(-1),
            json!(0),
            json!(1.5),
            json!("5000"),
            json!(300001),
        ] {
            let ctx = ToolContext::new(
                "call",
                json!({"prompt": "chair", "polycount": polycount}),
                None,
                &cancel,
            );
            assert!(tool.request(&ctx).is_err());
        }
    }

    #[test]
    fn every_schema_property_reaches_text_or_image_requests() {
        let tool = GenerateThreeDTool::from_keys(|_| Some("example-key".into()), PathBuf::new())
            .unwrap()
            .unwrap();
        let cancel = CancellationToken::new();
        let mut exercised = std::collections::BTreeSet::new();
        for input in [
            json!({"prompt": "chair"}),
            json!({"images": ["https://example.com/reference.png"]}),
        ] {
            let mut args = input;
            args["model"] = json!("meshy:meshy-6");
            args["texture"] = json!(true);
            args["polycount"] = json!(5000);
            exercised.extend(args.as_object().unwrap().keys().cloned());
            args["symmetry_mode"] = json!("auto");
            let (_, request) = tool
                .request(&ToolContext::new("call", args, None, &cancel))
                .unwrap();
            assert_eq!(request.model.name(), "meshy-6");
            assert!(request.texture);
            assert_eq!(request.extra_args["target_polycount"], 5000);
            assert_eq!(request.extra_args["symmetry_mode"], "auto");
            match request.input {
                ThreeDInput::Text(prompt) => assert_eq!(prompt, "chair"),
                ThreeDInput::Images(images) => {
                    assert_eq!(images.len(), 1);
                    assert!(
                        matches!(&images[0], ThreeDImage::Url(url) if url.as_str() == "https://example.com/reference.png")
                    );
                }
            }
        }
        let schema = tool.parameters();
        assert_eq!(schema["additionalProperties"], true);
        assert_eq!(
            exercised,
            schema["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        );
    }

    #[tokio::test]
    async fn validates_inputs_and_cancellation_without_context_provider() {
        let dir = tempfile::tempdir().unwrap();
        let working_dir = dir.path().to_path_buf();
        let tool = GenerateThreeDTool::from_keys(
            |key| (key == ApiKey::Meshy).then(|| "fake-key".into()),
            working_dir.clone(),
        )
        .unwrap()
        .unwrap();
        let cancel = CancellationToken::new();
        for args in [
            json!({}),
            json!({"prompt": " "}),
            json!({"images": []}),
            json!({"prompt": "chair", "images": ["a.png"]}),
            json!({"images": ["file:///image.png"]}),
            json!({"prompt": "chair", "model": "tripo:example"}),
        ] {
            assert!(
                tool.execute(ToolContext::new("call", args, Some(&working_dir), &cancel))
                    .await
                    .is_err()
            );
        }
        let ctx = ToolContext::new(
            "call",
            json!({"images": ["reference.png"]}),
            Some(&working_dir),
            &cancel,
        );
        let (_, request) = tool.request(&ctx).unwrap();
        assert!(
            matches!(request.input, ThreeDInput::Images(images) if matches!(&images[0], ThreeDImage::File(path) if path == &working_dir.join("reference.png")))
        );
        cancel.cancel();
        let error = tool
            .execute(ToolContext::new(
                "call",
                json!({"prompt": "chair"}),
                Some(&working_dir),
                &cancel,
            ))
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "3D generation cancelled");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn provider_dispatch_and_failed_download_leave_no_partial_file() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for (provider, model, responses) in [
            (
                ThreeDProvider::Meshy,
                "meshy:latest",
                [
                    json!({"result": "task"}),
                    json!({"status": "SUCCEEDED", "model_urls": {"glb": "http://example.com/model.glb"}}),
                ],
            ),
            (
                ThreeDProvider::Tripo,
                "tripo:v3.1-20260211",
                [
                    json!({"code": 0, "data": {"task_id": "task"}}),
                    json!({"code": 0, "data": {"status": "success", "output": {"model_url": "http://example.com/model.glb"}}}),
                ],
            ),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let working_dir = dir.path().to_path_buf();
            let tool = GenerateThreeDTool {
                clients: vec![(
                    model.parse().unwrap(),
                    ThreeDClient::new(provider, "fake-key".to_owned())
                        .unwrap()
                        .with_base_url(base),
                )],
                output_dir: "output".into(),
            };
            let server = async {
                for (index, response) in responses.into_iter().enumerate() {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut received = Vec::new();
                    let header_end = loop {
                        let mut buffer = [0; 4096];
                        let len = socket.read(&mut buffer).await.unwrap();
                        assert_ne!(len, 0);
                        received.extend_from_slice(&buffer[..len]);
                        if let Some(end) =
                            received.windows(4).position(|bytes| bytes == b"\r\n\r\n")
                        {
                            break end + 4;
                        }
                    };
                    let headers = String::from_utf8_lossy(&received[..header_end]).to_lowercase();
                    assert!(headers.contains("authorization: bearer fake-key"));
                    let length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .map(|length| length.trim().parse::<usize>().unwrap())
                        .unwrap_or(0);
                    while received.len() < header_end + length {
                        let mut buffer = [0; 4096];
                        let len = socket.read(&mut buffer).await.unwrap();
                        assert_ne!(len, 0);
                        received.extend_from_slice(&buffer[..len]);
                    }
                    if index == 0 {
                        let body: Value =
                            serde_json::from_slice(&received[header_end..header_end + length])
                                .unwrap();
                        assert_eq!(body["prompt"], "chair");
                        assert_eq!(body["symmetry_mode"], "auto");
                        assert!(body.get("polycount").is_none());
                        match provider {
                            ThreeDProvider::Meshy => {
                                assert_eq!(body["target_polycount"], 5000);
                                assert_eq!(body["should_remesh"], true);
                                assert_eq!(body["topology"], "triangle");
                            }
                            ThreeDProvider::Tripo => assert_eq!(body["face_limit"], 5000),
                        }
                        assert_eq!(
                            body[if matches!(provider, ThreeDProvider::Meshy) {
                                "ai_model"
                            } else {
                                "model"
                            }],
                            model.split_once(':').unwrap().1
                        );
                    }
                    let body = response.to_string();
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                }
            };
            let cancel = CancellationToken::new();
            let (result, ()) = tokio::join!(
                tool.execute(ToolContext::new(
                    "call",
                    json!({"prompt": "chair", "polycount": 5000, "symmetry_mode": "auto"}),
                    Some(&working_dir),
                    &cancel
                )),
                server,
            );
            assert_eq!(
                result.unwrap_err().to_string(),
                "3D asset URL must use HTTPS"
            );
            assert_eq!(
                std::fs::read_dir(dir.path().join("output"))
                    .unwrap()
                    .count(),
                0
            );
        }
    }

    #[tokio::test]
    async fn unknown_model_reaches_provider_and_preserves_its_error() {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

        for (provider, id, model_field) in [
            (ThreeDProvider::Meshy, "meshy:future-model", "ai_model"),
            (ThreeDProvider::Tripo, "tripo:future-model", "model"),
            (ThreeDProvider::Tripo, "tripo:latest", "model"),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let tool = GenerateThreeDTool {
                clients: vec![(
                    id.parse().unwrap(),
                    ThreeDClient::new(provider, "fake-key")
                        .unwrap()
                        .with_base_url(base),
                )],
                output_dir: PathBuf::new(),
            };
            let server = async {
                let (socket, _) = listener.accept().await.unwrap();
                let mut reader = BufReader::new(socket);
                let mut length = None;
                loop {
                    let mut line = String::new();
                    assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = Some(value.trim().parse::<usize>().unwrap());
                    }
                }
                let mut body = vec![0; length.unwrap()];
                reader.read_exact(&mut body).await.unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    body[model_field],
                    if id == "tripo:latest" {
                        TRIPO_LATEST
                    } else {
                        "future-model"
                    }
                );
                let body = "unknown model from provider";
                reader.get_mut().write_all(format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()
                ).as_bytes()).await.unwrap();
            };
            let cancel = CancellationToken::new();
            let (result, ()) = tokio::join!(
                tool.execute(ToolContext::new(
                    "call",
                    json!({"prompt": "chair", "model": id}),
                    None,
                    &cancel,
                )),
                server
            );
            let error = result.unwrap_err().to_string();
            assert!(error.contains("unknown model from provider"), "{error}");
        }
    }

    #[test]
    fn model_ids_are_free_strings_and_only_tripo_latest_is_mapped() {
        let tool = GenerateThreeDTool::from_keys(|_| Some("fake-key".into()), PathBuf::new())
            .unwrap()
            .unwrap();
        let cancel = CancellationToken::new();
        assert!(
            tool.parameters()["properties"]["model"]
                .get("enum")
                .is_none()
        );
        for (id, expected) in [
            ("meshy:future-version", "future-version"),
            ("tripo:future-version", "future-version"),
            ("meshy:latest", "latest"),
            ("tripo:latest", TRIPO_LATEST),
        ] {
            let ctx = ToolContext::new(
                "call",
                json!({"prompt": "chair", "model": id}),
                None,
                &cancel,
            );
            let (_, request) = tool.request(&ctx).unwrap();
            assert_eq!(request.model.name(), expected);
        }
    }

    #[test]
    fn schema_exposes_text_or_images_and_both_defaults() {
        let tool = GenerateThreeDTool::from_keys(|_| Some("fake-key".into()), PathBuf::new())
            .unwrap()
            .unwrap();
        assert_eq!(tool.name(), "generate_3d");
        assert_eq!(tool.parameters()["properties"]["images"]["maxItems"], 4);
        assert_eq!(tool.parameters()["oneOf"].as_array().unwrap().len(), 2);
        assert_eq!(tool.clients[0].0.provider, Provider::Custom("meshy".into()));
    }
}
