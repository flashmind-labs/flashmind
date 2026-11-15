//! A tool for one Meshy task endpoint: submit, wait, and save the result files.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::bail;
use async_trait::async_trait;
use flashmind_llm::three_d::{MeshyTaskOutput, ThreeDClient};
use serde_json::{Map, Value, json};
use url::Url;

use super::files::{FileKind, inline_local_files};
use crate::three_d::download;
use crate::{Tool, ToolContext, ToolResult};

/// Where a tool sends its task.
pub(super) enum Route {
    Fixed(&'static str),
    /// `product` and `stage` arguments pick the Creative Lab endpoint.
    CreativeLab,
}

/// Static description of one Meshy task tool.
pub(super) struct TaskSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub route: Route,
    pub schema: fn() -> Value,
    /// Arguments that may hold local files, sent to Meshy as data URIs.
    pub files: &'static [(&'static str, FileKind)],
    /// Request GLB only unless the caller asks for formats, since Meshy
    /// otherwise returns every format.
    pub default_glb: bool,
    /// Task fields worth reporting besides files, such as printability reports.
    pub extras: &'static [&'static str],
}

/// Creative Lab products, each with its own endpoints.
pub(super) const CREATIVE_LAB_PRODUCTS: [&str; 9] = [
    "keychain",
    "fridge-magnet",
    "figure",
    "vinyl-figure",
    "brick-figure",
    "lamp",
    "keycap",
    "fidget-collapsible",
    "fidget-pixel",
];

/// A Meshy task endpoint exposed as a tool.
pub struct MeshyTaskTool {
    client: ThreeDClient,
    output_dir: PathBuf,
    spec: &'static TaskSpec,
    schema: Value,
}

impl MeshyTaskTool {
    pub(super) fn new(client: ThreeDClient, output_dir: PathBuf, spec: &'static TaskSpec) -> Self {
        Self {
            client,
            output_dir,
            spec,
            schema: (spec.schema)(),
        }
    }

    fn resolve(&self, args: &mut Map<String, Value>) -> anyhow::Result<String> {
        match self.spec.route {
            Route::Fixed(path) => Ok(path.to_owned()),
            Route::CreativeLab => creative_lab_path(args),
        }
    }

    /// Download every result file and describe the outcome.
    async fn report(
        &self,
        ctx: &ToolContext<'_>,
        output: &MeshyTaskOutput,
    ) -> anyhow::Result<ToolResult> {
        let dir = ctx
            .working_dir
            .map_or_else(|| self.output_dir.clone(), |dir| dir.join(&self.output_dir));
        let found = result_files(&output.task);
        if !found.is_empty() {
            tokio::fs::create_dir_all(&dir).await?;
        }
        let stem = format!(
            "{}_{}",
            self.spec.name,
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let mut text = format!("Meshy task {} succeeded", output.id);
        if let Some(credits) = output.consumed_credits {
            let _ = write!(text, " ({credits} credits)");
        }
        text.push_str(".\n");
        let mut saved = Vec::new();
        for (label, url) in found {
            let path = dir.join(format!("{stem}_{label}.{}", extension(&url)));
            match download(ctx, &self.client, &url, &path).await {
                Ok(()) => {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
                    let _ = writeln!(text, "- {label}: [{name}]({})", path.display());
                    saved.push(path);
                }
                Err(error) if ctx.cancel_token().is_cancelled() => return Err(error),
                Err(error) => {
                    let _ = writeln!(text, "- {label}: download failed: {error}");
                }
            }
        }
        for key in self.spec.extras {
            if let Some(value) = output.task.get(*key).filter(|value| !value.is_null()) {
                let _ = writeln!(text, "{key}: {value}");
            }
        }
        if !saved.is_empty() {
            text.push_str("Include a file's link in your response to send it.");
        }
        Ok(ToolResult::success(ctx.tool_call_id, text.trim_end()).with_files(saved))
    }
}

#[async_trait]
impl Tool for MeshyTaskTool {
    fn name(&self) -> &str {
        self.spec.name
    }

    fn description(&self) -> &str {
        self.spec.description
    }

    fn parameters(&self) -> Value {
        self.schema.clone()
    }

    fn timeout_secs(&self) -> Option<u64> {
        Some(1920)
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> anyhow::Result<ToolResult> {
        let Value::Object(mut args) = ctx.args.clone() else {
            bail!("Tool '{}': arguments must be an object", self.name());
        };
        check_arguments(self.name(), &self.schema, &args)?;
        let path = self.resolve(&mut args)?;
        if self.spec.default_glb && !args.contains_key("target_formats") {
            args.insert("target_formats".into(), json!(["glb"]));
        }
        inline_local_files(&mut args, self.spec.files, ctx.working_dir).await?;

        // The caller may cancel polling, but Meshy can still charge for submitted jobs.
        let output = tokio::select! {
            biased;
            _ = ctx.cancel_token().cancelled() => bail!("{} cancelled", self.name()),
            result = self.client.meshy_task(&path, Value::Object(args), |id| ctx.progress(&format!("Meshy task: {id}"))) => result?,
        };
        self.report(&ctx, &output).await
    }
}

/// Reject unknown arguments and missing required ones, as the harness does not
/// validate schemas.
fn check_arguments(name: &str, schema: &Value, args: &Map<String, Value>) -> anyhow::Result<()> {
    let properties = schema["properties"].as_object();
    for key in args.keys() {
        if !properties.is_some_and(|properties| properties.contains_key(key)) {
            bail!("Tool '{name}': unknown argument '{key}'");
        }
    }
    for required in schema["required"].as_array().into_iter().flatten() {
        if let Some(key) = required.as_str()
            && !args.contains_key(key)
        {
            bail!("Tool '{name}': missing required argument '{key}'");
        }
    }
    Ok(())
}

/// Pick the Creative Lab endpoint and remove the routing arguments from the body.
fn creative_lab_path(args: &mut Map<String, Value>) -> anyhow::Result<String> {
    let product = match args.remove("product") {
        Some(Value::String(product)) if CREATIVE_LAB_PRODUCTS.contains(&product.as_str()) => {
            product
        }
        _ => bail!(
            "product must be one of: {}",
            CREATIVE_LAB_PRODUCTS.join(", ")
        ),
    };
    let stage = args.remove("stage");
    if product == "fidget-collapsible" {
        return Ok(format!("openapi/creative-lab/{product}/v1"));
    }
    match stage.as_ref().and_then(Value::as_str) {
        Some(stage @ ("prototype" | "build")) => {
            Ok(format!("openapi/creative-lab/{product}/v1/{stage}"))
        }
        _ => bail!("stage must be prototype or build for {product}"),
    }
}

/// Downloadable files in a finished task, as `(label, url)`.
fn result_files(task: &Value) -> Vec<(String, Url)> {
    let mut found = Vec::new();
    let mut add = |label: String, value: &Value| {
        if let Some(url) = value
            .as_str()
            .filter(|url| !url.is_empty())
            .and_then(|url| Url::parse(url).ok())
        {
            found.push((label, url));
        }
    };
    for (key, value) in task["model_urls"].as_object().into_iter().flatten() {
        add(label(key), value);
    }
    for (index, value) in task["image_urls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        add(format!("image{}", index + 1), value);
    }
    add("depth_map".into(), &task["depth_map_url"]);
    found
}

fn label(key: &str) -> String {
    key.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// File extension from the URL path, or `bin`.
fn extension(url: &Url) -> String {
    url.path_segments()
        .and_then(|mut segments| segments.next_back())
        .and_then(|name| name.rsplit_once('.'))
        .map(|(_, ext)| ext)
        .filter(|ext| {
            (1..=5).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric())
        })
        .map_or_else(|| "bin".to_owned(), str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creative_lab_routes_by_product_and_stage() {
        let route = |value: Value| creative_lab_path(&mut value.as_object().unwrap().clone());
        assert_eq!(
            route(json!({"product": "keychain", "stage": "prototype"})).unwrap(),
            "openapi/creative-lab/keychain/v1/prototype"
        );
        assert_eq!(
            route(json!({"product": "lamp", "stage": "build"})).unwrap(),
            "openapi/creative-lab/lamp/v1/build"
        );
        assert_eq!(
            route(json!({"product": "fidget-collapsible"})).unwrap(),
            "openapi/creative-lab/fidget-collapsible/v1"
        );
        assert!(route(json!({"product": "lamp"})).is_err());
        assert!(route(json!({"product": "toaster", "stage": "build"})).is_err());
        assert!(route(json!({"stage": "build"})).is_err());
    }

    #[test]
    fn routing_arguments_are_removed_from_the_body() {
        let mut args = json!({"product": "keycap", "stage": "build", "candidate_id": "c"})
            .as_object()
            .unwrap()
            .clone();
        creative_lab_path(&mut args).unwrap();
        assert_eq!(Value::Object(args), json!({"candidate_id": "c"}));
    }

    #[test]
    fn arguments_are_checked_against_the_schema() {
        let schema = json!({
            "properties": {"a": {}, "b": {}},
            "required": ["a"],
        });
        let check = |value: Value| check_arguments("t", &schema, value.as_object().unwrap());
        assert!(check(json!({"a": 1, "b": 2})).is_ok());
        assert!(
            check(json!({"b": 2}))
                .unwrap_err()
                .to_string()
                .contains("'a'")
        );
        assert!(
            check(json!({"a": 1, "c": 2}))
                .unwrap_err()
                .to_string()
                .contains("'c'")
        );
    }

    #[test]
    fn result_files_cover_models_images_and_depth_maps() {
        let task = json!({
            "model_urls": {"glb": "https://x.test/m.glb?sig=1", "3mf": "", "bundle_zip": "https://x.test/b.zip"},
            "image_urls": ["https://x.test/i.png"],
            "depth_map_url": "https://x.test/d.png",
            "thumbnail_url": "https://x.test/t.png",
        });
        let files: Vec<_> = result_files(&task)
            .into_iter()
            .map(|(label, url)| (label, extension(&url)))
            .collect();
        for expected in [
            ("bundle_zip", "zip"),
            ("glb", "glb"),
            ("image1", "png"),
            ("depth_map", "png"),
        ] {
            assert!(
                files.contains(&(expected.0.to_owned(), expected.1.to_owned())),
                "{expected:?} in {files:?}"
            );
        }
        assert_eq!(files.len(), 4);
        assert_eq!(
            extension(&Url::parse("https://x.test/noext").unwrap()),
            "bin"
        );
    }

    /// Serve one canned JSON reply per entry and return the request bodies.
    async fn serve(
        listener: tokio::net::TcpListener,
        replies: Vec<(&'static str, Value)>,
    ) -> Vec<Value> {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
        let mut bodies = Vec::new();
        for (request_line, reply) in replies {
            let (socket, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(socket);
            let mut first = String::new();
            reader.read_line(&mut first).await.unwrap();
            assert!(first.starts_with(request_line), "{first}");
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).await.unwrap();
            bodies.push(if length == 0 {
                Value::Null
            } else {
                serde_json::from_slice(&body).unwrap()
            });
            let reply = reply.to_string();
            reader
                .get_mut()
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                        reply.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        }
        bodies
    }

    async fn tool_for(name: &str, listener: &tokio::net::TcpListener) -> MeshyTaskTool {
        let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let spec = super::super::specs::TASKS
            .iter()
            .find(|spec| spec.name == name)
            .unwrap();
        let client = ThreeDClient::new(flashmind_llm::three_d::ThreeDProvider::Meshy, "key")
            .unwrap()
            .with_base_url(base);
        MeshyTaskTool::new(client, "out".into(), spec)
    }

    fn text(result: &ToolResult) -> String {
        match result {
            ToolResult::Success { content, .. } => content
                .iter()
                .filter_map(|part| match part {
                    flashmind_types::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn retexture_inlines_the_local_model_defaults_to_glb_and_reports_failed_downloads() {
        use tokio_util::sync::CancellationToken;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tool = tool_for("retexture_3d", &listener).await;
        let dir = tempfile::tempdir().unwrap();
        let working_dir = dir.path().to_path_buf();
        std::fs::write(working_dir.join("hero.glb"), b"glTF").unwrap();
        let server = serve(
            listener,
            vec![
                ("POST /openapi/v1/retexture ", json!({"result": "task-1"})),
                (
                    "GET /openapi/v1/retexture/task-1 ",
                    json!({"status": "SUCCEEDED", "consumed_credits": 10, "model_urls": {"glb": "http://example.com/a.glb"}}),
                ),
            ],
        );
        let cancel = CancellationToken::new();
        let (result, bodies) = tokio::join!(
            tool.execute(ToolContext::new(
                "call",
                json!({"model_url": "hero.glb", "text_style_prompt": "rusty"}),
                Some(&working_dir),
                &cancel,
            )),
            server,
        );
        let result = text(&result.unwrap());
        assert!(
            result.contains("Meshy task task-1 succeeded (10 credits)"),
            "{result}"
        );
        assert!(result.contains("glb: download failed"), "{result}");
        assert_eq!(
            bodies[0],
            json!({
                "model_url": "data:application/octet-stream;base64,Z2xURg==",
                "text_style_prompt": "rusty",
                "target_formats": ["glb"],
            })
        );
        let leftovers = std::fs::read_dir(working_dir.join("out"))
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert_eq!(leftovers, 0);
    }

    #[tokio::test]
    async fn analyze_reports_the_printability_without_downloads_and_creative_lab_routes() {
        use tokio_util::sync::CancellationToken;
        let cancel = CancellationToken::new();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tool = tool_for("analyze_printability_3d", &listener).await;
        let server = serve(
            listener,
            vec![
                ("POST /openapi/v1/print/analyze ", json!({"result": "a"})),
                (
                    "GET /openapi/v1/print/analyze/a ",
                    json!({"status": "SUCCEEDED", "consumed_credits": 0, "printability": {"status": "healthy"}}),
                ),
            ],
        );
        let (result, bodies) = tokio::join!(
            tool.execute(ToolContext::new(
                "call",
                json!({"input_task_id": "src", "printer_type": "fdm"}),
                None,
                &cancel,
            )),
            server,
        );
        let result = text(&result.unwrap());
        assert!(
            result.contains("printability: {\"status\":\"healthy\"}"),
            "{result}"
        );
        assert_eq!(
            bodies[0],
            json!({"input_task_id": "src", "printer_type": "fdm"})
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tool = tool_for("creative_lab_3d", &listener).await;
        let server = serve(
            listener,
            vec![
                (
                    "POST /openapi/creative-lab/keycap/v1/prototype ",
                    json!({"result": "p"}),
                ),
                (
                    "GET /openapi/creative-lab/keycap/v1/prototype/p ",
                    json!({"status": "SUCCEEDED", "candidate_ids": ["c1", "c2"]}),
                ),
            ],
        );
        let (result, bodies) = tokio::join!(
            tool.execute(ToolContext::new(
                "call",
                json!({"product": "keycap", "stage": "prototype", "image_url": "https://example.com/i.png"}),
                None,
                &cancel,
            )),
            server,
        );
        let result = text(&result.unwrap());
        assert!(
            result.contains("candidate_ids: [\"c1\",\"c2\"]"),
            "{result}"
        );
        assert_eq!(bodies[0], json!({"image_url": "https://example.com/i.png"}));
    }

    #[tokio::test]
    async fn unknown_and_missing_arguments_fail_before_any_request() {
        use tokio_util::sync::CancellationToken;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tool = tool_for("convert_3d", &listener).await;
        let cancel = CancellationToken::new();
        for args in [
            json!({"input_task_id": "x"}),
            json!({"input_task_id": "x", "target_formats": ["glb"], "bogus": 1}),
            json!("not an object"),
        ] {
            let ctx = ToolContext::new("call", args, None, &cancel);
            assert!(tool.execute(ctx).await.is_err());
        }
    }
}
